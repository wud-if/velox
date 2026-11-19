/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "velox/experimental/stateful/LocalWindowAggregator.h"
#include <glog/logging.h>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include "velox/experimental/stateful/window/SliceAssigner.h"
#include "velox/experimental/stateful/window/TimeWindowUtil.h"

namespace facebook::velox::stateful {

namespace {
/// Wraps the first 'size' rows selected by 'indices' of 'input' as a
/// dictionary-encoded RowVector (O(size), no copy) - the same technique
/// SliceAssigner::assignSliceEnd uses for its slice groups.
RowVectorPtr wrapChunkByIndices(
    const RowVectorPtr& input,
    vector_size_t size,
    const BufferPtr& indices,
    memory::MemoryPool* pool) {
  auto result = std::make_shared<RowVector>(
      pool,
      input->type(),
      nullptr,
      size,
      std::vector<VectorPtr>(input->childrenSize()));
  for (vector_size_t i = 0; i < input->childrenSize(); ++i) {
    result->childAt(i) =
        BaseVector::wrapInDictionary(nullptr, indices, size, input->childAt(i));
  }
  result->updateContainsLazyNotLoaded();
  return result;
}
} // namespace

// LocalWindowAggregator
LocalWindowAggregator::LocalWindowAggregator(
    std::unique_ptr<exec::Operator> op,
    std::vector<std::unique_ptr<StatefulOperator>> targets,
    std::unique_ptr<KeySelector> keySelector,
    std::unique_ptr<SliceAssigner> sliceAssigner,
    const int64_t windowInterval,
    const bool useDayLightSaving,
    RowTypePtr outputType)
    : StatefulOperator(std::move(op), std::move(targets)),
      keySelector_(std::move(keySelector)),
      sliceAssigner_(std::move(sliceAssigner)),
      windowInterval_(windowInterval),
      useDayLightSaving_(useDayLightSaving),
      outputType_(std::move(outputType)) {
  windowBuffer_ = std::make_shared<RecordsWindowBuffer>();
}

void LocalWindowAggregator::addInput(StreamElementPtr input) {
  VELOX_CHECK(!input_, "Last input has not been processed");
  auto record = std::static_pointer_cast<StreamRecord>(input);
  // TODO: preserve rowKind across this operator.
  input_ = record->record();
}

void LocalWindowAggregator::advance() {
  if (!input_) {
    return;
  }

  if (sliceAssigner_->windowType() == WindowType::TUMBLE ||
      sliceAssigner_->windowType() == WindowType::HOP) {
    // Tumble/hop fast path: keep the arriving batches whole - no per-key
    // splitting. The watermark side sorts key-adjacent chunks through the
    // aggregation operator; see processWatermarkInternal.
    auto sliceEndToData = sliceAssigner_->assignSliceEnd(input_);
    for (auto& [sliceEnd, data] : sliceEndToData) {
      // Buffer every slice; a slice arriving after its own fire time still
      // feeds windows that end later (hopping windows), so the drop of
      // already-fired windows happens on the global side per window.
      pendingWholeBatches_[sliceEnd].push_back(data);
    }
    input_.reset();
    return;
  }
  std::map<int64_t, RowVectorPtr> keyToData = keySelector_->partition(input_);
  for (const auto& [key, data] : keyToData) {
    std::map<int64_t, RowVectorPtr> sliceEndToData =
        sliceAssigner_->assignSliceEnd(data);
    for (const auto& [sliceEnd, data] : sliceEndToData) {
      // Buffer every slice; a slice arriving after its own fire time still
      // feeds windows that end later (hopping windows), so the drop of
      // already-fired windows happens on the global side per window.
      auto windowData = data;
      windowBuffer_->addElement(key, sliceEnd, windowData);
    }
  }
  input_.reset();
}

void LocalWindowAggregator::processWatermarkInternal(int64_t timestamp) {
  if (timestamp > currentWatermark_) {
    currentWatermark_ = timestamp;
    if (currentWatermark_ >= nextTriggerWatermark_) {
      // we only need to call advanceProgress() when current watermark may
      // trigger window
      auto windowKeyToData = windowBuffer_->advanceProgress(currentWatermark_);
      // Batch fire: collect the aggregation results and emit as merged
      // output batches instead of one 1-row record per key-window. For
      // tumble the fired slices are additionally aggregated in key-sorted
      // chunks (see WindowAggregator's tumble fast path): a fired slice
      // groups ALL of its whole batches through a handful of operator
      // cycles instead of one cycle per key.
      std::list<RowVectorPtr> fireOutputs;
      for (const auto& [windowKey, datas] : windowKeyToData) {
        if (datas.empty()) {
          continue;
        }
        op()->addInput(TimeWindowUtil::mergeVectors(datas, op()->pool()));
        RowVectorPtr output = op()->getOutput();
        if (output) {
          fireOutputs.push_back(
              addWindowEndToVector(std::move(output), windowKey.window()));
        }
      }
      for (auto it = pendingWholeBatches_.begin();
           it != pendingWholeBatches_.end();) {
        if (!TimeWindowUtil::isWindowFired(
                it->first, currentWatermark_, shiftTimeZone_)) {
          ++it;
          continue;
        }
        const int64_t sliceEnd = it->first;
        RowVectorPtr merged = it->second.size() == 1
            ? it->second.front()
            : TimeWindowUtil::mergeVectors(it->second, op()->pool());
        const vector_size_t numRows = merged->size();
        std::vector<int64_t> ids;
        keySelector_->rowKeyIds(merged, ids);
        std::vector<vector_size_t> order(numRows);
        std::iota(order.begin(), order.end(), 0);
        std::sort(
            order.begin(), order.end(), [&](vector_size_t a, vector_size_t b) {
              return ids[a] < ids[b];
            });
        std::vector<int64_t> sortedIds(numRows);
        for (vector_size_t i = 0; i < numRows; ++i) {
          sortedIds[i] = ids[order[i]];
        }
        constexpr vector_size_t kMaxGroupsPerChunk = 1024;
        constexpr vector_size_t kMaxRowsPerChunk = 65536;
        vector_size_t begin = 0;
        while (begin < numRows) {
          vector_size_t end = begin + 1;
          vector_size_t groups = 1;
          while (end < numRows && groups < kMaxGroupsPerChunk &&
                 end - begin < kMaxRowsPerChunk) {
            if (sortedIds[end] != sortedIds[end - 1]) {
              ++groups;
            }
            ++end;
          }
          const vector_size_t chunkRows = end - begin;
          BufferPtr indices = allocateIndices(chunkRows, op()->pool());
          auto* raw = indices->asMutable<vector_size_t>();
          for (vector_size_t i = 0; i < chunkRows; ++i) {
            raw[i] = order[begin + i];
          }
          RowVectorPtr chunk =
              wrapChunkByIndices(merged, chunkRows, indices, op()->pool());
          op()->addInput(chunk);
          RowVectorPtr output = op()->getOutput();
          if (output) {
            fireOutputs.push_back(
                addWindowEndToVector(std::move(output), sliceEnd));
          }
          begin = end;
        }
        it = pendingWholeBatches_.erase(it);
      }
      if (!fireOutputs.empty()) {
        pushOutput(std::make_shared<StreamRecord>(
            getPlanNodeId(),
            TimeWindowUtil::mergeVectors(fireOutputs, op()->pool())));
      }
      nextTriggerWatermark_ = TimeWindowUtil::getNextTriggerWatermark(
          currentWatermark_,
          windowInterval_,
          shiftTimeZone_,
          useDayLightSaving_);
    }
  }
}

void LocalWindowAggregator::close() {
  processWatermarkInternal(INT_MAX);
  StatefulOperator::close();
  input_.reset();
  windowBuffer_->clear();
  pendingWholeBatches_.clear();
  currentWatermark_ = 0;
  nextTriggerWatermark_ = 0;
}

RowVectorPtr LocalWindowAggregator::addWindowEndToVector(
    RowVectorPtr vector,
    int64_t sliceEnd) {
  auto newColumn = BaseVector::create(BIGINT(), vector->size(), vector->pool());
  auto windowEndCol = newColumn->as<FlatVector<int64_t>>();

  for (int i = 0; i < vector->size(); ++i) {
    windowEndCol->set(i, sliceEnd);
  }

  std::vector<VectorPtr> newChildren(vector->children());
  newChildren.push_back(newColumn);

  auto newRowVector = std::make_shared<RowVector>(
      vector->pool(),
      outputType_,
      vector->nulls(), // 保留原 nulls
      vector->size(),
      std::move(newChildren),
      vector->getNullCount());
  return newRowVector;
}
} // namespace facebook::velox::stateful
