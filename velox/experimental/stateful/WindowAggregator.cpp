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
#include "velox/experimental/stateful/WindowAggregator.h"
#include <experimental/stateful/InternalTimerService.h>
#include <algorithm>
#include <cstring>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <unordered_map>
#include <vector>
#include "velox/experimental/stateful/TimerHeapInternalTimer.h"
#include "velox/experimental/stateful/window/TimeWindowUtil.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::stateful {

namespace {
/// Wraps the first 'size' rows selected by 'indices' of 'input' as a
/// dictionary-encoded RowVector (O(size), no copy) - used by the
/// tumble/hop fast path to feed sorted key-adjacent chunks to the
/// streaming aggregation without materializing.
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

/// Forward declaration: defined below, used by the tumble/hop fast path.
RowVectorPtr addWindowTimestampToOutput(
    const RowVectorPtr& output,
    const std::string& fieldName,
    const int64_t fieldValue,
    const int32_t fieldIndex);

WindowAggregator::WindowAggregator(
    std::unique_ptr<exec::Operator> localAggregator,
    std::unique_ptr<exec::Operator> globalAggregator,
    std::vector<std::unique_ptr<StatefulOperator>> targets,
    std::unique_ptr<KeySelector> keySelector,
    std::unique_ptr<SliceAssigner> sliceAssigner,
    const long windowInterval,
    const bool useDayLightSaving,
    const bool isEventTime,
    const int windowStartIndex,
    const int windowEndIndex,
    const bool inputTagged)
    : StatefulOperator(std::move(globalAggregator), std::move(targets)),
      Triggerable<int64_t, long>(),
      localAggregator_(std::move(localAggregator)),
      keySelector_(std::move(keySelector)),
      sliceAssigner_(std::move(sliceAssigner)),
      windowInterval_(windowInterval),
      useDayLightSaving_(useDayLightSaving),
      isEventTime_(isEventTime),
      windowStartIndex_(windowStartIndex),
      windowEndIndex_(windowEndIndex),
      inputTagged_(inputTagged) {
  windowBuffer_ = std::make_shared<RecordsWindowBuffer>();
}

void WindowAggregator::initialize() {
  StatefulOperator::initialize();
  if (localAggregator_) {
    localAggregator_->initialize();
  }
}

void WindowAggregator::initializeStateBackend(StateBackend* stateBackend) {
  // The base class no longer creates a state handler (the operator-owned
  // keyed state migration is still in flight upstream); create our own
  // from the task-level backend, as the base class documents.
  if (!stateHandler()) {
    stateHandler_ = std::make_shared<StreamOperatorStateHandler>(
        op()->operatorId(), stateBackend->createKeyedStateBackend());
  }
  StatefulOperator::initializeStateBackend(stateBackend);
}

void WindowAggregator::initializeState() {
  StateDescriptor stateDesc("window-aggs", "", memoryPool());
  windowState_ = stateHandler()->getValueState(stateDesc);
  windowTimerService_ = stateHandler()->createTimerService(this);
}

void WindowAggregator::addInput(StreamElementPtr input) {
  VELOX_CHECK(!input_, "Last input has not been processed");
  auto record = std::static_pointer_cast<StreamRecord>(input);
  // TODO: preserve rowKind across this operator.
  input_ = record->record();
}

void WindowAggregator::advance() {
  if (!input_) {
    return;
  }

  if (inputTagged_ && isEventTime_ &&
      (sliceAssigner_->windowType() == WindowType::TUMBLE ||
       sliceAssigner_->windowType() == WindowType::HOP)) {
    // Tumble/hop fast path: keep the arriving batches whole - no per-key
    // splitting. The watermark side sorts key-adjacent chunks through the
    // aggregation operators; see processWatermarkInternal.
    auto sliceEndToData = sliceAssigner_->assignSliceEndFromTag(input_);
    for (auto& [sliceEnd, data] : sliceEndToData) {
      int64_t lastWindowEnd = sliceAssigner_->getLastWindowEnd(sliceEnd);
      if (TimeWindowUtil::isWindowFired(
              lastWindowEnd, currentProgress_, shiftTimeZone_)) {
        // every window containing this slice has fired: the data is late
        // and can be dropped.
        continue;
      }
      auto& slice = pendingWholeBatches_[sliceEnd];
      slice.batches.push_back(data);
      if (slice.nextWindow == 0) {
        slice.nextWindow = sliceEnd;
      }
    }
    input_.reset();
    return;
  }
  std::map<int64_t, RowVectorPtr> keyToData = keySelector_->partition(input_);
  for (const auto& [key, data] : keyToData) {
    // Two-phase input carries the local aggregator's slice-end tag in the
    // last column (use it directly); single-phase input (session) holds raw
    // rows and is sliced from the event-time column.
    std::map<int64_t, RowVectorPtr> sliceEndToData = inputTagged_
        ? sliceAssigner_->assignSliceEndFromTag(data)
        : sliceAssigner_->assignSliceEnd(data);
    for (const auto& [sliceEnd, data] : sliceEndToData) {
      if (!isEventTime_) {
        windowTimerService_->registerProcessingTimeTimer(
            key, sliceEnd, sliceEnd);
        continue;
      }
      // A slice is only fully consumed once its LAST containing window has
      // fired (for tumble the slice is the window; for hop the slice feeds
      // size/slide windows ending up to sliceEnd - slice + size).
      int64_t lastWindowEnd = sliceAssigner_->getLastWindowEnd(sliceEnd);
      if (TimeWindowUtil::isWindowFired(
              lastWindowEnd, currentProgress_, shiftTimeZone_)) {
        // every window containing this slice has fired: the data is late
        // and can be dropped.
        continue;
      }
      {
        std::lock_guard<std::mutex> lock(*mtx_);
        auto windowData = data;
        windowBuffer_->addElement(key, sliceEnd, windowData);
      }
    }
  }
  input_.reset();
}

void WindowAggregator::processWatermarkInternal(int64_t timestamp) {
  if (!isEventTime_ || timestamp <= currentProgress_) {
    return;
  }
  currentProgress_ = timestamp;
  const bool session = sliceAssigner_->windowType() == WindowType::SESSION;
  // Session windows are unaligned: a window can end at any timestamp, so
  // the slide-boundary trigger optimization does not apply - check the
  // buffer on every watermark advance.
  if (!session && currentProgress_ < nextTriggerWatermark_) {
    return;
  }
  auto windowKeyToData = windowBuffer_->advanceProgress(currentProgress_);
  // Tumble/hop fast path, fire side. Group the windows that fire at this
  // watermark by window end; a window's input is the concatenation of all
  // still-held slices feeding it (a tumble slice IS its window; a hop
  // slice feeds every window in [sliceEnd, sliceEnd + size - slide] and
  // is held until its last window has fired, so late contributions are
  // impossible by construction). Sort the window's partials by
  // hash-derived key id so equal user keys are adjacent (the streaming
  // aggregation's pre-grouped contract), then drive the local and global
  // aggregation over chunks of at most 1024 distinct ids - each operator
  // cycle then holds at most 1024 groups (the configured max output
  // batch), so its single getOutput flushes completely. Every key lands
  // in exactly one chunk, so the operator outputs are final rows;
  // collect them for the batched flushPendingFireOutputs emission. No
  // per-key splitting, keyed state or timers.
  if (!pendingWholeBatches_.empty()) {
    VELOX_CHECK(localAggregator_, "Local aggregator not set");
    auto emitWindow = [&](const RowVectorPtr& merged, int64_t windowEnd) {
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
        localAggregator_->addInput(chunk);
        RowVectorPtr localAccs = localAggregator_->getOutput();
        if (localAccs) {
          op()->addInput(localAccs);
          RowVectorPtr newAccs = op()->getOutput();
          if (newAccs) {
            RowVectorPtr output = newAccs;
            if (windowStartIndex_ >= 0) {
              output = addWindowTimestampToOutput(
                  output,
                  "window_start",
                  sliceAssigner_->getWindowStart(windowEnd),
                  windowStartIndex_);
            }
            if (windowEndIndex_ >= 0) {
              output = addWindowTimestampToOutput(
                  output, "window_end", windowEnd, windowEndIndex_);
            }
            if (output) {
              pendingFireOutputs_.push_back(std::move(output));
            }
          }
        }
        begin = end;
      }
    };
    int64_t step = sliceAssigner_->step();
    if (step <= 0) {
      step = sliceAssigner_->size();
    }
    // windows firing at this watermark, grouped by window end
    std::map<int64_t, std::vector<int64_t>> firing;
    for (const auto& [sliceEnd, ps] : pendingWholeBatches_) {
      const int64_t lastWindowEnd = sliceAssigner_->getLastWindowEnd(sliceEnd);
      for (int64_t W = ps.nextWindow; W <= lastWindowEnd &&
           TimeWindowUtil::isWindowFired(W, currentProgress_, shiftTimeZone_);
           W += step) {
        firing[W].push_back(sliceEnd);
      }
    }
    for (const auto& [W, slices] : firing) {
      std::list<RowVectorPtr> all;
      for (int64_t S : slices) {
        const auto& batches = pendingWholeBatches_[S].batches;
        all.insert(all.end(), batches.begin(), batches.end());
      }
      RowVectorPtr merged = all.size() == 1
          ? all.front()
          : TimeWindowUtil::mergeVectors(all, op()->pool());
      emitWindow(merged, W);
    }
    // advance each slice past its fired windows; drop exhausted slices
    for (auto it = pendingWholeBatches_.begin();
         it != pendingWholeBatches_.end();) {
      const int64_t lastWindowEnd = sliceAssigner_->getLastWindowEnd(it->first);
      while (it->second.nextWindow <= lastWindowEnd &&
             TimeWindowUtil::isWindowFired(
                 it->second.nextWindow, currentProgress_, shiftTimeZone_)) {
        it->second.nextWindow += step;
      }
      if (it->second.nextWindow > lastWindowEnd) {
        it = pendingWholeBatches_.erase(it);
      } else {
        ++it;
      }
    }
  }
  for (const auto& [windowKey, datas] : windowKeyToData) {
    if (datas.empty()) {
      continue;
    }
    VELOX_CHECK(localAggregator_, "Local aggregator not set");
    localAggregator_->addInput(
        TimeWindowUtil::mergeVectors(datas, op()->pool()));
    RowVectorPtr localAcc = localAggregator_->getOutput();
    if (!localAcc) {
      continue;
    }
    if (session) {
      mergeSessionWindow(windowKey.key(), windowKey.window(), localAcc);
      continue;
    }
    // Tumble/hop: distribute the fired slice's accumulator into every
    // window containing the slice. For tumble the containing window set
    // is the slice itself; for hop it is every window ending in
    // [sliceEnd, sliceEnd - slice + size] stepping by the slide.
    const int64_t firstWindow = windowKey.window();
    const int64_t lastWindow =
        sliceAssigner_->getLastWindowEnd(windowKey.window());
    int64_t step = sliceAssigner_->step();
    if (step <= 0) {
      step = sliceAssigner_->size();
    }
    for (int64_t windowEnd = firstWindow; windowEnd <= lastWindow;
         windowEnd += step) {
      auto stateAcc = windowState_->value(windowKey.key(), windowEnd);
      std::list<RowVectorPtr> allDatas;
      allDatas.push_back(localAcc);
      if (stateAcc) {
        allDatas.push_back(stateAcc);
      }
      op()->addInput(TimeWindowUtil::mergeVectors(allDatas, op()->pool()));
      auto newAcc = op()->getOutput();
      if (newAcc) {
        windowState_->update(windowKey.key(), windowEnd, newAcc);
      }
      // Registering a timer at or below the current watermark is safe:
      // advanceWatermark below fires it in this same cycle, after the
      // state write, so the window emits with the complete accumulator.
      windowTimerService_->registerEventTimeTimer(
          windowKey.key(), windowEnd, windowEnd - 1);
    }
  }
  windowTimerService_->advanceWatermark(currentProgress_);
  // All timers for this watermark have fired; flush the collected
  // window outputs as one merged batch (one crossing instead of one
  // per (key, window)).
  flushPendingFireOutputs();
  if (session) {
    nextTriggerWatermark_ = currentProgress_;
  } else {
    nextTriggerWatermark_ = TimeWindowUtil::getNextTriggerWatermark(
        currentProgress_, windowInterval_, shiftTimeZone_, useDayLightSaving_);
  }
}

void WindowAggregator::flushPendingFireOutputs() {
  if (pendingFireOutputs_.empty()) {
    return;
  }
  // All fireWindow outputs share the same schema (agg output +
  // window_start/window_end columns added by addWindowTimestampToOutput),
  // so they merge into one RowVector.
  auto merged = TimeWindowUtil::mergeVectors(pendingFireOutputs_, op()->pool());
  pendingFireOutputs_.clear();
  pushOutput(
      std::make_shared<StreamRecord>(getPlanNodeId(), std::move(merged)));
}

void WindowAggregator::mergeSessionWindow(
    int64_t key,
    int64_t sliceEnd,
    RowVectorPtr localAcc) {
  const int64_t gap = sliceAssigner_->size();
  const int64_t tentativeStart = sliceEnd - gap;
  auto& live = sessionLiveWindows_[key];
  // Collect the live windows overlapping [tentativeStart, sliceEnd):
  // [ws, we) overlaps iff ws < sliceEnd && we > tentativeStart.
  std::vector<std::pair<int64_t, int64_t>> overlapped; // (end, start)
  for (const auto& [we, ws] : live) {
    if (ws < sliceEnd && we > tentativeStart) {
      overlapped.emplace_back(we, ws);
    }
  }
  int64_t mergedStart = tentativeStart;
  int64_t mergedEnd = sliceEnd;
  std::list<RowVectorPtr> allDatas;
  allDatas.push_back(localAcc);
  for (const auto& [we, ws] : overlapped) {
    auto acc = windowState_->value(key, we);
    if (acc) {
      allDatas.push_back(acc);
    }
    windowState_->remove(key, we);
    live.erase(we);
    mergedStart = std::min(mergedStart, ws);
    mergedEnd = std::max(mergedEnd, we);
  }
  op()->addInput(TimeWindowUtil::mergeVectors(allDatas, op()->pool()));
  auto newAcc = op()->getOutput();
  if (newAcc) {
    windowState_->update(key, mergedEnd, newAcc);
  }
  live[mergedEnd] = mergedStart;
  // Sessions are only complete once no further event arrives within the
  // gap: a window merged at watermark W can still absorb a slice that
  // fires at the NEXT crossing (slices are keyed by event + gap and are
  // processed one crossing later than Flink's per-event merging). Delay
  // the cleanup timer by one gap so late-arriving slices within the gap
  // merge into the live window instead of opening a duplicate session.
  windowTimerService_->registerEventTimeTimer(
      key, mergedEnd, mergedEnd + gap - 1);
}

void WindowAggregator::onEventTime(
    std::shared_ptr<TimerHeapInternalTimer<int64_t, int64_t>> timer) {
  onTimer(timer);
}

/// Add window_start / window_end timestamp to output
RowVectorPtr addWindowTimestampToOutput(
    const RowVectorPtr& output,
    const std::string& fieldName,
    const int64_t fieldValue,
    const int32_t fieldIndex) {
  auto createTimestampVector =
      [&](const Timestamp& val,
          const size_t size,
          velox::memory::MemoryPool* pool) -> VectorPtr {
    const TypePtr windowStartType = std::make_shared<const TimestampType>();
    VectorPtr windowStartVec =
        BaseVector::create(windowStartType, output->size(), output->pool());
    FlatVector<Timestamp>* timestampVector =
        windowStartVec->asFlatVector<Timestamp>();
    for (size_t i = 0; i < size; ++i) {
      timestampVector->set(i, val);
    }
    return windowStartVec;
  };
  const TypePtr& outputType = output->type();
  const RowTypePtr& outputRowType =
      std::dynamic_pointer_cast<const RowType>(outputType);
  const std::vector<std::string>& outputFieldNames = outputRowType->names();
  const std::vector<TypePtr>& outputFieldTypes = outputRowType->children();
  const std::vector<VectorPtr>& outputFields = output->children();
  std::vector<std::string> newOutputFieldNames;
  std::vector<TypePtr> newOutputFieldTypes;
  std::vector<VectorPtr> newOutputFields;
  VectorPtr windowStartVec = createTimestampVector(
      Timestamp::fromMillis(fieldValue), output->size(), output->pool());
  for (int i = 0; i < fieldIndex; ++i) {
    newOutputFieldTypes.emplace_back(outputFieldTypes[i]);
    newOutputFieldNames.emplace_back(outputFieldNames[i]);
    newOutputFields.emplace_back(outputFields[i]);
  }
  newOutputFieldTypes.emplace_back(std::make_shared<const TimestampType>());
  newOutputFieldNames.emplace_back(fieldName);
  newOutputFields.emplace_back(windowStartVec);
  for (int i = fieldIndex + 1; i < output->childrenSize() + 1; ++i) {
    newOutputFieldTypes.emplace_back(outputFieldTypes[i - 1]);
    newOutputFieldNames.emplace_back(outputFieldNames[i - 1]);
    newOutputFields.emplace_back(outputFields[i - 1]);
  }
  auto newOutputRowType = std::make_shared<const RowType>(
      std::move(newOutputFieldNames), std::move(newOutputFieldTypes));
  return std::make_shared<RowVector>(
      output->pool(),
      newOutputRowType,
      output->nulls(),
      output->size(),
      newOutputFields,
      output->getNullCount());
}

void WindowAggregator::onTimer(
    std::shared_ptr<TimerHeapInternalTimer<int64_t, int64_t>> timer) {
  fireWindow(timer->key(), timer->timestamp(), timer->ns());
  clearWindow(timer->key(), timer->timestamp(), timer->ns());
}

template <typename K>
void WindowAggregator::fireWindow(
    const K& key,
    int64_t timerTimestamp,
    int64_t windowEnd) {
  RowVectorPtr output = windowState_->value(key, windowEnd);
  if (!output) {
    LOG(INFO) << "No output found for key: " << key
              << ", window end: " << windowEnd;
    return;
  }
  {
    if (windowStartIndex_ >= 0) {
      // window_start = windowEnd - size for aligned windows (tumble/hop);
      // session windows keep each window's own start (windows merge into
      // arbitrary spans, so the start is not derivable from the end).
      int64_t windowStart;
      if (sliceAssigner_->windowType() == WindowType::SESSION) {
        auto kit = sessionLiveWindows_.find(static_cast<int64_t>(key));
        auto it = kit != sessionLiveWindows_.end() ? kit->second.find(windowEnd)
                                                   : kit->second.end();
        windowStart = it != kit->second.end()
            ? it->second
            : windowEnd - sliceAssigner_->size();
      } else {
        windowStart = sliceAssigner_->getWindowStart(windowEnd);
      }
      output = addWindowTimestampToOutput(
          output, "window_start", windowStart, windowStartIndex_);
    }
    if (windowEndIndex_ >= 0) {
      output = addWindowTimestampToOutput(
          output, "window_end", windowEnd, windowEndIndex_);
    }
  }
  if (output) {
    pendingFireOutputs_.push_back(std::move(output));
  }
}

template <typename K>
void WindowAggregator::clearWindow(
    const K& key,
    int64_t timerTimestamp,
    int64_t windowEnd) {
  windowState_->remove(key, windowEnd);
  if (sliceAssigner_->windowType() == WindowType::SESSION) {
    auto kit = sessionLiveWindows_.find(static_cast<int64_t>(key));
    if (kit != sessionLiveWindows_.end()) {
      kit->second.erase(windowEnd);
    }
  }
}

void WindowAggregator::onProcessingTime(
    std::shared_ptr<TimerHeapInternalTimer<int64_t, int64_t>> timer) {
  if (timer->timestamp() > lastTriggeredProcessingTime_) {
    lastTriggeredProcessingTime_ = timer->timestamp();
    auto windowKeyToData = windowBuffer_->advanceProgress(timer->timestamp());
    for (auto& [windowKey, datas] : windowKeyToData) {
      if (datas.empty()) {
        continue;
      }
      auto stateAcc = windowState_->value(windowKey.key(), windowKey.window());
      if (stateAcc) {
        datas.push_back(stateAcc);
      }
      RowVectorPtr opInput = TimeWindowUtil::mergeVectors(datas, op()->pool());
      op()->addInput(opInput);
      auto newAcc = op()->getOutput();
      if (newAcc) {
        windowState_->update(windowKey.key(), windowKey.window(), newAcc);
      }
    }
    if (!windowKeyToData.empty()) {
      windowBuffer_->clear();
    }
  }
  onTimer(timer);
}

void WindowAggregator::onProcessingTime(int64_t timestamp) {
  auto callbackBridge = nativeCallbackBridge();
  VELOX_CHECK_NOT_NULL(
      callbackBridge,
      "Native callback bridge is not bound for processing-time callback");
  callbackBridge->onProcessingTime(timestamp);
}

void WindowAggregator::close() {
  // processWatermarkInternal(std::numeric_limits<int64_t>::max());
  StatefulOperator::close();
  if (localAggregator_) {
    localAggregator_->close();
  }
  input_.reset();
  // State members are created in initializeState(), which fails fast when the
  // operator has no state handler; they can be null at close time.
  if (windowTimerService_) {
    windowTimerService_->close();
  }
  windowBuffer_->clear();
  sessionLiveWindows_.clear();
  pendingWholeBatches_.clear();
  pendingFireOutputs_.clear();
  if (windowState_) {
    windowState_->clear();
  }
  currentProgress_ = 0;
  nextTriggerWatermark_ = 0;
}

} // namespace facebook::velox::stateful
