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
#include "velox/experimental/stateful/WindowJoin.h"

#include <cstdint>
#include <utility>

#include "velox/experimental/stateful/TypeSerializer.h"
#include "velox/experimental/stateful/state/StateDescriptor.h"
#include "velox/experimental/stateful/window/TimeWindowUtil.h"
#include "velox/type/Timestamp.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::stateful {

WindowJoin::WindowJoin(
    std::unique_ptr<exec::Operator> leftInput,
    std::unique_ptr<exec::Operator> rightInput,
    std::unique_ptr<exec::Operator> probe,
    std::vector<std::unique_ptr<StatefulOperator>> targets,
    std::shared_ptr<const StreamWindowJoinNode> node)
    : StatefulOperator(std::move(probe), std::move(targets)),
      leftInput_(std::move(leftInput)),
      rightInput_(std::move(rightInput)),
      node_(std::move(node)),
      probe_(static_cast<exec::NestedLoopJoinProbe*>(op().get())),
      leftWindowEndIndex_(node_->leftWindowEndIndex()),
      rightWindowEndIndex_(node_->rightWindowEndIndex()),
      ownWatermarkStatus_(2) {
  VELOX_CHECK_NOT_NULL(
      node_->keyedStateSpec(), "StreamWindowJoinNode needs a KeyedStateSpec");
}

void WindowJoin::initialize() {
  StatefulOperator::initialize();
  // The keyed state buckets store rows under (joinKey, windowEnd); every
  // pair of left/right rows in a fired window already satisfies all
  // key-equality conjuncts. When the probe carries no residual condition
  // (the planner dropped the tautological equalities) and the join is
  // INNER, firing reduces to a plain cross product.
  fastPath_ = node_->probe()->joinCondition() == nullptr &&
      isInnerJoin(node_->probe()->joinType());
  // The wrapped probe is driven exclusively by this operator (one private
  // build per due window, no peer probe operators), so its build-mismatch
  // flush skips the multi-driver peer election.
  probe_->markOperatorOwned();
  leftInput_->initialize();
  rightInput_->initialize();
}

void WindowJoin::initializeStateBackend(StateBackend* stateBackend) {
  const auto& spec = *node_->keyedStateSpec();
  keyedBackend_ = stateBackend->createKeyedStateBackend<RowContainerStateKey>(
      spec, memoryPool());
  VELOX_CHECK_EQ(
      spec.rightKeyChannels().size(),
      spec.keyChannels().size(),
      "WindowJoin needs the key channels of both inputs");
  leftKeySelector_ = std::make_unique<KeySelector>(
      keyedBackend_->keyIndex(), spec.keyChannels());
  rightKeySelector_ = std::make_unique<KeySelector>(
      keyedBackend_->keyIndex(), spec.rightKeyChannels());

  // Row-granularity list state: one element per stored row - a standalone
  // deep copy - serialized element-wise through the shared vector serde.
  // Flink's ListState<RowData> keyed by (remaining key, window end) is the
  // semantic counterpart. The element types are plan-side data: the wrapped
  // scan operators keep their output type protected.
  leftState_ =
      keyedBackend_->getOrCreateListState<WindowEndNamespace, RowVectorPtr>(
          ListStateDescriptor<RowVectorPtr>(
              "left-records",
              std::make_shared<ComplexVectorSerializer<RowVectorPtr>>(
                  node_->sources()[0]->outputType(), memoryPool()),
              memoryPool()));
  rightState_ =
      keyedBackend_->getOrCreateListState<WindowEndNamespace, RowVectorPtr>(
          ListStateDescriptor<RowVectorPtr>(
              "right-records",
              std::make_shared<ComplexVectorSerializer<RowVectorPtr>>(
                  node_->sources()[1]->outputType(), memoryPool()),
              memoryPool()));

  StatefulOperator::initializeStateBackend(stateBackend);
}

bool WindowJoin::isFinished() {
  return leftInput_->isFinished() && rightInput_->isFinished();
}

void WindowJoin::addInput(StreamElementPtr input) {
  // Two-input operators are pull-driven in production - advance() drains
  // both external streams - but the push interface stays a real input
  // path: the side is the record's node id (the producing scan's plan node
  // id, which is what both the pulled batches and any pushed records carry),
  // so the operator can be driven record-by-record outside the scan chain.
  auto record = std::dynamic_pointer_cast<StreamRecord>(input);
  VELOX_CHECK_NOT_NULL(record, "WindowJoin expects StreamRecord input");
  VELOX_CHECK_NOT_NULL(
      keyedBackend_, "WindowJoin needs initializeStateBackend before input");
  const auto& nodeId = input->nodeId();
  if (nodeId == leftInput_->planNodeId()) {
    fileRecord(
        *record,
        leftKeySelector_.get(),
        leftWindowEndIndex_,
        leftState_.get(),
        &leftLateRecordsDropped_);
  } else if (nodeId == rightInput_->planNodeId()) {
    fileRecord(
        *record,
        rightKeySelector_.get(),
        rightWindowEndIndex_,
        rightState_.get(),
        &rightLateRecordsDropped_);
  } else {
    VELOX_FAIL(
        "WindowJoin cannot route a record from node {}; expected one of the "
        "two input scans ({} / {})",
        nodeId,
        leftInput_->planNodeId(),
        rightInput_->planNodeId());
  }
  // Batch boundary: recycle key rows whose state entries are all gone.
  keyedBackend_->advance();
}

void WindowJoin::advance() {
  bool pulledAny = false;
  pulledAny |= processSide(
      leftInput_.get(),
      leftKeySelector_.get(),
      leftWindowEndIndex_,
      leftState_.get(),
      &leftLateRecordsDropped_);
  pulledAny |= processSide(
      rightInput_.get(),
      rightKeySelector_.get(),
      rightWindowEndIndex_,
      rightState_.get(),
      &rightLateRecordsDropped_);
  // Batch boundary: recycle key rows whose state entries are all gone, so
  // dead keys never survive a batch (the contract's step 7).
  if (keyedBackend_) {
    keyedBackend_->advance();
  }
  setSourceEmpty(!pulledAny);
}

void WindowJoin::close() {
  StatefulOperator::close();
  leftInput_->close();
  rightInput_->close();
  // Reverse wiring order: the handles and selectors go before the backend
  // that owns the key table they probe.
  leftState_.reset();
  rightState_.reset();
  leftKeySelector_.reset();
  rightKeySelector_.reset();
  keyedBackend_.reset();
}

void WindowJoin::processWatermark(int64_t timestamp, int32_t index) {
  // Fire the windows due under this operator's mirror of the combined
  // watermark before the base class propagates it: emitted records reach
  // downstream ahead of the watermark, Flink's timer-before-watermark
  // order. The base emitWatermark is not virtual (StatefulOperator.h), so
  // the merge state is mirrored here and handed to the base afterwards;
  // both copies see the same updates.
  if (ownWatermarkStatus_.updateWatermark(index, timestamp)) {
    lastCombinedWatermark_ = ownWatermarkStatus_.getCombinedWatermark();
    fireDueWindows(lastCombinedWatermark_);
  }
  StatefulOperator::processWatermark(timestamp, index);
}

void WindowJoin::processWatermarkStatus(bool idle, int32_t index) {
  // Mirror the idle flip: the merge skips idle partials, so an input going
  // idle can advance the combined watermark. The mirror must see it or the
  // lateness criterion and due index would diverge from the watermark the
  // base emits downstream. Fire whatever became due before the base
  // re-emits the combined watermark.
  if (ownWatermarkStatus_.updateStatus(index, idle)) {
    lastCombinedWatermark_ = ownWatermarkStatus_.getCombinedWatermark();
    fireDueWindows(lastCombinedWatermark_);
  }
  StatefulOperator::processWatermarkStatus(idle, index);
}

void WindowJoin::onStateRestored() {
  VELOX_CHECK(
      dueIndex_.empty() && scheduled_.empty(),
      "The due index must be rebuilt on a fresh (empty) schedule");
  const auto rebuild = [this](const std::shared_ptr<WindowListState>& state) {
    auto entries = std::dynamic_pointer_cast<
        HeapListState<RowContainerStateKey, WindowEndNamespace, RowVectorPtr>>(
        state);
    VELOX_CHECK_NOT_NULL(entries, "The join state must be a HeapListState");
    entries->forEachEntry(
        [this](const RowContainerStateKey& key, const WindowEndNamespace& ns) {
          registerDueWindow(key, ns.windowEnd());
        });
  };
  rebuild(leftState_);
  rebuild(rightState_);
  // Windows whose trigger time already passed under the last seen watermark
  // fire immediately: a restored operator does not wait for new input.
  if (lastCombinedWatermark_ > INT64_MIN) {
    fireDueWindows(lastCombinedWatermark_);
  }
}

bool WindowJoin::processSide(
    exec::Operator* input,
    KeySelector* keySelector,
    int windowEndIndex,
    WindowListState* state,
    uint64_t* lateRecordsDropped) {
  auto merged = input->getOutput();
  if (!merged) {
    return false;
  }
  // Split the merged input vector (user columns + trailing $row_kind) into
  // the record and its per-row kinds; an append-only input carries no
  // $row_kind column and splits to an append-only record.
  const auto record =
      StreamRecord::create(input->planNodeId(), std::move(merged));
  fileRecord(*record, keySelector, windowEndIndex, state, lateRecordsDropped);
  return true;
}

void WindowJoin::fileRecord(
    const StreamRecord& record,
    KeySelector* keySelector,
    int windowEndIndex,
    WindowListState* state,
    uint64_t* lateRecordsDropped) {
  const auto& inputVector = record.record();
  const auto& rowKinds = record.rowKind();
  for (auto& child : inputVector->children()) {
    child->loadedVector();
  }
  keySelector->probe(inputVector);
  const auto perRowKeys = keySelector->keys();

  const auto windowEndChild = inputVector->childAt(windowEndIndex).get();
  const auto windowEndBigint = windowEndChild->asFlatVector<int64_t>();
  const auto windowEndTimestamp = windowEndBigint == nullptr
      ? windowEndChild->asFlatVector<Timestamp>()
      : nullptr;
  VELOX_CHECK(
      windowEndBigint != nullptr || windowEndTimestamp != nullptr,
      "Window join needs a BIGINT or TIMESTAMP window-end column, got {}",
      windowEndChild->type()->toString());
  const auto numRows = inputVector->size();

  // Pre-read all window-end values (sequential access, local buffer - no
  // class member to avoid layout changes that break ODR with the planner).
  std::vector<int64_t> windowEnds(numRows);
  if (windowEndBigint != nullptr) {
    const auto* rawValues = windowEndBigint->rawValues();
    std::memcpy(windowEnds.data(), rawValues, numRows * sizeof(int64_t));
  } else {
    for (vector_size_t i = 0; i < numRows; ++i) {
      windowEnds[i] = windowEndTimestamp->valueAt(i).toMillis();
    }
  }

  // Batch processing: detect runs of consecutive rows with the same
  // (key, windowEnd) and process each run with ONE dictionary view and
  // ONE state add. The input is pre-grouped by key (keyed exchange),
  // so same-key rows are adjacent.
  vector_size_t runStart = 0;
  while (runStart < numRows) {
    const auto windowEnd = windowEnds[runStart];
    if (TimeWindowUtil::isWindowFired(
            windowEnd, lastCombinedWatermark_, shiftTimeZone_)) {
      ++*lateRecordsDropped;
      ++runStart;
      continue;
    }
    if (rowKinds != nullptr &&
        rowKinds->valueAt(runStart) != static_cast<int8_t>(RowKind::INSERT) &&
        rowKinds->valueAt(runStart) !=
            static_cast<int8_t>(RowKind::UPDATE_AFTER)) {
      VELOX_USER_FAIL(
          "WindowJoin received a retraction row (UPDATE_BEFORE or DELETE); "
          "window join cannot handle retraction input");
    }
    vector_size_t runEnd = runStart + 1;
    while (runEnd < numRows && windowEnds[runEnd] == windowEnd &&
           perRowKeys[runStart].hash() == perRowKeys[runEnd].hash()) {
      if (rowKinds != nullptr &&
          rowKinds->valueAt(runEnd) != static_cast<int8_t>(RowKind::INSERT) &&
          rowKinds->valueAt(runEnd) !=
              static_cast<int8_t>(RowKind::UPDATE_AFTER)) {
        VELOX_USER_FAIL(
            "WindowJoin received a retraction row (UPDATE_BEFORE or DELETE); "
            "window join cannot handle retraction input");
      }
      ++runEnd;
    }
    const auto runSize = runEnd - runStart;
    auto indices = allocateIndices(runSize, memoryPool());
    auto* rawIndices = indices->asMutable<vector_size_t>();
    if (runSize == 1) {
      rawIndices[0] = runStart;
    } else {
      for (vector_size_t i = 0; i < runSize; ++i) {
        rawIndices[i] = runStart + i;
      }
    }
    auto runView = std::make_shared<RowVector>(
        memoryPool(),
        inputVector->type(),
        BufferPtr(nullptr),
        runSize,
        std::vector<VectorPtr>(inputVector->childrenSize()));
    for (vector_size_t c = 0; c < inputVector->childrenSize(); ++c) {
      runView->childAt(c) = BaseVector::wrapInDictionary(
          nullptr, indices, runSize, inputVector->childAt(c));
    }
    runView->updateContainsLazyNotLoaded();
    state->add(
        perRowKeys[runStart],
        WindowEndNamespace(windowEnd),
        std::move(runView));
    registerDueWindow(perRowKeys[runStart], windowEnd);
    runStart = runEnd;
  }
}

RowVectorPtr WindowJoin::copyRow(const RowVectorPtr& input, vector_size_t row) {
  // Dictionary wrap: zero-copy 1-row view (avoids per-row allocation+copy)
  BufferPtr indices = allocateIndices(1, memoryPool());
  indices->asMutable<vector_size_t>()[0] = row;
  auto result = std::make_shared<RowVector>(
      memoryPool(),
      input->type(),
      nullptr,
      1,
      std::vector<VectorPtr>(input->childrenSize()));
  for (vector_size_t c = 0; c < input->childrenSize(); ++c) {
    result->childAt(c) =
        BaseVector::wrapInDictionary(nullptr, indices, 1, input->childAt(c));
  }
  result->updateContainsLazyNotLoaded();
  return result;
}

void WindowJoin::registerDueWindow(
    const RowContainerStateKey& key,
    int64_t windowEnd) {
  if (windowEnd == INT64_MAX) {
    // The min-window-end flag never fires (isWindowFired special-cases it);
    // its rows are stored but never scheduled.
    return;
  }
  if (!scheduled_.emplace(key.row(), windowEnd).second) {
    return; // Already scheduled: later rows of the same window join it.
  }
  const auto dueTs =
      TimeWindowUtil::toEpochMillsForTimer(windowEnd - 1, shiftTimeZone_);
  dueIndex_[dueTs].push_back(DueWindow{key, windowEnd});
}

void WindowJoin::fireDueWindows(int64_t watermark) {
  while (!dueIndex_.empty() && dueIndex_.begin()->first <= watermark) {
    auto batch = std::move(dueIndex_.begin()->second);
    dueIndex_.erase(dueIndex_.begin());
    for (const auto& due : batch) {
      if (fastPath_) {
        fastFireWindow(due);
      } else {
        fireWindow(due);
      }
    }
  }
  if (fastPath_) {
    flushCrossOutputs();
  }
}

void WindowJoin::fireWindow(const DueWindow& due) {
  const WindowEndNamespace ns(due.windowEnd);
  auto leftRows = leftState_->get(due.key, ns);
  auto rightRows = rightState_->get(due.key, ns);

  // One complete probe lifecycle for the window. The build side is the
  // window's right rows; each left row is fed as its own single-row batch -
  // the batch shape the streaming path always produces - and end-of-input
  // flushes the right/full outer mismatches. resetForNextWindow rewinds
  // the wrapped probe's state so the same instance serves every window:
  // one memory pool, one compiled join condition.
  probe_->resetForNextWindow(rightRows);
  // Split multi-row stored vectors back to individual rows for the probe:
  // the NestedLoopJoinProbe's per-input lifecycle produces output batches
  // per input, and UT tests assert on that batching shape.
  for (const auto& leftVec : leftRows) {
    const auto vecRows = leftVec->size();
    if (vecRows == 1) {
      probe_->addInput(leftVec);
      drainProbeOutput(*probe_);
    } else {
      for (vector_size_t row = 0; row < vecRows; ++row) {
        probe_->addInput(copyRow(leftVec, row));
        drainProbeOutput(*probe_);
      }
    }
  }
  probe_->noMoreInput();
  drainProbeOutput(*probe_);

  // Emission and clearing are atomic per window, as Flink's onEventTime.
  // Erase the schedule entry while the key row is still alive, then clear
  // the window's state; once all entries are gone the backend recycles
  // the key row, which would leave a stale pointer in the schedule set.
  scheduled_.erase({due.key.row(), due.windowEnd});
  leftState_->remove(due.key, ns);
  rightState_->remove(due.key, ns);
}

void WindowJoin::fastFireWindow(const DueWindow& due) {
  const WindowEndNamespace ns(due.windowEnd);
  auto leftRows = leftState_->get(due.key, ns);
  auto rightRows = rightState_->get(due.key, ns);
  if (!leftRows.empty() && !rightRows.empty()) {
    const auto L = leftRows.size() == 1
        ? leftRows[0]
        : TimeWindowUtil::mergeVectors(
              std::list<RowVectorPtr>(leftRows.begin(), leftRows.end()),
              memoryPool());
    const auto R = rightRows.size() == 1
        ? rightRows[0]
        : TimeWindowUtil::mergeVectors(
              std::list<RowVectorPtr>(rightRows.begin(), rightRows.end()),
              memoryPool());
    // INNER semantics: a NULL in any key column makes the equality
    // unsatisfiable, so null-key rows can never join - drop them instead
    // of probing. The common case (no nulls anywhere) costs one
    // mayHaveNulls() check per key column.
    const auto Lf = dropNullKeys(L, node_->keyedStateSpec()->keyChannels());
    const auto Rf =
        dropNullKeys(R, node_->keyedStateSpec()->rightKeyChannels());
    if (Lf != nullptr && Rf != nullptr && Lf->size() > 0 && Rf->size() > 0) {
      const auto nl = Lf->size();
      const auto nr = Rf->size();
      const auto total = nl * nr;
      // Stored rows carry a trailing $row_kind column; the join output
      // projection is bounded by the probe plan's input widths.
      const auto lw = node_->probe()->sources()[0]->outputType()->size();
      const auto rw = node_->probe()->sources()[1]->outputType()->size();
      auto leftIdx = allocateIndices(total, memoryPool());
      auto rightIdx = allocateIndices(total, memoryPool());
      auto* li = leftIdx->asMutable<vector_size_t>();
      auto* ri = rightIdx->asMutable<vector_size_t>();
      for (vector_size_t i = 0, k = 0; i < nl; ++i) {
        for (vector_size_t j = 0; j < nr; ++j, ++k) {
          li[k] = i;
          ri[k] = j;
        }
      }
      std::vector<VectorPtr> children(lw + rw);
      for (vector_size_t c = 0; c < lw; ++c) {
        children[c] = BaseVector::wrapInDictionary(
            nullptr, leftIdx, total, Lf->childAt(c));
      }
      for (vector_size_t c = 0; c < rw; ++c) {
        children[lw + c] = BaseVector::wrapInDictionary(
            nullptr, rightIdx, total, Rf->childAt(c));
      }
      auto out = std::make_shared<RowVector>(
          memoryPool(),
          node_->outputType(),
          BufferPtr(nullptr),
          total,
          std::move(children));
      out->updateContainsLazyNotLoaded();
      crossOutputs_.push_back(std::move(out));
      crossOutputRows_ += total;
      if (crossOutputs_.size() >= 64 || crossOutputRows_ >= 4096) {
        flushCrossOutputs();
      }
    }
  }
  // Firing clears the window's state, as Flink's onEventTime does. Erase
  // the schedule entry first, while the key row is still alive: removing
  // the state entries lets the backend recycle the row, and a recycled
  // pointer left in the set could suppress a future window registration.
  scheduled_.erase({due.key.row(), due.windowEnd});
  leftState_->remove(due.key, ns);
  rightState_->remove(due.key, ns);
}

RowVectorPtr WindowJoin::dropNullKeys(
    const RowVectorPtr& v,
    const std::vector<uint32_t>& chans) {
  bool mayNull = false;
  for (auto ch : chans) {
    if (v->childAt(ch)->mayHaveNulls()) {
      mayNull = true;
      break;
    }
  }
  if (!mayNull) {
    return v;
  }
  std::vector<vector_size_t> keep;
  keep.reserve(v->size());
  for (vector_size_t row = 0; row < v->size(); ++row) {
    bool null = false;
    for (auto ch : chans) {
      if (v->childAt(ch)->isNullAt(row)) {
        null = true;
        break;
      }
    }
    if (!null) {
      keep.push_back(row);
    }
  }
  if (keep.size() == v->size()) {
    return v;
  }
  if (keep.empty()) {
    return nullptr;
  }
  auto indices = allocateIndices(keep.size(), memoryPool());
  std::copy(keep.begin(), keep.end(), indices->asMutable<vector_size_t>());
  std::vector<VectorPtr> children(v->childrenSize());
  for (vector_size_t c = 0; c < v->childrenSize(); ++c) {
    children[c] = BaseVector::wrapInDictionary(
        nullptr, indices, keep.size(), v->childAt(c));
  }
  auto result = std::make_shared<RowVector>(
      memoryPool(),
      v->type(),
      BufferPtr(nullptr),
      keep.size(),
      std::move(children));
  result->updateContainsLazyNotLoaded();
  return result;
}

void WindowJoin::flushCrossOutputs() {
  if (crossOutputs_.empty()) {
    return;
  }
  RowVectorPtr merged = crossOutputs_.size() == 1
      ? crossOutputs_[0]
      : TimeWindowUtil::mergeVectors(
            std::list<RowVectorPtr>(
                std::make_move_iterator(crossOutputs_.begin()),
                std::make_move_iterator(crossOutputs_.end())),
            memoryPool());
  crossOutputs_.clear();
  crossOutputRows_ = 0;
  pushOutput(
      std::make_shared<StreamRecord>(getPlanNodeId(), std::move(merged)));
}

void WindowJoin::drainProbeOutput(exec::NestedLoopJoinProbe& probe) {
  while (auto output = probe.getOutput()) {
    pushOutput(
        std::make_shared<StreamRecord>(getPlanNodeId(), std::move(output)));
  }
}

} // namespace facebook::velox::stateful
