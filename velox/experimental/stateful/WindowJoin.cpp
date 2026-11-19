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

  // The window-end column arrives as TIMESTAMP(3) (velox Timestamp) or
  // BIGINT depending on the bridge path; read epoch millis from either.
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
  for (vector_size_t row = 0; row < numRows; ++row) {
    const auto windowEnd = windowEndBigint != nullptr
        ? windowEndBigint->valueAt(row)
        : windowEndTimestamp->valueAt(row).toMillis();
    // Lateness first, as in Flink: a late row is dropped silently, ahead of
    // any other check.
    if (TimeWindowUtil::isWindowFired(
            windowEnd, lastCombinedWatermark_, shiftTimeZone_)) {
      ++*lateRecordsDropped;
      continue;
    }
    // Accumulate rows (INSERT or UPDATE_AFTER, as in Flink
    // RowDataUtil.isAccumulateMsg) take the add path below. Retraction rows
    // (UPDATE_BEFORE or DELETE) fail loud: window join cannot handle a
    // retraction input stream.
    if (rowKinds != nullptr &&
        rowKinds->valueAt(row) != static_cast<int8_t>(RowKind::INSERT) &&
        rowKinds->valueAt(row) != static_cast<int8_t>(RowKind::UPDATE_AFTER)) {
      VELOX_USER_FAIL(
          "WindowJoin received a retraction row (UPDATE_BEFORE or DELETE); "
          "window join cannot handle retraction input");
    }
    state->add(
        perRowKeys[row],
        WindowEndNamespace(windowEnd),
        copyRow(inputVector, row));
    registerDueWindow(perRowKeys[row], windowEnd);
  }
}

RowVectorPtr WindowJoin::copyRow(const RowVectorPtr& input, vector_size_t row) {
  auto copy = BaseVector::create(input->type(), 1, memoryPool());
  copy->copy(input.get(), 0, row, 1);
  return std::static_pointer_cast<RowVector>(copy);
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
      fireWindow(due);
    }
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
  for (const auto& leftRow : leftRows) {
    probe_->addInput(leftRow);
    drainProbeOutput(*probe_);
  }
  probe_->noMoreInput();
  drainProbeOutput(*probe_);

  // Emission and clearing are atomic per window, as Flink's onEventTime.
  leftState_->remove(due.key, ns);
  rightState_->remove(due.key, ns);
  scheduled_.erase({due.key.row(), due.windowEnd});
}

void WindowJoin::drainProbeOutput(exec::NestedLoopJoinProbe& probe) {
  while (auto output = probe.getOutput()) {
    pushOutput(
        std::make_shared<StreamRecord>(getPlanNodeId(), std::move(output)));
  }
}

} // namespace facebook::velox::stateful
