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
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "velox/exec/NestedLoopJoinProbe.h"
#include "velox/experimental/stateful/CombinedWatermarkStatus.h"
#include "velox/experimental/stateful/KeySelector.h"
#include "velox/experimental/stateful/RowKind.h"
#include "velox/experimental/stateful/StatefulOperator.h"
#include "velox/experimental/stateful/StatefulPlanNode.h"
#include "velox/experimental/stateful/StreamElement.h"
#include "velox/experimental/stateful/state/HeapKeyedStateBackend.h"
#include "velox/experimental/stateful/state/HeapListState.h"
#include "velox/experimental/stateful/state/Namespace.h"

namespace facebook::velox::stateful {

/// Window TVF join over the operator-owned keyed state backend: the
/// counterpart of Flink's WindowJoinOperator family (window TVF syntax,
/// planner node StreamExecWindowJoin). Both inputs are windowed streams;
/// rows pair only inside their window, and a window fires exactly once,
/// when the two-input combined watermark passes its trigger time
/// (window end - 1), emitting every match in one batch and clearing both
/// sides' state for that window. Output is always INSERT-only.
///
/// State: two row-granularity list states ("left-records" / "right-records")
/// addressed by (remaining equi-join key, window end); the window equality
/// pairs are stripped from the join condition by the planner and carried
/// structurally by the namespace. Triggering: an in-memory due index
/// (dueTs -> (key, window end)) instead of the legacy timer service; the
/// index is derived from the state and rebuilt on restore, so it needs no
/// persistence of its own. Lateness: rows whose window already fired under
/// the combined watermark are dropped and counted per side, as in Flink.
/// Join execution: each due window materializes both sides' rows through a
/// fresh NestedLoopJoinProbe built from the plan's probe node - one complete
/// probe lifecycle per window (build data, one-row probe batches as in the
/// streaming path, end-of-input flush for the outer join mismatches).
class WindowJoin : public StatefulOperator {
 public:
  WindowJoin(
      std::unique_ptr<exec::Operator> leftInput,
      std::unique_ptr<exec::Operator> rightInput,
      std::unique_ptr<exec::Operator> probe,
      std::vector<std::unique_ptr<StatefulOperator>> targets,
      std::shared_ptr<const StreamWindowJoinNode> node);

  void initialize() override;

  void initializeStateBackend(StateBackend* stateBackend) override;

  bool isFinished() override;

  void addInput(StreamElementPtr input) override;

  void advance() override;

  void close() override;

  void processWatermark(int64_t timestamp, int32_t index) override;

  void processWatermarkStatus(bool idle, int32_t index) override;

  std::string name() const override {
    return "WindowJoin";
  }

  /// The keyed state backend this operator owns. Checkpoint drivers snapshot
  /// and restore the join state through it; the task-level wiring that
  /// persists the checkpoint bytes is delivered separately.
  HeapKeyedStateBackend<RowContainerStateKey>& keyedStateBackend() {
    return *keyedBackend_;
  }

  /// Rebuilds the due index from the restored list-state entries. Call once
  /// after a backend restore, before feeding input: every restored window is
  /// scheduled at its trigger time, and windows already due under the last
  /// seen watermark fire immediately.
  void onStateRestored();

  /// Read-only lateness accounting: the per-side counts of rows dropped as
  /// late (the counters the drop path in fileRecord increments).
  uint64_t leftLateRecordsDropped() const {
    return leftLateRecordsDropped_;
  }

  uint64_t rightLateRecordsDropped() const {
    return rightLateRecordsDropped_;
  }

 protected:
  int numInputs() const override {
    return 2;
  }

  bool needsFinishDrain() const override {
    return false;
  }

 private:
  // One (key, window end) pair scheduled to fire at dueTs. The key is the
  // selector-resolved state key; its RowContainer row stays alive while the
  // window's state entries exist.
  struct DueWindow {
    RowContainerStateKey key;
    int64_t windowEnd;
  };

  using WindowListState =
      ListState<RowContainerStateKey, WindowEndNamespace, RowVectorPtr>;

  // Pulls one batch from one input side and files its rows into the side's
  // list state, window by window: lateness check, per-row kind check, deep
  // row copy, due-index registration. Returns whether a batch was pulled.
  bool processSide(
      exec::Operator* input,
      KeySelector* keySelector,
      int windowEndIndex,
      WindowListState* state,
      uint64_t* lateRecordsDropped);

  // Files one record's rows into a side's list state. The shared body of
  // both input paths: advance() pulls merged batches from the external
  // streams, addInput() receives pushed records (the test surface - and
  // any future push-driven chain) routed by the record's node id.
  void fileRecord(
      const StreamRecord& record,
      KeySelector* keySelector,
      int windowEndIndex,
      WindowListState* state,
      uint64_t* lateRecordsDropped);

  // A standalone deep copy of row 'row': one list-state element, independent
  // of the reused input batch buffers.
  RowVectorPtr copyRow(const RowVectorPtr& input, vector_size_t row);

  // Schedules (key, windowEnd) at its trigger time, unless already
  // scheduled: registration is idempotent while later batches add rows to
  // the same window.
  void registerDueWindow(const RowContainerStateKey& key, int64_t windowEnd);

  // Fires every window whose trigger time has passed under 'watermark';
  // emitted records reach downstream before the watermark itself.
  void fireDueWindows(int64_t watermark);

  // Joins and clears one (key, window): both sides' rows through a fresh
  // probe lifecycle, then remove the state entries and the schedule mark.
  void fireWindow(const DueWindow& due);

  // Drains a probe's output into the downstream stream as append-only
  // records (the join never emits retractions).
  void drainProbeOutput(exec::NestedLoopJoinProbe& probe);

  const std::unique_ptr<exec::Operator> leftInput_;
  const std::unique_ptr<exec::Operator> rightInput_;
  std::shared_ptr<const StreamWindowJoinNode> node_;
  // The wrapped probe operator runs one complete lifecycle per due window
  // (resetForNextWindow rewinds its state); it is never driven by a driver
  // pipeline, so the join owns it exclusively.
  exec::NestedLoopJoinProbe* probe_;
  const int leftWindowEndIndex_;
  const int rightWindowEndIndex_;
  // UTC (0) on this base: TimeWindowUtil's timer math is the identity today
  // and the LTZ plumbing remains limitation L4 of the design.
  const int shiftTimeZone_ = 0;

  std::unique_ptr<HeapKeyedStateBackend<RowContainerStateKey>> keyedBackend_;
  std::unique_ptr<KeySelector> leftKeySelector_;
  std::unique_ptr<KeySelector> rightKeySelector_;
  std::shared_ptr<WindowListState> leftState_;
  std::shared_ptr<WindowListState> rightState_;

  // The operator's own mirror of the two-input combined watermark. The base
  // class merges and emits internally, but its emitWatermark is not virtual
  // (StatefulOperator.h), so the join keeps this second copy - fed the same
  // updates - to drive the due index and the lateness criterion, then hands
  // the update to the base so records still precede the watermark.
  CombinedWatermarkStatus ownWatermarkStatus_;
  int64_t lastCombinedWatermark_ = INT64_MIN;

  // Due index: trigger time -> the windows due at it, ordered by time.
  std::map<int64_t, std::vector<DueWindow>> dueIndex_;
  // (key row, window end) pairs already scheduled.
  std::set<std::pair<const char*, int64_t>> scheduled_;

  uint64_t leftLateRecordsDropped_ = 0;
  uint64_t rightLateRecordsDropped_ = 0;
};

} // namespace facebook::velox::stateful
