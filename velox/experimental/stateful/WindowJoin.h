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
#include <vector>

#include "velox/core/PlanNode.h"
#include "velox/exec/NestedLoopJoinProbe.h"
#include "velox/experimental/stateful/InternalTimerService.h"
#include "velox/experimental/stateful/KeySelector.h"
#include "velox/experimental/stateful/StatefulOperator.h"
#include "velox/experimental/stateful/StatefulPlanNode.h"
#include "velox/experimental/stateful/StreamElement.h"
#include "velox/experimental/stateful/TimerHeapInternalTimer.h"
#include "velox/experimental/stateful/Triggerable.h"
#include "velox/experimental/stateful/join/JoinRecordStateView.h"

namespace facebook::velox::stateful {

/// Streaming window join operator. Joins two streams whose rows carry
/// window_end columns: rows are bucketed by (join key, window_end), and each
/// bucket is joined exactly once when the combined watermark passes the
/// window end. The join computation itself is delegated to a velox
/// NestedLoopJoinProbe, which handles the join type (inner/left/right/full/
/// semi/anti), the join condition (equi and non-equi), and vectorized
/// evaluation.
class WindowJoin : public StatefulOperator,
                   public Triggerable<int64_t, int64_t> {
 public:
  WindowJoin(
      std::unique_ptr<exec::Operator> leftInput,
      std::unique_ptr<exec::Operator> rightInput,
      std::unique_ptr<KeySelector> leftKeySelector,
      std::unique_ptr<KeySelector> rightKeySelector,
      std::unique_ptr<exec::Operator> probe,
      std::vector<std::unique_ptr<StatefulOperator>> targets,
      core::JoinType joinType,
      int leftWindowEndIndex,
      int rightWindowEndIndex,
      RowTypePtr leftInputType,
      RowTypePtr rightInputType);

  void initialize() override;

  void initializeState() override;

  bool isFinished() override;

  void addInput(StreamElementPtr input) override;

  void advance() override;

  void close() override;

  std::string name() const override {
    return "WindowJoin";
  }

  /// Dual-input watermark: combines per-input watermarks (min) and advances
  /// the timer service, which triggers bucket joins for windows that have
  /// closed.
  void processWatermark(int64_t timestamp, int32_t index) override;

  void onEventTime(
      std::shared_ptr<TimerHeapInternalTimer<int64_t, int64_t>> timer) override;

  void onProcessingTime(
      std::shared_ptr<TimerHeapInternalTimer<int64_t, int64_t>> timer) override;

 protected:
  int numInputs() const override {
    return 2;
  }

  bool needsFinishDrain() const override {
    return false;
  }

 private:
  /// Joins the bucket for (key, windowEnd): reads both sides' state,
  /// dispatches on joinType_, delegates to the probe, and clears state.
  void join(uint32_t key, int64_t windowEnd);

  /// Pulls one batch from one input side, filters late rows, partitions by
  /// join key, then by window end, and stores into state with a timer.
  void processData(
      exec::Operator* input,
      KeySelector* keySelector,
      int windowEndIndex,
      ListState<uint32_t, int64_t, RowVectorPtr>* state);

  /// Drops rows whose window has already been fired (watermark >= windowEnd).
  RowVectorPtr filterWindowFiredRows(
      RowVectorPtr& input, int windowEndIndex);

  /// Partitions a (key-tagged) vector by the window_end column. Fast path:
  /// if all rows share one windowEnd, the vector is passed through as-is.
  std::map<int64_t, RowVectorPtr> partitionWindowData(
      RowVectorPtr& input, int windowEndIndex);

  /// Inner/outer join via NestedLoopJoinProbe: build side set once, probe
  /// side fed vector by vector.
  void probeAndCollect(
      const std::vector<RowVectorPtr>& leftValues,
      const std::vector<RowVectorPtr>& rightValues);

  /// Emits rows with null padding on the other side (for outer joins where
  /// one side's bucket is empty).
  void emitNullPadded(
      const std::vector<RowVectorPtr>& values, bool padRight);

  /// Semi/Anti join: emits probe-side rows that have (semi) or do not have
  /// (anti) a match on the build side. Uses the probe's kLeftSemiFilter /
  /// kAnti join types when available; falls back to a kLeft join with
  /// post-filtering otherwise.
  void probeSemi(
      const std::vector<RowVectorPtr>& leftValues,
      const std::vector<RowVectorPtr>& rightValues,
      bool anti);

  /// Emits all rows as-is (for anti join when the right bucket is empty).
  void emitAll(const std::vector<RowVectorPtr>& values);

  /// Wraps a subset of rows (by indices) into a new RowVector.
  RowVectorPtr wrapRows(
      const RowVectorPtr& input,
      const std::vector<vector_size_t>& indices);

  /// Advances the timer service to the given combined watermark.
  void advanceTimers(int64_t combinedWatermark);

  const std::unique_ptr<exec::Operator> leftInput_;
  const std::unique_ptr<exec::Operator> rightInput_;
  const std::unique_ptr<KeySelector> leftKeySelector_;
  const std::unique_ptr<KeySelector> rightKeySelector_;
  exec::NestedLoopJoinProbe* probe_;
  const core::JoinType joinType_;
  std::shared_ptr<ListState<uint32_t, int64_t, RowVectorPtr>> leftWindowState_;
  std::shared_ptr<ListState<uint32_t, int64_t, RowVectorPtr>> rightWindowState_;
  const int leftWindowEndIndex_;
  const int rightWindowEndIndex_;
  std::shared_ptr<InternalTimerService<int64_t, int64_t>> timerService_;
  int64_t lastTriggeredWatermark_ = -1;

  // Cached types for null padding.
  RowTypePtr leftInputType_;
  RowTypePtr rightInputType_;
  memory::MemoryPool* pool_ = nullptr;
};

} // namespace facebook::velox::stateful
