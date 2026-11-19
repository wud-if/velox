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

#include <memory>
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/stateful/InternalTimerService.h"
#include "velox/experimental/stateful/KeySelector.h"
#include "velox/experimental/stateful/StatefulOperator.h"
#include "velox/experimental/stateful/StreamElement.h"
#include "velox/experimental/stateful/TimerHeapInternalTimer.h"
#include "velox/experimental/stateful/Triggerable.h"
#include "velox/experimental/stateful/window/SliceAssigner.h"
#include "velox/experimental/stateful/window/WindowBuffer.h"

namespace facebook::velox::stateful {

/// This class is related to XXXWindowAggProcessor in Flink.
/// Its work includes both WindowAggOperator and XXXWindowAggOperator.
class WindowAggregator : public StatefulOperator,
                         public Triggerable<int64_t, int64_t> {
 public:
  WindowAggregator(
      std::unique_ptr<exec::Operator> localAggregator,
      std::unique_ptr<exec::Operator> globalAggregator,
      std::vector<std::unique_ptr<StatefulOperator>> targets,
      std::unique_ptr<KeySelector> keySelector,
      std::unique_ptr<SliceAssigner> sliceAssigner,
      const int64_t windowInterval,
      const bool useDayLightSaving,
      const bool isEventTime,
      const int32_t windowStartIndex,
      const int32_t windowEndIndex,
      bool inputTagged);

  void initialize() override;

  void initializeStateBackend(StateBackend* stateBackend) override;

  void initializeState() override;

  void addInput(StreamElementPtr input) override;

  void advance() override;

  void close() override;

  std::string name() const override {
    return "WindowAggregator";
  }

  void processWatermark(int64_t timestamp) override {
    // Fire windows before forwarding the watermark downstream (see
    // LocalWindowAggregator::processWatermark for the ordering rationale).
    processWatermarkInternal(timestamp);
    StatefulOperator::processWatermark(timestamp);
  }

  void processWatermark(int64_t timestamp, int32_t index) override {
    // Fire windows for the incoming watermark BEFORE the base class
    // forwards the combined watermark downstream, so that the fired
    // records precede the watermark in the output stream (see
    // LocalWindowAggregator::processWatermark for the rationale).
    processWatermarkInternal(timestamp);
    StatefulOperator::processWatermark(timestamp, index);
  }

  void onEventTime(
      std::shared_ptr<TimerHeapInternalTimer<int64_t, int64_t>> timer) override;

  void onProcessingTime(
      std::shared_ptr<TimerHeapInternalTimer<int64_t, int64_t>> timer) override;

  void onProcessingTime(int64_t timestamp) override;

 private:
  void processWatermarkInternal(int64_t timestamp);

  /// Flushes pendingFireOutputs_ as one merged output batch. Called after
  /// windowTimerService_->advanceWatermark() completes (all timers for the
  /// current watermark have fired and collected their outputs).
  void flushPendingFireOutputs();

  /// Session windows: merges the fired slice's local accumulator into the
  /// live (unfired) windows of the key that overlap the slice's tentative
  /// window [sliceEnd - gap, sliceEnd), merging overlapping windows into
  /// one (Flink session semantics).
  void mergeSessionWindow(int64_t key, int64_t sliceEnd, RowVectorPtr localAcc);

  void onTimer(std::shared_ptr<TimerHeapInternalTimer<int64_t, int64_t>> timer);

  template <typename K>
  void fireWindow(const K& key, int64_t timerTimestamp, int64_t windowEnd);

  template <typename K>
  void clearWindow(const K& key, int64_t timerTimestamp, int64_t windowEnd);

  std::unique_ptr<exec::Operator> localAggregator_;
  std::unique_ptr<KeySelector> keySelector_;
  std::unique_ptr<SliceAssigner> sliceAssigner_;
  WindowBufferPtr windowBuffer_;
  const int64_t windowInterval_;
  const bool useDayLightSaving_;
  const int shiftTimeZone_ = 0; // TODO: support time zone shift
  const bool isEventTime_ = true;
  const int32_t windowStartIndex_ = -1;
  const int32_t windowEndIndex_ = -1;

  /// True when the input rows carry the local aggregator's slice-end tag
  /// in their last column (two-phase global); false for single-phase input
  /// (raw rows, e.g. session windows).
  bool inputTagged_ = true;
  /// Session windows: live (unfired) windows per key, window end -> start.
  /// In-memory only: rebuilt as windows accumulate; fired windows are
  /// erased. Not part of the checkpoint (the e2e checkpoint path is out
  /// of scope, limitation L2).
  std::unordered_map<int64_t, std::map<int64_t, int64_t>> sessionLiveWindows_;
  /// Batch fireWindow: outputs collected from each timer's fireWindow
  /// during one advanceWatermark pass, flushed as one merged batch after
  /// all timers for that watermark have fired.
  std::list<RowVectorPtr> pendingFireOutputs_;

  /// Tumble/hop fast path, input side: whole input batches (already
  /// local-aggregated, slice-end tagged) bucketed by slice end, held
  /// until every window containing the slice has fired. Feeding these to
  /// the aggregation operators in key-sorted chunks replaces the per-key
  /// splitting, keyed state and timers of the generic path.
  struct PendingSlice {
    std::list<RowVectorPtr> batches;
    /// Next window end of this slice that has not been emitted yet
    /// (for tumble the only window is the slice end itself; a hop slice
    /// walks [sliceEnd, sliceEnd + size - slide] stepping by the slide).
    int64_t nextWindow;
  };
  std::map<int64_t, PendingSlice> pendingWholeBatches_;

  RowVectorPtr input_;
  int64_t currentProgress_ = 0;
  int64_t nextTriggerWatermark_ = 0;
  int64_t lastTriggeredProcessingTime_ = 0;
  std::shared_ptr<ValueState<int64_t, int64_t, RowVectorPtr>> windowState_;
  std::shared_ptr<InternalTimerService<int64_t, int64_t>> windowTimerService_;
};

} // namespace facebook::velox::stateful
