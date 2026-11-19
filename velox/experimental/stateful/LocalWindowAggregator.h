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

#include <list>
#include <map>

#include "velox/experimental/stateful/KeySelector.h"
#include "velox/experimental/stateful/StatefulOperator.h"
#include "velox/experimental/stateful/StreamElement.h"
#include "velox/experimental/stateful/window/SliceAssigner.h"
#include "velox/experimental/stateful/window/WindowBuffer.h"

namespace facebook::velox::stateful {

/// It is related to LocalSlicingWindowAggOperator in Flink.
/// It aggregates the records when window fires.
class LocalWindowAggregator : public StatefulOperator {
 public:
  LocalWindowAggregator(
      std::unique_ptr<exec::Operator> op,
      std::vector<std::unique_ptr<StatefulOperator>> targets,
      std::unique_ptr<KeySelector> keySelector,
      std::unique_ptr<SliceAssigner> sliceAssigner,
      const int64_t windowInterval,
      const bool useDayLightSaving,
      RowTypePtr outputType);

  void addInput(StreamElementPtr input) override;

  void advance() override;

  void close() override;

  void processWatermark(int64_t timestamp) override {
    // Fire windows BEFORE forwarding the watermark downstream, so that the
    // fired records precede the watermark element in the task output. If the
    // watermark is forwarded first, the downstream window operator sees it
    // before the window data, fires empty windows and then drops the
    // late-arriving data, losing all output.
    processWatermarkInternal(timestamp);
    StatefulOperator::processWatermark(timestamp);
  }

  std::string name() const override {
    return "LocalWindowAggregator";
  }

 private:
  void processWatermarkInternal(int64_t timestamp);
  RowVectorPtr addWindowEndToVector(RowVectorPtr vector, int64_t sliceEnd);

  std::unique_ptr<KeySelector> keySelector_;
  std::unique_ptr<SliceAssigner> sliceAssigner_;
  WindowBufferPtr windowBuffer_;

  /// Tumble fast path: whole input batches bucketed by slice end, awaiting
  /// the watermark that fires their slice. Feeding these to the
  /// aggregation operator in key-sorted chunks replaces the per-key
  /// splitting and per-key operator cycles of the generic path.
  std::map<int64_t, std::list<RowVectorPtr>> pendingWholeBatches_;
  const int64_t windowInterval_;
  const bool useDayLightSaving_;
  const int shiftTimeZone_ = 0; // TODO: support time zone shift
  RowTypePtr outputType_;

  RowVectorPtr input_;
  int64_t currentWatermark_ = 0;
  int64_t nextTriggerWatermark_ = 0;
};
} // namespace facebook::velox::stateful
