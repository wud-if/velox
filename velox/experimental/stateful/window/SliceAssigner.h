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

#include "velox/experimental/stateful/KeySelector.h"
#include "velox/experimental/stateful/window/Window.h"

namespace facebook::velox::stateful {

/// This class is relevant to Flink SliceAssigner.
class SliceAssigner {
 public:
  SliceAssigner(
      std::unique_ptr<KeySelector> keySelector,
      int64_t size,
      int64_t step,
      int64_t offset,
      WindowType windowType,
      int rowtimeIndex);

  std::map<int64_t, RowVectorPtr> assignSliceEnd(const RowVectorPtr& input);

  /// Groups the input rows by the slice end tag carried in the LAST column of
  /// the input. The local window aggregator appends the slice end as the last
  /// column of every pre-aggregated batch; the global window aggregator must
  /// use that value directly as the slice end. Re-deriving a window from the
  /// tag via assignSliceEnd() would shift the slice by one window
  /// (align(end) + size == end + size for an already aligned end).
  std::map<int64_t, RowVectorPtr> assignSliceEndFromTag(
      const RowVectorPtr& input);

  int64_t getLastWindowEnd(int64_t sliceEnd);

  int64_t getWindowStart(int64_t windowEnd);

  // Iterable<Long> expiredSlices(int64_t windowEnd);

  int64_t getSliceEndInterval();

  /// Index of the rowtime column this assigner slices on (-1 when the
  /// assigner runs on processing time).
  int rowtimeIndex() const {
    return rowtimeIndex_;
  }

  WindowType windowType() const {
    return windowType_;
  }

  int64_t size() const {
    return size_;
  }

  int64_t step() const {
    return step_;
  }

 private:
  const std::unique_ptr<KeySelector> keySelector_;
  const int64_t size_;
  const int64_t step_;
  const int64_t offset_;
  const WindowType windowType_;
  int64_t sliceSize_;
  int rowtimeIndex_;
};

} // namespace facebook::velox::stateful
