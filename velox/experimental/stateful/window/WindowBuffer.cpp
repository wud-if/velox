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
#include "velox/experimental/stateful/window/WindowBuffer.h"
#include <cstdint>
#include "velox/experimental/stateful/window/TimeWindowUtil.h"

namespace facebook::velox::stateful {

void RecordsWindowBuffer::addElement(
    uint32_t key,
    int64_t sliceEnd,
    RowVectorPtr& element) {
  minSliceEnd_ = std::min(sliceEnd, minSliceEnd_);
  WindowKey windowKey(key, sliceEnd);
  auto it = buffer_.find(windowKey);
  if (it != buffer_.end()) {
    // If the key already exists, we can append the element to the existing
    // list.
    it->second.push_back(element);
  } else {
    // If the key does not exist, we create a new list and add the element.
    std::list<RowVectorPtr> newList;
    newList.push_back(element);
    buffer_[windowKey] = std::move(newList);
  }
}

std::unordered_map<WindowKey, std::list<RowVectorPtr>>
RecordsWindowBuffer::advanceProgress(int64_t progress) {
  // Flush-and-erase semantics, mirroring Flink's RecordsWindowBuffer: return
  // ONLY the slices whose window has actually fired and remove them from the
  // buffer. The previous implementation returned the whole buffer without
  // clearing it, so already-emitted slices were re-emitted on every
  // watermark crossing (and unfired slices were emitted prematurely);
  // downstream received duplicate partial aggregates and double counted
  // them (verified in e2e: ids emitted at a premature flush plus their
  // window's full flush produced COUNT=2).
  std::unordered_map<WindowKey, std::list<RowVectorPtr>> fired;
  for (auto it = buffer_.begin(); it != buffer_.end();) {
    if (TimeWindowUtil::isWindowFired(
            it->first.window(), progress, shiftTimeZone_)) {
      fired.emplace(it->first, std::move(it->second));
      it = buffer_.erase(it);
    } else {
      ++it;
    }
  }
  minSliceEnd_ = INT64_MAX;
  for (const auto& entry : buffer_) {
    minSliceEnd_ = std::min(minSliceEnd_, entry.first.window());
  }
  return fired;
}

} // namespace facebook::velox::stateful
