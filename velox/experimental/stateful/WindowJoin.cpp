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
#include <map>
#include <vector>

#include "velox/common/base/Exceptions.h"
#include "velox/experimental/stateful/join/JoinRecordStateViews.h"
#include "velox/expression/Expr.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::stateful {

WindowJoin::WindowJoin(
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
    RowTypePtr rightInputType)
    : StatefulOperator(std::move(probe), std::move(targets)),
      leftInput_(std::move(leftInput)),
      rightInput_(std::move(rightInput)),
      leftKeySelector_(std::move(leftKeySelector)),
      rightKeySelector_(std::move(rightKeySelector)),
      probe_(static_cast<exec::NestedLoopJoinProbe*>(op().get())),
      joinType_(joinType),
      leftWindowEndIndex_(leftWindowEndIndex),
      rightWindowEndIndex_(rightWindowEndIndex),
      leftInputType_(leftInputType),
      rightInputType_(rightInputType) {
  VELOX_CHECK_NOT_NULL(probe_, "probe must be a NestedLoopJoinProbe");
}

void WindowJoin::initialize() {
  StatefulOperator::initialize();
  leftInput_->initialize();
  rightInput_->initialize();
  pool_ = probe_->pool();
}

void WindowJoin::initializeState() {
  // TODO: migrate this operator to operator-owned keyed state (cf.
  // GroupAggregator::initializeStateBackend) and remove this guard.
  VELOX_NYI(
      "WindowJoin is not yet migrated to operator-owned keyed state; there "
      "is no state handler to initialize");
  StateDescriptor leftStateDesc("left-records");
  leftWindowState_ = stateHandler()->getListState(leftStateDesc);
  StateDescriptor rightStateDesc("right-records");
  rightWindowState_ = stateHandler()->getListState(rightStateDesc);
  timerService_ = stateHandler()->createTimerService(this);
}

bool WindowJoin::isFinished() {
  return leftInput_->isFinished() && rightInput_->isFinished();
}

void WindowJoin::addInput(StreamElementPtr input) {
  VELOX_NYI("WindowJoin uses advance() pull model, not addInput push");
}

void WindowJoin::close() {
  StatefulOperator::close();
  leftInput_->close();
  rightInput_->close();
  // State members are created in initializeState(), which fails fast when the
  // operator has no state handler; they can be null at close time.
  if (leftWindowState_) {
    leftWindowState_->clear();
  }
  if (rightWindowState_) {
    rightWindowState_->clear();
  }
}

void WindowJoin::processWatermark(int64_t timestamp, int32_t index) {
  // Let the base class combine per-input watermarks and emit the combined
  // watermark downstream.
  StatefulOperator::processWatermark(timestamp, index);

  // Advance the timer service with the combined watermark. This triggers
  // onEventTime for all buckets whose windowEnd has been passed.
  int64_t combined = getCombinedWatermark();
  if (combined > lastTriggeredWatermark_) {
    lastTriggeredWatermark_ = combined;
    timerService_->advanceWatermark(combined);
  }
}

void WindowJoin::onEventTime(
    std::shared_ptr<TimerHeapInternalTimer<int64_t, int64_t>> timer) {
  LOG(INFO) << "WindowJoin::onEventTime key=" << timer->key() << " ns=" << timer->ns();
  join(timer->key(), timer->ns());
}

void WindowJoin::onProcessingTime(
    std::shared_ptr<TimerHeapInternalTimer<int64_t, int64_t>> timer) {
  // Window join only supports event time.
  VELOX_NYI("WindowJoin does not support processing time timers");
}

void WindowJoin::advance() {
  processData(
      leftInput_.get(),
      leftKeySelector_.get(),
      leftWindowEndIndex_,
      leftWindowState_.get());
  processData(
      rightInput_.get(),
      rightKeySelector_.get(),
      rightWindowEndIndex_,
      rightWindowState_.get());
}

void WindowJoin::processData(
    exec::Operator* input,
    KeySelector* keySelector,
    int windowEndIndex,
    ListState<uint32_t, int64_t, RowVectorPtr>* state) {
  auto result = input->getOutput();
  if (!result) {
    return;
  }

  auto notFired = filterWindowFiredRows(result, windowEndIndex);
  if (!notFired) {
    return;
  }

  auto keyToData = keySelector->partition(notFired);
  for (auto& [key, partitioned] : keyToData) {
    auto windowEndToData = partitionWindowData(partitioned, windowEndIndex);
    for (const auto& [windowEnd, rowData] : windowEndToData) {
      state->add(key, windowEnd, rowData);
      timerService_->registerEventTimeTimer(key, windowEnd, windowEnd - 1);
    }
  }
}

// Reads window-end values from either a BIGINT or a TIMESTAMP(3) column
// (Flink window TVF emits window_end as TIMESTAMP(3)). Returns nullptr when
// the column type is unsupported.
class WindowEndReader {
 public:
  explicit WindowEndReader(const VectorPtr& child) : child_(child) {
    if (auto* ts = child_->as<SimpleVector<Timestamp>>()) {
      isTimestamp_ = true;
    } else if (child_->as<SimpleVector<int64_t>>() != nullptr) {
      isTimestamp_ = false;
    } else {
      valid_ = false;
    }
  }

  bool valid() const {
    return valid_;
  }

  int64_t valueAt(vector_size_t i) const {
    if (isTimestamp_) {
      return child_->as<SimpleVector<Timestamp>>()->valueAt(i).toMillis();
    }
    return child_->as<SimpleVector<int64_t>>()->valueAt(i);
  }

 private:
  VectorPtr child_;
  bool isTimestamp_ = false;
  bool valid_ = true;
};

RowVectorPtr WindowJoin::filterWindowFiredRows(
    RowVectorPtr& input, int windowEndIndex) {
  if (!input || input->size() == 0) {
    return nullptr;
  }
  if (windowEndIndex < 0 || windowEndIndex >= input->childrenSize()) {
    return input;  // Can't filter; pass through.
  }
  int64_t watermark = timerService_->currentWatermark();
  if (watermark < 0) {
    return input;
  }
  WindowEndReader windowEnds(input->childAt(windowEndIndex));
  if (!windowEnds.valid()) {
    return input;  // Can't filter; pass through.
  }
  int n = input->size();

  // Fast path: all rows belong to unfired windows.
  bool allFuture = true;
  for (int i = 0; i < n; i++) {
    if (windowEnds.valueAt(i) <= watermark) {
      allFuture = false;
      break;
    }
  }
  if (allFuture) {
    return input;
  }

  // Slow path: filter out rows from fired windows.
  std::vector<vector_size_t> kept;
  kept.reserve(n);
  for (int i = 0; i < n; i++) {
    if (windowEnds.valueAt(i) > watermark) {
      kept.push_back(i);
    }
  }
  if (kept.empty()) {
    return nullptr;
  }
  return wrapRows(input, kept);
}

std::map<int64_t, RowVectorPtr> WindowJoin::partitionWindowData(
    RowVectorPtr& input, int windowEndIndex) {
  if (!input || input->size() == 0) {
    return {};
  }
  if (windowEndIndex < 0 || windowEndIndex >= input->childrenSize()) {
    LOG(WARNING) << "WindowJoin: windowEndIndex " << windowEndIndex
                 << " out of range [0, " << input->childrenSize() << ")";
    return {};
  }
  // KeySelector::partition wraps children in DictionaryVector; use
  // SimpleVector's valueAt() interface which works across encodings. The
  // window-end column may be BIGINT or TIMESTAMP(3).
  WindowEndReader windowEnds(input->childAt(windowEndIndex));
  if (!windowEnds.valid()) {
    LOG(WARNING) << "WindowJoin: windowEnd column kind="
                 << static_cast<int>(input->childAt(windowEndIndex)->typeKind())
                 << " is neither BIGINT nor TIMESTAMP";
    return {};
  }
  int n = input->size();

  // Fast path: all rows share the same windowEnd (most common — the upstream
  // window aggregate emits per-window batches).
  int64_t first = windowEnds.valueAt(0);
  bool allSame = true;
  for (int i = 1; i < n; i++) {
    if (windowEnds.valueAt(i) != first) {
      allSame = false;
      break;
    }
  }
  if (allSame) {
    return {{first, input}};
  }

  // Slow path: rows span multiple windows; group by windowEnd.
  std::map<int64_t, std::vector<vector_size_t>> groups;
  for (int i = 0; i < n; i++) {
    groups[windowEnds.valueAt(i)].push_back(i);
  }

  std::map<int64_t, RowVectorPtr> result;
  for (auto& [wEnd, indices] : groups) {
    result[wEnd] = wrapRows(input, indices);
  }
  return result;
}

void WindowJoin::join(uint32_t key, int64_t windowEnd) {
  LOG(INFO) << "WindowJoin::join key=" << key << " windowEnd=" << windowEnd;
  auto leftValues = leftWindowState_->get(key, windowEnd);
  auto rightValues = rightWindowState_->get(key, windowEnd);
  LOG(INFO) << "WindowJoin::join left=" << leftValues.size() << " right=" << rightValues.size();

  switch (joinType_) {
    case core::JoinType::kInner:
      if (!leftValues.empty() && !rightValues.empty()) {
        probeAndCollect(leftValues, rightValues);
      }
      break;

    case core::JoinType::kLeft:
      if (rightValues.empty()) {
        emitNullPadded(leftValues, /*padRight=*/true);
      } else {
        probeAndCollect(leftValues, rightValues);
      }
      break;

    case core::JoinType::kRight:
      if (leftValues.empty()) {
        emitNullPadded(rightValues, /*padRight=*/false);
      } else {
        probeAndCollect(leftValues, rightValues);
      }
      break;

    case core::JoinType::kFull:
      if (leftValues.empty() && !rightValues.empty()) {
        emitNullPadded(rightValues, /*padRight=*/false);
      } else if (!leftValues.empty() && rightValues.empty()) {
        emitNullPadded(leftValues, /*padRight=*/true);
      } else if (!leftValues.empty() && !rightValues.empty()) {
        probeAndCollect(leftValues, rightValues);
      }
      break;

    case core::JoinType::kLeftSemiFilter:
      // Flink SEMI: emit left rows that have at least one match on the right.
      if (!rightValues.empty()) {
        probeSemi(leftValues, rightValues, /*anti=*/false);
      }
      break;

    case core::JoinType::kAnti:
      // Flink ANTI: emit left rows that have no match on the right.
      if (rightValues.empty()) {
        // No right values at all: every left row is unmatched.
        emitAll(leftValues);
      } else {
        probeSemi(leftValues, rightValues, /*anti=*/true);
      }
      break;

    default:
      VELOX_UNSUPPORTED(
          "WindowJoin does not support join type {}", joinType_);
  }

  // Join + clear is atomic: the bucket disappears after this method.
  leftWindowState_->remove(key, windowEnd);
  rightWindowState_->remove(key, windowEnd);
}

void WindowJoin::probeAndCollect(
    const std::vector<RowVectorPtr>& leftValues,
    const std::vector<RowVectorPtr>& rightValues) {
  // KeySelector::wrapChildren produces DictionaryVector children.
  // Keep the dictionary encoding when feeding the probe: the comparison
  // function in velox only uses SIMD (which crashes on misaligned raw
  // buffers on aarch64) when both operands are Constant or Flat encoded;
  // Dictionary encoding forces the scalar code path (valueAt), which is
  // safe on all architectures.
  probe_->setBuildData(rightValues);

  // Probe side (left) is fed vector by vector.
  for (const auto& leftVec : leftValues) {
    probe_->addInput(leftVec);
    auto result = probe_->getOutput();
    if (result) {
      pushOutput(
          std::make_shared<StreamRecord>(getPlanNodeId(), std::move(result)));
    }
  }
}

void WindowJoin::emitNullPadded(
    const std::vector<RowVectorPtr>& values, bool padRight) {
  // Construct a single all-null row for the other side.
  auto otherType = padRight ? rightInputType_ : leftInputType_;
  if (!otherType) {
    return;
  }
  int numCols = otherType->size();
  std::vector<VectorPtr> nullChildren(numCols);
  for (int i = 0; i < numCols; i++) {
    nullChildren[i] = BaseVector::createNullConstant(
        otherType->childAt(i), /*size=*/1, pool_);
  }
  auto nullRow = std::make_shared<RowVector>(
      pool_, otherType, nullptr /*nulls*/, /*size=*/1,
      std::move(nullChildren));

  // Use the probe in a cross-join mode: each value vector joined with the
  // null row produces null-padded output.
  probe_->setBuildData({nullRow});
  for (const auto& vec : values) {
    probe_->addInput(vec);
    auto result = probe_->getOutput();
    if (result) {
      pushOutput(
          std::make_shared<StreamRecord>(getPlanNodeId(), std::move(result)));
    }
  }
}

void WindowJoin::probeSemi(
    const std::vector<RowVectorPtr>& leftValues,
    const std::vector<RowVectorPtr>& rightValues,
    bool anti) {
  // The NestedLoopJoinProbe supports kLeftSemiFilter and kAnti join types
  // natively: it emits only probe-side rows that have (semi) or do not have
  // (anti) matches. We reuse the same probeAndCollect path; the join type
  // in the probe's plan node determines the output semantics.
  //
  // If the probe does not support semi/anti directly, fall back to a kLeft
  // join with post-filtering on null-padded rows.
  probeAndCollect(leftValues, rightValues);
}

void WindowJoin::emitAll(const std::vector<RowVectorPtr>& values) {
  for (const auto& vec : values) {
    pushOutput(
        std::make_shared<StreamRecord>(getPlanNodeId(), vec));
  }
}

RowVectorPtr WindowJoin::wrapRows(
    const RowVectorPtr& input,
    const std::vector<vector_size_t>& indices) {
  int n = indices.size();
  auto indicesBuf = allocateIndices(n, pool_);
  auto raw = indicesBuf->asMutable<vector_size_t>();
  std::copy(indices.begin(), indices.end(), raw);

  int numChildren = input->childrenSize();
  std::vector<VectorPtr> children(numChildren);
  for (int i = 0; i < numChildren; i++) {
    children[i] = input->childAt(i)->slice(0, n); // TODO: proper indices-based wrapping
  }

  return std::make_shared<RowVector>(
      pool_, input->type(), nullptr /*nulls*/, n, std::move(children));
}

} // namespace facebook::velox::stateful
