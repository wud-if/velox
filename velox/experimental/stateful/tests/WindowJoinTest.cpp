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

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include <optional>
#include <set>
#include <string>
#include <vector>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/Expressions.h"
#include "velox/core/PlanFragment.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Driver.h"
#include "velox/exec/NestedLoopJoinProbe.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/experimental/stateful/KeyedStateSpec.h"
#include "velox/experimental/stateful/RowKind.h"
#include "velox/experimental/stateful/StatefulOperator.h"
#include "velox/experimental/stateful/StatefulPlanNode.h"
#include "velox/experimental/stateful/StreamElement.h"
#include "velox/experimental/stateful/state/HashMapStateBackend.h"
#include "velox/experimental/stateful/state/StateBackend.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::stateful::test {
namespace {

// The default two-input shape:
//   left  ROW<l_k BIGINT, l_ws BIGINT, l_we BIGINT, l_v VARCHAR>
//   right ROW<r_k BIGINT, r_ws BIGINT, r_we BIGINT, r_v VARCHAR>
// with the key on channel 0, window start 1, window end 2, payload 3.
constexpr int64_t kWindowSize = 10000;

// A terminal scan stub with its own plan node id: the join routes pushed
// records by the record's node id, so each side's scan needs a distinct one.
class ScanStub : public exec::Operator {
 public:
  ScanStub(
      exec::DriverCtx* driverCtx,
      const RowTypePtr& inputType,
      std::string planNodeId)
      : Operator(driverCtx, inputType, 0, std::move(planNodeId), "ScanStub") {}

  bool needsInput() const override {
    return false;
  }

  void addInput(RowVectorPtr /*input*/) override {}

  RowVectorPtr getOutput() override {
    return nullptr;
  }

  exec::BlockingReason isBlocked(ContinueFuture*) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return true;
  }
};

// Downstream collector: keeps every record the join pushes plus a marker for
// every watermark it receives, so tests can assert that fired records reach
// downstream ahead of the watermark.
class CaptureTarget : public StatefulOperator {
 public:
  CaptureTarget(
      exec::DriverCtx* driverCtx,
      const RowTypePtr& rowType,
      std::string spyNodeId)
      : StatefulOperator(
            std::make_unique<ScanStub>(
                driverCtx,
                rowType,
                std::move(spyNodeId)),
            {}) {}

  void addInput(StreamElementPtr input) override {
    records_.push_back(std::move(input));
    events_.push_back("record");
  }

  void advance() override {}

  void processWatermark(int64_t timestamp) override {
    events_.push_back("watermark:" + std::to_string(timestamp));
  }

  const std::vector<StreamElementPtr>& records() const {
    return records_;
  }

  const std::vector<std::string>& events() const {
    return events_;
  }

  void clear() {
    records_.clear();
    events_.clear();
  }

 private:
  std::vector<StreamElementPtr> records_;
  std::vector<std::string> events_;
};

class WindowJoinTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    // The join conditions compile through the expression engine with the
    // sparksql names the gluten translation emits (equalto,
    // greaterorequal).
    functions::sparksql::registerFunctions();

    core::PlanFragment planFragment;
    planFragment.planNode = std::make_shared<core::ValuesNode>(
        core::PlanNodeId{"values"},
        std::vector<RowVectorPtr>{makeRowVector({"x"}, {bigints({1})})});
    executor_ = std::make_shared<folly::CPUThreadPoolExecutor>(1);
    task_ = exec::Task::create(
        "WindowJoinTest_task",
        std::move(planFragment),
        0,
        core::QueryCtx::create(executor_.get()),
        exec::Task::ExecutionMode::kParallel);
    driver_ = exec::Driver::testingCreate();
    driverCtx_ = std::make_unique<exec::DriverCtx>(task_, 0, 0, 0, 0);
    driverCtx_->driver = driver_.get();

    taskBackend_ = std::make_unique<HashMapStateBackend>(
        std::make_shared<const KeyedStateBackendParameters>(
            StateBackendType::HEAP, "job", "window-join"));
    // The scan ids are assigned per operator in makeJoin: looped tests put
    // several joins on one driver and pool names key on the plan node id.
  }

  void TearDown() override {
    driverCtx_.reset();
    driver_.reset();
    task_.reset();
    executor_.reset();
    OperatorTestBase::TearDown();
  }

  RowTypePtr leftType() const {
    return ROW(
        {"l_k", "l_ws", "l_we", "l_v"},
        {BIGINT(), BIGINT(), BIGINT(), VARCHAR()});
  }

  RowTypePtr rightType() const {
    return ROW(
        {"r_k", "r_ws", "r_we", "r_v"},
        {BIGINT(), BIGINT(), BIGINT(), VARCHAR()});
  }

  // The joined output for inner/left/right/full: all four payload columns.
  RowTypePtr joinedType() const {
    return ROW(
        {"l_k", "l_v", "r_k", "r_v"},
        {BIGINT(), VARCHAR(), BIGINT(), VARCHAR()});
  }

  // The projected output for semi/anti: left columns only.
  RowTypePtr leftProjectionType() const {
    return ROW({"l_k", "l_v"}, {BIGINT(), VARCHAR()});
  }

  core::TypedExprPtr equiCondition() const {
    return std::make_shared<core::CallTypedExpr>(
        BOOLEAN(),
        std::vector<core::TypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "l_k"),
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "r_k")},
        "equalto");
  }

  std::shared_ptr<const StreamWindowJoinNode> makeNode(
      core::JoinType joinType,
      const RowTypePtr& outputType,
      const std::vector<column_index_t>& leftKeyChannels,
      const std::vector<column_index_t>& rightKeyChannels,
      const std::vector<TypePtr>& keyTypes,
      core::TypedExprPtr joinCondition,
      const RowTypePtr& leftInputType = {},
      const RowTypePtr& rightInputType = {},
      int leftWindowEndIndex = 2,
      int rightWindowEndIndex = 2) {
    auto leftInput = leftInputType ? leftInputType : leftType();
    auto rightInput = rightInputType ? rightInputType : rightType();
    auto probeNode = std::make_shared<core::NestedLoopJoinNode>(
        core::PlanNodeId{nextNodeId()},
        joinType,
        joinCondition,
        std::make_shared<EmptyNode>(leftInput),
        std::make_shared<EmptyNode>(rightInput),
        outputType);
    auto keyedStateSpec = std::make_shared<const KeyedStateSpec>(
        leftKeyChannels, rightKeyChannels, keyTypes);
    return std::make_shared<const StreamWindowJoinNode>(
        core::PlanNodeId{nextNodeId()},
        std::vector<core::PlanNodePtr>{
            std::make_shared<EmptyNode>(leftInput),
            std::make_shared<EmptyNode>(rightInput)},
        std::make_shared<core::GatherPartitionFunctionSpec>(),
        std::make_shared<core::GatherPartitionFunctionSpec>(),
        probeNode,
        keyedStateSpec,
        outputType,
        /*numPartitions=*/1,
        leftWindowEndIndex,
        rightWindowEndIndex);
  }

  std::unique_ptr<WindowJoin> makeJoin(
      const std::shared_ptr<const StreamWindowJoinNode>& node,
      std::unique_ptr<StatefulOperator> capture) {
    return makeJoin(node, std::move(capture), taskBackend_.get());
  }

  // The rescale face: a join keyed onto a caller-chosen task backend, e.g.
  // one whose parameters cover a key-group sub-range (a scaled-down
  // operator's slice of the key-group space).
  std::unique_ptr<WindowJoin> makeJoin(
      const std::shared_ptr<const StreamWindowJoinNode>& node,
      std::unique_ptr<StatefulOperator> capture,
      StateBackend* backend) {
    std::vector<StatefulOperatorPtr> targets;
    targets.push_back(std::move(capture));
    // Fresh scan ids per operator: the pool names key on the plan node id,
    // so two joins on one driver need distinct scans (looped tests).
    leftScanId_ = "left-scan-" + nextNodeId();
    rightScanId_ = "right-scan-" + nextNodeId();
    auto leftOp =
        std::make_unique<ScanStub>(driverCtx_.get(), leftType(), leftScanId_);
    auto rightOp =
        std::make_unique<ScanStub>(driverCtx_.get(), rightType(), rightScanId_);
    auto probeOp = std::make_unique<exec::NestedLoopJoinProbe>(
        2, driverCtx_.get(), node->probe());
    auto join = std::make_unique<WindowJoin>(
        std::move(leftOp),
        std::move(rightOp),
        std::move(probeOp),
        std::move(targets),
        node);
    join->initialize();
    join->initializeStateBackend(backend);
    return join;
  }

  // An append-only left record (no rowKind column). The window start is
  // windowEnd - kWindowSize, the shape of a tumbling window.
  StreamElementPtr leftAppendOnly(
      const std::vector<std::optional<int64_t>>& k,
      const std::vector<int64_t>& windowEnd,
      const std::vector<std::string>& v) {
    std::vector<int64_t> ws;
    ws.reserve(windowEnd.size());
    for (auto end : windowEnd) {
      ws.push_back(end - kWindowSize);
    }
    return std::make_shared<StreamRecord>(
        leftScanId_,
        makeRowVector(
            {"l_k", "l_ws", "l_we", "l_v"},
            {makeNullableFlatVector<int64_t>(k),
             bigints(ws),
             bigints(windowEnd),
             makeFlatVector<std::string>(v)}));
  }

  StreamElementPtr rightAppendOnly(
      const std::vector<std::optional<int64_t>>& k,
      const std::vector<int64_t>& windowEnd,
      const std::vector<std::string>& v) {
    std::vector<int64_t> ws;
    ws.reserve(windowEnd.size());
    for (auto end : windowEnd) {
      ws.push_back(end - kWindowSize);
    }
    return std::make_shared<StreamRecord>(
        rightScanId_,
        makeRowVector(
            {"r_k", "r_ws", "r_we", "r_v"},
            {makeNullableFlatVector<int64_t>(k),
             bigints(ws),
             bigints(windowEnd),
             makeFlatVector<std::string>(v)}));
  }

  // A left record carrying an explicit per-row rowKind column.
  StreamElementPtr leftWithKinds(
      const std::vector<int64_t>& k,
      const std::vector<int64_t>& windowEnd,
      const std::vector<std::string>& v,
      const std::vector<RowKind>& kinds) {
    std::vector<int64_t> ws;
    ws.reserve(windowEnd.size());
    for (auto end : windowEnd) {
      ws.push_back(end - kWindowSize);
    }
    std::vector<int8_t> rawKinds;
    rawKinds.reserve(kinds.size());
    for (auto kind : kinds) {
      rawKinds.push_back(static_cast<int8_t>(kind));
    }
    return std::make_shared<StreamRecord>(
        leftScanId_,
        makeRowVector(
            {"l_k", "l_ws", "l_we", "l_v"},
            {bigints(k),
             bigints(ws),
             bigints(windowEnd),
             makeFlatVector<std::string>(v)}),
        makeFlatVector<int8_t>(rawKinds));
  }

  // A right record carrying an explicit per-row rowKind column.
  StreamElementPtr rightWithKinds(
      const std::vector<int64_t>& k,
      const std::vector<int64_t>& windowEnd,
      const std::vector<std::string>& v,
      const std::vector<RowKind>& kinds) {
    std::vector<int64_t> ws;
    ws.reserve(windowEnd.size());
    for (auto end : windowEnd) {
      ws.push_back(end - kWindowSize);
    }
    std::vector<int8_t> rawKinds;
    rawKinds.reserve(kinds.size());
    for (auto kind : kinds) {
      rawKinds.push_back(static_cast<int8_t>(kind));
    }
    return std::make_shared<StreamRecord>(
        rightScanId_,
        makeRowVector(
            {"r_k", "r_ws", "r_we", "r_v"},
            {bigints(k),
             bigints(ws),
             bigints(windowEnd),
             makeFlatVector<std::string>(v)}),
        makeFlatVector<int8_t>(rawKinds));
  }

  // A session-window record: an explicit, arbitrary window span - the
  // shape the upstream session merger leaves behind. The join reads only
  // the window-end column, so the unaligned span must pair by window end
  // exactly as given, with no correction.
  StreamElementPtr leftSessionRows(
      const std::vector<int64_t>& k,
      const std::vector<int64_t>& windowStart,
      const std::vector<int64_t>& windowEnd,
      const std::vector<std::string>& v) {
    return std::make_shared<StreamRecord>(
        leftScanId_,
        makeRowVector(
            {"l_k", "l_ws", "l_we", "l_v"},
            {bigints(k),
             bigints(windowStart),
             bigints(windowEnd),
             makeFlatVector<std::string>(v)}));
  }

  StreamElementPtr rightSessionRows(
      const std::vector<int64_t>& k,
      const std::vector<int64_t>& windowStart,
      const std::vector<int64_t>& windowEnd,
      const std::vector<std::string>& v) {
    return std::make_shared<StreamRecord>(
        rightScanId_,
        makeRowVector(
            {"r_k", "r_ws", "r_we", "r_v"},
            {bigints(k),
             bigints(windowStart),
             bigints(windowEnd),
             makeFlatVector<std::string>(v)}));
  }

  StreamRecordPtr recordAt(const CaptureTarget& capture, size_t index) {
    VELOX_CHECK_LT(index, capture.records().size());
    auto record =
        std::static_pointer_cast<StreamRecord>(capture.records()[index]);
    EXPECT_TRUE(record->isRecord());
    EXPECT_TRUE(record->appendOnly()) << "window join emits INSERT only";
    return record;
  }

  // Flattens one BIGINT output channel across every captured record, nulls
  // preserved (outer-join padding). Rows iterate by the record's row count
  // - the probe pre-allocates its output columns at the batch size, so the
  // column vectors can claim more rows than the record holds. The
  // dictionary-encoded probe columns read the same way as flat ones through
  // SimpleVector.
  std::vector<std::optional<int64_t>> bigintsOf(
      const CaptureTarget& capture,
      int channel) {
    std::vector<std::optional<int64_t>> out;
    for (const auto& element : capture.records()) {
      auto record = std::static_pointer_cast<StreamRecord>(element);
      const auto numRows = record->record()->size();
      auto vector =
          record->record()->childAt(channel)->as<SimpleVector<int64_t>>();
      VELOX_CHECK_NOT_NULL(vector, "expected a BIGINT channel");
      for (vector_size_t row = 0; row < numRows; ++row) {
        out.push_back(
            vector->isNullAt(row)
                ? std::nullopt
                : std::optional<int64_t>(vector->valueAt(row)));
      }
    }
    return out;
  }

  // Flattens one VARCHAR output channel across every captured record.
  std::vector<std::optional<std::string>> varcharsOf(
      const CaptureTarget& capture,
      int channel) {
    std::vector<std::optional<std::string>> out;
    for (const auto& element : capture.records()) {
      auto record = std::static_pointer_cast<StreamRecord>(element);
      const auto numRows = record->record()->size();
      auto vector =
          record->record()->childAt(channel)->as<SimpleVector<StringView>>();
      VELOX_CHECK_NOT_NULL(vector, "expected a VARCHAR channel");
      for (vector_size_t row = 0; row < numRows; ++row) {
        out.push_back(
            vector->isNullAt(row)
                ? std::nullopt
                : std::optional<std::string>(vector->valueAt(row).str()));
      }
    }
    return out;
  }

  // Operator pool names are keyed by plan node id, so every node a test puts
  // on one driver needs a distinct id.
  std::string nextNodeId() {
    return std::to_string(nextNodeId_++);
  }

  // A single-element braced argument to makeFlatVector resolves to the
  // (vector_size_t size) overload and builds a garbage vector; routing
  // through these helpers always picks the value-list overload.
  VectorPtr bigints(const std::vector<int64_t>& values) {
    return makeFlatVector<int64_t>(values);
  }

  std::string leftScanId_;
  std::string rightScanId_;
  int64_t nextNodeId_ = 0;
  std::shared_ptr<folly::CPUThreadPoolExecutor> executor_;
  std::shared_ptr<exec::Task> task_;
  std::shared_ptr<exec::Driver> driver_;
  std::unique_ptr<exec::DriverCtx> driverCtx_;
  std::unique_ptr<HashMapStateBackend> taskBackend_;
};

// The core inner-join path: rows pair inside their window only, the window
// fires exactly once - when BOTH inputs' watermarks pass its trigger time -
// and every match of the key group appears in the output. One side alone
// never fires the window (two-input min).
TEST_F(WindowJoinTest, testInnerJoinFiresOnMinWatermark) {
  auto capture = std::make_unique<CaptureTarget>(
      driverCtx_.get(), joinedType(), nextNodeId());
  auto* capturePtr = capture.get();
  auto join = makeJoin(
      makeNode(
          core::JoinType::kInner,
          joinedType(),
          /*leftKeyChannels=*/{0},
          /*rightKeyChannels=*/{0},
          /*keyTypes=*/{BIGINT()},
          equiCondition()),
      std::move(capture));

  // One window [0, 10010): left key 1 twice, key 2 once; right key 1 twice.
  join->addInput(
      leftAppendOnly({1, 1, 2}, {10010, 10010, 10010}, {"L1", "L2", "L3"}));
  join->addInput(rightAppendOnly({1, 1}, {10010, 10010}, {"R1", "R2"}));

  // Left watermark alone: the min does not reach the trigger time.
  join->processWatermark(10009, 0);
  EXPECT_TRUE(capturePtr->records().empty());

  // Right watermark completes the min: the window fires.
  join->processWatermark(10009, 1);
  ASSERT_EQ(capturePtr->records().size(), 2);
  recordAt(*capturePtr, 0);
  recordAt(*capturePtr, 1);
  // Key 1: 2 left rows x 2 right rows; key 2: no right side, so the cross
  // product within the key group covers every pairing.
  EXPECT_EQ(
      bigintsOf(*capturePtr, 0),
      (std::vector<std::optional<int64_t>>{1, 1, 1, 1}));
  EXPECT_EQ(
      varcharsOf(*capturePtr, 1),
      (std::vector<std::optional<std::string>>{"L1", "L1", "L2", "L2"}));
  EXPECT_EQ(
      bigintsOf(*capturePtr, 2),
      (std::vector<std::optional<int64_t>>{1, 1, 1, 1}));
  EXPECT_EQ(
      varcharsOf(*capturePtr, 3),
      (std::vector<std::optional<std::string>>{"R1", "R2", "R1", "R2"}));

  // The fired records reached downstream ahead of the watermark marker.
  const auto& events = capturePtr->events();
  ASSERT_GE(events.size(), 3);
  EXPECT_EQ(events[0], "record");
  EXPECT_EQ(events[1], "record");
  EXPECT_EQ(events[2], "watermark:10009");

  // Refiring is impossible: a later watermark emits nothing more.
  join->processWatermark(20000, 0);
  join->processWatermark(20000, 1);
  EXPECT_EQ(capturePtr->records().size(), 2);
}

// Windows only pair within themselves: rows of different windows never
// meet, and windows fire in trigger-time order.
TEST_F(WindowJoinTest, testWindowIsolation) {
  auto capture = std::make_unique<CaptureTarget>(
      driverCtx_.get(), joinedType(), nextNodeId());
  auto* capturePtr = capture.get();
  auto join = makeJoin(
      makeNode(
          core::JoinType::kInner,
          joinedType(),
          {0},
          {0},
          {BIGINT()},
          equiCondition()),
      std::move(capture));

  // Two windows: [0, 10010) and [10010, 20010). The left row of window 2
  // must not pair with the right row of window 1 despite the shared key.
  join->addInput(leftAppendOnly({1, 1}, {10010, 20010}, {"L1", "L2"}));
  join->addInput(rightAppendOnly({1, 1}, {10010, 20010}, {"R1", "R2"}));
  join->processWatermark(10009, 0);
  join->processWatermark(10009, 1);
  join->processWatermark(20009, 0);
  join->processWatermark(20009, 1);

  ASSERT_EQ(capturePtr->records().size(), 2);
  EXPECT_EQ(
      varcharsOf(*capturePtr, 1),
      (std::vector<std::optional<std::string>>{"L1", "L2"}));
  EXPECT_EQ(
      varcharsOf(*capturePtr, 3),
      (std::vector<std::optional<std::string>>{"R1", "R2"}));
}

// Late rows: after a window fired, rows of that window are dropped and never
// pollute the emitted output; an untouched window still fires afterwards.
TEST_F(WindowJoinTest, testLateRowsDropped) {
  auto capture = std::make_unique<CaptureTarget>(
      driverCtx_.get(), joinedType(), nextNodeId());
  auto* capturePtr = capture.get();
  auto join = makeJoin(
      makeNode(
          core::JoinType::kInner,
          joinedType(),
          {0},
          {0},
          {BIGINT()},
          equiCondition()),
      std::move(capture));

  join->addInput(leftAppendOnly({1}, {10010}, {"L1"}));
  join->addInput(rightAppendOnly({1}, {10010}, {"R1"}));
  join->processWatermark(10009, 0);
  join->processWatermark(10009, 1);
  ASSERT_EQ(capturePtr->records().size(), 1);

  // Late left and right rows for the fired window: dropped silently.
  join->addInput(leftAppendOnly({1}, {10010}, {"LATE"}));
  join->addInput(rightAppendOnly({1}, {10010}, {"LATE"}));

  // A fresh window still fires and contains no late payload.
  join->addInput(leftAppendOnly({2}, {20010}, {"L2"}));
  join->addInput(rightAppendOnly({2}, {20010}, {"R2"}));
  join->processWatermark(20009, 0);
  join->processWatermark(20009, 1);
  ASSERT_EQ(capturePtr->records().size(), 2);
  EXPECT_EQ(
      varcharsOf(*capturePtr, 1),
      (std::vector<std::optional<std::string>>{"L1", "L2"}));
  EXPECT_EQ(
      varcharsOf(*capturePtr, 3),
      (std::vector<std::optional<std::string>>{"R1", "R2"}));

  // The window-end sentinel (INT64_MAX) never fires: the row is stored but
  // no watermark ever schedules it.
  join->addInput(
      leftAppendOnly({3}, {std::numeric_limits<int64_t>::max()}, {"L3"}));
  join->processWatermark(std::numeric_limits<int64_t>::max(), 0);
  join->processWatermark(std::numeric_limits<int64_t>::max(), 1);
  EXPECT_EQ(capturePtr->records().size(), 2);
}

// Retraction rows fail loud: window join accepts accumulate rows only
// (INSERT/UPDATE_AFTER, as in Flink RowDataUtil.isAccumulateMsg), so an
// UPDATE_BEFORE or DELETE batch must not be silently dropped.
TEST_F(WindowJoinTest, testRetractionFailsLoud) {
  auto capture = std::make_unique<CaptureTarget>(
      driverCtx_.get(), joinedType(), nextNodeId());
  auto join = makeJoin(
      makeNode(
          core::JoinType::kInner,
          joinedType(),
          {0},
          {0},
          {BIGINT()},
          equiCondition()),
      std::move(capture));

  VELOX_ASSERT_THROW(
      join->addInput(
          leftWithKinds({1}, {10010}, {"L1"}, {RowKind::UPDATE_BEFORE})),
      "retraction");
  VELOX_ASSERT_THROW(
      join->addInput(leftWithKinds({1}, {10010}, {"L1"}, {RowKind::DELETE})),
      "retraction");
}

// Design UT matrix example 3 (mixed kinds): accumulate rows (INSERT or
// UPDATE_AFTER, as in Flink RowDataUtil.isAccumulateMsg) are accepted into
// the window state and join on fire, and the rowKind check runs per row -
// accumulate rows in front of a retraction row land in state before the
// retraction row aborts the batch.
TEST_F(WindowJoinTest, testMixedKindsAccumulateAndRetract) {
  auto capture = std::make_unique<CaptureTarget>(
      driverCtx_.get(), joinedType(), nextNodeId());
  auto* capturePtr = capture.get();
  auto join = makeJoin(
      makeNode(
          core::JoinType::kInner,
          joinedType(),
          {0},
          {0},
          {BIGINT()},
          equiCondition()),
      std::move(capture));

  // A pure UPDATE_AFTER batch is accumulate input: both rows enter the
  // window state and join on fire. The fire emits one record per left row
  // (each probe batch is a single left row crossed with the build side).
  join->addInput(leftWithKinds(
      {1, 1},
      {10010, 10010},
      {"L1", "L2"},
      {RowKind::UPDATE_AFTER, RowKind::UPDATE_AFTER}));
  join->addInput(rightAppendOnly({1}, {10010}, {"R1"}));
  join->processWatermark(10009, 0);
  join->processWatermark(10009, 1);
  ASSERT_EQ(capturePtr->records().size(), 2);
  EXPECT_EQ(
      varcharsOf(*capturePtr, 1),
      (std::vector<std::optional<std::string>>{"L1", "L2"}));
  EXPECT_EQ(
      varcharsOf(*capturePtr, 3),
      (std::vector<std::optional<std::string>>{"R1", "R1"}));

  // The rowKind check runs per row inside a batch: the UPDATE_AFTER and
  // INSERT rows in front of the UPDATE_BEFORE row are accumulated before
  // the retraction row aborts the batch.
  VELOX_ASSERT_THROW(
      join->addInput(leftWithKinds(
          {2, 2, 2},
          {20010, 20010, 20010},
          {"L3", "L4", "L5"},
          {RowKind::UPDATE_AFTER, RowKind::INSERT, RowKind::UPDATE_BEFORE})),
      "retraction");
  join->addInput(rightAppendOnly({2}, {20010}, {"R2"}));
  join->processWatermark(20009, 0);
  join->processWatermark(20009, 1);
  // Window 2 fires with the accumulated rows only: the retraction row L5
  // never entered the state. Two records, one per accumulated left row.
  ASSERT_EQ(capturePtr->records().size(), 4);
  EXPECT_EQ(
      varcharsOf(*capturePtr, 1),
      (std::vector<std::optional<std::string>>{"L1", "L2", "L3", "L4"}));
  EXPECT_EQ(
      varcharsOf(*capturePtr, 3),
      (std::vector<std::optional<std::string>>{"R1", "R1", "R2", "R2"}));
}

// The six-type matrix over one window: key 1 matches, key 2 has no right
// match, key 3 has no left match.
TEST_F(WindowJoinTest, testJoinTypeMatrix) {
  struct Shape {
    core::JoinType joinType;
    RowTypePtr outputType;
    std::vector<std::optional<int64_t>> l_k;
    std::vector<std::optional<std::string>> l_v;
    std::vector<std::optional<int64_t>> r_k;
    std::vector<std::optional<std::string>> r_v;
  };
  const std::vector<Shape> shapes = {
      {core::JoinType::kInner, joinedType(), {1}, {"L1"}, {1}, {"R1"}},
      {core::JoinType::kLeft,
       joinedType(),
       {1, 2},
       {"L1", "L2"},
       {1, std::nullopt},
       {"R1", std::nullopt}},
      {core::JoinType::kRight,
       joinedType(),
       {1, std::nullopt},
       {"L1", std::nullopt},
       {1, 3},
       {"R1", "R3"}},
      {core::JoinType::kFull,
       joinedType(),
       {1, 2, std::nullopt},
       {"L1", "L2", std::nullopt},
       {1, std::nullopt, 3},
       {"R1", std::nullopt, "R3"}},
      {core::JoinType::kLeftSemiFilter,
       leftProjectionType(),
       {1},
       {"L1"},
       {},
       {}},
      {core::JoinType::kAnti, leftProjectionType(), {2}, {"L2"}, {}, {}}};

  for (const auto& shape : shapes) {
    SCOPED_TRACE("joinType " + std::to_string(int(shape.joinType)));
    auto capture = std::make_unique<CaptureTarget>(
        driverCtx_.get(), shape.outputType, nextNodeId());
    auto* capturePtr = capture.get();
    auto join = makeJoin(
        makeNode(
            shape.joinType,
            shape.outputType,
            {0},
            {0},
            {BIGINT()},
            equiCondition()),
        std::move(capture));

    join->addInput(leftAppendOnly({1, 2}, {10010, 10010}, {"L1", "L2"}));
    join->addInput(rightAppendOnly({1, 3}, {10010, 10010}, {"R1", "R3"}));
    join->processWatermark(10009, 0);
    join->processWatermark(10009, 1);

    EXPECT_EQ(bigintsOf(*capturePtr, 0), shape.l_k);
    EXPECT_EQ(varcharsOf(*capturePtr, 1), shape.l_v);
    if (shape.outputType->size() > 2) {
      EXPECT_EQ(bigintsOf(*capturePtr, 2), shape.r_k);
      EXPECT_EQ(varcharsOf(*capturePtr, 3), shape.r_v);
    }
  }
}

// Empty sides per window type: a right-empty window pads on left/full and
// stays silent on inner/semi/anti; a left-empty window pads on right/full.
TEST_F(WindowJoinTest, testEmptySideWindows) {
  struct Shape {
    core::JoinType joinType;
    bool leftEmpty;
    std::vector<std::optional<int64_t>> l_k;
    std::vector<std::optional<std::string>> l_v;
    std::vector<std::optional<int64_t>> r_k;
    std::vector<std::optional<std::string>> r_v;
  };
  const std::vector<Shape> shapes = {
      // Window A: right empty, left has key 1.
      {core::JoinType::kInner, false, {}, {}, {}, {}},
      {core::JoinType::kLeft,
       false,
       {1},
       {"L1"},
       {std::nullopt},
       {std::nullopt}},
      {core::JoinType::kFull,
       false,
       {1},
       {"L1"},
       {std::nullopt},
       {std::nullopt}},
      {core::JoinType::kLeftSemiFilter, false, {}, {}, {}, {}},
      {core::JoinType::kAnti, false, {1}, {"L1"}, {}, {}},
      // Window B: left empty, right has key 3.
      {core::JoinType::kInner, true, {}, {}, {}, {}},
      {core::JoinType::kRight,
       true,
       {std::nullopt},
       {std::nullopt},
       {3},
       {"R3"}},
      {core::JoinType::kFull,
       true,
       {std::nullopt},
       {std::nullopt},
       {3},
       {"R3"}},
      {core::JoinType::kLeftSemiFilter, true, {}, {}, {}, {}},
      {core::JoinType::kAnti, true, {}, {}, {}, {}}};

  for (const auto& shape : shapes) {
    SCOPED_TRACE(
        std::string("joinType ") + std::to_string(int(shape.joinType)) +
        (shape.leftEmpty ? " leftEmpty" : " rightEmpty"));
    auto outputType = shape.joinType == core::JoinType::kLeftSemiFilter ||
            shape.joinType == core::JoinType::kAnti
        ? leftProjectionType()
        : joinedType();
    auto capture = std::make_unique<CaptureTarget>(
        driverCtx_.get(), outputType, nextNodeId());
    auto* capturePtr = capture.get();
    auto join = makeJoin(
        makeNode(
            shape.joinType, outputType, {0}, {0}, {BIGINT()}, equiCondition()),
        std::move(capture));

    if (shape.leftEmpty) {
      join->addInput(rightAppendOnly({3}, {10010}, {"R3"}));
    } else {
      join->addInput(leftAppendOnly({1}, {10010}, {"L1"}));
    }
    join->processWatermark(10009, 0);
    join->processWatermark(10009, 1);

    EXPECT_EQ(bigintsOf(*capturePtr, 0), shape.l_k);
    EXPECT_EQ(varcharsOf(*capturePtr, 1), shape.l_v);
    if (outputType->size() > 2) {
      EXPECT_EQ(bigintsOf(*capturePtr, 2), shape.r_k);
      EXPECT_EQ(varcharsOf(*capturePtr, 3), shape.r_v);
    }
  }
}

// Null join keys under ordinary equality never pair (three-valued logic),
// so the null-key left row is dropped by inner, padded by left, and emitted
// by anti - Flink's JoinConditionWithNullFilters semantics.
TEST_F(WindowJoinTest, testNullKeyOrdinaryEquality) {
  struct Shape {
    core::JoinType joinType;
    RowTypePtr outputType;
    std::vector<std::optional<int64_t>> l_k;
    std::vector<std::optional<std::string>> l_v;
    std::vector<std::optional<int64_t>> r_k;
    std::vector<std::optional<std::string>> r_v;
  };
  const std::vector<Shape> shapes = {
      {core::JoinType::kInner, joinedType(), {}, {}, {}, {}},
      {core::JoinType::kLeft,
       joinedType(),
       {std::nullopt},
       {"LN"},
       {std::nullopt},
       {std::nullopt}},
      {core::JoinType::kAnti,
       leftProjectionType(),
       {std::nullopt},
       {"LN"},
       {},
       {}}};

  for (const auto& shape : shapes) {
    SCOPED_TRACE("joinType " + std::to_string(int(shape.joinType)));
    auto outputType = shape.joinType == core::JoinType::kAnti
        ? leftProjectionType()
        : joinedType();
    auto capture = std::make_unique<CaptureTarget>(
        driverCtx_.get(), outputType, nextNodeId());
    auto* capturePtr = capture.get();
    auto join = makeJoin(
        makeNode(
            shape.joinType, outputType, {0}, {0}, {BIGINT()}, equiCondition()),
        std::move(capture));

    // A null-key left row and a null-key right row in one window: the null
    // group pairs the rows, but ordinary equality never matches them.
    join->addInput(leftAppendOnly({std::optional<int64_t>()}, {10010}, {"LN"}));
    join->addInput(
        rightAppendOnly({std::optional<int64_t>()}, {10010}, {"RN"}));
    join->processWatermark(10009, 0);
    join->processWatermark(10009, 1);

    EXPECT_EQ(bigintsOf(*capturePtr, 0), shape.l_k);
    EXPECT_EQ(varcharsOf(*capturePtr, 1), shape.l_v);
    if (outputType->size() > 2) {
      EXPECT_EQ(bigintsOf(*capturePtr, 2), shape.r_k);
      EXPECT_EQ(varcharsOf(*capturePtr, 3), shape.r_v);
    }
  }
}

// Semi and anti window joins over the full behavior set: matched probe rows
// (semi emits once, anti emits nothing), unmatched probe rows (semi emits
// nothing, anti emits the row), empty sides, and the no-condition shapes
// where every probe row matches (semi emits its probe row once, anti emits
// nothing).
TEST_F(WindowJoinTest, testSemiAntiNoConditionShapes) {
  struct Shape {
    core::JoinType joinType;
    bool rightEmpty;
    std::vector<std::optional<int64_t>> l_k;
    std::vector<std::optional<std::string>> l_v;
  };
  // No join condition: every probe row with a non-empty build matches.
  const std::vector<Shape> shapes = {
      // Semi: key 1 has a right row -> emits L1; key 2 has none -> nothing.
      {core::JoinType::kLeftSemiFilter, false, {1}, {"L1"}},
      // Anti: key 1 matches -> nothing; key 2 has an empty build (no match
      // possible) -> emits L2.
      {core::JoinType::kAnti, false, {2}, {"L2"}},
      // Semi with an empty build everywhere: no matches, nothing emits.
      {core::JoinType::kLeftSemiFilter, true, {}, {}},
      // Anti with an empty build everywhere: no matches possible, both emit.
      {core::JoinType::kAnti, true, {1, 2}, {"L1", "L2"}}};

  for (const auto& shape : shapes) {
    SCOPED_TRACE(
        std::string("joinType ") + std::to_string(int(shape.joinType)) +
        (shape.rightEmpty ? " rightEmpty" : " rightPresent"));
    auto capture = std::make_unique<CaptureTarget>(
        driverCtx_.get(), leftProjectionType(), nextNodeId());
    auto* capturePtr = capture.get();
    auto join = makeJoin(
        makeNode(
            shape.joinType,
            leftProjectionType(),
            {0},
            {0},
            {BIGINT()},
            /*joinCondition=*/nullptr),
        std::move(capture));

    join->addInput(leftAppendOnly({1, 2}, {10010, 10010}, {"L1", "L2"}));
    if (!shape.rightEmpty) {
      join->addInput(rightAppendOnly({1}, {10010}, {"R1"}));
    }
    join->processWatermark(10009, 0);
    join->processWatermark(10009, 1);

    EXPECT_EQ(bigintsOf(*capturePtr, 0), shape.l_k);
    EXPECT_EQ(varcharsOf(*capturePtr, 1), shape.l_v);
  }
}

// The pure window join (q5 shape): no equi key remains, so the window start
// compensates as the key channel and the residual non-equi condition drives
// the pairing. Rows of one window pair across the full key group.
TEST_F(WindowJoinTest, testEmptyKeyWindowCompensation) {
  auto leftInputType =
      ROW({"l_ws", "l_we", "l_num"}, {BIGINT(), BIGINT(), BIGINT()});
  auto rightInputType =
      ROW({"r_ws", "r_we", "r_num"}, {BIGINT(), BIGINT(), BIGINT()});
  auto outputType = ROW({"l_num", "r_num"}, {BIGINT(), BIGINT()});
  auto condition = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "l_num"),
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "r_num")},
      "greaterthanorequal");

  auto capture = std::make_unique<CaptureTarget>(
      driverCtx_.get(), outputType, nextNodeId());
  auto* capturePtr = capture.get();
  auto join = makeJoin(
      makeNode(
          core::JoinType::kInner,
          outputType,
          /*leftKeyChannels=*/{0},
          /*rightKeyChannels=*/{0},
          {BIGINT()},
          condition,
          leftInputType,
          rightInputType,
          /*leftWindowEndIndex=*/1,
          /*rightWindowEndIndex=*/1),
      std::move(capture));

  // One window [0, 10010): l_num 5 pairs with r_num 3 and 7 under >=; rows
  // of another window never meet them.
  join->addInput(std::make_shared<StreamRecord>(
      leftScanId_,
      makeRowVector(
          {"l_ws", "l_we", "l_num"},
          {bigints({0}), bigints({10010}), bigints({5})})));
  join->addInput(std::make_shared<StreamRecord>(
      rightScanId_,
      makeRowVector(
          {"r_ws", "r_we", "r_num"},
          {bigints({0, 0}), bigints({10010, 10010}), bigints({3, 7})})));
  join->addInput(std::make_shared<StreamRecord>(
      leftScanId_,
      makeRowVector(
          {"l_ws", "l_we", "l_num"},
          {bigints({10010}), bigints({20010}), bigints({5})})));
  join->processWatermark(10009, 0);
  join->processWatermark(10009, 1);

  EXPECT_EQ(
      bigintsOf(*capturePtr, 0), (std::vector<std::optional<int64_t>>{5}));
  EXPECT_EQ(
      bigintsOf(*capturePtr, 1), (std::vector<std::optional<int64_t>>{3}));
}

// An idle input stops holding back the combined watermark: the window fires
// once the remaining side's watermark passes the trigger time.
TEST_F(WindowJoinTest, testIdleSideAdvancesCombinedWatermark) {
  auto capture = std::make_unique<CaptureTarget>(
      driverCtx_.get(), joinedType(), nextNodeId());
  auto* capturePtr = capture.get();
  auto join = makeJoin(
      makeNode(
          core::JoinType::kInner,
          joinedType(),
          {0},
          {0},
          {BIGINT()},
          equiCondition()),
      std::move(capture));

  join->addInput(leftAppendOnly({1}, {10010}, {"L1"}));
  join->addInput(rightAppendOnly({1}, {10010}, {"R1"}));
  // Left turns idle without ever sending a watermark; the right watermark
  // alone then completes the combined progress.
  join->processWatermarkStatus(/*idle=*/true, 0);
  join->processWatermark(10009, 1);
  ASSERT_EQ(capturePtr->records().size(), 1);
  EXPECT_EQ(
      varcharsOf(*capturePtr, 1),
      (std::vector<std::optional<std::string>>{"L1"}));
  EXPECT_EQ(
      varcharsOf(*capturePtr, 3),
      (std::vector<std::optional<std::string>>{"R1"}));
}

// Rows accumulate across batches into their window; the window fires once
// with every row of both sides present.
TEST_F(WindowJoinTest, testMultiBatchAccumulation) {
  auto capture = std::make_unique<CaptureTarget>(
      driverCtx_.get(), joinedType(), nextNodeId());
  auto* capturePtr = capture.get();
  auto join = makeJoin(
      makeNode(
          core::JoinType::kInner,
          joinedType(),
          {0},
          {0},
          {BIGINT()},
          equiCondition()),
      std::move(capture));

  join->addInput(leftAppendOnly({1}, {10010}, {"L1"}));
  join->addInput(rightAppendOnly({1}, {10010}, {"R1"}));
  join->addInput(leftAppendOnly({1}, {10010}, {"L2"}));
  join->addInput(rightAppendOnly({1}, {10010}, {"R2"}));
  join->processWatermark(10009, 0);
  join->processWatermark(10009, 1);

  EXPECT_EQ(
      varcharsOf(*capturePtr, 1),
      (std::vector<std::optional<std::string>>{"L1", "L1", "L2", "L2"}));
  EXPECT_EQ(
      varcharsOf(*capturePtr, 3),
      (std::vector<std::optional<std::string>>{"R1", "R2", "R1", "R2"}));
}

// The due index is derived from the state: a restored join fires its
// windows on the first watermark after the restore, without new rows, and
// the output equals the uninterrupted run.
TEST_F(WindowJoinTest, testSnapshotRestoreRoundTrip) {
  auto makeWarmJoin = [&](std::unique_ptr<StatefulOperator> capture) {
    return makeJoin(
        makeNode(
            core::JoinType::kInner,
            joinedType(),
            {0},
            {0},
            {BIGINT()},
            equiCondition()),
        std::move(capture));
  };

  // Uninterrupted reference run.
  auto refCapture = std::make_unique<CaptureTarget>(
      driverCtx_.get(), joinedType(), nextNodeId());
  auto* refPtr = refCapture.get();
  auto ref = makeWarmJoin(std::move(refCapture));
  ref->addInput(leftAppendOnly({1, 2}, {10010, 10010}, {"L1", "L2"}));
  ref->addInput(rightAppendOnly({1, 2}, {10010, 10010}, {"R1", "R2"}));
  ref->processWatermark(10009, 0);
  ref->processWatermark(10009, 1);
  ref->addInput(leftAppendOnly({1}, {20010}, {"L3"}));
  ref->addInput(rightAppendOnly({1}, {20010}, {"R3"}));
  ref->processWatermark(20009, 0);
  ref->processWatermark(20009, 1);

  // Snapshotted run: warm up with the first window, snapshot mid-stream,
  // restore into a fresh operator, continue with the second window.
  auto captureA = std::make_unique<CaptureTarget>(
      driverCtx_.get(), joinedType(), nextNodeId());
  auto joinA = makeWarmJoin(std::move(captureA));
  joinA->addInput(leftAppendOnly({1, 2}, {10010, 10010}, {"L1", "L2"}));
  joinA->addInput(rightAppendOnly({1, 2}, {10010, 10010}, {"R1", "R2"}));
  const auto bytes = joinA->keyedStateBackend().snapshot();

  auto captureB = std::make_unique<CaptureTarget>(
      driverCtx_.get(), joinedType(), nextNodeId());
  auto* captureBPtr = captureB.get();
  auto joinB = makeWarmJoin(std::move(captureB));
  joinB->keyedStateBackend().restore(bytes);
  joinB->onStateRestored();

  // The first window fires on the first watermark after the restore - no
  // new rows needed: the due index was rebuilt from the restored state.
  joinB->processWatermark(10009, 0);
  joinB->processWatermark(10009, 1);
  ASSERT_EQ(captureBPtr->records().size(), 2);

  // New data continues from the restored state.
  joinB->addInput(leftAppendOnly({1}, {20010}, {"L3"}));
  joinB->addInput(rightAppendOnly({1}, {20010}, {"R3"}));
  joinB->processWatermark(20009, 0);
  joinB->processWatermark(20009, 1);

  EXPECT_EQ(varcharsOf(*captureBPtr, 1), varcharsOf(*refPtr, 1));
  EXPECT_EQ(varcharsOf(*captureBPtr, 3), varcharsOf(*refPtr, 3));
  EXPECT_EQ(bigintsOf(*captureBPtr, 0), bigintsOf(*refPtr, 0));
  EXPECT_EQ(bigintsOf(*captureBPtr, 2), bigintsOf(*refPtr, 2));
}

// The keyed-state spec is a plan constraint: a spec whose second channel
// list does not pair with the first fails loud at construction.
TEST_F(WindowJoinTest, testUnpairedKeyChannelsFailLoud) {
  VELOX_ASSERT_THROW(
      makeNode(
          core::JoinType::kInner,
          joinedType(),
          /*leftKeyChannels=*/{0},
          /*rightKeyChannels=*/{0, 1},
          /*keyTypes=*/{BIGINT()},
          equiCondition()),
      "key channel");
}

// Design UT matrix example 2 (explicit INSERT): both inputs carry an
// explicit $row_kind column with every row INSERT(0) - the rowKind
// validation takes its accept path and the join fires the same cross
// product as the append-only (no column) shape.
TEST_F(WindowJoinTest, testExplicitInsertKindsAccepted) {
  auto capture = std::make_unique<CaptureTarget>(
      driverCtx_.get(), joinedType(), nextNodeId());
  auto* capturePtr = capture.get();
  auto join = makeJoin(
      makeNode(
          core::JoinType::kInner,
          joinedType(),
          {0},
          {0},
          {BIGINT()},
          equiCondition()),
      std::move(capture));

  join->addInput(leftWithKinds(
      {1, 1},
      {10010, 10010},
      {"L1", "L2"},
      {RowKind::INSERT, RowKind::INSERT}));
  join->addInput(rightWithKinds(
      {1, 1},
      {10010, 10010},
      {"R1", "R2"},
      {RowKind::INSERT, RowKind::INSERT}));
  join->processWatermark(10009, 0);
  join->processWatermark(10009, 1);

  ASSERT_EQ(capturePtr->records().size(), 2);
  recordAt(*capturePtr, 0);
  recordAt(*capturePtr, 1);
  EXPECT_EQ(
      varcharsOf(*capturePtr, 1),
      (std::vector<std::optional<std::string>>{"L1", "L1", "L2", "L2"}));
  EXPECT_EQ(
      varcharsOf(*capturePtr, 3),
      (std::vector<std::optional<std::string>>{"R1", "R2", "R1", "R2"}));
}

// Design UT matrix example 4 (changelog assertion): for every join type a
// representative fire emits StreamRecords that are all append-only - the
// window join never emits a rowKind column, its output is INSERT-only
// (no updating stream).
TEST_F(WindowJoinTest, testOutputIsAppendOnlyAcrossJoinTypes) {
  const std::vector<core::JoinType> joinTypes = {
      core::JoinType::kInner,
      core::JoinType::kLeft,
      core::JoinType::kRight,
      core::JoinType::kFull,
      core::JoinType::kLeftSemiFilter,
      core::JoinType::kAnti};

  for (auto joinType : joinTypes) {
    SCOPED_TRACE("joinType " + std::to_string(int(joinType)));
    auto outputType = joinType == core::JoinType::kLeftSemiFilter ||
            joinType == core::JoinType::kAnti
        ? leftProjectionType()
        : joinedType();
    auto capture = std::make_unique<CaptureTarget>(
        driverCtx_.get(), outputType, nextNodeId());
    auto* capturePtr = capture.get();
    auto join = makeJoin(
        makeNode(joinType, outputType, {0}, {0}, {BIGINT()}, equiCondition()),
        std::move(capture));

    // Key 1 matches, key 2 is left-only, key 3 is right-only: every type
    // emits at least one record, so no type's check is vacuous.
    join->addInput(leftAppendOnly({1, 2}, {10010, 10010}, {"L1", "L2"}));
    join->addInput(rightAppendOnly({1, 3}, {10010, 10010}, {"R1", "R3"}));
    join->processWatermark(10009, 0);
    join->processWatermark(10009, 1);

    EXPECT_FALSE(capturePtr->records().empty());
    for (size_t i = 0; i < capturePtr->records().size(); ++i) {
      // recordAt asserts the record is a StreamRecord and appendOnly
      // (no rowKind column - INSERT by construction).
      recordAt(*capturePtr, i);
    }
  }
}

// Design UT matrix example 7, counter face: a late row increments the
// dropping side's counter (and only that side's), per row and per batch,
// while on-time rows and the fired output stay unaffected - late rows
// never enter the state.
TEST_F(WindowJoinTest, testLateRecordCounters) {
  auto capture = std::make_unique<CaptureTarget>(
      driverCtx_.get(), joinedType(), nextNodeId());
  auto* capturePtr = capture.get();
  auto join = makeJoin(
      makeNode(
          core::JoinType::kInner,
          joinedType(),
          {0},
          {0},
          {BIGINT()},
          equiCondition()),
      std::move(capture));

  EXPECT_EQ(join->leftLateRecordsDropped(), 0);
  EXPECT_EQ(join->rightLateRecordsDropped(), 0);

  join->addInput(leftAppendOnly({1}, {10010}, {"L1"}));
  join->addInput(rightAppendOnly({1}, {10010}, {"R1"}));
  // On-time rows never count.
  EXPECT_EQ(join->leftLateRecordsDropped(), 0);
  EXPECT_EQ(join->rightLateRecordsDropped(), 0);
  join->processWatermark(10009, 0);
  join->processWatermark(10009, 1);
  ASSERT_EQ(capturePtr->records().size(), 1);

  // A late left row counts on its own side only.
  join->addInput(leftAppendOnly({1}, {10010}, {"LATE_L"}));
  EXPECT_EQ(join->leftLateRecordsDropped(), 1);
  EXPECT_EQ(join->rightLateRecordsDropped(), 0);

  // A late right row counts on its own side only.
  join->addInput(rightAppendOnly({1}, {10010}, {"LATE_R"}));
  EXPECT_EQ(join->leftLateRecordsDropped(), 1);
  EXPECT_EQ(join->rightLateRecordsDropped(), 1);

  // Mixed late batches count per row, per side.
  join->addInput(leftAppendOnly({1, 2}, {10010, 10010}, {"L1", "L2"}));
  join->addInput(rightAppendOnly({1, 3}, {10010, 10010}, {"R1", "R3"}));
  EXPECT_EQ(join->leftLateRecordsDropped(), 3);
  EXPECT_EQ(join->rightLateRecordsDropped(), 3);

  // The late rows never entered the state: the next window fires with the
  // fresh rows only, and the counters hold steady.
  join->addInput(leftAppendOnly({2}, {20010}, {"L2"}));
  join->addInput(rightAppendOnly({2}, {20010}, {"R2"}));
  EXPECT_EQ(join->leftLateRecordsDropped(), 3);
  EXPECT_EQ(join->rightLateRecordsDropped(), 3);
  join->processWatermark(20009, 0);
  join->processWatermark(20009, 1);
  ASSERT_EQ(capturePtr->records().size(), 2);
  EXPECT_EQ(
      varcharsOf(*capturePtr, 1),
      (std::vector<std::optional<std::string>>{"L1", "L2"}));
  EXPECT_EQ(
      varcharsOf(*capturePtr, 3),
      (std::vector<std::optional<std::string>>{"R1", "R2"}));
}

// Design UT matrix example 9, the full-match cell of the six-type x
// four-shape matrix: both sides hold several rows of the same key in one
// window and every row matches, so the whole cross product ships - no
// padding on left/right/full, one emission per probe row on semi, nothing
// on anti.
TEST_F(WindowJoinTest, testJoinTypeMatrixFullMatchCells) {
  struct Shape {
    core::JoinType joinType;
    RowTypePtr outputType;
    std::vector<std::optional<int64_t>> l_k;
    std::vector<std::optional<std::string>> l_v;
    std::vector<std::optional<int64_t>> r_k;
    std::vector<std::optional<std::string>> r_v;
  };
  const std::vector<Shape> shapes = {
      {core::JoinType::kInner,
       joinedType(),
       {1, 1, 1, 1},
       {"L1", "L1", "L2", "L2"},
       {1, 1, 1, 1},
       {"R1", "R2", "R1", "R2"}},
      {core::JoinType::kLeft,
       joinedType(),
       {1, 1, 1, 1},
       {"L1", "L1", "L2", "L2"},
       {1, 1, 1, 1},
       {"R1", "R2", "R1", "R2"}},
      {core::JoinType::kRight,
       joinedType(),
       {1, 1, 1, 1},
       {"L1", "L1", "L2", "L2"},
       {1, 1, 1, 1},
       {"R1", "R2", "R1", "R2"}},
      {core::JoinType::kFull,
       joinedType(),
       {1, 1, 1, 1},
       {"L1", "L1", "L2", "L2"},
       {1, 1, 1, 1},
       {"R1", "R2", "R1", "R2"}},
      {core::JoinType::kLeftSemiFilter,
       leftProjectionType(),
       {1, 1},
       {"L1", "L2"},
       {},
       {}},
      {core::JoinType::kAnti, leftProjectionType(), {}, {}, {}, {}}};

  for (const auto& shape : shapes) {
    SCOPED_TRACE("joinType " + std::to_string(int(shape.joinType)));
    auto capture = std::make_unique<CaptureTarget>(
        driverCtx_.get(), shape.outputType, nextNodeId());
    auto* capturePtr = capture.get();
    auto join = makeJoin(
        makeNode(
            shape.joinType,
            shape.outputType,
            {0},
            {0},
            {BIGINT()},
            equiCondition()),
        std::move(capture));

    join->addInput(leftAppendOnly({1, 1}, {10010, 10010}, {"L1", "L2"}));
    join->addInput(rightAppendOnly({1, 1}, {10010, 10010}, {"R1", "R2"}));
    join->processWatermark(10009, 0);
    join->processWatermark(10009, 1);

    EXPECT_EQ(bigintsOf(*capturePtr, 0), shape.l_k);
    EXPECT_EQ(varcharsOf(*capturePtr, 1), shape.l_v);
    if (shape.outputType->size() > 2) {
      EXPECT_EQ(bigintsOf(*capturePtr, 2), shape.r_k);
      EXPECT_EQ(varcharsOf(*capturePtr, 3), shape.r_v);
    }
  }
}

// Design UT matrix example 12 (many keys, scale): a thousand keys over a
// hundred windows - 100k (key, window end) buckets, one row per side
// each. Every bucket fires exactly its own pairing: the values name their
// bucket, so any cross-bucket pollution, loss or duplication shows up as
// a value or count mismatch in that window's fire. No performance
// threshold is asserted - correctness at scale, no crash.
TEST_F(WindowJoinTest, testManyKeysAndWindowsIsolation) {
  constexpr int kNumKeys = 1000;
  constexpr int kNumWindows = 100;

  auto capture = std::make_unique<CaptureTarget>(
      driverCtx_.get(), joinedType(), nextNodeId());
  auto* capturePtr = capture.get();
  auto join = makeJoin(
      makeNode(
          core::JoinType::kInner,
          joinedType(),
          {0},
          {0},
          {BIGINT()},
          equiCondition()),
      std::move(capture));

  // One batch per side: every (key, window) bucket holds one row on each
  // side, its value naming the bucket ("L<k>w<w>" / "R<k>w<w>"). The full
  // input sits in the state before anything fires - the state carries the
  // whole load at once.
  std::vector<std::optional<int64_t>> keys;
  std::vector<int64_t> windowEnds;
  std::vector<std::string> leftVals, rightVals;
  keys.reserve(kNumKeys * kNumWindows);
  windowEnds.reserve(kNumKeys * kNumWindows);
  leftVals.reserve(kNumKeys * kNumWindows);
  rightVals.reserve(kNumKeys * kNumWindows);
  for (int w = 0; w < kNumWindows; ++w) {
    const auto end = (w + 1) * kWindowSize;
    for (int k = 0; k < kNumKeys; ++k) {
      keys.push_back(k);
      windowEnds.push_back(end);
      leftVals.push_back("L" + std::to_string(k) + "w" + std::to_string(w));
      rightVals.push_back("R" + std::to_string(k) + "w" + std::to_string(w));
    }
  }
  join->addInput(leftAppendOnly(keys, windowEnds, leftVals));
  join->addInput(rightAppendOnly(keys, windowEnds, rightVals));

  // Fire window by window: each trigger time releases exactly its own
  // thousand buckets, keyed and valued as filed.
  for (int w = 0; w < kNumWindows; ++w) {
    SCOPED_TRACE("window " + std::to_string(w));
    const auto trigger = (w + 1) * kWindowSize - 1;
    join->processWatermark(trigger, 0);
    join->processWatermark(trigger, 1);

    EXPECT_EQ(capturePtr->records().size(), kNumKeys);
    std::vector<std::optional<int64_t>> expectKeys;
    std::vector<std::optional<std::string>> expectLeft, expectRight;
    expectKeys.reserve(kNumKeys);
    expectLeft.reserve(kNumKeys);
    expectRight.reserve(kNumKeys);
    for (int k = 0; k < kNumKeys; ++k) {
      expectKeys.push_back(k);
      expectLeft.push_back("L" + std::to_string(k) + "w" + std::to_string(w));
      expectRight.push_back("R" + std::to_string(k) + "w" + std::to_string(w));
    }
    EXPECT_EQ(bigintsOf(*capturePtr, 0), expectKeys);
    EXPECT_EQ(varcharsOf(*capturePtr, 1), expectLeft);
    EXPECT_EQ(bigintsOf(*capturePtr, 2), expectKeys);
    EXPECT_EQ(varcharsOf(*capturePtr, 3), expectRight);
    capturePtr->clear();
  }
}

// Design UT matrix example 15 (rescale restore): a snapshot written by a
// backend covering the whole key-group space restores into backends
// covering key-group sub-ranges - each half picks its own key groups out
// of the stream and skips the rest without error, and together the halves
// carry every key exactly once (the scale-down rescale shape of
// HeapKeyedStateBackend::restore).
TEST_F(WindowJoinTest, testRescaleRestorePartialKeyGroupRange) {
  constexpr uint32_t kMaxParallelism = 128;
  constexpr int64_t kNumKeys = 40;

  std::vector<std::optional<int64_t>> keys;
  std::vector<int64_t> windowEnds;
  std::vector<std::string> leftVals, rightVals;
  std::set<int64_t> expectedKeys;
  for (int64_t k = 1; k <= kNumKeys; ++k) {
    keys.push_back(k);
    windowEnds.push_back(10010);
    leftVals.push_back("L" + std::to_string(k));
    rightVals.push_back("R" + std::to_string(k));
    expectedKeys.insert(k);
  }

  // Source: the fixture backend covers the whole key-group space. The
  // window stays unfired so every entry rides the snapshot.
  auto src = makeJoin(
      makeNode(
          core::JoinType::kInner,
          joinedType(),
          {0},
          {0},
          {BIGINT()},
          equiCondition()),
      std::make_unique<CaptureTarget>(
          driverCtx_.get(), joinedType(), nextNodeId()));
  src->addInput(leftAppendOnly(keys, windowEnds, leftVals));
  src->addInput(rightAppendOnly(keys, windowEnds, rightVals));
  const auto bytes = src->keyedStateBackend().snapshot();

  auto rangeBackend = [&](uint32_t start, uint32_t num) {
    return std::make_unique<HashMapStateBackend>(
        std::make_shared<const KeyedStateBackendParameters>(
            StateBackendType::HEAP,
            "job",
            "window-join",
            kMaxParallelism,
            start,
            num));
  };

  // Restores the full-range stream into a backend covering [start,
  // start + num) and fires: the keys that come back are exactly the
  // stream's keys that hash into that sub-range, each with its rows
  // intact. ASSERTs return from this lambda only, leaving the out set
  // empty for the surrounding assertions to fail on.
  auto restoreHalfAndCollect = [&](uint32_t start,
                                   uint32_t num,
                                   std::set<int64_t>* firedOut) {
    auto capture = std::make_unique<CaptureTarget>(
        driverCtx_.get(), joinedType(), nextNodeId());
    auto* capturePtr = capture.get();
    auto backend = rangeBackend(start, num);
    auto join = makeJoin(
        makeNode(
            core::JoinType::kInner,
            joinedType(),
            {0},
            {0},
            {BIGINT()},
            equiCondition()),
        std::move(capture),
        backend.get());
    // Out-of-range key groups of the stream are skipped here, not an
    // error.
    join->keyedStateBackend().restore(bytes);
    join->onStateRestored();
    join->processWatermark(10009, 0);
    join->processWatermark(10009, 1);

    const auto firedKeys = bigintsOf(*capturePtr, 0);
    EXPECT_EQ(bigintsOf(*capturePtr, 2), firedKeys);
    const auto firedLeft = varcharsOf(*capturePtr, 1);
    const auto firedRight = varcharsOf(*capturePtr, 3);
    for (size_t i = 0; i < firedKeys.size(); ++i) {
      ASSERT_TRUE(firedKeys[i].has_value());
      EXPECT_EQ(firedLeft[i].value_or(""), "L" + std::to_string(*firedKeys[i]));
      EXPECT_EQ(
          firedRight[i].value_or(""), "R" + std::to_string(*firedKeys[i]));
      firedOut->insert(*firedKeys[i]);
    }
  };

  std::set<int64_t> lowerKeys, upperKeys;
  restoreHalfAndCollect(0, 64, &lowerKeys);
  restoreHalfAndCollect(64, 64, &upperKeys);

  // The halves are disjoint, together complete, and both carry state (the
  // key hash spreads the keys over the whole key-group space).
  EXPECT_FALSE(lowerKeys.empty());
  EXPECT_FALSE(upperKeys.empty());
  EXPECT_EQ(lowerKeys.size() + upperKeys.size(), expectedKeys.size());
  std::set<int64_t> allKeys = lowerKeys;
  allKeys.insert(upperKeys.begin(), upperKeys.end());
  EXPECT_EQ(allKeys, expectedKeys);
}

// Design UT matrix example 16 (fire clears state): after a window fires,
// its (key, window end) entries are gone from both sides - a FULL join
// restores the post-fire snapshot and stays silent at the fired window's
// trigger time (a leftover on either side would emit a padded row), while
// an unfired window from the same snapshot fires normally.
TEST_F(WindowJoinTest, testFireClearsStateBothSides) {
  auto makeFullJoin = [&](std::unique_ptr<StatefulOperator> capture) {
    return makeJoin(
        makeNode(
            core::JoinType::kFull,
            joinedType(),
            {0},
            {0},
            {BIGINT()},
            equiCondition()),
        std::move(capture));
  };

  auto captureA = std::make_unique<CaptureTarget>(
      driverCtx_.get(), joinedType(), nextNodeId());
  auto* captureAPtr = captureA.get();
  auto joinA = makeFullJoin(std::move(captureA));

  // Window [0, 10010) fires completely.
  joinA->addInput(leftAppendOnly({1}, {10010}, {"L1"}));
  joinA->addInput(rightAppendOnly({1}, {10010}, {"R1"}));
  joinA->processWatermark(10009, 0);
  joinA->processWatermark(10009, 1);
  ASSERT_EQ(captureAPtr->records().size(), 1);

  // An unfired window rides along as the restore control: it must survive
  // the same snapshot the fired window must not.
  joinA->addInput(leftAppendOnly({1}, {20010}, {"L2"}));
  joinA->addInput(rightAppendOnly({1}, {20010}, {"R2"}));
  const auto bytes = joinA->keyedStateBackend().snapshot();

  auto captureB = std::make_unique<CaptureTarget>(
      driverCtx_.get(), joinedType(), nextNodeId());
  auto* captureBPtr = captureB.get();
  auto joinB = makeFullJoin(std::move(captureB));
  joinB->keyedStateBackend().restore(bytes);
  joinB->onStateRestored();

  // The fired window is gone from both sides: no padded row at its
  // trigger time (a leaked left entry pads from the probe side, a leaked
  // right entry from the build side).
  joinB->processWatermark(10009, 0);
  joinB->processWatermark(10009, 1);
  EXPECT_EQ(captureBPtr->records().size(), 0);

  // The control window survived the fire and the snapshot: it fires with
  // its rows intact.
  joinB->processWatermark(20009, 0);
  joinB->processWatermark(20009, 1);
  ASSERT_EQ(captureBPtr->records().size(), 1);
  EXPECT_EQ(
      varcharsOf(*captureBPtr, 1),
      (std::vector<std::optional<std::string>>{"L2"}));
  EXPECT_EQ(
      varcharsOf(*captureBPtr, 3),
      (std::vector<std::optional<std::string>>{"R2"}));
}

// Design UT matrix example 17 (session window rows): session rows as the
// upstream merger leaves them - unaligned spans keyed by their final
// window end. Rows pair inside their (key, window end) bucket only, each
// session fires at its own end - 1, and the join replicates the window
// end as-is, with no correction or grid alignment (SPEC behavior spec).
TEST_F(WindowJoinTest, testSessionWindowRowsPairByWindowEnd) {
  auto capture = std::make_unique<CaptureTarget>(
      driverCtx_.get(), joinedType(), nextNodeId());
  auto* capturePtr = capture.get();
  auto join = makeJoin(
      makeNode(
          core::JoinType::kInner,
          joinedType(),
          {0},
          {0},
          {BIGINT()},
          equiCondition()),
      std::move(capture));

  // Key 1 owns two disjoint sessions; key 2 one session of its own. All
  // rows sit in the state before anything fires.
  join->addInput(
      leftSessionRows({1, 1}, {3000, 3000}, {15777, 15777}, {"L1", "L1b"}));
  join->addInput(rightSessionRows({1}, {3000}, {15777}, {"R1"}));
  join->addInput(leftSessionRows({2}, {8000}, {19211}, {"L3"}));
  join->addInput(rightSessionRows({2}, {8000}, {19211}, {"R3"}));
  join->addInput(leftSessionRows({1}, {20000}, {26123}, {"L2"}));
  join->addInput(rightSessionRows({1}, {20000}, {26123}, {"R2"}));

  // The first session fires at its own, unaligned end: both of its left
  // rows pair with its right row inside the (key 1, 15777) bucket.
  join->processWatermark(15776, 0);
  join->processWatermark(15776, 1);
  ASSERT_EQ(capturePtr->records().size(), 2);
  EXPECT_EQ(
      varcharsOf(*capturePtr, 1),
      (std::vector<std::optional<std::string>>{"L1", "L1b"}));
  EXPECT_EQ(
      varcharsOf(*capturePtr, 3),
      (std::vector<std::optional<std::string>>{"R1", "R1"}));

  // Key 2's session is an isolated bucket: it fires on its own end, never
  // pairing with key 1's rows.
  join->processWatermark(19210, 0);
  join->processWatermark(19210, 1);
  ASSERT_EQ(capturePtr->records().size(), 3);
  EXPECT_EQ(
      varcharsOf(*capturePtr, 1),
      (std::vector<std::optional<std::string>>{"L1", "L1b", "L3"}));
  EXPECT_EQ(
      varcharsOf(*capturePtr, 3),
      (std::vector<std::optional<std::string>>{"R1", "R1", "R3"}));

  // Key 1's second session fires last: the same key never pairs across
  // its two sessions.
  join->processWatermark(26122, 0);
  join->processWatermark(26122, 1);
  ASSERT_EQ(capturePtr->records().size(), 4);
  EXPECT_EQ(
      varcharsOf(*capturePtr, 1),
      (std::vector<std::optional<std::string>>{"L1", "L1b", "L3", "L2"}));
  EXPECT_EQ(
      varcharsOf(*capturePtr, 3),
      (std::vector<std::optional<std::string>>{"R1", "R1", "R3", "R2"}));
}

// Design UT matrix example 18 (session six types): the six-type behavior
// table over session window rows - key 1 matches across a shared session,
// key 2 is left-only in its own session, key 3 is right-only in its own
// session; every session has an unaligned window end.
TEST_F(WindowJoinTest, testSessionWindowJoinTypeMatrix) {
  struct Shape {
    core::JoinType joinType;
    RowTypePtr outputType;
    std::vector<std::optional<int64_t>> l_k;
    std::vector<std::optional<std::string>> l_v;
    std::vector<std::optional<int64_t>> r_k;
    std::vector<std::optional<std::string>> r_v;
  };
  const std::vector<Shape> shapes = {
      {core::JoinType::kInner, joinedType(), {1}, {"L1"}, {1}, {"R1"}},
      {core::JoinType::kLeft,
       joinedType(),
       {1, 2},
       {"L1", "L2"},
       {1, std::nullopt},
       {"R1", std::nullopt}},
      {core::JoinType::kRight,
       joinedType(),
       {1, std::nullopt},
       {"L1", std::nullopt},
       {1, 3},
       {"R1", "R3"}},
      {core::JoinType::kFull,
       joinedType(),
       {1, 2, std::nullopt},
       {"L1", "L2", std::nullopt},
       {1, std::nullopt, 3},
       {"R1", std::nullopt, "R3"}},
      {core::JoinType::kLeftSemiFilter,
       leftProjectionType(),
       {1},
       {"L1"},
       {},
       {}},
      {core::JoinType::kAnti, leftProjectionType(), {2}, {"L2"}, {}, {}}};

  for (const auto& shape : shapes) {
    SCOPED_TRACE("joinType " + std::to_string(int(shape.joinType)));
    auto capture = std::make_unique<CaptureTarget>(
        driverCtx_.get(), shape.outputType, nextNodeId());
    auto* capturePtr = capture.get();
    auto join = makeJoin(
        makeNode(
            shape.joinType,
            shape.outputType,
            {0},
            {0},
            {BIGINT()},
            equiCondition()),
        std::move(capture));

    // Session rows with three unaligned window ends, one key per session:
    // key 1 [3000, 15777) on both sides, key 2 [12000, 19211) left only,
    // key 3 [18000, 22333) right only.
    join->addInput(
        leftSessionRows({1, 2}, {3000, 12000}, {15777, 19211}, {"L1", "L2"}));
    join->addInput(
        rightSessionRows({1, 3}, {3000, 18000}, {15777, 22333}, {"R1", "R3"}));

    // One watermark past every session end fires them in end order.
    join->processWatermark(22332, 0);
    join->processWatermark(22332, 1);

    EXPECT_EQ(bigintsOf(*capturePtr, 0), shape.l_k);
    EXPECT_EQ(varcharsOf(*capturePtr, 1), shape.l_v);
    if (shape.outputType->size() > 2) {
      EXPECT_EQ(bigintsOf(*capturePtr, 2), shape.r_k);
      EXPECT_EQ(varcharsOf(*capturePtr, 3), shape.r_v);
    }
  }
}

} // namespace
} // namespace facebook::velox::stateful::test
