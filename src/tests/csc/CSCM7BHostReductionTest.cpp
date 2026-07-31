#include "csc/CSCDescriptorEngine.h"
#include "csc/CSCPartialResultPath.h"
#include "tests/csc/CSCLayout.h"

#include <gtest/gtest.h>

#include <cmath>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

using namespace csc_descriptor;

namespace csc_descriptor {

struct CSCM7BHostReductionTestAccess {
    static void corruptFrontRow(CSCPartialResultPath& path,
                                uint32_t row)
    {
        ASSERT_FALSE(path.host_return_queue_.empty());
        path.host_return_queue_.front()
            .burst.records[0].record.row_idx = row;
    }
};

}  // namespace csc_descriptor

namespace {

CSCPartialResultPathConfig reductionConfig()
{
    CSCPartialResultPathConfig config;
    config.global_bg_count = 8;
    config.channel_count = 2;
    config.ranks_per_channel = 1;
    config.bank_groups_per_rank = 4;
    config.buffer_capacity_bursts_per_bg = 8;
    config.pending_capacity_bursts_per_bg = 8;
    config.write_latency_cycles = 2;
    config.max_inflight_writes_per_bg = 8;
    config.write_issue_limit_per_rank_per_cycle = 4;
    config.read_latency_cycles = 3;
    config.read_issue_limit_per_channel_per_cycle = 1;
    config.max_inflight_reads_per_channel = 8;
    config.host_return_queue_capacity_bursts = 8;
    config.host_reduction_enabled = true;
    config.reduction_queue_capacity_records = 16;
    config.host_reduce_records_per_cycle = 4;
    config.host_reduce_latency_cycles = 2;
    return config;
}

CSCBGAOutput output(uint32_t row, uint64_t sequence, float value,
                    uint32_t contributions = 1)
{
    CSCBGAOutput result;
    result.row_idx = row;
    result.value = value;
    result.generation = 7;
    result.output_sequence = sequence;
    result.contribution_count = contributions;
    result.reason = CSCBGAOutputReason::FINAL_DRAIN;
    return result;
}

void accept(CSCPartialResultPath& path, uint32_t bg,
            uint32_t row, uint64_t sequence, float value,
            uint32_t contributions = 1)
{
    const auto item = output(row, sequence, value, contributions);
    bool available = true;
    CSCBGAOutputPortCallbacks source;
    source.has_output = [&](uint32_t) { return available; };
    source.peek_output =
        [&](uint32_t) -> const CSCBGAOutput& { return item; };
    source.accept_output = [&](uint32_t) { available = false; };
    ASSERT_EQ(transferCSCBGAOutput(bg, source, path),
              CSCBGAOutputTransferResult::ACCEPTED);
}

std::vector<bool> done(const CSCPartialResultPath& path)
{
    return std::vector<bool>(path.globalBGCount(), true);
}

void runThrough(CSCPartialResultPath& path, uint64_t last_cycle)
{
    for (uint64_t cycle = path.currentCycle() + 1;
         cycle <= last_cycle; ++cycle)
        path.step(cycle, done(path));
}

void finish(CSCPartialResultPath& path)
{
    for (uint64_t cycle = path.currentCycle() + 1;
         cycle < 500 && !path.hostReductionComplete(); ++cycle)
        path.step(cycle, done(path));
    ASSERT_TRUE(path.hostReductionComplete()) << path.errorMessage();
}

uint32_t bits(float value)
{
    uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float orderedAdd(std::initializer_list<float> values)
{
    float result = 0.0F;
    for (float value : values)
        result = static_cast<float>(result + value);
    return result;
}

struct Views {
    std::array<std::vector<float>, kCSCGlobalBGs> x;
    std::array<CSCBGImageView, kCSCGlobalBGs> views;

    Views(const CSCLayout& layout,
          const std::vector<float>& original_x)
    {
        for (uint32_t bg = 0; bg < kCSCGlobalBGs; ++bg) {
            for (uint32_t column :
                 layout.bg[bg].x_slot_to_original_col)
                x[bg].push_back(original_x[column]);
            views[bg] = {&layout.bg[bg].values,
                         &layout.bg[bg].row_indices,
                         &layout.bg[bg].descriptors, &x[bg]};
        }
    }
};

}  // namespace

TEST(CSCM7BHostReductionTest, ReturnedBurstWaitsForHostVisibleCycle)
{
    CSCPartialResultPath path(reductionConfig(), 16, 7);
    accept(path, 0, 2, 1, 1.0F);
    runThrough(path, 8);
    ASSERT_EQ(path.hostReturnQueueSize(), 1U);
    EXPECT_EQ(path.reductionQueueSize(), 0U);
    EXPECT_EQ(path.reductionCounters()
                  .first_returned_burst_accept_cycle, 0U);
    path.step(9, done(path));
    EXPECT_EQ(path.hostReturnQueueSize(), 0U);
    EXPECT_EQ(path.reductionQueueSize(), 1U);
    EXPECT_EQ(path.reductionCounters()
                  .first_returned_burst_accept_cycle, 9U);
}

TEST(CSCM7BHostReductionTest,
     BurstAcceptanceRequiresWholeRecordCapacity)
{
    auto config = reductionConfig();
    config.channel_count = 1;
    config.global_bg_count = 4;
    config.reduction_queue_capacity_records = 4;
    config.host_reduce_records_per_cycle = 1;
    config.host_reduce_latency_cycles = 2;
    CSCPartialResultPath path(config, 32, 7);
    for (uint32_t i = 0; i < 8; ++i)
        accept(path, 0, i, i + 1, float(i + 1));
    runThrough(path, 8);
    ASSERT_EQ(path.readbackCounters().returned_bursts_delivered, 1U);
    ASSERT_EQ(path.reductionQueueSize(), 4U);
    runThrough(path, 10);
    EXPECT_EQ(path.readbackCounters().returned_bursts_delivered, 1U);
    EXPECT_EQ(path.hostReturnQueueSize(), 1U);
    EXPECT_GT(path.reductionCounters()
                  .reduction_queue_backpressure_cycles, 0U);
    finish(path);
    EXPECT_EQ(path.readbackCounters().returned_bursts_delivered, 2U);
}

TEST(CSCM7BHostReductionTest, BurstExpansionPreservesRecordOrder)
{
    auto config = reductionConfig();
    config.host_reduce_records_per_cycle = 1;
    CSCPartialResultPath path(config, 8, 7);
    accept(path, 0, 0, 1, 1.0F);
    accept(path, 0, 0, 2, 2.0F);
    accept(path, 0, 0, 3, 3.0F);
    accept(path, 0, 0, 4, 4.0F);
    finish(path);
    EXPECT_EQ(bits(path.hostReducedResult()[0]),
              bits(orderedAdd({1.0F, 2.0F, 3.0F, 4.0F})));
    EXPECT_EQ(path.reductionCounters().reduced_partial_records, 4U);
}

TEST(CSCM7BHostReductionTest, AcceptedBurstCannotBeAcceptedTwice)
{
    CSCPartialResultPath path(reductionConfig(), 8, 7);
    accept(path, 0, 0, 1, 2.0F);
    finish(path);
    EXPECT_EQ(path.readbackCounters().returned_bursts_committed, 1U);
    EXPECT_EQ(path.readbackCounters().returned_bursts_delivered, 1U);
    EXPECT_EQ(path.reductionCounters().reduction_records_enqueued, 1U);
}

TEST(CSCM7BHostReductionTest,
     ReductionQueueBackpressuresReturnAndReadQueues)
{
    auto config = reductionConfig();
    config.channel_count = 1;
    config.global_bg_count = 4;
    config.reduction_queue_capacity_records = 4;
    config.host_reduce_records_per_cycle = 1;
    config.host_reduce_latency_cycles = 5;
    config.host_return_queue_capacity_bursts = 1;
    CSCPartialResultPath path(config, 32, 7);
    for (uint32_t i = 0; i < 12; ++i)
        accept(path, 0, i, i + 1, float(i));
    runThrough(path, 13);
    EXPECT_GT(path.reductionCounters()
                  .reduction_queue_backpressure_cycles, 0U);
    EXPECT_GT(path.readbackCounters()
                  .return_queue_backpressure_cycles, 0U);
    EXPECT_LT(path.readbackCounters().read_requests_issued,
              path.counters().write_requests_completed);
    finish(path);
}

TEST(CSCM7BHostReductionTest,
     ReductionStartsNextCycleAfterQueueCommit)
{
    CSCPartialResultPath path(reductionConfig(), 8, 7);
    accept(path, 0, 0, 1, 1.0F);
    runThrough(path, 9);
    EXPECT_EQ(path.reductionCounters()
                  .first_returned_burst_accept_cycle, 9U);
    EXPECT_EQ(path.reductionCounters()
                  .first_reduction_batch_issue_cycle, 0U);
    path.step(10, done(path));
    EXPECT_EQ(path.reductionCounters()
                  .first_reduction_batch_issue_cycle, 10U);
}

TEST(CSCM7BHostReductionTest, RecordsPerCycleAndLatencyAreHonored)
{
    auto config = reductionConfig();
    config.host_reduce_records_per_cycle = 2;
    config.host_reduce_latency_cycles = 3;
    CSCPartialResultPath path(config, 8, 7);
    for (uint32_t i = 0; i < 4; ++i)
        accept(path, 0, i, i + 1, float(i));
    runThrough(path, 9);
    EXPECT_EQ(path.reductionCounters().reduction_records_issued, 2U);
    EXPECT_TRUE(path.reductionBatchInflight());
    runThrough(path, 11);
    EXPECT_EQ(path.reductionCounters().reduced_partial_records, 0U);
    path.step(12, done(path));
    EXPECT_EQ(path.reductionCounters().reduced_partial_records, 2U);
    EXPECT_EQ(path.reductionCounters().reduction_batches_issued, 2U);
}

TEST(CSCM7BHostReductionTest, IndexedRowsReduceBitExactly)
{
    CSCPartialResultPath path(reductionConfig(), 6, 7);
    accept(path, 0, 1, 1, 1.25F);
    accept(path, 0, 4, 2, -2.5F);
    finish(path);
    const auto& result = path.hostReducedResult();
    EXPECT_EQ(bits(result[1]), bits(1.25F));
    EXPECT_EQ(bits(result[4]), bits(-2.5F));
    EXPECT_EQ(bits(result[0]), bits(0.0F));
}

TEST(CSCM7BHostReductionTest, CrossBGSameRowMerge)
{
    CSCPartialResultPath path(reductionConfig(), 8, 7);
    accept(path, 0, 3, 1, 1.25F, 2);
    accept(path, 5, 3, 1, 2.5F, 3);
    finish(path);
    EXPECT_EQ(bits(path.hostReducedResult()[3]),
              bits(orderedAdd({1.25F, 2.5F})));
    EXPECT_EQ(path.reductionCounters()
                  .same_row_cross_bg_merges, 1U);
    EXPECT_EQ(path.reductionCounters().reduced_contribution_count, 5U);
}

TEST(CSCM7BHostReductionTest, RepeatedSameRowFromOneBG)
{
    CSCPartialResultPath path(reductionConfig(), 16, 7);
    accept(path, 2, 9, 1, 1.0F, 3);
    accept(path, 2, 1, 2, 0.0F);
    accept(path, 2, 2, 3, 0.0F);
    accept(path, 2, 3, 4, 0.0F);
    accept(path, 2, 9, 5, 2.0F, 2);
    finish(path);
    EXPECT_EQ(bits(path.hostReducedResult()[9]), bits(3.0F));
    EXPECT_EQ(path.reductionCounters()
                  .same_row_repeated_record_merges, 1U);
    EXPECT_EQ(path.reductionCounters().reduced_partial_records, 5U);
    EXPECT_EQ(path.reductionCounters().reduced_contribution_count, 8U);
}

TEST(CSCM7BHostReductionTest, FP32OrderIsDeterministic)
{
    CSCPartialResultPath path(reductionConfig(), 4, 7);
    accept(path, 0, 0, 1, 16777216.0F);
    accept(path, 0, 0, 2, 1.0F);
    accept(path, 0, 0, 3, -16777216.0F);
    finish(path);
    const float expected =
        orderedAdd({16777216.0F, 1.0F, -16777216.0F});
    EXPECT_EQ(bits(path.hostReducedResult()[0]), bits(expected));
    EXPECT_EQ(bits(expected), bits(0.0F));
}

TEST(CSCM7BHostReductionTest, MultipleBurstsPreserveGlobalOrder)
{
    auto config = reductionConfig();
    config.host_reduce_records_per_cycle = 1;
    CSCPartialResultPath path(config, 4, 7);
    accept(path, 0, 0, 1, 16777216.0F);
    accept(path, 4, 0, 1, 1.0F);
    accept(path, 0, 0, 2, -16777216.0F);
    finish(path);
    EXPECT_EQ(bits(path.hostReducedResult()[0]), bits(1.0F));
    EXPECT_EQ(path.reductionCounters().reduced_partial_records, 3U);
}

TEST(CSCM7BHostReductionTest,
     ReadbackCompletePrecedesArithmeticDrain)
{
    auto config = reductionConfig();
    config.host_reduce_records_per_cycle = 1;
    config.host_reduce_latency_cycles = 5;
    CSCPartialResultPath path(config, 8, 7);
    for (uint32_t i = 0; i < 4; ++i)
        accept(path, 0, i, i + 1, 1.0F);
    runThrough(path, 8);
    ASSERT_TRUE(path.hostReadbackComplete());
    EXPECT_FALSE(path.hostReductionComplete());
    EXPECT_EQ(path.hostReturnQueueSize(), 0U);
    EXPECT_GT(path.reductionQueueSize(), 0U);
    finish(path);
}

TEST(CSCM7BHostReductionTest,
     ReducedResultRequiresArithmeticCompletion)
{
    CSCPartialResultPath path(reductionConfig(), 4, 7);
    accept(path, 0, 0, 1, 3.0F);
    EXPECT_THROW(path.hostReducedResult(), std::logic_error);
    finish(path);
    EXPECT_NO_THROW(path.hostReducedResult());
    EXPECT_EQ(bits(path.hostReducedResult()[0]), bits(3.0F));
}

TEST(CSCM7BHostReductionTest, EmptyReductionCompletesWithPositiveZero)
{
    CSCPartialResultPath path(reductionConfig(), 5, 7);
    finish(path);
    ASSERT_TRUE(path.hostReadbackComplete());
    const auto& result = path.hostReducedResult();
    ASSERT_EQ(result.size(), 5U);
    for (float value : result) EXPECT_EQ(bits(value), bits(0.0F));
    EXPECT_EQ(path.reductionCounters().reduced_partial_records, 0U);
}

TEST(CSCM7BHostReductionTest, InvalidRowIsStickyError)
{
    CSCPartialResultPath path(reductionConfig(), 4, 7);
    accept(path, 0, 0, 1, 1.0F);
    runThrough(path, 8);
    CSCM7BHostReductionTestAccess::corruptFrontRow(path, 4);
    path.step(9, done(path));
    EXPECT_TRUE(path.hasError());
    EXPECT_FALSE(path.hostReductionComplete());
    EXPECT_NE(path.errorMessage().find("invalid returned"),
              std::string::npos);
}

TEST(CSCM7BHostReductionTest, RecordAndContributionConservation)
{
    CSCPartialResultPath path(reductionConfig(), 8, 7);
    accept(path, 0, 0, 1, 1.0F, 2);
    accept(path, 0, 1, 2, 2.0F, 3);
    accept(path, 5, 2, 1, 3.0F, 4);
    finish(path);
    EXPECT_EQ(path.counters().partial_records_generated, 3U);
    EXPECT_EQ(path.readbackCounters().readback_records, 3U);
    EXPECT_EQ(path.reductionCounters().reduced_partial_records, 3U);
    EXPECT_EQ(path.counters().partial_record_contribution_sum, 9U);
    EXPECT_EQ(path.readbackCounters().readback_contribution_sum, 9U);
    EXPECT_EQ(path.reductionCounters().reduced_contribution_count, 9U);
    EXPECT_EQ(path.reductionCounters().fp32_host_add_count, 3U);
}

TEST(CSCM7BHostReductionTest, NaNInfBehaviorMatchesFloatReference)
{
    CSCPartialResultPath path(reductionConfig(), 4, 7);
    accept(path, 0, 0, 1,
           std::numeric_limits<float>::quiet_NaN());
    accept(path, 0, 1, 2,
           std::numeric_limits<float>::infinity());
    accept(path, 0, 2, 3,
           -std::numeric_limits<float>::infinity());
    finish(path);
    const auto& result = path.hostReducedResult();
    EXPECT_TRUE(std::isnan(result[0]));
    EXPECT_EQ(result[1], std::numeric_limits<float>::infinity());
    EXPECT_EQ(result[2], -std::numeric_limits<float>::infinity());
}

TEST(CSCM7BHostReductionTest, DeterministicExactCycle)
{
    CSCPartialResultPath path(reductionConfig(), 8, 7);
    accept(path, 0, 0, 1, 1.0F);
    runThrough(path, 12);
    EXPECT_EQ(path.counters().last_write_complete_cycle, 4U);
    EXPECT_EQ(path.counters().partial_writeback_complete_cycle, 4U);
    EXPECT_EQ(path.readbackCounters().first_read_issue_cycle, 5U);
    EXPECT_EQ(path.readbackCounters().first_read_complete_cycle, 8U);
    EXPECT_EQ(path.reductionCounters()
                  .first_returned_burst_accept_cycle, 9U);
    EXPECT_EQ(path.reductionCounters()
                  .first_reduction_batch_issue_cycle, 10U);
    EXPECT_EQ(path.reductionCounters().first_host_reduce_cycle, 12U);
    EXPECT_EQ(path.reductionCounters()
                  .host_readback_complete_cycle, 9U);
    EXPECT_EQ(path.reductionCounters()
                  .host_reduction_complete_cycle, 12U);
    EXPECT_TRUE(path.hostReductionComplete());
}

TEST(CSCM7BHostReductionTest, ResultInvalidUntilM7B5)
{
    const auto layout = buildLayout(
        makeCSC(1, 1, {{0, 0, 3.0F}}),
        MappingPolicy::External, {0});
    Views views(layout, {2.0F});
    CSCNativeExecution execution;
    auto path_config = reductionConfig();
    execution.configurePartialResultWriteback(path_config);
    CSCBGAIntegrationConfig bga;
    bga.enabled = true;
    bga.accumulator.rows = 1;
    bga.accumulator.accumulator_entries = 4;
    bga.accumulator.compare_width = 4;
    bga.accumulator.output_queue_depth = 4;
    bga.output_consumer_mode =
        CSCBGAOutputConsumerMode::PARTIAL_RESULT_WRITEBACK;
    execution.enableProductionBGAIntegration(bga);
    execution.launch(views.views, 1, 1);
    for (uint64_t guard = 0;
         guard < 200000 && !execution.hostReductionComplete();
         ++guard)
        execution.tick();
    ASSERT_TRUE(execution.hostReductionComplete())
        << execution.errorMessage();
    ASSERT_EQ(execution.hostReducedResult().size(), 1U);
    EXPECT_EQ(bits(execution.hostReducedResult()[0]), bits(6.0F));
    EXPECT_FALSE(execution.resultValid());
}
