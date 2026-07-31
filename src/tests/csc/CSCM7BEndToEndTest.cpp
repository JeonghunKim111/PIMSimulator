#include "csc/CSCDescriptorEngine.h"
#include "csc/CSCPartialResultPath.h"
#include "tests/csc/CSCExternalImage.h"
#include "tests/csc/CSCFunctionalModel.h"
#include "tests/csc/CSCLayout.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace csc_descriptor;

namespace csc_descriptor {

struct CSCM7BEndToEndTestAccess {
    static void corruptAcceptedContributionCount(
        CSCNativeExecution& execution)
    {
        ++execution.execution_stats_
              .bga_output_contributions_accepted;
    }
};

}  // namespace csc_descriptor

namespace {

uint32_t bits(float value)
{
    uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
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

CSCPartialResultPathConfig pathConfig()
{
    CSCPartialResultPathConfig config;
    config.buffer_capacity_bursts_per_bg = 16;
    config.pending_capacity_bursts_per_bg = 2;
    config.write_latency_cycles = 2;
    config.max_inflight_writes_per_bg = 1;
    config.write_issue_limit_per_rank_per_cycle = 1;
    config.read_latency_cycles = 3;
    config.read_issue_limit_per_channel_per_cycle = 1;
    config.max_inflight_reads_per_channel = 1;
    config.host_return_queue_capacity_bursts = 2;
    config.host_reduction_enabled = true;
    config.reduction_queue_capacity_records = 4;
    config.host_reduce_records_per_cycle = 1;
    config.host_reduce_latency_cycles = 2;
    return config;
}

CSCBGAIntegrationConfig bgaConfig(uint32_t rows,
                                  uint32_t entries = 4)
{
    CSCBGAIntegrationConfig config;
    config.enabled = true;
    config.accumulator.rows = rows;
    config.accumulator.accumulator_entries = entries;
    config.accumulator.compare_width = entries;
    config.accumulator.output_queue_depth = 4;
    config.output_consumer_mode =
        CSCBGAOutputConsumerMode::PARTIAL_RESULT_WRITEBACK;
    return config;
}

struct EndToEndHarness {
    CSCLayout layout;
    Views views;
    CSCNativeExecution execution;

    EndToEndHarness(const CSCMatrix& matrix, const std::vector<float>& x,
        MappingPolicy policy = MappingPolicy::RoundRobin,
        const std::vector<uint32_t>& owners = {},
        CSCPartialResultPathConfig path = pathConfig(),
        uint32_t accumulator_entries = 4)
        : layout(buildLayout(matrix, policy, owners)),
          views(layout, x)
    {
        execution.configurePartialResultWriteback(path);
        execution.enableProductionBGAIntegration(
            bgaConfig(matrix.rows, accumulator_entries));
        execution.launch(views.views, matrix.rows,
                         matrix.values.size());
    }

    void untilReduction()
    {
        for (uint64_t guard = 0;
             guard < 300000 &&
             !execution.hostReductionComplete() &&
             !execution.hasFailed();
             ++guard)
            execution.tick();
        ASSERT_TRUE(execution.hostReductionComplete())
            << execution.errorMessage();
    }

    void finish()
    {
        for (uint64_t guard = 0;
             guard < 300000 && !execution.isTerminal();
             ++guard)
            execution.tick();
        ASSERT_TRUE(execution.endToEndSpMVComplete())
            << execution.errorMessage();
    }
};

void expectResult(const CSCMatrix& matrix,
                  const std::vector<float>& x,
                  const std::vector<float>& actual)
{
    float error = 0;
    uint32_t row = UINT32_MAX;
    EXPECT_TRUE(compareResults(cpuReference(matrix, x), actual,
                               &error, &row))
        << "max_error=" << error << " row=" << row;
}

}  // namespace

TEST(CSCM7BEndToEndTest, ProductionModeCompletesEndToEnd)
{
    const auto matrix = makeCSC(1, 1, {{0, 0, 3.0F}});
    EndToEndHarness run(matrix, {2.0F});
    run.finish();
    EXPECT_TRUE(run.execution.isDone());
    EXPECT_TRUE(run.execution.resultValid());
    EXPECT_EQ(bits(run.execution.finalResult()[0]), bits(6.0F));
}

TEST(CSCM7BEndToEndTest, ResultValidOnlyAfterEndToEndCompletion)
{
    EndToEndHarness run(makeCSC(1, 1, {{0, 0, 2.0F}}), {4.0F});
    run.untilReduction();
    ASSERT_FALSE(run.execution.resultValid());
    const uint64_t reduction_cycle = run.execution.cycle();
    run.execution.tick();
    EXPECT_TRUE(run.execution.resultValid());
    EXPECT_GT(run.execution.counters().result_valid_cycle,
              reduction_cycle);
}

TEST(CSCM7BEndToEndTest, FinalResultRequiresValidState)
{
    EndToEndHarness run(makeCSC(1, 1, {{0, 0, 2.0F}}), {4.0F});
    EXPECT_THROW(run.execution.finalResult(), std::logic_error);
    run.finish();
    EXPECT_NO_THROW(run.execution.finalResult());
}

TEST(CSCM7BEndToEndTest,
     FinalResultUsesHostReducedAuthoritativeState)
{
    EndToEndHarness run(makeCSC(1, 1, {{0, 0, 2.0F}}), {4.0F});
    run.finish();
    EXPECT_EQ(&run.execution.finalResult(),
              &run.execution.hostReducedResult());
}

TEST(CSCM7BEndToEndTest, ValidationModeNeverProducesProductionResult)
{
    const auto matrix = makeCSC(1, 1, {{0, 0, 2.0F}});
    const auto layout =
        buildLayout(matrix, MappingPolicy::RoundRobin);
    Views views(layout, {4.0F});
    CSCNativeExecution execution;
    auto bga = bgaConfig(1);
    bga.output_consumer_mode =
        CSCBGAOutputConsumerMode::VALIDATION_ROUND_ROBIN;
    execution.enableProductionBGAIntegration(bga);
    execution.launch(views.views, 1, 1);
    for (uint64_t guard = 0;
         guard < 300000 && !execution.isTerminal(); ++guard)
        execution.tick();
    ASSERT_TRUE(execution.isDone()) << execution.errorMessage();
    EXPECT_FALSE(execution.resultValid());
    EXPECT_THROW(execution.finalResult(), std::logic_error);
}

TEST(CSCM7BEndToEndTest, ExternalModeResultInvalid)
{
    CSCNativeExecution execution;
    EXPECT_FALSE(execution.resultValid());
    EXPECT_THROW(execution.finalResult(), std::logic_error);
}

TEST(CSCM7BEndToEndTest, LegacyModeSemanticsUnchanged)
{
    const auto matrix = makeCSC(1, 1, {{0, 0, 2.0F}});
    const auto layout =
        buildLayout(matrix, MappingPolicy::RoundRobin);
    Views views(layout, {4.0F});
    CSCNativeExecution execution;
    execution.launch(views.views, 1, 1);
    for (uint64_t guard = 0;
         guard < 300000 && !execution.isTerminal(); ++guard)
        execution.tick();
    ASSERT_TRUE(execution.isDone()) << execution.errorMessage();
    EXPECT_TRUE(execution.resultValid());
    EXPECT_EQ(execution.hostAccumulate(),
              cpuReference(matrix, {4.0F}));
}

TEST(CSCM7BEndToEndTest, EmptyMatrixProducesValidZeroResult)
{
    EndToEndHarness run(makeCSC(5, 0, {}), {});
    run.finish();
    ASSERT_EQ(run.execution.finalResult().size(), 5U);
    for (float value : run.execution.finalResult())
        EXPECT_EQ(bits(value), bits(0.0F));
    EXPECT_EQ(run.execution.partialResultPath().counters()
                  .partial_records_generated, 0U);
}

TEST(CSCM7BEndToEndTest, PhysicalRecordConservationEndToEnd)
{
    EndToEndHarness run(makeCSC(4, 1, {{0, 0, 1.0F}, {1, 0, 2.0F},
                           {2, 0, 3.0F}, {3, 0, 4.0F}}),
            {2.0F});
    run.finish();
    const auto& path = run.execution.partialResultPath();
    EXPECT_EQ(run.execution.counters().bga_outputs_accepted,
              path.counters().partial_records_generated);
    EXPECT_EQ(path.counters().partial_records_generated,
              path.readbackCounters().readback_records);
    EXPECT_EQ(path.readbackCounters().readback_records,
              path.reductionCounters().reduced_partial_records);
}

TEST(CSCM7BEndToEndTest, ContributionConservationEqualsNNZ)
{
    const auto matrix = makeCSC(
        3, 2, {{0, 0, 1.0F}, {1, 0, 2.0F},
               {1, 1, 3.0F}, {2, 1, 4.0F}});
    EndToEndHarness run(matrix, {2.0F, 3.0F});
    run.finish();
    const auto& path = run.execution.partialResultPath();
    EXPECT_EQ(run.execution.counters().accepted_bga_partials,
              matrix.values.size());
    EXPECT_EQ(path.counters().partial_record_contribution_sum,
              matrix.values.size());
    EXPECT_EQ(path.reductionCounters()
                  .reduced_contribution_count,
              matrix.values.size());
}

TEST(CSCM7BEndToEndTest, ByteConservationEndToEnd)
{
    EndToEndHarness run(makeCSC(2, 1, {{0, 0, 1.0F}, {1, 0, 2.0F}}),
            {2.0F});
    run.finish();
    const auto& path = run.execution.partialResultPath();
    const auto& write = path.counters();
    const auto& read = path.readbackCounters();
    EXPECT_EQ(write.writeback_transferred_bytes,
              read.readback_transferred_bytes);
    EXPECT_EQ(write.writeback_useful_bytes,
              read.readback_useful_bytes);
    EXPECT_EQ(write.writeback_padding_bytes,
              read.readback_padding_bytes);
}

TEST(CSCM7BEndToEndTest, CrossBGSameRowCompletesEndToEnd)
{
    const auto matrix =
        makeCSC(1, 2, {{0, 0, 1.25F}, {0, 1, 2.5F}});
    EndToEndHarness run(matrix, {2.0F, 2.0F}, MappingPolicy::External,
            {0, 5});
    run.finish();
    EXPECT_EQ(bits(run.execution.finalResult()[0]), bits(7.5F));
    EXPECT_EQ(run.execution.partialResultPath()
                  .reductionCounters()
                  .same_row_cross_bg_merges, 1U);
}

TEST(CSCM7BEndToEndTest,
     RepeatedSameRowPhysicalOutputsCompleteEndToEnd)
{
    const auto matrix = makeCSC(
        2, 3, {{0, 0, 1.0F}, {1, 1, 2.0F},
               {0, 2, 3.0F}});
    EndToEndHarness run(matrix, {1.0F, 1.0F, 1.0F},
                        MappingPolicy::External, {0, 0, 0},
                        pathConfig(), 1);
    run.finish();
    EXPECT_EQ(bits(run.execution.finalResult()[0]), bits(4.0F));
    EXPECT_GT(run.execution.partialResultPath()
                  .reductionCounters()
                  .same_row_repeated_record_merges, 0U);
}

TEST(CSCM7BEndToEndTest, CapacityPressureRecoversEndToEnd)
{
    auto config = pathConfig();
    config.pending_capacity_bursts_per_bg = 1;
    config.max_inflight_writes_per_bg = 1;
    config.write_latency_cycles = 5;
    config.host_return_queue_capacity_bursts = 1;
    config.reduction_queue_capacity_records = 4;
    config.host_reduce_records_per_cycle = 1;
    config.host_reduce_latency_cycles = 5;
    std::vector<COOEntry> entries;
    for (uint32_t row = 0; row < 12; ++row)
        entries.push_back({row, 0, float(row + 1)});
    EndToEndHarness run(makeCSC(12, 1, entries), {1.0F},
            MappingPolicy::External, {0}, config, 1);
    run.finish();
    const auto& path = run.execution.partialResultPath();
    EXPECT_GT(path.reductionCounters()
                  .reduction_queue_backpressure_cycles, 0U);
    expectResult(run.layout.matrix, {1.0F},
                 run.execution.finalResult());
}

TEST(CSCM7BEndToEndTest, ErrorNeverBecomesValidResult)
{
    EndToEndHarness run(makeCSC(1, 1, {{0, 0, 2.0F}}), {4.0F});
    run.untilReduction();
    CSCM7BEndToEndTestAccess::
        corruptAcceptedContributionCount(run.execution);
    run.execution.tick();
    EXPECT_TRUE(run.execution.hasFailed());
    EXPECT_FALSE(run.execution.endToEndSpMVComplete());
    EXPECT_FALSE(run.execution.resultValid());
    EXPECT_THROW(run.execution.finalResult(), std::logic_error);
}

TEST(CSCM7BEndToEndTest, EndToEndExactCycle)
{
    EndToEndHarness run(makeCSC(1, 1, {{0, 0, 2.0F}}), {4.0F});
    run.finish();
    const auto counters = run.execution.counters();
    const auto& path = run.execution.partialResultPath();
    EXPECT_LT(counters.compute_submit_complete_cycle,
              counters.bga_drain_complete_cycle);
    EXPECT_LT(path.counters().last_write_complete_cycle,
              path.readbackCounters().first_read_issue_cycle);
    EXPECT_LT(path.readbackCounters().first_read_issue_cycle,
              path.readbackCounters().first_read_complete_cycle);
    EXPECT_LT(path.reductionCounters()
                  .first_returned_burst_accept_cycle,
              path.reductionCounters()
                  .first_reduction_batch_issue_cycle);
    EXPECT_LT(path.reductionCounters()
                  .first_reduction_batch_issue_cycle,
              path.reductionCounters().first_host_reduce_cycle);
    EXPECT_LT(path.reductionCounters()
                  .host_reduction_complete_cycle,
              counters.end_to_end_complete_cycle);
    EXPECT_EQ(counters.result_valid_cycle,
              counters.end_to_end_complete_cycle);
    EXPECT_EQ(counters.compute_submit_complete_cycle, 60U);
    EXPECT_EQ(counters.bga_drain_complete_cycle, 63U);
    EXPECT_EQ(path.counters().first_bga_output_accept_cycle, 62U);
    EXPECT_EQ(path.counters().first_write_issue_cycle, 64U);
    EXPECT_EQ(path.counters().last_write_complete_cycle, 66U);
    EXPECT_EQ(path.counters().partial_writeback_complete_cycle, 66U);
    EXPECT_EQ(path.readbackCounters().first_read_issue_cycle, 67U);
    EXPECT_EQ(path.readbackCounters().first_read_complete_cycle, 70U);
    EXPECT_EQ(path.reductionCounters()
                  .first_returned_burst_accept_cycle, 71U);
    EXPECT_EQ(path.reductionCounters()
                  .first_reduction_batch_issue_cycle, 72U);
    EXPECT_EQ(path.reductionCounters().first_host_reduce_cycle, 74U);
    EXPECT_EQ(path.reductionCounters()
                  .host_reduction_complete_cycle, 74U);
    EXPECT_EQ(counters.end_to_end_complete_cycle, 75U);
    RecordProperty("compute_submit_complete_cycle",
                   counters.compute_submit_complete_cycle);
    RecordProperty("bga_drain_complete_cycle",
                   counters.bga_drain_complete_cycle);
    RecordProperty("first_output_accept_cycle",
                   path.counters().first_bga_output_accept_cycle);
    RecordProperty("first_write_issue_cycle",
                   path.counters().first_write_issue_cycle);
    RecordProperty("last_write_complete_cycle",
                   path.counters().last_write_complete_cycle);
    RecordProperty("partial_writeback_complete_cycle",
                   path.counters().partial_writeback_complete_cycle);
    RecordProperty("first_read_issue_cycle",
                   path.readbackCounters().first_read_issue_cycle);
    RecordProperty("first_read_complete_cycle",
                   path.readbackCounters().first_read_complete_cycle);
    RecordProperty("first_return_accept_cycle",
                   path.reductionCounters()
                       .first_returned_burst_accept_cycle);
    RecordProperty("first_reduction_issue_cycle",
                   path.reductionCounters()
                       .first_reduction_batch_issue_cycle);
    RecordProperty("first_host_reduce_cycle",
                   path.reductionCounters().first_host_reduce_cycle);
    RecordProperty("host_reduction_complete_cycle",
                   path.reductionCounters()
                       .host_reduction_complete_cycle);
    RecordProperty("end_to_end_complete_cycle",
                   counters.end_to_end_complete_cycle);
}

TEST(CSCM7BEndToEndTest, EndToEndTimingCountersConsistent)
{
    EndToEndHarness run(makeCSC(1, 1, {{0, 0, 2.0F}}), {4.0F});
    run.finish();
    const auto counters = run.execution.counters();
    EXPECT_EQ(counters.t_end_to_end,
              counters.end_to_end_complete_cycle -
                  counters.execution_start_cycle);
    EXPECT_GT(counters.t_writeback, 0U);
    EXPECT_GT(counters.t_readback, 0U);
    EXPECT_GT(counters.t_host_reduce, 0U);
}

TEST(CSCM7BEndToEndTest,
     ExternalToyCompletesSparsePIMCompatiblePath)
{
    const char* image = std::getenv("CSC_EXTERNAL_IMAGE");
    if (!image || !*image)
        GTEST_SKIP() << "set CSC_EXTERNAL_IMAGE";
    auto loaded = loadExternalPhysicalImage(image);
    auto& layout = loaded.layout;
    ASSERT_EQ(layout.stats.nnz, 9U);
    std::vector<float> x(layout.matrix.cols);
    for (uint32_t i = 0; i < x.size(); ++i)
        x[i] = float((i % 13) + 1) / 7;
    Views views(layout, x);
    CSCNativeExecution execution;
    execution.configurePartialResultWriteback(pathConfig());
    execution.enableProductionBGAIntegration(
        bgaConfig(layout.matrix.rows));
    execution.launch(views.views, layout.matrix.rows,
                     layout.stats.nnz);
    for (uint64_t guard = 0;
         guard < 500000 && !execution.isTerminal(); ++guard)
        execution.tick();
    ASSERT_TRUE(execution.endToEndSpMVComplete())
        << execution.errorMessage();
    ASSERT_TRUE(execution.resultValid());
    const auto& path = execution.partialResultPath();
    EXPECT_EQ(execution.counters().accepted_bga_partials, 9U);
    EXPECT_EQ(execution.counters()
                  .bga_output_contributions_accepted, 9U);
    EXPECT_EQ(path.counters().partial_record_contribution_sum, 9U);
    EXPECT_EQ(path.readbackCounters().readback_contribution_sum, 9U);
    EXPECT_EQ(path.reductionCounters()
                  .reduced_contribution_count, 9U);
    EXPECT_EQ(path.counters().partial_records_generated,
              path.reductionCounters().reduced_partial_records);
    expectResult(layout.matrix, x, execution.finalResult());
    RecordProperty("toy_physical_records",
                   path.counters().partial_records_generated);
    RecordProperty("toy_full_bursts",
                   path.counters().full_writeback_bursts);
    RecordProperty("toy_tail_bursts",
                   path.counters().tail_writeback_bursts);
    RecordProperty("toy_transferred_bytes",
                   path.counters().writeback_transferred_bytes);
    RecordProperty("toy_useful_bytes",
                   path.counters().writeback_useful_bytes);
    RecordProperty("toy_padding_bytes",
                   path.counters().writeback_padding_bytes);
    RecordProperty("toy_end_to_end_cycle",
                   execution.counters().end_to_end_complete_cycle);
}
