#include "csc/CSCDescriptorEngine.h"
#include "csc/CSCPartialResultPath.h"
#include "tests/csc/CSCLayout.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

using namespace csc_descriptor;

namespace {

uint32_t fp32Bits(float value)
{
    uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

void hashU64(uint64_t& hash, uint64_t value)
{
    for (uint32_t byte = 0; byte < 8; ++byte) {
        hash ^= (value >> (byte * 8)) & 0xffU;
        hash *= 0x100000001b3ULL;
    }
}

struct Views {
    std::array<std::vector<float>, kCSCGlobalBGs> packed_x;
    std::array<CSCBGImageView, kCSCGlobalBGs> views;

    Views(const CSCLayout& layout, const std::vector<float>& x)
    {
        for (uint32_t bg = 0; bg < kCSCGlobalBGs; ++bg) {
            for (const uint32_t column :
                 layout.bg[bg].x_slot_to_original_col)
                packed_x[bg].push_back(x[column]);
            views[bg] = {&layout.bg[bg].values,
                         &layout.bg[bg].row_indices,
                         &layout.bg[bg].descriptors,
                         &packed_x[bg]};
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

CSCBGAIntegrationConfig bgaConfig(uint32_t rows)
{
    CSCBGAIntegrationConfig config;
    config.enabled = true;
    config.accumulator.rows = rows;
    config.accumulator.accumulator_entries = 4;
    config.accumulator.compare_width = 4;
    config.accumulator.output_queue_depth = 4;
    config.output_consumer_mode =
        CSCBGAOutputConsumerMode::PARTIAL_RESULT_WRITEBACK;
    return config;
}

CSCMatrix goldenMatrix()
{
    // Column NNZ counts are 0, 1, 7, 8, and 9. Row 1 is repeated both
    // within BG 0 and across BGs 0 and 5; column 4 contains a duplicate row.
    return makeCSC(12, 5,
        {{1, 1, 1.0F},
         {0, 2, 2.0F}, {1, 2, 3.0F}, {2, 2, 4.0F},
         {3, 2, 5.0F}, {4, 2, 6.0F}, {5, 2, 7.0F},
         {6, 2, 8.0F},
         {1, 3, 9.0F}, {2, 3, 10.0F}, {3, 3, 11.0F},
         {4, 3, 12.0F}, {5, 3, 13.0F}, {6, 3, 14.0F},
         {7, 3, 15.0F}, {8, 3, 16.0F},
         {1, 4, 17.0F}, {1, 4, 18.0F}, {2, 4, 19.0F},
         {3, 4, 20.0F}, {4, 4, 21.0F}, {5, 4, 22.0F},
         {6, 4, 23.0F}, {7, 4, 24.0F}, {8, 4, 25.0F}});
}

}  // namespace

TEST(CSCFP32GoldenBaselineTest, M7ArchitecturalBehaviorAndAccounting)
{
    const CSCMatrix matrix = goldenMatrix();
    const std::vector<uint32_t> owners = {0, 0, 0, 0, 5};
    const std::vector<float> x(matrix.cols, 1.0F);
    const auto layout =
        buildLayout(matrix, MappingPolicy::External, owners);
    Views views(layout, x);

    CSCNativeExecution execution;
    execution.configurePartialResultWriteback(pathConfig());
    execution.enableProductionBGAIntegration(bgaConfig(matrix.rows));
    execution.launch(views.views, matrix.rows, matrix.values.size());
    for (uint64_t guard = 0;
         guard < 500000 && !execution.isTerminal(); ++guard)
        execution.tick();

    ASSERT_TRUE(execution.endToEndSpMVComplete())
        << execution.errorMessage();
    ASSERT_TRUE(execution.resultValid());

    const auto counters = execution.counters();
    const auto& path = execution.partialResultPath();
    const auto& write = path.counters();
    const auto& read = path.readbackCounters();
    const auto& reduce = path.reductionCounters();

    uint64_t bga_merges = 0;
    uint64_t bga_inserts = 0;
    uint64_t bga_evictions = 0;
    uint64_t bga_final_drains = 0;
    uint64_t descriptor_nnz_sum = 0;
    for (uint32_t bg = 0; bg < kCSCGlobalBGs; ++bg) {
        for (const auto& descriptor : layout.bg[bg].descriptors)
            descriptor_nnz_sum += descriptor.nnz_count;
        const auto& bga = execution.bankGroupAccumulator(bg).counters();
        bga_merges += bga.fp32_merges;
        bga_inserts += bga.inserts;
        bga_evictions += bga.capacity_evictions;
        bga_final_drains += bga.final_drain_outputs;
    }

    EXPECT_EQ(layout.stats.descriptor_count, 4U);
    EXPECT_EQ(descriptor_nnz_sum, 25U);
    EXPECT_EQ(matrix.values.size(), 25U);
    EXPECT_EQ(counters.descriptor_fetches, 4U);
    EXPECT_EQ(counters.x_scalar_loads, 4U);
    EXPECT_EQ(counters.value_transactions, 5U);
    EXPECT_EQ(counters.index_transactions, 5U);
    EXPECT_EQ(counters.logical_mul_events, 5U);
    EXPECT_EQ(counters.total_chunks, 5U);
    EXPECT_EQ(counters.full_chunks, 2U);
    EXPECT_EQ(counters.tail_chunks, 3U);
    EXPECT_EQ(counters.active_lanes, 25U);
    EXPECT_EQ(counters.generated_partials, 25U);
    EXPECT_EQ(counters.accepted_bga_partials, 25U);
    EXPECT_EQ(counters.invalid_lane_multiplies, 0U);
    EXPECT_EQ(counters.invalid_lane_writes, 0U);

    EXPECT_EQ(counters.bga_outputs_accepted,
              write.partial_records_generated);
    EXPECT_EQ(write.partial_records_generated, read.readback_records);
    EXPECT_EQ(read.readback_records, reduce.reduced_partial_records);
    EXPECT_EQ(counters.bga_output_contributions_accepted, 25U);
    EXPECT_EQ(write.partial_record_contribution_sum, 25U);
    EXPECT_EQ(read.readback_contribution_sum, 25U);
    EXPECT_EQ(reduce.reduced_contribution_count, 25U);
    EXPECT_EQ(bga_merges + counters.bga_outputs_accepted, 25U);
    EXPECT_EQ(bga_evictions,
              counters.bga_capacity_outputs_accepted);
    EXPECT_EQ(bga_final_drains,
              counters.bga_final_drain_outputs_accepted);
    EXPECT_EQ(bga_merges, 2U);
    EXPECT_EQ(bga_inserts, 23U);
    EXPECT_EQ(bga_evictions, 15U);
    EXPECT_EQ(bga_final_drains, 8U);
    EXPECT_EQ(counters.bga_outputs_accepted, 23U);
    EXPECT_EQ(write.full_writeback_bursts, 5U);
    EXPECT_EQ(write.tail_writeback_bursts, 1U);
    EXPECT_EQ(write.total_writeback_bursts, 6U);
    EXPECT_EQ(write.writeback_useful_bytes, 184U);
    EXPECT_EQ(write.writeback_transferred_bytes, 192U);
    EXPECT_EQ(write.writeback_padding_bytes, 8U);
    EXPECT_EQ(read.read_requests_issued, 6U);
    EXPECT_EQ(read.read_requests_completed, 6U);
    EXPECT_EQ(reduce.fp32_host_add_count, 23U);
    EXPECT_EQ(counters.compute_submit_complete_cycle, 178U);
    EXPECT_EQ(counters.bga_drain_complete_cycle, 206U);
    EXPECT_EQ(write.partial_writeback_complete_cycle, 209U);
    EXPECT_EQ(read.read_transport_complete_cycle, 242U);
    EXPECT_EQ(reduce.host_reduction_complete_cycle, 261U);
    EXPECT_EQ(counters.end_to_end_complete_cycle, 262U);

    const std::array<uint32_t, 12> expected_y_bits = {
        0x40000000U, 0x42400000U, 0x42040000U, 0x42100000U,
        0x421c0000U, 0x42280000U, 0x42340000U, 0x421c0000U,
        0x42240000U, 0x00000000U, 0x00000000U, 0x00000000U};
    ASSERT_EQ(execution.finalResult().size(), expected_y_bits.size());
    for (uint32_t row = 0; row < expected_y_bits.size(); ++row)
        EXPECT_EQ(fp32Bits(execution.finalResult()[row]),
                  expected_y_bits[row]) << "row=" << row;

    std::cout << "FP32_GOLDEN_METRICS"
              << " bga_merges=" << bga_merges
              << " bga_inserts=" << bga_inserts
              << " bga_evictions=" << bga_evictions
              << " bga_final_drains=" << bga_final_drains
              << " bga_outputs=" << counters.bga_outputs_accepted
              << " write_bursts=" << write.total_writeback_bursts
              << " transferred_bytes=" << write.writeback_transferred_bytes
              << " compute_cycle=" << counters.compute_submit_complete_cycle
              << " bga_cycle=" << counters.bga_drain_complete_cycle
              << " write_cycle=" << write.partial_writeback_complete_cycle
              << " read_cycle=" << read.read_transport_complete_cycle
              << " reduce_cycle=" << reduce.host_reduction_complete_cycle
              << " end_cycle=" << counters.end_to_end_complete_cycle
              << '\n';
}

TEST(CSCFP32GoldenBaselineTest, BGAOutputEventTraceHash)
{
    const CSCMatrix matrix = goldenMatrix();
    const std::vector<uint32_t> owners = {0, 0, 0, 0, 5};
    const std::vector<float> x(matrix.cols, 1.0F);
    const auto layout =
        buildLayout(matrix, MappingPolicy::External, owners);
    Views views(layout, x);

    auto config = bgaConfig(matrix.rows);
    config.output_consumer_mode = CSCBGAOutputConsumerMode::EXTERNAL;
    CSCNativeExecution execution;
    execution.enableProductionBGAIntegration(config);
    execution.launch(views.views, matrix.rows, matrix.values.size());

    uint64_t trace_hash = 0xcbf29ce484222325ULL;
    uint64_t trace_records = 0;
    for (uint64_t guard = 0;
         guard < 500000 && !execution.bgaExecutionComplete(); ++guard) {
        execution.tick();
        for (uint32_t bg = 0; bg < kCSCGlobalBGs; ++bg) {
            if (!execution.hasBGAOutput(bg)) continue;
            const auto output = execution.peekBGAOutput(bg).payload;
            hashU64(trace_hash, bg);
            hashU64(trace_hash, output.output_sequence);
            hashU64(trace_hash, output.row_idx);
            hashU64(trace_hash, fp32Bits(output.value));
            hashU64(trace_hash, output.contribution_count);
            hashU64(trace_hash,
                    output.reason == CSCBGAOutputReason::CAPACITY_EVICTION
                        ? 0U : 1U);
            ++trace_records;
            execution.acceptBGAOutput(bg);
        }
    }

    ASSERT_FALSE(execution.hasFailed()) << execution.errorMessage();
    ASSERT_TRUE(execution.bgaExecutionComplete());
    EXPECT_EQ(trace_records, 23U);
    EXPECT_EQ(trace_hash, 0xf3459a33998d8acdULL);
    std::cout << "FP32_GOLDEN_BGA_TRACE records=" << trace_records
              << " fnv1a64=0x" << std::hex << trace_hash << std::dec
              << '\n';
}
