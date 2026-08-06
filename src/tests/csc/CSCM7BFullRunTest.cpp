#include "csc/CSCDescriptorEngine.h"
#include "csc/CSCPartialResultPath.h"
#include "tests/csc/CSCExternalImage.h"
#include "tests/csc/CSCFunctionalModel.h"
#include "tests/csc/CSCLayout.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace csc_descriptor;

namespace {

uint32_t envUint32(const char* name, uint32_t fallback)
{
    const char* text = std::getenv(name);
    if (!text || !*text) return fallback;

    std::size_t consumed = 0;
    const unsigned long value = std::stoul(text, &consumed, 10);
    if (consumed != std::strlen(text) || value == 0 ||
        value > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument(std::string(name) +
                                    " must be an integer in [1, UINT32_MAX]");
    return static_cast<uint32_t>(value);
}

struct Views {
    std::array<std::vector<float>, kCSCGlobalBGs> x;
    std::array<CSCBGImageView, kCSCGlobalBGs> views;

    Views(const CSCLayout& layout, const std::vector<float>& original_x)
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

CSCPartialResultPathConfig fullRunPathConfig(
    uint32_t resident_capacity_bursts_per_bg = 16)
{
    CSCPartialResultPathConfig config;
    config.buffer_capacity_bursts_per_bg =
        resident_capacity_bursts_per_bg;
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

CSCBGAIntegrationConfig fullRunBGAConfig(uint32_t rows)
{
    CSCBGAIntegrationConfig config;
    const uint32_t accumulator_entries =
        envUint32("CSC_BGA_ACCUMULATOR_ENTRIES", 64);
    config.enabled = true;
    config.accumulator.rows = rows;
    config.accumulator.accumulator_entries = accumulator_entries;
    // The current fully-associative BGA model requires comparison against
    // every resident accumulator entry.
    config.accumulator.compare_width = accumulator_entries;
    config.accumulator.output_queue_depth =
        envUint32("CSC_BGA_OUTPUT_QUEUE_DEPTH", 64);
    config.output_consumer_mode =
        CSCBGAOutputConsumerMode::PARTIAL_RESULT_WRITEBACK;
    config.partial_writeback_accepts_per_cycle =
        envUint32("CSC_BGA_WRITEBACK_ACCEPTS_PER_CYCLE", 1);
    return config;
}

CSCPartialResultPathConfig transportOnlyPathConfig(
    uint32_t resident_capacity_bursts_per_bg)
{
    auto config = fullRunPathConfig(resident_capacity_bursts_per_bg);
    config.max_inflight_reads_per_channel = 4;
    config.host_return_queue_capacity_bursts = 64;
    config.host_reduction_enabled = false;
    return config;
}

uint32_t floatBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

const char* passFail(bool pass)
{
    return pass ? "PASS" : "FAIL";
}

void printEvent(uint64_t cycle, const char* name)
{
    std::cout << "cycle " << std::setw(4) << cycle << "  " << name
              << '\n';
}

}  // namespace

TEST(CSCM7BFullRunTest, ExternalToyPrintsFullCycleBreakdown)
{
    const char* image_path = std::getenv("CSC_EXTERNAL_IMAGE");
    if (!image_path || !*image_path)
        GTEST_SKIP() << "set CSC_EXTERNAL_IMAGE";

    auto loaded = loadExternalPhysicalImage(image_path);
    const auto& layout = loaded.layout;
    std::vector<float> x(layout.matrix.cols);
    for (uint32_t i = 0; i < x.size(); ++i)
        x[i] = float((i % 13) + 1) / 7;
    const auto reference = cpuReference(layout.matrix, x);

    Views views(layout, x);
    CSCNativeExecution execution;
    execution.configurePartialResultWriteback(fullRunPathConfig());
    execution.enableProductionBGAIntegration(
        fullRunBGAConfig(layout.matrix.rows));
    execution.launch(views.views, layout.matrix.rows, layout.stats.nnz);

    for (uint64_t guard = 0;
         guard < 500000 && !execution.isTerminal(); ++guard)
        execution.tick();

    ASSERT_FALSE(execution.hasFailed()) << execution.errorMessage();
    ASSERT_TRUE(execution.endToEndSpMVComplete());
    ASSERT_TRUE(execution.resultValid());

    const auto& result = execution.finalResult();
    const auto execution_counters = execution.counters();
    const auto& path = execution.partialResultPath();
    const auto& write = path.counters();
    const auto& read = path.readbackCounters();
    const auto& reduce = path.reductionCounters();

    const uint64_t active_bg_count = std::count_if(
        layout.stats.bg_nnz.begin(), layout.stats.bg_nnz.end(),
        [](uint64_t count) { return count != 0; });
    uint64_t min_active_nnz = std::numeric_limits<uint64_t>::max();
    uint64_t max_active_nnz = 0;
    uint64_t min_active_descriptors =
        std::numeric_limits<uint64_t>::max();
    uint64_t max_active_descriptors = 0;
    for (uint32_t bg = 0; bg < layout.stats.bg_nnz.size(); ++bg) {
        if (!layout.stats.bg_nnz[bg]) continue;
        min_active_nnz =
            std::min(min_active_nnz, layout.stats.bg_nnz[bg]);
        max_active_nnz =
            std::max(max_active_nnz, layout.stats.bg_nnz[bg]);
        min_active_descriptors = std::min(
            min_active_descriptors,
            layout.stats.bg_descriptor_count[bg]);
        max_active_descriptors = std::max(
            max_active_descriptors,
            layout.stats.bg_descriptor_count[bg]);
    }

    std::cout << "\n=== MATRIX ===\n"
              << "rows: " << layout.matrix.rows << '\n'
              << "cols: " << layout.matrix.cols << '\n'
              << "nnz: " << layout.stats.nnz << '\n'
              << "descriptor_count: "
              << layout.stats.descriptor_count << '\n'
              << "global_bg_count: "
              << layout.stats.num_global_bg << '\n'
              << "active_bg_count: " << active_bg_count << '\n'
              << "active_bg_descriptor_min/max/avg: "
              << min_active_descriptors << '/' << max_active_descriptors
              << '/' << std::fixed << std::setprecision(3)
              << double(layout.stats.descriptor_count) /
                     double(active_bg_count)
              << '\n'
              << "active_bg_nnz_min/max/avg: " << min_active_nnz << '/'
              << max_active_nnz << '/'
              << double(layout.stats.nnz) / double(active_bg_count)
              << '\n';

    std::cout << "\n=== COMPUTE / BGA ===\n"
              << "execution_start_cycle: "
              << execution_counters.execution_start_cycle << '\n'
              << "compute_submit_complete_cycle: "
              << execution_counters.compute_submit_complete_cycle << '\n'
              << "bga_drain_complete_cycle: "
              << execution_counters.bga_drain_complete_cycle << '\n'
              << "accepted_nnz/contributions: "
              << execution_counters.accepted_bga_partials << '\n'
              << "bga_physical_output_records: "
              << execution_counters.bga_outputs_accepted << '\n'
              << "bga_output_contribution_sum: "
              << execution_counters.bga_output_contributions_accepted
              << '\n'
              << "capacity_eviction_outputs: "
              << execution_counters.bga_capacity_outputs_accepted << '\n'
              << "final_drain_outputs: "
              << execution_counters.bga_final_drain_outputs_accepted
              << '\n'
              << "bga_backpressure_cycles: "
              << execution_counters.bga_backpressure_cycles << '\n';

    std::cout << "\n=== PARTIAL WRITEBACK ===\n"
              << "first_bga_output_accept_cycle: "
              << write.first_bga_output_accept_cycle << '\n'
              << "first_write_issue_cycle: "
              << write.first_write_issue_cycle << '\n'
              << "first_write_complete_cycle: "
              << write.first_write_complete_cycle << '\n'
              << "last_write_issue_cycle: "
              << write.last_write_issue_cycle << '\n'
              << "last_write_complete_cycle: "
              << write.last_write_complete_cycle << '\n'
              << "partial_writeback_complete_cycle: "
              << write.partial_writeback_complete_cycle << '\n'
              << "partial_records_generated: "
              << write.partial_records_generated << '\n'
              << "full_writeback_bursts: "
              << write.full_writeback_bursts << '\n'
              << "tail_writeback_bursts: "
              << write.tail_writeback_bursts << '\n'
              << "total_writeback_bursts: "
              << write.total_writeback_bursts << '\n'
              << "writeback_useful_bytes: "
              << write.writeback_useful_bytes << '\n'
              << "writeback_transferred_bytes: "
              << write.writeback_transferred_bytes << '\n'
              << "writeback_padding_bytes: "
              << write.writeback_padding_bytes << '\n'
              << "write_requests_issued/completed: "
              << write.write_requests_issued << '/'
              << write.write_requests_completed << '\n'
              << "peak_inflight_writes: "
              << write.peak_inflight_writes << '\n'
              << "packer_backpressure_cycles: "
              << write.packer_backpressure_cycles << '\n'
              << "buffer_backpressure_cycles: "
              << write.buffer_backpressure_cycles << '\n'
              << "inflight_backpressure_cycles: "
              << write.inflight_backpressure_cycles << '\n';

    std::cout << "\n=== HOST READBACK ===\n"
              << "host_readback_start_cycle: "
              << read.host_readback_start_cycle << '\n'
              << "first_read_issue_cycle: "
              << read.first_read_issue_cycle << '\n'
              << "first_read_complete_cycle: "
              << read.first_read_complete_cycle << '\n'
              << "last_read_issue_cycle: "
              << read.last_read_issue_cycle << '\n'
              << "last_read_complete_cycle: "
              << read.last_read_complete_cycle << '\n'
              << "host_readback_complete_cycle: "
              << reduce.host_readback_complete_cycle << '\n'
              << "read_requests_issued/completed: "
              << read.read_requests_issued << '/'
              << read.read_requests_completed << '\n'
              << "peak_inflight_reads: "
              << read.peak_inflight_reads << '\n'
              << "returned_bursts_committed/delivered: "
              << read.returned_bursts_committed << '/'
              << read.returned_bursts_delivered << '\n'
              << "peak_host_return_queue_occupancy: "
              << read.peak_host_return_queue_occupancy << '\n'
              << "return_queue_backpressure_cycles: "
              << read.return_queue_backpressure_cycles << '\n'
              << "readback_useful_bytes: "
              << read.readback_useful_bytes << '\n'
              << "readback_transferred_bytes: "
              << read.readback_transferred_bytes << '\n'
              << "readback_padding_bytes: "
              << read.readback_padding_bytes << '\n';

    std::cout << "\n=== HOST REDUCTION ===\n"
              << "first_host_reduce_issue_cycle: "
              << reduce.first_reduction_batch_issue_cycle << '\n'
              << "first_host_reduce_cycle: "
              << reduce.first_host_reduce_cycle << '\n'
              << "last_host_reduce_cycle: "
              << reduce.last_host_reduce_cycle << '\n'
              << "host_reduction_complete_cycle: "
              << reduce.host_reduction_complete_cycle << '\n'
              << "reduced_partial_records: "
              << reduce.reduced_partial_records << '\n'
              << "reduced_contribution_count: "
              << reduce.reduced_contribution_count << '\n'
              << "fp32_host_add_count: "
              << reduce.fp32_host_add_count << '\n'
              << "same_row_cross_bg_merges: "
              << reduce.same_row_cross_bg_merges << '\n'
              << "same_row_repeated_record_merges: "
              << reduce.same_row_repeated_record_merges << '\n'
              << "peak_reduction_queue_occupancy: "
              << reduce.peak_reduction_queue_occupancy << '\n'
              << "reduction_queue_backpressure_cycles: "
              << reduce.reduction_queue_backpressure_cycles << '\n';

    std::cout << "\n=== END TO END ===\n"
              << "end_to_end_complete_cycle: "
              << execution_counters.end_to_end_complete_cycle << '\n'
              << "result_valid_cycle: "
              << execution_counters.result_valid_cycle << '\n'
              << "T_compute_submit: "
              << execution_counters.t_compute_submit << '\n'
              << "T_bga_drain: " << execution_counters.t_bga_drain
              << '\n'
              << "T_writeback: " << execution_counters.t_writeback
              << '\n'
              << "T_readback: " << execution_counters.t_readback
              << '\n'
              << "T_host_reduce: "
              << execution_counters.t_host_reduce << '\n'
              << "T_end_to_end: "
              << execution_counters.t_end_to_end << '\n';

    const bool contribution_conservation =
        layout.stats.nnz == execution_counters.accepted_bga_partials &&
        layout.stats.nnz ==
            execution_counters.bga_output_contributions_accepted &&
        layout.stats.nnz == write.partial_record_contribution_sum &&
        layout.stats.nnz == read.readback_contribution_sum &&
        layout.stats.nnz == reduce.reduced_contribution_count;
    const bool record_conservation =
        execution_counters.bga_outputs_accepted ==
            write.partial_records_generated &&
        write.partial_records_generated == read.readback_records &&
        read.readback_records == reduce.reduced_partial_records;
    const bool byte_conservation =
        write.writeback_useful_bytes == read.readback_useful_bytes &&
        write.writeback_transferred_bytes ==
            read.readback_transferred_bytes &&
        write.writeback_padding_bytes == read.readback_padding_bytes;

    std::cout << "\n=== CONSERVATION ===\n"
              << "manifest_nnz: " << layout.stats.nnz << '\n'
              << "engine_accepted_contributions: "
              << execution_counters.accepted_bga_partials << '\n'
              << "bga_output_contributions: "
              << execution_counters.bga_output_contributions_accepted
              << '\n'
              << "writeback_contributions: "
              << write.partial_record_contribution_sum << '\n'
              << "readback_contributions: "
              << read.readback_contribution_sum << '\n'
              << "reduced_contributions: "
              << reduce.reduced_contribution_count << '\n'
              << "physical_bga_output_records: "
              << execution_counters.bga_outputs_accepted << '\n'
              << "packed_records: " << write.partial_records_generated
              << '\n'
              << "readback_records: " << read.readback_records << '\n'
              << "reduced_records: "
              << reduce.reduced_partial_records << '\n'
              << "contribution_conservation: "
              << passFail(contribution_conservation) << '\n'
              << "record_conservation: "
              << passFail(record_conservation) << '\n'
              << "byte_conservation: "
              << passFail(byte_conservation) << '\n';

    bool bit_exact = result.size() == reference.size();
    float max_error = 0.0F;
    uint32_t first_mismatch = UINT32_MAX;
    const bool reference_match = compareResults(
        reference, result, &max_error, &first_mismatch);
    std::cout << "\n=== FINAL Y ===\n"
              << "row cpu_reference m7b_final absolute_error bit_exact\n";
    ASSERT_EQ(result.size(), reference.size());
    for (uint32_t row = 0; row < result.size(); ++row) {
        const bool row_bit_exact =
            floatBits(reference[row]) == floatBits(result[row]);
        bit_exact = bit_exact && row_bit_exact;
        std::cout << row << ' ' << std::setprecision(9)
                  << reference[row] << ' ' << result[row] << ' '
                  << std::fabs(reference[row] - result[row]) << ' '
                  << passFail(row_bit_exact) << '\n';
    }
    std::cout << "resultValid: " << passFail(execution.resultValid())
              << '\n'
              << "endToEndSpMVComplete: "
              << passFail(execution.endToEndSpMVComplete()) << '\n'
              << "CPU_reference_comparison: "
              << passFail(reference_match) << '\n'
              << "all_rows_bit_exact: " << passFail(bit_exact) << '\n'
              << "max_absolute_error: " << max_error << '\n';

    std::cout << "\n=== TIMELINE ===\n";
    printEvent(execution_counters.execution_start_cycle,
               "execution start");
    printEvent(execution_counters.compute_submit_complete_cycle,
               "compute submit complete");
    printEvent(execution_counters.first_bga_output_cycle,
               "first BGA output");
    printEvent(write.first_bga_output_accept_cycle,
               "first writeback consumer accept");
    printEvent(execution_counters.bga_drain_complete_cycle,
               "BGA drain complete");
    printEvent(write.first_write_issue_cycle, "first write issue");
    printEvent(write.last_write_complete_cycle,
               "last write complete");
    printEvent(read.first_read_issue_cycle, "first read issue");
    printEvent(read.last_read_complete_cycle, "last read complete");
    printEvent(reduce.first_host_reduce_cycle,
               "first FP32 reduction");
    printEvent(reduce.last_host_reduce_cycle,
               "last FP32 reduction");
    printEvent(execution_counters.result_valid_cycle, "resultValid");

    std::cout << "\n=== PER-BG SUMMARY ===\n"
              << "BG input_nnz descriptors\n";
    for (uint32_t bg = 0; bg < layout.stats.bg_nnz.size(); ++bg) {
        if (!layout.stats.bg_nnz[bg]) continue;
        std::cout << bg << ' ' << layout.stats.bg_nnz[bg] << ' '
                  << layout.stats.bg_descriptor_count[bg] << '\n';
    }
    std::cout << "=== M7B FULL RUN COMPLETE ===\n";

    EXPECT_TRUE(contribution_conservation);
    EXPECT_TRUE(record_conservation);
    EXPECT_TRUE(byte_conservation);
    EXPECT_TRUE(reference_match);
    EXPECT_EQ(layout.stats.nnz, 9U);
}

TEST(CSCM7BFullRunTest, ExternalMatrixPrintsFullCycleBreakdown)
{
    const char* image_path = std::getenv("CSC_EXTERNAL_IMAGE");
    if (!image_path || !*image_path)
        GTEST_SKIP() << "set CSC_EXTERNAL_IMAGE";

    auto loaded = loadExternalPhysicalImage(image_path);
    const auto& layout = loaded.layout;
    const uint64_t max_bg_nnz = *std::max_element(
        layout.stats.bg_nnz.begin(), layout.stats.bg_nnz.end());
    const uint32_t resident_capacity = static_cast<uint32_t>(
        (max_bg_nnz + kCSCPartialRecordsPerBurst - 1) /
        kCSCPartialRecordsPerBurst);
    const uint64_t resident_entries =
        uint64_t(resident_capacity) * layout.stats.num_global_bg;
    const uint64_t resident_physical_bytes =
        resident_entries * kCSCPartialResultBurstBytes;
    const uint64_t resident_shadow_bytes =
        resident_entries * sizeof(CSCPartialResultBurst);

    std::vector<float> x(layout.matrix.cols);
    for (uint32_t i = 0; i < x.size(); ++i)
        x[i] = float((i % 13) + 1) / 7;
    const auto reference = cpuReference(layout.matrix, x);

    Views views(layout, x);
    const auto bga_config = fullRunBGAConfig(layout.matrix.rows);
    CSCNativeExecution execution;
    execution.configurePartialResultWriteback(
        fullRunPathConfig(resident_capacity));
    execution.enableProductionBGAIntegration(bga_config);
    execution.launch(views.views, layout.matrix.rows, layout.stats.nnz);

    constexpr uint64_t kProgressIntervalCycles = 100000;
    uint64_t next_progress_cycle = kProgressIntervalCycles;
    for (uint64_t guard = 0;
         guard < 2000000000ULL && !execution.isTerminal(); ++guard) {
        execution.tick();
        if (execution.cycle() < next_progress_cycle) continue;
        const auto progress = execution.counters();
        const auto& progress_path = execution.partialResultPath();
        std::cout << "M7B_PROGRESS cycle=" << execution.cycle()
                  << " accepted_contributions="
                  << progress.accepted_bga_partials
                  << " bga_outputs="
                  << progress.bga_outputs_accepted
                  << " writes_completed="
                  << progress_path.counters().write_requests_completed
                  << " reads_completed="
                  << progress_path.readbackCounters()
                         .read_requests_completed
                  << " reduced_records="
                  << progress_path.reductionCounters()
                         .reduced_partial_records
                  << '\n'
                  << std::flush;
        next_progress_cycle += kProgressIntervalCycles;
    }

    ASSERT_FALSE(execution.hasFailed()) << execution.errorMessage();
    ASSERT_TRUE(execution.isTerminal());
    ASSERT_TRUE(execution.endToEndSpMVComplete());
    ASSERT_TRUE(execution.resultValid());

    const auto counters = execution.counters();
    const auto& path = execution.partialResultPath();
    const auto& write = path.counters();
    const auto& read = path.readbackCounters();
    const auto& reduce = path.reductionCounters();
    const auto& result = execution.finalResult();

    uint64_t active_bg_count = 0;
    uint64_t min_bg_nnz = std::numeric_limits<uint64_t>::max();
    uint64_t max_active_bg_nnz = 0;
    uint64_t min_bg_descriptors =
        std::numeric_limits<uint64_t>::max();
    uint64_t max_bg_descriptors = 0;
    uint32_t least_loaded_bg = 0;
    uint32_t most_loaded_bg = 0;
    for (uint32_t bg = 0; bg < layout.stats.bg_nnz.size(); ++bg) {
        if (!layout.stats.bg_nnz[bg]) continue;
        ++active_bg_count;
        if (layout.stats.bg_nnz[bg] < min_bg_nnz) {
            min_bg_nnz = layout.stats.bg_nnz[bg];
            least_loaded_bg = bg;
        }
        if (layout.stats.bg_nnz[bg] > max_active_bg_nnz) {
            max_active_bg_nnz = layout.stats.bg_nnz[bg];
            most_loaded_bg = bg;
        }
        min_bg_descriptors = std::min(
            min_bg_descriptors,
            layout.stats.bg_descriptor_count[bg]);
        max_bg_descriptors = std::max(
            max_bg_descriptors,
            layout.stats.bg_descriptor_count[bg]);
    }

    const double average_bg_nnz =
        double(layout.stats.nnz) / double(active_bg_count);
    const double average_bg_descriptors =
        double(layout.stats.descriptor_count) /
        double(active_bg_count);
    const double aggregation_ratio =
        double(counters.bga_outputs_accepted) /
        double(layout.stats.nnz);
    const double reduction_factor =
        double(layout.stats.nnz) /
        double(counters.bga_outputs_accepted);
    const double full_burst_ratio = write.total_writeback_bursts
        ? double(write.full_writeback_bursts) /
              double(write.total_writeback_bursts)
        : 0.0;
    const double packing_efficiency =
        write.writeback_transferred_bytes
        ? double(write.writeback_useful_bytes) /
              double(write.writeback_transferred_bytes)
        : 1.0;

    const bool contribution_conservation =
        layout.stats.nnz == counters.accepted_bga_partials &&
        layout.stats.nnz ==
            counters.bga_output_contributions_accepted &&
        layout.stats.nnz == write.partial_record_contribution_sum &&
        layout.stats.nnz == read.readback_contribution_sum &&
        layout.stats.nnz == reduce.reduced_contribution_count;
    const bool record_conservation =
        counters.bga_outputs_accepted ==
            write.partial_records_generated &&
        write.partial_records_generated == read.readback_records &&
        read.readback_records == reduce.reduced_partial_records;
    const bool byte_conservation =
        write.writeback_useful_bytes == read.readback_useful_bytes &&
        write.writeback_transferred_bytes ==
            read.readback_transferred_bytes &&
        write.writeback_padding_bytes == read.readback_padding_bytes;

    ASSERT_TRUE(contribution_conservation);
    ASSERT_TRUE(record_conservation);
    ASSERT_TRUE(byte_conservation);
    ASSERT_EQ(result.size(), reference.size());

    uint64_t nonzero_rows = 0;
    uint64_t nan_count = 0;
    uint64_t inf_count = 0;
    uint64_t mismatch_count = 0;
    double l1 = 0.0;
    double l2_squared = 0.0;
    double max_absolute_value = 0.0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    struct ErrorRow {
        double absolute_error = 0.0;
        uint32_t row = 0;
    };
    std::vector<ErrorRow> worst_rows;
    worst_rows.reserve(result.size());
    for (uint32_t row = 0; row < result.size(); ++row) {
        const float actual = result[row];
        const float expected = reference[row];
        if (std::isnan(actual)) {
            ++nan_count;
            if (!std::isnan(expected)) ++mismatch_count;
            continue;
        }
        if (std::isinf(actual)) {
            ++inf_count;
            if (actual != expected) ++mismatch_count;
            continue;
        }
        if (actual != 0.0F) ++nonzero_rows;
        const double magnitude = std::fabs(double(actual));
        l1 += magnitude;
        l2_squared += double(actual) * double(actual);
        max_absolute_value = std::max(max_absolute_value, magnitude);
        const double absolute_error =
            std::fabs(double(actual) - double(expected));
        const double relative_error =
            absolute_error /
            std::max(std::fabs(double(expected)), 1.0e-30);
        max_abs_error = std::max(max_abs_error, absolute_error);
        max_rel_error = std::max(max_rel_error, relative_error);
        if (absolute_error >
            kAbsoluteTolerance +
                kRelativeTolerance * std::fabs(double(expected)))
            ++mismatch_count;
        worst_rows.push_back({absolute_error, row});
    }
    std::partial_sort(
        worst_rows.begin(),
        worst_rows.begin() +
            std::min<size_t>(5, worst_rows.size()),
        worst_rows.end(),
        [](const ErrorRow& lhs, const ErrorRow& rhs) {
            if (lhs.absolute_error != rhs.absolute_error)
                return lhs.absolute_error > rhs.absolute_error;
            return lhs.row < rhs.row;
        });

    std::cout << std::fixed << std::setprecision(6)
              << "\n=== MATRIX / PREFLIGHT ===\n"
              << "rows: " << layout.matrix.rows << '\n'
              << "cols: " << layout.matrix.cols << '\n'
              << "nnz: " << layout.stats.nnz << '\n'
              << "descriptor_count: "
              << layout.stats.descriptor_count << '\n'
              << "global_bg_count: "
              << layout.stats.num_global_bg << '\n'
              << "active_bg_count: " << active_bg_count << '\n'
              << "descriptor_per_active_bg_min/max/avg: "
              << min_bg_descriptors << '/' << max_bg_descriptors << '/'
              << average_bg_descriptors << '\n'
              << "nnz_per_active_bg_min/max/avg: " << min_bg_nnz << '/'
              << max_active_bg_nnz << '/' << average_bg_nnz << '\n'
              << "least_loaded_bg: " << least_loaded_bg << '\n'
              << "most_loaded_bg: " << most_loaded_bg << '\n'
              << "max_to_average_imbalance: "
              << double(max_active_bg_nnz) / average_bg_nnz << '\n'
              << "resident_capacity_bursts_per_bg: "
              << resident_capacity << '\n'
              << "resident_physical_bound_bytes: "
              << resident_physical_bytes << '\n'
              << "resident_shadow_bound_bytes: "
              << resident_shadow_bytes << '\n'
              << "bga_accumulator_entries: "
              << bga_config.accumulator.accumulator_entries << '\n'
              << "bga_compare_width: "
              << bga_config.accumulator.compare_width << '\n'
              << "bga_output_queue_depth: "
              << bga_config.accumulator.output_queue_depth << '\n';

    std::cout << "\n=== COMPUTE / BGA ===\n"
              << "execution_start_cycle: "
              << counters.execution_start_cycle << '\n'
              << "compute_submit_complete_cycle: "
              << counters.compute_submit_complete_cycle << '\n'
              << "bga_drain_complete_cycle: "
              << counters.bga_drain_complete_cycle << '\n'
              << "manifest_nnz: " << layout.stats.nnz << '\n'
              << "engine_accepted_contributions: "
              << counters.accepted_bga_partials << '\n'
              << "physical_bga_output_records: "
              << counters.bga_outputs_accepted << '\n'
              << "bga_output_contribution_sum: "
              << counters.bga_output_contributions_accepted << '\n'
              << "capacity_eviction_outputs: "
              << counters.bga_capacity_outputs_accepted << '\n'
              << "final_drain_outputs: "
              << counters.bga_final_drain_outputs_accepted << '\n'
              << "bga_aggregation_ratio: " << aggregation_ratio << '\n'
              << "bga_reduction_factor: " << reduction_factor << '\n'
              << "bga_backpressure_cycles: "
              << counters.bga_backpressure_cycles << '\n';

    std::cout << "\n=== WRITEBACK ===\n"
              << "first_bga_output_accept_cycle: "
              << write.first_bga_output_accept_cycle << '\n'
              << "first_write_issue_cycle: "
              << write.first_write_issue_cycle << '\n'
              << "first_write_complete_cycle: "
              << write.first_write_complete_cycle << '\n'
              << "last_write_issue_cycle: "
              << write.last_write_issue_cycle << '\n'
              << "last_write_complete_cycle: "
              << write.last_write_complete_cycle << '\n'
              << "partial_writeback_complete_cycle: "
              << write.partial_writeback_complete_cycle << '\n'
              << "partial_records_generated: "
              << write.partial_records_generated << '\n'
              << "full_writeback_bursts: "
              << write.full_writeback_bursts << '\n'
              << "tail_writeback_bursts: "
              << write.tail_writeback_bursts << '\n'
              << "total_writeback_bursts: "
              << write.total_writeback_bursts << '\n'
              << "full_burst_ratio: " << full_burst_ratio << '\n'
              << "writeback_useful_bytes: "
              << write.writeback_useful_bytes << '\n'
              << "writeback_transferred_bytes: "
              << write.writeback_transferred_bytes << '\n'
              << "writeback_padding_bytes: "
              << write.writeback_padding_bytes << '\n'
              << "packing_efficiency: " << packing_efficiency << '\n'
              << "write_requests_issued/completed: "
              << write.write_requests_issued << '/'
              << write.write_requests_completed << '\n'
              << "peak_inflight_writes: "
              << write.peak_inflight_writes << '\n'
              << "packer/buffer/inflight_backpressure_cycles: "
              << write.packer_backpressure_cycles << '/'
              << write.buffer_backpressure_cycles << '/'
              << write.inflight_backpressure_cycles << '\n';

    std::cout << "\n=== READBACK ===\n"
              << "host_readback_start_cycle: "
              << read.host_readback_start_cycle << '\n'
              << "first_read_issue_cycle: "
              << read.first_read_issue_cycle << '\n'
              << "first_read_complete_cycle: "
              << read.first_read_complete_cycle << '\n'
              << "last_read_issue_cycle: "
              << read.last_read_issue_cycle << '\n'
              << "last_read_complete_cycle: "
              << read.last_read_complete_cycle << '\n'
              << "host_readback_complete_cycle: "
              << reduce.host_readback_complete_cycle << '\n'
              << "read_requests_issued/completed: "
              << read.read_requests_issued << '/'
              << read.read_requests_completed << '\n'
              << "peak_inflight_reads: "
              << read.peak_inflight_reads << '\n'
              << "peak_host_return_queue_occupancy: "
              << read.peak_host_return_queue_occupancy << '\n'
              << "return_queue_backpressure_cycles: "
              << read.return_queue_backpressure_cycles << '\n'
              << "readback_useful/transferred/padding_bytes: "
              << read.readback_useful_bytes << '/'
              << read.readback_transferred_bytes << '/'
              << read.readback_padding_bytes << '\n';

    std::cout << "\n=== HOST REDUCTION ===\n"
              << "first_host_reduce_cycle: "
              << reduce.first_host_reduce_cycle << '\n'
              << "last_host_reduce_cycle: "
              << reduce.last_host_reduce_cycle << '\n'
              << "host_reduction_complete_cycle: "
              << reduce.host_reduction_complete_cycle << '\n'
              << "reduced_partial_records: "
              << reduce.reduced_partial_records << '\n'
              << "reduced_contribution_count: "
              << reduce.reduced_contribution_count << '\n'
              << "fp32_host_add_count: "
              << reduce.fp32_host_add_count << '\n'
              << "same_row_cross_bg_merges: "
              << reduce.same_row_cross_bg_merges << '\n'
              << "same_row_repeated_record_merges: "
              << reduce.same_row_repeated_record_merges << '\n'
              << "cross_bg_merges_per_physical_record: "
              << double(reduce.same_row_cross_bg_merges) /
                     double(reduce.reduced_partial_records)
              << '\n'
              << "same_bg_repeats_per_physical_record: "
              << double(reduce.same_row_repeated_record_merges) /
                     double(reduce.reduced_partial_records)
              << '\n'
              << "peak_reduction_queue_occupancy: "
              << reduce.peak_reduction_queue_occupancy << '\n'
              << "reduction_queue_backpressure_cycles: "
              << reduce.reduction_queue_backpressure_cycles << '\n';

    std::cout << "\n=== END TO END ===\n"
              << "T_compute_submit: " << counters.t_compute_submit << '\n'
              << "T_bga_drain: " << counters.t_bga_drain << '\n'
              << "T_writeback: " << counters.t_writeback << '\n'
              << "T_readback: " << counters.t_readback << '\n'
              << "T_host_reduce: " << counters.t_host_reduce << '\n'
              << "T_end_to_end: " << counters.t_end_to_end << '\n'
              << "end_to_end_complete_cycle: "
              << counters.end_to_end_complete_cycle << '\n'
              << "result_valid_cycle: "
              << counters.result_valid_cycle << '\n';

    std::cout << "\n=== CONSERVATION ===\n"
              << "contribution_conservation: "
              << passFail(contribution_conservation) << '\n'
              << "record_conservation: "
              << passFail(record_conservation) << '\n'
              << "byte_conservation: "
              << passFail(byte_conservation) << '\n';

    std::cout << "\n=== FINAL RESULT SUMMARY ===\n"
              << "rows: " << result.size() << '\n'
              << "nonzero_output_rows: " << nonzero_rows << '\n'
              << "max_absolute_value: " << max_absolute_value << '\n'
              << "L1: " << l1 << '\n'
              << "L2: " << std::sqrt(l2_squared) << '\n'
              << "nan_count: " << nan_count << '\n'
              << "inf_count: " << inf_count << '\n'
              << "CPU_reference_comparison: "
              << passFail(mismatch_count == 0) << '\n'
              << "max_abs_error: " << max_abs_error << '\n'
              << "max_rel_error: " << max_rel_error << '\n'
              << "mismatch_count: " << mismatch_count << '\n'
              << "resultValid: " << passFail(execution.resultValid())
              << '\n'
              << "endToEndSpMVComplete: "
              << passFail(execution.endToEndSpMVComplete()) << '\n';

    auto print_row = [&](uint32_t row, const char* group) {
        std::cout << group << " row=" << row
                  << " reference=" << reference[row]
                  << " actual=" << result[row]
                  << " abs_error="
                  << std::fabs(double(reference[row]) -
                               double(result[row]))
                  << '\n';
    };
    const uint32_t first_count =
        std::min<uint32_t>(10, result.size());
    for (uint32_t row = 0; row < first_count; ++row)
        print_row(row, "sample_first");
    if (result.size() > 5) {
        const uint32_t middle = result.size() / 2;
        const uint32_t begin = middle >= 2 ? middle - 2 : 0;
        for (uint32_t row = begin;
             row < std::min<uint32_t>(begin + 5, result.size()); ++row)
            print_row(row, "sample_middle");
    }
    const uint32_t last_begin =
        result.size() > 10 ? result.size() - 10 : 0;
    for (uint32_t row = last_begin; row < result.size(); ++row)
        print_row(row, "sample_last");
    for (size_t i = 0; i < std::min<size_t>(5, worst_rows.size()); ++i)
        print_row(worst_rows[i].row, "sample_max_error");
    std::cout << "=== M7B EXTERNAL MATRIX RUN COMPLETE ===\n";

    EXPECT_EQ(mismatch_count, 0U);
}

TEST(CSCM7BFullRunTest, ExternalMatrixWritebackReadbackOnly)
{
    const char* image_path = std::getenv("CSC_EXTERNAL_IMAGE");
    if (!image_path || !*image_path)
        GTEST_SKIP() << "set CSC_EXTERNAL_IMAGE";

    auto loaded = loadExternalPhysicalImage(image_path);
    const auto& layout = loaded.layout;
    const uint64_t max_bg_nnz = *std::max_element(
        layout.stats.bg_nnz.begin(), layout.stats.bg_nnz.end());
    const uint32_t resident_capacity = static_cast<uint32_t>(
        (max_bg_nnz + kCSCPartialRecordsPerBurst - 1) /
        kCSCPartialRecordsPerBurst);

    std::vector<float> x(layout.matrix.cols);
    for (uint32_t i = 0; i < x.size(); ++i)
        x[i] = float((i % 13) + 1) / 7;
    Views views(layout, x);

    const auto bga_config = fullRunBGAConfig(layout.matrix.rows);
    const auto path_config = transportOnlyPathConfig(resident_capacity);
    CSCNativeExecution execution;
    execution.configurePartialResultWriteback(path_config);
    execution.enableProductionBGAIntegration(bga_config);
    execution.launch(views.views, layout.matrix.rows, layout.stats.nnz);

    const auto transport_complete = [&execution] {
        if (!execution.hostReadTransportComplete()) return false;
        const auto& read =
            execution.partialResultPath().readbackCounters();
        return read.returned_bursts_delivered ==
               read.returned_bursts_committed;
    };

    constexpr uint64_t kProgressIntervalCycles = 100000;
    uint64_t next_progress_cycle = kProgressIntervalCycles;
    for (uint64_t guard = 0;
         guard < 2000000000ULL && !transport_complete(); ++guard) {
        execution.tick();
        while (execution.hasHostReturnedBurst())
            execution.acceptHostReturnedBurst();
        if (execution.cycle() < next_progress_cycle) continue;
        const auto progress = execution.counters();
        const auto& path = execution.partialResultPath();
        std::cout << "M7_TRANSPORT_PROGRESS cycle="
                  << execution.cycle()
                  << " accepted_contributions="
                  << progress.accepted_bga_partials
                  << " bga_outputs="
                  << progress.bga_outputs_accepted
                  << " writes_completed="
                  << path.counters().write_requests_completed
                  << " reads_completed="
                  << path.readbackCounters().read_requests_completed
                  << '\n'
                  << std::flush;
        next_progress_cycle += kProgressIntervalCycles;
    }

    ASSERT_FALSE(execution.hasFailed()) << execution.errorMessage();
    ASSERT_TRUE(transport_complete());
    ASSERT_TRUE(execution.bgaComputeSubmitComplete());
    ASSERT_TRUE(execution.bgaDrainComplete());
    ASSERT_TRUE(execution.partialWritebackComplete());
    ASSERT_TRUE(execution.hostReadTransportComplete());

    const auto counters = execution.counters();
    const auto& path = execution.partialResultPath();
    const auto& write = path.counters();
    const auto& read = path.readbackCounters();
    const bool contribution_conservation =
        counters.accepted_bga_partials == layout.stats.nnz &&
        counters.bga_output_contributions_accepted == layout.stats.nnz &&
        write.partial_record_contribution_sum == layout.stats.nnz &&
        read.readback_contribution_sum == layout.stats.nnz;
    const bool record_conservation =
        counters.bga_outputs_accepted == write.partial_records_generated &&
        write.partial_records_generated == read.readback_records;
    const bool byte_conservation =
        write.writeback_useful_bytes == read.readback_useful_bytes &&
        write.writeback_transferred_bytes ==
            read.readback_transferred_bytes &&
        write.writeback_padding_bytes == read.readback_padding_bytes;
    const double aggregation_ratio = layout.stats.nnz
        ? double(counters.bga_outputs_accepted) /
              double(layout.stats.nnz)
        : 0.0;
    const double reduction_factor = counters.bga_outputs_accepted
        ? double(layout.stats.nnz) /
              double(counters.bga_outputs_accepted)
        : 0.0;
    const uint64_t writeback_cycles =
        write.partial_writeback_complete_cycle -
        write.first_bga_output_accept_cycle;
    const uint64_t readback_cycles =
        read.read_transport_complete_cycle -
        read.host_readback_start_cycle;

    std::cout << std::fixed << std::setprecision(6)
              << "\n=== M7 BGA + WRITEBACK + READBACK ONLY ===\n"
              << "rows: " << layout.matrix.rows << '\n'
              << "cols: " << layout.matrix.cols << '\n'
              << "nnz: " << layout.stats.nnz << '\n'
              << "bga_accumulator_entries: "
              << bga_config.accumulator.accumulator_entries << '\n'
              << "bga_output_queue_depth: "
              << bga_config.accumulator.output_queue_depth << '\n'
              << "writeback_accepts_per_cycle: "
              << bga_config.partial_writeback_accepts_per_cycle << '\n'
              << "compute_submit_complete_cycle: "
              << counters.compute_submit_complete_cycle << '\n'
              << "bga_drain_complete_cycle: "
              << counters.bga_drain_complete_cycle << '\n'
              << "partial_writeback_complete_cycle: "
              << write.partial_writeback_complete_cycle << '\n'
              << "read_transport_complete_cycle: "
              << read.read_transport_complete_cycle << '\n'
              << "transport_result_delivered_cycle: "
              << execution.cycle() << '\n'
              << "T_compute_submit: " << counters.t_compute_submit << '\n'
              << "T_bga_drain: " << counters.t_bga_drain << '\n'
              << "T_writeback: " << writeback_cycles << '\n'
              << "T_readback: " << readback_cycles << '\n'
              << "T_scope_matched_total: " << execution.cycle() << '\n'
              << "physical_bga_output_records: "
              << counters.bga_outputs_accepted << '\n'
              << "bga_aggregation_ratio: " << aggregation_ratio << '\n'
              << "bga_reduction_factor: " << reduction_factor << '\n'
              << "bga_backpressure_cycles: "
              << counters.bga_backpressure_cycles << '\n'
              << "write_requests_issued/completed: "
              << write.write_requests_issued << '/'
              << write.write_requests_completed << '\n'
              << "read_requests_issued/completed: "
              << read.read_requests_issued << '/'
              << read.read_requests_completed << '\n'
              << "writeback_useful/transferred/padding_bytes: "
              << write.writeback_useful_bytes << '/'
              << write.writeback_transferred_bytes << '/'
              << write.writeback_padding_bytes << '\n'
              << "readback_useful/transferred/padding_bytes: "
              << read.readback_useful_bytes << '/'
              << read.readback_transferred_bytes << '/'
              << read.readback_padding_bytes << '\n'
              << "return_queue_backpressure_cycles: "
              << read.return_queue_backpressure_cycles << '\n'
              << "host_reduction_enabled: false\n"
              << "contribution_conservation: "
              << passFail(contribution_conservation) << '\n'
              << "record_conservation: "
              << passFail(record_conservation) << '\n'
              << "byte_conservation: "
              << passFail(byte_conservation) << '\n'
              << "=== M7 TRANSPORT-ONLY RUN COMPLETE ===\n";

    EXPECT_TRUE(contribution_conservation);
    EXPECT_TRUE(record_conservation);
    EXPECT_TRUE(byte_conservation);
    EXPECT_FALSE(path.config().host_reduction_enabled);
    EXPECT_FALSE(execution.hostReductionComplete());
    EXPECT_FALSE(execution.resultValid());
    EXPECT_FALSE(execution.endToEndSpMVComplete());
}

TEST(CSCM7BFullRunTest, ExternalMatrixBGAValidationOnly)
{
    const char* image_path = std::getenv("CSC_EXTERNAL_IMAGE");
    if (!image_path || !*image_path)
        GTEST_SKIP() << "set CSC_EXTERNAL_IMAGE";

    auto loaded = loadExternalPhysicalImage(image_path);
    const auto& layout = loaded.layout;
    std::vector<float> x(layout.matrix.cols);
    for (uint32_t i = 0; i < x.size(); ++i)
        x[i] = float((i % 13) + 1) / 7;
    const auto reference = cpuReference(layout.matrix, x);

    Views views(layout, x);
    auto bga_config = fullRunBGAConfig(layout.matrix.rows);
    bga_config.output_consumer_mode =
        CSCBGAOutputConsumerMode::VALIDATION_ROUND_ROBIN;
    bga_config.validation_consumer_accepts_per_cycle =
        envUint32("CSC_BGA_VALIDATION_ACCEPTS_PER_CYCLE", 1);

    CSCNativeExecution execution;
    execution.enableProductionBGAIntegration(bga_config);
    execution.launch(views.views, layout.matrix.rows, layout.stats.nnz);

    constexpr uint64_t kProgressIntervalCycles = 100000;
    uint64_t next_progress_cycle = kProgressIntervalCycles;
    for (uint64_t guard = 0;
         guard < 2000000000ULL && !execution.isTerminal(); ++guard) {
        execution.tick();
        if (execution.cycle() < next_progress_cycle) continue;
        const auto progress = execution.counters();
        std::cout << "M7A_BGA_PROGRESS cycle=" << execution.cycle()
                  << " accepted_contributions="
                  << progress.accepted_bga_partials
                  << " bga_outputs="
                  << progress.bga_outputs_accepted << '\n'
                  << std::flush;
        next_progress_cycle += kProgressIntervalCycles;
    }

    ASSERT_FALSE(execution.hasFailed()) << execution.errorMessage();
    ASSERT_TRUE(execution.isTerminal());
    ASSERT_TRUE(execution.isDone());
    ASSERT_TRUE(execution.bgaExecutionComplete());

    const auto counters = execution.counters();
    const auto& validation = execution.bgaValidationCollector();
    ASSERT_EQ(validation.validation_y_double.size(), reference.size());

    uint64_t mismatch_count = 0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    uint32_t max_error_row = 0;
    for (uint32_t row = 0; row < reference.size(); ++row) {
        const double actual = validation.validation_y_double[row];
        const double expected = reference[row];
        const double absolute_error = std::fabs(actual - expected);
        const double relative_error = absolute_error /
            std::max(std::fabs(expected), 1.0e-30);
        if (absolute_error > max_abs_error) {
            max_abs_error = absolute_error;
            max_error_row = row;
        }
        max_rel_error = std::max(max_rel_error, relative_error);
        if (absolute_error >
            kAbsoluteTolerance +
                kRelativeTolerance * std::fabs(expected))
            ++mismatch_count;
    }

    const double aggregation_ratio = layout.stats.nnz
        ? double(counters.bga_outputs_accepted) / double(layout.stats.nnz)
        : 0.0;
    const double reduction_factor = counters.bga_outputs_accepted
        ? double(layout.stats.nnz) / double(counters.bga_outputs_accepted)
        : 0.0;

    std::cout << std::fixed << std::setprecision(6)
              << "\n=== M7A BGA VALIDATION ONLY ===\n"
              << "rows: " << layout.matrix.rows << '\n'
              << "cols: " << layout.matrix.cols << '\n'
              << "nnz: " << layout.stats.nnz << '\n'
              << "bga_accumulator_entries: "
              << bga_config.accumulator.accumulator_entries << '\n'
              << "bga_compare_width: "
              << bga_config.accumulator.compare_width << '\n'
              << "bga_output_queue_depth: "
              << bga_config.accumulator.output_queue_depth << '\n'
              << "validation_accepts_per_cycle: "
              << bga_config.validation_consumer_accepts_per_cycle << '\n'
              << "compute_submit_complete_cycle: "
              << counters.compute_submit_complete_cycle << '\n'
              << "bga_drain_complete_cycle: "
              << counters.bga_drain_complete_cycle << '\n'
              << "physical_bga_output_records: "
              << counters.bga_outputs_accepted << '\n'
              << "bga_output_contribution_sum: "
              << counters.bga_output_contributions_accepted << '\n'
              << "capacity_eviction_outputs: "
              << counters.bga_capacity_outputs_accepted << '\n'
              << "final_drain_outputs: "
              << counters.bga_final_drain_outputs_accepted << '\n'
              << "bga_aggregation_ratio: " << aggregation_ratio << '\n'
              << "bga_reduction_factor: " << reduction_factor << '\n'
              << "bga_backpressure_cycles: "
              << counters.bga_backpressure_cycles << '\n'
              << "contribution_conservation: "
              << passFail(counters.accepted_bga_partials == layout.stats.nnz &&
                          counters.bga_output_contributions_accepted ==
                              layout.stats.nnz &&
                          validation.accepted_contributions ==
                              layout.stats.nnz)
              << '\n'
              << "bga_semantic_validation: "
              << passFail(execution.bgaValidationSemanticMatches()) << '\n'
              << "CPU_reference_comparison: "
              << passFail(mismatch_count == 0) << '\n'
              << "max_abs_error: " << max_abs_error << '\n'
              << "max_rel_error: " << max_rel_error << '\n'
              << "mismatch_count: " << mismatch_count << '\n'
              << "sample_max_error row=" << max_error_row
              << " reference=" << reference[max_error_row]
              << " actual="
              << validation.validation_y_double[max_error_row] << '\n'
              << "resultValid: NOT_APPLICABLE (validation-only mode)\n"
              << "endToEndSpMVComplete: NOT_APPLICABLE "
                 "(validation-only mode)\n"
              << "=== M7A BGA VALIDATION RUN COMPLETE ===\n";

    EXPECT_EQ(counters.accepted_bga_partials, layout.stats.nnz);
    EXPECT_EQ(counters.bga_output_contributions_accepted, layout.stats.nnz);
    EXPECT_EQ(validation.accepted_contributions, layout.stats.nnz);
    EXPECT_TRUE(execution.bgaValidationSemanticMatches());
    EXPECT_EQ(mismatch_count, 0U);
    EXPECT_FALSE(execution.resultValid());
    EXPECT_FALSE(execution.endToEndSpMVComplete());
}
