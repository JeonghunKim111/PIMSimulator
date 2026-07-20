#include "tests/SpmvDrafBgaStructuralBV2.h"

#include <algorithm>
#include <iostream>

#include "tests/SpmvDrafBgaStructuralCommon.h"

namespace spmv
{
namespace
{
using namespace std;

constexpr unsigned kHostReduceWidth = 16;
constexpr uint64_t kMetadataCyclePerColumnGroup = 1;
constexpr uint64_t kMetadataCyclePerDrafRow = 2;
constexpr uint64_t kMetadataCyclePerCluster = 256;
constexpr uint64_t kCommandIssueCyclePerColumnGroup = 1;
constexpr uint64_t kSyncCyclePerCluster = 1024;
constexpr uint64_t kSyncCyclePerBgaGroup = 128;

} // namespace

StructuralModelResult runDrafBgaStructuralModelBV2(
    const SpmvInputs& inputs, const DrafStats& draf, const BgaStats& bga,
    const ShapeStats& shape, const DrafCriticalPathStats& critical,
    const DrafPaddingExposureStats& exposure, const StructuralBaseTiming& timing,
    const StructuralVariantSpec& variant_spec, const string& matrix_name,
    double gpu_ms, double target_speedup, double tck_ns)
{
    uint64_t gemv_like_compute_cycle =
        ceilDiv(draf.packed_nnz_capacity, kElementsPerBurst);
    uint64_t useful_compute_cycle = ceilDiv(draf.nnz, kElementsPerBurst);
    uint64_t padding_cycle = gemv_like_compute_cycle > useful_compute_cycle
                                 ? gemv_like_compute_cycle - useful_compute_cycle
                                 : 0;

    uint64_t bga_capacity_flushes =
        max(bga.max_estimated_flushes_per_group, bga.max_stream_capacity_flushes_per_group);
    uint64_t bga_accumulate_cycle =
        bga.max_bacc_instructions_per_group +
        kConservativeBgaFlushPenalty * bga_capacity_flushes;
    uint64_t final_reduce_cycle = ceilDiv(bga.host_reduce_ops_after_bga, kHostReduceWidth);

    uint64_t runtime_metadata_traversal_cycle =
        draf.column_groups * kMetadataCyclePerColumnGroup +
        draf.draf_rows * kMetadataCyclePerDrafRow +
        inputs.clusters.size() * kMetadataCyclePerCluster;
    uint64_t pim_command_issue_cycle =
        draf.column_groups * kCommandIssueCyclePerColumnGroup;
    uint64_t phase_sync_cycle =
        inputs.clusters.size() * kSyncCyclePerCluster +
        bga.groups.size() * kSyncCyclePerBgaGroup;

    uint64_t total_cycle =
        timing.setup_cycle +
        timing.draf_row_fetch_cycle +
        timing.pim_enable_cycle +
        timing.draf_compute_trigger_cycle +
        gemv_like_compute_cycle +
        bga_accumulate_cycle +
        timing.pim_disable_cycle +
        timing.pim_to_sb_cycle +
        timing.bga_output_readback_cycle +
        final_reduce_cycle +
        runtime_metadata_traversal_cycle +
        pim_command_issue_cycle +
        phase_sync_cycle;

    double model_ms = static_cast<double>(total_cycle) * tck_ns / 1000000.0;
    double model_speedup = model_ms == 0.0 ? 0.0 : gpu_ms / model_ms;
    double target_pim_ms = target_speedup == 0.0 ? 0.0 : gpu_ms / target_speedup;
    double mean_nnz_per_col =
        inputs.n_cols == 0 ? 0.0
                           : static_cast<double>(draf.nnz) /
                                 static_cast<double>(inputs.n_cols);

    StructuralModelResult result;
    result.matrix = matrix_name;
    result.gpu_ms = gpu_ms;
    result.target_speedup = target_speedup;
    result.target_pim_ms = target_pim_ms;
    result.model_ms = model_ms;
    result.model_speedup = model_speedup;
    result.speedup_error_ratio =
        target_speedup == 0.0 ? 0.0 : model_speedup / target_speedup;
    result.total_cycle = total_cycle;
    result.setup_cycle = timing.setup_cycle + runtime_metadata_traversal_cycle +
                         pim_command_issue_cycle + phase_sync_cycle;
    result.draf_row_fetch_cycle = timing.draf_row_fetch_cycle;
    result.draf_compute_trigger_cycle = timing.draf_compute_trigger_cycle +
                                        useful_compute_cycle;
    result.padding_cycle = padding_cycle;
    result.bga_accumulate_cycle = bga_accumulate_cycle;
    result.bga_output_readback_cycle = timing.bga_output_readback_cycle;
    result.final_reduce_cycle = final_reduce_cycle;
    result.draf_padding_ratio = draf.nze_padding_ratio;
    result.draf_memory_expansion = draf.expansion_ratio;
    result.critical_padding = critical.critical_padding;
    result.critical_padding_ratio = critical.critical_padding_ratio;
    result.bg_imbalance = critical.bg_imbalance;
    result.total_padding_steps = exposure.total_padding_steps;
    result.hidden_padding_steps = exposure.hidden_padding_steps;
    result.fragmentation = exposure.fragmentation;
    result.memory_pressure = exposure.memory_pressure;
    result.imbalance_pressure = exposure.imbalance_pressure;
    result.exposure_factor = exposure.exposure_factor;
    result.exposed_hidden_padding = exposure.exposed_hidden_padding;
    result.effective_padding_steps = exposure.effective_padding_steps;
    result.bga_raw_accumulate_cycle = bga_accumulate_cycle;
    result.row_nnz_gini = shape.row_nnz_gini;
    result.col_nnz_gini = shape.col_nnz_gini;
    result.mean_nnz_per_col = mean_nnz_per_col;
    result.single_nnz_column_ratio = shape.single_nnz_column_ratio;
    result.low_nnz_column_ratio = shape.low_nnz_column_ratio;
    result.bga_row_reuse_factor = shape.bga_row_reuse_factor;
    result.bga_unique_row_ratio = shape.bga_unique_row_ratio;
    result.bga_duplicate_partial_ratio = shape.bga_duplicate_partial_ratio;
    result.bga_near_duplicate_partial_ratio = bga.near_duplicate_partial_ratio;
    result.bga_far_duplicate_partial_ratio = bga.far_duplicate_partial_ratio;
    result.bga_near_duplicate_share = bga.near_duplicate_share;
    result.accumulator_flush_estimate = bga.accumulator_flush_estimate;
    result.bga_capacity_pressure = shape.bga_capacity_pressure;
    result.bga_stream_capacity_flushes = bga.max_stream_capacity_flushes_per_group;
    result.bga_critical_flushes = bga_capacity_flushes;
    result.bga_reuse_aware_accumulate_cycle = bga.max_reuse_aware_accumulate_cycle_per_group;
    result.draf_coo_bytes_per_nnz = kSparseValueBytes + kCooIndexBytesPerNnz;
    result.draf_bytes_per_nnz =
        kSparseValueBytes * draf.expansion_ratio +
        kDrafCompactMetadataBytesPerNnz;
    result.draf_vs_coo_memory_ratio =
        result.draf_coo_bytes_per_nnz == 0.0
            ? 0.0
            : result.draf_bytes_per_nnz / result.draf_coo_bytes_per_nnz;
    result.draf_memory_saving_factor =
        clamp01(1.0 - result.draf_vs_coo_memory_ratio);
    result.group_steps_p90 = critical.group_steps_p90;
    result.group_steps_p95 = critical.group_steps_p95;
    result.group_steps_max = critical.group_steps_max;
    result.tail_skew_max_over_p90 = critical.tail_skew_max_over_p90;
    result.tail_skew_max_over_p95 = critical.tail_skew_max_over_p95;
    result.p90_to_max_tail_steps = critical.p90_to_max_tail_steps;
    result.p95_to_max_tail_steps = critical.p95_to_max_tail_steps;

    cout << "  latency_scope: BV-1 + runtime metadata traversal"
         << " + command issue + phase synchronization" << endl;
    cout << "> setup_cycle: " << timing.setup_cycle << endl;
    cout << "> runtime_metadata_traversal_cycle: "
         << runtime_metadata_traversal_cycle
         << " column_groups=" << draf.column_groups
         << " draf_rows=" << draf.draf_rows
         << " clusters=" << inputs.clusters.size() << endl;
    cout << "> pim_command_issue_cycle: " << pim_command_issue_cycle
         << " per_column_group=" << kCommandIssueCyclePerColumnGroup << endl;
    cout << "> phase_sync_cycle: " << phase_sync_cycle
         << " clusters=" << inputs.clusters.size()
         << " bga_groups=" << bga.groups.size() << endl;
    cout << "> draf_row_fetch_cycle: " << timing.draf_row_fetch_cycle << endl;
    cout << "> draf_compute_trigger_cycle: "
         << timing.draf_compute_trigger_cycle << endl;
    cout << "> gemv_like_compute_cycle: " << gemv_like_compute_cycle
         << " useful_compute_cycle=" << useful_compute_cycle
         << " padding_cycle=" << padding_cycle
         << " packed_nnz_capacity=" << draf.packed_nnz_capacity
         << " nnz=" << draf.nnz << endl;
    cout << "> bga_accumulate_cycle: " << bga_accumulate_cycle
         << " max_bacc_per_group=" << bga.max_bacc_instructions_per_group
         << " selected_flushes_per_group=" << bga_capacity_flushes
         << " flush_penalty=" << kConservativeBgaFlushPenalty << endl;
    cout << "> bga_output_readback_cycle: "
         << timing.bga_output_readback_cycle << endl;
    cout << "> final_reduce_cycle: " << final_reduce_cycle << endl;
    cout << "> " << variant_spec.cycle_label << total_cycle
         << " ms=" << model_ms << endl;
    cout << "> gpu_baseline_ms: " << gpu_ms << endl;
    cout << "> paper_target_speedup: " << target_speedup
         << " target_pim_ms=" << target_pim_ms << endl;
    cout << "> " << variant_spec.speedup_label << model_speedup
         << " speedup_error_ratio=" << result.speedup_error_ratio << endl;
    cout << variant_spec.result_csv_tag
         << matrix_name << "," << gpu_ms << "," << target_speedup << ","
         << target_pim_ms << "," << model_ms << "," << model_speedup << ","
         << result.speedup_error_ratio << "," << total_cycle << ","
         << gemv_like_compute_cycle << "," << useful_compute_cycle << ","
         << padding_cycle << "," << bga_accumulate_cycle << ","
         << timing.draf_row_fetch_cycle << ","
         << timing.draf_compute_trigger_cycle << ","
         << timing.bga_output_readback_cycle << ","
         << final_reduce_cycle << ","
         << runtime_metadata_traversal_cycle << ","
         << pim_command_issue_cycle << ","
         << phase_sync_cycle << endl;

    return result;
}

} // namespace spmv
