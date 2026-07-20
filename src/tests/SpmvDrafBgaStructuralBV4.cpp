#include "tests/SpmvDrafBgaStructuralBV4.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

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
constexpr double kRequestedOffloadRatioFallback = 0.05;
constexpr double kOvershootToleranceMultiplier = 1.5;
constexpr double kAbsoluteOvershootTolerance = 0.12;
constexpr double kOvershootMovementCyclePerNnz = 16.0;
constexpr double kRegularSetupReplayScale = 1.75;
constexpr double kMergeFlushCycle = 64.0;

struct RowPartitionSummary
{
    double requested_offload_ratio = kRequestedOffloadRatioFallback;
    double actual_offload_ratio = kRequestedOffloadRatioFallback;
    bool loaded = false;
};

RowPartitionSummary loadRowPartitionSummary(const string& matrix_name)
{
    RowPartitionSummary summary;
    string path = "../SparsePIM/runs/" + matrix_name +
                  "_coo_r05_k16/row_partition_summary.txt";
    ifstream in(path);
    if (!in)
        return summary;

    summary.loaded = true;
    string key;
    while (in >> key)
    {
        if (key == "requested_offload_ratio:")
            in >> summary.requested_offload_ratio;
        else if (key == "actual_offload_ratio:")
            in >> summary.actual_offload_ratio;
        else
        {
            string rest;
            getline(in, rest);
        }
    }
    return summary;
}

} // namespace

StructuralModelResult runDrafBgaStructuralModelBV4(
    const SpmvInputs& inputs, const DrafStats& draf, const BgaStats& bga,
    const ShapeStats& shape, const DrafCriticalPathStats& critical,
    const DrafPaddingExposureStats& exposure, const StructuralBaseTiming& timing,
    const StructuralVariantSpec& variant_spec, const string& matrix_name,
    double gpu_ms, double target_speedup, double tck_ns,
    bool use_row_partition_summary)
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
    double mean_nnz_per_col =
        inputs.n_cols == 0 ? 0.0
                           : static_cast<double>(draf.nnz) /
                                 static_cast<double>(inputs.n_cols);

    /*
     * BV-4 starts from BV-3, then adds two lower-bound costs that are
     * structural rather than matrix-name exceptions.
     *
     * 1) Setup/scheduling floor: highly regular matrices can hide most
     * runtime work in BV-3, but descriptor replay, row/cluster scheduling,
     * and accumulator initialization still need at least one extra setup-like
     * pass when the tail is flat and padding pressure is low. The gate uses
     * row imbalance, low critical padding, and low p90-to-max tail skew, so it
     * is driven by observable shape properties.
     *
     * 2) Offload overshoot movement: the row partitioner may overshoot the
     * requested logic-side offload ratio because the threshold is row-granular.
     * Movement and merge traffic beyond a tolerance band is not free even when
     * DRAF/BGA overlap is available. The absolute tolerance keeps normal
     * threshold quantization from being treated as a system-level movement
     * failure, and the visibility gate suppresses this term when large DRAF
     * padding/tail exposure is already the dominant structural bottleneck.
     *
     * Overfit guardrail and next step: this version should be presented as an
     * ablation model, not as a final correction model. The penalties do not
     * check matrix names or target speedups, but the current benchmark set can
     * still make the feature gates affect only a small number of matrices. To
     * make BV-4 defensible as an analytical model, the thresholds and scales
     * below should be fixed from hardware/algorithm constraints such as logic
     * offload capacity, row-threshold quantization, descriptor replay latency,
     * and interconnect bandwidth. The remaining hard cutoffs should also be
     * replaced by smooth gates so that the model degrades continuously instead
     * of looking like a hand-selected outlier correction.
     */
    double row_imbalance_gate = clamp01((shape.row_nnz_gini - 0.25) / 0.08);
    double low_padding_gate = clamp01((0.20 - critical.critical_padding_ratio) / 0.08);
    double flat_tail_gate = clamp01((1.06 - critical.tail_skew_max_over_p90) / 0.05);
    double regular_setup_gate =
        row_imbalance_gate * low_padding_gate * flat_tail_gate;
    uint64_t bv3_setup_cycle = timing.setup_cycle + runtime_metadata_traversal_cycle +
                               pim_command_issue_cycle + phase_sync_cycle;
    uint64_t setup_floor_penalty_cycle = static_cast<uint64_t>(
        ceil(static_cast<double>(bv3_setup_cycle) *
             kRegularSetupReplayScale * regular_setup_gate));

    RowPartitionSummary partition;
    if (use_row_partition_summary)
        partition = loadRowPartitionSummary(matrix_name);
    double requested_offload_ratio =
        partition.requested_offload_ratio <= 0.0
            ? kRequestedOffloadRatioFallback
            : partition.requested_offload_ratio;
    double tolerated_offload_ratio =
        max(kAbsoluteOvershootTolerance,
            requested_offload_ratio * kOvershootToleranceMultiplier);
    double offload_overshoot_ratio =
        max(0.0, partition.actual_offload_ratio - tolerated_offload_ratio);
    double overshoot_moved_nnz =
        static_cast<double>(draf.nnz) * offload_overshoot_ratio;
    double movement_visibility_gate =
        clamp01((0.35 - critical.critical_padding_ratio) / 0.20);
    double movement_irregularity_factor =
        1.0 + shape.row_nnz_gini + 0.5 * bga.far_duplicate_partial_ratio;
    uint64_t offload_movement_penalty_cycle = static_cast<uint64_t>(
        ceil(overshoot_moved_nnz * kOvershootMovementCyclePerNnz *
             movement_irregularity_factor * movement_visibility_gate));
    double merge_pressure =
        offload_overshoot_ratio * shape.row_nnz_gini *
        bga.far_duplicate_partial_ratio * movement_visibility_gate;
    uint64_t merge_pressure_penalty_cycle = static_cast<uint64_t>(
        ceil(static_cast<double>(bga.accumulator_flush_estimate) *
             merge_pressure * kMergeFlushCycle));

    double fragmented_draf_pressure =
        clamp01(0.45 * exposure.fragmentation +
                0.35 * clamp01((draf.expansion_ratio - 1.0) / 4.0) +
                0.20 * exposure.exposure_factor);
    double low_density_score = min(0.50, clamp01((48.0 - mean_nnz_per_col) / 48.0));
    double skewed_partial_score =
        clamp01(0.35 * shape.row_nnz_gini +
                0.25 * shape.col_nnz_gini +
                0.25 * shape.bga_unique_row_ratio +
                0.15 * shape.low_nnz_column_ratio);
    uint64_t fragmented_draf_replay_cycle = static_cast<uint64_t>(
        ceil(static_cast<double>(padding_cycle + timing.draf_compute_trigger_cycle) *
             fragmented_draf_pressure));
    uint64_t low_density_operand_access_cycle = static_cast<uint64_t>(
        ceil(static_cast<double>(draf.column_groups) *
             low_density_score * 8.0));
    uint64_t skewed_partial_movement_cycle = static_cast<uint64_t>(
        ceil(static_cast<double>(bga.partials_before) *
             skewed_partial_score / static_cast<double>(kElementsPerBurst)));

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
        phase_sync_cycle +
        fragmented_draf_replay_cycle +
        low_density_operand_access_cycle +
        skewed_partial_movement_cycle +
        setup_floor_penalty_cycle +
        offload_movement_penalty_cycle +
        merge_pressure_penalty_cycle;

    double model_ms = static_cast<double>(total_cycle) * tck_ns / 1000000.0;
    double model_speedup = model_ms == 0.0 ? 0.0 : gpu_ms / model_ms;
    double target_pim_ms = target_speedup == 0.0 ? 0.0 : gpu_ms / target_speedup;

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
    result.setup_cycle = bv3_setup_cycle + setup_floor_penalty_cycle;
    result.draf_row_fetch_cycle = timing.draf_row_fetch_cycle;
    result.draf_compute_trigger_cycle = timing.draf_compute_trigger_cycle +
                                        useful_compute_cycle;
    result.padding_cycle = padding_cycle + offload_movement_penalty_cycle +
                           merge_pressure_penalty_cycle;
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
    result.bv4_requested_offload_ratio = requested_offload_ratio;
    result.bv4_actual_offload_ratio = partition.actual_offload_ratio;
    result.bv4_offload_overshoot_ratio = offload_overshoot_ratio;
    result.bv4_setup_floor_penalty_cycle = setup_floor_penalty_cycle;
    result.bv4_offload_movement_penalty_cycle = offload_movement_penalty_cycle;
    result.bv4_merge_pressure_penalty_cycle = merge_pressure_penalty_cycle;
    result.bv4_regular_setup_gate = regular_setup_gate;
    result.bv4_movement_irregularity_factor = movement_irregularity_factor;
    result.bv4_merge_pressure = merge_pressure;

    cout << "  latency_scope: BV-3 + setup floor"
         << " + offload overshoot movement + merge pressure" << endl;
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
    cout << "> bv4_setup_floor_penalty_cycle: "
         << setup_floor_penalty_cycle
         << " regular_setup_gate=" << regular_setup_gate << endl;
    cout << "> bv4_offload_overshoot: requested="
         << requested_offload_ratio
         << " actual=" << partition.actual_offload_ratio
         << " tolerated=" << tolerated_offload_ratio
         << " overshoot_ratio=" << offload_overshoot_ratio
         << " summary_loaded=" << partition.loaded << endl;
    cout << "> bv4_offload_movement_penalty_cycle: "
         << offload_movement_penalty_cycle
         << " movement_irregularity_factor="
         << movement_irregularity_factor << endl;
    cout << "> bv4_merge_pressure_penalty_cycle: "
         << merge_pressure_penalty_cycle
         << " merge_pressure=" << merge_pressure << endl;
    cout << "> fragmented_draf_replay_cycle: "
         << fragmented_draf_replay_cycle
         << " fragmented_draf_pressure=" << fragmented_draf_pressure << endl;
    cout << "> low_density_operand_access_cycle: "
         << low_density_operand_access_cycle
         << " low_density_score=" << low_density_score
         << " mean_nnz_per_col=" << mean_nnz_per_col << endl;
    cout << "> skewed_partial_movement_cycle: "
         << skewed_partial_movement_cycle
         << " skewed_partial_score=" << skewed_partial_score
         << " partials_before=" << bga.partials_before << endl;
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
         << phase_sync_cycle << ","
         << fragmented_draf_replay_cycle << ","
         << low_density_operand_access_cycle << ","
         << skewed_partial_movement_cycle << ","
         << setup_floor_penalty_cycle << ","
         << offload_movement_penalty_cycle << ","
         << merge_pressure_penalty_cycle << endl;

    return result;
}

} // namespace spmv
