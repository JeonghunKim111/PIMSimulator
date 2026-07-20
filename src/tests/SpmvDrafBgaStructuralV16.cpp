#include "tests/SpmvDrafBgaStructuralV16.h"

/*
 * Experimental coefficient-tuning branch derived from v15.
 *
 * This version tried to improve class separation among BGA/DRAF-heavy
 * underprediction cases, graph-like overprediction cases, and regular
 * false-positive overprediction cases by retuning the v15 scoring gates.
 * The full-suite result was very close to v15, so this file is kept as an
 * experimental reference rather than the preferred direction for the next
 * analytical model. Future work should favor phase-separated exposure/saving
 * budgets over additional score/threshold tuning.
 */

#include <algorithm>
#include <cmath>
#include <iostream>

#include "tests/SpmvDrafBgaStructuralCommon.h"

namespace spmv
{
namespace
{
using namespace std;

constexpr unsigned kConservativeHostReduceWidth = 16;

uint64_t proportionalGrant(uint64_t effective_saving, uint64_t demand,
                           uint64_t component_demand)
{
    if (demand == 0 || component_demand == 0)
        return 0;
    return min(component_demand,
               static_cast<uint64_t>(
                   floor(static_cast<double>(effective_saving) *
                         static_cast<double>(component_demand) /
                         static_cast<double>(demand))));
}

} // namespace

StructuralModelResult runDrafBgaStructuralModelV16(
    const SpmvInputs& inputs, const DrafStats& draf, const BgaStats& bga,
    const ShapeStats& shape, const DrafCriticalPathStats& critical,
    const DrafPaddingExposureStats& exposure, const StructuralBaseTiming& timing,
    const StructuralVariantSpec& variant_spec, const string& matrix_name,
    double gpu_ms, double target_speedup, double tck_ns)
{
    uint64_t bga_capacity_flushes =
        max(bga.max_estimated_flushes_per_group, bga.max_stream_capacity_flushes_per_group);
    uint64_t instruction_bga_accumulate_cycle =
        bga.max_bacc_instructions_per_group + kConservativeBgaFlushPenalty * bga_capacity_flushes;
    uint64_t reuse_aware_bga_accumulate_cycle =
        bga.max_reuse_aware_accumulate_cycle_per_group;

    double mean_nnz_per_col =
        inputs.n_cols == 0 ? 0.0
                           : static_cast<double>(draf.nnz) /
                                 static_cast<double>(inputs.n_cols);
    double instruction_phase_total =
        static_cast<double>(instruction_bga_accumulate_cycle +
                            timing.draf_row_fetch_cycle +
                            timing.draf_compute_trigger_cycle) +
        exposure.effective_padding_steps + 1.0;
    double instruction_bga_share =
        static_cast<double>(instruction_bga_accumulate_cycle) / instruction_phase_total;
    double weak_reuse_serialization_guard =
        clamp01(1.0 - (shape.bga_row_reuse_factor - 1.0) / 32.0);
    double bga_bound_score =
        clamp01(0.55 * instruction_bga_share +
                0.25 * clamp01(shape.bga_capacity_pressure / 16.0) +
                0.20 * weak_reuse_serialization_guard);
    double draf_access_bound_score =
        clamp01(0.45 * clamp01(draf.expansion_ratio - 1.0) +
                0.30 * shape.draf_padding_pressure +
                0.25 * clamp01(mean_nnz_per_col / 64.0));
    double padding_sync_bound_score =
        clamp01(0.50 * exposure.exposure_factor +
                0.30 * critical.critical_padding_ratio +
                0.20 * (1.0 - critical.tail_exposure_ratio));
    double percentile_tail_exposure_factor =
        clamp01(0.55 + 0.45 * critical.tail_exposure_ratio);
    double phase_class_code =
        bga_bound_score >= draf_access_bound_score &&
                bga_bound_score >= padding_sync_bound_score
            ? 1.0
            : (draf_access_bound_score >= padding_sync_bound_score ? 2.0 : 3.0);

    double conservative_bga_floor_factor =
        clamp01(0.35 + 0.45 * bga_bound_score +
                0.20 * padding_sync_bound_score -
                0.25 * clamp01((shape.bga_row_reuse_factor - 8.0) / 32.0));
    uint64_t conservative_bga_floor_cycle = static_cast<uint64_t>(
        ceil(conservative_bga_floor_factor *
             static_cast<double>(instruction_bga_accumulate_cycle)));
    reuse_aware_bga_accumulate_cycle =
        max(reuse_aware_bga_accumulate_cycle, conservative_bga_floor_cycle);
    uint64_t raw_bga_accumulate_cycle = reuse_aware_bga_accumulate_cycle;

    uint64_t exposed_padding_cycle = static_cast<uint64_t>(
        ceil(exposure.effective_padding_steps * percentile_tail_exposure_factor));
    uint64_t compute_overlap_window =
        timing.draf_compute_trigger_cycle + exposed_padding_cycle;
    BgaOverlapStats bga_overlap =
        buildBgaOverlapStats(raw_bga_accumulate_cycle, compute_overlap_window, shape, exposure);
    DrafStreamingOverlapStats draf_streaming =
        buildDrafStreamingOverlapStats(timing.draf_row_fetch_cycle,
                                       timing.draf_compute_trigger_cycle,
                                       timing.draf_row_fetch_cycle +
                                           timing.draf_compute_trigger_cycle +
                                           exposed_padding_cycle +
                                           raw_bga_accumulate_cycle + 1,
                                       compute_overlap_window, exposure, bga_overlap, true);
    DrafMemoryEfficiencyStats draf_memory =
        buildDrafMemoryEfficiencyStats(draf, exposure,
                                       timing.draf_row_fetch_cycle +
                                           timing.draf_compute_trigger_cycle);

    double graph_like_penalty_score =
        clamp01(0.24 * shape.row_nnz_gini + 0.26 * shape.col_nnz_gini +
                0.20 * shape.low_nnz_column_ratio +
                0.16 * padding_sync_bound_score +
                0.14 * exposure.exposure_factor);

    double bga_cost_gate =
        clamp01(static_cast<double>(reuse_aware_bga_accumulate_cycle) / 30000.0);
    double mean_nnz_gate = clamp01(mean_nnz_per_col / 120.0);
    double shared_bga_gate =
        clamp01(static_cast<double>(bga_overlap.hidden_accumulate_cycle) / 18000.0);
    double shared_draf_gate =
        clamp01(static_cast<double>(draf_streaming.hidden_access_cycle) / 18000.0);
    double shared_overlap_gate =
        clamp01(0.55 * shared_bga_gate + 0.45 * shared_draf_gate);
    double bga_draf_heavy_evidence =
        clamp01(0.34 * bga_cost_gate + 0.26 * mean_nnz_gate +
                0.25 * shared_overlap_gate +
                0.15 * clamp01(draf_streaming.bga_window_occupancy / 0.45));
    double high_speedup_exemption =
        max(max(clamp01(mean_nnz_per_col / 90.0),
                clamp01(static_cast<double>(reuse_aware_bga_accumulate_cycle) / 18000.0)),
            max(clamp01(static_cast<double>(bga_overlap.hidden_accumulate_cycle) / 12000.0),
                clamp01(static_cast<double>(draf_streaming.hidden_access_cycle) / 12000.0)));
    double low_gini_regular =
        (1.0 - clamp01(shape.row_nnz_gini / 0.35)) *
        (1.0 - clamp01(shape.col_nnz_gini / 0.35));
    double phase_class_1_guard = phase_class_code == 1.0 ? 1.0 : 0.0;
    double tail_flat_guard = clamp01(1.0 - (critical.tail_skew_max_over_p95 - 1.0) / 0.05);
    double low_padding_sync = 1.0 - clamp01(padding_sync_bound_score / 0.18);
    double low_graph_like = 1.0 - graph_like_penalty_score;
    double low_mean_regular = 1.0 - clamp01(mean_nnz_per_col / 80.0);
    double low_overlap_regular = 1.0 - shared_overlap_gate;
    double insufficient_bga_draf_heavy =
        (1.0 - high_speedup_exemption) *
        (1.0 - 0.35 * bga_draf_heavy_evidence);
    double tail_flat_low_padding_guard =
        phase_class_1_guard * tail_flat_guard * low_padding_sync;
    double regular_false_positive_score =
        clamp01(phase_class_1_guard * low_graph_like *
                (0.22 * low_gini_regular + 0.20 * low_padding_sync +
                 0.16 * tail_flat_guard + 0.24 * insufficient_bga_draf_heavy +
                 0.12 * low_mean_regular + 0.06 * low_overlap_regular));
    double low_mean_phase1_guard =
        clamp01(phase_class_1_guard * low_graph_like *
                (1.0 - clamp01(mean_nnz_per_col / 70.0)) *
                (1.0 - high_speedup_exemption));

    double true_high_raw =
        clamp01(clamp01(bga_bound_score / 0.55) *
                (0.38 * high_speedup_exemption +
                 0.30 * bga_draf_heavy_evidence +
                 0.22 * shared_overlap_gate +
                 0.10 * bga_cost_gate));
    double true_high_speedup_candidate_score =
        clamp01(true_high_raw * (1.0 - 0.65 * graph_like_penalty_score) *
                (1.0 - 0.18 * regular_false_positive_score));
    double strict_regular_budget_guard =
        clamp01(phase_class_1_guard * low_graph_like *
                (1.0 - clamp01(true_high_speedup_candidate_score / 0.35)) *
                (0.50 * low_mean_regular + 0.50 * low_overlap_regular));

    double fragmented_pressure =
        clamp01(0.60 * exposure.fragmentation +
                0.25 * shape.draf_padding_pressure +
                0.15 * critical.critical_padding_ratio);
    double draf_serial_exposure =
        clamp01(0.25 + 0.30 * fragmented_pressure +
                0.25 * regular_false_positive_score +
                0.15 * graph_like_penalty_score -
                0.15 * true_high_speedup_candidate_score +
                0.12 * low_mean_phase1_guard +
                0.08 * strict_regular_budget_guard);
    uint64_t draf_parallel_cycle =
        max(timing.draf_row_fetch_cycle, timing.draf_compute_trigger_cycle);
    uint64_t draf_pipeline_overlap_demand =
        min(timing.draf_row_fetch_cycle, timing.draf_compute_trigger_cycle);
    uint64_t draf_path_cycle =
        draf_parallel_cycle +
        static_cast<uint64_t>(ceil(draf_serial_exposure *
                                   static_cast<double>(draf_pipeline_overlap_demand)));

    uint64_t bga_hidden_demand = bga_overlap.hidden_accumulate_cycle;
    uint64_t draf_hidden_demand = draf_streaming.hidden_access_cycle;
    uint64_t memory_saved_demand = draf_memory.saved_access_cycle;
    double draf_memory_long_stream_score =
        static_cast<double>(draf.draf_rows) /
        (static_cast<double>(draf.draf_rows) + 32768.0);
    uint64_t padding_saved_demand = static_cast<uint64_t>(
        floor(static_cast<double>(exposed_padding_cycle) *
              draf_memory.exposed_saving_factor * draf_memory_long_stream_score));
    uint64_t tail_saved_demand = static_cast<uint64_t>(
        floor(static_cast<double>(timing.draf_compute_trigger_cycle +
                                  exposed_padding_cycle) *
              clamp01(draf_memory.exposed_saving_factor *
                      (1.0 - percentile_tail_exposure_factor))));
    uint64_t saving_demand = bga_hidden_demand + draf_hidden_demand +
                             memory_saved_demand + padding_saved_demand +
                             tail_saved_demand;

    double budget_factor =
        clamp01(0.46 - 0.30 * regular_false_positive_score -
                0.25 * graph_like_penalty_score +
                0.28 * true_high_speedup_candidate_score +
                0.08 * fragmented_pressure);
    budget_factor *= (1.0 - 0.45 * low_mean_phase1_guard);
    budget_factor *= (1.0 - 0.25 * strict_regular_budget_guard);
    uint64_t saving_budget = static_cast<uint64_t>(
        floor(static_cast<double>(compute_overlap_window) * budget_factor));
    double soft_alpha = 0.12;
    if (graph_like_penalty_score > 0.60)
        soft_alpha = 0.02 + 0.03 * true_high_speedup_candidate_score;
    else if (true_high_speedup_candidate_score > 0.55)
        soft_alpha = 0.20 + 0.24 * true_high_speedup_candidate_score -
                     0.08 * regular_false_positive_score;
    else if (regular_false_positive_score > 0.55)
        soft_alpha = 0.03 + 0.07 * true_high_speedup_candidate_score;
    else
        soft_alpha = 0.06 + 0.18 * true_high_speedup_candidate_score -
                     0.04 * regular_false_positive_score -
                     0.05 * graph_like_penalty_score;
    soft_alpha *= (1.0 - 0.55 * low_mean_phase1_guard);
    soft_alpha *= (1.0 - 0.35 * strict_regular_budget_guard);
    soft_alpha = clamp01(soft_alpha);
    uint64_t effective_saving = saving_demand;
    if (saving_demand > saving_budget)
    {
        uint64_t excess = saving_demand - saving_budget;
        effective_saving =
            saving_budget +
            static_cast<uint64_t>(floor(static_cast<double>(excess) * soft_alpha));
    }
    effective_saving = min(effective_saving, saving_demand);

    uint64_t protected_bga_hidden = static_cast<uint64_t>(
        floor(static_cast<double>(bga_hidden_demand) *
              bga.near_duplicate_share * bga_overlap.reuse_score *
              bga_bound_score *
              (0.06 + 0.68 * true_high_speedup_candidate_score) *
              (1.0 - 0.80 * graph_like_penalty_score) *
              (1.0 - 0.25 * regular_false_positive_score) *
              (1.0 - 0.50 * low_mean_phase1_guard) *
              (1.0 - 0.40 * strict_regular_budget_guard)));
    uint64_t allocated_bga_hidden =
        proportionalGrant(effective_saving, saving_demand, bga_hidden_demand);
    uint64_t bounded_bga_hidden =
        min(bga_hidden_demand, max(allocated_bga_hidden, protected_bga_hidden));
    uint64_t remaining_effective =
        effective_saving > bounded_bga_hidden ? effective_saving - bounded_bga_hidden : 0;
    uint64_t remaining_demand =
        saving_demand > bga_hidden_demand ? saving_demand - bga_hidden_demand : 0;
    uint64_t bounded_draf_hidden =
        proportionalGrant(remaining_effective, remaining_demand, draf_hidden_demand);
    remaining_effective =
        remaining_effective > bounded_draf_hidden ? remaining_effective - bounded_draf_hidden : 0;
    remaining_demand =
        remaining_demand > draf_hidden_demand ? remaining_demand - draf_hidden_demand : 0;
    uint64_t bounded_memory_saved =
        proportionalGrant(remaining_effective, remaining_demand, memory_saved_demand);
    remaining_effective =
        remaining_effective > bounded_memory_saved ? remaining_effective - bounded_memory_saved : 0;
    remaining_demand =
        remaining_demand > memory_saved_demand ? remaining_demand - memory_saved_demand : 0;
    uint64_t bounded_padding_saved =
        proportionalGrant(remaining_effective, remaining_demand, padding_saved_demand);
    remaining_effective =
        remaining_effective > bounded_padding_saved ? remaining_effective - bounded_padding_saved : 0;
    uint64_t bounded_tail_saved = min(tail_saved_demand, remaining_effective);

    uint64_t graph_padding_penalty = static_cast<uint64_t>(
        ceil(static_cast<double>(exposed_padding_cycle) *
             graph_like_penalty_score * 0.14));
    uint64_t padding_path_cycle =
        exposed_padding_cycle > bounded_padding_saved
            ? exposed_padding_cycle - bounded_padding_saved
            : 0;
    padding_path_cycle += graph_padding_penalty;
    uint64_t bga_accumulate_cycle =
        raw_bga_accumulate_cycle > bounded_bga_hidden
            ? raw_bga_accumulate_cycle - bounded_bga_hidden
            : 0;
    uint64_t final_reduce_cycle =
        ceilDiv(bga.host_reduce_ops_after_bga, kConservativeHostReduceWidth);
    uint64_t total_cycle =
        timing.setup_cycle + draf_path_cycle + timing.pim_enable_cycle +
        padding_path_cycle + bga_accumulate_cycle + timing.pim_disable_cycle +
        timing.pim_to_sb_cycle + timing.bga_output_readback_cycle +
        final_reduce_cycle;
    total_cycle = total_cycle > bounded_memory_saved
                      ? total_cycle - bounded_memory_saved
                      : 0;
    total_cycle = total_cycle > bounded_tail_saved ? total_cycle - bounded_tail_saved : 0;

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
    result.setup_cycle = timing.setup_cycle;
    result.draf_row_fetch_cycle = timing.draf_row_fetch_cycle;
    result.draf_compute_trigger_cycle = timing.draf_compute_trigger_cycle;
    result.padding_cycle = padding_path_cycle;
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
    result.bga_raw_accumulate_cycle = raw_bga_accumulate_cycle;
    result.bga_hidden_accumulate_cycle = bounded_bga_hidden;
    result.bga_regularity_score = bga_overlap.regularity_score;
    result.bga_reuse_score = bga_overlap.reuse_score;
    result.bga_padding_guard = bga_overlap.padding_guard;
    result.bga_overlap_factor = bga_overlap.overlap_factor;
    result.draf_stream_hidden_cycle = bounded_draf_hidden;
    result.draf_stream_regularity_score = draf_streaming.regularity_score;
    result.draf_stream_guard_score = draf_streaming.guard_score;
    result.draf_access_share = draf_streaming.access_share;
    result.draf_stream_overlap_factor = draf_streaming.overlap_factor;
    result.draf_bga_window_occupancy = draf_streaming.bga_window_occupancy;
    result.draf_bga_contention_score = draf_streaming.bga_contention_score;
    result.draf_stream_budget_factor = draf_streaming.budget_factor;
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
    result.bga_critical_flushes = bga.max_stream_capacity_flushes_per_group;
    result.bga_reuse_aware_accumulate_cycle = reuse_aware_bga_accumulate_cycle;
    result.draf_memory_saved_cycle = bounded_memory_saved;
    result.draf_coo_bytes_per_nnz = draf_memory.coo_bytes_per_nnz;
    result.draf_bytes_per_nnz = draf_memory.draf_bytes_per_nnz;
    result.draf_vs_coo_memory_ratio = draf_memory.draf_vs_coo_ratio;
    result.draf_memory_saving_factor = draf_memory.memory_saving_factor;
    result.draf_exposed_memory_saving_factor = draf_memory.exposed_saving_factor;
    result.conservative_bga_floor_cycle = conservative_bga_floor_cycle;
    result.memory_budgeted_stream_hidden_cycle = bounded_draf_hidden;
    result.draf_memory_padding_saved_cycle = bounded_padding_saved;
    result.draf_memory_long_stream_score = draf_memory_long_stream_score;
    result.conservative_bga_floor_factor = conservative_bga_floor_factor;
    result.stream_memory_budget_factor = budget_factor;
    result.phase_class_code = phase_class_code;
    result.bga_bound_score = bga_bound_score;
    result.draf_access_bound_score = draf_access_bound_score;
    result.padding_sync_bound_score = padding_sync_bound_score;
    result.percentile_tail_exposure_factor = percentile_tail_exposure_factor;
    result.group_steps_p90 = critical.group_steps_p90;
    result.group_steps_p95 = critical.group_steps_p95;
    result.group_steps_max = critical.group_steps_max;
    result.tail_skew_max_over_p90 = critical.tail_skew_max_over_p90;
    result.tail_skew_max_over_p95 = critical.tail_skew_max_over_p95;
    result.p90_to_max_tail_steps = critical.p90_to_max_tail_steps;
    result.p95_to_max_tail_steps = critical.p95_to_max_tail_steps;
    result.shared_overlap_window_cycle = compute_overlap_window;
    result.shared_bga_overlap_cycle = bounded_bga_hidden;
    result.shared_draf_overlap_cycle = bounded_draf_hidden;
    result.draf_step_tail_saved_cycle = bounded_tail_saved;
    result.v14_draf_serial_exposure = draf_serial_exposure;
    result.v14_draf_path_cycle = draf_path_cycle;
    result.v14_regular_false_positive_score = regular_false_positive_score;
    result.v14_fragmented_pressure = fragmented_pressure;
    result.v14_tail_flat_low_padding_guard = tail_flat_low_padding_guard;
    result.v14_saving_budget_cycle = saving_budget;
    result.v14_saving_demand_cycle = saving_demand;
    result.v14_bounded_bga_hidden_cycle = bounded_bga_hidden;
    result.v14_bounded_draf_hidden_cycle = bounded_draf_hidden;
    result.v14_bounded_memory_saved_cycle = bounded_memory_saved;
    result.v14_bounded_padding_saved_cycle = bounded_padding_saved;
    result.v14_bounded_tail_saved_cycle = bounded_tail_saved;
    result.v14_budget_saturation =
        saving_budget == 0
            ? 0.0
            : static_cast<double>(min(saving_demand, saving_budget)) /
                  static_cast<double>(saving_budget);
    result.v16_soft_alpha = soft_alpha;
    result.v16_true_high_speedup_candidate_score = true_high_speedup_candidate_score;
    result.v16_regular_false_positive_score = regular_false_positive_score;
    result.v16_graph_like_penalty_score = graph_like_penalty_score;
    result.v16_saving_budget_cycle = saving_budget;
    result.v16_saving_demand_cycle = saving_demand;
    result.v16_effective_saving_cycle = effective_saving;
    result.v16_protected_bga_hidden_cycle = protected_bga_hidden;
    result.v16_final_bounded_bga_hidden_cycle = bounded_bga_hidden;
    result.v16_budget_saturation_ratio =
        saving_budget == 0
            ? 0.0
            : static_cast<double>(effective_saving) /
                  static_cast<double>(saving_budget);

    cout << "  latency_scope: setup + DRAF pipeline path + soft-bounded saving budget"
         << " + reuse-protected BGA + graph padding penalty" << endl;
    cout << "> draf_pipeline_path_cycle: " << draf_path_cycle
         << " serial_exposure=" << draf_serial_exposure
         << " true_high_speedup_candidate_score="
         << true_high_speedup_candidate_score
         << " regular_false_positive_score=" << regular_false_positive_score
         << " graph_like_penalty_score=" << graph_like_penalty_score << endl;
    cout << "> soft_saving_budget: budget=" << saving_budget
         << " demand=" << saving_demand
         << " effective=" << effective_saving
         << " soft_alpha=" << soft_alpha
         << " saturation_ratio=" << result.v16_budget_saturation_ratio << endl;
    cout << "> bga_accumulate_cycle: " << bga_accumulate_cycle
         << " raw=" << raw_bga_accumulate_cycle
         << " protected_bga_hidden=" << protected_bga_hidden
         << " final_bounded_bga_hidden=" << bounded_bga_hidden << endl;
    cout << "> padding_path_cycle: " << padding_path_cycle
         << " exposed_padding_cycle=" << exposed_padding_cycle
         << " bounded_padding_saved_cycle=" << bounded_padding_saved
         << " graph_padding_penalty=" << graph_padding_penalty << endl;
    cout << "> " << variant_spec.cycle_label << total_cycle
         << " ms=" << model_ms << endl;
    cout << "> " << variant_spec.speedup_label << model_speedup
         << " speedup_error_ratio=" << result.speedup_error_ratio << endl;
    cout << variant_spec.result_csv_tag
         << matrix_name << "," << gpu_ms << "," << target_speedup << ","
         << target_pim_ms << "," << model_ms << "," << model_speedup << ","
         << result.speedup_error_ratio << "," << draf.nze_padding << ","
         << draf.nze_padding_ratio << "," << draf.expansion_ratio << ","
         << critical.critical_padding << "," << critical.critical_padding_ratio
         << "," << critical.group_steps_mean << "," << critical.group_steps_max
         << "," << critical.bg_imbalance << "," << exposure.total_padding_steps
         << "," << exposure.hidden_padding_steps << "," << exposure.fragmentation
         << "," << exposure.memory_pressure << "," << exposure.imbalance_pressure
         << "," << exposure.exposure_factor << "," << exposure.exposed_hidden_padding
         << "," << exposure.effective_padding_steps << ","
         << raw_bga_accumulate_cycle << "," << bounded_bga_hidden << ","
         << bga_overlap.regularity_score << "," << bga_overlap.reuse_score
         << "," << bga_overlap.padding_guard << "," << bga_overlap.overlap_factor
         << "," << bounded_draf_hidden << "," << draf_streaming.regularity_score
         << "," << draf_streaming.guard_score << "," << draf_streaming.access_share
         << "," << draf_streaming.overlap_factor << ","
         << draf_streaming.bga_window_occupancy << ","
         << draf_streaming.bga_contention_score << ","
         << draf_streaming.budget_factor << "," << bga.near_duplicate_partial_ratio
         << "," << bga.far_duplicate_partial_ratio << ","
         << bga.near_duplicate_share << "," << bga.accumulator_flush_estimate
         << "," << bounded_memory_saved << "," << draf_memory.coo_bytes_per_nnz
         << "," << draf_memory.draf_bytes_per_nnz << ","
         << draf_memory.draf_vs_coo_ratio << ","
         << draf_memory.memory_saving_factor << ","
         << draf_memory.exposed_saving_factor << ","
         << conservative_bga_floor_cycle << "," << bounded_draf_hidden << ","
         << bounded_padding_saved << "," << draf_memory_long_stream_score << ","
         << conservative_bga_floor_factor << "," << budget_factor << ","
         << result.phase_class_code << "," << bga_bound_score << ","
         << draf_access_bound_score << "," << padding_sync_bound_score << ","
         << percentile_tail_exposure_factor << "," << critical.group_steps_p90
         << "," << critical.group_steps_p95 << "," << critical.group_steps_max
         << "," << critical.tail_skew_max_over_p90 << ","
         << critical.tail_skew_max_over_p95 << ","
         << critical.p90_to_max_tail_steps << ","
         << critical.p95_to_max_tail_steps << "," << compute_overlap_window
         << "," << bounded_bga_hidden << "," << bounded_draf_hidden << ","
         << bounded_tail_saved << "," << draf_serial_exposure << ","
         << draf_path_cycle << "," << regular_false_positive_score << ","
         << fragmented_pressure << "," << tail_flat_low_padding_guard << ","
         << saving_budget << "," << saving_demand << "," << bounded_bga_hidden
         << "," << bounded_draf_hidden << "," << bounded_memory_saved << ","
         << bounded_padding_saved << "," << bounded_tail_saved << ","
         << result.v14_budget_saturation << "," << soft_alpha << ","
         << true_high_speedup_candidate_score << ","
         << regular_false_positive_score << "," << graph_like_penalty_score
         << "," << saving_budget << "," << saving_demand << ","
         << effective_saving << "," << protected_bga_hidden << ","
         << bounded_bga_hidden << "," << result.v16_budget_saturation_ratio
         << endl;
    return result;
}

} // namespace spmv
