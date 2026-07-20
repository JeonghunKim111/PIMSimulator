#ifndef SPMV_DRAF_BGA_STRUCTURAL_TYPES_H
#define SPMV_DRAF_BGA_STRUCTURAL_TYPES_H

#include <cstdint>
#include <string>
#include <vector>

namespace spmv
{

struct ClusterInfo
{
    unsigned id = 0;
    uint64_t nnz = 0;
    uint64_t num_cols = 0;
    uint64_t active_rows = 0;
};

struct SpmvInputs
{
    uint64_t n_rows = 0;
    uint64_t n_cols = 0;
    uint64_t nnz = 0;
    std::vector<ClusterInfo> clusters;
    std::vector<int> reordered_col_to_cluster;
    std::vector<std::vector<unsigned>> rows_by_col;
    uint64_t total_active_row_memberships = 0;
    uint64_t rows_with_any_partial = 0;
};

struct DrafClusterInfo
{
    unsigned id = 0;
    uint64_t column_groups = 0;
    uint64_t draf_rows = 0;
    uint64_t nnz = 0;
    uint64_t active_rows = 0;
};

struct DrafStats
{
    std::vector<DrafClusterInfo> clusters;
    uint64_t column_groups = 0;
    uint64_t draf_rows = 0;
    uint64_t packed_nnz_capacity = 0;
    uint64_t nnz = 0;
    uint64_t active_row_memberships = 0;
    uint64_t rows_with_any_partial = 0;
    uint64_t nze_padding = 0;
    uint64_t row_group_padding = 0;
    double nze_padding_ratio = 0.0;
    double row_group_padding_ratio = 0.0;
    double expansion_ratio = 0.0;
};

struct BgaGroupInfo
{
    unsigned channel = 0;
    unsigned bank_group = 0;
    uint64_t partials_before = 0;
    uint64_t partials_after = 0;
    uint64_t bacc_instructions = 0;
    uint64_t estimated_flushes = 0;
    uint64_t simple_capacity_flushes = 0;
    uint64_t stream_capacity_flushes = 0;
    double row_reuse_factor = 0.0;
    double effective_partials = 0.0;
    uint64_t reuse_aware_bacc_instructions = 0;
    uint64_t reuse_aware_accumulate_cycle = 0;
    uint64_t near_duplicate_partials = 0;
    uint64_t far_duplicate_partials = 0;
    double near_duplicate_ratio = 0.0;
    double far_duplicate_ratio = 0.0;
};

struct BgaStats
{
    std::vector<BgaGroupInfo> groups;
    uint64_t partials_before = 0;
    uint64_t partials_after = 0;
    uint64_t bga_reduce_ops = 0;
    uint64_t bacc_instructions = 0;
    uint64_t estimated_flushes = 0;
    uint64_t max_bacc_instructions_per_group = 0;
    uint64_t max_estimated_flushes_per_group = 0;
    uint64_t simple_capacity_flushes = 0;
    uint64_t stream_capacity_flushes = 0;
    uint64_t max_simple_capacity_flushes_per_group = 0;
    uint64_t max_stream_capacity_flushes_per_group = 0;
    uint64_t max_reuse_aware_bacc_instructions_per_group = 0;
    uint64_t max_reuse_aware_accumulate_cycle_per_group = 0;
    uint64_t output_readback_tx = 0;
    uint64_t host_reduce_ops_after_bga = 0;
    double row_reuse_factor = 0.0;
    double unique_row_ratio = 0.0;
    double duplicate_partial_ratio = 0.0;
    double max_row_reuse_factor_per_group = 0.0;
    uint64_t near_duplicate_partials = 0;
    uint64_t far_duplicate_partials = 0;
    double near_duplicate_partial_ratio = 0.0;
    double far_duplicate_partial_ratio = 0.0;
    double near_duplicate_share = 0.0;
    uint64_t accumulator_flush_estimate = 0;
    uint64_t max_accumulator_flush_estimate_per_group = 0;
};

struct SpmvDataset
{
    std::string name;
    std::string base;
};

struct ShapeStats
{
    double row_nnz_cv = 0.0;
    double col_nnz_cv = 0.0;
    double row_nnz_gini = 0.0;
    double col_nnz_gini = 0.0;
    double max_row_nnz_over_mean = 0.0;
    double max_col_nnz_over_mean = 0.0;
    double hot_top1pct_row_nnz_ratio = 0.0;
    double bga_partials_imbalance = 0.0;
    double bga_unique_rows_imbalance = 0.0;
    double hot_bga_group_ratio = 0.0;
    double skew_score = 0.0;
    double bga_reduction_ratio = 0.0;
    double bga_row_reuse_factor = 0.0;
    double bga_unique_row_ratio = 0.0;
    double bga_duplicate_partial_ratio = 0.0;
    double bga_capacity_pressure = 0.0;
    double single_nnz_column_ratio = 0.0;
    double low_nnz_column_ratio = 0.0;
    double cluster_row_alignment_ratio = 0.0;
    double draf_row_alignment_ratio = 0.0;
    double sampled_cluster_jaccard = 0.0;
    double draf_padding_pressure = 0.0;
};

struct DrafCriticalPathStats
{
    double ideal_steps = 0.0;
    double actual_steps = 0.0;
    double critical_padding = 0.0;
    double critical_padding_ratio = 0.0;
    double group_steps_mean = 0.0;
    double group_steps_max = 0.0;
    double group_steps_p90 = 0.0;
    double group_steps_p95 = 0.0;
    double tail_exposure_ratio = 0.0;
    double tail_skew_max_over_p90 = 0.0;
    double tail_skew_max_over_p95 = 0.0;
    double p90_to_max_tail_steps = 0.0;
    double p95_to_max_tail_steps = 0.0;
    double bg_imbalance = 0.0;
};

struct DrafPaddingExposureStats
{
    double total_padding_steps = 0.0;
    double critical_padding_steps = 0.0;
    double hidden_padding_steps = 0.0;
    double fragmentation = 0.0;
    double memory_pressure = 0.0;
    double imbalance_pressure = 0.0;
    double exposure_factor = 0.0;
    double exposed_hidden_padding = 0.0;
    double effective_padding_steps = 0.0;
};

struct BgaOverlapStats
{
    uint64_t raw_accumulate_cycle = 0;
    uint64_t exposed_accumulate_cycle = 0;
    uint64_t hidden_accumulate_cycle = 0;
    double regularity_score = 0.0;
    double reuse_score = 0.0;
    double padding_guard = 0.0;
    double overlap_factor = 0.0;
};

struct DrafStreamingOverlapStats
{
    uint64_t hidden_access_cycle = 0;
    double regularity_score = 0.0;
    double guard_score = 0.0;
    double access_share = 0.0;
    double overlap_factor = 0.0;
    double bga_window_occupancy = 0.0;
    double bga_contention_score = 0.0;
    double budget_factor = 1.0;
};

struct DrafMemoryEfficiencyStats
{
    uint64_t saved_access_cycle = 0;
    double coo_bytes_per_nnz = 0.0;
    double draf_bytes_per_nnz = 0.0;
    double draf_vs_coo_ratio = 0.0;
    double memory_saving_factor = 0.0;
    double exposed_saving_factor = 0.0;
};

struct StructuralBaseTiming
{
    uint64_t setup_cycle = 0;
    uint64_t draf_row_fetch_cycle = 0;
    uint64_t pim_enable_cycle = 0;
    uint64_t draf_compute_trigger_cycle = 0;
    uint64_t pim_disable_cycle = 0;
    uint64_t pim_to_sb_cycle = 0;
    uint64_t bga_output_readback_cycle = 0;
};

struct StructuralModelResult
{
    std::string matrix;
    double gpu_ms = 0.0;
    double target_speedup = 0.0;
    double target_pim_ms = 0.0;
    double model_ms = 0.0;
    double model_speedup = 0.0;
    double speedup_error_ratio = 0.0;
    uint64_t total_cycle = 0;
    uint64_t setup_cycle = 0;
    uint64_t draf_row_fetch_cycle = 0;
    uint64_t draf_compute_trigger_cycle = 0;
    uint64_t padding_cycle = 0;
    uint64_t bga_accumulate_cycle = 0;
    uint64_t bga_output_readback_cycle = 0;
    uint64_t final_reduce_cycle = 0;
    double draf_padding_ratio = 0.0;
    double draf_memory_expansion = 0.0;
    double critical_padding = 0.0;
    double critical_padding_ratio = 0.0;
    double bg_imbalance = 0.0;
    double total_padding_steps = 0.0;
    double hidden_padding_steps = 0.0;
    double fragmentation = 0.0;
    double memory_pressure = 0.0;
    double imbalance_pressure = 0.0;
    double exposure_factor = 0.0;
    double exposed_hidden_padding = 0.0;
    double effective_padding_steps = 0.0;
    uint64_t bga_raw_accumulate_cycle = 0;
    uint64_t bga_hidden_accumulate_cycle = 0;
    double bga_regularity_score = 0.0;
    double bga_reuse_score = 0.0;
    double bga_padding_guard = 0.0;
    double bga_overlap_factor = 0.0;
    uint64_t draf_stream_hidden_cycle = 0;
    double draf_stream_regularity_score = 0.0;
    double draf_stream_guard_score = 0.0;
    double draf_access_share = 0.0;
    double draf_stream_overlap_factor = 0.0;
    double draf_bga_window_occupancy = 0.0;
    double draf_bga_contention_score = 0.0;
    double draf_stream_budget_factor = 1.0;
    double row_nnz_gini = 0.0;
    double col_nnz_gini = 0.0;
    double mean_nnz_per_col = 0.0;
    double single_nnz_column_ratio = 0.0;
    double low_nnz_column_ratio = 0.0;
    double bga_row_reuse_factor = 0.0;
    double bga_unique_row_ratio = 0.0;
    double bga_duplicate_partial_ratio = 0.0;
    double bga_near_duplicate_partial_ratio = 0.0;
    double bga_far_duplicate_partial_ratio = 0.0;
    double bga_near_duplicate_share = 0.0;
    uint64_t accumulator_flush_estimate = 0;
    double bga_capacity_pressure = 0.0;
    uint64_t bga_stream_capacity_flushes = 0;
    uint64_t bga_critical_flushes = 0;
    uint64_t bga_reuse_aware_accumulate_cycle = 0;
    uint64_t draf_memory_saved_cycle = 0;
    double draf_coo_bytes_per_nnz = 0.0;
    double draf_bytes_per_nnz = 0.0;
    double draf_vs_coo_memory_ratio = 0.0;
    double draf_memory_saving_factor = 0.0;
    double draf_exposed_memory_saving_factor = 0.0;
    uint64_t conservative_bga_floor_cycle = 0;
    uint64_t memory_budgeted_stream_hidden_cycle = 0;
    uint64_t draf_memory_padding_saved_cycle = 0;
    double draf_memory_long_stream_score = 0.0;
    double conservative_bga_floor_factor = 0.0;
    double stream_memory_budget_factor = 1.0;
    double phase_class_code = 0.0;
    double bga_bound_score = 0.0;
    double draf_access_bound_score = 0.0;
    double padding_sync_bound_score = 0.0;
    double percentile_tail_exposure_factor = 1.0;
    double group_steps_p90 = 0.0;
    double group_steps_p95 = 0.0;
    double group_steps_max = 0.0;
    double tail_skew_max_over_p90 = 0.0;
    double tail_skew_max_over_p95 = 0.0;
    double p90_to_max_tail_steps = 0.0;
    double p95_to_max_tail_steps = 0.0;
    uint64_t shared_overlap_window_cycle = 0;
    uint64_t shared_bga_overlap_cycle = 0;
    uint64_t shared_draf_overlap_cycle = 0;
    uint64_t draf_step_tail_saved_cycle = 0;
    double v14_draf_serial_exposure = 0.0;
    uint64_t v14_draf_path_cycle = 0;
    double v14_regular_false_positive_score = 0.0;
    double v14_fragmented_pressure = 0.0;
    double v14_tail_flat_low_padding_guard = 0.0;
    uint64_t v14_saving_budget_cycle = 0;
    uint64_t v14_saving_demand_cycle = 0;
    uint64_t v14_bounded_bga_hidden_cycle = 0;
    uint64_t v14_bounded_draf_hidden_cycle = 0;
    uint64_t v14_bounded_memory_saved_cycle = 0;
    uint64_t v14_bounded_padding_saved_cycle = 0;
    uint64_t v14_bounded_tail_saved_cycle = 0;
    double v14_budget_saturation = 0.0;
    double v15_soft_alpha = 0.0;
    double v15_true_high_speedup_candidate_score = 0.0;
    double v15_regular_false_positive_score = 0.0;
    double v15_graph_like_penalty_score = 0.0;
    uint64_t v15_saving_budget_cycle = 0;
    uint64_t v15_saving_demand_cycle = 0;
    uint64_t v15_effective_saving_cycle = 0;
    uint64_t v15_protected_bga_hidden_cycle = 0;
    uint64_t v15_final_bounded_bga_hidden_cycle = 0;
    double v15_budget_saturation_ratio = 0.0;
    double v16_soft_alpha = 0.0;
    double v16_true_high_speedup_candidate_score = 0.0;
    double v16_regular_false_positive_score = 0.0;
    double v16_graph_like_penalty_score = 0.0;
    uint64_t v16_saving_budget_cycle = 0;
    uint64_t v16_saving_demand_cycle = 0;
    uint64_t v16_effective_saving_cycle = 0;
    uint64_t v16_protected_bga_hidden_cycle = 0;
    uint64_t v16_final_bounded_bga_hidden_cycle = 0;
    double v16_budget_saturation_ratio = 0.0;
    double v17_soft_alpha = 0.0;
    double v17_true_high_speedup_candidate_score = 0.0;
    double v17_regular_false_positive_score = 0.0;
    double v17_graph_like_penalty_score = 0.0;
    double v17_regular_leakage_score = 0.0;
    double v17_high_confidence_protection_score = 0.0;
    double v17_regular_soft_alpha_cap = 0.0;
    uint64_t v17_saving_budget_cycle = 0;
    uint64_t v17_saving_demand_cycle = 0;
    uint64_t v17_effective_saving_cycle = 0;
    uint64_t v17_protected_bga_hidden_cycle = 0;
    uint64_t v17_final_bounded_bga_hidden_cycle = 0;
    double v17_budget_saturation_ratio = 0.0;
    double v17_1_soft_alpha = 0.0;
    double v17_1_true_high_speedup_candidate_score = 0.0;
    double v17_1_regular_false_positive_score = 0.0;
    double v17_1_graph_like_penalty_score = 0.0;
    double v17_1_regular_leakage_score = 0.0;
    double v17_1_high_confidence_protection_score = 0.0;
    double v17_1_regular_soft_alpha_cap = 0.0;
    uint64_t v17_1_saving_budget_cycle = 0;
    uint64_t v17_1_saving_demand_cycle = 0;
    uint64_t v17_1_effective_saving_cycle = 0;
    uint64_t v17_1_protected_bga_hidden_cycle = 0;
    uint64_t v17_1_final_bounded_bga_hidden_cycle = 0;
    double v17_1_budget_saturation_ratio = 0.0;
    double v18_soft_alpha = 0.0;
    double v18_true_high_speedup_candidate_score = 0.0;
    double v18_regular_false_positive_score = 0.0;
    double v18_graph_like_penalty_score = 0.0;
    double v18_regular_leakage_score = 0.0;
    double v18_high_confidence_protection_score = 0.0;
    double v18_regular_soft_alpha_cap = 0.0;
    double v18_graph_sync_gate = 0.0;
    double v18_graph_tail_gate = 0.0;
    uint64_t v18_graph_sync_exposure_cycle = 0;
    uint64_t v18_saving_budget_cycle = 0;
    uint64_t v18_saving_demand_cycle = 0;
    uint64_t v18_effective_saving_cycle = 0;
    uint64_t v18_protected_bga_hidden_cycle = 0;
    uint64_t v18_final_bounded_bga_hidden_cycle = 0;
    double v18_budget_saturation_ratio = 0.0;
    double bv4_requested_offload_ratio = 0.0;
    double bv4_actual_offload_ratio = 0.0;
    double bv4_offload_overshoot_ratio = 0.0;
    uint64_t bv4_setup_floor_penalty_cycle = 0;
    uint64_t bv4_offload_movement_penalty_cycle = 0;
    uint64_t bv4_merge_pressure_penalty_cycle = 0;
    double bv4_regular_setup_gate = 0.0;
    double bv4_movement_irregularity_factor = 0.0;
    double bv4_merge_pressure = 0.0;
};

} // namespace spmv

#endif
