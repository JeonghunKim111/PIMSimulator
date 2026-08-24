/***************************************************************************************************
 * SparsePIM-style clustered SpMV cycle-only benchmark.
 **************************************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Burst.h"
#include "MultiChannelMemorySystem.h"
#include "PIMCmd.h"
#include "gtest/gtest.h"
#include "tests/PIMCmdGen.h"
#include "tests/PIMKernel.h"
#include "tests/SpmvDrafBgaStructuralTypes.h"
#include "tests/SpmvDrafBgaStructuralCommon.h"
#include "tests/SpmvDrafBgaStructuralVariants.h"
#include "tests/SpmvDrafBgaStructuralBV1.h"
#include "tests/SpmvDrafBgaStructuralBV2.h"
#include "tests/SpmvDrafBgaStructuralBV3.h"
#include "tests/SpmvDrafBgaStructuralBV4.h"
#include "tests/SpmvDrafBgaStructuralV14.h"
#include "tests/SpmvDrafBgaStructuralV15.h"
#include "tests/SpmvDrafBgaStructuralV16.h"
#include "tests/SpmvDrafBgaStructuralV17.h"
#include "tests/SpmvDrafBgaStructuralV17_1.h"
#include "tests/SpmvDrafBgaStructuralV18.h"

using namespace DRAMSim;
using namespace spmv;
using namespace std;

namespace
{
constexpr unsigned kPimRegRow = 0x3fff;
constexpr unsigned kMacBaseRow = 128;
constexpr unsigned kResultBaseRow = 8192;
constexpr unsigned kResultReadBaseRow = 12288;
constexpr unsigned kConservativeHostReduceWidth = 16;

struct V2Params
{
    uint64_t bga_acc_capacity = kDefaultV2BgaAccCapacity;
    double skew_alpha = 0.5;
    double final_reduce_skew_alpha = 0.5;
    double padding_alpha = 0.2;
    double locality_beta = 0.3;
    double min_locality_factor = 0.6;
    uint64_t flush_penalty = kConservativeBgaFlushPenalty;
    bool use_stream_flush = false;
};

struct PaperTarget
{
    string workload;
    string matrix;
    double speedup = 0.0;
};


void addTx(shared_ptr<MultiChannelMemorySystem> mem, PIMAddrManager& addr_mgr, bool is_write,
           unsigned chan, unsigned bg, unsigned bank, unsigned row, unsigned col, BurstType* bst)
{
    uint64_t addr = addr_mgr.addrGenSafe(chan, 0, bg, bank, row, col);
    mem->addTransaction(is_write, addr, bst);
}

uint64_t drain(PIMKernel& kernel)
{
    uint64_t before = kernel.getCycle();
    kernel.runPIM();
    return kernel.getCycle() - before;
}

uint64_t envLimit(const char* name)
{
    const char* value = getenv(name);
    if (value == nullptr || value[0] == '\0')
        return 0;
    return strtoull(value, nullptr, 10);
}

string envString(const char* name)
{
    const char* value = getenv(name);
    return value == nullptr ? string() : string(value);
}

double envDouble(const char* name, double default_value)
{
    const char* value = getenv(name);
    if (value == nullptr || value[0] == '\0')
        return default_value;
    return strtod(value, nullptr);
}

uint64_t envUint64(const char* name, uint64_t default_value)
{
    const char* value = getenv(name);
    if (value == nullptr || value[0] == '\0')
        return default_value;
    return strtoull(value, nullptr, 10);
}

bool envBool(const char* name, bool default_value)
{
    const char* value = getenv(name);
    if (value == nullptr || value[0] == '\0')
        return default_value;
    return string(value) != "0";
}

V2Params loadV2Params()
{
    V2Params params;
    params.bga_acc_capacity = envUint64("SPMV_V2_BGA_ACC_CAPACITY", params.bga_acc_capacity);
    params.skew_alpha = envDouble("SPMV_V2_SKEW_ALPHA", params.skew_alpha);
    params.final_reduce_skew_alpha =
        envDouble("SPMV_V2_FINAL_REDUCE_SKEW_ALPHA", params.final_reduce_skew_alpha);
    params.padding_alpha = envDouble("SPMV_V2_PADDING_ALPHA", params.padding_alpha);
    params.locality_beta = envDouble("SPMV_V2_LOCALITY_BETA", params.locality_beta);
    params.min_locality_factor =
        envDouble("SPMV_V2_MIN_LOCALITY_FACTOR", params.min_locality_factor);
    params.flush_penalty = envUint64("SPMV_V2_FLUSH_PENALTY", params.flush_penalty);
    params.use_stream_flush = envBool("SPMV_V2_USE_STREAM_FLUSH", params.use_stream_flush);
    if (params.bga_acc_capacity == 0)
        params.bga_acc_capacity = 1;
    return params;
}

double paperTargetSpeedup(const string& matrix)
{
    static const vector<PaperTarget> targets{
        {"w1", "cant", 2.2},
        {"w2", "crankseg_2", 3.0},
        {"w3", "lhr71", 1.8},
        {"w4", "pdb1HYS", 5.4},
        {"w5", "rma10", 2.3},
        {"w6", "soc-sign-epinions", 1.2},
        {"w7", "Stanford", 1.3},
        {"w8", "bcsstk32", 2.0},
        {"w9", "consph", 2.3},
        {"w10", "ct20stif", 2.1},
        {"w11", "ohne2", 3.2},
        {"w12", "pwtk", 3.6},
        {"w13", "shipsec1", 2.1},
        {"w14", "ASIC_100k", 1.4},
        {"w15", "xenon2", 5.6},
        {"w16", "webbase-1M", 0.7},
    };
    for (const PaperTarget& target : targets)
    {
        if (target.matrix == matrix)
            return target.speedup;
    }
    return 0.0;
}

double gpuBaselineMs(const string& matrix)
{
    // Original-author GPU baseline latency in ms. Local GPU kernel-only
    // measurements are archived in SPARSEPIM_GPU_KERNEL_BASELINES_LOCAL.md.
    static const unordered_map<string, double> baselines{
        {"ASIC_100k", 1.264486},
        {"Stanford", 3.311456},
        {"bcsstk32", 1.1594},
        {"cant", 2.388768},
        {"consph", 3.705472},
        {"crankseg_2", 9.807726},
        {"ct20stif", 1.673094},
        {"lhr71", 1.90591},
        {"ohne2", 18.24784},
        {"pdb1HYS", 5.907898},
        {"pwtk", 11.52014},
        {"rma10", 2.682394},
        {"shipsec1", 4.43507},
        {"soc-sign-epinions", 1.187932},
        {"webbase-1M", 5.887198},
        {"xenon2", 11.67992},
    };
    auto it = baselines.find(matrix);
    return it == baselines.end() ? 0.0 : it->second;
}

double geomeanSpeedup(const vector<StructuralModelResult>& results,
                      function<double(const StructuralModelResult&)> value_fn)
{
    double sum_log = 0.0;
    uint64_t count = 0;
    for (const StructuralModelResult& result : results)
    {
        double value = value_fn(result);
        if (value <= 0.0)
            continue;
        sum_log += log(value);
        count++;
    }
    return count == 0 ? 0.0 : exp(sum_log / count);
}

void writeStructuralResults(int structural_variant,
                            const vector<StructuralModelResult>& results)
{
    string path = structuralResultPath(structural_variant);
    ofstream out(path);
    if (!out)
        throw runtime_error("failed to open " + path);

    out << "# SparsePIM DRAF+BGA structural model results\n";
    out << "model_variant: " << structuralVariantName(structural_variant) << "\n";
    out << "latency_scope: setup + DRAF access + padding model + BGA + result readback\n";
    out << "columns: matrix gpu_ms target_speedup target_pim_ms model_ms model_speedup "
           "speedup_error_ratio total_cycle setup_cycle draf_row_fetch_cycle "
           "draf_compute_trigger_cycle padding_cycle bga_accumulate_cycle "
           "bga_output_readback_cycle final_reduce_cycle draf_padding_ratio "
           "draf_memory_expansion critical_padding critical_padding_ratio bg_imbalance "
           "total_padding_steps hidden_padding_steps fragmentation memory_pressure "
           "imbalance_pressure exposure_factor exposed_hidden_padding effective_padding_steps "
           "bga_raw_accumulate_cycle bga_hidden_accumulate_cycle bga_regularity_score "
           "bga_reuse_score bga_padding_guard bga_overlap_factor "
           "draf_stream_hidden_cycle draf_stream_regularity_score "
           "draf_stream_guard_score draf_access_share draf_stream_overlap_factor "
           "draf_bga_window_occupancy draf_bga_contention_score "
           "draf_stream_budget_factor row_nnz_gini col_nnz_gini mean_nnz_per_col "
           "bga_row_reuse_factor bga_unique_row_ratio bga_duplicate_partial_ratio "
           "bga_near_duplicate_partial_ratio bga_far_duplicate_partial_ratio "
           "bga_near_duplicate_share accumulator_flush_estimate "
           "bga_capacity_pressure bga_stream_capacity_flushes bga_critical_flushes "
           "bga_reuse_aware_accumulate_cycle draf_memory_saved_cycle "
           "draf_coo_bytes_per_nnz draf_bytes_per_nnz draf_vs_coo_memory_ratio "
           "draf_memory_saving_factor draf_exposed_memory_saving_factor "
           "conservative_bga_floor_cycle memory_budgeted_stream_hidden_cycle "
           "draf_memory_padding_saved_cycle draf_memory_long_stream_score "
           "conservative_bga_floor_factor stream_memory_budget_factor "
           "phase_class_code bga_bound_score draf_access_bound_score "
           "padding_sync_bound_score percentile_tail_exposure_factor "
           "group_steps_p90 group_steps_p95 group_steps_max "
           "tail_skew_max_over_p90 tail_skew_max_over_p95 "
           "p90_to_max_tail_steps p95_to_max_tail_steps "
           "shared_overlap_window_cycle shared_bga_overlap_cycle "
           "shared_draf_overlap_cycle draf_step_tail_saved_cycle "
           "v14_draf_serial_exposure v14_draf_path_cycle "
           "v14_regular_false_positive_score v14_fragmented_pressure "
           "v14_tail_flat_low_padding_guard v14_saving_budget_cycle "
           "v14_saving_demand_cycle v14_bounded_bga_hidden_cycle "
           "v14_bounded_draf_hidden_cycle v14_bounded_memory_saved_cycle "
           "v14_bounded_padding_saved_cycle v14_bounded_tail_saved_cycle "
           "v14_budget_saturation v15_soft_alpha "
           "v15_true_high_speedup_candidate_score "
           "v15_regular_false_positive_score v15_graph_like_penalty_score "
           "v15_saving_budget_cycle v15_saving_demand_cycle "
           "v15_effective_saving_cycle v15_protected_bga_hidden_cycle "
           "v15_final_bounded_bga_hidden_cycle v15_budget_saturation_ratio "
           "v16_soft_alpha v16_true_high_speedup_candidate_score "
           "v16_regular_false_positive_score v16_graph_like_penalty_score "
           "v16_saving_budget_cycle v16_saving_demand_cycle "
           "v16_effective_saving_cycle v16_protected_bga_hidden_cycle "
           "v16_final_bounded_bga_hidden_cycle v16_budget_saturation_ratio "
           "v17_soft_alpha v17_true_high_speedup_candidate_score "
           "v17_regular_false_positive_score v17_graph_like_penalty_score "
           "v17_regular_leakage_score v17_high_confidence_protection_score "
           "v17_regular_soft_alpha_cap v17_saving_budget_cycle "
           "v17_saving_demand_cycle v17_effective_saving_cycle "
           "v17_protected_bga_hidden_cycle "
           "v17_final_bounded_bga_hidden_cycle v17_budget_saturation_ratio "
           "v17_1_soft_alpha v17_1_true_high_speedup_candidate_score "
           "v17_1_regular_false_positive_score v17_1_graph_like_penalty_score "
           "v17_1_regular_leakage_score "
           "v17_1_high_confidence_protection_score "
           "v17_1_regular_soft_alpha_cap v17_1_saving_budget_cycle "
           "v17_1_saving_demand_cycle v17_1_effective_saving_cycle "
           "v17_1_protected_bga_hidden_cycle "
           "v17_1_final_bounded_bga_hidden_cycle "
           "v17_1_budget_saturation_ratio "
           "v18_soft_alpha v18_true_high_speedup_candidate_score "
           "v18_regular_false_positive_score v18_graph_like_penalty_score "
           "v18_regular_leakage_score v18_high_confidence_protection_score "
           "v18_regular_soft_alpha_cap v18_graph_sync_gate "
           "v18_graph_tail_gate v18_graph_sync_exposure_cycle "
           "v18_saving_budget_cycle v18_saving_demand_cycle "
           "v18_effective_saving_cycle v18_protected_bga_hidden_cycle "
           "v18_final_bounded_bga_hidden_cycle v18_budget_saturation_ratio "
           "bv4_requested_offload_ratio bv4_actual_offload_ratio "
           "bv4_offload_overshoot_ratio bv4_setup_floor_penalty_cycle "
           "bv4_offload_movement_penalty_cycle "
           "bv4_merge_pressure_penalty_cycle bv4_regular_setup_gate "
           "bv4_movement_irregularity_factor bv4_merge_pressure\n";

    for (const StructuralModelResult& result : results)
    {
        out << result.matrix << " "
            << result.gpu_ms << " "
            << result.target_speedup << " "
            << result.target_pim_ms << " "
            << result.model_ms << " "
            << result.model_speedup << " "
            << result.speedup_error_ratio << " "
            << result.total_cycle << " "
            << result.setup_cycle << " "
            << result.draf_row_fetch_cycle << " "
            << result.draf_compute_trigger_cycle << " "
            << result.padding_cycle << " "
            << result.bga_accumulate_cycle << " "
            << result.bga_output_readback_cycle << " "
            << result.final_reduce_cycle << " "
            << result.draf_padding_ratio << " "
            << result.draf_memory_expansion << " "
            << result.critical_padding << " "
            << result.critical_padding_ratio << " "
            << result.bg_imbalance << " "
            << result.total_padding_steps << " "
            << result.hidden_padding_steps << " "
            << result.fragmentation << " "
            << result.memory_pressure << " "
            << result.imbalance_pressure << " "
            << result.exposure_factor << " "
            << result.exposed_hidden_padding << " "
            << result.effective_padding_steps << " "
            << result.bga_raw_accumulate_cycle << " "
            << result.bga_hidden_accumulate_cycle << " "
            << result.bga_regularity_score << " "
            << result.bga_reuse_score << " "
            << result.bga_padding_guard << " "
            << result.bga_overlap_factor << " "
            << result.draf_stream_hidden_cycle << " "
            << result.draf_stream_regularity_score << " "
            << result.draf_stream_guard_score << " "
            << result.draf_access_share << " "
            << result.draf_stream_overlap_factor << " "
            << result.draf_bga_window_occupancy << " "
            << result.draf_bga_contention_score << " "
            << result.draf_stream_budget_factor << " "
            << result.row_nnz_gini << " "
            << result.col_nnz_gini << " "
            << result.mean_nnz_per_col << " "
            << result.bga_row_reuse_factor << " "
            << result.bga_unique_row_ratio << " "
            << result.bga_duplicate_partial_ratio << " "
            << result.bga_near_duplicate_partial_ratio << " "
            << result.bga_far_duplicate_partial_ratio << " "
            << result.bga_near_duplicate_share << " "
            << result.accumulator_flush_estimate << " "
            << result.bga_capacity_pressure << " "
            << result.bga_stream_capacity_flushes << " "
            << result.bga_critical_flushes << " "
            << result.bga_reuse_aware_accumulate_cycle << " "
            << result.draf_memory_saved_cycle << " "
            << result.draf_coo_bytes_per_nnz << " "
            << result.draf_bytes_per_nnz << " "
            << result.draf_vs_coo_memory_ratio << " "
            << result.draf_memory_saving_factor << " "
            << result.draf_exposed_memory_saving_factor << " "
            << result.conservative_bga_floor_cycle << " "
            << result.memory_budgeted_stream_hidden_cycle << " "
            << result.draf_memory_padding_saved_cycle << " "
            << result.draf_memory_long_stream_score << " "
            << result.conservative_bga_floor_factor << " "
            << result.stream_memory_budget_factor << " "
            << result.phase_class_code << " "
            << result.bga_bound_score << " "
            << result.draf_access_bound_score << " "
            << result.padding_sync_bound_score << " "
            << result.percentile_tail_exposure_factor << " "
            << result.group_steps_p90 << " "
            << result.group_steps_p95 << " "
            << result.group_steps_max << " "
            << result.tail_skew_max_over_p90 << " "
            << result.tail_skew_max_over_p95 << " "
            << result.p90_to_max_tail_steps << " "
            << result.p95_to_max_tail_steps << " "
            << result.shared_overlap_window_cycle << " "
            << result.shared_bga_overlap_cycle << " "
            << result.shared_draf_overlap_cycle << " "
            << result.draf_step_tail_saved_cycle << " "
            << result.v14_draf_serial_exposure << " "
            << result.v14_draf_path_cycle << " "
            << result.v14_regular_false_positive_score << " "
            << result.v14_fragmented_pressure << " "
            << result.v14_tail_flat_low_padding_guard << " "
            << result.v14_saving_budget_cycle << " "
            << result.v14_saving_demand_cycle << " "
            << result.v14_bounded_bga_hidden_cycle << " "
            << result.v14_bounded_draf_hidden_cycle << " "
            << result.v14_bounded_memory_saved_cycle << " "
            << result.v14_bounded_padding_saved_cycle << " "
            << result.v14_bounded_tail_saved_cycle << " "
            << result.v14_budget_saturation << " "
            << result.v15_soft_alpha << " "
            << result.v15_true_high_speedup_candidate_score << " "
            << result.v15_regular_false_positive_score << " "
            << result.v15_graph_like_penalty_score << " "
            << result.v15_saving_budget_cycle << " "
            << result.v15_saving_demand_cycle << " "
            << result.v15_effective_saving_cycle << " "
            << result.v15_protected_bga_hidden_cycle << " "
            << result.v15_final_bounded_bga_hidden_cycle << " "
            << result.v15_budget_saturation_ratio << " "
            << result.v16_soft_alpha << " "
            << result.v16_true_high_speedup_candidate_score << " "
            << result.v16_regular_false_positive_score << " "
            << result.v16_graph_like_penalty_score << " "
            << result.v16_saving_budget_cycle << " "
            << result.v16_saving_demand_cycle << " "
            << result.v16_effective_saving_cycle << " "
            << result.v16_protected_bga_hidden_cycle << " "
            << result.v16_final_bounded_bga_hidden_cycle << " "
            << result.v16_budget_saturation_ratio << " "
            << result.v17_soft_alpha << " "
            << result.v17_true_high_speedup_candidate_score << " "
            << result.v17_regular_false_positive_score << " "
            << result.v17_graph_like_penalty_score << " "
            << result.v17_regular_leakage_score << " "
            << result.v17_high_confidence_protection_score << " "
            << result.v17_regular_soft_alpha_cap << " "
            << result.v17_saving_budget_cycle << " "
            << result.v17_saving_demand_cycle << " "
            << result.v17_effective_saving_cycle << " "
            << result.v17_protected_bga_hidden_cycle << " "
            << result.v17_final_bounded_bga_hidden_cycle << " "
            << result.v17_budget_saturation_ratio << " "
            << result.v17_1_soft_alpha << " "
            << result.v17_1_true_high_speedup_candidate_score << " "
            << result.v17_1_regular_false_positive_score << " "
            << result.v17_1_graph_like_penalty_score << " "
            << result.v17_1_regular_leakage_score << " "
            << result.v17_1_high_confidence_protection_score << " "
            << result.v17_1_regular_soft_alpha_cap << " "
            << result.v17_1_saving_budget_cycle << " "
            << result.v17_1_saving_demand_cycle << " "
            << result.v17_1_effective_saving_cycle << " "
            << result.v17_1_protected_bga_hidden_cycle << " "
            << result.v17_1_final_bounded_bga_hidden_cycle << " "
            << result.v17_1_budget_saturation_ratio << " "
            << result.v18_soft_alpha << " "
            << result.v18_true_high_speedup_candidate_score << " "
            << result.v18_regular_false_positive_score << " "
            << result.v18_graph_like_penalty_score << " "
            << result.v18_regular_leakage_score << " "
            << result.v18_high_confidence_protection_score << " "
            << result.v18_regular_soft_alpha_cap << " "
            << result.v18_graph_sync_gate << " "
            << result.v18_graph_tail_gate << " "
            << result.v18_graph_sync_exposure_cycle << " "
            << result.v18_saving_budget_cycle << " "
            << result.v18_saving_demand_cycle << " "
            << result.v18_effective_saving_cycle << " "
            << result.v18_protected_bga_hidden_cycle << " "
            << result.v18_final_bounded_bga_hidden_cycle << " "
            << result.v18_budget_saturation_ratio << " "
            << result.bv4_requested_offload_ratio << " "
            << result.bv4_actual_offload_ratio << " "
            << result.bv4_offload_overshoot_ratio << " "
            << result.bv4_setup_floor_penalty_cycle << " "
            << result.bv4_offload_movement_penalty_cycle << " "
            << result.bv4_merge_pressure_penalty_cycle << " "
            << result.bv4_regular_setup_gate << " "
            << result.bv4_movement_irregularity_factor << " "
            << result.bv4_merge_pressure << "\n";
    }

    out << "gmean_target_speedup: "
        << geomeanSpeedup(results,
                          [](const StructuralModelResult& result) {
                              return result.target_speedup;
                          })
        << "\n";
    out << "gmean_model_speedup: "
        << geomeanSpeedup(results,
                          [](const StructuralModelResult& result) {
                              return result.model_speedup;
                          })
        << "\n";
    out << "gmean_speedup_error_ratio: "
        << geomeanSpeedup(results,
                          [](const StructuralModelResult& result) {
                              return result.speedup_error_ratio;
                          })
        << "\n";
}

void writeStructuralDiagnostics(int structural_variant,
                                const vector<StructuralModelResult>& results)
{
    string path = structuralDiagnosticPath(structural_variant);
    ofstream out(path);
    if (!out)
        throw runtime_error("failed to open " + path);

    out << "# SparsePIM DRAF+BGA phase diagnostics\n";
    out << "model_variant: " << structuralVariantName(structural_variant) << "\n";
    out << "source_results: " << structuralResultPath(structural_variant) << "\n";
    out << "notes: alternative totals are diagnostic only, not adopted models\n";
    out << "columns: matrix target_speedup model_speedup model_over_target "
           "target_pim_ms model_ms excess_ms setup_pct fetch_pct compute_pct "
           "padding_pct bga_pct readback_pct final_pct mode_cycle "
           "compute_bga_overlap_ms compute_bga_overlap_speedup "
           "compute_bga_overlap_over_target aggressive_overlap_ms "
           "aggressive_overlap_speedup aggressive_overlap_over_target "
           "wide_final_ms wide_final_speedup wide_final_over_target "
           "compute_bga_overlap_wide_final_ms compute_bga_overlap_wide_final_speedup "
           "compute_bga_overlap_wide_final_over_target phase_class_code "
           "bga_bound_score draf_access_bound_score padding_sync_bound_score "
           "percentile_tail_exposure_factor shared_overlap_window_cycle "
           "shared_bga_overlap_cycle shared_draf_overlap_cycle "
           "draf_step_tail_saved_cycle group_steps_p90 group_steps_p95 "
           "group_steps_max tail_skew_max_over_p90 tail_skew_max_over_p95 "
           "p90_to_max_tail_steps p95_to_max_tail_steps "
           "bga_near_duplicate_partial_ratio bga_far_duplicate_partial_ratio "
           "bga_near_duplicate_share accumulator_flush_estimate\n";

    for (const StructuralModelResult& result : results)
    {
        uint64_t known_cycle = result.setup_cycle + result.draf_row_fetch_cycle +
                               result.draf_compute_trigger_cycle + result.padding_cycle +
                               result.bga_accumulate_cycle +
                               result.bga_output_readback_cycle +
                               result.final_reduce_cycle;
        known_cycle = known_cycle > result.draf_stream_hidden_cycle
                          ? known_cycle - result.draf_stream_hidden_cycle
                          : 0;
        uint64_t mode_cycle =
            result.total_cycle > known_cycle ? result.total_cycle - known_cycle : 0;
        uint64_t compute_path_cycle =
            result.draf_compute_trigger_cycle + result.padding_cycle;
        uint64_t compute_bga_overlap_cycle =
            result.setup_cycle + result.draf_row_fetch_cycle +
            max(compute_path_cycle, result.bga_accumulate_cycle) +
            result.bga_output_readback_cycle + result.final_reduce_cycle + mode_cycle;
        compute_bga_overlap_cycle =
            compute_bga_overlap_cycle > result.draf_stream_hidden_cycle
                ? compute_bga_overlap_cycle - result.draf_stream_hidden_cycle
                : 0;
        uint64_t aggressive_overlap_cycle =
            result.setup_cycle +
            max(result.draf_row_fetch_cycle,
                max(compute_path_cycle, result.bga_accumulate_cycle)) +
            result.bga_output_readback_cycle + result.final_reduce_cycle + mode_cycle;
        aggressive_overlap_cycle =
            aggressive_overlap_cycle > result.draf_stream_hidden_cycle
                ? aggressive_overlap_cycle - result.draf_stream_hidden_cycle
                : 0;
        uint64_t wide_final_cycle = ceilDiv(result.final_reduce_cycle, 4);
        uint64_t wide_final_total_cycle =
            result.total_cycle - result.final_reduce_cycle + wide_final_cycle;
        uint64_t overlap_wide_final_cycle =
            compute_bga_overlap_cycle - result.final_reduce_cycle + wide_final_cycle;

        double ms_per_cycle =
            result.total_cycle == 0 ? 0.0 : result.model_ms / result.total_cycle;
        auto cycleToMs = [ms_per_cycle](uint64_t cycle) {
            return static_cast<double>(cycle) * ms_per_cycle;
        };
        auto speedupFromMs = [&result](double ms) {
            return ms == 0.0 ? 0.0 : result.gpu_ms / ms;
        };
        auto overTarget = [&result](double speedup) {
            return result.target_speedup == 0.0 ? 0.0 : speedup / result.target_speedup;
        };
        auto pct = [&result](uint64_t cycle) {
            return result.total_cycle == 0
                       ? 0.0
                       : 100.0 * static_cast<double>(cycle) / result.total_cycle;
        };

        double compute_bga_overlap_ms = cycleToMs(compute_bga_overlap_cycle);
        double aggressive_overlap_ms = cycleToMs(aggressive_overlap_cycle);
        double wide_final_ms = cycleToMs(wide_final_total_cycle);
        double overlap_wide_final_ms = cycleToMs(overlap_wide_final_cycle);
        double compute_bga_overlap_speedup = speedupFromMs(compute_bga_overlap_ms);
        double aggressive_overlap_speedup = speedupFromMs(aggressive_overlap_ms);
        double wide_final_speedup = speedupFromMs(wide_final_ms);
        double overlap_wide_final_speedup = speedupFromMs(overlap_wide_final_ms);

        out << result.matrix << " "
            << result.target_speedup << " "
            << result.model_speedup << " "
            << result.speedup_error_ratio << " "
            << result.target_pim_ms << " "
            << result.model_ms << " "
            << (result.model_ms - result.target_pim_ms) << " "
            << pct(result.setup_cycle) << " "
            << pct(result.draf_row_fetch_cycle) << " "
            << pct(result.draf_compute_trigger_cycle) << " "
            << pct(result.padding_cycle) << " "
            << pct(result.bga_accumulate_cycle) << " "
            << pct(result.bga_output_readback_cycle) << " "
            << pct(result.final_reduce_cycle) << " "
            << mode_cycle << " "
            << compute_bga_overlap_ms << " "
            << compute_bga_overlap_speedup << " "
            << overTarget(compute_bga_overlap_speedup) << " "
            << aggressive_overlap_ms << " "
            << aggressive_overlap_speedup << " "
            << overTarget(aggressive_overlap_speedup) << " "
            << wide_final_ms << " "
            << wide_final_speedup << " "
            << overTarget(wide_final_speedup) << " "
            << overlap_wide_final_ms << " "
            << overlap_wide_final_speedup << " "
            << overTarget(overlap_wide_final_speedup) << " "
            << result.phase_class_code << " "
            << result.bga_bound_score << " "
            << result.draf_access_bound_score << " "
            << result.padding_sync_bound_score << " "
            << result.percentile_tail_exposure_factor << " "
            << result.shared_overlap_window_cycle << " "
            << result.shared_bga_overlap_cycle << " "
            << result.shared_draf_overlap_cycle << " "
            << result.draf_step_tail_saved_cycle << " "
            << result.group_steps_p90 << " "
            << result.group_steps_p95 << " "
            << result.group_steps_max << " "
            << result.tail_skew_max_over_p90 << " "
            << result.tail_skew_max_over_p95 << " "
            << result.p90_to_max_tail_steps << " "
            << result.p95_to_max_tail_steps << " "
            << result.bga_near_duplicate_partial_ratio << " "
            << result.bga_far_duplicate_partial_ratio << " "
            << result.bga_near_duplicate_share << " "
            << result.accumulator_flush_estimate << "\n";
    }
}

double pearsonCorrelation(const vector<double>& lhs, const vector<double>& rhs)
{
    if (lhs.size() != rhs.size() || lhs.size() < 2)
        return 0.0;
    double lhs_sum = 0.0;
    double rhs_sum = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i)
    {
        lhs_sum += lhs[i];
        rhs_sum += rhs[i];
    }
    double lhs_mean = lhs_sum / lhs.size();
    double rhs_mean = rhs_sum / rhs.size();
    double cov = 0.0;
    double lhs_var = 0.0;
    double rhs_var = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i)
    {
        double lhs_diff = lhs[i] - lhs_mean;
        double rhs_diff = rhs[i] - rhs_mean;
        cov += lhs_diff * rhs_diff;
        lhs_var += lhs_diff * lhs_diff;
        rhs_var += rhs_diff * rhs_diff;
    }
    if (lhs_var == 0.0 || rhs_var == 0.0)
        return 0.0;
    return cov / sqrt(lhs_var * rhs_var);
}

void writeStructuralResidualAnalysis(int structural_variant,
                                     const vector<StructuralModelResult>& results)
{
    string path = structuralResidualPath(structural_variant);
    ofstream out(path);
    if (!out)
        throw runtime_error("failed to open " + path);

    out << "# SparsePIM structural residual feature analysis\n";
    out << "model_variant: " << structuralVariantName(structural_variant) << "\n";
    out << "source_results: " << structuralResultPath(structural_variant) << "\n";
    out << "residual_definition: log(model_speedup) - log(target_speedup)\n";
    out << "notes: correlations are diagnostic only; model equations do not fit to targets\n";
    out << "columns: matrix residual_log model_speedup target_speedup "
           "mean_nnz_per_col row_nnz_gini col_nnz_gini single_nnz_column_ratio "
           "low_nnz_column_ratio draf_memory_expansion critical_padding_ratio "
           "exposure_factor bga_reduction_ratio bga_row_reuse_factor "
           "bga_unique_row_ratio bga_duplicate_partial_ratio "
           "bga_near_duplicate_partial_ratio bga_far_duplicate_partial_ratio "
           "bga_near_duplicate_share accumulator_flush_estimate bga_capacity_pressure "
           "bga_stream_capacity_flushes bga_critical_flushes "
           "bga_reuse_aware_accumulate_cycle draf_access_share "
           "draf_bga_window_occupancy draf_bga_contention_score "
           "draf_vs_coo_memory_ratio draf_exposed_memory_saving_factor "
           "phase_class_code bga_bound_score draf_access_bound_score "
           "padding_sync_bound_score percentile_tail_exposure_factor "
           "tail_skew_max_over_p90 tail_skew_max_over_p95 "
           "p90_to_max_tail_steps p95_to_max_tail_steps "
           "shared_bga_overlap_cycle shared_draf_overlap_cycle "
           "draf_step_tail_saved_cycle\n";

    vector<double> residuals;
    vector<pair<string, vector<double>>> features{
        {"mean_nnz_per_col", {}},
        {"row_nnz_gini", {}},
        {"col_nnz_gini", {}},
        {"single_nnz_column_ratio", {}},
        {"low_nnz_column_ratio", {}},
        {"draf_memory_expansion", {}},
        {"critical_padding_ratio", {}},
        {"exposure_factor", {}},
        {"bga_reduction_ratio", {}},
        {"bga_row_reuse_factor", {}},
        {"bga_unique_row_ratio", {}},
        {"bga_duplicate_partial_ratio", {}},
        {"bga_near_duplicate_partial_ratio", {}},
        {"bga_far_duplicate_partial_ratio", {}},
        {"bga_near_duplicate_share", {}},
        {"accumulator_flush_estimate", {}},
        {"bga_capacity_pressure", {}},
        {"bga_stream_capacity_flushes", {}},
        {"bga_critical_flushes", {}},
        {"bga_reuse_aware_accumulate_cycle", {}},
        {"draf_access_share", {}},
        {"draf_bga_window_occupancy", {}},
        {"draf_bga_contention_score", {}},
        {"draf_vs_coo_memory_ratio", {}},
        {"draf_exposed_memory_saving_factor", {}},
        {"phase_class_code", {}},
        {"bga_bound_score", {}},
        {"draf_access_bound_score", {}},
        {"padding_sync_bound_score", {}},
        {"percentile_tail_exposure_factor", {}},
        {"tail_skew_max_over_p90", {}},
        {"tail_skew_max_over_p95", {}},
        {"p90_to_max_tail_steps", {}},
        {"p95_to_max_tail_steps", {}},
        {"shared_bga_overlap_cycle", {}},
        {"shared_draf_overlap_cycle", {}},
        {"draf_step_tail_saved_cycle", {}},
    };

    for (const StructuralModelResult& result : results)
    {
        if (result.model_speedup <= 0.0 || result.target_speedup <= 0.0)
            continue;
        double residual = log(result.model_speedup) - log(result.target_speedup);
        residuals.push_back(residual);
        vector<double> values{
            result.mean_nnz_per_col,
            result.row_nnz_gini,
            result.col_nnz_gini,
            result.single_nnz_column_ratio,
            result.low_nnz_column_ratio,
            result.draf_memory_expansion,
            result.critical_padding_ratio,
            result.exposure_factor,
            result.bga_reuse_score,
            result.bga_row_reuse_factor,
            result.bga_unique_row_ratio,
            result.bga_duplicate_partial_ratio,
            result.bga_near_duplicate_partial_ratio,
            result.bga_far_duplicate_partial_ratio,
            result.bga_near_duplicate_share,
            static_cast<double>(result.accumulator_flush_estimate),
            result.bga_capacity_pressure,
            static_cast<double>(result.bga_stream_capacity_flushes),
            static_cast<double>(result.bga_critical_flushes),
            static_cast<double>(result.bga_reuse_aware_accumulate_cycle),
            result.draf_access_share,
            result.draf_bga_window_occupancy,
            result.draf_bga_contention_score,
            result.draf_vs_coo_memory_ratio,
            result.draf_exposed_memory_saving_factor,
            result.phase_class_code,
            result.bga_bound_score,
            result.draf_access_bound_score,
            result.padding_sync_bound_score,
            result.percentile_tail_exposure_factor,
            result.tail_skew_max_over_p90,
            result.tail_skew_max_over_p95,
            result.p90_to_max_tail_steps,
            result.p95_to_max_tail_steps,
            static_cast<double>(result.shared_bga_overlap_cycle),
            static_cast<double>(result.shared_draf_overlap_cycle),
            static_cast<double>(result.draf_step_tail_saved_cycle),
        };
        for (size_t i = 0; i < features.size(); ++i)
            features[i].second.push_back(values[i]);

        out << result.matrix << " "
            << residual << " "
            << result.model_speedup << " "
            << result.target_speedup << " "
            << result.mean_nnz_per_col << " "
            << result.row_nnz_gini << " "
            << result.col_nnz_gini << " "
            << result.single_nnz_column_ratio << " "
            << result.low_nnz_column_ratio << " "
            << result.draf_memory_expansion << " "
            << result.critical_padding_ratio << " "
            << result.exposure_factor << " "
            << result.bga_reuse_score << " "
            << result.bga_row_reuse_factor << " "
            << result.bga_unique_row_ratio << " "
            << result.bga_duplicate_partial_ratio << " "
            << result.bga_near_duplicate_partial_ratio << " "
            << result.bga_far_duplicate_partial_ratio << " "
            << result.bga_near_duplicate_share << " "
            << result.accumulator_flush_estimate << " "
            << result.bga_capacity_pressure << " "
            << result.bga_stream_capacity_flushes << " "
            << result.bga_critical_flushes << " "
            << result.bga_reuse_aware_accumulate_cycle << " "
            << result.draf_access_share << " "
            << result.draf_bga_window_occupancy << " "
            << result.draf_bga_contention_score << " "
            << result.draf_vs_coo_memory_ratio << " "
            << result.draf_exposed_memory_saving_factor << " "
            << result.phase_class_code << " "
            << result.bga_bound_score << " "
            << result.draf_access_bound_score << " "
            << result.padding_sync_bound_score << " "
            << result.percentile_tail_exposure_factor << " "
            << result.tail_skew_max_over_p90 << " "
            << result.tail_skew_max_over_p95 << " "
            << result.p90_to_max_tail_steps << " "
            << result.p95_to_max_tail_steps << " "
            << result.shared_bga_overlap_cycle << " "
            << result.shared_draf_overlap_cycle << " "
            << result.draf_step_tail_saved_cycle << "\n";
    }

    out << "\n# Pearson correlation with residual_log\n";
    out << "columns: feature pearson_r\n";
    for (const auto& feature : features)
        out << feature.first << " "
            << pearsonCorrelation(residuals, feature.second) << "\n";
}

void addPhaseBarriers(shared_ptr<MultiChannelMemorySystem> mem, const vector<char>& active_channels)
{
    for (size_t chan = 0; chan < active_channels.size(); ++chan)
    {
        if (active_channels[chan])
            mem->addBarrier(static_cast<int>(chan));
    }
}

class ClusteredSpmvBenchFixture : public testing::Test
{
  protected:
    void SetUp() override
    {
        resetPIMKernel();
    }

    void resetPIMKernel()
    {
        mem_ = make_shared<MultiChannelMemorySystem>("ini/HBM2_samsung_2M_16B_x64.ini",
                                                     "system_hbm.ini", ".", "spmv_bench",
                                                     256 * kSparsePimPseudoChannels);
        kernel_ = make_shared<PIMKernel>(mem_, kSparsePimPseudoChannels,
                                         kSparsePimRanks);
    }

    shared_ptr<MultiChannelMemorySystem> mem_;
    shared_ptr<PIMKernel> kernel_;
    void runDrafBgaModel(const string& base, const string& input_name,
                         bool conservative_model = false, bool v2_model = false,
                         bool v21_model = false);
    StructuralBaseTiming runDrafBgaStructuralBaseTiming(const DrafStats& draf,
                                                        const BgaStats& bga);
    StructuralModelResult runDrafBgaStructuralModel(const string& base,
                                                    const string& input_name,
                                                    const string& matrix_name,
                                                    int structural_variant = 3);
    StructuralModelResult runDrafBgaStructuralModelFromInputs(
        SpmvInputs inputs, const string& input_name, const string& matrix_name,
        int structural_variant = 3);
    void runGuidedKmeansDrafBgaSuite(bool conservative_model = false,
                                     bool v2_model = false, bool v21_model = false);
    void runGuidedKmeansDrafBgaStructuralSuite(int structural_variant = 3);
    void runNaiveCooRoundRobinBV4Suite();
};
}  // namespace

TEST(SpmvDrafBgaStructuralCommonTest, LoadsNaiveCooRoundRobinMapping)
{
    SpmvInputs inputs = loadNaiveCooInputs("src/tests/data/naive_coo_toy.txt",
                                           kSparsePimBankGroups);

    EXPECT_EQ(inputs.n_rows, 4);
    EXPECT_EQ(inputs.n_cols, 66);
    EXPECT_EQ(inputs.nnz, 4);
    ASSERT_EQ(inputs.clusters.size(), kSparsePimBankGroups);
    ASSERT_EQ(inputs.reordered_col_to_cluster.size(), 66);
    EXPECT_EQ(inputs.reordered_col_to_cluster[0], 0);
    EXPECT_EQ(inputs.reordered_col_to_cluster[64], 0);
    EXPECT_EQ(inputs.reordered_col_to_cluster[65], 1);
    EXPECT_EQ(inputs.clusters[0].num_cols, 2);
    EXPECT_EQ(inputs.clusters[1].num_cols, 2);
    EXPECT_EQ(inputs.clusters[0].nnz, 3);
    EXPECT_EQ(inputs.clusters[1].nnz, 1);
    EXPECT_EQ(inputs.clusters[0].active_rows, 3);
    EXPECT_EQ(inputs.clusters[1].active_rows, 1);
    EXPECT_EQ(inputs.rows_with_any_partial, 4);
    EXPECT_EQ(inputs.total_active_row_memberships, 4);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_cluster_cantcoo)
{
    const string base = "../SparsePIM/cluster_cantcoo/";
    SpmvInputs inputs = loadSparsePIMInputs(base + "reordered_matrix.txt",
                                            base + "column_permutation.txt", base + "clusters.txt");
    uint64_t max_clusters = envLimit("SPMV_BENCH_MAX_CLUSTERS");
    applyClusterLimit(inputs, max_clusters);

    BurstType null_bst;
    vector<PIMCmd> mac_cmds{
        PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK, 1),
        PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A, PIMOpdType::ODD_BANK, 1),
        PIMCmd(PIMCmdType::NOP, 7),
        PIMCmd(PIMCmdType::EXIT, 0),
    };

    uint64_t input_load_tx = 0;
    uint64_t mac_tx = 0;
    uint64_t partial_write_tx = 0;
    uint64_t partial_read_tx = 0;
    for (const ClusterInfo& cluster : inputs.clusters)
    {
        input_load_tx += ceilDiv(cluster.num_cols, kElementsPerBurst);
        mac_tx += ceilDiv(cluster.nnz, kElementsPerBurst);
        uint64_t result_bursts = ceilDiv(cluster.active_rows, kElementsPerBurst);
        partial_write_tx += result_bursts;
        partial_read_tx += result_bursts;
    }

    cout << ">>Clustered SpMV DRAMSim transaction plan" << endl;
    cout << "  input: SparsePIM/cluster_cantcoo" << endl;
    cout << "  rows: " << inputs.n_rows << endl;
    cout << "  cols: " << inputs.n_cols << endl;
    cout << "  nnz: " << inputs.nnz << endl;
    cout << "  clusters: " << inputs.clusters.size() << endl;
    if (max_clusters > 0)
        cout << "  SPMV_BENCH_MAX_CLUSTERS: " << max_clusters << endl;
    cout << "  planned_input_load_tx: " << input_load_tx << endl;
    cout << "  planned_mac_tx: " << mac_tx << endl;
    cout << "  planned_partial_write_tx: " << partial_write_tx << endl;
    cout << "  planned_partial_read_tx: " << partial_read_tx << endl;
    cout << "  note: cycle-accurate DRAMSim drain can take several minutes for full input"
         << endl;

    cout << "  phase: setup" << endl;
    kernel_->configurePIMControl();
    kernel_->parkIn();
    kernel_->changePIMMode(dramMode::SB, dramMode::HAB);
    kernel_->programCrf(mac_cmds);
    uint64_t excluded_setup_cycle = drain(*kernel_);

    cout << "  phase: input_load" << endl;
    vector<char> input_channels(kernel_->num_pim_chans_, 0);
    for (const ClusterInfo& cluster : inputs.clusters)
    {
        unsigned chan = cluster.id / 4;
        input_channels[chan] = 1;
        uint64_t bursts = ceilDiv(cluster.num_cols, kElementsPerBurst);
        for (uint64_t i = 0; i < bursts; ++i)
        {
            unsigned col = 0x8 + (i % 8);
            uint64_t addr = kernel_->pim_addr_mgr_->addrGen(chan, 0, 0, 1, kPimRegRow, col);
            mem_->addTransaction(true, addr, &null_bst);
        }
    }
    addPhaseBarriers(mem_, input_channels);
    uint64_t input_load_cycle = drain(*kernel_);

    cout << "  phase: pim_enable" << endl;
    kernel_->changePIMMode(dramMode::HAB, dramMode::HAB_PIM);
    uint64_t pim_enable_cycle = drain(*kernel_);

    cout << "  phase: pim_compute" << endl;
    vector<char> compute_channels(kernel_->num_pim_chans_, 0);
    for (const ClusterInfo& cluster : inputs.clusters)
    {
        unsigned chan = cluster.id / 4;
        uint64_t mac_bursts = ceilDiv(cluster.nnz, kElementsPerBurst);
        compute_channels[chan] = 1;

        for (uint64_t i = 0; i < mac_bursts; ++i)
        {
            unsigned row = kMacBaseRow + cluster.id;
            unsigned col = i;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, chan, 0, i % 2, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, compute_channels);

    for (const ClusterInfo& cluster : inputs.clusters)
    {
        unsigned chan = cluster.id / 4;
        uint64_t result_bursts = ceilDiv(cluster.active_rows, kElementsPerBurst);
        for (uint64_t i = 0; i < result_bursts; ++i)
        {
            unsigned row = kResultBaseRow + cluster.id;
            unsigned col = i;
            addTx(mem_, *kernel_->pim_addr_mgr_, true, chan, 0, 0, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, compute_channels);
    uint64_t pim_compute_cycle = drain(*kernel_);

    cout << "  phase: pim_disable" << endl;
    kernel_->changePIMMode(dramMode::HAB_PIM, dramMode::HAB);
    uint64_t pim_disable_cycle = drain(*kernel_);
    cout << "  phase: pim_to_sb" << endl;
    kernel_->changePIMMode(dramMode::HAB, dramMode::SB);
    uint64_t pim_to_sb_cycle = drain(*kernel_);

    cout << "  phase: partial_readback" << endl;
    vector<char> readback_channels(kernel_->num_pim_chans_, 0);
    for (const ClusterInfo& cluster : inputs.clusters)
    {
        unsigned chan = cluster.id / 4;
        uint64_t result_bursts = ceilDiv(cluster.active_rows, kElementsPerBurst);
        readback_channels[chan] = 1;
        for (uint64_t i = 0; i < result_bursts; ++i)
        {
            unsigned row = kResultReadBaseRow + cluster.id;
            unsigned col = i;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, chan, 0, 0, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, readback_channels);
    uint64_t partial_readback_cycle = drain(*kernel_);

    uint64_t host_reduce_ops =
        inputs.total_active_row_memberships > inputs.rows_with_any_partial
            ? inputs.total_active_row_memberships - inputs.rows_with_any_partial
            : 0;

    cout << ">>Clustered SpMV Per-Run Cycle Model" << endl;
    cout << "  input: SparsePIM/cluster_cantcoo" << endl;
    cout << "  rows: " << inputs.n_rows << endl;
    cout << "  cols: " << inputs.n_cols << endl;
    cout << "  nnz: " << inputs.nnz << endl;
    cout << "  clusters: " << inputs.clusters.size() << endl;
    cout << "  mapping: cluster_id -> channel=floor(id/4), bank_group=id%4" << endl;
    cout << "  simulator_physical_pim_bg: 0" << endl;
    cout << "  excluded_setup_cycle: " << excluded_setup_cycle << endl;
    cout << "> input_load_cycle: " << input_load_cycle << " tx=" << input_load_tx << endl;
    cout << "> pim_enable_cycle: " << pim_enable_cycle << endl;
    cout << "> pim_compute_cycle: " << pim_compute_cycle << " mac_tx=" << mac_tx
         << " partial_write_tx=" << partial_write_tx << endl;
    cout << "> pim_disable_cycle: " << pim_disable_cycle << endl;
    cout << "> pim_to_sb_cycle: " << pim_to_sb_cycle << endl;
    cout << "> partial_readback_cycle: " << partial_readback_cycle
         << " tx=" << partial_read_tx << endl;
    cout << "> host_reduce_ops: " << host_reduce_ops << endl;
    cout << "> per_spmv_simulated_cycle: "
         << input_load_cycle + pim_enable_cycle + pim_compute_cycle + pim_disable_cycle +
                pim_to_sb_cycle + partial_readback_cycle
         << endl;
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_cluster_cantcoo_draf_model)
{
    const string base = "../SparsePIM/cluster_cantcoo/";
    SpmvInputs inputs = loadSparsePIMInputs(base + "reordered_matrix.txt",
                                            base + "column_permutation.txt", base + "clusters.txt");
    uint64_t max_clusters = envLimit("SPMV_BENCH_MAX_CLUSTERS");
    applyClusterLimit(inputs, max_clusters);
    DrafStats draf = buildDrafStats(inputs);

    BurstType null_bst;
    vector<PIMCmd> mac_cmds{
        PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK, 1),
        PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A, PIMOpdType::ODD_BANK, 1),
        PIMCmd(PIMCmdType::NOP, 7),
        PIMCmd(PIMCmdType::EXIT, 0),
    };

    cout << ">>DRAF-aware SpMV synthetic transaction plan" << endl;
    cout << "  model: DRAF-aware synthetic DRAMSim model" << endl;
    cout << "  note: not a bit-accurate DRAF memory image" << endl;
    cout << "  input: SparsePIM/cluster_cantcoo" << endl;
    cout << "  rows: " << inputs.n_rows << endl;
    cout << "  cols: " << inputs.n_cols << endl;
    cout << "  nnz: " << draf.nnz << endl;
    cout << "  clusters: " << inputs.clusters.size() << endl;
    if (max_clusters > 0)
        cout << "  SPMV_BENCH_MAX_CLUSTERS: " << max_clusters << endl;
    cout << "  draf_column_group_capacity_nnz: " << kDrafNzesPerColumnGroup << endl;
    cout << "  draf_column_groups_per_row: " << kDrafColumnGroupsPerRow << endl;
    cout << "  draf_column_groups: " << draf.column_groups << endl;
    cout << "  draf_rows: " << draf.draf_rows << endl;
    cout << "  packed_nnz_capacity: " << draf.packed_nnz_capacity << endl;
    cout << "  planned_draf_row_fetch_tx: " << draf.draf_rows << endl;
    cout << "  planned_draf_compute_trigger_tx: " << draf.draf_rows << endl;
    cout << "  planned_draf_partial_write_tx: " << draf.draf_rows << endl;
    cout << "  planned_draf_partial_read_tx: " << draf.draf_rows << endl;

    cout << "  phase: setup" << endl;
    kernel_->configurePIMControl();
    kernel_->parkIn();
    kernel_->changePIMMode(dramMode::SB, dramMode::HAB);
    kernel_->programCrf(mac_cmds);
    uint64_t excluded_setup_cycle = drain(*kernel_);

    cout << "  phase: draf_row_fetch" << endl;
    vector<char> draf_channels(kernel_->num_pim_chans_, 0);
    uint64_t global_draf_row = 0;
    for (const DrafClusterInfo& cluster : draf.clusters)
    {
        unsigned chan = cluster.id / 4;
        draf_channels[chan] = cluster.draf_rows > 0 ? 1 : draf_channels[chan];
        for (uint64_t i = 0; i < cluster.draf_rows; ++i, ++global_draf_row)
        {
            unsigned row = kMacBaseRow + (global_draf_row / 32);
            unsigned col = global_draf_row % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, chan, 0, 0, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, draf_channels);
    uint64_t draf_row_fetch_cycle = drain(*kernel_);

    cout << "  phase: pim_enable" << endl;
    kernel_->changePIMMode(dramMode::HAB, dramMode::HAB_PIM);
    uint64_t pim_enable_cycle = drain(*kernel_);

    cout << "  phase: draf_compute" << endl;
    global_draf_row = 0;
    for (const DrafClusterInfo& cluster : draf.clusters)
    {
        unsigned chan = cluster.id / 4;
        for (uint64_t i = 0; i < cluster.draf_rows; ++i, ++global_draf_row)
        {
            unsigned row = kMacBaseRow + (global_draf_row / 32);
            unsigned col = global_draf_row % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, chan, 0, i % 2, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, draf_channels);

    global_draf_row = 0;
    for (const DrafClusterInfo& cluster : draf.clusters)
    {
        unsigned chan = cluster.id / 4;
        for (uint64_t i = 0; i < cluster.draf_rows; ++i, ++global_draf_row)
        {
            unsigned row = kResultBaseRow + (global_draf_row / 32);
            unsigned col = global_draf_row % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, true, chan, 0, 0, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, draf_channels);
    uint64_t draf_compute_cycle = drain(*kernel_);

    cout << "  phase: pim_disable" << endl;
    kernel_->changePIMMode(dramMode::HAB_PIM, dramMode::HAB);
    uint64_t pim_disable_cycle = drain(*kernel_);
    cout << "  phase: pim_to_sb" << endl;
    kernel_->changePIMMode(dramMode::HAB, dramMode::SB);
    uint64_t pim_to_sb_cycle = drain(*kernel_);

    cout << "  phase: draf_partial_readback" << endl;
    global_draf_row = 0;
    for (const DrafClusterInfo& cluster : draf.clusters)
    {
        unsigned chan = cluster.id / 4;
        for (uint64_t i = 0; i < cluster.draf_rows; ++i, ++global_draf_row)
        {
            unsigned row = kResultReadBaseRow + (global_draf_row / 32);
            unsigned col = global_draf_row % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, chan, 0, 0, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, draf_channels);
    uint64_t draf_partial_readback_cycle = drain(*kernel_);

    uint64_t host_reduce_ops =
        draf.active_row_memberships > draf.rows_with_any_partial
            ? draf.active_row_memberships - draf.rows_with_any_partial
            : 0;

    cout << ">>DRAF-aware SpMV Per-Run Cycle Model" << endl;
    cout << "  input: SparsePIM/cluster_cantcoo" << endl;
    cout << "  rows: " << inputs.n_rows << endl;
    cout << "  cols: " << inputs.n_cols << endl;
    cout << "  nnz: " << draf.nnz << endl;
    cout << "  clusters: " << inputs.clusters.size() << endl;
    cout << "  draf_column_groups: " << draf.column_groups << endl;
    cout << "  draf_rows: " << draf.draf_rows << endl;
    cout << "  simulator_physical_pim_bg: 0" << endl;
    cout << "  excluded_setup_cycle: " << excluded_setup_cycle << endl;
    cout << "> draf_row_fetch_cycle: " << draf_row_fetch_cycle
         << " tx=" << draf.draf_rows << endl;
    cout << "> pim_enable_cycle: " << pim_enable_cycle << endl;
    cout << "> draf_compute_cycle: " << draf_compute_cycle
         << " compute_trigger_tx=" << draf.draf_rows
         << " partial_write_tx=" << draf.draf_rows << endl;
    cout << "> pim_disable_cycle: " << pim_disable_cycle << endl;
    cout << "> pim_to_sb_cycle: " << pim_to_sb_cycle << endl;
    cout << "> draf_partial_readback_cycle: " << draf_partial_readback_cycle
         << " tx=" << draf.draf_rows << endl;
    cout << "> host_reduce_ops: " << host_reduce_ops << endl;
    cout << "> per_spmv_draf_model_cycle: "
         << draf_row_fetch_cycle + pim_enable_cycle + draf_compute_cycle + pim_disable_cycle +
                pim_to_sb_cycle + draf_partial_readback_cycle
         << endl;
}

void ClusteredSpmvBenchFixture::runDrafBgaModel(const string& base, const string& input_name,
                                                bool conservative_model, bool v2_model,
                                                bool v21_model)
{
    if (v21_model)
        v2_model = true;
    SpmvInputs inputs = loadSparsePIMInputs(base + "reordered_matrix.txt",
                                            base + "column_permutation.txt", base + "clusters.txt");
    uint64_t max_clusters = envLimit("SPMV_BENCH_MAX_CLUSTERS");
    applyClusterLimit(inputs, max_clusters);
    V2Params v2_params = loadV2Params();
    DrafStats draf = buildDrafStats(inputs);
    BgaStats bga = buildBgaStats(inputs, v2_params.bga_acc_capacity);
    ShapeStats shape = buildShapeStats(inputs, draf, bga);

    BurstType null_bst;
    vector<PIMCmd> mac_cmds{
        PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK, 1),
        PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A, PIMOpdType::ODD_BANK, 1),
        PIMCmd(PIMCmdType::NOP, 7),
        PIMCmd(PIMCmdType::EXIT, 0),
    };

    cout << ">>DRAF+BGA-aware SpMV synthetic transaction plan" << endl;
    cout << "  model: "
         << (v21_model ? "DRAF+BGA-aware v2.1 correction synthetic DRAMSim model"
                       : (v2_model ? "DRAF+BGA-aware v2 correction synthetic DRAMSim model"
                      : (conservative_model
                             ? "DRAF+BGA-aware conservative synthetic DRAMSim model"
                             : "DRAF+BGA-aware synthetic DRAMSim model")))
         << endl;
    cout << "  note: not a bit-accurate DRAF/BGA microarchitecture" << endl;
    cout << "  input: " << input_name << endl;
    cout << "  rows: " << inputs.n_rows << endl;
    cout << "  cols: " << inputs.n_cols << endl;
    cout << "  nnz: " << draf.nnz << endl;
    cout << "  clusters: " << inputs.clusters.size() << endl;
    if (max_clusters > 0)
        cout << "  SPMV_BENCH_MAX_CLUSTERS: " << max_clusters << endl;
    cout << "  draf_column_group_capacity_nnz: " << kDrafNzesPerColumnGroup << endl;
    cout << "  draf_column_groups_per_row: " << kDrafColumnGroupsPerRow << endl;
    cout << "  bga_entries_per_bacc: " << kBgaEntriesPerBacc << endl;
    cout << "  bga_queue_depth: " << kBgaQueueDepth << endl;
    cout << "  bga_flush_threshold: " << kBgaFlushThreshold << endl;
    if (conservative_model)
    {
        cout << "  conservative_bga_flush_penalty: " << kConservativeBgaFlushPenalty << endl;
        cout << "  conservative_host_reduce_width: " << kConservativeHostReduceWidth << endl;
    }
    if (v2_model)
    {
        cout << "  v2_bga_acc_capacity: " << v2_params.bga_acc_capacity << endl;
        cout << "  v2_use_stream_flush: " << (v2_params.use_stream_flush ? 1 : 0) << endl;
        cout << "  v2_skew_alpha: " << v2_params.skew_alpha << endl;
        cout << "  v2_final_reduce_skew_alpha: " << v2_params.final_reduce_skew_alpha
             << endl;
        cout << "  v2_padding_alpha: " << v2_params.padding_alpha << endl;
        cout << "  v2_locality_beta: " << v2_params.locality_beta << endl;
        cout << "  v2_min_locality_factor: " << v2_params.min_locality_factor << endl;
        cout << "  v2_flush_penalty: " << v2_params.flush_penalty << endl;
    }
    cout << "  draf_column_groups: " << draf.column_groups << endl;
    cout << "  draf_rows: " << draf.draf_rows << endl;
    cout << "  draf_expansion_ratio: " << draf.expansion_ratio << endl;
    cout << "  draf_nze_padding_ratio: " << draf.nze_padding_ratio << endl;
    cout << "  draf_row_group_padding_ratio: " << draf.row_group_padding_ratio << endl;
    cout << "  active_bga_groups: " << bga.groups.size() << endl;
    cout << "  bga_partials_before: " << bga.partials_before << endl;
    cout << "  bga_partials_after: " << bga.partials_after << endl;
    cout << "  bga_reduce_ops: " << bga.bga_reduce_ops << endl;
    cout << "  bacc_instructions: " << bga.bacc_instructions << endl;
    cout << "  max_bacc_instructions_per_group: " << bga.max_bacc_instructions_per_group
         << endl;
    cout << "  estimated_bga_flushes: " << bga.estimated_flushes << endl;
    cout << "  max_estimated_flushes_per_group: " << bga.max_estimated_flushes_per_group
         << endl;
    cout << "  simple_capacity_flushes: " << bga.simple_capacity_flushes << endl;
    cout << "  max_simple_capacity_flushes_per_group: "
         << bga.max_simple_capacity_flushes_per_group << endl;
    cout << "  stream_capacity_flushes: " << bga.stream_capacity_flushes << endl;
    cout << "  max_stream_capacity_flushes_per_group: "
         << bga.max_stream_capacity_flushes_per_group << endl;
    cout << "  bga_output_readback_tx: " << bga.output_readback_tx << endl;
    cout << "  host_reduce_ops_after_bga: " << bga.host_reduce_ops_after_bga << endl;
    if (v2_model)
    {
        cout << "  row_nnz_cv: " << shape.row_nnz_cv << endl;
        cout << "  col_nnz_cv: " << shape.col_nnz_cv << endl;
        cout << "  row_nnz_gini: " << shape.row_nnz_gini << endl;
        cout << "  col_nnz_gini: " << shape.col_nnz_gini << endl;
        cout << "  max_row_nnz_over_mean: " << shape.max_row_nnz_over_mean << endl;
        cout << "  max_col_nnz_over_mean: " << shape.max_col_nnz_over_mean << endl;
        cout << "  hot_top1pct_row_nnz_ratio: " << shape.hot_top1pct_row_nnz_ratio << endl;
        cout << "  bga_partials_imbalance: " << shape.bga_partials_imbalance << endl;
        cout << "  bga_unique_rows_imbalance: " << shape.bga_unique_rows_imbalance << endl;
        cout << "  hot_bga_group_ratio: " << shape.hot_bga_group_ratio << endl;
        cout << "  bga_reduction_ratio: " << shape.bga_reduction_ratio << endl;
        cout << "  single_nnz_column_ratio: " << shape.single_nnz_column_ratio << endl;
        cout << "  low_nnz_column_ratio: " << shape.low_nnz_column_ratio << endl;
        cout << "  cluster_row_alignment_ratio: " << shape.cluster_row_alignment_ratio << endl;
        cout << "  draf_row_alignment_ratio: " << shape.draf_row_alignment_ratio << endl;
        cout << "  sampled_cluster_jaccard: " << shape.sampled_cluster_jaccard << endl;
        cout << "  draf_padding_pressure: " << shape.draf_padding_pressure << endl;
        cout << "  v2_skew_score: " << shape.skew_score << endl;
    }

    cout << "  phase: setup" << endl;
    kernel_->configurePIMControl();
    kernel_->parkIn();
    kernel_->changePIMMode(dramMode::SB, dramMode::HAB);
    kernel_->programCrf(mac_cmds);
    uint64_t excluded_setup_cycle = drain(*kernel_);

    cout << "  phase: draf_row_fetch" << endl;
    vector<char> draf_channels(kernel_->num_pim_chans_, 0);
    uint64_t global_draf_row = 0;
    for (const DrafClusterInfo& cluster : draf.clusters)
    {
        unsigned chan = cluster.id / 4;
        draf_channels[chan] = cluster.draf_rows > 0 ? 1 : draf_channels[chan];
        for (uint64_t i = 0; i < cluster.draf_rows; ++i, ++global_draf_row)
        {
            unsigned row = kMacBaseRow + (global_draf_row / 32);
            unsigned col = global_draf_row % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, chan, 0, 0, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, draf_channels);
    uint64_t draf_row_fetch_cycle = drain(*kernel_);

    cout << "  phase: pim_enable" << endl;
    kernel_->changePIMMode(dramMode::HAB, dramMode::HAB_PIM);
    uint64_t pim_enable_cycle = drain(*kernel_);

    cout << "  phase: draf_compute" << endl;
    global_draf_row = 0;
    for (const DrafClusterInfo& cluster : draf.clusters)
    {
        unsigned chan = cluster.id / 4;
        for (uint64_t i = 0; i < cluster.draf_rows; ++i, ++global_draf_row)
        {
            unsigned row = kMacBaseRow + (global_draf_row / 32);
            unsigned col = global_draf_row % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, chan, 0, i % 2, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, draf_channels);
    uint64_t draf_compute_cycle = drain(*kernel_);

    cout << "  phase: bga_accumulate" << endl;
    uint64_t bga_accumulate_cycle =
        bga.max_bacc_instructions_per_group +
        bga.max_estimated_flushes_per_group *
            (conservative_model ? kConservativeBgaFlushPenalty : 1);

    /*
     * BGA is modeled as an internal bank-group accumulator. Active BGAs run in parallel and
     * reuse the idle SIMD adders during BACC, so this phase is not emitted as serialized DRAM
     * traffic. The synthetic cycle is the longest per-BGA BACC stream plus its flush penalty.
     */
    vector<char> bga_channels(kernel_->num_pim_chans_, 0);
    for (size_t group_idx = 0; group_idx < bga.groups.size(); ++group_idx)
    {
        const BgaGroupInfo& group = bga.groups[group_idx];
        bga_channels[group.channel] = 1;
    }

    cout << "  phase: pim_disable" << endl;
    kernel_->changePIMMode(dramMode::HAB_PIM, dramMode::HAB);
    uint64_t pim_disable_cycle = drain(*kernel_);
    cout << "  phase: pim_to_sb" << endl;
    kernel_->changePIMMode(dramMode::HAB, dramMode::SB);
    uint64_t pim_to_sb_cycle = drain(*kernel_);

    cout << "  phase: bga_output_readback" << endl;
    for (size_t group_idx = 0; group_idx < bga.groups.size(); ++group_idx)
    {
        const BgaGroupInfo& group = bga.groups[group_idx];
        uint64_t readback_tx = ceilDiv(group.partials_after, kElementsPerBurst);
        for (uint64_t i = 0; i < readback_tx; ++i)
        {
            unsigned row = kResultReadBaseRow + static_cast<unsigned>(group_idx * 16 + (i / 32));
            unsigned col = i % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, group.channel, 0, 0, row, col,
                  &null_bst);
        }
    }
    addPhaseBarriers(mem_, bga_channels);
    uint64_t bga_output_readback_cycle = drain(*kernel_);

    uint64_t pim_kernel_compute_cycle = draf_compute_cycle + bga_accumulate_cycle;
    uint64_t final_reduce_cycle =
        conservative_model ? ceilDiv(bga.host_reduce_ops_after_bga, kConservativeHostReduceWidth)
                           : 0;
    uint64_t per_spmv_draf_bga_model_cycle =
        draf_row_fetch_cycle + pim_enable_cycle + pim_kernel_compute_cycle + pim_disable_cycle +
        pim_to_sb_cycle + bga_output_readback_cycle + final_reduce_cycle;
    double tck_ns = getConfigParam(FLOAT, "tCK");
    double pim_kernel_compute_ms = pim_kernel_compute_cycle * tck_ns / 1000000.0;
    double per_spmv_draf_bga_model_ms = per_spmv_draf_bga_model_cycle * tck_ns / 1000000.0;

    uint64_t v2_capacity_max_flushes =
        v2_params.use_stream_flush ? bga.max_stream_capacity_flushes_per_group
                                   : bga.max_simple_capacity_flushes_per_group;
    uint64_t v2_selected_max_flushes =
        max(bga.max_estimated_flushes_per_group, v2_capacity_max_flushes);
    uint64_t v2_flush_penalty = conservative_model ? v2_params.flush_penalty : 1;
    double v2_padding_factor =
        1.0 + v2_params.padding_alpha *
                  min(1.0, 0.5 * draf.nze_padding_ratio + 0.5 * draf.row_group_padding_ratio);
    double v2_acc_skew_factor = 1.0 + v2_params.skew_alpha * shape.skew_score;
    double v2_readback_skew_factor =
        1.0 + 0.5 * v2_params.skew_alpha *
                  normalizeRatio(shape.bga_unique_rows_imbalance, 16.0);
    double v2_final_skew_factor = 1.0 + v2_params.final_reduce_skew_alpha * shape.skew_score;
    double v2_locality_factor =
        max(v2_params.min_locality_factor,
            1.0 - v2_params.locality_beta * shape.bga_reduction_ratio);
    double v2_final_locality_factor =
        max(0.75, 1.0 - 0.5 * v2_params.locality_beta * shape.bga_reduction_ratio);

    uint64_t v2_draf_row_fetch_cycle =
        static_cast<uint64_t>(ceil(draf_row_fetch_cycle * v2_padding_factor));
    uint64_t v2_draf_compute_cycle =
        static_cast<uint64_t>(ceil(draf_compute_cycle * v2_padding_factor));
    uint64_t v2_bga_accumulate_cycle = static_cast<uint64_t>(
        ceil((bga.max_bacc_instructions_per_group +
              v2_flush_penalty * v2_selected_max_flushes) *
             v2_acc_skew_factor * v2_locality_factor));
    uint64_t v2_bga_output_readback_cycle =
        static_cast<uint64_t>(ceil(bga_output_readback_cycle * v2_readback_skew_factor));
    uint64_t v2_final_reduce_cycle =
        conservative_model
            ? static_cast<uint64_t>(
                  ceil(final_reduce_cycle * v2_final_skew_factor * v2_final_locality_factor))
            : 0;
    uint64_t v2_pim_kernel_compute_cycle = v2_draf_compute_cycle + v2_bga_accumulate_cycle;
    uint64_t v2_per_spmv_draf_bga_model_cycle =
        v2_draf_row_fetch_cycle + pim_enable_cycle + v2_pim_kernel_compute_cycle +
        pim_disable_cycle + pim_to_sb_cycle + v2_bga_output_readback_cycle +
        v2_final_reduce_cycle;
    double v2_pim_kernel_compute_ms = v2_pim_kernel_compute_cycle * tck_ns / 1000000.0;
    double v2_per_spmv_draf_bga_model_ms =
        v2_per_spmv_draf_bga_model_cycle * tck_ns / 1000000.0;

    double v21_padding_factor =
        1.0 + 2.0 * v2_params.padding_alpha * shape.draf_padding_pressure;
    double v21_alignment_factor =
        1.0 + v2_params.skew_alpha *
                  min(1.0, 0.5 * shape.cluster_row_alignment_ratio +
                               0.5 * shape.draf_row_alignment_ratio);
    double v21_effective_locality_beta =
        v2_params.locality_beta *
        max(0.0, 1.0 - 0.75 * shape.draf_padding_pressure);
    double v21_jaccard_boost =
        min(0.15, 0.5 * v21_effective_locality_beta * shape.sampled_cluster_jaccard);
    double v21_locality_factor =
        max(v2_params.min_locality_factor,
            1.0 - v21_effective_locality_beta * shape.bga_reduction_ratio -
                v21_jaccard_boost);
    double v21_final_locality_factor =
        max(0.75,
            1.0 - 0.5 * v21_effective_locality_beta * shape.bga_reduction_ratio -
                0.5 * v21_jaccard_boost);
    double v21_readback_skew_factor =
        v2_readback_skew_factor *
        (1.0 + 0.5 * v2_params.skew_alpha * shape.draf_padding_pressure);
    double v21_final_skew_factor = v2_final_skew_factor * v21_alignment_factor;

    uint64_t v21_draf_row_fetch_cycle =
        static_cast<uint64_t>(ceil(draf_row_fetch_cycle * v21_padding_factor));
    uint64_t v21_draf_compute_cycle =
        static_cast<uint64_t>(ceil(draf_compute_cycle * v21_padding_factor *
                                   v21_alignment_factor));
    uint64_t v21_bga_accumulate_cycle = static_cast<uint64_t>(
        ceil((bga.max_bacc_instructions_per_group +
              v2_flush_penalty * v2_selected_max_flushes) *
             v2_acc_skew_factor * v21_alignment_factor * v21_locality_factor));
    uint64_t v21_bga_output_readback_cycle =
        static_cast<uint64_t>(ceil(bga_output_readback_cycle * v21_readback_skew_factor));
    uint64_t v21_final_reduce_cycle =
        conservative_model
            ? static_cast<uint64_t>(
                  ceil(final_reduce_cycle * v21_final_skew_factor *
                       v21_final_locality_factor))
            : 0;
    uint64_t v21_pim_kernel_compute_cycle =
        v21_draf_compute_cycle + v21_bga_accumulate_cycle;
    uint64_t v21_per_spmv_draf_bga_model_cycle =
        v21_draf_row_fetch_cycle + pim_enable_cycle + v21_pim_kernel_compute_cycle +
        pim_disable_cycle + pim_to_sb_cycle + v21_bga_output_readback_cycle +
        v21_final_reduce_cycle;
    double v21_pim_kernel_compute_ms = v21_pim_kernel_compute_cycle * tck_ns / 1000000.0;
    double v21_per_spmv_draf_bga_model_ms =
        v21_per_spmv_draf_bga_model_cycle * tck_ns / 1000000.0;

    cout << ">>DRAF+BGA-aware SpMV Per-Run Cycle Model" << endl;
    cout << "  input: " << input_name << endl;
    cout << "  model_variant: " << (conservative_model ? "conservative" : "base") << endl;
    cout << "  rows: " << inputs.n_rows << endl;
    cout << "  cols: " << inputs.n_cols << endl;
    cout << "  nnz: " << draf.nnz << endl;
    cout << "  clusters: " << inputs.clusters.size() << endl;
    cout << "  draf_column_groups: " << draf.column_groups << endl;
    cout << "  draf_rows: " << draf.draf_rows << endl;
    cout << "  active_bga_groups: " << bga.groups.size() << endl;
    cout << "  bga_partials_before: " << bga.partials_before << endl;
    cout << "  bga_partials_after: " << bga.partials_after << endl;
    cout << "  simulator_physical_pim_bg: 0" << endl;
    cout << "  excluded_setup_cycle: " << excluded_setup_cycle << endl;
    cout << "> draf_row_fetch_cycle: " << draf_row_fetch_cycle
         << " tx=" << draf.draf_rows << endl;
    cout << "> pim_enable_cycle: " << pim_enable_cycle << endl;
    cout << "> draf_compute_cycle: " << draf_compute_cycle
         << " compute_trigger_tx=" << draf.draf_rows << endl;
    cout << "> bga_accumulate_cycle: " << bga_accumulate_cycle
         << " bacc_instructions=" << bga.bacc_instructions
         << " max_bacc_per_group=" << bga.max_bacc_instructions_per_group
         << " estimated_flushes=" << bga.estimated_flushes
         << " max_flushes_per_group=" << bga.max_estimated_flushes_per_group;
    if (conservative_model)
        cout << " flush_penalty=" << kConservativeBgaFlushPenalty;
    cout << endl;
    cout << "> pim_disable_cycle: " << pim_disable_cycle << endl;
    cout << "> pim_to_sb_cycle: " << pim_to_sb_cycle << endl;
    cout << "> bga_output_readback_cycle: " << bga_output_readback_cycle
         << " tx=" << bga.output_readback_tx << endl;
    cout << "> bga_reduce_ops: " << bga.bga_reduce_ops << endl;
    cout << "> host_reduce_ops_after_bga: " << bga.host_reduce_ops_after_bga << endl;
    if (conservative_model)
    {
        cout << "> final_reduce_cycle: " << final_reduce_cycle
             << " reduce_width=" << kConservativeHostReduceWidth << endl;
    }
    cout << "> pim_kernel_compute_cycle: " << pim_kernel_compute_cycle
         << " ms=" << pim_kernel_compute_ms << endl;
    cout << "> per_spmv_draf_bga_model_cycle: " << per_spmv_draf_bga_model_cycle
         << " ms=" << per_spmv_draf_bga_model_ms << endl;
    if (v2_model)
    {
        cout << ">>DRAF+BGA-aware SpMV "
             << (v21_model ? "V2.1" : "V2") << " Corrected Cycle Model" << endl;
        cout << "  input: " << input_name << endl;
        cout << "  model_variant: "
             << (v21_model ? (conservative_model ? "v2.1_conservative" : "v2.1_base")
                           : (conservative_model ? "v2_conservative" : "v2_base"))
             << endl;
        cout << "  flush_mode: " << (v2_params.use_stream_flush ? "stream" : "simple")
             << endl;
        cout << "  capacity_max_flushes_per_group: " << v2_capacity_max_flushes << endl;
        cout << "  selected_max_flushes_per_group: " << v2_selected_max_flushes << endl;
        cout << "  padding_factor: " << v2_padding_factor << endl;
        cout << "  acc_skew_factor: " << v2_acc_skew_factor << endl;
        cout << "  readback_skew_factor: " << v2_readback_skew_factor << endl;
        cout << "  final_skew_factor: " << v2_final_skew_factor << endl;
        cout << "  locality_factor: " << v2_locality_factor << endl;
        cout << "  final_locality_factor: " << v2_final_locality_factor << endl;
        if (v21_model)
        {
            cout << "  v2.1_padding_factor: " << v21_padding_factor << endl;
            cout << "  v2.1_alignment_factor: " << v21_alignment_factor << endl;
            cout << "  v2.1_effective_locality_beta: " << v21_effective_locality_beta
                 << endl;
            cout << "  v2.1_jaccard_boost: " << v21_jaccard_boost << endl;
            cout << "  v2.1_locality_factor: " << v21_locality_factor << endl;
            cout << "  v2.1_final_locality_factor: " << v21_final_locality_factor << endl;
            cout << "  v2.1_readback_skew_factor: " << v21_readback_skew_factor << endl;
        }
        cout << "> v1_per_spmv_draf_bga_model_cycle: " << per_spmv_draf_bga_model_cycle
             << " ms=" << per_spmv_draf_bga_model_ms << endl;
        uint64_t reported_draf_row_fetch_cycle =
            v21_model ? v21_draf_row_fetch_cycle : v2_draf_row_fetch_cycle;
        uint64_t reported_draf_compute_cycle =
            v21_model ? v21_draf_compute_cycle : v2_draf_compute_cycle;
        uint64_t reported_bga_accumulate_cycle =
            v21_model ? v21_bga_accumulate_cycle : v2_bga_accumulate_cycle;
        uint64_t reported_bga_output_readback_cycle =
            v21_model ? v21_bga_output_readback_cycle : v2_bga_output_readback_cycle;
        uint64_t reported_final_reduce_cycle =
            v21_model ? v21_final_reduce_cycle : v2_final_reduce_cycle;
        uint64_t reported_pim_kernel_compute_cycle =
            v21_model ? v21_pim_kernel_compute_cycle : v2_pim_kernel_compute_cycle;
        uint64_t reported_total_cycle =
            v21_model ? v21_per_spmv_draf_bga_model_cycle
                      : v2_per_spmv_draf_bga_model_cycle;
        double reported_pim_kernel_compute_ms =
            v21_model ? v21_pim_kernel_compute_ms : v2_pim_kernel_compute_ms;
        double reported_total_ms =
            v21_model ? v21_per_spmv_draf_bga_model_ms : v2_per_spmv_draf_bga_model_ms;

        cout << "> v2_draf_row_fetch_cycle: " << reported_draf_row_fetch_cycle
             << " v1=" << draf_row_fetch_cycle << endl;
        cout << "> v2_draf_compute_cycle: " << reported_draf_compute_cycle
             << " v1=" << draf_compute_cycle << endl;
        cout << "> v2_bga_accumulate_cycle: " << reported_bga_accumulate_cycle
             << " v1=" << bga_accumulate_cycle << endl;
        cout << "> v2_bga_output_readback_cycle: " << reported_bga_output_readback_cycle
             << " v1=" << bga_output_readback_cycle << endl;
        if (conservative_model)
            cout << "> v2_final_reduce_cycle: " << reported_final_reduce_cycle
                 << " v1=" << final_reduce_cycle << endl;
        cout << "> v2_pim_kernel_compute_cycle: " << reported_pim_kernel_compute_cycle
             << " ms=" << reported_pim_kernel_compute_ms << endl;
        cout << "> v2_per_spmv_draf_bga_model_cycle: " << reported_total_cycle
             << " ms=" << reported_total_ms << endl;
    }
}

void ClusteredSpmvBenchFixture::runGuidedKmeansDrafBgaSuite(bool conservative_model,
                                                            bool v2_model,
                                                            bool v21_model)
{
    if (v21_model)
        v2_model = true;
    const vector<SpmvDataset> datasets{
        {"ASIC_100k", "../SparsePIM/guided_kmeans_coo_results/ASIC_100k/"},
        {"Stanford", "../SparsePIM/guided_kmeans_coo_results/Stanford/"},
        {"bcsstk32", "../SparsePIM/guided_kmeans_coo_results/bcsstk32/"},
        {"cant", "../SparsePIM/guided_kmeans_coo_results/cant/"},
        {"consph", "../SparsePIM/guided_kmeans_coo_results/consph/"},
        {"crankseg_2", "../SparsePIM/guided_kmeans_coo_results/crankseg_2/"},
        {"ct20stif", "../SparsePIM/guided_kmeans_coo_results/ct20stif/"},
        {"lhr71", "../SparsePIM/guided_kmeans_coo_results/lhr71/"},
        {"ohne2", "../SparsePIM/guided_kmeans_coo_results/ohne2/"},
        {"pdb1HYS", "../SparsePIM/guided_kmeans_coo_results/pdb1HYS/"},
        {"pwtk", "../SparsePIM/guided_kmeans_coo_results/pwtk/"},
        {"rma10", "../SparsePIM/guided_kmeans_coo_results/rma10/"},
        {"shipsec1", "../SparsePIM/guided_kmeans_coo_results/shipsec1/"},
        {"soc-sign-epinions", "../SparsePIM/guided_kmeans_coo_results/soc-sign-epinions/"},
        {"webbase-1M", "../SparsePIM/guided_kmeans_coo_results/webbase-1M/"},
        {"xenon2", "../SparsePIM/guided_kmeans_coo_results/xenon2/"},
    };

    string only_matrix = envString("SPMV_BENCH_MATRIX");
    cout << ">>Guided K-means COO DRAF+BGA suite" << endl;
    cout << "  model_variant: "
         << (v21_model ? (conservative_model ? "v2.1_conservative" : "v2.1_base")
                       : (v2_model ? (conservative_model ? "v2_conservative" : "v2_base")
                                   : (conservative_model ? "conservative" : "base")))
         << endl;
    if (!only_matrix.empty())
        cout << "  SPMV_BENCH_MATRIX: " << only_matrix << endl;

    bool matched = false;
    for (const SpmvDataset& dataset : datasets)
    {
        if (!only_matrix.empty() && dataset.name != only_matrix)
            continue;

        matched = true;
        resetPIMKernel();
        runDrafBgaModel(dataset.base,
                        "SparsePIM/guided_kmeans_coo_results/" + dataset.name,
                        conservative_model, v2_model, v21_model);
    }

    if (!only_matrix.empty())
    {
        ASSERT_TRUE(matched) << "unknown SPMV_BENCH_MATRIX=" << only_matrix;
    }
}

StructuralBaseTiming ClusteredSpmvBenchFixture::runDrafBgaStructuralBaseTiming(
    const DrafStats& draf, const BgaStats& bga)
{
    StructuralBaseTiming timing;
    BurstType null_bst;
    vector<PIMCmd> mac_cmds{
        PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK, 1),
        PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A, PIMOpdType::ODD_BANK, 1),
        PIMCmd(PIMCmdType::NOP, 7),
        PIMCmd(PIMCmdType::EXIT, 0),
    };

    kernel_->configurePIMControl();
    kernel_->parkIn();
    kernel_->changePIMMode(dramMode::SB, dramMode::HAB);
    kernel_->programCrf(mac_cmds);
    timing.setup_cycle = drain(*kernel_);

    vector<char> draf_channels(kernel_->num_pim_chans_, 0);
    uint64_t global_draf_row = 0;
    for (const DrafClusterInfo& cluster : draf.clusters)
    {
        unsigned chan = cluster.id / 4;
        draf_channels[chan] = cluster.draf_rows > 0 ? 1 : draf_channels[chan];
        for (uint64_t i = 0; i < cluster.draf_rows; ++i, ++global_draf_row)
        {
            unsigned row = kMacBaseRow + (global_draf_row / 32);
            unsigned col = global_draf_row % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, chan, 0, 0, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, draf_channels);
    timing.draf_row_fetch_cycle = drain(*kernel_);

    kernel_->changePIMMode(dramMode::HAB, dramMode::HAB_PIM);
    timing.pim_enable_cycle = drain(*kernel_);

    global_draf_row = 0;
    for (const DrafClusterInfo& cluster : draf.clusters)
    {
        unsigned chan = cluster.id / 4;
        for (uint64_t i = 0; i < cluster.draf_rows; ++i, ++global_draf_row)
        {
            unsigned row = kMacBaseRow + (global_draf_row / 32);
            unsigned col = global_draf_row % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, chan, 0, i % 2, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, draf_channels);
    timing.draf_compute_trigger_cycle = drain(*kernel_);

    kernel_->changePIMMode(dramMode::HAB_PIM, dramMode::HAB);
    timing.pim_disable_cycle = drain(*kernel_);
    kernel_->changePIMMode(dramMode::HAB, dramMode::SB);
    timing.pim_to_sb_cycle = drain(*kernel_);

    vector<char> bga_channels(kernel_->num_pim_chans_, 0);
    for (const BgaGroupInfo& group : bga.groups)
        bga_channels[group.channel] = 1;
    for (const BgaGroupInfo& group : bga.groups)
    {
        uint64_t readback_tx = ceilDiv(group.partials_after, kElementsPerBurst);
        if (readback_tx > 0 &&
            kResultReadBaseRow + (readback_tx - 1) / 32 >=
                kernel_->pim_addr_mgr_->num_rows_)
            throw runtime_error("BGA readback exceeds physical row capacity");
        for (uint64_t i = 0; i < readback_tx; ++i)
        {
            unsigned row = kResultReadBaseRow + static_cast<unsigned>(i / 32);
            unsigned col = i % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, group.channel, group.bank_group, 0,
                  row, col,
                  &null_bst);
        }
    }
    addPhaseBarriers(mem_, bga_channels);
    timing.bga_output_readback_cycle = drain(*kernel_);
    return timing;
}

StructuralModelResult ClusteredSpmvBenchFixture::runDrafBgaStructuralModel(
    const string& base, const string& input_name, const string& matrix_name,
    int structural_variant)
{
    SpmvInputs inputs = loadSparsePIMInputs(base + "reordered_matrix.txt",
                                            base + "column_permutation.txt", base + "clusters.txt");
    return runDrafBgaStructuralModelFromInputs(
        move(inputs), input_name, matrix_name, structural_variant);
}

StructuralModelResult ClusteredSpmvBenchFixture::runDrafBgaStructuralModelFromInputs(
    SpmvInputs inputs, const string& input_name, const string& matrix_name,
    int structural_variant)
{
    uint64_t max_clusters = envLimit("SPMV_BENCH_MAX_CLUSTERS");
    applyClusterLimit(inputs, max_clusters);
    DrafStats draf = buildDrafStats(inputs);
    BgaStats bga = buildBgaStats(inputs, kDefaultV2BgaAccCapacity);
    ShapeStats shape = buildShapeStats(inputs, draf, bga);
    DrafCriticalPathStats critical = buildDrafCriticalPathStats(draf);
    DrafPaddingExposureStats exposure =
        buildDrafPaddingExposureStats(draf, shape, critical);
    const StructuralVariantSpec& variant_spec = structuralVariantSpec(structural_variant);
    bool phase_exposure_model = variant_spec.phase_exposure;
    bool critical_padding_model = variant_spec.critical_padding;
    bool exposure_padding_model = variant_spec.exposure_padding;
    bool bga_overlap_model = variant_spec.bga_overlap;
    bool draf_streaming_overlap_model = variant_spec.draf_streaming_overlap;
    bool draf_bga_budget_model = variant_spec.draf_bga_budget;
    bool reuse_aware_bga_model = variant_spec.reuse_aware_bga;
    bool draf_memory_efficiency_model = variant_spec.draf_memory_efficiency;
    bool conservative_overlap_reuse_model = variant_spec.conservative_overlap_reuse;
    bool stronger_draf_memory_conservative_model =
        variant_spec.stronger_draf_memory_conservative;

    cout << ">>DRAF+BGA-aware SpMV ";
    cout << variant_spec.model_banner;
    cout << endl;
    cout << "  input: " << input_name << endl;
    cout << "  matrix: " << matrix_name << endl;
    cout << "  note: structural model; target speedup is used only for reporting" << endl;
    cout << "  rows: " << inputs.n_rows << endl;
    cout << "  cols: " << inputs.n_cols << endl;
    cout << "  nnz: " << draf.nnz << endl;
    cout << "  clusters: " << inputs.clusters.size() << endl;
    cout << "  draf_column_groups: " << draf.column_groups << endl;
    cout << "  draf_rows: " << draf.draf_rows << endl;
    cout << "  draf_packed_nnz_capacity: " << draf.packed_nnz_capacity << endl;
    cout << "  draf_nze_padding: " << draf.nze_padding << endl;
    cout << "  draf_padding_ratio: " << draf.nze_padding_ratio << endl;
    cout << "  draf_expansion_ratio: " << draf.expansion_ratio << endl;
    cout << "  draf_nze_padding_ratio: " << draf.nze_padding_ratio << endl;
    cout << "  draf_memory_expansion: " << draf.expansion_ratio << endl;
    cout << "  critical_padding: " << critical.critical_padding << endl;
    cout << "  critical_padding_ratio: " << critical.critical_padding_ratio << endl;
    cout << "  group_steps_mean: " << critical.group_steps_mean << endl;
    cout << "  group_steps_max: " << critical.group_steps_max << endl;
    cout << "  group_steps_p90: " << critical.group_steps_p90 << endl;
    cout << "  group_steps_p95: " << critical.group_steps_p95 << endl;
    cout << "  tail_skew_max_over_p90: " << critical.tail_skew_max_over_p90 << endl;
    cout << "  tail_skew_max_over_p95: " << critical.tail_skew_max_over_p95 << endl;
    cout << "  p90_to_max_tail_steps: " << critical.p90_to_max_tail_steps << endl;
    cout << "  p95_to_max_tail_steps: " << critical.p95_to_max_tail_steps << endl;
    cout << "  tail_exposure_ratio: " << critical.tail_exposure_ratio << endl;
    cout << "  bg_imbalance: " << critical.bg_imbalance << endl;
    cout << "  total_padding_steps: " << exposure.total_padding_steps << endl;
    cout << "  hidden_padding_steps: " << exposure.hidden_padding_steps << endl;
    cout << "  fragmentation: " << exposure.fragmentation << endl;
    cout << "  memory_pressure: " << exposure.memory_pressure << endl;
    cout << "  imbalance_pressure: " << exposure.imbalance_pressure << endl;
    cout << "  exposure_factor: " << exposure.exposure_factor << endl;
    cout << "  exposed_hidden_padding: " << exposure.exposed_hidden_padding << endl;
    cout << "  effective_padding_steps: " << exposure.effective_padding_steps << endl;
    cout << "  single_nnz_column_ratio: " << shape.single_nnz_column_ratio << endl;
    cout << "  low_nnz_column_ratio: " << shape.low_nnz_column_ratio << endl;
    cout << "  draf_padding_pressure: " << shape.draf_padding_pressure << endl;
    cout << "  bga_reduction_ratio: " << shape.bga_reduction_ratio << endl;

    StructuralBaseTiming timing = runDrafBgaStructuralBaseTiming(draf, bga);
    uint64_t setup_cycle = timing.setup_cycle;
    uint64_t draf_row_fetch_cycle = timing.draf_row_fetch_cycle;
    uint64_t pim_enable_cycle = timing.pim_enable_cycle;
    uint64_t draf_compute_trigger_cycle = timing.draf_compute_trigger_cycle;

    double tck_ns = getConfigParam(FLOAT, "tCK");
    double gpu_ms = gpuBaselineMs(matrix_name);
    double target_speedup = paperTargetSpeedup(matrix_name);
    if (variant_spec.id == 1001)
    {
        return runDrafBgaStructuralModelBV1(inputs, draf, bga, shape, critical,
                                           exposure, timing, variant_spec,
                                           matrix_name, gpu_ms, target_speedup,
                                           tck_ns);
    }
    if (variant_spec.id == 1002)
    {
        return runDrafBgaStructuralModelBV2(inputs, draf, bga, shape, critical,
                                           exposure, timing, variant_spec,
                                           matrix_name, gpu_ms, target_speedup,
                                           tck_ns);
    }
    if (variant_spec.id == 1003)
    {
        return runDrafBgaStructuralModelBV3(inputs, draf, bga, shape, critical,
                                           exposure, timing, variant_spec,
                                           matrix_name, gpu_ms, target_speedup,
                                           tck_ns);
    }
    if (variant_spec.id == 1004 || variant_spec.id == 2004)
    {
        return runDrafBgaStructuralModelBV4(inputs, draf, bga, shape, critical,
                                           exposure, timing, variant_spec,
                                           matrix_name, gpu_ms, target_speedup,
                                           tck_ns, true);
    }
    if (variant_spec.id == 14)
    {
        return runDrafBgaStructuralModelV14(inputs, draf, bga, shape, critical,
                                           exposure, timing, variant_spec,
                                           matrix_name, gpu_ms, target_speedup,
                                           tck_ns);
    }
    if (variant_spec.id == 15)
    {
        return runDrafBgaStructuralModelV15(inputs, draf, bga, shape, critical,
                                           exposure, timing, variant_spec,
                                           matrix_name, gpu_ms, target_speedup,
                                           tck_ns);
    }
    if (variant_spec.id == 16)
    {
        return runDrafBgaStructuralModelV16(inputs, draf, bga, shape, critical,
                                           exposure, timing, variant_spec,
                                           matrix_name, gpu_ms, target_speedup,
                                           tck_ns);
    }
    if (variant_spec.id == 17)
    {
        return runDrafBgaStructuralModelV17(inputs, draf, bga, shape, critical,
                                           exposure, timing, variant_spec,
                                           matrix_name, gpu_ms, target_speedup,
                                           tck_ns);
    }
    if (variant_spec.id == 171)
    {
        return runDrafBgaStructuralModelV17_1(inputs, draf, bga, shape, critical,
                                             exposure, timing, variant_spec,
                                             matrix_name, gpu_ms, target_speedup,
                                             tck_ns);
    }
    if (variant_spec.id == 18)
    {
        return runDrafBgaStructuralModelV18(inputs, draf, bga, shape, critical,
                                           exposure, timing, variant_spec,
                                           matrix_name, gpu_ms, target_speedup,
                                           tck_ns);
    }

    uint64_t bga_capacity_flushes =
        max(bga.max_estimated_flushes_per_group, bga.max_stream_capacity_flushes_per_group);
    uint64_t instruction_bga_accumulate_cycle =
        bga.max_bacc_instructions_per_group + kConservativeBgaFlushPenalty * bga_capacity_flushes;
    uint64_t reuse_aware_bga_accumulate_cycle =
        bga.max_reuse_aware_accumulate_cycle_per_group;
    double mean_nnz_per_col =
        inputs.n_cols == 0
            ? 0.0
            : static_cast<double>(draf.nnz) / static_cast<double>(inputs.n_cols);
    double instruction_phase_total =
        static_cast<double>(instruction_bga_accumulate_cycle + draf_row_fetch_cycle +
                            draf_compute_trigger_cycle) +
        exposure.effective_padding_steps + 1.0;
    double instruction_bga_share =
        static_cast<double>(instruction_bga_accumulate_cycle) / instruction_phase_total;
    double weak_reuse_serialization_guard =
        clamp01(1.0 - (shape.bga_row_reuse_factor - 1.0) / 32.0);
    double pre_phase_bga_score =
        clamp01(0.55 * instruction_bga_share +
                0.25 * clamp01(shape.bga_capacity_pressure / 16.0) +
                0.20 * weak_reuse_serialization_guard);
    double pre_phase_draf_score =
        clamp01(0.45 * clamp01(draf.expansion_ratio - 1.0) +
                0.30 * shape.draf_padding_pressure +
                0.25 * clamp01(mean_nnz_per_col / 64.0));
    double pre_phase_padding_score =
        clamp01(0.50 * exposure.exposure_factor +
                0.30 * critical.critical_padding_ratio +
                0.20 * (1.0 - critical.tail_exposure_ratio));
    double conservative_bga_floor_factor = 0.0;
    if (phase_exposure_model)
    {
        double reuse_relief = clamp01((shape.bga_row_reuse_factor - 8.0) / 32.0);
        conservative_bga_floor_factor =
            clamp01(0.35 + 0.45 * pre_phase_bga_score +
                    0.20 * pre_phase_padding_score - 0.25 * reuse_relief);
    }
    else
    {
        conservative_bga_floor_factor =
            stronger_draf_memory_conservative_model
                ? 0.75
                : (conservative_overlap_reuse_model ? 0.50 : 0.0);
    }
    uint64_t conservative_bga_floor_cycle =
        static_cast<uint64_t>(ceil(conservative_bga_floor_factor *
                                   static_cast<double>(instruction_bga_accumulate_cycle)));
    if (conservative_overlap_reuse_model)
        reuse_aware_bga_accumulate_cycle =
            max(reuse_aware_bga_accumulate_cycle, conservative_bga_floor_cycle);
    uint64_t raw_bga_accumulate_cycle =
        reuse_aware_bga_model ? reuse_aware_bga_accumulate_cycle
                              : instruction_bga_accumulate_cycle;

    uint64_t pim_disable_cycle = timing.pim_disable_cycle;
    uint64_t pim_to_sb_cycle = timing.pim_to_sb_cycle;
    uint64_t bga_output_readback_cycle = timing.bga_output_readback_cycle;

    /*
     * V3 charges all padded NZE slots. V4 charges only the padding exposed on the synchronous
     * bank-group critical path, so regular padding that is hidden inside shorter banks does not
     * receive the same cost as useful nonzero work.
     */
    uint64_t padded_zero_compute_cycle = 0;
    if (phase_exposure_model)
    {
        double tail_exposure_factor =
            clamp01(0.55 + 0.45 * critical.tail_exposure_ratio);
        padded_zero_compute_cycle = static_cast<uint64_t>(
            ceil(exposure.effective_padding_steps * tail_exposure_factor));
    }
    else if (exposure_padding_model)
    {
        padded_zero_compute_cycle =
            static_cast<uint64_t>(ceil(exposure.effective_padding_steps));
    }
    else if (critical_padding_model)
    {
        padded_zero_compute_cycle =
            static_cast<uint64_t>(ceil(critical.critical_padding));
    }
    else
    {
        padded_zero_compute_cycle = ceilDiv(draf.nze_padding, kElementsPerBurst);
    }
    uint64_t compute_overlap_window = draf_compute_trigger_cycle + padded_zero_compute_cycle;
    BgaOverlapStats bga_overlap =
        buildBgaOverlapStats(raw_bga_accumulate_cycle, compute_overlap_window, shape, exposure);
    uint64_t bga_hidden_accumulate_cycle =
        bga_overlap_model ? bga_overlap.hidden_accumulate_cycle : 0;
    uint64_t bga_accumulate_cycle =
        raw_bga_accumulate_cycle > bga_hidden_accumulate_cycle
            ? raw_bga_accumulate_cycle - bga_hidden_accumulate_cycle
            : 0;
    uint64_t final_reduce_cycle = ceilDiv(bga.host_reduce_ops_after_bga,
                                          kConservativeHostReduceWidth);
    uint64_t pre_streaming_total_cycle =
        setup_cycle + draf_row_fetch_cycle + pim_enable_cycle + draf_compute_trigger_cycle +
        padded_zero_compute_cycle + bga_accumulate_cycle + pim_disable_cycle + pim_to_sb_cycle +
        bga_output_readback_cycle + final_reduce_cycle;
    DrafStreamingOverlapStats draf_streaming =
        buildDrafStreamingOverlapStats(draf_row_fetch_cycle, draf_compute_trigger_cycle,
                                       pre_streaming_total_cycle, compute_overlap_window,
                                       exposure, bga_overlap, draf_bga_budget_model);
    uint64_t draf_stream_hidden_cycle =
        draf_streaming_overlap_model ? draf_streaming.hidden_access_cycle : 0;
    double bga_bound_score = pre_phase_bga_score;
    double draf_access_bound_score = pre_phase_draf_score;
    double padding_sync_bound_score = pre_phase_padding_score;
    double phase_class_code = 0.0;
    uint64_t shared_overlap_window_cycle = 0;
    uint64_t shared_bga_overlap_cycle = bga_hidden_accumulate_cycle;
    uint64_t shared_draf_overlap_cycle = draf_stream_hidden_cycle;
    if (phase_exposure_model)
    {
        uint64_t bga_demand = bga_overlap.hidden_accumulate_cycle;
        uint64_t draf_demand = draf_stream_hidden_cycle;
        shared_overlap_window_cycle = compute_overlap_window;
        if (bga_bound_score >= draf_access_bound_score &&
            bga_bound_score >= padding_sync_bound_score)
            phase_class_code = 1.0;
        else if (draf_access_bound_score >= padding_sync_bound_score)
            phase_class_code = 2.0;
        else
            phase_class_code = 3.0;

        double bga_weight = 0.35 + bga_bound_score;
        double draf_weight = 0.35 + draf_access_bound_score;
        if (phase_class_code == 1.0)
            bga_weight += 0.35;
        else if (phase_class_code == 2.0)
            draf_weight += 0.35;
        double weight_sum = bga_weight + draf_weight;
        uint64_t bga_budget = static_cast<uint64_t>(
            floor(static_cast<double>(shared_overlap_window_cycle) *
                  bga_weight / weight_sum));
        shared_bga_overlap_cycle = min(bga_demand, bga_budget);
        uint64_t remaining_window =
            shared_overlap_window_cycle > shared_bga_overlap_cycle
                ? shared_overlap_window_cycle - shared_bga_overlap_cycle
                : 0;
        shared_draf_overlap_cycle = min(draf_demand, remaining_window);
        bga_hidden_accumulate_cycle = shared_bga_overlap_cycle;
        draf_stream_hidden_cycle = shared_draf_overlap_cycle;
        bga_accumulate_cycle =
            raw_bga_accumulate_cycle > bga_hidden_accumulate_cycle
                ? raw_bga_accumulate_cycle - bga_hidden_accumulate_cycle
                : 0;
        pre_streaming_total_cycle =
            setup_cycle + draf_row_fetch_cycle + pim_enable_cycle +
            draf_compute_trigger_cycle + padded_zero_compute_cycle + bga_accumulate_cycle +
            pim_disable_cycle + pim_to_sb_cycle + bga_output_readback_cycle +
            final_reduce_cycle;
    }
    uint64_t memory_bound_access_cycle = draf_row_fetch_cycle + draf_compute_trigger_cycle;
    DrafMemoryEfficiencyStats draf_memory =
        buildDrafMemoryEfficiencyStats(draf, exposure, memory_bound_access_cycle);
    uint64_t draf_memory_saved_cycle =
        draf_memory_efficiency_model ? draf_memory.saved_access_cycle : 0;
    double draf_memory_long_stream_score = 0.0;
    uint64_t draf_memory_padding_saved_cycle = 0;
    if (stronger_draf_memory_conservative_model && draf_memory_efficiency_model)
    {
        /*
         * DRAF's COO-index reduction also lowers sustained dummy/padding traffic once a long
         * column stream is established. Keep this separate from the direct row-fetch/trigger
         * saving so the overlap diagnostics can expose whether the gain comes from memory
         * format efficiency or from optimistic phase hiding.
         */
        draf_memory_long_stream_score =
            static_cast<double>(draf.draf_rows) /
            (static_cast<double>(draf.draf_rows) + 32768.0);
        draf_memory_padding_saved_cycle = static_cast<uint64_t>(
            floor(static_cast<double>(padded_zero_compute_cycle) *
                  draf_memory.exposed_saving_factor * draf_memory_long_stream_score));
        draf_memory_saved_cycle += draf_memory_padding_saved_cycle;
    }
    uint64_t draf_step_tail_saved_cycle = 0;
    double percentile_tail_exposure_factor =
        phase_exposure_model ? clamp01(0.55 + 0.45 * critical.tail_exposure_ratio) : 1.0;
    if (phase_exposure_model && draf_memory_efficiency_model)
    {
        double tail_reduction_factor =
            draf_memory.exposed_saving_factor * (1.0 - percentile_tail_exposure_factor);
        if (phase_class_code == 2.0)
            tail_reduction_factor *= 1.5;
        if (phase_class_code == 3.0)
            tail_reduction_factor *= 0.5;
        tail_reduction_factor = clamp01(tail_reduction_factor);
        draf_step_tail_saved_cycle = static_cast<uint64_t>(
            floor(static_cast<double>(draf_compute_trigger_cycle + padded_zero_compute_cycle) *
                  tail_reduction_factor));
        draf_memory_saved_cycle += draf_step_tail_saved_cycle;
    }
    uint64_t memory_budgeted_stream_hidden_cycle = draf_stream_hidden_cycle;
    double stream_memory_budget_factor = 1.0;
    if (conservative_overlap_reuse_model)
    {
        stream_memory_budget_factor = 1.0 - draf_memory.exposed_saving_factor;
        if (stronger_draf_memory_conservative_model)
            stream_memory_budget_factor *= 0.50;
        memory_budgeted_stream_hidden_cycle = static_cast<uint64_t>(
            floor(static_cast<double>(draf_stream_hidden_cycle) *
                  stream_memory_budget_factor));
    }
    uint64_t hidden_cycle = memory_budgeted_stream_hidden_cycle + draf_memory_saved_cycle;
    uint64_t v3_total_cycle =
        pre_streaming_total_cycle > hidden_cycle ? pre_streaming_total_cycle - hidden_cycle : 0;

    double v3_total_ms = v3_total_cycle * tck_ns / 1000000.0;
    double v3_speedup = v3_total_ms == 0.0 ? 0.0 : gpu_ms / v3_total_ms;
    double target_pim_ms =
        target_speedup == 0.0 ? 0.0 : gpu_ms / target_speedup;
    StructuralModelResult result;
    result.matrix = matrix_name;
    result.gpu_ms = gpu_ms;
    result.target_speedup = target_speedup;
    result.target_pim_ms = target_pim_ms;
    result.model_ms = v3_total_ms;
    result.model_speedup = v3_speedup;
    result.speedup_error_ratio =
        target_speedup == 0.0 ? 0.0 : v3_speedup / target_speedup;
    result.total_cycle = v3_total_cycle;
    result.setup_cycle = setup_cycle;
    result.draf_row_fetch_cycle = draf_row_fetch_cycle;
    result.draf_compute_trigger_cycle = draf_compute_trigger_cycle;
    result.padding_cycle = padded_zero_compute_cycle;
    result.bga_accumulate_cycle = bga_accumulate_cycle;
    result.bga_output_readback_cycle = bga_output_readback_cycle;
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
    result.bga_hidden_accumulate_cycle = bga_hidden_accumulate_cycle;
    result.bga_regularity_score = bga_overlap.regularity_score;
    result.bga_reuse_score = bga_overlap.reuse_score;
    result.bga_padding_guard = bga_overlap.padding_guard;
    result.bga_overlap_factor = bga_overlap.overlap_factor;
    result.draf_stream_hidden_cycle = draf_stream_hidden_cycle;
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
    result.bga_critical_flushes =
        reuse_aware_bga_model ? bga.max_stream_capacity_flushes_per_group : bga_capacity_flushes;
    result.bga_reuse_aware_accumulate_cycle = reuse_aware_bga_accumulate_cycle;
    result.draf_memory_saved_cycle = draf_memory_saved_cycle;
    result.draf_coo_bytes_per_nnz = draf_memory.coo_bytes_per_nnz;
    result.draf_bytes_per_nnz = draf_memory.draf_bytes_per_nnz;
    result.draf_vs_coo_memory_ratio = draf_memory.draf_vs_coo_ratio;
    result.draf_memory_saving_factor = draf_memory.memory_saving_factor;
    result.draf_exposed_memory_saving_factor = draf_memory.exposed_saving_factor;
    result.conservative_bga_floor_cycle = conservative_bga_floor_cycle;
    result.memory_budgeted_stream_hidden_cycle = memory_budgeted_stream_hidden_cycle;
    result.draf_memory_padding_saved_cycle = draf_memory_padding_saved_cycle;
    result.draf_memory_long_stream_score = draf_memory_long_stream_score;
    result.conservative_bga_floor_factor = conservative_bga_floor_factor;
    result.stream_memory_budget_factor = stream_memory_budget_factor;
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
    result.shared_overlap_window_cycle = shared_overlap_window_cycle;
    result.shared_bga_overlap_cycle = shared_bga_overlap_cycle;
    result.shared_draf_overlap_cycle = shared_draf_overlap_cycle;
    result.draf_step_tail_saved_cycle = draf_step_tail_saved_cycle;

    cout << "  latency_scope: setup + DRAF access + "
         << (exposure_padding_model
                 ? "exposed hidden padding"
                 : (critical_padding_model ? "critical-path padding" : "padded zero work"))
         << " + BGA + result readback" << endl;
    cout << "> setup_cycle: " << setup_cycle << endl;
    cout << "> draf_row_fetch_cycle: " << draf_row_fetch_cycle << endl;
    cout << "> draf_compute_trigger_cycle: " << draf_compute_trigger_cycle << endl;
    cout << "> "
         << (exposure_padding_model
                 ? "exposed_padding_cycle: "
                 : (critical_padding_model ? "critical_padding_cycle: "
                                           : "padded_zero_compute_cycle: "))
         << padded_zero_compute_cycle;
    if (exposure_padding_model)
    {
        cout << " effective_padding_steps=" << exposure.effective_padding_steps
             << " critical_padding=" << critical.critical_padding
             << " hidden_padding_steps=" << exposure.hidden_padding_steps
             << " exposure_factor=" << exposure.exposure_factor;
    }
    else if (critical_padding_model)
    {
        cout << " critical_padding=" << critical.critical_padding
             << " ideal_steps=" << critical.ideal_steps
             << " actual_steps=" << critical.actual_steps;
    }
    else
    {
        cout << " padded_slots=" << draf.nze_padding
             << " simd_width=" << kElementsPerBurst;
    }
    cout << endl;
    cout << "> bga_accumulate_cycle: " << bga_accumulate_cycle
         << " raw_bga_accumulate_cycle=" << raw_bga_accumulate_cycle
         << " hidden_bga_accumulate_cycle=" << bga_hidden_accumulate_cycle
         << " bga_overlap_factor=" << bga_overlap.overlap_factor
         << " bga_regularity_score=" << bga_overlap.regularity_score
         << " bga_reuse_score=" << bga_overlap.reuse_score
         << " bga_padding_guard=" << bga_overlap.padding_guard
         << " max_bacc_per_group=" << bga.max_bacc_instructions_per_group
         << " reuse_aware_bga_cycle=" << reuse_aware_bga_accumulate_cycle
         << " conservative_bga_floor_cycle=" << conservative_bga_floor_cycle
         << " conservative_bga_floor_factor=" << conservative_bga_floor_factor
         << " bga_row_reuse_factor=" << shape.bga_row_reuse_factor
         << " bga_unique_row_ratio=" << shape.bga_unique_row_ratio
         << " bga_duplicate_partial_ratio=" << shape.bga_duplicate_partial_ratio
         << " near_duplicate_partial_ratio=" << bga.near_duplicate_partial_ratio
         << " far_duplicate_partial_ratio=" << bga.far_duplicate_partial_ratio
         << " near_duplicate_share=" << bga.near_duplicate_share
         << " accumulator_flush_estimate=" << bga.accumulator_flush_estimate
         << " selected_flushes_per_group=" << bga_capacity_flushes
         << " flush_penalty=" << kConservativeBgaFlushPenalty << endl;
    cout << "> bga_output_readback_cycle: " << bga_output_readback_cycle << endl;
    cout << "> final_reduce_cycle: " << final_reduce_cycle << endl;
    cout << "> draf_stream_hidden_cycle: " << draf_stream_hidden_cycle
         << " draf_stream_overlap_factor=" << draf_streaming.overlap_factor
         << " draf_stream_regularity_score=" << draf_streaming.regularity_score
         << " draf_stream_guard_score=" << draf_streaming.guard_score
         << " draf_access_share=" << draf_streaming.access_share
         << " draf_bga_window_occupancy=" << draf_streaming.bga_window_occupancy
         << " draf_bga_contention_score=" << draf_streaming.bga_contention_score
         << " draf_stream_budget_factor=" << draf_streaming.budget_factor
         << " stream_memory_budget_factor=" << stream_memory_budget_factor
         << " memory_budgeted_stream_hidden_cycle="
         << memory_budgeted_stream_hidden_cycle << endl;
    cout << "> draf_memory_saved_cycle: " << draf_memory_saved_cycle
         << " draf_memory_padding_saved_cycle=" << draf_memory_padding_saved_cycle
         << " draf_memory_long_stream_score=" << draf_memory_long_stream_score
         << " draf_vs_coo_memory_ratio=" << draf_memory.draf_vs_coo_ratio
         << " draf_memory_saving_factor=" << draf_memory.memory_saving_factor
         << " draf_exposed_memory_saving_factor=" << draf_memory.exposed_saving_factor
         << " draf_bytes_per_nnz=" << draf_memory.draf_bytes_per_nnz
         << " coo_bytes_per_nnz=" << draf_memory.coo_bytes_per_nnz << endl;
    if (phase_exposure_model)
    {
        cout << "> phase_exposure_class: " << phase_class_code
             << " bga_bound_score=" << bga_bound_score
             << " draf_access_bound_score=" << draf_access_bound_score
             << " padding_sync_bound_score=" << padding_sync_bound_score
             << " percentile_tail_exposure_factor="
             << percentile_tail_exposure_factor
             << " group_steps_p90=" << critical.group_steps_p90
             << " group_steps_p95=" << critical.group_steps_p95
             << " group_steps_max=" << critical.group_steps_max
             << " tail_skew_max_over_p90=" << critical.tail_skew_max_over_p90
             << " tail_skew_max_over_p95=" << critical.tail_skew_max_over_p95
             << " p90_to_max_tail_steps=" << critical.p90_to_max_tail_steps
             << " p95_to_max_tail_steps=" << critical.p95_to_max_tail_steps
             << " shared_overlap_window_cycle=" << shared_overlap_window_cycle
             << " shared_bga_overlap_cycle=" << shared_bga_overlap_cycle
             << " shared_draf_overlap_cycle=" << shared_draf_overlap_cycle
             << " draf_step_tail_saved_cycle=" << draf_step_tail_saved_cycle
             << endl;
    }
    cout << "> " << variant_spec.cycle_label << v3_total_cycle
         << " ms=" << v3_total_ms << endl;
    cout << "> gpu_baseline_ms: " << gpu_ms << endl;
    cout << "> paper_target_speedup: " << target_speedup
         << " target_pim_ms=" << target_pim_ms << endl;
    cout << "> " << variant_spec.speedup_label << v3_speedup
         << " speedup_error_ratio="
         << (target_speedup == 0.0 ? 0.0 : v3_speedup / target_speedup) << endl;
    cout << variant_spec.result_csv_tag
         << matrix_name << "," << gpu_ms << "," << target_speedup << "," << target_pim_ms
         << "," << v3_total_ms << "," << v3_speedup << ","
         << (target_speedup == 0.0 ? 0.0 : v3_speedup / target_speedup) << ","
         << draf.nze_padding << "," << draf.nze_padding_ratio << ","
         << draf.expansion_ratio << "," << critical.critical_padding << ","
         << critical.critical_padding_ratio << "," << critical.group_steps_mean << ","
         << critical.group_steps_max << "," << critical.bg_imbalance << ","
         << exposure.total_padding_steps << "," << exposure.hidden_padding_steps << ","
         << exposure.fragmentation << "," << exposure.memory_pressure << ","
         << exposure.imbalance_pressure << "," << exposure.exposure_factor << ","
         << exposure.exposed_hidden_padding << "," << exposure.effective_padding_steps
         << "," << raw_bga_accumulate_cycle << "," << bga_hidden_accumulate_cycle
         << "," << bga_overlap.regularity_score << "," << bga_overlap.reuse_score
         << "," << bga_overlap.padding_guard << "," << bga_overlap.overlap_factor
         << "," << draf_stream_hidden_cycle << "," << draf_streaming.regularity_score
         << "," << draf_streaming.guard_score << "," << draf_streaming.access_share
         << "," << draf_streaming.overlap_factor
         << "," << draf_streaming.bga_window_occupancy
         << "," << draf_streaming.bga_contention_score
         << "," << draf_streaming.budget_factor
         << "," << bga.near_duplicate_partial_ratio
         << "," << bga.far_duplicate_partial_ratio
         << "," << bga.near_duplicate_share
         << "," << bga.accumulator_flush_estimate
         << "," << draf_memory_saved_cycle
         << "," << draf_memory.coo_bytes_per_nnz
         << "," << draf_memory.draf_bytes_per_nnz
         << "," << draf_memory.draf_vs_coo_ratio
         << "," << draf_memory.memory_saving_factor
         << "," << draf_memory.exposed_saving_factor
         << "," << conservative_bga_floor_cycle
         << "," << memory_budgeted_stream_hidden_cycle
         << "," << draf_memory_padding_saved_cycle
         << "," << draf_memory_long_stream_score
         << "," << conservative_bga_floor_factor
         << "," << stream_memory_budget_factor
         << "," << phase_class_code
         << "," << bga_bound_score
         << "," << draf_access_bound_score
         << "," << padding_sync_bound_score
         << "," << percentile_tail_exposure_factor
         << "," << critical.group_steps_p90
         << "," << critical.group_steps_p95
         << "," << critical.group_steps_max
         << "," << critical.tail_skew_max_over_p90
         << "," << critical.tail_skew_max_over_p95
         << "," << critical.p90_to_max_tail_steps
         << "," << critical.p95_to_max_tail_steps
         << "," << shared_overlap_window_cycle
         << "," << shared_bga_overlap_cycle
         << "," << shared_draf_overlap_cycle
         << "," << draf_step_tail_saved_cycle
         << endl;
    return result;
}

void ClusteredSpmvBenchFixture::runGuidedKmeansDrafBgaStructuralSuite(
    int structural_variant)
{
    const vector<SpmvDataset> datasets{
        {"ASIC_100k", "../SparsePIM/guided_kmeans_coo_results/ASIC_100k/"},
        {"Stanford", "../SparsePIM/guided_kmeans_coo_results/Stanford/"},
        {"bcsstk32", "../SparsePIM/guided_kmeans_coo_results/bcsstk32/"},
        {"cant", "../SparsePIM/guided_kmeans_coo_results/cant/"},
        {"consph", "../SparsePIM/guided_kmeans_coo_results/consph/"},
        {"crankseg_2", "../SparsePIM/guided_kmeans_coo_results/crankseg_2/"},
        {"ct20stif", "../SparsePIM/guided_kmeans_coo_results/ct20stif/"},
        {"lhr71", "../SparsePIM/guided_kmeans_coo_results/lhr71/"},
        {"ohne2", "../SparsePIM/guided_kmeans_coo_results/ohne2/"},
        {"pdb1HYS", "../SparsePIM/guided_kmeans_coo_results/pdb1HYS/"},
        {"pwtk", "../SparsePIM/guided_kmeans_coo_results/pwtk/"},
        {"rma10", "../SparsePIM/guided_kmeans_coo_results/rma10/"},
        {"shipsec1", "../SparsePIM/guided_kmeans_coo_results/shipsec1/"},
        {"soc-sign-epinions", "../SparsePIM/guided_kmeans_coo_results/soc-sign-epinions/"},
        {"webbase-1M", "../SparsePIM/guided_kmeans_coo_results/webbase-1M/"},
        {"xenon2", "../SparsePIM/guided_kmeans_coo_results/xenon2/"},
    };

    string only_matrix = envString("SPMV_BENCH_MATRIX");
    const StructuralVariantSpec& variant_spec = structuralVariantSpec(structural_variant);
    cout << ">>Guided K-means COO DRAF+BGA ";
    cout << variant_spec.suite_label;
    cout << endl;
    if (!only_matrix.empty())
        cout << "  SPMV_BENCH_MATRIX: " << only_matrix << endl;

    bool matched = false;
    vector<StructuralModelResult> results;
    for (const SpmvDataset& dataset : datasets)
    {
        if (!only_matrix.empty() && dataset.name != only_matrix)
            continue;

        matched = true;
        resetPIMKernel();
        StructuralModelResult result =
            runDrafBgaStructuralModel(dataset.base,
                                      "SparsePIM/guided_kmeans_coo_results/" + dataset.name,
                                      dataset.name, structural_variant);
        results.push_back(result);
    }

    if (!only_matrix.empty())
    {
        ASSERT_TRUE(matched) << "unknown SPMV_BENCH_MATRIX=" << only_matrix;
    }
    else
    {
        writeStructuralResults(structural_variant, results);
        writeStructuralDiagnostics(structural_variant, results);
        writeStructuralResidualAnalysis(structural_variant, results);
        cout << "  wrote_structural_results: "
             << structuralResultPath(structural_variant) << endl;
        cout << "  wrote_phase_diagnostics: "
             << structuralDiagnosticPath(structural_variant) << endl;
        cout << "  wrote_residual_features: "
             << structuralResidualPath(structural_variant) << endl;
    }
}

void ClusteredSpmvBenchFixture::runNaiveCooRoundRobinBV4Suite()
{
    constexpr uint64_t kNaiveClusters = kSparsePimBankGroups;
    const vector<string> matrices{
        "ASIC_100k", "Stanford", "bcsstk32", "cant", "consph", "crankseg_2",
        "ct20stif", "lhr71", "ohne2", "pdb1HYS", "pwtk", "rma10", "shipsec1",
        "soc-sign-epinions", "webbase-1M", "xenon2",
    };

    string only_matrix = envString("SPMV_BENCH_MATRIX");
    bool matched = false;
    vector<StructuralModelResult> results;
    cout << ">>Naive original-order COO round-robin DRAF+BGA BV-4 suite\n"
         << "  mapping: cluster_id = original_column_id % " << kNaiveClusters << "\n"
         << "  clustering: none\n"
         << "  column_reordering: none" << endl;

    for (const string& matrix : matrices)
    {
        if (!only_matrix.empty() && matrix != only_matrix)
            continue;
        matched = true;
        resetPIMKernel();
        string path = "../SparsePIM/sparse_matrix_coo/" + matrix + "_coo.txt";
        SpmvInputs inputs = loadNaiveCooInputs(path, kNaiveClusters);
        results.push_back(runDrafBgaStructuralModelFromInputs(
            move(inputs), "SparsePIM/sparse_matrix_coo/" + matrix + "_coo.txt",
            matrix, 1004));
    }
    if (!only_matrix.empty())
    {
        ASSERT_TRUE(matched) << "unknown SPMV_BENCH_MATRIX=" << only_matrix;
    }

    const string output_path = "../SparsePIM/naive_coo_round_robin_bv4_64bg_results.csv";
    ofstream out(output_path);
    if (!out)
        throw runtime_error("failed to open " + output_path);
    out << "matrix,mapping,clusters,model_ms,total_cycle,setup_cycle,"
           "draf_row_fetch_cycle,draf_compute_trigger_cycle,padding_cycle,"
           "bga_accumulate_cycle,bga_output_readback_cycle,final_reduce_cycle,"
           "draf_padding_ratio,draf_memory_expansion,bg_imbalance\n";
    for (const StructuralModelResult& result : results)
    {
        out << result.matrix << ",column_mod_64," << kNaiveClusters << ","
            << result.model_ms << "," << result.total_cycle << ","
            << result.setup_cycle << "," << result.draf_row_fetch_cycle << ","
            << result.draf_compute_trigger_cycle << "," << result.padding_cycle << ","
            << result.bga_accumulate_cycle << "," << result.bga_output_readback_cycle << ","
            << result.final_reduce_cycle << "," << result.draf_padding_ratio << ","
            << result.draf_memory_expansion << "," << result.bg_imbalance << "\n";
    }
    cout << "  wrote_results: " << output_path << endl;
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_cluster_cantcoo_draf_bga_model)
{
    runDrafBgaModel("../SparsePIM/cluster_cantcoo/", "SparsePIM/cluster_cantcoo");
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_cluster_cantcoo_draf_bga_conservative_model)
{
    runDrafBgaModel("../SparsePIM/cluster_cantcoo/", "SparsePIM/cluster_cantcoo", true);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_hybrid_hash_partition_v2_cantcoo_draf_bga_model)
{
    runDrafBgaModel("../SparsePIM/hybrid_hash_partition_v2_cantcoo/",
                    "SparsePIM/hybrid_hash_partition_v2_cantcoo");
}

TEST_F(ClusteredSpmvBenchFixture,
       sparsepim_hybrid_hash_partition_v2_cantcoo_draf_bga_conservative_model)
{
    runDrafBgaModel("../SparsePIM/hybrid_hash_partition_v2_cantcoo/",
                    "SparsePIM/hybrid_hash_partition_v2_cantcoo", true);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_model)
{
    runGuidedKmeansDrafBgaSuite();
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_conservative_model)
{
    runGuidedKmeansDrafBgaSuite(true);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v2_model)
{
    runGuidedKmeansDrafBgaSuite(false, true);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v2_conservative_model)
{
    runGuidedKmeansDrafBgaSuite(true, true);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v21_model)
{
    runGuidedKmeansDrafBgaSuite(false, true, true);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v21_conservative_model)
{
    runGuidedKmeansDrafBgaSuite(true, true, true);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v3_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(3);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v4_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(4);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v5_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(5);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v6_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(6);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v7_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(7);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v8_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(8);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v9_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(9);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v10_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(10);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v11_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(11);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v12_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(12);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v13_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(13);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v14_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(14);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v15_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(15);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v16_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(16);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v17_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(17);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v17_1_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(171);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v18_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(18);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_bv1_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(1001);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_bv2_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(1002);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_bv3_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(1003);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_bv4_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(1004);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_naive_coo_round_robin_draf_bga_bv4_structural_model)
{
    runNaiveCooRoundRobinBV4Suite();
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_minhash_binpack_v1_cantcoo_draf_bga_bv4_structural_model)
{
    vector<StructuralModelResult> results;
    resetPIMKernel();
    results.push_back(runDrafBgaStructuralModel(
        "../SparsePIM/minhash_binpack_v1_cantcoo/",
        "SparsePIM/minhash_binpack_v1_cantcoo", "cant", 2004));
    writeStructuralResults(2004, results);
    writeStructuralDiagnostics(2004, results);
    writeStructuralResidualAnalysis(2004, results);
    cout << "  wrote_structural_results: " << structuralResultPath(2004) << endl;
    cout << "  wrote_phase_diagnostics: " << structuralDiagnosticPath(2004) << endl;
    cout << "  wrote_residual_features: " << structuralResidualPath(2004) << endl;
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v4_draf_memory_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(104);
}
