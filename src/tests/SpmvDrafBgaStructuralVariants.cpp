#include "tests/SpmvDrafBgaStructuralVariants.h"

#include <stdexcept>

namespace spmv
{

const std::vector<StructuralVariantSpec>& structuralVariantSpecs()
{
    static const std::vector<StructuralVariantSpec> specs{
        {3,
         "v3_total_padding",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v3_structural_results.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v3_phase_diagnostics.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v3_residual_features.txt",
         "V3 Structural Model",
         "V3 structural suite",
         "v3_structural_cycle: ",
         "v3_structural_speedup: ",
         "V3_RESULT_CSV,",
         false, false, false, false, false, false, false, false, false, false, false},
        {4,
         "v4_critical_path",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v4_structural_results.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v4_phase_diagnostics.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v4_residual_features.txt",
         "V4 Critical-Path Structural Model",
         "V4 critical-path structural suite",
         "v4_structural_cycle: ",
         "v4_structural_speedup: ",
         "V4_RESULT_CSV,",
         false, true, false, false, false, false, false, false, false, false, false},
        {5,
         "v5_exposure",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v5_structural_results.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v5_phase_diagnostics.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v5_residual_features.txt",
         "V5 Exposure Structural Model",
         "V5 exposure structural suite",
         "v5_structural_cycle: ",
         "v5_structural_speedup: ",
         "V5_RESULT_CSV,",
         false, true, true, false, false, false, false, false, false, false, false},
        {6,
         "v6_bga_overlap",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v6_structural_results.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v6_phase_diagnostics.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v6_residual_features.txt",
         "V6 BGA-Overlap Structural Model",
         "V6 BGA-overlap structural suite",
         "v6_structural_cycle: ",
         "v6_structural_speedup: ",
         "V6_RESULT_CSV,",
         false, true, true, true, false, false, false, false, false, false, false},
        {7,
         "v7_draf_streaming_overlap",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v7_structural_results.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v7_phase_diagnostics.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v7_residual_features.txt",
         "V7 DRAF-Streaming-Overlap Structural Model",
         "V7 DRAF-streaming-overlap structural suite",
         "v7_structural_cycle: ",
         "v7_structural_speedup: ",
         "V7_RESULT_CSV,",
         false, true, true, true, true, false, false, false, false, false, false},
        {8,
         "v8_draf_bga_budgeted_streaming_overlap",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v8_structural_results.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v8_phase_diagnostics.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v8_residual_features.txt",
         "V8 DRAF-BGA-Budgeted-Streaming Structural Model",
         "V8 DRAF-BGA-budgeted-streaming structural suite",
         "v8_structural_cycle: ",
         "v8_structural_speedup: ",
         "V8_RESULT_CSV,",
         false, true, true, true, true, true, false, false, false, false, false},
        {9,
         "v9_reuse_aware_bga",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v9_structural_results.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v9_phase_diagnostics.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v9_residual_features.txt",
         "V9 Reuse-Aware-BGA Structural Model",
         "V9 reuse-aware-BGA structural suite",
         "v9_structural_cycle: ",
         "v9_structural_speedup: ",
         "V9_RESULT_CSV,",
         false, true, true, true, true, true, true, false, false, false, false},
        {10,
         "v10_draf_memory_efficiency",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v10_structural_results.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v10_phase_diagnostics.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v10_residual_features.txt",
         "V10 DRAF-Memory-Efficiency Structural Model",
         "V10 DRAF-memory-efficiency structural suite",
         "v10_structural_cycle: ",
         "v10_structural_speedup: ",
         "V10_RESULT_CSV,",
         false, true, true, true, true, true, true, true, false, false, false},
        {11,
         "v11_conservative_overlap_reuse",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v11_structural_results.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v11_phase_diagnostics.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v11_residual_features.txt",
         "V11 Conservative-Overlap-Reuse Structural Model",
         "V11 conservative-overlap-reuse structural suite",
         "v11_structural_cycle: ",
         "v11_structural_speedup: ",
         "V11_RESULT_CSV,",
         false, true, true, true, true, true, true, true, true, false, false},
        {12,
         "v12_stronger_draf_memory_conservative_overlap",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v12_structural_results.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v12_phase_diagnostics.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v12_residual_features.txt",
         "V12 Stronger-DRAF-Memory Conservative-Overlap Structural Model",
         "V12 stronger-DRAF-memory conservative-overlap structural suite",
         "v12_structural_cycle: ",
         "v12_structural_speedup: ",
         "V12_RESULT_CSV,",
         false, true, true, true, true, true, true, true, true, true, false},
        {13,
         "v13_phase_exposure_classified_shared_overlap",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v13_structural_results.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v13_phase_diagnostics.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v13_residual_features.txt",
         "V13 Phase-Exposure Classified Shared-Overlap Structural Model",
         "V13 phase-exposure classified shared-overlap structural suite",
         "v13_structural_cycle: ",
         "v13_structural_speedup: ",
         "V13_RESULT_CSV,",
         false, true, true, true, true, true, true, true, true, true, true},
        {14,
         "v14_bounded_draf_exposure_budget",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v14_structural_results.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v14_phase_diagnostics.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v14_residual_features.txt",
         "V14 Bounded-DRAF-Exposure-Budget Structural Model",
         "V14 bounded-DRAF-exposure-budget structural suite",
         "v14_structural_cycle: ",
         "v14_structural_speedup: ",
         "V14_RESULT_CSV,",
         false, true, true, true, true, true, true, true, true, true, true},
        {15,
         "v15_soft_bounded_reuse_protected_bga",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v15_structural_results.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v15_phase_diagnostics.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v15_residual_features.txt",
         "V15 Soft-Bounded Saving Budget with Reuse-Protected BGA Structural Model",
         "V15 soft-bounded reuse-protected-BGA structural suite",
         "v15_structural_cycle: ",
         "v15_structural_speedup: ",
         "V15_RESULT_CSV,",
         false, true, true, true, true, true, true, true, true, true, true},
        {104,
         "v4_draf_memory_saving_only",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v4_draf_memory_structural_results.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v4_draf_memory_phase_diagnostics.txt",
         "Guided_kmeans_result/spmv_guided_kmeans_draf_bga_v4_draf_memory_residual_features.txt",
         "V4 DRAF-Memory-Saving-Only Structural Model",
         "V4 DRAF-memory-saving-only structural suite",
         "v4_draf_memory_structural_cycle: ",
         "v4_draf_memory_structural_speedup: ",
         "V4_DRAF_MEMORY_RESULT_CSV,",
         true, true, false, false, false, false, false, true, false, false, false},
    };
    return specs;
}

const StructuralVariantSpec& structuralVariantSpec(int structural_variant)
{
    const std::vector<StructuralVariantSpec>& specs = structuralVariantSpecs();
    for (const StructuralVariantSpec& spec : specs)
    {
        if (spec.id == structural_variant)
            return spec;
    }
    throw std::invalid_argument("unknown structural variant: " +
                                std::to_string(structural_variant));
}

std::string structuralVariantName(int structural_variant)
{
    return structuralVariantSpec(structural_variant).name;
}

std::string structuralResultPath(int structural_variant)
{
    return structuralVariantSpec(structural_variant).result_path;
}

std::string structuralDiagnosticPath(int structural_variant)
{
    return structuralVariantSpec(structural_variant).diagnostic_path;
}

std::string structuralResidualPath(int structural_variant)
{
    return structuralVariantSpec(structural_variant).residual_path;
}

} // namespace spmv
