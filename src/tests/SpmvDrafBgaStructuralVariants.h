#ifndef SPMV_DRAF_BGA_STRUCTURAL_VARIANTS_H
#define SPMV_DRAF_BGA_STRUCTURAL_VARIANTS_H

#include <string>
#include <vector>

namespace spmv
{

struct StructuralVariantSpec
{
    int id = 0;
    const char* name = "";
    const char* result_path = "";
    const char* diagnostic_path = "";
    const char* residual_path = "";
    const char* model_banner = "";
    const char* suite_label = "";
    const char* cycle_label = "";
    const char* speedup_label = "";
    const char* result_csv_tag = "";
    bool v4_draf_memory_only = false;
    bool critical_padding = false;
    bool exposure_padding = false;
    bool bga_overlap = false;
    bool draf_streaming_overlap = false;
    bool draf_bga_budget = false;
    bool reuse_aware_bga = false;
    bool draf_memory_efficiency = false;
    bool conservative_overlap_reuse = false;
    bool stronger_draf_memory_conservative = false;
    bool phase_exposure = false;
};

const std::vector<StructuralVariantSpec>& structuralVariantSpecs();
const StructuralVariantSpec& structuralVariantSpec(int structural_variant);
std::string structuralVariantName(int structural_variant);
std::string structuralResultPath(int structural_variant);
std::string structuralDiagnosticPath(int structural_variant);
std::string structuralResidualPath(int structural_variant);

} // namespace spmv

#endif
