#ifndef SPMV_DRAF_BGA_STRUCTURAL_V17_1_H
#define SPMV_DRAF_BGA_STRUCTURAL_V17_1_H

#include <string>

#include "tests/SpmvDrafBgaStructuralTypes.h"
#include "tests/SpmvDrafBgaStructuralVariants.h"

namespace spmv
{

StructuralModelResult runDrafBgaStructuralModelV17_1(
    const SpmvInputs& inputs, const DrafStats& draf, const BgaStats& bga,
    const ShapeStats& shape, const DrafCriticalPathStats& critical,
    const DrafPaddingExposureStats& exposure, const StructuralBaseTiming& timing,
    const StructuralVariantSpec& variant_spec, const std::string& matrix_name,
    double gpu_ms, double target_speedup, double tck_ns);

} // namespace spmv

#endif
