#ifndef SPMV_DRAF_BGA_STRUCTURAL_BV4_H
#define SPMV_DRAF_BGA_STRUCTURAL_BV4_H

#include <string>

#include "tests/SpmvDrafBgaStructuralTypes.h"
#include "tests/SpmvDrafBgaStructuralVariants.h"

namespace spmv
{

StructuralModelResult runDrafBgaStructuralModelBV4(
    const SpmvInputs& inputs, const DrafStats& draf, const BgaStats& bga,
    const ShapeStats& shape, const DrafCriticalPathStats& critical,
    const DrafPaddingExposureStats& exposure, const StructuralBaseTiming& timing,
    const StructuralVariantSpec& variant_spec, const std::string& matrix_name,
    double gpu_ms, double target_speedup, double tck_ns,
    bool use_row_partition_summary = true);

} // namespace spmv

#endif
