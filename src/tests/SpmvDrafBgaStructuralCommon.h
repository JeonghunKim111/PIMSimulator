#ifndef SPMV_DRAF_BGA_STRUCTURAL_COMMON_H
#define SPMV_DRAF_BGA_STRUCTURAL_COMMON_H

#include <cstdint>
#include <string>

#include "tests/SpmvDrafBgaStructuralTypes.h"

namespace spmv
{

inline constexpr unsigned kElementsPerBurst = 16;
inline constexpr unsigned kDrafNzesPerColumnGroup = 16;
inline constexpr unsigned kDrafColumnGroupsPerRow = 7;
inline constexpr unsigned kBgaEntriesPerBacc = 8;
inline constexpr unsigned kBgaQueueDepth = 16;
inline constexpr unsigned kBgaFlushThreshold = 9;
inline constexpr unsigned kConservativeBgaFlushPenalty = 4;
inline constexpr unsigned kDefaultV2BgaAccCapacity = 64;
inline constexpr double kBgaDuplicatePartialWeight = 0.25;
inline constexpr double kSparseValueBytes = 2.0;
inline constexpr double kCooIndexBytesPerNnz = 8.0;
inline constexpr double kDrafCompactMetadataBytesPerNnz = 5.0;

uint64_t ceilDiv(uint64_t value, uint64_t divisor);
double normalizeRatio(double value, double scale);
double clamp01(double value);

SpmvInputs loadSparsePIMInputs(const std::string& matrix_path,
                               const std::string& permutation_path,
                               const std::string& clusters_path);
void applyClusterLimit(SpmvInputs& inputs, uint64_t max_clusters);
DrafStats buildDrafStats(const SpmvInputs& inputs);
BgaStats buildBgaStats(const SpmvInputs& inputs,
                       uint64_t bga_acc_capacity = kDefaultV2BgaAccCapacity);
ShapeStats buildShapeStats(const SpmvInputs& inputs, const DrafStats& draf,
                           const BgaStats& bga);
DrafCriticalPathStats buildDrafCriticalPathStats(const DrafStats& draf);
DrafPaddingExposureStats buildDrafPaddingExposureStats(
    const DrafStats& draf, const ShapeStats& shape,
    const DrafCriticalPathStats& critical);
BgaOverlapStats buildBgaOverlapStats(uint64_t raw_bga_accumulate_cycle,
                                     uint64_t compute_overlap_window,
                                     const ShapeStats& shape,
                                     const DrafPaddingExposureStats& exposure);
DrafStreamingOverlapStats buildDrafStreamingOverlapStats(
    uint64_t draf_row_fetch_cycle, uint64_t draf_compute_trigger_cycle,
    uint64_t pre_overlap_total_cycle, uint64_t compute_overlap_window,
    const DrafPaddingExposureStats& exposure,
    const BgaOverlapStats& bga_overlap, bool budget_by_bga_overlap);
DrafMemoryEfficiencyStats buildDrafMemoryEfficiencyStats(
    const DrafStats& draf, const DrafPaddingExposureStats& exposure,
    uint64_t memory_bound_access_cycle);

} // namespace spmv

#endif
