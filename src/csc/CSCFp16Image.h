#ifndef CSC_FP16_IMAGE_H
#define CSC_FP16_IMAGE_H

#include "csc/CSCFp16.h"
#include "csc/CSCTypes.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace csc_descriptor {

constexpr uint32_t kCSCFp16ImageVersion = 2;
constexpr uint32_t kCSCFp16ImageBGCount = 64;
constexpr uint32_t kCSCFp16ImageBurstBytes = 32;
constexpr uint32_t kCSCFp16ImageFutureSIMDWidth = 16;
constexpr uint32_t kCSCFp16ImageValueBytes = 2;
constexpr uint32_t kCSCFp16ImageIndexBytes = 4;
constexpr uint32_t kCSCFp16ImageDescriptorBytes = 32;

struct CSCFp16ImageSource {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<uint64_t> col_ptr;
    std::vector<uint32_t> row_idx;
    std::vector<double> values;
    std::vector<double> x;
    std::vector<uint32_t> column_to_bg;

    void validate() const;
};

struct CSCFp16ConversionStats {
    uint64_t conversion_count = 0;
    uint64_t positive_infinity_count = 0;
    uint64_t negative_infinity_count = 0;
    uint64_t nan_count = 0;
    uint64_t subnormal_count = 0;
    uint64_t underflow_to_zero_count = 0;
    uint64_t signed_zero_count = 0;
};

struct CSCFp16ImageAccounting {
    uint64_t descriptor_count = 0;
    uint64_t logical_value_bytes = 0;
    uint64_t physical_value_bytes = 0;
    uint64_t value_padding_bytes = 0;
    uint64_t logical_index_bytes = 0;
    uint64_t physical_index_bytes = 0;
    uint64_t index_padding_bytes = 0;
    uint64_t logical_x_bytes = 0;
    uint64_t physical_x_bytes = 0;
    uint64_t x_padding_bytes = 0;
    uint64_t descriptor_bytes = 0;
};

struct CSCFp16ImageBG {
    std::vector<uint8_t> values;
    std::vector<uint8_t> row_indices;
    std::vector<uint8_t> descriptors;
    std::vector<uint8_t> x_permutation;
    std::vector<CSCDescriptor> parsed_descriptors;
};

struct CSCFp16LoadedImage {
    CSCFp16LogicalMatrix matrix;
    std::vector<CSCFp16Bits> x_bits;
    std::vector<uint32_t> column_to_bg;
    std::array<CSCFp16ImageBG, kCSCFp16ImageBGCount> bg;
    CSCFp16ConversionStats conversion;
    CSCFp16ImageAccounting accounting;
};

uint64_t cscFp16ImageFnv1a64(const std::vector<uint8_t>& bytes);
void exportCSCFp16ImageV2(const CSCFp16ImageSource& source,
                         const std::string& directory);
CSCFp16LoadedImage loadCSCFp16ImageV2(const std::string& directory);

}  // namespace csc_descriptor

#endif

