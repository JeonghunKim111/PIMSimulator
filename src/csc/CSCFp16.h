#ifndef CSC_FP16_H
#define CSC_FP16_H

#include "FP16.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace csc_descriptor {

using CSCFp16 = half_float::half;
using CSCFp16Bits = uint16_t;

CSCFp16 cscFp16FromBits(CSCFp16Bits bits);
CSCFp16Bits cscFp16ToBits(CSCFp16 value);
CSCFp16 cscFp16FromFloat(float value);
CSCFp16 cscFp16FromDouble(double value);
float cscFp16ToFloat(CSCFp16 value);
double cscFp16ToDouble(CSCFp16 value);
CSCFp16 cscFp16Mul(CSCFp16 lhs, CSCFp16 rhs);
CSCFp16 cscFp16Add(CSCFp16 lhs, CSCFp16 rhs);
CSCFp16 cscFp16MulThenAdd(CSCFp16 accumulator,
                          CSCFp16 lhs,
                          CSCFp16 rhs);

void cscAppendU16LE(std::vector<uint8_t>& bytes, uint16_t value);
uint16_t cscReadU16LE(const std::vector<uint8_t>& bytes, std::size_t offset);
void cscAppendFp16LE(std::vector<uint8_t>& bytes, CSCFp16 value);
CSCFp16 cscReadFp16LE(const std::vector<uint8_t>& bytes,
                     std::size_t offset);

struct CSCFp16LogicalMatrix {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<uint64_t> col_ptr;
    std::vector<uint32_t> row_idx;
    std::vector<CSCFp16Bits> value_bits;

    void validate() const;
};

// Conventional column-major traversal. This is a sequential FP16 reference,
// not the future architectural BGA/transport event-trace reference.
std::vector<CSCFp16Bits> cscFp16SequentialSpMV(
    const CSCFp16LogicalMatrix& matrix,
    const std::vector<CSCFp16Bits>& x_bits);

}  // namespace csc_descriptor

#endif

