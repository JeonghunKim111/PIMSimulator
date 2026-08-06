#include "csc/CSCFp16.h"

#include <stdexcept>

namespace csc_descriptor {

static_assert(sizeof(CSCFp16) == sizeof(CSCFp16Bits),
              "half_float::half must have a 16-bit representation");
CSCFp16 cscFp16FromBits(CSCFp16Bits bits)
{
    return fp16i(bits).fval;
}

CSCFp16Bits cscFp16ToBits(CSCFp16 value)
{
    return fp16i(value).ival;
}

CSCFp16 cscFp16FromFloat(float value)
{
    return CSCFp16(value);
}

CSCFp16 cscFp16FromDouble(double value)
{
    return CSCFp16(value);
}

float cscFp16ToFloat(CSCFp16 value)
{
    return static_cast<float>(value);
}

double cscFp16ToDouble(CSCFp16 value)
{
    return static_cast<double>(value);
}

CSCFp16 cscFp16Mul(CSCFp16 lhs, CSCFp16 rhs)
{
    return lhs * rhs;
}

CSCFp16 cscFp16Add(CSCFp16 lhs, CSCFp16 rhs)
{
    return lhs + rhs;
}

CSCFp16 cscFp16MulThenAdd(CSCFp16 accumulator,
                          CSCFp16 lhs,
                          CSCFp16 rhs)
{
    const CSCFp16 product = cscFp16Mul(lhs, rhs);
    return cscFp16Add(accumulator, product);
}

void cscAppendU16LE(std::vector<uint8_t>& bytes, uint16_t value)
{
    bytes.push_back(static_cast<uint8_t>(value & 0xffU));
    bytes.push_back(static_cast<uint8_t>((value >> 8) & 0xffU));
}

uint16_t cscReadU16LE(const std::vector<uint8_t>& bytes, std::size_t offset)
{
    if (offset > bytes.size() || bytes.size() - offset < 2)
        throw std::out_of_range("FP16 little-endian read outside buffer");
    return static_cast<uint16_t>(bytes[offset]) |
           static_cast<uint16_t>(bytes[offset + 1]) << 8;
}

void cscAppendFp16LE(std::vector<uint8_t>& bytes, CSCFp16 value)
{
    cscAppendU16LE(bytes, cscFp16ToBits(value));
}

CSCFp16 cscReadFp16LE(const std::vector<uint8_t>& bytes,
                     std::size_t offset)
{
    return cscFp16FromBits(cscReadU16LE(bytes, offset));
}

void CSCFp16LogicalMatrix::validate() const
{
    if (col_ptr.size() != static_cast<std::size_t>(cols) + 1 ||
        row_idx.size() != value_bits.size() ||
        col_ptr.empty() || col_ptr.front() != 0 ||
        col_ptr.back() != value_bits.size())
        throw std::invalid_argument("invalid logical FP16 CSC arrays");
    for (std::size_t column = 0; column < cols; ++column)
        if (col_ptr[column] > col_ptr[column + 1])
            throw std::invalid_argument("non-monotonic FP16 CSC column pointer");
    for (const uint32_t row : row_idx)
        if (row >= rows)
            throw std::invalid_argument("FP16 CSC row index outside matrix");
}

std::vector<CSCFp16Bits> cscFp16SequentialSpMV(
    const CSCFp16LogicalMatrix& matrix,
    const std::vector<CSCFp16Bits>& x_bits)
{
    matrix.validate();
    if (x_bits.size() != matrix.cols)
        throw std::invalid_argument("FP16 x length does not match matrix columns");
    std::vector<CSCFp16> y(matrix.rows, cscFp16FromBits(0));
    for (uint32_t column = 0; column < matrix.cols; ++column) {
        const CSCFp16 x = cscFp16FromBits(x_bits[column]);
        for (uint64_t index = matrix.col_ptr[column];
             index < matrix.col_ptr[column + 1]; ++index) {
            const uint32_t row = matrix.row_idx[index];
            y[row] = cscFp16MulThenAdd(
                y[row], cscFp16FromBits(matrix.value_bits[index]), x);
        }
    }
    std::vector<CSCFp16Bits> result;
    result.reserve(y.size());
    for (const CSCFp16 value : y) result.push_back(cscFp16ToBits(value));
    return result;
}

}  // namespace csc_descriptor
