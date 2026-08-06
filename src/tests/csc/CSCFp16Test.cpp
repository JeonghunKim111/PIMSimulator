#include "csc/CSCFp16.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace csc_descriptor;

TEST(CSCFp16BitsTest, AllPatternsRoundTripThroughBitsAndLittleEndian)
{
    std::vector<uint8_t> serialized;
    serialized.reserve(65536 * 2);
    for (uint32_t raw = 0; raw <= UINT16_MAX; ++raw) {
        const auto value = cscFp16FromBits(static_cast<uint16_t>(raw));
        ASSERT_EQ(cscFp16ToBits(value), raw);
        cscAppendFp16LE(serialized, value);
    }
    ASSERT_EQ(serialized.size(), 65536U * 2U);
    for (uint32_t raw = 0; raw <= UINT16_MAX; ++raw)
        ASSERT_EQ(cscFp16ToBits(cscReadFp16LE(serialized, raw * 2)), raw);
    EXPECT_THROW(cscReadU16LE(serialized, serialized.size() - 1),
                 std::out_of_range);
}

TEST(CSCFp16BitsTest, SerializationIsExplicitLittleEndian)
{
    std::vector<uint8_t> bytes;
    cscAppendU16LE(bytes, 0x1234U);
    cscAppendFp16LE(bytes, cscFp16FromBits(0xabcdU));
    ASSERT_EQ(bytes, (std::vector<uint8_t>{0x34, 0x12, 0xcd, 0xab}));
    EXPECT_EQ(cscReadU16LE(bytes, 0), 0x1234U);
    EXPECT_EQ(cscFp16ToBits(cscReadFp16LE(bytes, 2)), 0xabcdU);
}

TEST(CSCFp16ArithmeticTest, ContractSpecialValuesAndRounding)
{
    EXPECT_EQ(cscFp16ToBits(cscFp16FromFloat(0.0F)), 0x0000U);
    EXPECT_EQ(cscFp16ToBits(cscFp16FromFloat(-0.0F)), 0x8000U);
    EXPECT_EQ(cscFp16ToBits(cscFp16FromDouble(
                  1.0 + std::ldexp(1.0, -11))), 0x3c00U);
    EXPECT_EQ(cscFp16ToBits(cscFp16FromDouble(65504.0)), 0x7bffU);
    EXPECT_EQ(cscFp16ToBits(cscFp16FromDouble(70000.0)), 0x7c00U);
    EXPECT_EQ(cscFp16ToBits(cscFp16FromDouble(std::ldexp(1.0, -14))),
              0x0400U);
    EXPECT_EQ(cscFp16ToBits(cscFp16FromDouble(std::ldexp(1.0, -24))),
              0x0001U);
    EXPECT_EQ(cscFp16ToBits(cscFp16FromDouble(std::ldexp(1.0, -25))),
              0x0000U);
    EXPECT_EQ(cscFp16ToBits(cscFp16FromFloat(
                  std::numeric_limits<float>::infinity())), 0x7c00U);
    EXPECT_EQ(cscFp16ToBits(cscFp16FromFloat(
                  -std::numeric_limits<float>::infinity())), 0xfc00U);
    const auto nan = cscFp16ToBits(cscFp16FromFloat(
        std::numeric_limits<float>::quiet_NaN()));
    EXPECT_EQ(nan & 0x7c00U, 0x7c00U);
    EXPECT_NE(nan & 0x03ffU, 0U);
    EXPECT_EQ(cscFp16ToBits(cscFp16Add(cscFp16FromBits(0x3c00U),
                                      cscFp16FromBits(0xbc00U))), 0x0000U);
    EXPECT_EQ(cscFp16ToBits(cscFp16Mul(cscFp16FromBits(0x8000U),
                                      cscFp16FromBits(0x3c00U))), 0x8000U);
}

TEST(CSCFp16ArithmeticTest, MulThenAddIsSeparatedAndNotFused)
{
    const auto lhs = cscFp16FromBits(0x3c01U);
    const auto rhs = cscFp16FromBits(0x3c01U);
    const auto accumulator = cscFp16FromBits(0xbc02U);
    const auto product = cscFp16Mul(lhs, rhs);
    const auto separated = cscFp16Add(accumulator, product);
    const auto helper = cscFp16MulThenAdd(accumulator, lhs, rhs);
    const auto fused = half_float::fma(lhs, rhs, accumulator);
    EXPECT_EQ(cscFp16ToBits(product), 0x3c02U);
    EXPECT_EQ(cscFp16ToBits(helper), cscFp16ToBits(separated));
    EXPECT_NE(cscFp16ToBits(helper), cscFp16ToBits(fused));
}

TEST(CSCFp16BitsTest, FiniteFloatRoundTripPreservesRepresentableBits)
{
    for (uint32_t raw = 0; raw <= UINT16_MAX; ++raw) {
        if ((raw & 0x7c00U) == 0x7c00U) continue;
        const auto value = cscFp16FromBits(static_cast<uint16_t>(raw));
        EXPECT_EQ(cscFp16ToBits(cscFp16FromFloat(cscFp16ToFloat(value))), raw);
        EXPECT_EQ(cscFp16ToBits(cscFp16FromDouble(cscFp16ToDouble(value))), raw);
    }
}

TEST(CSCFp16SequentialReferenceTest, RoundsEveryMultiplyAndAddInCSCOrder)
{
    CSCFp16LogicalMatrix matrix;
    matrix.rows = 3;
    matrix.cols = 3;
    matrix.col_ptr = {0, 2, 4, 5};
    matrix.row_idx = {0, 1, 0, 2, 0};
    matrix.value_bits = {0x3c01U, 0x4000U, 0x3c01U, 0xc200U, 0x3c01U};
    const std::vector<CSCFp16Bits> x = {0x3c01U, 0x3c01U, 0xbc02U};

    const auto y = cscFp16SequentialSpMV(matrix, x);
    ASSERT_EQ(y.size(), 3U);
    CSCFp16 row0 = cscFp16FromBits(0);
    row0 = cscFp16MulThenAdd(row0, cscFp16FromBits(0x3c01U),
                            cscFp16FromBits(x[0]));
    row0 = cscFp16MulThenAdd(row0, cscFp16FromBits(0x3c01U),
                            cscFp16FromBits(x[1]));
    row0 = cscFp16MulThenAdd(row0, cscFp16FromBits(0x3c01U),
                            cscFp16FromBits(x[2]));
    EXPECT_EQ(y[0], cscFp16ToBits(row0));
    EXPECT_EQ(y[1], cscFp16ToBits(cscFp16Mul(
                        cscFp16FromBits(0x4000U), cscFp16FromBits(x[0]))));
    EXPECT_EQ(y[2], cscFp16ToBits(cscFp16Mul(
                        cscFp16FromBits(0xc200U), cscFp16FromBits(x[1]))));
}

TEST(CSCFp16SequentialReferenceTest, RejectsMalformedMatrixAndX)
{
    CSCFp16LogicalMatrix matrix{1, 1, {0, 1}, {0}, {0x3c00U}};
    EXPECT_THROW(cscFp16SequentialSpMV(matrix, {}), std::invalid_argument);
    matrix.row_idx[0] = 1;
    EXPECT_THROW(cscFp16SequentialSpMV(matrix, {0x3c00U}),
                 std::invalid_argument);
}
