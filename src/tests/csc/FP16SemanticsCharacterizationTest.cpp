#include "FP16.h"
#include "PIMBlock.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>

using namespace DRAMSim;

namespace {

uint16_t bits(fp16 value)
{
    return fp16i(value).ival;
}

fp16 fromBits(uint16_t value)
{
    return fp16i(value).fval;
}

}  // namespace

TEST(FP16SemanticsCharacterizationTest, AllBitPatternsPreserveStoredBits)
{
    for (uint32_t raw = 0; raw <= UINT16_MAX; ++raw)
        ASSERT_EQ(bits(fromBits(static_cast<uint16_t>(raw))), raw)
            << "raw=" << raw;
}

TEST(FP16SemanticsCharacterizationTest, ConversionUsesBinary16RoundToNearestEven)
{
    EXPECT_EQ(bits(fp16(0.0F)), 0x0000U);
    EXPECT_EQ(bits(fp16(-0.0F)), 0x8000U);
    EXPECT_EQ(bits(fp16(1.0F)), 0x3c00U);
    EXPECT_EQ(bits(fp16(1.0F + std::ldexp(1.0F, -11))), 0x3c00U);
    EXPECT_EQ(bits(fp16(1.0F + 3.0F * std::ldexp(1.0F, -11))),
              0x3c02U);
    EXPECT_EQ(bits(fp16(65504.0F)), 0x7bffU);
    EXPECT_EQ(bits(fp16(70000.0F)), 0x7c00U);
    EXPECT_EQ(bits(fp16(std::ldexp(1.0F, -14))), 0x0400U);
    EXPECT_EQ(bits(fp16(std::ldexp(1.0F, -24))), 0x0001U);
    EXPECT_EQ(bits(fp16(std::ldexp(1.0F, -25))), 0x0000U);
}

TEST(FP16SemanticsCharacterizationTest, SpecialValuesAndSignedZero)
{
    EXPECT_EQ(bits(fp16(std::numeric_limits<float>::infinity())),
              0x7c00U);
    EXPECT_EQ(bits(fp16(-std::numeric_limits<float>::infinity())),
              0xfc00U);
    const uint16_t nan_bits =
        bits(fp16(std::numeric_limits<float>::quiet_NaN()));
    EXPECT_EQ(nan_bits & 0x7c00U, 0x7c00U);
    EXPECT_NE(nan_bits & 0x03ffU, 0U);
    EXPECT_NE(nan_bits & 0x0200U, 0U);
    EXPECT_EQ(bits(fromBits(0x3c00U) + fromBits(0xbc00U)), 0x0000U);
    EXPECT_EQ(bits(fromBits(0x8000U) * fromBits(0x3c00U)), 0x8000U);
}

TEST(FP16SemanticsCharacterizationTest, AddAndMulRoundToHalfResults)
{
    EXPECT_EQ(bits(fromBits(0x3c01U) + fromBits(0x1000U)), 0x3c02U);
    EXPECT_EQ(bits(fromBits(0x3c01U) * fromBits(0x3c01U)), 0x3c02U);
    EXPECT_EQ(bits(fromBits(0x7bffU) + fromBits(0x7bffU)), 0x7c00U);
    EXPECT_EQ(bits(fromBits(0x0001U) * fromBits(0x3800U)), 0x0000U);
}

TEST(FP16SemanticsCharacterizationTest,
     PIMBlockUsesExistingHalfOperatorsAndSixteenLanes)
{
    PIMBlock pim(FP16);
    BurstType lhs;
    BurstType rhs;
    BurstType dst;
    for (uint32_t lane = 0; lane < 16; ++lane) {
        lhs.fp16Data_[lane] = fromBits(0x3c01U);
        rhs.fp16Data_[lane] = fromBits(0x3c01U);
        dst.fp16Data_[lane] = fromBits(0x3555U);
    }
    pim.mul(dst, lhs, rhs, 16);
    for (uint32_t lane = 0; lane < 16; ++lane)
        EXPECT_EQ(bits(dst.fp16Data_[lane]), 0x3c02U);
    EXPECT_EQ(pim.simdCounters().active_lanes, 16U);
    EXPECT_EQ(pim.simdCounters().masked_lanes, 0U);
}

TEST(FP16SemanticsCharacterizationTest,
     PIMBlockMacMaterializesMultiplyBeforeAdd)
{
    PIMBlock pim(FP16);
    BurstType lhs;
    BurstType rhs;
    BurstType dst;
    lhs.fp16Data_[0] = fromBits(0x3c01U);
    rhs.fp16Data_[0] = fromBits(0x3c01U);
    dst.fp16Data_[0] = fromBits(0xbc02U);

    const fp16 rounded_product = lhs.fp16Data_[0] * rhs.fp16Data_[0];
    const fp16 separated = rounded_product + dst.fp16Data_[0];
    const fp16 fused = half_float::fma(
        lhs.fp16Data_[0], rhs.fp16Data_[0], dst.fp16Data_[0]);
    ASSERT_NE(bits(separated), bits(fused));

    pim.mac(dst, lhs, rhs, 1);
    EXPECT_EQ(bits(dst.fp16Data_[0]), bits(separated));
    EXPECT_NE(bits(dst.fp16Data_[0]), bits(fused));
}

