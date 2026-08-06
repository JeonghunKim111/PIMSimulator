#include "csc/CSCFp16BankGroupAccumulator.h"

#include <gtest/gtest.h>

#include <cmath>

namespace csc_descriptor {
namespace {

CSCFp16PartialEvent partial(uint32_t row, CSCFp16Bits bits, uint32_t bg = 3,
                            uint32_t ordinal = 0)
{
    return {row, bits, bg, ordinal, 0, 0};
}

std::vector<CSCFp16BGAOutputEvent> run(
    CSCFp16BankGroupAccumulator& bga,
    const std::vector<CSCFp16PartialEvent>& input,
    uint32_t output_stall_cycles = 0)
{
    CSCFp16BoundedBGAOutputSink sink(64);
    std::size_t next = 0;
    bool done_sent = false;
    for (uint32_t guard = 0; guard < 10000 && !bga.finalDrainComplete(); ++guard) {
        bga.step();
        if (next < input.size() && bga.ready()) bga.accept(input[next++]);
        if (next == input.size() && !done_sent) {
            bga.markProducerDone();
            EXPECT_TRUE(bga.requestFinalDrain());
            done_sent = true;
        }
        if (bga.hasOutput() && guard >= output_stall_cycles && sink.ready()) {
            sink.accept(bga.peekOutput());
            bga.acceptOutput();
        }
    }
    EXPECT_TRUE(bga.finalDrainComplete());
    EXPECT_EQ(next, input.size());
    return sink.trace();
}

TEST(CSCFp16BankGroupAccumulatorTest, SameRowUsesImmediateFp16Rounding)
{
    CSCFp16BGAConfig config;
    config.rows = 8;
    config.accumulator_entries = config.compare_width = 4;
    CSCFp16BankGroupAccumulator bga(3, config);
    const auto one = cscFp16ToBits(cscFp16FromFloat(1.0f));
    const auto tiny = cscFp16ToBits(cscFp16FromFloat(0.00048828125f));
    const auto trace = run(bga, {partial(2, one), partial(2, tiny),
                                 partial(2, tiny)});
    ASSERT_EQ(trace.size(), 1U);
    CSCFp16 expected = cscFp16Add(cscFp16FromBits(one), cscFp16FromBits(tiny));
    expected = cscFp16Add(expected, cscFp16FromBits(tiny));
    EXPECT_EQ(trace[0].value_bits, cscFp16ToBits(expected));
    EXPECT_EQ(trace[0].reason, CSCFp16BGAOutputReason::FINAL_DRAIN);
    EXPECT_EQ(trace[0].contribution_count, 3U);
    EXPECT_EQ(bga.counters().fp16_adds, 2U);
    EXPECT_EQ(bga.counters().merges, 2U);
    EXPECT_TRUE(bga.conservationInvariant());
}

TEST(CSCFp16BankGroupAccumulatorTest, SpecialValuesUseSharedHalfSemantics)
{
    CSCFp16BGAConfig config;
    config.rows = 16;
    config.accumulator_entries = config.compare_width = 8;
    CSCFp16BankGroupAccumulator bga(3, config);
    const auto pos_zero = CSCFp16Bits{0x0000};
    const auto neg_zero = CSCFp16Bits{0x8000};
    const auto max = CSCFp16Bits{0x7bff};
    const auto subnormal = CSCFp16Bits{0x0001};
    const auto pos_inf = CSCFp16Bits{0x7c00};
    const auto neg_inf = CSCFp16Bits{0xfc00};
    const auto trace = run(bga, {
        partial(0, pos_zero), partial(0, neg_zero),
        partial(1, subnormal), partial(1, subnormal),
        partial(2, max), partial(2, max),
        partial(3, pos_inf), partial(3, neg_inf)});
    ASSERT_EQ(trace.size(), 4U);
    EXPECT_EQ(trace[0].value_bits,
              cscFp16ToBits(cscFp16Add(cscFp16FromBits(pos_zero),
                                       cscFp16FromBits(neg_zero))));
    EXPECT_EQ(trace[1].value_bits,
              cscFp16ToBits(cscFp16Add(cscFp16FromBits(subnormal),
                                       cscFp16FromBits(subnormal))));
    EXPECT_EQ(trace[2].value_bits, pos_inf);
    EXPECT_TRUE(std::isnan(cscFp16ToFloat(cscFp16FromBits(trace[3].value_bits))));
}

TEST(CSCFp16BankGroupAccumulatorTest, FullHitAvoidsEvictionAndMissEvictsOldest)
{
    CSCFp16BGAConfig config;
    config.rows = 16;
    config.input_queue_depth = 4;
    config.accumulator_entries = config.compare_width = 2;
    config.output_queue_depth = 1;
    CSCFp16BankGroupAccumulator bga(3, config);
    const auto one = cscFp16ToBits(cscFp16FromFloat(1.0f));
    const auto trace = run(bga, {partial(1, one), partial(2, one),
                                 partial(1, one), partial(3, one)}, 8);
    ASSERT_EQ(trace.size(), 3U);
    EXPECT_EQ(trace[0].row_idx, 1U);
    EXPECT_EQ(trace[0].reason, CSCFp16BGAOutputReason::CAPACITY_EVICTION);
    EXPECT_EQ(trace[0].contribution_count, 2U);
    EXPECT_EQ(trace[1].row_idx, 2U);
    EXPECT_EQ(trace[2].row_idx, 3U);
    EXPECT_EQ(bga.counters().capacity_evictions, 1U);
    EXPECT_EQ(bga.counters().final_drain_outputs, 2U);
    EXPECT_EQ(bga.counters().queue_high_water, 2U);
}

TEST(CSCFp16BankGroupAccumulatorTest, IngressAndOutputBackpressureAreBounded)
{
    CSCFp16BGAConfig config;
    config.rows = 16;
    config.input_queue_depth = 1;
    config.accumulator_entries = config.compare_width = 1;
    config.output_queue_depth = 1;
    CSCFp16BankGroupAccumulator bga(3, config);
    const auto one = cscFp16ToBits(cscFp16FromFloat(1.0f));
    ASSERT_TRUE(bga.ready());
    bga.accept(partial(1, one));
    EXPECT_FALSE(bga.ready());
    bga.step();
    EXPECT_FALSE(bga.ready());
    bga.step();
    EXPECT_TRUE(bga.ready());
    bga.accept(partial(2, one));
    bga.markProducerDone();
    ASSERT_TRUE(bga.requestFinalDrain());
    for (uint32_t i = 0; i < 10; ++i) bga.step();
    ASSERT_TRUE(bga.hasOutput());
    const auto held = bga.peekOutput();
    for (uint32_t i = 0; i < 4; ++i) {
        bga.step();
        EXPECT_EQ(bga.peekOutput(), held);
    }
    EXPECT_EQ(bga.peekOutput(), held);
    bga.acceptOutput();
    while (!bga.finalDrainComplete()) {
        bga.step();
        if (bga.hasOutput()) bga.acceptOutput();
    }
    EXPECT_EQ(bga.counters().ingress_accepted, 2U);
    EXPECT_EQ(bga.counters().retired_contributions, 2U);
}

TEST(CSCFp16BankGroupAccumulatorTest, RejectsWrongBankGroup)
{
    CSCFp16BGAConfig config;
    config.rows = 4;
    CSCFp16BankGroupAccumulator bga(3, config);
    EXPECT_THROW(bga.accept(partial(0, 0, 4)), std::invalid_argument);
}

CSCFp16PartialBatch batch(uint8_t valid, uint8_t batch_id = 0)
{
    CSCFp16PartialBatch result;
    result.valid_count = valid;
    result.global_bg_id = 3;
    result.descriptor_id = 4;
    result.chunk_id = 5;
    result.batch_id = batch_id;
    for (uint8_t i = 0; i < valid; ++i)
        result.entries[i] = partial(i, cscFp16ToBits(cscFp16FromFloat(i + 1)),
                                    3, 4);
    for (uint8_t i = 0; i < valid; ++i) {
        result.entries[i].descriptor_id = 4;
        result.entries[i].chunk_id = 5;
        result.entries[i].lane_id = batch_id * 8 + i;
        result.entries[i].chunk_valid_count = batch_id ? 8 + valid : valid;
    }
    return result;
}

TEST(CSCFp16BankGroupAccumulatorTest, Batch8AcceptanceIsAtomicAndOrdered)
{
    auto config = makeFp16IsoStructureProductionConfig(32);
    CSCFp16BankGroupAccumulator bga(3, config);
    const auto low = batch(8, 0);
    const auto high = batch(5, 1);
    EXPECT_TRUE(bga.canAcceptBatch(low));
    EXPECT_TRUE(bga.acceptBatch(low));
    EXPECT_FALSE(bga.canAcceptBatch(high));
    EXPECT_FALSE(bga.acceptBatch(high));
    EXPECT_EQ(bga.counters().ingress_accepted, 8U);
    bga.step();
    EXPECT_TRUE(bga.canAcceptBatch(high));
    EXPECT_TRUE(bga.acceptBatch(high));
    bga.step();
    EXPECT_EQ(bga.counters().batches_accepted, 2U);
    EXPECT_EQ(bga.counters().batch0_count, 1U);
    EXPECT_EQ(bga.counters().batch1_count, 1U);
    EXPECT_EQ(bga.counters().ingress_accepted, 13U);
    // One entry begins service before the second accepted batch commits.
    EXPECT_EQ(bga.counters().input_high_water, 12U);
}

TEST(CSCFp16BankGroupAccumulatorTest, NamedPresetsFixProductionAndStressShape)
{
    const auto serial = makeFp16SerialCompatibilityConfig(64);
    const auto production = makeFp16IsoStructureProductionConfig(64);
    const auto stress = makeFp16Q8StressConfig(64);
    EXPECT_EQ(serial.ingress_mode, CSCFp16BGAIngressMode::SERIAL_EVENT);
    EXPECT_EQ(serial.ingress_batch_width, 1U);
    EXPECT_EQ(serial.accumulator_entries, 16U);
    EXPECT_EQ(production.ingress_mode, CSCFp16BGAIngressMode::BATCH8);
    EXPECT_EQ(production.ingress_batch_width, 8U);
    EXPECT_EQ(production.input_queue_depth, 16U);
    EXPECT_EQ(production.accumulator_entries, 16U);
    EXPECT_EQ(production.compare_width, 16U);
    EXPECT_EQ(production.output_queue_depth, 16U);
    EXPECT_EQ(stress.ingress_mode, CSCFp16BGAIngressMode::BATCH8);
    EXPECT_EQ(stress.accumulator_entries, 8U);
    EXPECT_EQ(stress.compare_width, 8U);
}

}  // namespace
}  // namespace csc_descriptor
