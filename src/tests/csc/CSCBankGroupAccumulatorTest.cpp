#include "csc/CSCBankGroupAccumulator.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

using namespace csc_descriptor;

namespace {

CSCBGAConfig config(uint32_t streams = 1, uint32_t queue = 16,
                    uint32_t entries = 16, uint32_t output = 16,
                    uint32_t compare_latency = 1, uint32_t add_latency = 1)
{
    CSCBGAConfig c;
    c.input_streams = streams;
    c.input_queue_depth = queue;
    c.accumulator_entries = entries;
    c.compare_width = entries;
    c.compare_latency = compare_latency;
    c.add_latency = add_latency;
    c.output_queue_depth = output;
    c.rows = 1024;
    return c;
}

CSCBGABatch batch(uint32_t stream, uint32_t generation, uint64_t sequence,
                  std::initializer_list<CSCBGAPartial> partials)
{
    return {stream, generation, sequence, partials};
}

const CSCBGAAccumulatorEntry* findRow(const CSCBankGroupAccumulator& bga,
                                      uint32_t row)
{
    for (const auto& entry : bga.accumulator())
        if (entry.valid && entry.row_idx == row) return &entry;
    return nullptr;
}

void runUntil(CSCBankGroupAccumulator& bga,
              const std::function<bool()>& predicate, uint32_t limit = 1000)
{
    uint32_t steps = 0;
    while (!predicate() && steps++ < limit) bga.step();
    ASSERT_TRUE(predicate()) << "cycle guard exceeded at " << bga.cycle();
}

std::vector<CSCBGAOutput> drainOutputs(CSCBankGroupAccumulator& bga)
{
    std::vector<CSCBGAOutput> result;
    while (bga.hasOutput()) {
        result.push_back(bga.peekOutput());
        bga.acceptOutput();
        bga.step();
    }
    return result;
}

uint32_t bits(float value)
{
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float fromBits(uint32_t value)
{
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

}  // namespace

TEST(CSCBGAConfigTest, RejectsUnsupportedCompareWidthAndZeroFields)
{
    auto c = config();
    c.compare_width = c.accumulator_entries - 1;
    EXPECT_THROW((void)CSCBankGroupAccumulator{c}, std::invalid_argument);
    c = config();
    c.compare_latency = 0;
    EXPECT_THROW((void)CSCBankGroupAccumulator{c}, std::invalid_argument);
    c = config();
    c.rows = 0;
    EXPECT_THROW((void)CSCBankGroupAccumulator{c}, std::invalid_argument);
}

TEST(CSCBGAHandshakeTest, AtomicBatchDelayBackpressureAndIndependentStreams)
{
    CSCBankGroupAccumulator bga(config(2, 2, 4));
    EXPECT_EQ(bga.offerBatch(batch(0, 1, 1, {{3, 1.0F}, {4, 2.0F}})),
              CSCBGAInputResult::ACCEPTED);
    EXPECT_EQ(bga.inputQueueSize(0), 0U);
    bga.step();
    EXPECT_EQ(bga.inputQueueSize(0), 2U);
    EXPECT_EQ(bga.counters().accepted_valid_partials, 2U);

    EXPECT_EQ(bga.offerBatch(batch(0, 1, 2, {{5, 3.0F}, {6, 4.0F}})),
              CSCBGAInputResult::BACKPRESSURE);
    EXPECT_EQ(bga.offerBatch(batch(1, 1, 1, {{7, 5.0F}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();
    EXPECT_EQ(bga.inputQueueSize(1), 1U);
    EXPECT_EQ(bga.counters().accepted_valid_partials, 3U);
    EXPECT_TRUE(bga.conservationInvariant());

    runUntil(bga, [&] { return bga.inputQueueSize(0) == 0; });
    EXPECT_EQ(bga.offerBatch(batch(0, 1, 2, {{5, 3.0F}, {6, 4.0F}})),
              CSCBGAInputResult::ACCEPTED);
    EXPECT_EQ(bga.counters().accepted_batches, 3U);
    EXPECT_EQ(bga.counters().accepted_valid_partials, 5U);
    EXPECT_TRUE(bga.conservationInvariant());
}

TEST(CSCBGAHandshakeTest, BoundedExactlyOnceAndProtocolErrors)
{
    CSCBankGroupAccumulator bga(config());
    auto first = batch(0, 1, 1, {{3, 1.0F}});
    EXPECT_EQ(bga.offerBatch(first), CSCBGAInputResult::ACCEPTED);
    bga.step();
    EXPECT_EQ(bga.offerBatch(first), CSCBGAInputResult::DUPLICATE);
    EXPECT_EQ(bga.counters().accepted_valid_partials, 1U);

    auto conflict = first;
    conflict.partials[0].value = 9.0F;
    EXPECT_EQ(bga.offerBatch(conflict), CSCBGAInputResult::PROTOCOL_ERROR);
    EXPECT_EQ(bga.offerBatch(batch(0, 1, 3, {{4, 1.0F}})),
              CSCBGAInputResult::PROTOCOL_ERROR);
    EXPECT_EQ(bga.offerBatch(batch(0, 0, 2, {{4, 1.0F}})),
              CSCBGAInputResult::PROTOCOL_ERROR);

    EXPECT_EQ(bga.offerBatch(batch(0, 1, 2, {{4, 1.0F}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();
    EXPECT_EQ(bga.offerBatch(batch(0, 1, 1, {{3, 1.0F}})),
              CSCBGAInputResult::PROTOCOL_ERROR);
    EXPECT_GE(bga.counters().protocol_errors, 4U);
}

TEST(CSCBGAPipelineTest, EnqueueCompareAndAddLatenciesAreExplicit)
{
    CSCBankGroupAccumulator bga(config(1, 16, 4, 16, 2, 3));
    EXPECT_EQ(bga.offerBatch(batch(0, 1, 1, {{9, 1.0F}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();  // enqueue commit
    EXPECT_EQ(findRow(bga, 9), nullptr);
    bga.step();  // lookup starts
    bga.step();  // compare cycle 1
    EXPECT_EQ(findRow(bga, 9), nullptr);
    bga.step();  // compare completes, insert starts
    EXPECT_EQ(findRow(bga, 9), nullptr);
    bga.step();
    bga.step();
    EXPECT_EQ(findRow(bga, 9), nullptr);
    bga.step();  // add/insert latency 3 completes
    ASSERT_NE(findRow(bga, 9), nullptr);

    EXPECT_EQ(bga.offerBatch(batch(0, 1, 2, {{9, 2.0F}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();
    runUntil(bga, [&] { return bga.counters().lookup_hits == 1; });
    const uint64_t hit_cycle = bga.cycle();
    EXPECT_FLOAT_EQ(findRow(bga, 9)->value, 1.0F);
    bga.step();
    bga.step();
    EXPECT_FLOAT_EQ(findRow(bga, 9)->value, 1.0F);
    bga.step();
    EXPECT_EQ(bga.cycle(), hit_cycle + 3);
    EXPECT_FLOAT_EQ(findRow(bga, 9)->value, 3.0F);
}

TEST(CSCBGAAssociativeTest, FullyAssociativeHitAndFp32Rounding)
{
    CSCBankGroupAccumulator bga(config(1, 16, 4));
    EXPECT_EQ(bga.offerBatch(
                  batch(0, 1, 1, {{1, 16777216.0F}, {2, 4.0F}, {3, 5.0F}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();
    runUntil(bga, [&] { return bga.counters().inserts == 3; });
    const uint64_t age = findRow(bga, 1)->insertion_age;
    EXPECT_EQ(bga.offerBatch(batch(0, 1, 2, {{1, 1.0F}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();
    runUntil(bga, [&] { return bga.counters().fp32_merges == 1; });
    ASSERT_NE(findRow(bga, 1), nullptr);
    EXPECT_EQ(bits(findRow(bga, 1)->value), bits(16777216.0F));
    EXPECT_EQ(findRow(bga, 1)->insertion_age, age);
    EXPECT_EQ(findRow(bga, 1)->contribution_count, 2U);
}

TEST(CSCBGAVictimTest, FifoAgeSurvivesHitAndRetirementReleasesVictimSlot)
{
    CSCBankGroupAccumulator bga(config(1, 16, 2, 1));
    ASSERT_EQ(bga.offerBatch(batch(0, 1, 1, {{10, 1.0F}, {20, 2.0F}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();
    runUntil(bga, [&] { return bga.counters().inserts == 2; });
    const uint64_t age10 = findRow(bga, 10)->insertion_age;
    ASSERT_EQ(bga.offerBatch(batch(0, 1, 2, {{10, 3.0F}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();
    runUntil(bga, [&] { return bga.counters().fp32_merges == 1; });
    EXPECT_EQ(findRow(bga, 10)->insertion_age, age10);

    ASSERT_EQ(bga.offerBatch(batch(0, 1, 3, {{30, 4.0F}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();
    runUntil(bga, [&] { return bga.counters().capacity_evictions == 1; });
    ASSERT_TRUE(bga.hasOutput());
    EXPECT_EQ(bga.peekOutput().row_idx, 10U);
    EXPECT_EQ(bga.peekOutput().reason, CSCBGAOutputReason::CAPACITY_EVICTION);
    ASSERT_NE(findRow(bga, 10), nullptr);
    EXPECT_TRUE(findRow(bga, 10)->reserved);
    EXPECT_EQ(findRow(bga, 30), nullptr);
    const auto stable = bga.peekOutput();
    bga.step();
    EXPECT_EQ(bga.peekOutput().output_sequence, stable.output_sequence);
    EXPECT_EQ(findRow(bga, 30), nullptr);

    bga.acceptOutput();
    EXPECT_TRUE(findRow(bga, 10)->reserved);
    EXPECT_EQ(findRow(bga, 30), nullptr);
    bga.step();
    EXPECT_EQ(findRow(bga, 10), nullptr);
    EXPECT_EQ(findRow(bga, 30), nullptr);
    runUntil(bga, [&] { return findRow(bga, 30) != nullptr; });
    EXPECT_TRUE(bga.validateAccumulatorInvariant());
}

TEST(CSCBGAFlushTest, CapacityEvictionFinalDrainAndAbortAreDistinct)
{
    CSCBankGroupAccumulator bga(config(1, 16, 2, 8));
    EXPECT_EQ(bga.offerBatch(batch(0, 1, 1, {{4, 1.0F}, {5, 2.0F}, {6, 3.0F}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();
    runUntil(bga, [&] { return bga.counters().capacity_evictions == 1; });
    EXPECT_EQ(bga.peekOutput().reason, CSCBGAOutputReason::CAPACITY_EVICTION);
    const CSCBGAOutput capacity = bga.peekOutput();
    bga.acceptOutput();
    bga.step();
    runUntil(bga, [&] { return bga.counters().inserts == 3; });

    bga.markProducerDone(0);
    ASSERT_TRUE(bga.requestFinalDrain());
    runUntil(bga, [&] { return bga.counters().final_drain_outputs == 2; });
    auto outputs = drainOutputs(bga);
    while (!bga.finalDrainComplete()) {
        bga.step();
        auto more = drainOutputs(bga);
        outputs.insert(outputs.end(), more.begin(), more.end());
    }
    ASSERT_EQ(outputs.size(), 2U);
    EXPECT_EQ(capacity.row_idx, 4U);
    EXPECT_EQ(outputs[0].row_idx, 5U);
    EXPECT_EQ(outputs[1].row_idx, 6U);
    EXPECT_EQ(outputs[0].reason, CSCBGAOutputReason::FINAL_DRAIN);
    EXPECT_EQ(outputs[1].reason, CSCBGAOutputReason::FINAL_DRAIN);
    EXPECT_LT(capacity.output_sequence, outputs[0].output_sequence);
    EXPECT_LT(outputs[0].output_sequence, outputs[1].output_sequence);
    EXPECT_TRUE(bga.finalDrainComplete());
    EXPECT_TRUE(bga.conservationInvariant());

    CSCBankGroupAccumulator aborted(config());
    EXPECT_EQ(aborted.offerBatch(batch(0, 1, 1, {{7, 9.0F}})),
              CSCBGAInputResult::ACCEPTED);
    aborted.step();
    runUntil(aborted, [&] { return aborted.counters().inserts == 1; });
    aborted.abortReset(2);
    EXPECT_FALSE(aborted.hasOutput());
    EXPECT_TRUE(aborted.quiescent());
    EXPECT_EQ(aborted.counters().aborted_contributions, 1U);
    EXPECT_TRUE(aborted.conservationInvariant());
}

TEST(CSCBGAFlushTest, FinalDrainDoesNotCascadeFromInsertCommit)
{
    CSCBankGroupAccumulator bga(config(1, 8, 2));
    EXPECT_EQ(bga.offerBatch(batch(0, 1, 1, {{8, 1.0F}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();
    bga.markProducerDone(0);
    ASSERT_TRUE(bga.requestFinalDrain());
    runUntil(bga, [&] { return bga.counters().inserts == 1; });
    EXPECT_FALSE(bga.hasOutput());
    bga.step();
    ASSERT_TRUE(bga.hasOutput());
    EXPECT_EQ(bga.peekOutput().row_idx, 8U);
    EXPECT_EQ(bga.peekOutput().reason, CSCBGAOutputReason::FINAL_DRAIN);
}

TEST(CSCBGAFairnessTest, LogicalStreamsAreServicedRoundRobin)
{
    CSCBankGroupAccumulator bga(config(3, 8, 8));
    EXPECT_EQ(bga.offerBatch(batch(0, 1, 1, {{10, 1.0F}, {11, 1.0F}})),
              CSCBGAInputResult::ACCEPTED);
    EXPECT_EQ(bga.offerBatch(batch(1, 1, 1, {{20, 1.0F}, {21, 1.0F}})),
              CSCBGAInputResult::ACCEPTED);
    EXPECT_EQ(bga.offerBatch(batch(2, 1, 1, {{30, 1.0F}, {31, 1.0F}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();
    runUntil(bga, [&] { return bga.counters().inserts >= 3; });
    ASSERT_NE(findRow(bga, 10), nullptr);
    ASSERT_NE(findRow(bga, 20), nullptr);
    ASSERT_NE(findRow(bga, 30), nullptr);
    EXPECT_LT(findRow(bga, 10)->insertion_age, findRow(bga, 20)->insertion_age);
    EXPECT_LT(findRow(bga, 20)->insertion_age, findRow(bga, 30)->insertion_age);
}

TEST(CSCBGAInvariantTest, ContributionConservationAcrossMergeEvictAndRetire)
{
    CSCBankGroupAccumulator bga(config(1, 16, 2, 2));
    EXPECT_EQ(bga.offerBatch(
                  batch(0, 1, 1, {{1, 1.0F}, {1, 2.0F}, {2, 3.0F}, {3, 4.0F}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();
    for (uint32_t i = 0; i < 50; ++i) {
        EXPECT_TRUE(bga.conservationInvariant());
        bga.step();
        if (bga.hasOutput()) {
            bga.acceptOutput();
            bga.step();
        }
        if (bga.counters().inserts == 3 && bga.counters().fp32_merges == 1) break;
    }
    EXPECT_EQ(bga.counters().accepted_valid_partials, 4U);
    EXPECT_EQ(bga.counters().fp32_merges, 1U);
    EXPECT_TRUE(bga.conservationInvariant());
}


TEST(CSCBGAHandshakeTest, AcceptanceIsImmediateAndDonePreservesPendingBatch)
{
    CSCBankGroupAccumulator bga(config());
    const auto accepted = batch(0, 1, 1, {{7, 1.0F}, {8, 2.0F}});
    EXPECT_EQ(bga.offerBatch(accepted), CSCBGAInputResult::ACCEPTED);
    EXPECT_EQ(bga.lastAcceptedSequence(0), 1U);
    EXPECT_EQ(bga.counters().accepted_batches, 1U);
    EXPECT_EQ(bga.counters().accepted_valid_partials, 2U);
    EXPECT_TRUE(bga.hasAcceptedPending(0));
    EXPECT_EQ(bga.inputQueueSize(0), 0U);
    EXPECT_EQ(bga.liveContributionCount(), 2U);
    EXPECT_TRUE(bga.conservationInvariant());

    bga.markProducerDone(0);
    EXPECT_TRUE(bga.requestFinalDrain());
    EXPECT_FALSE(bga.quiescent());
    bga.step();
    EXPECT_FALSE(bga.hasAcceptedPending(0));
    EXPECT_EQ(bga.inputQueueSize(0), 2U);
    runUntil(bga, [&] { return bga.counters().inserts == 2; });
    EXPECT_FALSE(bga.finalDrainComplete());
    while (!bga.finalDrainComplete()) {
        bga.step();
        if (bga.hasOutput()) {
            bga.acceptOutput();
            bga.step();
        }
    }
    EXPECT_EQ(bga.counters().retired_output_contributions, 2U);
    EXPECT_TRUE(bga.conservationInvariant());
}

TEST(CSCBGAHandshakeTest, BackpressuredPayloadIsHeldBitwiseUntilRetry)
{
    CSCBankGroupAccumulator bga(config(1, 1, 4));
    ASSERT_EQ(bga.offerBatch(batch(0, 1, 1, {{1, 1.0F}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();
    const auto held = batch(0, 1, 2, {{2, 2.0F}});
    EXPECT_EQ(bga.offerBatch(held), CSCBGAInputResult::BACKPRESSURE);
    const uint64_t accepted = bga.counters().accepted_valid_partials;
    const uint64_t live = bga.liveContributionCount();
    EXPECT_EQ(bga.offerBatch(held), CSCBGAInputResult::BACKPRESSURE);
    EXPECT_EQ(bga.counters().accepted_valid_partials, accepted);
    EXPECT_EQ(bga.liveContributionCount(), live);

    auto changed_row = held;
    changed_row.partials[0].row_idx = 3;
    EXPECT_EQ(bga.offerBatch(changed_row), CSCBGAInputResult::PROTOCOL_ERROR);
    auto changed_bits = held;
    changed_bits.partials[0].value = -2.0F;
    EXPECT_EQ(bga.offerBatch(changed_bits), CSCBGAInputResult::PROTOCOL_ERROR);
    EXPECT_EQ(bga.counters().accepted_valid_partials, accepted);
    EXPECT_EQ(bga.liveContributionCount(), live);

    bga.step();
    EXPECT_EQ(bga.offerBatch(held), CSCBGAInputResult::ACCEPTED);
    EXPECT_EQ(bga.lastAcceptedSequence(0), 2U);
    EXPECT_TRUE(bga.conservationInvariant());
}

TEST(CSCBGAHandshakeTest, FloatIdentityUsesExactFp32Bits)
{
    const float nan1 = fromBits(0x7fc00001U);
    const float nan2 = fromBits(0x7fc00002U);
    CSCBankGroupAccumulator nan_bga(config());
    const auto n1 = batch(0, 1, 1, {{1, nan1}});
    EXPECT_EQ(nan_bga.offerBatch(n1), CSCBGAInputResult::ACCEPTED);
    EXPECT_EQ(nan_bga.offerBatch(n1), CSCBGAInputResult::DUPLICATE);
    auto n2 = n1;
    n2.partials[0].value = nan2;
    EXPECT_EQ(nan_bga.offerBatch(n2), CSCBGAInputResult::PROTOCOL_ERROR);

    CSCBankGroupAccumulator zero_bga(config());
    const auto plus_zero = batch(0, 1, 1, {{1, 0.0F}});
    EXPECT_EQ(zero_bga.offerBatch(plus_zero), CSCBGAInputResult::ACCEPTED);
    auto minus_zero = plus_zero;
    minus_zero.partials[0].value = -0.0F;
    EXPECT_EQ(zero_bga.offerBatch(minus_zero), CSCBGAInputResult::PROTOCOL_ERROR);

    CSCBankGroupAccumulator finite_bga(config());
    const auto finite = batch(0, 1, 1, {{1, 3.25F}});
    EXPECT_EQ(finite_bga.offerBatch(finite), CSCBGAInputResult::ACCEPTED);
    finite_bga.step();
    EXPECT_EQ(finite_bga.offerBatch(finite), CSCBGAInputResult::DUPLICATE);
}

TEST(CSCBGAResetTest, ResetRejectsOldGenerationAndClearsPendingOwnership)
{
    CSCBankGroupAccumulator bga(config());
    const auto old = batch(0, 1, 1, {{4, 1.0F}});
    ASSERT_EQ(bga.offerBatch(old), CSCBGAInputResult::ACCEPTED);
    EXPECT_TRUE(bga.hasAcceptedPending(0));
    bga.abortReset(2);
    EXPECT_FALSE(bga.hasAcceptedPending(0));
    EXPECT_TRUE(bga.quiescent());
    EXPECT_EQ(bga.counters().aborted_contributions, 1U);
    EXPECT_EQ(bga.offerBatch(old), CSCBGAInputResult::PROTOCOL_ERROR);
    EXPECT_THROW(bga.abortReset(1), std::invalid_argument);
    EXPECT_EQ(bga.offerBatch(batch(0, 2, 1, {{4, 2.0F}})),
              CSCBGAInputResult::ACCEPTED);
    EXPECT_EQ(bga.lastAcceptedSequence(0), 1U);
    EXPECT_TRUE(bga.conservationInvariant());
}

TEST(CSCBGAHandshakeTest, InvalidInputOnlyChangesAttemptAndErrorStatistics)
{
    CSCBankGroupAccumulator bga(config());
    const auto check_unchanged = [&](uint64_t accepted, uint64_t live) {
        EXPECT_EQ(bga.counters().accepted_valid_partials, accepted);
        EXPECT_EQ(bga.liveContributionCount(), live);
        EXPECT_EQ(bga.inputQueueSize(0), 0U);
        EXPECT_EQ(bga.outputQueueSize(), 0U);
        EXPECT_TRUE(bga.quiescent());
    };
    EXPECT_EQ(bga.offerBatch(batch(0, 1, 1, {{1024, 1.0F}})),
              CSCBGAInputResult::PROTOCOL_ERROR);
    check_unchanged(0, 0);
    EXPECT_EQ(bga.offerBatch(batch(1, 1, 1, {{1, 1.0F}})),
              CSCBGAInputResult::PROTOCOL_ERROR);
    check_unchanged(0, 0);
    EXPECT_EQ(bga.offerBatch(CSCBGABatch{0, 1, 1, {}}),
              CSCBGAInputResult::PROTOCOL_ERROR);
    check_unchanged(0, 0);
    std::vector<CSCBGAPartial> too_many(17, {1, 1.0F});
    EXPECT_EQ(bga.offerBatch(CSCBGABatch{0, 1, 1, too_many}),
              CSCBGAInputResult::PROTOCOL_ERROR);
    check_unchanged(0, 0);
    EXPECT_EQ(bga.counters().enqueue_attempts, 4U);
    EXPECT_EQ(bga.counters().protocol_errors, 4U);
}

TEST(CSCBGAEvictionTest, OutputAcceptanceDoesNotImmediatelyReleaseReservedSlot)
{
    CSCBankGroupAccumulator bga(config(1, 8, 1, 1, 1, 2));
    ASSERT_EQ(bga.offerBatch(batch(0, 1, 1, {{1, 1.0F}, {2, 2.0F}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();
    runUntil(bga, [&] { return bga.counters().capacity_evictions == 1; });
    ASSERT_TRUE(bga.hasOutput());
    ASSERT_NE(findRow(bga, 1), nullptr);
    EXPECT_TRUE(findRow(bga, 1)->reserved);
    EXPECT_EQ(findRow(bga, 2), nullptr);
    EXPECT_FALSE(bga.quiescent());

    bga.acceptOutput();
    EXPECT_TRUE(findRow(bga, 1)->reserved);
    EXPECT_EQ(findRow(bga, 2), nullptr);
    bga.step();
    EXPECT_EQ(findRow(bga, 1), nullptr);
    EXPECT_EQ(findRow(bga, 2), nullptr);
    bga.step();
    EXPECT_EQ(findRow(bga, 2), nullptr);
    bga.step();
    ASSERT_NE(findRow(bga, 2), nullptr);
    EXPECT_FALSE(findRow(bga, 2)->reserved);
    EXPECT_TRUE(bga.conservationInvariant());
}

TEST(CSCBGAResetTest, AbortDropsQueuedAndPendingRetirementOutputs)
{
    for (bool accept_before_abort : {false, true}) {
        CSCBankGroupAccumulator bga(config(1, 8, 1));
        ASSERT_EQ(bga.offerBatch(batch(0, 1, 1, {{1, 1.0F}, {2, 2.0F}})),
                  CSCBGAInputResult::ACCEPTED);
        bga.step();
        runUntil(bga, [&] { return bga.counters().capacity_evictions == 1; });
        ASSERT_TRUE(bga.hasOutput());
        if (accept_before_abort) bga.acceptOutput();
        bga.abortReset(2);
        EXPECT_FALSE(bga.hasOutput());
        EXPECT_TRUE(bga.quiescent());
        EXPECT_EQ(bga.counters().aborted_contributions, 2U);
        EXPECT_EQ(bga.counters().retired_output_contributions, 0U);
        EXPECT_TRUE(bga.conservationInvariant());
    }
}

TEST(CSCBGAFp32Test, NanAndInfinityMergesAreDeterministic)
{
    CSCBankGroupAccumulator bga(config(1, 8, 4));
    ASSERT_EQ(bga.offerBatch(batch(0, 1, 1, {{1, 1.0F}, {1, INFINITY}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();
    runUntil(bga, [&] { return bga.counters().fp32_merges == 1; });
    ASSERT_TRUE(std::isinf(findRow(bga, 1)->value));
    ASSERT_FALSE(std::signbit(findRow(bga, 1)->value));
    ASSERT_EQ(bga.offerBatch(batch(0, 1, 2,
                                   {{1, -INFINITY},
                                    {1, fromBits(0x7fc00011U)}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();
    runUntil(bga, [&] { return bga.counters().fp32_merges == 3; });
    ASSERT_NE(findRow(bga, 1), nullptr);
    EXPECT_TRUE(std::isnan(findRow(bga, 1)->value));
    EXPECT_EQ(findRow(bga, 1)->contribution_count, 4U);
    EXPECT_TRUE(bga.conservationInvariant());

    CSCBankGroupAccumulator negative(config());
    ASSERT_EQ(negative.offerBatch(
                  batch(0, 1, 1, {{2, 1.0F}, {2, -INFINITY}})),
              CSCBGAInputResult::ACCEPTED);
    negative.step();
    runUntil(negative, [&] { return negative.counters().fp32_merges == 1; });
    EXPECT_TRUE(std::isinf(findRow(negative, 2)->value));
    EXPECT_TRUE(std::signbit(findRow(negative, 2)->value));
}

TEST(CSCBGAAssociativeTest, LookupScansEveryValidNonReservedTag)
{
    CSCBankGroupAccumulator bga(config(1, 8, 4));
    ASSERT_EQ(bga.offerBatch(batch(0, 1, 1,
                                   {{10, 1.0F}, {20, 2.0F}, {30, 3.0F}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();
    runUntil(bga, [&] { return bga.counters().inserts == 3; });
    const uint64_t before = bga.counters().tag_comparisons;
    ASSERT_EQ(bga.offerBatch(batch(0, 1, 2, {{20, 4.0F}})),
              CSCBGAInputResult::ACCEPTED);
    bga.step();
    runUntil(bga, [&] { return bga.counters().lookup_hits == 1; });
    EXPECT_EQ(bga.counters().tag_comparisons - before, 3U);
    EXPECT_TRUE(bga.validateAccumulatorInvariant());
}

TEST(CSCBGACounterTest, CountersAreCumulativeAcrossResetAndTrackBusyIdleStalls)
{
    CSCBankGroupAccumulator idle(config(1, 8, 2, 1));
    idle.step();
    EXPECT_EQ(idle.counters().cycles_idle, 1U);
    ASSERT_EQ(idle.offerBatch(batch(0, 1, 1, {{1, 1.0F}, {2, 2.0F}})),
              CSCBGAInputResult::ACCEPTED);
    idle.step();
    EXPECT_GT(idle.counters().cycles_busy, 0U);
    runUntil(idle, [&] { return idle.counters().inserts == 2; });
    idle.markProducerDone(0);
    ASSERT_TRUE(idle.requestFinalDrain());
    runUntil(idle, [&] { return idle.hasOutput(); });
    idle.step();
    EXPECT_GT(idle.counters().output_stalls, 0U);
    const uint64_t accepted = idle.counters().accepted_valid_partials;
    const uint64_t busy = idle.counters().cycles_busy;
    idle.abortReset(2);
    EXPECT_EQ(idle.counters().accepted_valid_partials, accepted);
    EXPECT_EQ(idle.counters().cycles_busy, busy);
    EXPECT_EQ(idle.counters().aborted_contributions, 2U);
    EXPECT_TRUE(idle.conservationInvariant());
}
