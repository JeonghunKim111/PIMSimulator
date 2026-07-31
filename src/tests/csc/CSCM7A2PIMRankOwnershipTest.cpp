#include <gtest/gtest.h>

#include <memory>

#include "MultiChannelMemorySystem.h"
#include "PIMRank.h"

using namespace csc_descriptor;
using namespace DRAMSim;

namespace {

class BGAOwnershipHarness {
  public:
    BGAOwnershipHarness()
        : memory(std::make_shared<MultiChannelMemorySystem>(
              "ini/HBM2_samsung_2M_16B_x64.ini",
              "system_hbm_csc_fp32.ini", ".", "m7a2_bga_ownership",
              256 * 16))
    {
    }

    PIMRank& rank(uint32_t channel = 0)
    {
        return *memory->channels.at(channel)->ranks->at(0)->pimRank;
    }

    std::shared_ptr<MultiChannelMemorySystem> memory;
};

CSCBGAIntegrationConfig bgaConfig(uint32_t rows = 64,
                                  uint32_t entries = 16,
                                  uint32_t output_depth = 16)
{
    CSCBGAIntegrationConfig config;
    config.enabled = true;
    config.accumulator.rows = rows;
    config.accumulator.accumulator_entries = entries;
    config.accumulator.compare_width = entries;
    config.accumulator.output_queue_depth = output_depth;
    return config;
}

CSCBGAPartialBatch partialBatch(uint32_t global_bg, uint32_t generation,
                                uint64_t sequence,
                                std::initializer_list<CSCBGAPartial> partials)
{
    CSCBGAPartialBatch batch;
    batch.global_bg_id = global_bg;
    batch.descriptor_index = uint32_t(sequence - 1);
    batch.chunk_index = 0;
    batch.valid_count = partials.size();
    batch.descriptor_chunk_ordinal = sequence;
    batch.payload.logical_stream_id = kCSCM7ALogicalStream;
    batch.payload.generation = generation;
    batch.payload.sequence = sequence;
    batch.payload.partials = partials;
    return batch;
}

template <typename Predicate>
void stepUntil(PIMRank& rank, Predicate predicate, uint32_t limit = 1000)
{
    uint32_t steps = 0;
    while (!predicate() && steps++ < limit) rank.stepCSCBGAs();
    ASSERT_TRUE(predicate());
}

}  // namespace

TEST(CSCM7A2PIMRankOwnershipTest, DisabledByDefaultAndManualStepIsNoOp)
{
    BGAOwnershipHarness h;
    auto& rank = h.rank();
    EXPECT_FALSE(rank.cscBGAEnabled());
    EXPECT_FALSE(rank.hasCSCBGA(0));
    EXPECT_THROW(rank.cscGlobalBG(0), std::logic_error);
    EXPECT_THROW(rank.hasCSCBGAOutput(0), std::logic_error);
    EXPECT_THROW(rank.submitCSCBGAPartialBatch(
                     partialBatch(0, 1, 1, {{1, 1.0F}})),
                 std::logic_error);
    const uint64_t cycle = rank.currentClockCycle;
    const auto grants = rank.targetedStatistics().rank_targeted_grants;
    EXPECT_NO_THROW(rank.stepCSCBGAs());
    EXPECT_EQ(rank.currentClockCycle, cycle);
    EXPECT_EQ(rank.targetedStatistics().rank_targeted_grants, grants);
}

TEST(CSCM7A2PIMRankOwnershipTest, CreatesOneIndependentBGAForEveryLocalBG)
{
    BGAOwnershipHarness h;
    auto& rank = h.rank();
    rank.configureCSCBGAs(bgaConfig(), 3);
    ASSERT_TRUE(rank.cscBGAEnabled());
    for (uint32_t local = 0; local < kBGsPerRank; ++local) {
        EXPECT_TRUE(rank.hasCSCBGA(local));
        EXPECT_EQ(rank.cscGlobalBG(local), local);
        EXPECT_EQ(rank.cscLocalBG(local), local);
        EXPECT_EQ(rank.cscBGAGeneration(local), 3U);
        EXPECT_EQ(rank.cscBGA(local).config().rows, 64U);
        for (uint32_t other = local + 1; other < kBGsPerRank; ++other)
            EXPECT_NE(&rank.cscBGA(local), &rank.cscBGA(other));
    }
}

TEST(CSCM7A2PIMRankOwnershipTest, TopologyMappingPreservesRankDimension)
{
    EXPECT_EQ(PIMRank::computeCSCGlobalBG(0, 0, 8, 2, 4, 0), 0U);
    EXPECT_EQ(PIMRank::computeCSCGlobalBG(0, 1, 8, 2, 4, 0), 4U);
    EXPECT_EQ(PIMRank::computeCSCGlobalBG(3, 1, 8, 2, 4, 2), 30U);
    EXPECT_THROW(PIMRank::computeCSCGlobalBG(8, 0, 8, 2, 4, 0),
                 std::out_of_range);
    EXPECT_THROW(PIMRank::computeCSCGlobalBG(0, 0, 0, 2, 4, 0),
                 std::invalid_argument);

    BGAOwnershipHarness h;
    auto& channel3 = h.rank(3);
    channel3.configureCSCBGAs(bgaConfig(), 1);
    EXPECT_EQ(channel3.cscGlobalBG(0), 12U);
    EXPECT_EQ(channel3.cscGlobalBG(3), 15U);
    EXPECT_EQ(channel3.cscLocalBG(14), 2U);
    EXPECT_THROW(channel3.cscLocalBG(0), std::invalid_argument);
    EXPECT_THROW(channel3.cscLocalBG(kCSCGlobalBGs), std::out_of_range);
}

TEST(CSCM7A2PIMRankOwnershipTest, SubmitRoutesToExactlyOneLocalBG)
{
    BGAOwnershipHarness h;
    auto& rank = h.rank();
    rank.configureCSCBGAs(bgaConfig(), 1);
    EXPECT_EQ(rank.submitCSCBGAPartialBatch(
                  partialBatch(2, 1, 1, {{7, 2.0F}})),
              CSCBGAInputResult::ACCEPTED);
    for (uint32_t local = 0; local < kBGsPerRank; ++local)
        EXPECT_EQ(rank.cscBGACounters(local).accepted_valid_partials,
                  local == 2 ? 1U : 0U);

    auto invalid_stream = partialBatch(2, 1, 2, {{8, 3.0F}});
    invalid_stream.payload.logical_stream_id = 1;
    EXPECT_THROW(rank.submitCSCBGAPartialBatch(invalid_stream),
                 std::invalid_argument);
    EXPECT_THROW(rank.submitCSCBGAPartialBatch(
                     partialBatch(4, 1, 1, {{1, 1.0F}})),
                 std::invalid_argument);
    EXPECT_EQ(rank.cscBGACounters(2).accepted_valid_partials, 1U);
}

TEST(CSCM7A2PIMRankOwnershipTest, SequenceAndGenerationAreIndependentPerBG)
{
    BGAOwnershipHarness h;
    auto& rank = h.rank();
    rank.configureCSCBGAs(bgaConfig(), 5);
    auto a = partialBatch(0, 5, 1, {{1, 1.0F}});
    auto b = partialBatch(1, 5, 1, {{2, 2.0F}});
    EXPECT_EQ(rank.submitCSCBGAPartialBatch(a), CSCBGAInputResult::ACCEPTED);
    EXPECT_EQ(rank.submitCSCBGAPartialBatch(b), CSCBGAInputResult::ACCEPTED);
    EXPECT_EQ(rank.submitCSCBGAPartialBatch(a), CSCBGAInputResult::DUPLICATE);
    EXPECT_EQ(rank.cscBGA(0).lastAcceptedSequence(0), 1U);
    EXPECT_EQ(rank.cscBGA(1).lastAcceptedSequence(0), 1U);
    EXPECT_EQ(rank.cscBGACounters(1).protocol_errors, 0U);

    EXPECT_THROW(rank.submitCSCBGAPartialBatch(
                     partialBatch(0, 6, 2, {{3, 1.0F}})),
                 std::invalid_argument);
    EXPECT_EQ(rank.cscBGAGeneration(1), 5U);
    EXPECT_EQ(rank.cscBGACounters(1).accepted_valid_partials, 1U);
}

TEST(CSCM7A2PIMRankOwnershipTest, ProducerDoneAndFinalDrainAreSeparate)
{
    BGAOwnershipHarness h;
    auto& rank = h.rank();
    rank.configureCSCBGAs(bgaConfig(), 2);
    rank.markCSCBGAProducerDone({0, 0, 2});
    EXPECT_FALSE(rank.cscBGA(0).finalDrainRequested());
    EXPECT_FALSE(rank.requestCSCBGAFinalDrain(1));
    EXPECT_TRUE(rank.requestCSCBGAFinalDrain(0));
    EXPECT_FALSE(rank.cscBGA(1).finalDrainRequested());
    EXPECT_THROW(rank.markCSCBGAProducerDone({1, 0, 3}),
                 std::invalid_argument);
    EXPECT_THROW(rank.markCSCBGAProducerDone({4, 0, 2}),
                 std::invalid_argument);
}

TEST(CSCM7A2PIMRankOwnershipTest, ManualStepCommitsThenServicesIndependently)
{
    BGAOwnershipHarness h;
    auto& rank = h.rank();
    rank.configureCSCBGAs(bgaConfig(), 1);
    ASSERT_EQ(rank.submitCSCBGAPartialBatch(
                  partialBatch(0, 1, 1, {{4, 1.0F}})),
              CSCBGAInputResult::ACCEPTED);
    EXPECT_TRUE(rank.cscBGA(0).hasAcceptedPending(0));
    rank.stepCSCBGAs();
    EXPECT_FALSE(rank.cscBGA(0).hasAcceptedPending(0));
    EXPECT_EQ(rank.cscBGA(0).inputQueueSize(0), 1U);
    EXPECT_EQ(rank.cscBGACounters(0).lookup_misses, 0U);
    EXPECT_EQ(rank.cscBGA(0).cycle(), 1U);
    EXPECT_EQ(rank.cscBGA(1).cycle(), 1U);
    rank.stepCSCBGAs();
    EXPECT_EQ(rank.cscBGA(0).inputQueueSize(0), 0U);
    EXPECT_EQ(rank.cscBGACounters(1).accepted_valid_partials, 0U);
    EXPECT_EQ(rank.currentClockCycle, 0U);
}

TEST(CSCM7A2PIMRankOwnershipTest, OutputPortPreservesOwnerAndRetirementLifecycle)
{
    BGAOwnershipHarness h;
    auto& rank = h.rank();
    rank.configureCSCBGAs(bgaConfig(64, 1, 1), 7);
    ASSERT_EQ(rank.submitCSCBGAPartialBatch(
                  partialBatch(0, 7, 1, {{1, 1.0F}, {2, 2.0F}})),
              CSCBGAInputResult::ACCEPTED);
    stepUntil(rank, [&] { return rank.hasCSCBGAOutput(0); });
    EXPECT_FALSE(rank.hasCSCBGAOutput(1));
    const auto first = rank.peekCSCBGAOutput(0);
    const auto stable = rank.peekCSCBGAOutput(0);
    EXPECT_EQ(first.global_bg_id, 0U);
    EXPECT_EQ(first.payload.generation, 7U);
    EXPECT_EQ(first.payload.output_sequence, stable.payload.output_sequence);
    EXPECT_EQ(first.payload.reason, CSCBGAOutputReason::CAPACITY_EVICTION);
    rank.acceptCSCBGAOutput(0);
    EXPECT_TRUE(rank.hasCSCBGAOutput(0));
    EXPECT_TRUE(rank.cscBGA(0).accumulator()[0].reserved);
    rank.stepCSCBGAs();
    EXPECT_FALSE(rank.hasCSCBGAOutput(0));
    EXPECT_TRUE(rank.cscBGA(0).conservationInvariant());
}

TEST(CSCM7A2PIMRankOwnershipTest, ExplicitFinalDrainCompletesAfterRetirement)
{
    BGAOwnershipHarness h;
    auto& rank = h.rank();
    rank.configureCSCBGAs(bgaConfig(64, 2), 1);
    ASSERT_EQ(rank.submitCSCBGAPartialBatch(
                  partialBatch(0, 1, 1, {{3, 4.0F}})),
              CSCBGAInputResult::ACCEPTED);
    stepUntil(rank, [&] { return rank.cscBGACounters(0).inserts == 1; });
    rank.markCSCBGAProducerDone({0, 0, 1});
    rank.stepCSCBGAs();
    EXPECT_FALSE(rank.hasCSCBGAOutput(0));
    ASSERT_TRUE(rank.requestCSCBGAFinalDrain(0));
    stepUntil(rank, [&] { return rank.hasCSCBGAOutput(0); });
    EXPECT_FALSE(rank.cscBGAFinalDrainComplete(0));
    rank.acceptCSCBGAOutput(0);
    EXPECT_FALSE(rank.cscBGAFinalDrainComplete(0));
    rank.stepCSCBGAs();
    EXPECT_TRUE(rank.cscBGAFinalDrainComplete(0));
}

TEST(CSCM7A2PIMRankOwnershipTest, ConfigurationIsAllOrNoneAndRetryable)
{
    BGAOwnershipHarness h;
    auto& rank = h.rank();
    auto invalid = bgaConfig();
    invalid.accumulator.rows = 0;
    EXPECT_THROW(rank.configureCSCBGAs(invalid, 1), std::invalid_argument);
    EXPECT_FALSE(rank.cscBGAEnabled());
    for (uint32_t local = 0; local < kBGsPerRank; ++local)
        EXPECT_FALSE(rank.hasCSCBGA(local));
    EXPECT_NO_THROW(rank.configureCSCBGAs(bgaConfig(), 1));
    EXPECT_TRUE(rank.cscBGAEnabled());

    ASSERT_EQ(rank.submitCSCBGAPartialBatch(
                  partialBatch(0, 1, 1, {{1, 1.0F}})),
              CSCBGAInputResult::ACCEPTED);
    EXPECT_THROW(rank.configureCSCBGAs(bgaConfig(), 2), std::logic_error);
    EXPECT_EQ(rank.cscBGAGeneration(0), 1U);
    EXPECT_TRUE(rank.cscBGA(0).conservationInvariant());
}
