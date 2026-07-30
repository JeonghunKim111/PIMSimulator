#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

#include "MultiChannelMemorySystem.h"
#include "PIMRank.h"
#include "csc/CSCDescriptorEngine.h"

namespace DRAMSim
{
struct PIMRankM6TestAccess
{
    static BGTargetedCompletionValidation injectCompletion(
        PIMRank& rank, uint32_t local_bg, const BGTargetedCompletion& candidate)
    {
        BGTargetedCompletion ignored;
        return rank.validateAndRetireBGTargetedCompletion(local_bg, candidate, ignored);
    }
    static bool corruptBusyMaskAndValidate(PIMRank& rank, uint8_t extra_bit)
    {
        rank.pimblock_busy_mask_ |= extra_bit;
        return rank.validateSharedOwnership();
    }
};
}

namespace csc_descriptor
{
struct CSCM6CompletionTestAccess
{
    static uint32_t generation(const CSCNativeExecution& execution)
    {
        return execution.generation_;
    }
    static bool submitTarget(CSCNativeExecution& execution, uint32_t bg,
                             const DRAMSim::BGTargetedOperation& operation)
    {
        return execution.submitTarget(execution.engines_.at(bg).get(), operation);
    }
    static DRAMSim::PIMRank& rank(CSCNativeExecution& execution, uint32_t bg)
    {
        return execution.targetRank(bg);
    }
    static uint64_t memoryOutstanding(const CSCNativeExecution& execution)
    {
        return execution.outstanding_.size();
    }
};
}

namespace
{
using namespace DRAMSim;

class TargetHarness
{
  public:
    TargetHarness()
        : memory(std::make_shared<MultiChannelMemorySystem>(
              "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_csc_fp32.ini", ".",
              "csc_m6_target_port", 256 * 16)),
          pim(*memory->channels.at(0)->ranks->at(0)->pimRank)
    {
    }

    BGTargetedOperation operation(uint32_t bg, uint64_t id, uint8_t mask = 1,
                                  uint32_t valid_count = 8)
    {
        BGTargetedOperation op;
        op.identity.operation_id = id;
        op.identity.channel = 0;
        op.identity.rank = 0;
        op.identity.local_bg = bg;
        op.identity.pimblock_mask = mask;
        op.identity.worker_id = bg;
        op.identity.sequence = id;
        op.identity.generation = 1;
        op.valid_count = valid_count;
        for (uint32_t bit = 0; bit < 2; ++bit)
            if (mask & (1U << bit))
            {
                op.contexts[bit].valid = true;
                for (uint32_t lane = 0; lane < 8; ++lane)
                {
                    op.contexts[bit].lhs.fp32Data_[lane] = float(id + lane);
                    op.contexts[bit].rhs.fp32Data_[lane] = float(bit + 2);
                }
            }
        return op;
    }

    void nextCycle(bool command_busy = false, bool data_busy = false)
    {
        pim.step();
        pim.serviceBGTargeted(command_busy, data_busy);
    }

    std::shared_ptr<MultiChannelMemorySystem> memory;
    PIMRank& pim;
};
}  // namespace

TEST(CSCM6TargetedExecutionTest, StaticBGToPIMBlockPairBinding)
{
    TargetHarness h;
    for (uint32_t bg = 0; bg < 4; ++bg)
        EXPECT_EQ(h.pim.physicalPIMBlockPair(bg),
                  (std::array<uint32_t, 2>{bg * 2, bg * 2 + 1}));
}

TEST(CSCM6TargetedExecutionTest, TopologyMismatchFailsFast)
{
    TargetHarness h;
    EXPECT_EQ(h.pim.pimBlocks.size(), 8);
    EXPECT_THROW(h.pim.physicalPIMBlockPair(4), std::out_of_range);
}

TEST(CSCM6TargetedExecutionTest, OneHotMaskSelectsFirstPIMBlock)
{
    TargetHarness h;
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(h.operation(1, 11, 1)));
    h.pim.serviceBGTargeted(false, false);
    EXPECT_EQ(h.pim.targetedPIMBlockBusyMask(), 1U << 2);
    h.nextCycle();
    BGTargetedCompletion completion;
    ASSERT_TRUE(h.pim.pollBGTargetedCompletion(1, completion));
    EXPECT_FLOAT_EQ(completion.results[0].fp32Data_[3], float(14 * 2));
    EXPECT_EQ(h.pim.pimBlocks[2].simdCounters().issued_operations, 1);
    EXPECT_EQ(h.pim.pimBlocks[3].simdCounters().issued_operations, 0);
}

TEST(CSCM6TargetedExecutionTest, OneHotMaskSelectsSecondPIMBlock)
{
    TargetHarness h;
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(h.operation(2, 7, 2)));
    h.pim.serviceBGTargeted(false, false);
    EXPECT_EQ(h.pim.targetedPIMBlockBusyMask(), 1U << 5);
    h.nextCycle();
    BGTargetedCompletion completion;
    ASSERT_TRUE(h.pim.pollBGTargetedCompletion(2, completion));
    EXPECT_FLOAT_EQ(completion.results[1].fp32Data_[2], float(9 * 3));
    EXPECT_EQ(h.pim.pimBlocks[4].simdCounters().issued_operations, 0);
    EXPECT_EQ(h.pim.pimBlocks[5].simdCounters().issued_operations, 1);
}

TEST(CSCM6TargetedExecutionTest, PairMaskRequiresTwoValidContexts)
{
    TargetHarness h;
    auto invalid = h.operation(0, 1, 1);
    invalid.identity.pimblock_mask = 3;
    EXPECT_THROW(h.pim.submitBGTargetedOperation(invalid), std::invalid_argument);
    auto valid = h.operation(0, 2, 3);
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(valid));
    h.pim.serviceBGTargeted(false, false);
    EXPECT_EQ(h.pim.targetedPIMBlockBusyMask(), 3);
    h.nextCycle();
    BGTargetedCompletion completion;
    ASSERT_TRUE(h.pim.pollBGTargetedCompletion(0, completion));
    EXPECT_EQ(completion.identity.pimblock_mask, 3);
}

TEST(CSCM6TargetedExecutionTest, BGTargetDoesNotTouchOtherPIMBlocks)
{
    TargetHarness h;
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(h.operation(3, 4)));
    h.pim.serviceBGTargeted(false, false);
    h.nextCycle();
    BGTargetedCompletion completion;
    ASSERT_TRUE(h.pim.pollBGTargetedCompletion(3, completion));
    for (uint32_t block = 0; block < 8; ++block)
        EXPECT_EQ(h.pim.pimBlocks[block].simdCounters().issued_operations, block == 6 ? 1 : 0);
}

TEST(CSCM6TargetedExecutionTest, OneGrantPerRankPerCycle)
{
    TargetHarness h;
    for (uint32_t bg = 0; bg < 4; ++bg)
        ASSERT_TRUE(h.pim.submitBGTargetedOperation(h.operation(bg, bg + 1)));
    h.pim.serviceBGTargeted(false, false);
    EXPECT_EQ(h.pim.targetedStatistics().rank_targeted_grants, 1);
    EXPECT_EQ(h.pim.nextBGRoundRobin(), 1);
}

TEST(CSCM6TargetedExecutionTest, RoundRobinPreventsStarvation)
{
    TargetHarness h;
    for (uint32_t bg = 0; bg < 4; ++bg)
        ASSERT_TRUE(h.pim.submitBGTargetedOperation(h.operation(bg, bg + 1)));
    h.pim.serviceBGTargeted(false, false);
    for (uint32_t expected = 0; expected < 4; ++expected)
    {
        h.nextCycle();
        BGTargetedCompletion completion;
        ASSERT_TRUE(h.pim.pollBGTargetedCompletion(expected, completion));
        EXPECT_EQ(completion.identity.operation_id, expected + 1);
        EXPECT_EQ(completion.grant_cycle, expected);
    }
    EXPECT_EQ(h.pim.targetedStatistics().rank_targeted_grants, 4);
}

TEST(CSCM6TargetedExecutionTest, CommandBusBusyStallsGrant)
{
    TargetHarness h;
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(h.operation(0, 1)));
    h.pim.serviceBGTargeted(true, false);
    EXPECT_EQ(h.pim.targetedStatistics().rank_targeted_grants, 0);
    EXPECT_EQ(h.pim.targetedStatistics().rank_command_bus_stall_cycles, 1);
    EXPECT_TRUE(h.pim.bgHasPendingOperation(0));
}

TEST(CSCM6TargetedExecutionTest, GrantCompletesOnNextCycle)
{
    TargetHarness h;
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(h.operation(0, 1)));
    h.pim.serviceBGTargeted(false, false);
    BGTargetedCompletion completion;
    EXPECT_FALSE(h.pim.pollBGTargetedCompletion(0, completion));
    h.nextCycle();
    ASSERT_TRUE(h.pim.pollBGTargetedCompletion(0, completion));
    EXPECT_EQ(completion.completion_cycle, completion.grant_cycle + 1);
    EXPECT_FALSE(h.pim.pollBGTargetedCompletion(0, completion));
    EXPECT_EQ(h.pim.targetedPIMBlockBusyMask(), 0);
}

TEST(CSCM6FlushIsolationTest, OneBGFlushDoesNotStopOtherBGs)
{
    TargetHarness h;
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(h.operation(0, 1)));
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(h.operation(1, 2)));
    h.pim.flushBG(0);
    EXPECT_EQ(h.pim.bgLifecycle(0), BGLifecycleState::FLUSHED);
    EXPECT_TRUE(h.pim.bgHasPendingOperation(1));
    h.pim.serviceBGTargeted(false, false);
    EXPECT_EQ(h.pim.targetedStatistics().rank_targeted_grants, 1);
}

TEST(CSCM6FlushIsolationTest, GrantedTargetedOpCompletesAfterFlush)
{
    TargetHarness h;
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(h.operation(0, 1)));
    h.pim.serviceBGTargeted(false, false);
    h.pim.flushBG(0);
    EXPECT_EQ(h.pim.bgLifecycle(0), BGLifecycleState::DRAINING);
    h.nextCycle();
    BGTargetedCompletion completion;
    ASSERT_TRUE(h.pim.pollBGTargetedCompletion(0, completion));
    h.nextCycle();
    EXPECT_EQ(h.pim.bgLifecycle(0), BGLifecycleState::FLUSHED);
}

TEST(CSCM6FlushIsolationTest, ResetRequiresQuiescentBGAndIncrementsGeneration)
{
    TargetHarness h;
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(h.operation(0, 1)));
    EXPECT_FALSE(h.pim.resetBG(0));
    EXPECT_EQ(h.pim.bgGeneration(0), 1);
    h.nextCycle();
    EXPECT_TRUE(h.pim.resetBG(0));
    EXPECT_EQ(h.pim.bgGeneration(0), 2);
    EXPECT_EQ(h.pim.bgLifecycle(0), BGLifecycleState::IDLE);
}

TEST(CSCM6ModeIsolationTest, TargetedModeExitUsesDrain)
{
    TargetHarness h;
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(h.operation(0, 1)));
    ASSERT_TRUE(h.pim.requestTargetedModeExit());
    EXPECT_EQ(h.pim.executionMode(), RankExecutionMode::DRAINING);
    h.pim.serviceBGTargeted(false, false);
    EXPECT_EQ(h.pim.executionMode(), RankExecutionMode::DRAINING);
    h.pim.flushBG(0);
    h.nextCycle();
    EXPECT_EQ(h.pim.executionMode(), RankExecutionMode::IDLE);
}

TEST(CSCM6IntegrationTest, InternalImageUsesPhysicalPIMBlockAndMatchesCPU)
{
    using namespace csc_descriptor;
    std::array<std::vector<uint8_t>, 64> values, rows;
    std::array<std::vector<CSCDescriptor>, 64> descriptors;
    std::array<std::vector<float>, 64> x;
    std::array<CSCBGImageView, 64> views;
    for (uint32_t bg = 0; bg < 64; ++bg)
    {
        values[bg].resize(32);
        rows[bg].resize(32);
        x[bg].push_back(bg == 0 ? 3.0F : 0.0F);
        views[bg] = {&values[bg], &rows[bg], &descriptors[bg], &x[bg]};
    }
    const float value = 2.5F;
    const uint32_t row = 0;
    std::memcpy(values[0].data(), &value, sizeof(value));
    std::memcpy(rows[0].data(), &row, sizeof(row));
    descriptors[0].push_back({0, 0, 1, 0, 0, 0});

    CSCNativeExecution execution;
    execution.launch(views, 1, 1);
    uint64_t guard = 0;
    while (!execution.done() && guard++ < 10000) execution.tick();
    ASSERT_TRUE(execution.isDone());
    ASSERT_LT(guard, 10000);
    const auto result = execution.hostAccumulate();
    ASSERT_EQ(result.size(), 1);
    EXPECT_FLOAT_EQ(result[0], 7.5F);
    const auto counters = execution.counters();
    EXPECT_EQ(counters.logical_mul_events, 1);
    EXPECT_EQ(counters.rank_targeted_grants, 1);
    EXPECT_EQ(counters.per_pimblock_targeted_ops[0], 1);
    EXPECT_EQ(counters.per_pimblock_targeted_ops[1], 0);
    EXPECT_EQ(counters.per_bg_executing_cycles[0], 1);
}

TEST(CSCM6CycleComparisonTest, DecoupledCountersComeFromContendedSimulatorCycles)
{
    TargetHarness h;
    for (uint32_t bg = 0; bg < 4; ++bg)
        ASSERT_TRUE(h.pim.submitBGTargetedOperation(h.operation(bg, bg + 1)));
    h.pim.serviceBGTargeted(true, false);
    for (uint32_t cycle = 0; cycle < 4; ++cycle) h.nextCycle();
    const auto& counters = h.pim.targetedStatistics();
    EXPECT_EQ(counters.rank_command_bus_stall_cycles, 1);
    EXPECT_EQ(counters.rank_targeted_grants, 4);
    EXPECT_EQ(counters.round_robin_skip_count, 0);
    EXPECT_EQ(h.pim.targetedPIMBlockBusyMask(), 1U << 6);
    h.nextCycle();
    EXPECT_EQ(h.pim.targetedPIMBlockBusyMask(), 0);
}

TEST(CSCM6TargetedExecutionTest, StaleGenerationOperationIsRejectedAfterReset)
{
    TargetHarness h;
    auto first = h.operation(0, 1);
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(first));
    h.pim.flushBG(0);
    ASSERT_TRUE(h.pim.resetBG(0));
    EXPECT_EQ(h.pim.bgGeneration(0), 2);
    auto stale = h.operation(0, 2);
    stale.identity.generation = 1;
    EXPECT_THROW(h.pim.submitBGTargetedOperation(stale), std::invalid_argument);
}

TEST(CSCM6ModeIsolationTest, LegacyAndTargetedModesAreMutuallyExclusive)
{
    TargetHarness targeted;
    ASSERT_TRUE(targeted.pim.submitBGTargetedOperation(targeted.operation(0, 1)));
    BurstType control_data;
    control_data.u8Data_[0] = 1;
    BusPacket enter_legacy(WRITE, 0, 0, 0, 0, 0, &control_data,
                           targeted.memory->getLogFile());
    EXPECT_THROW(targeted.pim.controlPIM(&enter_legacy), std::logic_error);

    TargetHarness legacy;
    BusPacket legacy_packet(WRITE, 0, 0, 0, 0, 0, &control_data,
                            legacy.memory->getLogFile());
    legacy.pim.controlPIM(&legacy_packet);
    EXPECT_EQ(legacy.pim.executionMode(), RankExecutionMode::LEGACY_RANK_WIDE);
    EXPECT_FALSE(legacy.pim.submitBGTargetedOperation(legacy.operation(0, 3)));
}

TEST(CSCM6ModeIsolationTest, DifferentRanksMayUseDifferentModes)
{
    TargetHarness h;
    auto& other = *h.memory->channels.at(1)->ranks->at(0)->pimRank;
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(h.operation(0, 1)));
    auto op = h.operation(0, 2);
    op.identity.channel = 1;
    ASSERT_TRUE(other.submitBGTargetedOperation(op));
    h.pim.serviceBGTargeted(false, false);
    other.serviceBGTargeted(true, false);
    EXPECT_EQ(h.pim.targetedStatistics().rank_targeted_grants, 1);
    EXPECT_EQ(other.targetedStatistics().rank_targeted_grants, 0);
    EXPECT_EQ(other.targetedStatistics().rank_command_bus_stall_cycles, 1);
}

TEST(CSCM6SchedulingComparisonTest, BarrierWaitsAtChunkBoundaryAndFinishedBGDoesNotDeadlock)
{
    using namespace csc_descriptor;
    std::array<std::vector<uint8_t>, 64> values, rows;
    std::array<std::vector<CSCDescriptor>, 64> descriptors;
    std::array<std::vector<float>, 64> x;
    std::array<CSCBGImageView, 64> views;
    for (uint32_t bg = 0; bg < 64; ++bg)
    {
        values[bg].resize(32);
        rows[bg].resize(32);
        x[bg].push_back(2.0F);
        views[bg] = {&values[bg], &rows[bg], &descriptors[bg], &x[bg]};
    }
    for (uint32_t bg = 0; bg < 2; ++bg)
    {
        const float value = float(bg + 1);
        const uint32_t row = bg;
        std::memcpy(values[bg].data(), &value, sizeof(value));
        std::memcpy(rows[bg].data(), &row, sizeof(row));
        descriptors[bg].push_back({0, 0, 1, 0, 0, bg});
    }
    CSCNativeExecution execution(CSCRequestPolicy::OVERLAPPED,
                                 CSCSchedulingPolicy::BARRIER_LOCKSTEP_REFERENCE);
    EXPECT_EQ(execution.schedulingPolicy(),
              CSCSchedulingPolicy::BARRIER_LOCKSTEP_REFERENCE);
    execution.setSubmitRejectBudget(0, 20);
    execution.launch(views, 2, 2);
    uint64_t guard = 0;
    while (!execution.done() && guard++ < 20000) execution.tick();
    ASSERT_TRUE(execution.isDone()) << " failed=" << execution.hasFailed() << " error=" << execution.errorMessage() << " bg0state=" << int(execution.engine(0).state()) << " bg1state=" << int(execution.engine(1).state()) << " e0=" << execution.engine(0).progressEpoch() << " e1=" << execution.engine(1).progressEpoch();
    EXPECT_LT(guard, 20000);
    const auto result = execution.hostAccumulate();
    ASSERT_EQ(result.size(), 2);
    EXPECT_FLOAT_EQ(result[0], 2.0F);
    EXPECT_FLOAT_EQ(result[1], 4.0F);
    const auto counters = execution.counters();
    EXPECT_GT(counters.per_bg_barrier_wait_cycles[1], 0);
    EXPECT_GT(counters.per_bg_completion_cycle[0], 0);
    EXPECT_GT(counters.per_bg_completion_cycle[1], 0);
}

TEST(CSCM6SchedulingComparisonTest, ProductionDefaultRemainsBGDecoupled)
{
    csc_descriptor::CSCNativeExecution execution;
    EXPECT_EQ(execution.schedulingPolicy(),
              csc_descriptor::CSCSchedulingPolicy::BG_DECOUPLED);
}

namespace
{
struct SchedulingRun
{
    std::vector<float> result;
    csc_descriptor::CSCExecutionCounters counters;
};

SchedulingRun runSchedulingWorkload(const std::array<uint32_t, 4>& chunks,
                                    csc_descriptor::CSCSchedulingPolicy policy,
                                    uint64_t bg0_reject_budget = 0)
{
    using namespace csc_descriptor;
    std::array<std::vector<uint8_t>, 64> values, rows;
    std::array<std::vector<CSCDescriptor>, 64> descriptors;
    std::array<std::vector<float>, 64> x;
    std::array<CSCBGImageView, 64> views;
    uint64_t nnz = 0;
    for (uint32_t bg = 0; bg < 64; ++bg)
    {
        x[bg].push_back(2.0F);
        if (bg < 4 && chunks[bg])
        {
            const uint32_t count = chunks[bg] * 8;
            values[bg].resize(chunks[bg] * 32);
            rows[bg].resize(chunks[bg] * 32);
            for (uint32_t lane = 0; lane < count; ++lane)
            {
                const float value = float(bg + 1);
                const uint32_t row = bg;
                std::memcpy(values[bg].data() + lane * 4, &value, 4);
                std::memcpy(rows[bg].data() + lane * 4, &row, 4);
            }
            descriptors[bg].push_back({0, 0, count, 0, bg, bg});
            nnz += count;
        }
        views[bg] = {&values[bg], &rows[bg], &descriptors[bg], &x[bg]};
    }
    CSCNativeExecution execution(CSCRequestPolicy::OVERLAPPED, policy);
    execution.setSubmitRejectBudget(0, bg0_reject_budget);
    execution.launch(views, 4, nnz);
    uint64_t guard = 0;
    while (!execution.done() && guard++ < 200000) execution.tick();
    EXPECT_TRUE(execution.isDone()) << execution.errorMessage();
    SchedulingRun run;
    if (execution.isDone()) run.result = execution.hostAccumulate();
    run.counters = execution.counters();
    return run;
}
}

TEST(CSCM6SchedulingComparisonTest, BalancedLockstepAndDecoupledProduceSameResultAndCounts)
{
    using namespace csc_descriptor;
    const auto decoupled = runSchedulingWorkload({2, 2, 2, 2},
                                                 CSCSchedulingPolicy::BG_DECOUPLED);
    const auto lockstep = runSchedulingWorkload(
        {2, 2, 2, 2}, CSCSchedulingPolicy::BARRIER_LOCKSTEP_REFERENCE);
    EXPECT_EQ(decoupled.result, lockstep.result);
    EXPECT_EQ(decoupled.counters.memory_requests_accepted, lockstep.counters.memory_requests_accepted);
    EXPECT_EQ(decoupled.counters.targeted_ops_accepted, lockstep.counters.targeted_ops_accepted);
    EXPECT_EQ(decoupled.counters.targeted_ops_completed, lockstep.counters.targeted_ops_completed);
    EXPECT_EQ(decoupled.counters.partial_results_emitted, lockstep.counters.partial_results_emitted);
    EXPECT_EQ(decoupled.counters.targeted_ops_accepted, 8);
}

TEST(CSCM6SchedulingComparisonTest, ImbalancedDecoupledCompletesBeforeLockstep)
{
    using namespace csc_descriptor;
    const auto decoupled = runSchedulingWorkload({1, 4, 1, 0},
                                                 CSCSchedulingPolicy::BG_DECOUPLED, 80);
    const auto lockstep = runSchedulingWorkload(
        {1, 4, 1, 0}, CSCSchedulingPolicy::BARRIER_LOCKSTEP_REFERENCE, 80);
    EXPECT_EQ(decoupled.result, lockstep.result);
    EXPECT_LT(decoupled.counters.total_cycles, lockstep.counters.total_cycles);
    EXPECT_EQ(decoupled.counters.total_barrier_wait_cycles, 0);
    EXPECT_GT(lockstep.counters.total_barrier_wait_cycles, 0);
    EXPECT_LT(decoupled.counters.per_bg_completion_cycle[1],
              lockstep.counters.per_bg_completion_cycle[1]);
    EXPECT_EQ(decoupled.counters.memory_requests_accepted, lockstep.counters.memory_requests_accepted);
    EXPECT_EQ(decoupled.counters.targeted_ops_accepted, lockstep.counters.targeted_ops_accepted);
    EXPECT_EQ(decoupled.counters.rank_targeted_grants, decoupled.counters.targeted_ops_completed);
    EXPECT_EQ(lockstep.counters.rank_targeted_grants, lockstep.counters.targeted_ops_completed);
    RecordProperty("decoupled_cycles", decoupled.counters.total_cycles);
    RecordProperty("lockstep_cycles", lockstep.counters.total_cycles);
    RecordProperty("barrier_wait_cycles", lockstep.counters.total_barrier_wait_cycles);
}

TEST(CSCM6CompletionNegativeTest, DuplicateCompletionIsRejectedExactlyOnce)
{
    TargetHarness h;
    const auto operation = h.operation(0, 101);
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(operation));
    h.pim.serviceBGTargeted(false, false);
    h.nextCycle();
    BGTargetedCompletion completion;
    ASSERT_TRUE(h.pim.pollBGTargetedCompletion(0, completion));
    EXPECT_EQ(PIMRankM6TestAccess::injectCompletion(h.pim, 0, completion),
              BGTargetedCompletionValidation::DUPLICATE);
    EXPECT_EQ(h.pim.targetedStatistics().targeted_completions_retired, 1);
    EXPECT_EQ(h.pim.targetedStatistics().duplicate_completion_rejections, 1);
}

TEST(CSCM6CompletionNegativeTest, UnknownCompletionIsRejected)
{
    TargetHarness h;
    BGTargetedCompletion completion;
    completion.identity = h.operation(0, 404).identity;
    EXPECT_EQ(PIMRankM6TestAccess::injectCompletion(h.pim, 0, completion),
              BGTargetedCompletionValidation::UNKNOWN_OPERATION);
    EXPECT_EQ(h.pim.targetedStatistics().unknown_completion_rejections, 1);
    EXPECT_EQ(h.pim.targetedStatistics().targeted_completions_retired, 0);
}

TEST(CSCM6CompletionNegativeTest, CompletionOpcodeMismatchIsRejected)
{
    TargetHarness h;
    const auto operation = h.operation(1, 202);
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(operation));
    h.pim.serviceBGTargeted(false, false);
    h.nextCycle();
    BGTargetedCompletion corrupted;
    corrupted.identity = operation.identity;
    corrupted.identity.opcode = static_cast<BGTargetedOpcode>(0xff);
    EXPECT_EQ(PIMRankM6TestAccess::injectCompletion(h.pim, 1, corrupted),
              BGTargetedCompletionValidation::IDENTITY_MISMATCH);
    BGTargetedCompletion valid;
    EXPECT_TRUE(h.pim.pollBGTargetedCompletion(1, valid));
    EXPECT_EQ(h.pim.targetedStatistics().identity_mismatch_rejections, 1);
    EXPECT_EQ(h.pim.targetedStatistics().targeted_completions_retired, 1);
}

TEST(CSCM6CompletionNegativeTest, CompletionMaskMismatchIsRejected)
{
    TargetHarness h;
    const auto operation = h.operation(2, 303);
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(operation));
    h.pim.serviceBGTargeted(false, false);
    h.nextCycle();
    BGTargetedCompletion corrupted;
    corrupted.identity = operation.identity;
    corrupted.identity.pimblock_mask = kBGTargetSecondPIMBlock;
    EXPECT_EQ(PIMRankM6TestAccess::injectCompletion(h.pim, 2, corrupted),
              BGTargetedCompletionValidation::IDENTITY_MISMATCH);
    BGTargetedCompletion valid;
    EXPECT_TRUE(h.pim.pollBGTargetedCompletion(2, valid));
    EXPECT_EQ(h.pim.targetedPIMBlockBusyMask(), 0);
    EXPECT_EQ(h.pim.targetedStatistics().targeted_completions_retired, 1);
}

TEST(CSCM6ErrorIsolationTest, BGLocalCompletionErrorDrainsOnlyThatBG)
{
    TargetHarness h;
    const auto bad = h.operation(0, 501);
    const auto good = h.operation(1, 502);
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(bad));
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(good));
    h.pim.serviceBGTargeted(false, false);
    h.nextCycle();
    BGTargetedCompletion corrupted;
    corrupted.identity = bad.identity;
    corrupted.identity.pimblock_mask = kBGTargetSecondPIMBlock;
    EXPECT_EQ(PIMRankM6TestAccess::injectCompletion(h.pim, 0, corrupted),
              BGTargetedCompletionValidation::IDENTITY_MISMATCH);
    EXPECT_EQ(h.pim.bgLifecycle(0), BGLifecycleState::ERROR_DRAINING);
    EXPECT_EQ(h.pim.bgLifecycle(1), BGLifecycleState::RUNNING);
    BGTargetedCompletion bad_completion;
    ASSERT_TRUE(h.pim.pollBGTargetedCompletion(0, bad_completion));
    h.nextCycle();
    EXPECT_EQ(h.pim.bgLifecycle(0), BGLifecycleState::ERROR);
    EXPECT_EQ(h.pim.executionMode(), RankExecutionMode::CSC_BG_TARGETED);
    h.nextCycle();
    BGTargetedCompletion good_completion;
    ASSERT_TRUE(h.pim.pollBGTargetedCompletion(1, good_completion));
    EXPECT_EQ(good_completion.identity.operation_id, good.identity.operation_id);
    EXPECT_NE(h.pim.executionMode(), RankExecutionMode::ERROR);
}

TEST(CSCM6ErrorIsolationTest, SharedPIMBlockOwnershipErrorEscalatesRank)
{
    TargetHarness h;
    ASSERT_TRUE(h.pim.submitBGTargetedOperation(h.operation(0, 601)));
    h.pim.serviceBGTargeted(false, false);
    ASSERT_NE(h.pim.targetedPIMBlockBusyMask(), 0);
    EXPECT_FALSE(PIMRankM6TestAccess::corruptBusyMaskAndValidate(h.pim, 1U << 1));
    EXPECT_EQ(h.pim.executionMode(), RankExecutionMode::ERROR_DRAINING);
    h.nextCycle();
    BGTargetedCompletion completion;
    ASSERT_TRUE(h.pim.pollBGTargetedCompletion(0, completion));
    h.nextCycle();
    EXPECT_EQ(h.pim.executionMode(), RankExecutionMode::ERROR);
    EXPECT_EQ(h.pim.bgLifecycle(0), BGLifecycleState::ERROR);
    EXPECT_EQ(h.pim.targetedOutstandingCount(), 0);
    EXPECT_EQ(h.pim.targetedPendingCompletionCount(), 0);
    EXPECT_EQ(h.pim.targetedPIMBlockBusyMask(), 0);
    EXPECT_FALSE(h.pim.submitBGTargetedOperation(h.operation(1, 602)));
}

TEST(CSCM6ErrorIsolationTest, AcceptedMemoryAndTargetedOpDrainTogether)
{
    using namespace csc_descriptor;
    std::array<std::vector<uint8_t>, 64> values, rows;
    std::array<std::vector<CSCDescriptor>, 64> descriptors;
    std::array<std::vector<float>, 64> x;
    std::array<CSCBGImageView, 64> views;
    for (uint32_t bg = 0; bg < 64; ++bg)
    {
        values[bg].resize(32);
        rows[bg].resize(32);
        x[bg].push_back(2.0F);
        views[bg] = {&values[bg], &rows[bg], &descriptors[bg], &x[bg]};
    }
    const float value = 3.0F;
    const uint32_t row0 = 0, row1 = 1;
    std::memcpy(values[0].data(), &value, 4);
    std::memcpy(rows[0].data(), &row0, 4);
    std::memcpy(values[1].data(), &value, 4);
    std::memcpy(rows[1].data(), &row1, 4);
    descriptors[0].push_back({0, 0, 1, 0, 0, 0});
    descriptors[1].push_back({0, 0, 1, 0, 1, 1});

    CSCNativeExecution execution;
    execution.launch(views, 2, 2);
    uint64_t guard = 0;
    while (!execution.hasOutstandingRequest() && guard++ < 1000) execution.tick();
    ASSERT_TRUE(execution.hasOutstandingRequest());

    BGTargetedOperation operation;
    operation.identity.operation_id = 0xf000000000000001ULL;
    operation.identity.channel = 0;
    operation.identity.rank = 0;
    operation.identity.local_bg = 0;
    operation.identity.pimblock_mask = kBGTargetFirstPIMBlock;
    operation.identity.worker_id = 0;
    operation.identity.sequence = 1;
    operation.identity.generation = CSCM6CompletionTestAccess::generation(execution);
    operation.valid_count = 1;
    operation.contexts[0].valid = true;
    operation.contexts[0].lhs.fp32Data_[0] = 3.0F;
    operation.contexts[0].rhs.fp32Data_[0] = 2.0F;
    ASSERT_TRUE(CSCM6CompletionTestAccess::submitTarget(execution, 0, operation));
    auto& rank = CSCM6CompletionTestAccess::rank(execution, 0);
    while (!rank.targetedPIMBlockBusyMask() && guard++ < 2000) execution.tick();
    ASSERT_NE(rank.targetedPIMBlockBusyMask(), 0);
    ASSERT_GT(CSCM6CompletionTestAccess::memoryOutstanding(execution), 0);

    execution.flushBG(0);
    BGTargetedCompletion completion;
    while ((execution.hasOutstandingRequest() || rank.targetedOutstandingCount()) &&
           guard++ < 200000)
    {
        execution.tick();
        rank.pollBGTargetedCompletion(0, completion);
    }
    ASSERT_LT(guard, 200000);
    while (!execution.done() && guard++ < 200000) execution.tick();
    EXPECT_TRUE(execution.isDone());
    EXPECT_EQ(execution.bgResultStatus(0), CSCResultStatus::INCOMPLETE_FLUSHED);
    EXPECT_EQ(CSCM6CompletionTestAccess::memoryOutstanding(execution), 0);
    EXPECT_EQ(rank.targetedOutstandingCount(), 0);
    EXPECT_EQ(rank.targetedPendingCompletionCount(), 0);
    EXPECT_EQ(rank.targetedPIMBlockBusyMask(), 0);
    bool bg1_partial = false;
    for (const auto& partial : execution.partials())
        if (partial.global_bg_id == 1 && partial.row_idx == 1 && partial.value == 6.0F)
            bg1_partial = true;
    EXPECT_TRUE(bg1_partial);
}
