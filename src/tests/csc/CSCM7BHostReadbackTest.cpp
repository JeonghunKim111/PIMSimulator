#include "csc/CSCPartialResultPath.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

using namespace csc_descriptor;

namespace csc_descriptor {

struct CSCM7BHostReadbackTestAccess {
    static void duplicateFirstRead(CSCPartialResultPath& path)
    {
        ASSERT_FALSE(path.inflight_reads_by_channel_.empty());
        ASSERT_FALSE(path.inflight_reads_by_channel_[0].empty());
        path.inflight_reads_by_channel_[0].push_back(
            path.inflight_reads_by_channel_[0].front());
        ++path.reserved_host_return_slots_;
    }

    static void discardReturnReservation(CSCPartialResultPath& path)
    {
        path.reserved_host_return_slots_ = 0;
    }
};

}  // namespace csc_descriptor

namespace {

CSCPartialResultPathConfig config(uint32_t channels = 1,
                                  uint32_t ranks = 1,
                                  uint32_t bgs = 4)
{
    CSCPartialResultPathConfig result;
    result.channel_count = channels;
    result.ranks_per_channel = ranks;
    result.bank_groups_per_rank = bgs;
    result.global_bg_count = channels * ranks * bgs;
    result.buffer_capacity_bursts_per_bg = 8;
    result.pending_capacity_bursts_per_bg = 8;
    result.write_latency_cycles = 2;
    result.max_inflight_writes_per_bg = 8;
    result.write_issue_limit_per_rank_per_cycle = 4;
    result.read_latency_cycles = 3;
    result.read_issue_limit_per_channel_per_cycle = 1;
    result.max_inflight_reads_per_channel = 4;
    result.host_return_queue_capacity_bursts = 16;
    return result;
}

CSCBGAOutput output(uint32_t row, uint64_t sequence,
                    uint32_t contributions = 1, float value = 1.0F)
{
    CSCBGAOutput result;
    result.row_idx = row;
    result.value = value;
    result.generation = 7;
    result.output_sequence = sequence;
    result.contribution_count = contributions;
    result.reason = CSCBGAOutputReason::FINAL_DRAIN;
    return result;
}

void accept(CSCPartialResultPath& path, uint32_t bg, uint32_t count,
            uint64_t first_sequence = 1, uint32_t contributions = 1)
{
    for (uint32_t index = 0; index < count; ++index) {
        const auto value =
            output(index, first_sequence + index, contributions,
                   float(index) + 0.25F);
        bool available = true;
        CSCBGAOutputPortCallbacks source;
        source.has_output = [&](uint32_t) { return available; };
        source.peek_output =
            [&](uint32_t) -> const CSCBGAOutput& { return value; };
        source.accept_output = [&](uint32_t) { available = false; };
        ASSERT_EQ(transferCSCBGAOutput(bg, source, path),
                  CSCBGAOutputTransferResult::ACCEPTED);
    }
}

std::vector<bool> done(const CSCPartialResultPath& path)
{
    return std::vector<bool>(path.globalBGCount(), true);
}

void stepUntilWriteback(CSCPartialResultPath& path,
                        uint64_t first_cycle = 1)
{
    for (uint64_t cycle = first_cycle;
         cycle < 200 && !path.partialWritebackComplete(); ++cycle)
        path.step(cycle, done(path));
    ASSERT_TRUE(path.partialWritebackComplete()) << path.errorMessage();
}

void stepUntilTransport(CSCPartialResultPath& path,
                        uint64_t first_cycle)
{
    for (uint64_t cycle = first_cycle;
         cycle < 400 && !path.hostReadTransportComplete(); ++cycle)
        path.step(cycle, done(path));
    ASSERT_TRUE(path.hostReadTransportComplete()) << path.errorMessage();
}

}  // namespace

TEST(CSCM7BHostReadbackTest, PostWritebackGatesReadIssue)
{
    CSCPartialResultPath path(config(), 64, 7);
    accept(path, 0, 4);
    path.step(1, done(path));
    path.step(2, done(path));
    EXPECT_FALSE(path.partialWritebackComplete());
    EXPECT_EQ(path.readbackCounters().read_requests_issued, 0U);
    path.step(3, done(path));
    ASSERT_TRUE(path.partialWritebackComplete());
    EXPECT_EQ(path.readbackCounters().host_readback_start_cycle, 3U);
    EXPECT_EQ(path.readbackCounters().read_requests_issued, 0U);
    path.step(4, done(path));
    EXPECT_EQ(path.readbackCounters().first_read_issue_cycle, 4U);
}

TEST(CSCM7BHostReadbackTest, ResidentOwnershipHeldUntilCompletion)
{
    CSCPartialResultPath path(config(), 64, 7);
    accept(path, 0, 4);
    stepUntilWriteback(path);
    path.step(4, done(path));
    EXPECT_EQ(path.residentBurstCount(0), 1U);
    EXPECT_EQ(path.readInflightBufferSlots(0), 1U);
    EXPECT_EQ(path.reservedHostReturnSlots(), 1U);
    path.step(6, done(path));
    EXPECT_EQ(path.residentBurstCount(0), 1U);
    path.step(7, done(path));
    EXPECT_EQ(path.residentBurstCount(0), 0U);
    EXPECT_EQ(path.readInflightBufferSlots(0), 0U);
    EXPECT_EQ(path.hostReturnQueueSize(), 1U);
}

TEST(CSCM7BHostReadbackTest, ReadLatencyAndReturnReservationAreExact)
{
    auto cfg = config();
    cfg.read_latency_cycles = 5;
    CSCPartialResultPath path(cfg, 64, 7);
    accept(path, 0, 4);
    stepUntilWriteback(path);
    path.step(4, done(path));
    ASSERT_EQ(path.readbackCounters().first_read_issue_cycle, 4U);
    EXPECT_EQ(path.reservedHostReturnSlots(), 1U);
    path.step(8, done(path));
    EXPECT_EQ(path.readbackCounters().read_requests_completed, 0U);
    path.step(9, done(path));
    EXPECT_EQ(path.readbackCounters().first_read_complete_cycle, 9U);
    EXPECT_EQ(path.reservedHostReturnSlots(), 0U);
}

TEST(CSCM7BHostReadbackTest, ReturnQueuePressureStopsAndResumesIssue)
{
    auto cfg = config();
    cfg.host_return_queue_capacity_bursts = 1;
    CSCPartialResultPath path(cfg, 64, 7);
    accept(path, 0, 8);
    stepUntilWriteback(path);
    path.step(4, done(path));
    path.step(7, done(path));
    ASSERT_EQ(path.hostReturnQueueSize(), 1U);
    EXPECT_EQ(path.readbackCounters().read_requests_issued, 1U);
    path.step(8, done(path));
    ASSERT_TRUE(path.hasHostReturnedBurst());
    path.acceptHostReturnedBurst();
    path.step(9, done(path));
    EXPECT_EQ(path.readbackCounters().read_requests_issued, 2U);
    EXPECT_GT(path.readbackCounters().return_queue_backpressure_cycles, 0U);
}

TEST(CSCM7BHostReadbackTest, ChannelInflightLimitStopsAndResumesIssue)
{
    auto cfg = config();
    cfg.max_inflight_reads_per_channel = 1;
    cfg.read_issue_limit_per_channel_per_cycle = 2;
    CSCPartialResultPath path(cfg, 64, 7);
    accept(path, 0, 4);
    accept(path, 1, 4);
    stepUntilWriteback(path);
    path.step(4, done(path));
    EXPECT_EQ(path.readbackCounters().read_requests_issued, 1U);
    path.step(5, done(path));
    EXPECT_EQ(path.readbackCounters().read_requests_issued, 1U);
    path.step(7, done(path));
    EXPECT_EQ(path.readbackCounters().read_requests_issued, 2U);
    EXPECT_GT(path.readbackCounters().read_inflight_backpressure_cycles, 0U);
}

TEST(CSCM7BHostReadbackTest, ChannelRoundRobinAndSameBGSequence)
{
    auto cfg = config();
    cfg.read_issue_limit_per_channel_per_cycle = 1;
    CSCPartialResultPath path(cfg, 64, 7);
    accept(path, 0, 8);
    accept(path, 1, 4);
    stepUntilWriteback(path);
    for (uint64_t cycle = 4; cycle <= 12; ++cycle)
        path.step(cycle, done(path));
    while (path.hasHostReturnedBurst()) {
        const auto& returned = path.peekHostReturnedBurst();
        if (returned.burst.global_bg_id == 0 &&
            returned.burst.writeback_burst_sequence == 2) {
            EXPECT_GE(path.readbackCounters().read_requests_issued, 3U);
        }
        path.acceptHostReturnedBurst();
    }
    EXPECT_EQ(path.readbackCounters().read_requests_issued, 3U);
    EXPECT_FALSE(path.hasError()) << path.errorMessage();
}

TEST(CSCM7BHostReadbackTest, IndependentChannelsCompleteInTotalOrder)
{
    auto cfg = config(2, 1, 2);
    cfg.read_issue_limit_per_channel_per_cycle = 1;
    CSCPartialResultPath path(cfg, 64, 7);
    accept(path, 2, 4);
    accept(path, 0, 4);
    stepUntilWriteback(path);
    const uint64_t start =
        path.counters().partial_writeback_complete_cycle + 1;
    path.step(start, done(path));
    EXPECT_EQ(path.readbackCounters().read_requests_issued, 2U);
    path.step(start + 3, done(path));
    ASSERT_EQ(path.hostReturnQueueSize(), 2U);
    path.step(start + 4, done(path));
    ASSERT_TRUE(path.hasHostReturnedBurst());
    EXPECT_EQ(path.peekHostReturnedBurst().channel_id, 0U);
    path.acceptHostReturnedBurst();
    ASSERT_TRUE(path.hasHostReturnedBurst());
    EXPECT_EQ(path.peekHostReturnedBurst().channel_id, 1U);
}

TEST(CSCM7BHostReadbackTest, FullAndTailPayloadAndBytesConserve)
{
    CSCPartialResultPath path(config(), 64, 7);
    accept(path, 0, 4, 1, 2);
    accept(path, 1, 3, 1, 5);
    stepUntilWriteback(path);
    stepUntilTransport(
        path, path.counters().partial_writeback_complete_cycle + 1);
    const auto& reads = path.readbackCounters();
    EXPECT_EQ(reads.read_requests_completed, 2U);
    EXPECT_EQ(reads.readback_records, 7U);
    EXPECT_EQ(reads.readback_contribution_sum, 23U);
    EXPECT_EQ(reads.readback_transferred_bytes, 64U);
    EXPECT_EQ(reads.readback_useful_bytes, 56U);
    EXPECT_EQ(reads.readback_padding_bytes, 8U);
    EXPECT_EQ(reads.readback_transferred_bytes,
              path.counters().writeback_transferred_bytes);
    ASSERT_TRUE(path.hasHostReturnedBurst());
    const auto& first = path.peekHostReturnedBurst().burst;
    EXPECT_EQ(first.valid_record_count, 4U);
    EXPECT_FALSE(first.tail);
    EXPECT_EQ(std::memcmp(&first.records[0].record.value,
                          &path.peekHostReturnedBurst()
                               .burst.records[0].record.value,
                          sizeof(float)), 0);
}

TEST(CSCM7BHostReadbackTest, TransportCompletionDoesNotRequireDelivery)
{
    CSCPartialResultPath path(config(), 64, 7);
    accept(path, 0, 1);
    stepUntilWriteback(path);
    stepUntilTransport(
        path, path.counters().partial_writeback_complete_cycle + 1);
    EXPECT_EQ(path.hostReturnQueueSize(), 1U);
    EXPECT_EQ(path.readbackCounters().returned_bursts_delivered, 0U);
    EXPECT_TRUE(path.hostReadTransportComplete());
    EXPECT_FALSE(path.hasError());
}

TEST(CSCM7BHostReadbackTest, TopologyConfiguredChannelRankMapping)
{
    CSCPartialResultPath path(config(2, 2, 2), 64, 7);
    accept(path, 7, 4);
    stepUntilWriteback(path);
    stepUntilTransport(
        path, path.counters().partial_writeback_complete_cycle + 1);
    path.step(path.currentCycle() + 1, done(path));
    ASSERT_TRUE(path.hasHostReturnedBurst());
    const auto& returned = path.peekHostReturnedBurst();
    EXPECT_EQ(returned.channel_id, 1U);
    EXPECT_EQ(returned.rank_id, 1U);
    EXPECT_EQ(returned.local_bg_id, 1U);
}

TEST(CSCM7BHostReadbackTest, ExactCycleContractIsDeterministic)
{
    CSCPartialResultPath path(config(), 64, 7);
    accept(path, 0, 4);
    for (uint64_t cycle = 1; cycle <= 7; ++cycle)
        path.step(cycle, done(path));
    const auto& writes = path.counters();
    const auto& reads = path.readbackCounters();
    EXPECT_EQ(writes.last_write_issue_cycle, 1U);
    EXPECT_EQ(writes.last_write_complete_cycle, 3U);
    EXPECT_EQ(writes.partial_writeback_complete_cycle, 3U);
    EXPECT_EQ(reads.host_readback_start_cycle, 3U);
    EXPECT_EQ(reads.first_read_issue_cycle, 4U);
    EXPECT_EQ(reads.first_read_complete_cycle, 7U);
    EXPECT_EQ(reads.read_transport_complete_cycle, 7U);
    EXPECT_FALSE(path.hasError()) << path.errorMessage();
}

TEST(CSCM7BHostReadbackTest, ResidentBurstIssuesReadOnceWithOwner)
{
    CSCPartialResultPath path(config(2, 2, 2), 64, 7);
    accept(path, 5, 4);
    stepUntilWriteback(path);
    path.step(path.currentCycle() + 1, done(path));
    EXPECT_EQ(path.readbackCounters().read_requests_issued, 1U);
    EXPECT_EQ(path.readInflightBufferSlots(5), 1U);
}

TEST(CSCM7BHostReadbackTest, ReadCompletionReleasesOnlyCompletedSlot)
{
    CSCPartialResultPath path(config(), 64, 7);
    accept(path, 0, 4);
    stepUntilWriteback(path);
    const uint64_t issue = path.currentCycle() + 1;
    path.step(issue, done(path));
    path.step(issue + 2, done(path));
    EXPECT_EQ(path.residentBurstCount(0), 1U);
    path.step(issue + 3, done(path));
    EXPECT_EQ(path.residentBurstCount(0), 0U);
}

TEST(CSCM7BHostReadbackTest, ReturnSlotIsReservedAtIssueNotCompletion)
{
    CSCPartialResultPath path(config(), 64, 7);
    accept(path, 0, 4);
    stepUntilWriteback(path);
    path.step(path.currentCycle() + 1, done(path));
    EXPECT_EQ(path.reservedHostReturnSlots(), 1U);
    EXPECT_EQ(path.hostReturnQueueSize(), 0U);
}

TEST(CSCM7BHostReadbackTest, BlockedBGDoesNotBlockReadyPeer)
{
    auto cfg = config();
    cfg.read_issue_limit_per_channel_per_cycle = 2;
    CSCPartialResultPath path(cfg, 64, 7);
    accept(path, 3, 4);
    stepUntilWriteback(path);
    path.step(path.currentCycle() + 1, done(path));
    EXPECT_EQ(path.readInflightBufferSlots(3), 1U);
    EXPECT_EQ(path.readbackCounters().read_requests_issued, 1U);
}

TEST(CSCM7BHostReadbackTest, SameBGBurstSequenceIsReturnedInOrder)
{
    auto cfg = config();
    cfg.read_issue_limit_per_channel_per_cycle = 2;
    CSCPartialResultPath path(cfg, 64, 7);
    accept(path, 0, 8);
    stepUntilWriteback(path);
    stepUntilTransport(
        path, path.counters().partial_writeback_complete_cycle + 1);
    path.step(path.currentCycle() + 1, done(path));
    ASSERT_TRUE(path.hasHostReturnedBurst());
    EXPECT_EQ(path.peekHostReturnedBurst().burst.writeback_burst_sequence,
              1U);
    path.acceptHostReturnedBurst();
    ASSERT_TRUE(path.hasHostReturnedBurst());
    EXPECT_EQ(path.peekHostReturnedBurst().burst.writeback_burst_sequence,
              2U);
}

TEST(CSCM7BHostReadbackTest, ReturnedBurstIsVisibleNextCycle)
{
    CSCPartialResultPath path(config(), 64, 7);
    accept(path, 0, 4);
    stepUntilWriteback(path);
    const uint64_t issue = path.currentCycle() + 1;
    path.step(issue, done(path));
    path.step(issue + 3, done(path));
    EXPECT_EQ(path.hostReturnQueueSize(), 1U);
    EXPECT_FALSE(path.hasHostReturnedBurst());
    path.step(issue + 4, done(path));
    EXPECT_TRUE(path.hasHostReturnedBurst());
}

TEST(CSCM7BHostReadbackTest, GenericConsumerTracksDeliveredBursts)
{
    CSCPartialResultPath path(config(), 64, 7);
    accept(path, 0, 4);
    stepUntilWriteback(path);
    stepUntilTransport(
        path, path.counters().partial_writeback_complete_cycle + 1);
    path.step(path.currentCycle() + 1, done(path));
    path.acceptHostReturnedBurst();
    EXPECT_EQ(path.readbackCounters().returned_bursts_delivered, 1U);
    EXPECT_EQ(path.hostReturnQueueSize(), 0U);
}

TEST(CSCM7BHostReadbackTest, DuplicateReadCompletionIsStickyError)
{
    CSCPartialResultPath path(config(), 64, 7);
    accept(path, 0, 4);
    stepUntilWriteback(path);
    const uint64_t issue = path.currentCycle() + 1;
    path.step(issue, done(path));
    CSCM7BHostReadbackTestAccess::duplicateFirstRead(path);
    path.step(issue + 3, done(path));
    EXPECT_TRUE(path.hasError());
    EXPECT_FALSE(path.hostReadTransportComplete());
    EXPECT_NE(path.errorMessage().find("in-flight"),
              std::string::npos);
}

TEST(CSCM7BHostReadbackTest, CompletionWithoutReservationIsStickyError)
{
    CSCPartialResultPath path(config(), 64, 7);
    accept(path, 0, 4);
    stepUntilWriteback(path);
    const uint64_t issue = path.currentCycle() + 1;
    path.step(issue, done(path));
    CSCM7BHostReadbackTestAccess::discardReturnReservation(path);
    path.step(issue + 3, done(path));
    EXPECT_TRUE(path.hasError());
    EXPECT_FALSE(path.hostReadTransportComplete());
    EXPECT_NE(path.errorMessage().find("reserved return slot"),
              std::string::npos);
}

TEST(CSCM7BHostReadbackTest, InvalidTopologyIsRejected)
{
    auto cfg = config(2, 1, 4);
    cfg.global_bg_count = 7;
    EXPECT_THROW(CSCPartialResultPath(cfg, 64, 7),
                 std::invalid_argument);
}
