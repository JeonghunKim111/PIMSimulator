#include "csc/CSCDescriptorEngine.h"
#include "csc/CSCPartialResultPath.h"
#include "tests/csc/CSCLayout.h"
#include "tests/csc/CSCMatrixLoader.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

using namespace csc_descriptor;

namespace {

std::vector<bool> lifecycle(bool value = true,
                            uint32_t global_bg_count = kCSCGlobalBGs)
{
    return std::vector<bool>(global_bg_count, value);
}

CSCBGAOutput output(uint32_t row, uint64_t sequence, float value = 1.0F,
                    uint32_t contributions = 1, uint32_t generation = 7)
{
    CSCBGAOutput result;
    result.row_idx = row;
    result.value = value;
    result.generation = generation;
    result.output_sequence = sequence;
    result.contribution_count = contributions;
    result.reason = CSCBGAOutputReason::FINAL_DRAIN;
    return result;
}

CSCBGAOutputTransferResult transfer(CSCPartialResultPath& path, uint32_t bg,
                                    const CSCBGAOutput& value,
                                    uint32_t* accepts = nullptr)
{
    bool available = true;
    CSCBGAOutputPortCallbacks source;
    source.has_output = [&](uint32_t) { return available; };
    source.peek_output =
        [&](uint32_t) -> const CSCBGAOutput& { return value; };
    source.accept_output = [&](uint32_t) {
        available = false;
        if (accepts) ++*accepts;
    };
    return transferCSCBGAOutput(bg, source, path);
}

CSCPartialResultPathConfig pathConfig(uint32_t buffer = 8,
                                      uint32_t pending = 4,
                                      uint32_t latency = 2,
                                      uint32_t inflight = 2,
                                      uint32_t issue = 1)
{
    CSCPartialResultPathConfig config;
    config.global_bg_count = kCSCGlobalBGs;
    config.bank_groups_per_rank = 4;
    config.buffer_capacity_bursts_per_bg = buffer;
    config.pending_capacity_bursts_per_bg = pending;
    config.write_latency_cycles = latency;
    config.max_inflight_writes_per_bg = inflight;
    config.write_issue_limit_per_rank_per_cycle = issue;
    return config;
}

void acceptRecords(CSCPartialResultPath& path, uint32_t bg, uint32_t count,
                   uint64_t first_sequence = 1,
                   uint32_t contributions = 1)
{
    for (uint32_t i = 0; i < count; ++i)
        ASSERT_EQ(transfer(path, bg,
                           output(i, first_sequence + i,
                                  float(i) + 0.25F, contributions)),
                  CSCBGAOutputTransferResult::ACCEPTED);
}

void finish(CSCPartialResultPath& path, uint64_t first_cycle = 1,
            uint64_t limit = 100)
{
    const auto done = lifecycle();
    for (uint64_t cycle = first_cycle;
         !path.partialWritebackComplete() && cycle <= limit; ++cycle)
        path.step(cycle, done);
    ASSERT_TRUE(path.partialWritebackComplete())
        << path.errorMessage();
}

struct Views {
    std::array<std::vector<float>, kCSCGlobalBGs> x;
    std::array<CSCBGImageView, kCSCGlobalBGs> views;

    Views(const CSCLayout& layout, const std::vector<float>& original_x)
    {
        for (uint32_t bg = 0; bg < kCSCGlobalBGs; ++bg) {
            for (uint32_t column : layout.bg[bg].x_slot_to_original_col)
                x[bg].push_back(original_x[column]);
            views[bg] = {&layout.bg[bg].values,
                         &layout.bg[bg].row_indices,
                         &layout.bg[bg].descriptors, &x[bg]};
        }
    }
};

CSCBGAIntegrationConfig integrationConfig(uint32_t rows,
                                          uint32_t entries = 16,
                                          uint32_t output_depth = 16)
{
    CSCBGAIntegrationConfig config;
    config.enabled = true;
    config.accumulator.rows = rows;
    config.accumulator.accumulator_entries = entries;
    config.accumulator.compare_width = entries;
    config.accumulator.output_queue_depth = output_depth;
    config.output_consumer_mode =
        CSCBGAOutputConsumerMode::PARTIAL_RESULT_WRITEBACK;
    return config;
}

}  // namespace

TEST(CSCM7BPartialWritebackTest, RecordContractPreservesBGAOutput)
{
    CSCPartialResultPath path(pathConfig(), 16, 7);
    const float value = -3.125F;
    ASSERT_EQ(transfer(path, 5, output(9, 1, value, 6)),
              CSCBGAOutputTransferResult::ACCEPTED);
    finish(path);
    const auto& burst = path.residentBurst(5, 0);
    ASSERT_EQ(burst.valid_record_count, 1U);
    const auto& envelope = burst.records[0];
    EXPECT_EQ(envelope.record.row_idx, 9U);
    EXPECT_EQ(std::memcmp(&envelope.record.value, &value, sizeof(value)), 0);
    EXPECT_EQ(envelope.global_bg_id, 5U);
    EXPECT_EQ(envelope.generation, 7U);
    EXPECT_EQ(envelope.output_sequence, 1U);
    EXPECT_EQ(envelope.contribution_count, 6U);
    EXPECT_EQ(sizeof(CSCPartialResultRecord), 8U);
    EXPECT_EQ(path.counters().writeback_useful_bytes, 8U);
}

TEST(CSCM7BPartialWritebackTest, StorageUsesConfiguredTopology)
{
    auto config = pathConfig();
    config.global_bg_count = 8;
    config.bank_groups_per_rank = 4;
    CSCPartialResultPath path(config, 16, 7);
    EXPECT_EQ(path.globalBGCount(), 8U);
    EXPECT_EQ(path.counters().peak_inflight_writes_by_bg.size(), 8U);
    acceptRecords(path, 7, 4);
    for (uint64_t cycle = 1; !path.partialWritebackComplete() &&
                             cycle < 20; ++cycle)
        path.step(cycle, lifecycle(true, 8));
    ASSERT_TRUE(path.partialWritebackComplete());
    EXPECT_EQ(path.residentBurstCount(7), 1U);

    CSCPartialResultPath invalid(config, 16, 7);
    EXPECT_EQ(transfer(invalid, 8, output(0, 1)),
              CSCBGAOutputTransferResult::DESTINATION_BACKPRESSURE);
    EXPECT_TRUE(invalid.hasError());
    EXPECT_NE(invalid.errorMessage().find("out of range"),
              std::string::npos);
}

TEST(CSCM7BPartialWritebackTest, FourRecordsPackOneFullBurst)
{
    CSCPartialResultPath path(pathConfig(), 16, 7);
    acceptRecords(path, 2, 4);
    EXPECT_EQ(path.pendingBurstCount(2), 1U);
    EXPECT_EQ(path.packerRecordCount(2), 0U);
    finish(path);
    const auto& burst = path.residentBurst(2, 0);
    EXPECT_FALSE(burst.tail);
    EXPECT_EQ(burst.valid_record_count, 4U);
    for (uint32_t i = 0; i < 4; ++i)
        EXPECT_EQ(burst.records[i].output_sequence, i + 1);
    const auto& counters = path.counters();
    EXPECT_EQ(counters.full_writeback_bursts, 1U);
    EXPECT_EQ(counters.tail_writeback_bursts, 0U);
    EXPECT_EQ(counters.writeback_useful_bytes, 32U);
    EXPECT_EQ(counters.writeback_transferred_bytes, 32U);
    EXPECT_EQ(counters.writeback_padding_bytes, 0U);
}

void expectTailPacking(uint32_t count)
{
    CSCPartialResultPath path(pathConfig(), 16, 7);
    acceptRecords(path, 1, count);
    finish(path);
    const auto& burst = path.residentBurst(1, 0);
    EXPECT_TRUE(burst.tail);
    EXPECT_EQ(burst.valid_record_count, count);
    EXPECT_EQ(path.counters().tail_writeback_bursts, 1U);
    EXPECT_EQ(path.counters().writeback_useful_bytes, count * 8U);
    EXPECT_EQ(path.counters().writeback_transferred_bytes, 32U);
    EXPECT_EQ(path.counters().writeback_padding_bytes, 32U - count * 8U);
}

TEST(CSCM7BPartialWritebackTest, TailOneRecord)
{
    expectTailPacking(1);
}

TEST(CSCM7BPartialWritebackTest, TailTwoRecords)
{
    expectTailPacking(2);
}

TEST(CSCM7BPartialWritebackTest, TailThreeRecords)
{
    expectTailPacking(3);
}

TEST(CSCM7BPartialWritebackTest, EmptyBGProducesNoTailAndTailFlushesOnce)
{
    CSCPartialResultPath empty(pathConfig(), 8, 7);
    finish(empty);
    EXPECT_EQ(empty.counters().total_writeback_bursts, 0U);
    EXPECT_EQ(empty.counters().writeback_transferred_bytes, 0U);

    CSCPartialResultPath one(pathConfig(), 8, 7);
    acceptRecords(one, 0, 1);
    finish(one);
    const auto bursts = one.counters().total_writeback_bursts;
    for (uint64_t cycle = 20; cycle < 30; ++cycle)
        one.step(cycle, lifecycle());
    EXPECT_EQ(one.counters().total_writeback_bursts, bursts);
}

TEST(CSCM7BPartialWritebackTest, ReservationPreventsFourthRecordOverflow)
{
    CSCPartialResultPath path(pathConfig(4, 1, 5, 1, 1), 32, 7);
    acceptRecords(path, 0, 4);
    acceptRecords(path, 0, 3, 5);
    uint32_t accepts = 0;
    const auto fourth = output(7, 8);
    EXPECT_EQ(transfer(path, 0, fourth, &accepts),
              CSCBGAOutputTransferResult::DESTINATION_BACKPRESSURE);
    EXPECT_EQ(accepts, 0U);
    EXPECT_EQ(path.packerRecordCount(0), 3U);
    path.step(1, lifecycle(false));
    EXPECT_EQ(transfer(path, 0, fourth, &accepts),
              CSCBGAOutputTransferResult::ACCEPTED);
    EXPECT_EQ(accepts, 1U);
    EXPECT_EQ(path.pendingBurstCount(0), 1U);
    EXPECT_EQ(path.counters().partial_records_generated, 8U);
}

TEST(CSCM7BPartialWritebackTest, InflightLimitBackpressuresThenResumes)
{
    CSCPartialResultPath path(pathConfig(4, 4, 3, 1, 1), 32, 7);
    acceptRecords(path, 0, 8);
    path.step(1, lifecycle(false));
    ASSERT_EQ(path.writeTrace().size(), 1U);
    path.step(2, lifecycle(false));
    path.step(3, lifecycle(false));
    EXPECT_EQ(path.writeTrace().size(), 1U);
    EXPECT_GT(path.counters().inflight_backpressure_cycles, 0U);
    path.step(4, lifecycle(false));
    ASSERT_EQ(path.writeTrace().size(), 2U);
    EXPECT_EQ(path.writeTrace()[0].burst_sequence, 1U);
    EXPECT_EQ(path.writeTrace()[1].burst_sequence, 2U);
    EXPECT_EQ(path.writeTrace()[1].issue_cycle, 4U);
}

TEST(CSCM7BPartialWritebackTest, BufferSlotIsReservedAtIssue)
{
    CSCPartialResultPath path(pathConfig(2, 4, 3, 2, 2), 32, 7);
    acceptRecords(path, 0, 8);
    path.step(1, lifecycle(false));
    EXPECT_EQ(path.inflightWriteCount(0), 2U);
    EXPECT_EQ(path.reservedBufferSlots(0), 2U);
    EXPECT_EQ(path.residentBurstCount(0), 0U);
    path.step(4, lifecycle(false));
    EXPECT_EQ(path.inflightWriteCount(0), 0U);
    EXPECT_EQ(path.reservedBufferSlots(0), 0U);
    EXPECT_EQ(path.residentBurstCount(0), 2U);
    EXPECT_FALSE(path.hasError());
}

TEST(CSCM7BPartialWritebackTest, RankRoundRobinAndIndependentRanks)
{
    CSCPartialResultPath path(pathConfig(4, 4, 4, 2, 1), 32, 7);
    acceptRecords(path, 0, 4);
    acceptRecords(path, 1, 4);
    acceptRecords(path, 4, 4);
    path.step(1, lifecycle(false));
    ASSERT_EQ(path.writeTrace().size(), 2U);
    EXPECT_EQ(path.writeTrace()[0].global_bg_id, 0U);
    EXPECT_EQ(path.writeTrace()[1].global_bg_id, 4U);
    path.step(2, lifecycle(false));
    ASSERT_EQ(path.writeTrace().size(), 3U);
    EXPECT_EQ(path.writeTrace()[2].global_bg_id, 1U);
}

TEST(CSCM7BPartialWritebackTest, BlockedBGDoesNotStopReadyPeer)
{
    CSCPartialResultPath path(pathConfig(1, 1, 2, 1, 1), 32, 7);
    acceptRecords(path, 0, 4);
    path.step(1, lifecycle(false));
    path.step(3, lifecycle(false));
    ASSERT_EQ(path.residentBurstCount(0), 1U);

    acceptRecords(path, 0, 4, 5);
    acceptRecords(path, 0, 3, 9);
    EXPECT_EQ(transfer(path, 0, output(11, 12)),
              CSCBGAOutputTransferResult::DESTINATION_BACKPRESSURE);
    acceptRecords(path, 1, 4);
    path.step(4, lifecycle(false));

    ASSERT_GE(path.writeTrace().size(), 2U);
    EXPECT_EQ(path.writeTrace().back().global_bg_id, 1U);
    EXPECT_EQ(path.inflightWriteCount(1), 1U);
    EXPECT_EQ(path.residentBurstCount(0), 1U);
}

TEST(CSCM7BPartialWritebackTest, WriteLatencyAndCausalSeparation)
{
    CSCPartialResultPath path(pathConfig(4, 4, 3, 2, 1), 16, 7);
    acceptRecords(path, 0, 4);
    EXPECT_EQ(path.counters().first_bga_output_accept_cycle, 0U);
    path.step(1, lifecycle(false));
    ASSERT_EQ(path.writeTrace().size(), 1U);
    EXPECT_EQ(path.writeTrace()[0].issue_cycle, 1U);
    EXPECT_EQ(path.writeTrace()[0].completion_cycle, 4U);
    path.step(3, lifecycle(false));
    EXPECT_EQ(path.residentBurstCount(0), 0U);
    path.step(4, lifecycle(false));
    EXPECT_EQ(path.residentBurstCount(0), 1U);
    EXPECT_EQ(path.counters().first_write_complete_cycle, 4U);
}

TEST(CSCM7BPartialWritebackTest, PhysicalBytesAndContributionsConserve)
{
    CSCPartialResultPath path(pathConfig(), 32, 7);
    acceptRecords(path, 0, 5, 1, 3);
    finish(path);
    const auto& counters = path.counters();
    EXPECT_EQ(counters.partial_records_generated, 5U);
    EXPECT_EQ(counters.partial_record_contribution_sum, 15U);
    EXPECT_EQ(counters.packed_record_contribution_sum, 15U);
    EXPECT_EQ(counters.resident_record_contribution_sum, 15U);
    EXPECT_EQ(counters.full_writeback_bursts, 1U);
    EXPECT_EQ(counters.tail_writeback_bursts, 1U);
    EXPECT_EQ(counters.writeback_useful_bytes, 40U);
    EXPECT_EQ(counters.writeback_transferred_bytes, 64U);
    EXPECT_EQ(counters.writeback_padding_bytes, 24U);
}

TEST(CSCM7BPartialWritebackTest, InvalidOrDuplicateIdentityIsError)
{
    CSCPartialResultPath invalid(pathConfig(), 4, 7);
    EXPECT_EQ(transfer(invalid, 0, output(4, 1)),
              CSCBGAOutputTransferResult::DESTINATION_BACKPRESSURE);
    EXPECT_TRUE(invalid.hasError());
    EXPECT_FALSE(invalid.partialWritebackComplete());

    CSCPartialResultPath duplicate(pathConfig(), 4, 7);
    ASSERT_EQ(transfer(duplicate, 0, output(0, 1)),
              CSCBGAOutputTransferResult::ACCEPTED);
    EXPECT_EQ(transfer(duplicate, 0, output(0, 1)),
              CSCBGAOutputTransferResult::DESTINATION_BACKPRESSURE);
    EXPECT_TRUE(duplicate.hasError());
    EXPECT_NE(duplicate.errorMessage().find("nonmonotonic"),
              std::string::npos);
}

TEST(CSCM7BPartialWritebackTest, CompletionWaitsForLastResidentWrite)
{
    CSCPartialResultPath path(pathConfig(4, 4, 3, 2, 1), 16, 7);
    acceptRecords(path, 0, 1);
    path.step(1, lifecycle());
    EXPECT_FALSE(path.partialWritebackComplete());
    path.step(2, lifecycle());
    EXPECT_FALSE(path.partialWritebackComplete());
    path.step(4, lifecycle());
    EXPECT_FALSE(path.partialWritebackComplete());
    path.step(5, lifecycle());
    EXPECT_TRUE(path.partialWritebackComplete());
    EXPECT_EQ(path.counters().partial_writeback_complete_cycle, 5U);
}

TEST(CSCM7BPartialWritebackTest,
     ProductionAcceptRetireWriteAndCompletionAreSeparated)
{
    auto layout = buildLayout(
        makeCSC(5, 1, {{0,0,1.0F},{1,0,2.0F},
                       {2,0,3.0F},{3,0,4.0F},{4,0,5.0F}}),
        MappingPolicy::External, {0});
    Views views(layout, {2.0F});
    CSCNativeExecution execution;
    execution.configurePartialResultWriteback(pathConfig(8, 4, 3, 2, 1));
    execution.enableProductionBGAIntegration(integrationConfig(5));
    execution.launch(views.views, 5, 5);

    uint64_t first_available = 0;
    uint64_t first_accept = 0;
    uint64_t first_retirement = 0;
    uint64_t final_drain_complete = 0;
    for (uint64_t guard = 0;
         !execution.partialWritebackComplete() && guard < 200000; ++guard) {
        const bool before = execution.hasBGAOutput(0);
        if (before && !first_available) first_available = execution.cycle();
        const auto records_before =
            execution.partialResultPath().counters().partial_records_generated;
        execution.tick();
        const auto& path = execution.partialResultPath();
        if (!first_accept &&
            path.counters().partial_records_generated > records_before)
            first_accept = execution.cycle();
        if (first_accept && !first_retirement &&
            execution.bankGroupAccumulator(0).counters()
                    .retired_output_contributions)
            first_retirement = execution.cycle();
        if (!final_drain_complete &&
            execution.bankGroupAccumulator(0).finalDrainComplete())
            final_drain_complete = execution.cycle();
    }
    ASSERT_TRUE(execution.partialWritebackComplete())
        << execution.errorMessage();
    const auto& counters = execution.partialResultPath().counters();
    EXPECT_GT(first_available, 0U);
    EXPECT_EQ(first_accept, first_available);
    EXPECT_GT(first_retirement, first_accept);
    EXPECT_GT(counters.first_write_complete_cycle, first_accept);
    EXPECT_EQ(counters.partial_records_generated, 5U);
    EXPECT_EQ(counters.partial_record_contribution_sum, 5U);
    EXPECT_EQ(counters.full_writeback_bursts, 1U);
    EXPECT_EQ(counters.tail_writeback_bursts, 1U);
    EXPECT_FALSE(execution.resultValid());
    EXPECT_EQ(first_available, 70U);
    EXPECT_EQ(first_accept, 70U);
    EXPECT_EQ(first_retirement, 71U);
    EXPECT_EQ(execution.partialResultPath().residentBurst(0, 0)
                  .formation_cycle, 73U);
    EXPECT_EQ(counters.first_write_issue_cycle, 74U);
    EXPECT_EQ(counters.first_write_complete_cycle, 77U);
    EXPECT_EQ(final_drain_complete, 75U);
    EXPECT_EQ(execution.partialResultPath().residentBurst(0, 1)
                  .formation_cycle, 75U);
    EXPECT_EQ(counters.last_write_issue_cycle, 76U);
    EXPECT_EQ(counters.last_write_complete_cycle, 79U);
    EXPECT_EQ(counters.partial_writeback_complete_cycle, 79U);
    RecordProperty("first_output_available_cycle", first_available);
    RecordProperty("first_output_accept_cycle", first_accept);
    RecordProperty("first_output_retirement_cycle", first_retirement);
    RecordProperty("full_burst_formation_cycle",
                   execution.partialResultPath().residentBurst(0, 0)
                       .formation_cycle);
    RecordProperty("first_write_issue_cycle",
                   counters.first_write_issue_cycle);
    RecordProperty("first_write_complete_cycle",
                   counters.first_write_complete_cycle);
    RecordProperty("final_bga_drain_complete_cycle",
                   final_drain_complete);
    RecordProperty("tail_formation_cycle",
                   execution.partialResultPath().residentBurst(0, 1)
                       .formation_cycle);
    RecordProperty("last_write_issue_cycle",
                   counters.last_write_issue_cycle);
    RecordProperty("last_write_complete_cycle",
                   counters.last_write_complete_cycle);
    RecordProperty("partial_writeback_complete_cycle",
                   counters.partial_writeback_complete_cycle);
}

TEST(CSCM7BPartialWritebackTest, BufferPressureKeepsBGAFrontStable)
{
    std::vector<COOEntry> entries;
    for (uint32_t row = 0; row < 12; ++row)
        entries.push_back({row, 0, float(row + 1)});
    auto layout =
        buildLayout(makeCSC(12, 1, entries), MappingPolicy::External, {0});
    Views views(layout, {1.0F});
    CSCNativeExecution execution;
    execution.configurePartialResultWriteback(pathConfig(1, 1, 2, 1, 1));
    execution.enableProductionBGAIntegration(
        integrationConfig(12, 4, 1));
    execution.launch(views.views, 12, entries.size());

    uint64_t guard = 0;
    while (!execution.hasBGAOutput(0) && guard++ < 200000) execution.tick();
    ASSERT_TRUE(execution.hasBGAOutput(0));
    while (execution.partialResultPath().residentBurstCount(0) == 0 &&
           guard++ < 200000)
        execution.tick();
    while (execution.partialResultPath().counters()
                   .packer_backpressure_cycles == 0 &&
           guard++ < 200000)
        execution.tick();
    ASSERT_TRUE(execution.hasBGAOutput(0));
    const auto held = execution.peekBGAOutput(0);
    const auto accepted =
        execution.partialResultPath().counters().partial_records_generated;
    for (uint32_t i = 0; i < 8; ++i) execution.tick();
    ASSERT_TRUE(execution.hasBGAOutput(0));
    EXPECT_EQ(execution.peekBGAOutput(0).payload.output_sequence,
              held.payload.output_sequence);
    EXPECT_EQ(execution.partialResultPath().counters().partial_records_generated,
              accepted);
    EXPECT_GT(execution.partialResultPath().counters()
                  .packer_backpressure_cycles, 0U);
    EXPECT_FALSE(execution.partialWritebackComplete());
    EXPECT_FALSE(execution.resultValid());
}
