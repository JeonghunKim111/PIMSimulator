#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "MultiChannelMemorySystem.h"
#include "PIMRank.h"
#include "csc/CSCDescriptorEngine.h"
#include "tests/csc/CSCExternalImage.h"
#include "tests/csc/CSCLayout.h"
#include "tests/csc/CSCMatrixLoader.h"

using namespace csc_descriptor;
using namespace DRAMSim;

namespace {
struct Views {
    std::array<std::vector<float>, 64> x;
    std::array<CSCBGImageView, 64> views;
    Views(const CSCLayout& layout, const std::vector<float>& original_x) {
        for (uint32_t bg = 0; bg < 64; ++bg) {
            for (uint32_t column : layout.bg[bg].x_slot_to_original_col)
                x[bg].push_back(original_x[column]);
            views[bg] = {&layout.bg[bg].values, &layout.bg[bg].row_indices,
                         &layout.bg[bg].descriptors, &x[bg]};
        }
    }
};
CSCBGAIntegrationConfig config(uint32_t rows, uint32_t entries = 16,
                               uint32_t input_depth = 16,
                               uint32_t output_depth = 16) {
    CSCBGAIntegrationConfig c;
    c.enabled = true;
    c.accumulator.rows = rows;
    c.accumulator.accumulator_entries = entries;
    c.accumulator.compare_width = entries;
    c.accumulator.input_queue_depth = input_depth;
    c.accumulator.output_queue_depth = output_depth;
    c.output_consumer_mode = CSCBGAOutputConsumerMode::VALIDATION_ROUND_ROBIN;
    return c;
}
uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
void runUntil(CSCNativeExecution& execution, const std::function<bool()>& done,
              uint64_t limit = 200000) {
    uint64_t guard = 0;
    while (!done() && guard++ < limit) execution.tick();
    ASSERT_TRUE(done()) << "cycle guard=" << limit;
}
CSCBGAOutputPortCallbacks unusedOutput() {
    static CSCBGAOutput output;
    CSCBGAOutputPortCallbacks c;
    c.has_output = [](uint32_t) { return false; };
    c.peek_output = [](uint32_t) -> const CSCBGAOutput& { return output; };
    c.accept_output = [](uint32_t) {};
    return c;
}
}  // namespace

TEST(CSCM7A2CycleHandshakeTest, ProductionUpdateStepsEnabledBGAsExactlyOnce) {
    auto memory = std::make_shared<MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_csc_fp32.ini", ".",
        "m7a2_cycle_step", 256 * 16);
    auto& rank = *memory->channels.at(0)->ranks->at(0)->pimRank;
    const uint64_t rank_cycle = rank.currentClockCycle;
    memory->update();
    EXPECT_EQ(rank.currentClockCycle, rank_cycle + 1);
    EXPECT_FALSE(rank.cscBGAEnabled());
    rank.configureCSCBGAs(config(64), 1);
    memory->update();
    for (uint32_t bg = 0; bg < kBGsPerRank; ++bg)
        EXPECT_EQ(rank.cscBGA(bg).cycle(), 1U);
    memory->update();
    for (uint32_t bg = 0; bg < kBGsPerRank; ++bg)
        EXPECT_EQ(rank.cscBGA(bg).cycle(), 2U);
}

TEST(CSCM7A2CycleHandshakeTest, ExactAcceptedFifoLookupAndInsertCycles) {
    std::vector<COOEntry> entries;
    for (uint32_t row = 0; row < 16; ++row)
        entries.push_back({row, 0, float(row + 1)});
    auto layout = buildLayout(makeCSC(16, 1, entries), MappingPolicy::External, {0});
    Views views(layout, {2.0F});
    CSCNativeExecution execution;
    execution.enableProductionBGAIntegration(config(16));
    execution.launch(views.views, 16, 16);
    uint64_t request_issue = 0, emit_entry = 0;
    while (execution.engine(0).counters().accepted_bga_batches == 0) {
        execution.tick();
        if (!request_issue && execution.engine(0).state() ==
                                  CSCDescriptorState::WAIT_TARGET_GRANT)
            request_issue = execution.cycle();
        if (!emit_entry && execution.engine(0).state() ==
                                CSCDescriptorState::EMIT_PARTIALS)
            emit_entry = execution.cycle();
        ASSERT_LT(execution.cycle(), 200000U);
    }
    const auto& engine = execution.engine(0);
    const uint64_t accepted = execution.cycle();
    ASSERT_TRUE(execution.bankGroupAccumulator(0).hasAcceptedPending(0));
    EXPECT_EQ(engine.lastBGAMaterializeEngineCycle(), accepted);
    EXPECT_EQ(engine.lastBGAAcceptEngineCycle(), accepted);
    EXPECT_EQ(engine.lastTargetPollEngineCycle(), emit_entry);
    EXPECT_EQ(engine.lastTargetCompletionCycle(), engine.lastTargetGrantCycle() + 1);
    execution.tick();
    const uint64_t fifo_commit = execution.cycle();
    EXPECT_EQ(execution.bankGroupAccumulator(0).inputQueueSize(0), 8U);
    EXPECT_EQ(execution.bankGroupAccumulator(0).counters().lookup_misses, 0U);
    execution.tick();
    const uint64_t lookup_start = execution.cycle();
    EXPECT_EQ(execution.bankGroupAccumulator(0).inputQueueSize(0), 7U);
    EXPECT_EQ(execution.bankGroupAccumulator(0).counters().lookup_misses, 0U);
    execution.tick();
    const uint64_t lookup_complete = execution.cycle();
    EXPECT_EQ(execution.bankGroupAccumulator(0).counters().lookup_misses, 1U);
    execution.tick();
    const uint64_t insert_commit = execution.cycle();
    EXPECT_EQ(execution.bankGroupAccumulator(0).counters().inserts, 1U);
    EXPECT_EQ(fifo_commit, accepted + 1);
    EXPECT_EQ(lookup_start, accepted + 2);
    EXPECT_EQ(lookup_complete, accepted + 3);
    EXPECT_EQ(insert_commit, accepted + 4);
    EXPECT_LT(request_issue, emit_entry);
    EXPECT_EQ(accepted, emit_entry + 1);
    RecordProperty("target_request_issue_cycle", request_issue);
    RecordProperty("target_grant_cycle", engine.lastTargetGrantCycle());
    RecordProperty("target_completion_cycle", engine.lastTargetCompletionCycle());
    RecordProperty("engine_completion_poll_cycle", emit_entry);
    RecordProperty("emit_entry_cycle", emit_entry);
    RecordProperty("batch_materialize_accept_cycle", accepted);
    RecordProperty("bga_fifo_commit_cycle", fifo_commit);
    RecordProperty("bga_lookup_start_cycle", lookup_start);
    RecordProperty("bga_lookup_complete_cycle", lookup_complete);
    RecordProperty("bga_insert_commit_cycle", insert_commit);
}

TEST(CSCM7A2CycleHandshakeTest, FullTailAndDescriptorBoundaryMaterialization) {
    std::vector<COOEntry> entries;
    for (uint32_t row = 0; row < 8; ++row)
        entries.push_back({row, 0, float(row + 1)});
    entries.push_back({8, 1, -1.0F});
    entries.push_back({9, 1, 0.5F});
    entries.push_back({10, 1, 7.0F});
    auto layout = buildLayout(makeCSC(11, 2, entries), MappingPolicy::External,
                              {0, 0});
    Views views(layout, {2.0F, -3.0F});
    CSCNativeExecution execution;
    execution.enableProductionBGAIntegration(config(11));
    execution.launch(views.views, 11, 11);
    runUntil(execution, [&] {
        return execution.engine(0).counters().accepted_bga_batches == 1;
    });
    const auto first = *execution.engine(0).lastAcceptedBGABatch();
    ASSERT_EQ(first.valid_count, 8U);
    ASSERT_EQ(first.payload.partials.size(), 8U);
    EXPECT_EQ(first.descriptor_index, 0U);
    EXPECT_EQ(first.chunk_index, 0U);
    EXPECT_EQ(first.descriptor_chunk_ordinal, 1U);
    EXPECT_EQ(first.payload.sequence, 1U);
    EXPECT_EQ(first.payload.generation, 1U);
    for (uint32_t lane = 0; lane < 8; ++lane) {
        EXPECT_EQ(first.payload.partials[lane].row_idx, lane);
        EXPECT_EQ(bits(first.payload.partials[lane].value),
                  bits(float(lane + 1) * 2.0F));
    }
    runUntil(execution, [&] {
        return execution.engine(0).counters().accepted_bga_batches == 2;
    });
    const auto second = *execution.engine(0).lastAcceptedBGABatch();
    ASSERT_EQ(second.valid_count, 3U);
    ASSERT_EQ(second.payload.partials.size(), 3U);
    EXPECT_EQ(second.descriptor_index, 1U);
    EXPECT_EQ(second.chunk_index, 0U);
    EXPECT_EQ(second.descriptor_chunk_ordinal, 2U);
    EXPECT_EQ(second.payload.sequence, 2U);
    EXPECT_EQ(second.payload.generation, 1U);
    EXPECT_EQ(second.payload.partials[0].row_idx, 8U);
    EXPECT_EQ(second.payload.partials[1].row_idx, 9U);
    EXPECT_EQ(second.payload.partials[2].row_idx, 10U);
    EXPECT_EQ(bits(second.payload.partials[0].value), bits(3.0F));
    EXPECT_EQ(bits(second.payload.partials[1].value), bits(-1.5F));
    EXPECT_EQ(bits(second.payload.partials[2].value), bits(-21.0F));
    runUntil(execution, [&] { return execution.done(); });
    const auto counters = execution.counters();
    EXPECT_EQ(counters.generated_bga_batches, 2U);
    EXPECT_EQ(counters.generated_bga_partials, 11U);
    EXPECT_EQ(counters.accepted_bga_batches, 2U);
    EXPECT_EQ(counters.accepted_bga_partials, 11U);
    EXPECT_TRUE(execution.partials().empty());
    EXPECT_FALSE(execution.resultValid());
    EXPECT_TRUE(execution.bgaComputeSubmitComplete());
    EXPECT_TRUE(execution.bgaOutputCompletionImplemented());
    EXPECT_THROW(execution.hostAccumulate(), std::logic_error);
    EXPECT_TRUE(execution.bankGroupAccumulator(0).conservationInvariant());
}

TEST(CSCM7A2CycleHandshakeTest, BackpressureRetriesOneStableBatch) {
    auto layout = buildLayout(makeCSC(2, 1, {{0, 0, 1.5F}, {1, 0, -2.0F}}),
                              MappingPolicy::External, {0});
    Views views(layout, {4.0F});
    CSCNativeExecution execution;
    auto c = config(2);
    std::vector<CSCBGAPartialBatch> observed;
    uint32_t rejects = 4;
    CSCBGAProducerCallbacks producer;
    producer.submit = [&](const CSCBGAPartialBatch& batch) {
        observed.push_back(batch);
        if (rejects) { --rejects; return CSCBGAInputResult::BACKPRESSURE; }
        return CSCBGAInputResult::ACCEPTED;
    };
    producer.producer_done = [](const CSCBGAProducerIdentity&) {};
    producer.request_final_drain = [](const CSCBGAProducerIdentity&) { return true; };
    execution.configureBGAIntegration(c, producer, unusedOutput());
    execution.launch(views.views, 2, 2);
    runUntil(execution, [&] { return execution.engine(0).computeSubmitDone(); });
    ASSERT_EQ(observed.size(), 5U);
    for (const auto& retry : observed) {
        EXPECT_EQ(retry.global_bg_id, observed[0].global_bg_id);
        EXPECT_EQ(retry.descriptor_index, observed[0].descriptor_index);
        EXPECT_EQ(retry.chunk_index, observed[0].chunk_index);
        EXPECT_EQ(retry.payload.generation, observed[0].payload.generation);
        EXPECT_EQ(retry.payload.sequence, observed[0].payload.sequence);
        ASSERT_EQ(retry.payload.partials.size(), observed[0].payload.partials.size());
        for (size_t lane = 0; lane < retry.payload.partials.size(); ++lane) {
            EXPECT_EQ(retry.payload.partials[lane].row_idx,
                      observed[0].payload.partials[lane].row_idx);
            EXPECT_EQ(bits(retry.payload.partials[lane].value),
                      bits(observed[0].payload.partials[lane].value));
        }
    }
    const auto counters = execution.counters();
    EXPECT_EQ(counters.generated_bga_batches, 1U);
    EXPECT_EQ(counters.accepted_bga_batches, 1U);
    EXPECT_EQ(counters.bga_backpressure_cycles, 4U);
    EXPECT_EQ(execution.engine(0).nextBGASequence(), 2U);
}

TEST(CSCM7A2CycleHandshakeTest, DuplicateAndProtocolResponsesBecomeEngineErrors) {
    for (const auto response : {CSCBGAInputResult::DUPLICATE,
                                CSCBGAInputResult::PROTOCOL_ERROR}) {
        auto layout = buildLayout(makeCSC(1, 1, {{0, 0, 2.0F}}),
                                  MappingPolicy::External, {0});
        Views views(layout, {3.0F});
        CSCNativeExecution execution;
        CSCBGAProducerCallbacks producer;
        producer.submit = [=](const CSCBGAPartialBatch&) { return response; };
        producer.producer_done = [](const CSCBGAProducerIdentity&) {};
        producer.request_final_drain = [](const CSCBGAProducerIdentity&) { return true; };
        execution.configureBGAIntegration(config(1), producer, unusedOutput());
        execution.launch(views.views, 1, 1);
        runUntil(execution, [&] { return execution.done(); });
        EXPECT_TRUE(execution.hasFailed());
        EXPECT_EQ(execution.failedEngine(), 0);
        EXPECT_NE(execution.errorMessage().find(
                      response == CSCBGAInputResult::DUPLICATE
                          ? "DUPLICATE" : "PROTOCOL_ERROR"),
                  std::string::npos);
        const auto counters = execution.counters();
        EXPECT_EQ(counters.bga_duplicate_errors,
                  response == CSCBGAInputResult::DUPLICATE ? 1U : 0U);
        EXPECT_EQ(counters.bga_protocol_errors,
                  response == CSCBGAInputResult::PROTOCOL_ERROR ? 1U : 0U);
    }
}

TEST(CSCM7A2CycleHandshakeTest, ProductionBackpressureIsBGLocal) {
    std::vector<COOEntry> entries;
    std::vector<uint32_t> mapping;
    for (uint32_t column = 0; column < 8; ++column) {
        entries.push_back({column, column, float(column + 1)});
        mapping.push_back(0);
    }
    entries.push_back({15, 8, 2.0F});
    mapping.push_back(1);
    auto layout = buildLayout(makeCSC(16, 9, entries), MappingPolicy::External,
                              mapping);
    Views views(layout, std::vector<float>(9, 1.0F));
    CSCNativeExecution execution;
    auto pressure_config = config(16, 1, 1, 1);
    pressure_config.output_consumer_mode = CSCBGAOutputConsumerMode::EXTERNAL;
    execution.enableProductionBGAIntegration(pressure_config);
    execution.launch(views.views, 16, entries.size());
    runUntil(execution, [&] {
        return execution.engine(0).counters().bga_backpressure_cycles >= 3;
    });
    ASSERT_EQ(execution.engine(0).state(), CSCDescriptorState::EMIT_PARTIALS);
    ASSERT_TRUE(execution.engine(0).pendingBGABatch());
    const auto held = *execution.engine(0).pendingBGABatch();
    const auto bg1_accepted = execution.engine(1).counters().accepted_bga_batches;
    EXPECT_EQ(bg1_accepted, 1U);
    EXPECT_TRUE(execution.engine(1).isDone());
    EXPECT_EQ(execution.bankGroupAccumulator(1).counters().accepted_batches, 1U);
    EXPECT_EQ(execution.bankGroupAccumulator(2).counters().accepted_batches, 0U);
    EXPECT_EQ(execution.bankGroupAccumulator(3).counters().accepted_batches, 0U);
    execution.tick();
    execution.tick();
    ASSERT_TRUE(execution.engine(0).pendingBGABatch());
    const auto& retry = *execution.engine(0).pendingBGABatch();
    EXPECT_EQ(retry.payload.sequence, held.payload.sequence);
    EXPECT_EQ(retry.descriptor_index, held.descriptor_index);
    EXPECT_EQ(retry.chunk_index, held.chunk_index);
    EXPECT_EQ(execution.engine(1).counters().accepted_bga_batches, bg1_accepted);
}

TEST(CSCM7A2CycleHandshakeTest, DisabledPathRemainsHostAuthoritative) {
    auto layout = buildLayout(makeCSC(2, 1, {{0, 0, 2.0F}, {1, 0, 3.0F}}),
                              MappingPolicy::RoundRobin);
    Views views(layout, {4.0F});
    CSCNativeExecution execution;
    execution.launch(views.views, 2, 2);
    runUntil(execution, [&] { return execution.done(); });
    EXPECT_TRUE(execution.resultValid());
    EXPECT_EQ(execution.hostAccumulate(), (std::vector<float>{8.0F, 12.0F}));
    EXPECT_EQ(execution.counters().generated_bga_batches, 0U);
    EXPECT_FALSE(execution.productionBGAEnabled());
}

TEST(CSCM7A2CycleHandshakeTest, ComparisonTraceRecordsOnceAtAcceptance) {
    auto layout = buildLayout(makeCSC(1, 1, {{0, 0, 2.0F}}),
                              MappingPolicy::External, {0});
    Views views(layout, {3.0F});
    CSCNativeExecution execution;
    auto c = config(1);
    c.record_comparison_trace = true;
    execution.enableProductionBGAIntegration(c);
    execution.launch(views.views, 1, 1);
    runUntil(execution, [&] { return execution.done(); });
    ASSERT_EQ(execution.partials().size(), 1U);
    EXPECT_EQ(execution.partials()[0].row_idx, 0U);
    EXPECT_EQ(bits(execution.partials()[0].value), bits(6.0F));
    EXPECT_EQ(execution.counters().accepted_bga_partials, 1U);
    EXPECT_THROW(execution.hostAccumulate(), std::logic_error);
}

TEST(CSCM7A2CycleHandshakeTest, ExternalToySubmitsEveryNNZToBGA) {
    const char* path = std::getenv("CSC_EXTERNAL_IMAGE");
    if (!path || !*path) GTEST_SKIP() << "set CSC_EXTERNAL_IMAGE";
    auto loaded = loadExternalPhysicalImage(path);
    auto& layout = loaded.layout;
    std::vector<float> x(layout.matrix.cols);
    for (uint32_t column = 0; column < x.size(); ++column)
        x[column] = float((column % 13) + 1) / 7;
    Views views(layout, x);
    CSCNativeExecution execution;
    execution.enableProductionBGAIntegration(config(layout.matrix.rows));
    execution.launch(views.views, layout.matrix.rows, layout.stats.nnz);
    runUntil(execution, [&] { return execution.done(); }, 2000000);
    const auto counters = execution.counters();
    EXPECT_EQ(counters.generated_bga_partials, layout.stats.nnz);
    EXPECT_EQ(counters.accepted_bga_partials, layout.stats.nnz);
    EXPECT_EQ(counters.generated_bga_batches, counters.total_chunks);
    EXPECT_EQ(counters.accepted_bga_batches, counters.total_chunks);
    EXPECT_TRUE(execution.partials().empty());
    EXPECT_FALSE(execution.resultValid());
    uint64_t accepted = 0;
    for (uint32_t bg = 0; bg < 64; ++bg) {
        accepted += execution.bankGroupAccumulator(bg)
                        .counters().accepted_valid_partials;
        EXPECT_TRUE(execution.bankGroupAccumulator(bg).conservationInvariant());
    }
    EXPECT_EQ(accepted, layout.stats.nnz);
}
