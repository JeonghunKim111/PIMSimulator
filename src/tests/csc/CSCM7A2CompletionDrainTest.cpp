#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <vector>

#include "csc/CSCDescriptorEngine.h"
#include "tests/csc/CSCExternalImage.h"
#include "tests/csc/CSCLayout.h"
#include "tests/csc/CSCMatrixLoader.h"

using namespace csc_descriptor;

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
CSCBGAIntegrationConfig config(uint32_t rows,
                               CSCBGAOutputConsumerMode mode,
                               uint32_t entries = 16,
                               uint32_t output_depth = 16) {
    CSCBGAIntegrationConfig c;
    c.enabled = true;
    c.accumulator.rows = rows;
    c.accumulator.accumulator_entries = entries;
    c.accumulator.compare_width = entries;
    c.accumulator.output_queue_depth = output_depth;
    c.output_consumer_mode = mode;
    return c;
}
void runUntil(CSCNativeExecution& execution, const std::function<bool()>& done,
              uint64_t limit = 500000) {
    uint64_t guard = 0;
    while (!done() && guard++ < limit) execution.tick();
    ASSERT_TRUE(done()) << "cycle guard=" << limit << " state="
                        << int(execution.bgaExecutionState());
}
}  // namespace

TEST(CSCM7A2CompletionDrainTest, LastAcceptedCompletesProducerAndEmptyBGsExactlyOnce) {
    auto layout = buildLayout(makeCSC(2, 1, {{0, 0, 2.0F}, {1, 0, -3.0F}}),
                              MappingPolicy::External, {0});
    Views views(layout, {4.0F});
    CSCNativeExecution execution;
    execution.enableProductionBGAIntegration(
        config(2, CSCBGAOutputConsumerMode::VALIDATION_ROUND_ROBIN));
    execution.launch(views.views, 2, 2);

    runUntil(execution, [&] { return execution.engine(0).computeSubmitDone(); });
    EXPECT_TRUE(execution.engine(0).producerDoneSent());
    EXPECT_TRUE(execution.engine(0).finalDrainRequested());
    EXPECT_EQ(execution.engine(0).counters().bga_producer_done_calls, 1U);
    EXPECT_EQ(execution.engine(0).counters().bga_final_drain_requests, 1U);
    for (uint32_t bg = 1; bg < 64; ++bg) {
        EXPECT_TRUE(execution.engine(bg).computeSubmitDone());
        EXPECT_EQ(execution.engine(bg).counters().empty_bg_producer_completions, 1U);
        EXPECT_EQ(execution.engine(bg).counters().generated_bga_batches, 0U);
    }
    EXPECT_TRUE(execution.bgaComputeSubmitComplete());
    EXPECT_FALSE(execution.bgaExecutionComplete());
    runUntil(execution, [&] { return execution.bgaExecutionComplete(); });
    EXPECT_TRUE(execution.bgaDrainComplete());
    EXPECT_FALSE(execution.resultValid());
    EXPECT_THROW(execution.hostAccumulate(), std::logic_error);
    const auto counters = execution.counters();
    EXPECT_EQ(counters.bga_producer_done_calls, 64U);
    EXPECT_EQ(counters.bga_final_drain_requests, 64U);
    EXPECT_EQ(counters.empty_bg_producer_completions, 63U);
    EXPECT_EQ(counters.bga_output_contributions_accepted, 2U);
}

TEST(CSCM7A2CompletionDrainTest, ExternalAcceptRetiresOnlyOnFollowingGlobalTick) {
    auto layout = buildLayout(makeCSC(1, 1, {{0, 0, 2.0F}}),
                              MappingPolicy::External, {0});
    Views views(layout, {3.0F});
    CSCNativeExecution execution;
    execution.enableProductionBGAIntegration(
        config(1, CSCBGAOutputConsumerMode::EXTERNAL));
    execution.launch(views.views, 1, 1);
    runUntil(execution, [&] { return execution.hasBGAOutput(0); });
    ASSERT_TRUE(execution.bgaComputeSubmitComplete());
    EXPECT_EQ(execution.bgaExecutionState(),
              CSCBGAExecutionState::WAITING_FOR_EXTERNAL_OUTPUT_CONSUMER);
    const auto output = execution.peekBGAOutput(0);
    const uint64_t accept_cycle = execution.cycle();
    execution.acceptBGAOutput(0);
    EXPECT_TRUE(execution.hasBGAOutput(0));
    EXPECT_FALSE(execution.bgaDrainComplete());
    EXPECT_FALSE(execution.bgaExecutionComplete());
    execution.tick();
    EXPECT_EQ(execution.cycle(), accept_cycle + 1);
    EXPECT_FALSE(execution.hasBGAOutput(0));
    EXPECT_TRUE(execution.bgaDrainComplete());
    EXPECT_TRUE(execution.bgaExecutionComplete());
    EXPECT_EQ(output.payload.reason, CSCBGAOutputReason::FINAL_DRAIN);
    EXPECT_EQ(execution.bgaValidationCollector().accepted_contributions, 1U);
}

TEST(CSCM7A2CompletionDrainTest, CapacityPressureConsumesEvictionAndFinalDrain) {
    std::vector<COOEntry> entries{{0,0,1.0F},{1,0,2.0F},{2,0,3.0F},
                                  {3,0,4.0F},{0,1,5.0F},{4,1,6.0F}};
    auto layout = buildLayout(makeCSC(5, 2, entries), MappingPolicy::External,
                              {0, 0});
    Views views(layout, {1.0F, 1.0F});
    CSCNativeExecution execution;
    execution.enableProductionBGAIntegration(config(
        5, CSCBGAOutputConsumerMode::VALIDATION_ROUND_ROBIN, 2, 1));
    execution.launch(views.views, 5, entries.size());
    runUntil(execution, [&] { return execution.bgaExecutionComplete(); });
    const auto& collector = execution.bgaValidationCollector();
    EXPECT_GT(collector.capacity_outputs, 0U);
    EXPECT_GT(collector.final_drain_outputs, 0U);
    EXPECT_EQ(collector.accepted_contributions, entries.size());
    EXPECT_EQ(collector.validation_y_double[0], 6.0);
    EXPECT_EQ(collector.validation_y_double[4], 6.0);
    EXPECT_TRUE(execution.bgaValidationSemanticMatches());
    EXPECT_TRUE(execution.bankGroupAccumulator(0).conservationInvariant());
    EXPECT_EQ(execution.bankGroupAccumulator(0).liveContributionCount(), 0U);
    RecordProperty("accepted_partials", entries.size());
    RecordProperty("capacity_outputs", collector.capacity_outputs);
    RecordProperty("final_drain_outputs", collector.final_drain_outputs);
    RecordProperty("physical_outputs", collector.accepted_outputs);
    RecordProperty("output_contributions", collector.accepted_contributions);
    RecordProperty("complete_cycle", execution.cycle());
}

TEST(CSCM7A2CompletionDrainTest, ExternalPauseBackpressuresOutputThenResumesWithoutLoss) {
    auto layout = buildLayout(makeCSC(4, 1,
        {{0,0,1.0F},{1,0,2.0F},{2,0,3.0F},{3,0,4.0F}}),
        MappingPolicy::External, {0});
    Views views(layout, {1.0F});
    CSCNativeExecution execution;
    execution.enableProductionBGAIntegration(
        config(4, CSCBGAOutputConsumerMode::EXTERNAL, 1, 1));
    execution.launch(views.views, 4, 4);
    runUntil(execution, [&] { return execution.hasBGAOutput(0); });
    const auto held = execution.peekBGAOutput(0);
    const uint64_t pause_cycle = execution.cycle();
    for (int i = 0; i < 8; ++i) execution.tick();
    ASSERT_TRUE(execution.hasBGAOutput(0));
    const auto still_held = execution.peekBGAOutput(0);
    EXPECT_EQ(still_held.payload.output_sequence, held.payload.output_sequence);
    EXPECT_EQ(execution.cycle(), pause_cycle + 8);
    EXPECT_FALSE(execution.bgaExecutionComplete());
    uint64_t accepted_outputs = 0;
    for (uint64_t guard = 0; !execution.bgaExecutionComplete() && guard < 1000;
         ++guard) {
        if (execution.hasBGAOutput(0)) {
            execution.acceptBGAOutput(0);
            ++accepted_outputs;
        }
        execution.tick();
    }
    EXPECT_TRUE(execution.bgaExecutionComplete());
    EXPECT_GT(accepted_outputs, 1U);
    EXPECT_EQ(execution.bgaValidationCollector().accepted_contributions, 4U);
    EXPECT_TRUE(execution.bgaValidationSemanticMatches());
}

TEST(CSCM7A2CompletionDrainTest, RoundRobinCollectorMergesSameRowAcrossBGsSemantically) {
    auto layout = buildLayout(makeCSC(1, 2, {{0,0,1.25F},{0,1,2.75F}}),
                              MappingPolicy::External, {0, 1});
    Views views(layout, {2.0F, 3.0F});
    CSCNativeExecution execution;
    auto c = config(1, CSCBGAOutputConsumerMode::VALIDATION_ROUND_ROBIN);
    c.validation_consumer_accepts_per_cycle = 1;
    execution.enableProductionBGAIntegration(c);
    execution.launch(views.views, 1, 2);
    runUntil(execution, [&] { return execution.bgaExecutionComplete(); });
    const auto& collector = execution.bgaValidationCollector();
    ASSERT_EQ(collector.accepted_contributions, 2U);
    EXPECT_DOUBLE_EQ(collector.validation_y_double[0], 10.75);
    EXPECT_TRUE(execution.bgaValidationSemanticMatches());
    EXPECT_GE(collector.accepted_outputs_by_bg[0], 1U);
    EXPECT_GE(collector.accepted_outputs_by_bg[1], 1U);
    ASSERT_GE(collector.accepted_bg_trace_sample.size(), 2U);
    EXPECT_EQ(collector.accepted_bg_trace_sample[0], 0U);
    EXPECT_EQ(collector.accepted_bg_trace_sample[1], 1U);
    EXPECT_FALSE(execution.resultValid());
}

TEST(CSCM7A2CompletionDrainTest, ProducerCompletionExceptionIsNotSuccess) {
    auto layout = buildLayout(makeCSC(1, 1, {{0,0,1.0F}}),
                              MappingPolicy::External, {0});
    Views views(layout, {1.0F});
    CSCNativeExecution execution;
    auto c = config(1, CSCBGAOutputConsumerMode::EXTERNAL);
    CSCBGAProducerCallbacks producer;
    producer.submit = [](const CSCBGAPartialBatch&) {
        return CSCBGAInputResult::ACCEPTED;
    };
    producer.producer_done = [](const CSCBGAProducerIdentity&) {
        throw std::runtime_error("injected producer done failure");
    };
    producer.request_final_drain = [](const CSCBGAProducerIdentity&) {
        return true;
    };
    static CSCBGAOutput unused;
    CSCBGAOutputPortCallbacks output;
    output.has_output = [](uint32_t) { return false; };
    output.peek_output = [](uint32_t) -> const CSCBGAOutput& { return unused; };
    output.accept_output = [](uint32_t) {};
    execution.configureBGAIntegration(c, producer, output);
    execution.launch(views.views, 1, 1);
    runUntil(execution, [&] { return execution.hasFailed(); });
    EXPECT_FALSE(execution.bgaExecutionComplete());
    EXPECT_FALSE(execution.resultValid());
    EXPECT_NE(execution.errorMessage().find("producer done failure"),
              std::string::npos);
}

TEST(CSCM7A2CompletionDrainTest, OutputAcceptExceptionIsNotCommittedOrSuccessful) {
    auto layout = buildLayout(makeCSC(1, 0, {}), MappingPolicy::RoundRobin);
    Views views(layout, {});
    CSCNativeExecution execution;
    auto c = config(1, CSCBGAOutputConsumerMode::VALIDATION_ROUND_ROBIN);
    CSCBGAProducerCallbacks producer;
    producer.submit = [](const CSCBGAPartialBatch&) {
        return CSCBGAInputResult::ACCEPTED;
    };
    producer.producer_done = [](const CSCBGAProducerIdentity&) {};
    producer.request_final_drain = [](const CSCBGAProducerIdentity&) {
        return true;
    };
    static CSCBGAOutput output_value{0, 1.0F, 0, 1, 1, 1, 1,
                                     CSCBGAOutputReason::FINAL_DRAIN};
    CSCBGAOutputPortCallbacks output;
    output.has_output = [](uint32_t bg) { return bg == 0; };
    output.peek_output = [](uint32_t) -> const CSCBGAOutput& {
        return output_value;
    };
    output.accept_output = [](uint32_t) {
        throw std::runtime_error("injected output accept failure");
    };
    execution.configureBGAIntegration(c, producer, output);
    execution.launch(views.views, 1, 0);
    runUntil(execution, [&] { return execution.hasFailed(); });
    EXPECT_EQ(execution.bgaValidationCollector().accepted_outputs, 0U);
    EXPECT_EQ(execution.counters().bga_output_accept_errors, 1U);
    EXPECT_FALSE(execution.bgaExecutionComplete());
    EXPECT_NE(execution.errorMessage().find("output accept failure"),
              std::string::npos);
}

TEST(CSCM7A2CompletionDrainTest, ExactCompletionCycleOrdering) {
    auto layout = buildLayout(makeCSC(2, 1, {{0,0,2.0F},{1,0,4.0F}}),
                              MappingPolicy::External, {0});
    Views views(layout, {3.0F});
    CSCNativeExecution execution;
    execution.enableProductionBGAIntegration(
        config(2, CSCBGAOutputConsumerMode::EXTERNAL));
    execution.launch(views.views, 2, 2);
    uint64_t request_issue = 0, completion_poll = 0;
    uint64_t accepted = 0, producer_done = 0, drain_request = 0;
    uint64_t fifo_commit = 0, last_lookup_start = 0, last_lookup_complete = 0;
    uint64_t last_insert = 0;
    while (!execution.hasBGAOutput(0) && execution.cycle() < 200000) {
        execution.tick();
        const auto& engine = execution.engine(0);
        const auto& bga = execution.bankGroupAccumulator(0);
        if (!request_issue && engine.state() == CSCDescriptorState::WAIT_TARGET_GRANT)
            request_issue = execution.cycle();
        if (!completion_poll && engine.state() == CSCDescriptorState::EMIT_PARTIALS)
            completion_poll = execution.cycle();
        if (!accepted && engine.counters().accepted_bga_batches) accepted = execution.cycle();
        if (accepted && !fifo_commit && bga.inputQueueSize(0) == 2)
            fifo_commit = execution.cycle();
        if (!last_lookup_start && bga.inputQueueSize(0) == 0 &&
            bga.counters().lookup_misses == 1 && bga.counters().inserts == 1)
            last_lookup_start = execution.cycle();
        if (!last_lookup_complete && bga.counters().lookup_misses == 2)
            last_lookup_complete = execution.cycle();
        if (!producer_done && engine.producerDoneSent()) producer_done = execution.cycle();
        if (!drain_request && engine.finalDrainRequested()) drain_request = execution.cycle();
        if (!last_insert && execution.bankGroupAccumulator(0).counters().inserts == 2)
            last_insert = execution.cycle();
    }
    ASSERT_TRUE(execution.hasBGAOutput(0));
    const uint64_t first_output = execution.cycle();
    ASSERT_GT(accepted, 0U);
    EXPECT_EQ(producer_done, accepted + 3);
    EXPECT_EQ(drain_request, producer_done);
    EXPECT_GT(last_insert, drain_request);
    EXPECT_GT(first_output, last_insert);
    execution.acceptBGAOutput(0);
    EXPECT_FALSE(execution.bgaExecutionComplete());
    execution.tick();
    const uint64_t first_retirement = execution.cycle();
    ASSERT_TRUE(execution.hasBGAOutput(0));
    const uint64_t second_output = execution.cycle();
    execution.acceptBGAOutput(0);
    EXPECT_FALSE(execution.bgaExecutionComplete());
    execution.tick();
    const uint64_t last_retirement = execution.cycle();
    EXPECT_TRUE(execution.bgaExecutionComplete());
    EXPECT_EQ(execution.counters().bga_execution_complete_cycle,
              last_retirement);
    RecordProperty("last_target_request_issue_cycle", request_issue);
    RecordProperty("last_target_grant_cycle",
                   execution.engine(0).lastTargetGrantCycle());
    RecordProperty("last_target_completion_cycle",
                   execution.engine(0).lastTargetCompletionCycle());
    RecordProperty("engine_completion_poll_cycle", completion_poll);
    RecordProperty("last_batch_materialize_cycle", accepted);
    RecordProperty("last_batch_accepted_cycle", accepted);
    RecordProperty("accepted_pending_fifo_commit_cycle", fifo_commit);
    RecordProperty("last_lookup_start_cycle", last_lookup_start);
    RecordProperty("last_lookup_complete_cycle", last_lookup_complete);
    RecordProperty("producer_done_cycle", producer_done);
    RecordProperty("final_drain_request_cycle", drain_request);
    RecordProperty("last_insert_commit_cycle", last_insert);
    RecordProperty("first_final_output_cycle", first_output);
    RecordProperty("first_output_accept_cycle", first_output);
    RecordProperty("first_output_retirement_cycle", first_retirement);
    RecordProperty("next_final_output_cycle", second_output);
    RecordProperty("last_output_accept_cycle", second_output);
    RecordProperty("last_output_retirement_cycle", last_retirement);
    RecordProperty("final_drain_complete_cycle", last_retirement);
    RecordProperty("execution_complete_cycle", last_retirement);
}

TEST(CSCM7A2CompletionDrainTest, ExternalToyCompletesBGADrain) {
    const char* path = std::getenv("CSC_EXTERNAL_IMAGE");
    if (!path || !*path) GTEST_SKIP() << "set CSC_EXTERNAL_IMAGE";
    auto loaded = loadExternalPhysicalImage(path);
    std::vector<float> x(loaded.layout.matrix.cols);
    for (uint32_t c = 0; c < x.size(); ++c) x[c] = float(c + 1) / 7.0F;
    Views views(loaded.layout, x);
    CSCNativeExecution execution;
    execution.enableProductionBGAIntegration(config(
        loaded.layout.matrix.rows,
        CSCBGAOutputConsumerMode::VALIDATION_ROUND_ROBIN));
    execution.launch(views.views, loaded.layout.matrix.rows, loaded.layout.stats.nnz);
    runUntil(execution, [&] { return execution.bgaExecutionComplete(); }, 2000000);
    const auto counters = execution.counters();
    EXPECT_EQ(counters.accepted_bga_partials, loaded.layout.stats.nnz);
    EXPECT_EQ(counters.bga_output_contributions_accepted, loaded.layout.stats.nnz);
    EXPECT_EQ(execution.bgaValidationCollector().accepted_contributions,
              loaded.layout.stats.nnz);
    EXPECT_TRUE(execution.bgaValidationSemanticMatches());
    EXPECT_FALSE(execution.resultValid());
    RecordProperty("toy_nnz", loaded.layout.stats.nnz);
    RecordProperty("accepted_output_entries",
                   execution.bgaValidationCollector().accepted_outputs);
    RecordProperty("capacity_outputs",
                   execution.bgaValidationCollector().capacity_outputs);
    RecordProperty("final_drain_outputs",
                   execution.bgaValidationCollector().final_drain_outputs);
    RecordProperty("producer_done_calls", counters.bga_producer_done_calls);
    RecordProperty("empty_bg_completions",
                   counters.empty_bg_producer_completions);
    RecordProperty("execution_complete_cycle", execution.cycle());
}
