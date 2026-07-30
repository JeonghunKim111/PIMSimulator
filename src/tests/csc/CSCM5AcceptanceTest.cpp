#include "Callback.h"
#include "MultiChannelMemorySystem.h"
#include "SystemConfiguration.h"
#include "csc/CSCDescriptorEngine.h"
#include "tests/KernelAddrGen.h"
#include "tests/csc/CSCFunctionalModel.h"
#include "tests/csc/CSCLayout.h"

#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

using namespace csc_descriptor;

namespace {

struct AcceptanceViews {
    std::array<std::vector<float>, 64> packed_x;
    std::array<CSCBGImageView, 64> views;

    AcceptanceViews(const CSCLayout& layout, const std::vector<float>& x) {
        for (uint32_t bg = 0; bg < 64; ++bg) {
            for (uint32_t column : layout.bg[bg].x_slot_to_original_col)
                packed_x[bg].push_back(x[column]);
            views[bg] = {&layout.bg[bg].values, &layout.bg[bg].row_indices,
                         &layout.bg[bg].descriptors, &packed_x[bg]};
        }
    }
};

std::vector<float> accumulatePartials(uint32_t rows,
                                      const std::vector<CSCPartial>& partials) {
    std::vector<double> sums(rows);
    for (const auto& partial : partials) {
        EXPECT_LT(partial.row_idx, rows);
        if (partial.row_idx < rows) sums[partial.row_idx] += partial.value;
    }
    std::vector<float> result(rows);
    for (uint32_t row = 0; row < rows; ++row) result[row] = float(sums[row]);
    return result;
}

DRAMSim::RequestToken fillerToken(uint64_t id) {
    DRAMSim::RequestToken token;
    token.request_id = id;
    token.request_kind = DRAMSim::RequestKind::VALUE_READ;
    token.global_bg_id = 0;
    token.worker_id = 0;
    token.sequence_number = id;
    token.client_id = 99;
    token.generation = 1;
    return token;
}

struct QueueCompletionSink {
    std::map<uint64_t, CSCDescriptorEngine*> owners;
    std::vector<uint64_t> completed;
    uint64_t filler_completions = 0;

    void complete(unsigned channel, const DRAMSim::RequestToken& token, uint64_t cycle) {
        auto owner = owners.find(token.request_id);
        if (owner == owners.end()) {
            filler_completions++;
            return;
        }
        EXPECT_TRUE(owner->second->onRequestComplete(channel, token, cycle));
        completed.push_back(token.request_id);
        owners.erase(owner);
    }
};

}  // namespace

namespace csc_descriptor {

struct CSCAcceptanceTestAccess {
    static void installControlledSameAddressSubmit(
        CSCNativeExecution& execution, uint32_t bg, uint64_t address,
        std::vector<DRAMSim::RequestToken>* issued) {
        auto* engine = execution.engines_.at(bg).get();
        engine->address_ = [address](uint32_t, CSCRequestKind, uint64_t) {
            return address;
        };
        engine->token_submit_ =
            [&execution, issued](CSCDescriptorEngine* owner,
                                 const DRAMSim::RequestToken& token, uint64_t request_address) {
                issued->push_back(token);
                auto inserted = execution.outstanding_.emplace(token.request_id, owner);
                if (!inserted.second) return false;
                execution.outstanding_addresses_[token.request_id] = request_address;
                return true;
            };
    }

    static void forceCompletion(CSCNativeExecution& execution, unsigned channel,
                                const DRAMSim::RequestToken& token, uint64_t cycle) {
        execution.tokenComplete(channel, token, cycle);
    }
};

}  // namespace csc_descriptor

TEST(CSCM5AcceptanceTest, ForcedOutOfOrderCompletionPreservesIdentity) {
    auto matrix = makeCSC(3, 1, {{0, 0, 2.0f}, {1, 0, -3.0f}, {2, 0, 4.0f}});
    auto layout = buildLayout(matrix, MappingPolicy::RoundRobin);
    const std::vector<float> x{5.0f};
    AcceptanceViews views(layout, x);
    CSCNativeExecution execution;
    std::vector<DRAMSim::RequestToken> issue_order;

    CSCAcceptanceTestAccess::installControlledSameAddressSubmit(execution, 0, 0,
                                                                &issue_order);
    execution.launch(views.views, matrix.rows, matrix.values.size());

    uint64_t guard = 0;
    while (issue_order.size() < 3 && guard++ < 100)
        execution.tick();
    ASSERT_LT(guard, 100);
    ASSERT_EQ(issue_order.size(), 3);
    EXPECT_EQ(issue_order[0].request_kind, DRAMSim::RequestKind::X_READ);
    EXPECT_EQ(issue_order[1].request_kind, DRAMSim::RequestKind::VALUE_READ);
    EXPECT_EQ(issue_order[2].request_kind, DRAMSim::RequestKind::INDEX_READ);
    EXPECT_NE(issue_order[0].request_id, issue_order[1].request_id);
    EXPECT_NE(issue_order[1].request_id, issue_order[2].request_id);
    EXPECT_EQ(execution.counters().maximum_global_outstanding, 3);

    std::vector<uint64_t> completion_order;
    for (auto it = issue_order.rbegin(); it != issue_order.rend(); ++it) {
        completion_order.push_back(it->request_id);
        CSCAcceptanceTestAccess::forceCompletion(execution, 0, *it, ++guard);
    }

    ASSERT_EQ(completion_order.size(), issue_order.size());
    EXPECT_EQ(completion_order.front(), issue_order.back().request_id);
    EXPECT_EQ(completion_order.back(), issue_order.front().request_id);
    EXPECT_NE(completion_order, std::vector<uint64_t>(
                                    {issue_order[0].request_id, issue_order[1].request_id,
                                     issue_order[2].request_id}));

    while (!execution.done() && guard++ < 1000)
        execution.tick();
    ASSERT_LT(guard, 1000);
    ASSERT_TRUE(execution.isDone());
    EXPECT_FALSE(execution.hasOutstandingRequest());
    EXPECT_FALSE(execution.hasPendingTransactions());
    EXPECT_FALSE(execution.hasUnconsumedCompletion());
    EXPECT_EQ(execution.hostAccumulate(), cpuReference(matrix, x));

    const auto counters = execution.counters();
    EXPECT_EQ(counters.accepted_requests, 3);
    EXPECT_EQ(counters.completion_count, 3);
    EXPECT_EQ(counters.created_requests, counters.accepted_requests);
    EXPECT_EQ(counters.unknown_completions, 0);
    EXPECT_EQ(counters.duplicate_completions, 0);
    EXPECT_EQ(counters.stale_generation_completions, 0);
    for (const auto& token : issue_order) {
        const auto* entry = execution.engine(0).tracker().find(token.request_id);
        ASSERT_NE(entry, nullptr);
        EXPECT_EQ(entry->state, CSCRequestState::RETIRED);
    }
}

TEST(CSCM5AcceptanceTest, RealQueueBackpressureAcrossMultipleBGs) {
    auto memory = std::make_shared<DRAMSim::MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_csc_fp32.ini", ".",
        "m5_real_queue_acceptance", 256 * 16);
    QueueCompletionSink sink;
    auto* read_callback =
        new DRAMSim::Callback<QueueCompletionSink, void, unsigned,
                              const DRAMSim::RequestToken&, uint64_t>(
            &sink, &QueueCompletionSink::complete);
    memory->RegisterTokenCallbacks(read_callback, nullptr);

    DRAMSim::BurstType burst;
    const uint64_t queue_depth = getConfigParam(UINT, "TRANS_QUEUE_DEPTH");
    ASSERT_GT(queue_depth, 0);
    for (uint64_t id = 0; id < queue_depth; ++id) {
        ASSERT_TRUE(memory->addTransaction(false, 0, "queue-filler", &burst,
                                           fillerToken(1000000 + id)));
    }
    EXPECT_FALSE(memory->addTransaction(false, 0, "queue-full-proof", &burst,
                                        fillerToken(2000000)));

    auto matrix = makeCSC(2, 2, {{0, 0, 2.0f}, {1, 1, -3.0f}});
    auto layout = buildLayout(matrix, MappingPolicy::External, {0, 17});
    const std::vector<float> x{4.0f, 5.0f};
    AcceptanceViews views(layout, x);
    PIMAddrManager addresses(16, 1);
    std::vector<CSCPartial> partials;
    std::vector<std::unique_ptr<DRAMSim::PIMBlock>> blocks;
    std::vector<std::unique_ptr<CSCDescriptorEngine>> engines;
    std::map<uint32_t, std::vector<DRAMSim::RequestToken>> attempts;
    uint64_t next_request_id = 1;

    auto address_for = [&addresses](uint32_t bg, CSCRequestKind kind, uint64_t offset) {
        const uint32_t bank = kind == CSCRequestKind::X_READ
                                  ? 2
                                  : (kind == CSCRequestKind::VALUE_READ ? 0 : 1);
        unsigned row = kind == CSCRequestKind::X_READ
                           ? 8192
                           : (kind == CSCRequestKind::VALUE_READ ? 0 : 4096);
        unsigned column = offset / 32;
        return addresses.addrGenSafe(bg / 4, 0, bg % 4, bank, row, column);
    };
    auto token_factory = [&next_request_id](uint32_t bg, CSCRequestKind kind,
                                            uint64_t sequence) {
        DRAMSim::RequestToken token;
        token.request_id = next_request_id++;
        token.request_kind = kind;
        token.global_bg_id = bg;
        token.worker_id = bg;
        token.sequence_number = sequence;
        token.client_id = 1;
        token.generation = 1;
        return token;
    };

    for (uint32_t bg : {0u, 17u}) {
        blocks.emplace_back(new DRAMSim::PIMBlock(FP32));
        engines.emplace_back(new CSCDescriptorEngine(
            bg, blocks.back().get(),
            [&memory, &sink, &burst, &attempts](
                CSCDescriptorEngine* owner, const DRAMSim::RequestToken& token,
                uint64_t address) {
                attempts[token.global_bg_id].push_back(token);
                if (!memory->addTransaction(false, address, "CSC_REAL_QUEUE", &burst,
                                            token))
                    return false;
                sink.owners[token.request_id] = owner;
                return true;
            },
            token_factory, address_for));
        engines.back()->launch(views.views[bg], &partials);
    }

    engines[0]->tick();
    engines[1]->tick();
    engines[0]->tick();
    engines[1]->tick();
    ASSERT_FALSE(attempts[0].empty());
    ASSERT_FALSE(attempts[17].empty());
    EXPECT_EQ(engines[0]->tracker().stats().issued, 0);
    EXPECT_EQ(engines[0]->tracker().stats().submit_rejected, 1);
    EXPECT_EQ(engines[1]->tracker().stats().issued, 1);
    const auto rejected_token = attempts[0].front();

    uint64_t guard = 0;
    while ((!engines[0]->isDone() || !engines[1]->isDone() ||
            memory->hasPendingTransactions() || !sink.owners.empty()) &&
           guard++ < 200000) {
        memory->update();
        engines[0]->tick();
        engines[1]->tick();
    }
    ASSERT_LT(guard, 200000);
    ASSERT_GE(attempts[0].size(), 2);
    EXPECT_TRUE(attempts[0][1] == rejected_token);
    EXPECT_EQ(attempts[0][1].sequence_number, rejected_token.sequence_number);
    EXPECT_GE(engines[0]->tracker().stats().retries, 1);
    EXPECT_GE(engines[0]->tracker().stats().submit_rejected, 1);
    EXPECT_TRUE(engines[0]->isDone());
    EXPECT_TRUE(engines[1]->isDone());
    EXPECT_TRUE(sink.owners.empty());
    EXPECT_EQ(memory->hasPendingTransactions(), 0);
    EXPECT_EQ(engines[0]->tracker().pendingCount(), 0);
    EXPECT_EQ(engines[1]->tracker().pendingCount(), 0);
    EXPECT_EQ(engines[0]->tracker().stats().issued,
              engines[0]->tracker().stats().completed);
    EXPECT_EQ(engines[1]->tracker().stats().issued,
              engines[1]->tracker().stats().completed);
    EXPECT_EQ(sink.completed.size(), 6);
    EXPECT_EQ(accumulatePartials(matrix.rows, partials),
              cpuReference(matrix, x));

    for (const auto& engine : engines) {
        const auto& stats = engine->tracker().stats();
        EXPECT_EQ(stats.unknown_completions, 0);
        EXPECT_EQ(stats.duplicate_completions, 0);
        EXPECT_EQ(stats.stale_generation_completions, 0);
        EXPECT_TRUE(engine->tracker().terminalAccountingValid());
    }

    delete read_callback;
}
