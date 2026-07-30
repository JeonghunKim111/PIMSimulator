#include "Callback.h"
#include "MemoryController.h"
#include "MultiChannelMemorySystem.h"
#include "Transaction.h"

#include "gtest/gtest.h"

#include <map>
#include <memory>
#include <vector>

namespace DRAMSim {

struct CSCM6PreparationTestAccess {
    static void addTokenizedPending(MemoryController& controller, const RequestToken& token,
                                    uint64_t address) {
        auto* transaction = new Transaction(DATA_READ, address, "test-pending", nullptr, token);
        transaction->timeAdded = 0;
        controller.addPendingRead(transaction);
        controller.parentMemorySystem->numOnTheFlyTransactions++;
    }

    static void addDuplicateTokenizedPending(MemoryController& controller,
                                             const RequestToken& token, uint64_t address) {
        auto* transaction = new Transaction(DATA_READ, address, "test-duplicate", nullptr, token);
        controller.addPendingRead(transaction);
    }

    static void addLegacyPending(MemoryController& controller, uint64_t address) {
        auto* transaction = new Transaction(DATA_READ, address, nullptr);
        transaction->timeAdded = 0;
        controller.pendingReadTransactions.push_back(transaction);
        controller.parentMemorySystem->numOnTheFlyTransactions++;
    }

    static void addReturn(MemoryController& controller, const RequestToken& token,
                          uint64_t address) {
        controller.returnTransaction.push_back(
            new Transaction(RETURN_DATA, address, "test-return", nullptr, token));
    }
};

}  // namespace DRAMSim

namespace {

using DRAMSim::CSCM6PreparationTestAccess;
using DRAMSim::MemoryController;
using DRAMSim::MultiChannelMemorySystem;
using DRAMSim::PendingReadLookupStats;
using DRAMSim::RequestKind;
using DRAMSim::RequestToken;

RequestToken token(uint64_t id, RequestKind kind = RequestKind::VALUE_READ,
                   uint32_t bg = 0, uint64_t sequence = 1) {
    RequestToken result;
    result.request_id = id;
    result.request_kind = kind;
    result.global_bg_id = bg;
    result.worker_id = bg;
    result.sequence_number = sequence;
    result.client_id = 7;
    result.generation = 3;
    return result;
}

std::shared_ptr<MultiChannelMemorySystem> memory(const char* trace) {
    return std::make_shared<MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_csc_fp32.ini", ".", trace,
        256 * 16);
}

MemoryController& controller(MultiChannelMemorySystem& memory_system, unsigned channel = 0) {
    return *memory_system.channels.at(channel)->memoryController;
}

struct CompletionSink {
    std::vector<RequestToken> tokens;
    std::vector<uint64_t> legacy_addresses;

    void tokenComplete(unsigned, const RequestToken& request_token, uint64_t) {
        tokens.push_back(request_token);
    }
    void legacyComplete(unsigned, uint64_t address, uint64_t) {
        legacy_addresses.push_back(address);
    }
};

struct RegisteredCallbacks {
    CompletionSink sink;
    DRAMSim::TokenCompleteCB* token_read = nullptr;
    DRAMSim::TransactionCompleteCB* legacy_read = nullptr;

    explicit RegisteredCallbacks(MultiChannelMemorySystem& memory_system) {
        token_read = new DRAMSim::Callback<CompletionSink, void, unsigned,
                                           const RequestToken&, uint64_t>(
            &sink, &CompletionSink::tokenComplete);
        legacy_read = new DRAMSim::Callback<CompletionSink, void, unsigned, uint64_t,
                                            uint64_t>(
            &sink, &CompletionSink::legacyComplete);
        memory_system.RegisterCallbacks(legacy_read, nullptr, nullptr);
        memory_system.RegisterTokenCallbacks(token_read, nullptr);
    }
    ~RegisteredCallbacks() {
        delete token_read;
        delete legacy_read;
    }
};

void drain(MultiChannelMemorySystem& memory_system, uint64_t limit = 200000) {
    uint64_t cycles = 0;
    while (memory_system.hasPendingTransactions() && cycles++ < limit)
        memory_system.update();
    ASSERT_LT(cycles, limit);
}

}  // namespace

TEST(CSCM6PreparationTest, TokenizedPendingReadUsesRequestIdLookup) {
    auto mem = memory("m6_prep_id_lookup");
    RegisteredCallbacks callbacks(*mem);
    DRAMSim::BurstType burst;
    const auto request = token(11);
    ASSERT_TRUE(mem->addTransaction(false, 0, "tokenized", &burst, request));
    drain(*mem);

    const auto& stats = controller(*mem).pendingReadLookupStats();
    EXPECT_EQ(stats.tokenized_map_lookup_count, 1);
    EXPECT_EQ(stats.tokenized_linear_lookup_count, 0);
    EXPECT_EQ(stats.legacy_linear_lookup_count, 0);
    ASSERT_EQ(callbacks.sink.tokens.size(), 1);
    EXPECT_TRUE(callbacks.sink.tokens[0] == request);
    EXPECT_EQ(controller(*mem).pendingTokenizedReadCount(), 0);
    EXPECT_EQ(controller(*mem).pendingLegacyReadCount(), 0);
}

TEST(CSCM6PreparationTest, SameAddressTokenizedReadsRemainIndependent) {
    auto mem = memory("m6_prep_same_address");
    RegisteredCallbacks callbacks(*mem);
    DRAMSim::BurstType bursts[4];
    std::vector<RequestToken> requests = {
        token(21, RequestKind::X_READ, 0, 1), token(22, RequestKind::VALUE_READ, 0, 2),
        token(23, RequestKind::INDEX_READ, 0, 3), token(24, RequestKind::VALUE_READ, 0, 4)};
    for (size_t i = 0; i < requests.size(); ++i)
        ASSERT_TRUE(mem->addTransaction(false, 0, "same-address", &bursts[i], requests[i]));
    drain(*mem);

    std::map<uint64_t, RequestToken> returned;
    for (const auto& completed : callbacks.sink.tokens) returned[completed.request_id] = completed;
    ASSERT_EQ(returned.size(), requests.size());
    for (const auto& request : requests) EXPECT_TRUE(returned.at(request.request_id) == request);
    const auto& stats = controller(*mem).pendingReadLookupStats();
    EXPECT_EQ(stats.tokenized_map_lookup_count, requests.size());
    EXPECT_EQ(stats.tokenized_linear_lookup_count, 0);
    EXPECT_EQ(controller(*mem).pendingTokenizedReadCount(), 0);
}

TEST(CSCM6PreparationTest, TokenizedAndLegacyReadsCanShareAddress) {
    auto mem = memory("m6_prep_mixed_address");
    RegisteredCallbacks callbacks(*mem);
    DRAMSim::BurstType bursts[2];
    const auto request = token(31);
    ASSERT_TRUE(mem->addTransaction(false, 0, "tokenized", &bursts[0], request));
    ASSERT_TRUE(mem->addTransaction(false, 0, "legacy", &bursts[1]));
    drain(*mem);

    ASSERT_EQ(callbacks.sink.tokens.size(), 1);
    EXPECT_TRUE(callbacks.sink.tokens[0] == request);
    EXPECT_EQ(callbacks.sink.legacy_addresses.size(), 2);
    const auto& stats = controller(*mem).pendingReadLookupStats();
    EXPECT_EQ(stats.tokenized_map_lookup_count, 1);
    EXPECT_EQ(stats.tokenized_linear_lookup_count, 0);
    EXPECT_EQ(stats.legacy_linear_lookup_count, 1);
    EXPECT_EQ(controller(*mem).pendingTokenizedReadCount(), 0);
    EXPECT_EQ(controller(*mem).pendingLegacyReadCount(), 0);
}

TEST(CSCM6PreparationTest, DuplicateRequestIdInsertionIsRejected) {
    EXPECT_DEATH(
        {
            auto mem = memory("m6_prep_duplicate_id");
            const auto duplicate = token(41);
            CSCM6PreparationTestAccess::addTokenizedPending(controller(*mem), duplicate, 0);
            CSCM6PreparationTestAccess::addDuplicateTokenizedPending(
                controller(*mem), duplicate, 32);
        },
        "Duplicate tokenized pending read request ID");
}

TEST(CSCM6PreparationTest, UnknownReturnedRequestIdIsDetected) {
    EXPECT_DEATH(
        {
            auto mem = memory("m6_prep_unknown_id");
            CSCM6PreparationTestAccess::addReturn(controller(*mem), token(51), 0);
            controller(*mem).update();
        },
        "Unknown tokenized pending read request ID");
}

TEST(CSCM6PreparationTest, FullTokenMismatchIsDetected) {
    EXPECT_DEATH(
        {
            auto mem = memory("m6_prep_token_mismatch");
            auto accepted = token(61, RequestKind::VALUE_READ, 0, 1);
            auto returned = accepted;
            returned.request_kind = RequestKind::INDEX_READ;
            CSCM6PreparationTestAccess::addTokenizedPending(controller(*mem), accepted, 0);
            CSCM6PreparationTestAccess::addReturn(controller(*mem), returned, 0);
            controller(*mem).update();
        },
        "full-token mismatch");
}

TEST(CSCM6PreparationTest, OutOfOrderCompletionRetiresExactlyOnce) {
    auto mem = memory("m6_prep_forced_reverse");
    RegisteredCallbacks callbacks(*mem);
    std::vector<RequestToken> issue_order = {
        token(71, RequestKind::X_READ, 0, 1), token(72, RequestKind::VALUE_READ, 0, 2),
        token(73, RequestKind::INDEX_READ, 0, 3)};
    for (const auto& request : issue_order)
        CSCM6PreparationTestAccess::addTokenizedPending(controller(*mem), request, 0);
    for (auto it = issue_order.rbegin(); it != issue_order.rend(); ++it)
        CSCM6PreparationTestAccess::addReturn(controller(*mem), *it, 0);
    for (size_t i = 0; i < issue_order.size(); ++i) controller(*mem).update();

    ASSERT_EQ(callbacks.sink.tokens.size(), issue_order.size());
    EXPECT_EQ(callbacks.sink.tokens.front().request_id, issue_order.back().request_id);
    EXPECT_EQ(callbacks.sink.tokens.back().request_id, issue_order.front().request_id);
    EXPECT_EQ(controller(*mem).pendingTokenizedReadCount(), 0);
    EXPECT_EQ(mem->hasPendingTransactions(), 0);
    const auto& stats = controller(*mem).pendingReadLookupStats();
    EXPECT_EQ(stats.tokenized_map_lookup_count, issue_order.size());
    EXPECT_EQ(stats.tokenized_linear_lookup_count, 0);
    EXPECT_EQ(stats.unknown_request_id_count, 0);
}

TEST(CSCM6PreparationTest, ControllerCleanupLeavesNoIndexedPendingReads) {
    {
        auto mem = memory("m6_prep_cleanup");
        CSCM6PreparationTestAccess::addTokenizedPending(controller(*mem), token(81), 0);
        CSCM6PreparationTestAccess::addLegacyPending(controller(*mem), 32);
        EXPECT_EQ(controller(*mem).pendingTokenizedReadCount(), 1);
        EXPECT_EQ(controller(*mem).pendingLegacyReadCount(), 1);
    }
    SUCCEED();
}

TEST(CSCM6PreparationTest, LookupCountIsIndependentOfPendingDepth) {
    for (uint64_t depth : {1ULL, 16ULL, 64ULL, 256ULL}) {
        auto mem = memory("m6_prep_lookup_depth");
        RegisteredCallbacks callbacks(*mem);
        for (uint64_t index = 0; index < depth; ++index)
            CSCM6PreparationTestAccess::addTokenizedPending(
                controller(*mem), token(1000 + index, RequestKind::VALUE_READ, 0, index), 0);
        for (uint64_t index = 0; index < depth; ++index)
            CSCM6PreparationTestAccess::addReturn(
                controller(*mem), token(1000 + index, RequestKind::VALUE_READ, 0, index), 0);
        for (uint64_t index = 0; index < depth; ++index) controller(*mem).update();

        const PendingReadLookupStats& stats = controller(*mem).pendingReadLookupStats();
        EXPECT_EQ(stats.tokenized_map_lookup_count, depth) << "depth=" << depth;
        EXPECT_EQ(stats.tokenized_linear_lookup_count, 0) << "depth=" << depth;
        EXPECT_EQ(stats.legacy_linear_lookup_count, 0) << "depth=" << depth;
        EXPECT_EQ(callbacks.sink.tokens.size(), depth) << "depth=" << depth;
        EXPECT_EQ(controller(*mem).pendingTokenizedReadCount(), 0) << "depth=" << depth;
        EXPECT_EQ(mem->hasPendingTransactions(), 0) << "depth=" << depth;
    }
}
