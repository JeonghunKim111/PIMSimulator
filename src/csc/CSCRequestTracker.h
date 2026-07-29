#ifndef CSC_REQUEST_TRACKER_H
#define CSC_REQUEST_TRACKER_H

#include "RequestToken.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

namespace csc_descriptor {

using CSCRequestKind = DRAMSim::RequestKind;
constexpr std::size_t kCSCRequestKindCount = 6;
constexpr std::size_t kCSCGlobalBGCount = 64;

enum class CSCRequestState {
    CREATED,
    WAITING_TO_SUBMIT,
    ACCEPTED,
    COMPLETED,
    RETIRED,
    ABANDONED
};

struct CSCRequestEntry {
    DRAMSim::RequestToken token{};
    uint64_t address = 0;
    unsigned channel = 0;
    CSCRequestState state = CSCRequestState::CREATED;
    uint64_t first_attempt_cycle = 0;
    uint64_t accepted_cycle = 0;
    uint64_t completion_cycle = 0;
    uint64_t abandoned_cycle = 0;
    uint64_t submit_attempts = 0;
};

struct CSCRequestStats {
    uint64_t created_requests = 0;
    uint64_t submit_attempts = 0;
    uint64_t issued = 0;
    uint64_t completed = 0;
    uint64_t submit_rejected = 0;
    uint64_t retries = 0;
    uint64_t abandoned_waiting_requests = 0;
    uint64_t unknown_completions = 0;
    uint64_t duplicate_completions = 0;
    uint64_t stale_generation_completions = 0;
    uint64_t kind_mismatches = 0;
    uint64_t owner_mismatches = 0;
    uint64_t completion_after_retirement = 0;
    uint64_t maximum_waiting_to_submit_depth = 0;
    std::array<uint64_t, kCSCRequestKindCount> issued_by_kind{};
    std::array<uint64_t, kCSCRequestKindCount> completed_by_kind{};
    std::array<uint64_t, kCSCRequestKindCount> abandoned_by_kind{};
    std::array<uint64_t, kCSCGlobalBGCount> abandoned_by_bg{};
};

class CSCRequestTracker {
  public:
    explicit CSCRequestTracker(uint32_t generation = 1) : generation_(generation) {}

    bool create(const DRAMSim::RequestToken&, uint64_t address, unsigned channel, uint64_t cycle);
    bool recordAttempt(uint64_t request_id, uint64_t cycle);
    bool recordAccepted(uint64_t request_id, unsigned channel, uint64_t cycle);
    bool recordRejected(uint64_t request_id);
    bool complete(unsigned channel, const DRAMSim::RequestToken&, uint64_t cycle);
    bool retire(uint64_t request_id);
    bool abandon(uint64_t request_id, uint64_t cycle);
    uint64_t abandonWaiting(uint64_t cycle);

    const CSCRequestEntry* find(uint64_t request_id) const;
    std::size_t pendingCount() const;
    std::size_t acceptedCount() const;
    std::size_t waitingCount() const;
    bool empty() const { return pendingCount() == 0; }
    bool terminalAccountingValid() const;
    void reset(uint32_t generation = 0);

    const CSCRequestStats& stats() const { return stats_; }
    const std::string& error() const { return error_; }
    uint32_t generation() const { return generation_; }

    // M4 serialized-engine compatibility helpers.
    bool pending() const { return acceptedCount() != 0; }
    bool completionReady() const { return legacy_ready_; }
    CSCRequestKind kind() const { return legacy_kind_; }
    uint64_t address() const { return legacy_address_; }
    void recordIssue(CSCRequestKind, uint64_t, unsigned, uint32_t);
    bool complete(unsigned, uint64_t, uint64_t);
    void consumeCompletion();
    void recordSubmitRejected() { stats_.submit_rejected++; }

  private:
    void fail(const std::string& message) {
        if (error_.empty()) error_ = message;
    }

    std::map<uint64_t, CSCRequestEntry> entries_;
    std::map<uint64_t, CSCRequestEntry> history_;
    uint32_t generation_ = 1;
    CSCRequestStats stats_;
    std::string error_;
    uint64_t legacy_next_id_ = 1;
    uint64_t legacy_id_ = 0;
    uint64_t legacy_address_ = 0;
    CSCRequestKind legacy_kind_ = CSCRequestKind::X_READ;
    bool legacy_ready_ = false;
};

}  // namespace csc_descriptor
#endif
