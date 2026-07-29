#include "csc/CSCRequestTracker.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace csc_descriptor {
namespace {
std::size_t kindIndex(DRAMSim::RequestKind kind) {
    return static_cast<std::size_t>(kind);
}
}  // namespace

bool CSCRequestTracker::create(const DRAMSim::RequestToken& token, uint64_t address,
                               unsigned channel, uint64_t cycle) {
    if (!token.valid() || token.generation != generation_ ||
        entries_.count(token.request_id) || history_.count(token.request_id)) {
        fail("invalid or duplicate request id");
        return false;
    }
    if (kindIndex(token.request_kind) >= kCSCRequestKindCount ||
        token.global_bg_id >= kCSCGlobalBGCount) {
        fail("request token kind or BG out of range");
        return false;
    }
    CSCRequestEntry entry;
    entry.token = token;
    entry.address = address;
    entry.channel = channel;
    entry.state = CSCRequestState::WAITING_TO_SUBMIT;
    entry.first_attempt_cycle = cycle;
    entries_.emplace(token.request_id, entry);
    stats_.created_requests++;
    stats_.maximum_waiting_to_submit_depth =
        std::max<uint64_t>(stats_.maximum_waiting_to_submit_depth, waitingCount());
    return true;
}

bool CSCRequestTracker::recordAttempt(uint64_t id, uint64_t cycle) {
    auto it = entries_.find(id);
    if (it == entries_.end() || it->second.state != CSCRequestState::WAITING_TO_SUBMIT)
        return false;
    if (!it->second.submit_attempts)
        it->second.first_attempt_cycle = cycle;
    else
        stats_.retries++;
    it->second.submit_attempts++;
    stats_.submit_attempts++;
    return true;
}

bool CSCRequestTracker::recordAccepted(uint64_t id, unsigned channel, uint64_t cycle) {
    auto it = entries_.find(id);
    if (it == entries_.end() || it->second.state != CSCRequestState::WAITING_TO_SUBMIT)
        return false;
    it->second.channel = channel;
    it->second.accepted_cycle = cycle;
    it->second.state = CSCRequestState::ACCEPTED;
    stats_.issued++;
    stats_.issued_by_kind[kindIndex(it->second.token.request_kind)]++;
    return true;
}

bool CSCRequestTracker::recordRejected(uint64_t id) {
    auto it = entries_.find(id);
    if (it == entries_.end() || it->second.state != CSCRequestState::WAITING_TO_SUBMIT)
        return false;
    stats_.submit_rejected++;
    return true;
}

bool CSCRequestTracker::complete(unsigned channel, const DRAMSim::RequestToken& token,
                                 uint64_t cycle) {
    auto it = entries_.find(token.request_id);
    if (it == entries_.end()) {
        auto history = history_.find(token.request_id);
        if (history != history_.end()) {
            stats_.completion_after_retirement++;
            fail(history->second.state == CSCRequestState::ABANDONED
                     ? "completion after abandonment"
                     : "completion after retirement");
        } else {
            stats_.unknown_completions++;
            fail("unknown completion");
        }
        return false;
    }
    auto& entry = it->second;
    if (entry.state == CSCRequestState::COMPLETED) {
        stats_.duplicate_completions++;
        fail("duplicate completion");
        return false;
    }
    if (token.generation != generation_) {
        stats_.stale_generation_completions++;
        fail("stale generation completion");
        return false;
    }
    if (!(entry.token == token)) {
        if (entry.token.request_kind != token.request_kind)
            stats_.kind_mismatches++;
        else
            stats_.owner_mismatches++;
        fail("completion token mismatch");
        return false;
    }
    if (entry.state != CSCRequestState::ACCEPTED || entry.channel != channel) {
        fail("completion before acceptance or wrong channel");
        return false;
    }
    entry.completion_cycle = cycle;
    entry.state = CSCRequestState::COMPLETED;
    stats_.completed++;
    stats_.completed_by_kind[kindIndex(token.request_kind)]++;
    return true;
}

bool CSCRequestTracker::retire(uint64_t id) {
    auto it = entries_.find(id);
    if (it == entries_.end() || it->second.state != CSCRequestState::COMPLETED) return false;
    it->second.state = CSCRequestState::RETIRED;
    history_[id] = it->second;
    entries_.erase(it);
    return true;
}

bool CSCRequestTracker::abandon(uint64_t id, uint64_t cycle) {
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        auto old = history_.find(id);
        if (old != history_.end() && old->second.state == CSCRequestState::ABANDONED)
            fail("duplicate request abandonment");
        else
            fail("abandon unknown request");
        return false;
    }
    if (it->second.state != CSCRequestState::CREATED &&
        it->second.state != CSCRequestState::WAITING_TO_SUBMIT) {
        fail("only unaccepted request may be abandoned");
        return false;
    }
    auto& entry = it->second;
    entry.state = CSCRequestState::ABANDONED;
    entry.abandoned_cycle = cycle;
    stats_.abandoned_waiting_requests++;
    stats_.abandoned_by_kind[kindIndex(entry.token.request_kind)]++;
    stats_.abandoned_by_bg[entry.token.global_bg_id]++;
    history_[id] = entry;
    entries_.erase(it);
    return true;
}

uint64_t CSCRequestTracker::abandonWaiting(uint64_t cycle) {
    std::vector<uint64_t> ids;
    for (const auto& pair : entries_) {
        if (pair.second.state == CSCRequestState::CREATED ||
            pair.second.state == CSCRequestState::WAITING_TO_SUBMIT)
            ids.push_back(pair.first);
    }
    uint64_t abandoned = 0;
    for (uint64_t id : ids)
        if (abandon(id, cycle)) abandoned++;
    return abandoned;
}

const CSCRequestEntry* CSCRequestTracker::find(uint64_t id) const {
    auto active = entries_.find(id);
    if (active != entries_.end()) return &active->second;
    auto history = history_.find(id);
    return history == history_.end() ? nullptr : &history->second;
}

std::size_t CSCRequestTracker::pendingCount() const { return entries_.size(); }

std::size_t CSCRequestTracker::acceptedCount() const {
    std::size_t count = 0;
    for (const auto& pair : entries_)
        count += pair.second.state == CSCRequestState::ACCEPTED;
    return count;
}

std::size_t CSCRequestTracker::waitingCount() const {
    std::size_t count = 0;
    for (const auto& pair : entries_)
        count += pair.second.state == CSCRequestState::CREATED ||
                 pair.second.state == CSCRequestState::WAITING_TO_SUBMIT;
    return count;
}

bool CSCRequestTracker::terminalAccountingValid() const {
    const uint64_t abandoned_kind_sum =
        std::accumulate(stats_.abandoned_by_kind.begin(), stats_.abandoned_by_kind.end(), 0ULL);
    const uint64_t abandoned_bg_sum =
        std::accumulate(stats_.abandoned_by_bg.begin(), stats_.abandoned_by_bg.end(), 0ULL);
    return stats_.created_requests == stats_.issued + stats_.abandoned_waiting_requests &&
           stats_.issued == stats_.completed &&
           stats_.abandoned_waiting_requests == abandoned_kind_sum &&
           stats_.abandoned_waiting_requests == abandoned_bg_sum;
}

void CSCRequestTracker::reset(uint32_t generation) {
    if (acceptedCount()) throw std::logic_error("reset with accepted request");
    entries_.clear();
    history_.clear();
    legacy_ready_ = false;
    legacy_id_ = 0;
    error_.clear();
    stats_ = {};
    if (generation) generation_ = generation;
}

void CSCRequestTracker::recordIssue(CSCRequestKind kind, uint64_t address, unsigned channel,
                                    uint32_t bg) {
    DRAMSim::RequestToken token;
    token.request_id = legacy_next_id_++;
    token.request_kind = kind;
    token.global_bg_id = bg;
    token.worker_id = bg;
    token.sequence_number = token.request_id;
    token.generation = generation_;
    if (!create(token, address, channel, 0) ||
        !recordAttempt(token.request_id, 0) ||
        !recordAccepted(token.request_id, channel, 0))
        throw std::logic_error(error_);
    legacy_id_ = token.request_id;
    legacy_kind_ = kind;
    legacy_address_ = address;
}

bool CSCRequestTracker::complete(unsigned channel, uint64_t address, uint64_t cycle) {
    auto it = entries_.find(legacy_id_);
    if (it == entries_.end() || it->second.address != address) return false;
    if (!complete(channel, it->second.token, cycle)) return false;
    legacy_ready_ = true;
    return true;
}

void CSCRequestTracker::consumeCompletion() {
    if (!legacy_ready_ || !retire(legacy_id_))
        throw std::logic_error("CSC completion not ready");
    legacy_ready_ = false;
    legacy_id_ = 0;
}

}  // namespace csc_descriptor
