#include "csc/CSCPartialResultPath.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace csc_descriptor {

void CSCPartialResultPathConfig::validate() const
{
    if (!global_bg_count || !bank_groups_per_rank ||
        global_bg_count % bank_groups_per_rank ||
        !buffer_capacity_bursts_per_bg ||
        !pending_capacity_bursts_per_bg ||
        !max_inflight_writes_per_bg ||
        !write_issue_limit_per_rank_per_cycle)
        throw std::invalid_argument("invalid partial-result writeback capacity");
}

CSCPartialResultPath::CSCPartialResultPath(
    const CSCPartialResultPathConfig& config, uint32_t rows,
    uint32_t generation)
    : config_(config), rows_(rows), generation_(generation)
{
    config_.validate();
    if (!rows_ || !generation_)
        throw std::invalid_argument("invalid partial-result path identity");
    bg_.resize(config_.global_bg_count);
    rank_rr_cursor_.resize(
        config_.global_bg_count / config_.bank_groups_per_rank);
    counters_.peak_inflight_writes_by_bg.resize(config_.global_bg_count);
    for (auto& state : bg_) {
        state.resident.resize(config_.buffer_capacity_bursts_per_bg);
        state.buffer_reserved.resize(
            config_.buffer_capacity_bursts_per_bg, false);
    }
    write_trace_capacity_ =
        uint64_t(config_.global_bg_count) *
        (uint64_t(config_.buffer_capacity_bursts_per_bg) +
         config_.pending_capacity_bursts_per_bg);
    write_trace_.reserve(write_trace_capacity_);
}

void CSCPartialResultPath::fail(const std::string& message) noexcept
{
    if (error_) return;
    error_ = true;
    error_message_ = message;
}

bool CSCPartialResultPath::validateReservation(
    const CSCBGAOutputPortValue& value)
{
    const auto& output = value.payload;
    if (value.global_bg_id >= config_.global_bg_count) {
        fail("partial-result global BG out of range");
        return false;
    }
    if (output.row_idx >= rows_) {
        fail("partial-result row out of range");
        return false;
    }
    if (!output.generation || output.generation != generation_) {
        fail("partial-result generation mismatch");
        return false;
    }
    if (!output.output_sequence || !output.contribution_count) {
        fail("invalid partial-result identity");
        return false;
    }
    const auto& state = bg_[value.global_bg_id];
    if (state.last_output_sequence == std::numeric_limits<uint64_t>::max() ||
        output.output_sequence != state.last_output_sequence + 1) {
        fail("duplicate or nonmonotonic partial-result output sequence");
        return false;
    }
    if (output.reason != CSCBGAOutputReason::CAPACITY_EVICTION &&
        output.reason != CSCBGAOutputReason::FINAL_DRAIN) {
        fail("invalid partial-result output reason");
        return false;
    }
    return true;
}

bool CSCPartialResultPath::reserve(const CSCBGAOutputPortValue& value)
{
    if (error_ || reservation_) {
        if (reservation_) fail("partial-result reservation mismatch");
        return false;
    }
    if (!validateReservation(value)) return false;
    const auto& state = bg_[value.global_bg_id];
    if (state.packer_count >= kCSCPartialRecordsPerBurst) {
        fail("partial-result packer overflow");
        return false;
    }
    if (state.packer_count == kCSCPartialRecordsPerBurst - 1 &&
        state.pending.size() >= config_.pending_capacity_bursts_per_bg) {
        ++counters_.packer_backpressure_cycles;
        ++counters_.writeback_backpressure_cycles;
        return false;
    }
    reservation_ = value;
    return true;
}

void CSCPartialResultPath::commitReserved() noexcept
{
    if (!reservation_) {
        fail("partial-result commit without reservation");
        return;
    }
    const auto value = *reservation_;
    reservation_.reset();
    auto& state = bg_[value.global_bg_id];
    if (state.packer_count >= kCSCPartialRecordsPerBurst) {
        fail("partial-result packer commit overflow");
        return;
    }
    CSCPartialResultRecordEnvelope envelope;
    envelope.record.row_idx = value.payload.row_idx;
    envelope.record.value = value.payload.value;
    envelope.global_bg_id = value.global_bg_id;
    envelope.generation = value.payload.generation;
    envelope.output_sequence = value.payload.output_sequence;
    envelope.contribution_count = value.payload.contribution_count;
    envelope.reason = value.payload.reason;
    state.packer[state.packer_count++] = envelope;
    state.last_output_sequence = envelope.output_sequence;
    ++counters_.partial_records_generated;
    counters_.partial_record_contribution_sum += envelope.contribution_count;
    if (!counters_.first_bga_output_accept_cycle)
        counters_.first_bga_output_accept_cycle = cycle_;
    if (state.packer_count == kCSCPartialRecordsPerBurst)
        formBurst(value.global_bg_id, false);
}

void CSCPartialResultPath::cancelReserved() noexcept
{
    if (!reservation_) {
        fail("partial-result cancel without reservation");
        return;
    }
    reservation_.reset();
}

void CSCPartialResultPath::formBurst(uint32_t global_bg, bool tail) noexcept
{
    auto& state = bg_[global_bg];
    const uint32_t count = state.packer_count;
    if ((!tail && count != kCSCPartialRecordsPerBurst) ||
        (tail && (!count || count >= kCSCPartialRecordsPerBurst)) ||
        state.pending.size() >= config_.pending_capacity_bursts_per_bg) {
        fail("invalid partial-result burst formation");
        return;
    }
    CSCPartialResultBurst burst;
    burst.global_bg_id = global_bg;
    burst.writeback_burst_sequence = state.next_burst_sequence++;
    burst.valid_record_count = count;
    burst.tail = tail;
    burst.formation_cycle = cycle_;
    uint64_t contributions = 0;
    for (uint32_t i = 0; i < count; ++i) {
        burst.records[i] = state.packer[i];
        contributions += state.packer[i].contribution_count;
    }
    state.pending.push_back(burst);
    state.packer_count = 0;
    counters_.packed_record_contribution_sum += contributions;
    ++counters_.total_writeback_bursts;
    if (tail)
        ++counters_.tail_writeback_bursts;
    else
        ++counters_.full_writeback_bursts;
    counters_.writeback_useful_bytes +=
        uint64_t(count) * kCSCPartialResultRecordBytes;
    counters_.writeback_transferred_bytes += kCSCPartialResultBurstBytes;
    counters_.writeback_padding_bytes +=
        kCSCPartialResultBurstBytes -
        uint64_t(count) * kCSCPartialResultRecordBytes;
}

void CSCPartialResultPath::completeWrites()
{
    for (uint32_t global_bg = 0; global_bg < bg_.size(); ++global_bg) {
        auto& state = bg_[global_bg];
        while (!state.inflight.empty() &&
               state.inflight.front().burst.completion_cycle <= cycle_) {
            auto transaction = state.inflight.front();
            state.inflight.pop_front();
            if (!state.reserved_buffer_slots ||
                transaction.buffer_slot >= state.resident.size() ||
                state.resident[transaction.buffer_slot] ||
                !state.buffer_reserved[transaction.buffer_slot]) {
                fail("write completion without reserved buffer slot");
                return;
            }
            state.resident[transaction.buffer_slot] = transaction.burst;
            state.buffer_reserved[transaction.buffer_slot] = false;
            --state.reserved_buffer_slots;
            ++counters_.write_requests_completed;
            counters_.last_write_complete_cycle = cycle_;
            if (!counters_.first_write_complete_cycle)
                counters_.first_write_complete_cycle = cycle_;
            for (uint32_t i = 0;
                 i < transaction.burst.valid_record_count; ++i)
                counters_.resident_record_contribution_sum +=
                    transaction.burst.records[i].contribution_count;
        }
    }
}

void CSCPartialResultPath::flushEligibleTails(
    const std::vector<bool>& final_drain_complete)
{
    if (final_drain_complete.size() != bg_.size()) {
        fail("partial-result lifecycle topology mismatch");
        return;
    }
    for (uint32_t global_bg = 0; global_bg < bg_.size(); ++global_bg) {
        auto& state = bg_[global_bg];
        if (final_drain_complete[global_bg])
            state.lifecycle_complete = true;
        if (!state.lifecycle_complete || state.tail_flushed) continue;
        if (!state.packer_count) {
            state.tail_flushed = true;
            continue;
        }
        if (state.pending.size() >= config_.pending_capacity_bursts_per_bg) {
            ++counters_.packer_backpressure_cycles;
            ++counters_.writeback_backpressure_cycles;
            continue;
        }
        formBurst(global_bg, true);
        if (!error_) state.tail_flushed = true;
    }
}

uint32_t CSCPartialResultPath::freeBufferSlot(uint32_t global_bg) const
{
    const auto& state = bg_.at(global_bg);
    for (uint32_t slot = 0; slot < state.resident.size(); ++slot)
        if (!state.resident[slot] && !state.buffer_reserved[slot]) return slot;
    return state.resident.size();
}

void CSCPartialResultPath::issueWrites()
{
    const uint32_t bank_groups_per_rank = config_.bank_groups_per_rank;
    for (uint32_t rank = 0; rank < rank_rr_cursor_.size(); ++rank) {
        uint32_t issued = 0;
        uint32_t scanned_without_issue = 0;
        while (issued < config_.write_issue_limit_per_rank_per_cycle &&
               scanned_without_issue < bank_groups_per_rank) {
            const uint32_t local = rank_rr_cursor_[rank];
            rank_rr_cursor_[rank] =
                (local + 1) % bank_groups_per_rank;
            const uint32_t global_bg =
                rank * bank_groups_per_rank + local;
            auto& state = bg_[global_bg];
            bool did_issue = false;
            if (!state.pending.empty() &&
                state.pending.front().formation_cycle < cycle_) {
                if (state.inflight.size() >=
                    config_.max_inflight_writes_per_bg) {
                    ++counters_.inflight_backpressure_cycles;
                    ++counters_.writeback_backpressure_cycles;
                } else if (state.reserved_buffer_slots +
                               residentBurstCount(global_bg) >=
                           config_.buffer_capacity_bursts_per_bg) {
                    ++counters_.buffer_backpressure_cycles;
                    ++counters_.writeback_backpressure_cycles;
                } else {
                    const uint32_t slot = freeBufferSlot(global_bg);
                    if (slot >= state.resident.size()) {
                        fail("partial-result buffer reservation mismatch");
                        return;
                    }
                    auto burst = state.pending.front();
                    state.pending.pop_front();
                    burst.issue_cycle = cycle_;
                    burst.completion_cycle =
                        cycle_ + std::max<uint32_t>(
                                     1, config_.write_latency_cycles);
                    state.inflight.push_back({burst, slot});
                    state.buffer_reserved[slot] = true;
                    ++state.reserved_buffer_slots;
                    ++counters_.write_requests_issued;
                    counters_.last_write_issue_cycle = cycle_;
                    if (!counters_.first_write_issue_cycle)
                        counters_.first_write_issue_cycle = cycle_;
                    counters_.peak_inflight_writes =
                        std::max(counters_.peak_inflight_writes,
                                 totalInflightWrites());
                    counters_.peak_inflight_writes_by_bg[global_bg] =
                        std::max<uint64_t>(
                            counters_.peak_inflight_writes_by_bg[global_bg],
                            state.inflight.size());
                    if (write_trace_.size() < write_trace_capacity_)
                        write_trace_.push_back(
                            {global_bg, burst.writeback_burst_sequence,
                             burst.issue_cycle, burst.completion_cycle});
                    ++issued;
                    did_issue = true;
                }
            }
            if (did_issue)
                scanned_without_issue = 0;
            else
                ++scanned_without_issue;
        }
    }
}

uint64_t CSCPartialResultPath::totalInflightWrites() const
{
    uint64_t total = 0;
    for (const auto& state : bg_) total += state.inflight.size();
    return total;
}

bool CSCPartialResultPath::conservationInvariant() const
{
    uint64_t packed_records = 0;
    uint64_t resident_records = 0;
    uint64_t generated_bursts = 0;
    for (const auto& state : bg_) {
        for (const auto& burst : state.pending) {
            packed_records += burst.valid_record_count;
            ++generated_bursts;
        }
        for (const auto& transaction : state.inflight) {
            packed_records += transaction.burst.valid_record_count;
            ++generated_bursts;
        }
        for (const auto& burst : state.resident)
            if (burst) {
                packed_records += burst->valid_record_count;
                resident_records += burst->valid_record_count;
                ++generated_bursts;
            }
    }
    return generated_bursts == counters_.total_writeback_bursts &&
           counters_.total_writeback_bursts ==
               counters_.full_writeback_bursts +
                   counters_.tail_writeback_bursts &&
           counters_.writeback_transferred_bytes ==
               uint64_t(kCSCPartialResultBurstBytes) *
                   counters_.total_writeback_bursts &&
           counters_.writeback_useful_bytes ==
               uint64_t(kCSCPartialResultRecordBytes) *
                   counters_.partial_records_generated &&
           counters_.writeback_padding_bytes ==
               counters_.writeback_transferred_bytes -
                   counters_.writeback_useful_bytes &&
           packed_records == counters_.partial_records_generated &&
           resident_records == counters_.partial_records_generated &&
           counters_.partial_record_contribution_sum ==
               counters_.packed_record_contribution_sum &&
           counters_.packed_record_contribution_sum ==
               counters_.resident_record_contribution_sum;
}

void CSCPartialResultPath::updateCompletion()
{
    if (error_ || counters_.partial_writeback_complete_cycle) return;
    for (const auto& state : bg_)
        if (!state.lifecycle_complete || !state.tail_flushed ||
            state.packer_count || !state.pending.empty() ||
            !state.inflight.empty())
            return;
    if (!conservationInvariant()) {
        fail("partial-result writeback conservation mismatch");
        return;
    }
    counters_.partial_writeback_complete_cycle = cycle_;
}

void CSCPartialResultPath::step(
    uint64_t cycle,
    const std::vector<bool>& final_drain_complete)
{
    if (cycle < cycle_) {
        fail("partial-result path cycle rollback");
        return;
    }
    cycle_ = cycle;
    if (error_) return;
    completeWrites();
    if (error_) return;
    flushEligibleTails(final_drain_complete);
    if (error_) return;
    issueWrites();
    if (error_) return;
    updateCompletion();
}

bool CSCPartialResultPath::partialWritebackComplete() const
{
    return !error_ && counters_.partial_writeback_complete_cycle != 0;
}

uint32_t CSCPartialResultPath::packerRecordCount(uint32_t bg) const
{
    return bg_.at(bg).packer_count;
}

uint32_t CSCPartialResultPath::pendingBurstCount(uint32_t bg) const
{
    return bg_.at(bg).pending.size();
}

uint32_t CSCPartialResultPath::inflightWriteCount(uint32_t bg) const
{
    return bg_.at(bg).inflight.size();
}

uint32_t CSCPartialResultPath::residentBurstCount(uint32_t bg) const
{
    uint32_t count = 0;
    for (const auto& burst : bg_.at(bg).resident)
        if (burst) ++count;
    return count;
}

uint32_t CSCPartialResultPath::reservedBufferSlots(uint32_t bg) const
{
    return bg_.at(bg).reserved_buffer_slots;
}

const CSCPartialResultBurst& CSCPartialResultPath::residentBurst(
    uint32_t bg, uint32_t ordinal) const
{
    uint32_t found = 0;
    for (const auto& burst : bg_.at(bg).resident)
        if (burst) {
            if (found++ == ordinal) return *burst;
        }
    throw std::out_of_range("partial-result resident burst");
}

}  // namespace csc_descriptor
