#include "csc/CSCPartialResultPath.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace csc_descriptor {

void CSCPartialResultPathConfig::validate() const
{
    const uint64_t topology_bg_count =
        uint64_t(channel_count) * ranks_per_channel *
        bank_groups_per_rank;
    if (!global_bg_count || !channel_count || !ranks_per_channel ||
        !bank_groups_per_rank || topology_bg_count != global_bg_count ||
        !buffer_capacity_bursts_per_bg ||
        !pending_capacity_bursts_per_bg ||
        !max_inflight_writes_per_bg ||
        !write_issue_limit_per_rank_per_cycle ||
        !read_issue_limit_per_channel_per_cycle ||
        !max_inflight_reads_per_channel ||
        !host_return_queue_capacity_bursts ||
        (host_reduction_enabled &&
         (!reduction_queue_capacity_records ||
          !host_reduce_records_per_cycle)))
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
    readback_counters_.peak_inflight_reads_by_channel.resize(
        config_.channel_count);
    inflight_reads_by_channel_.resize(config_.channel_count);
    channel_read_rr_cursor_.resize(config_.channel_count);
    if (config_.host_reduction_enabled) {
        reduction_queue_.resize(
            config_.reduction_queue_capacity_records);
        reduction_batch_.reserve(
            config_.host_reduce_records_per_cycle);
    }
    final_y_fp32_.assign(rows_, 0.0F);
    row_reduced_.assign(rows_, false);
    row_first_global_bg_.assign(rows_, 0);
    for (auto& state : bg_) {
        state.resident.resize(config_.buffer_capacity_bursts_per_bg);
        state.buffer_reserved.resize(
            config_.buffer_capacity_bursts_per_bg, false);
        state.read_inflight.resize(
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
                state.read_inflight[transaction.buffer_slot] ||
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
        if (!state.resident[slot] && !state.buffer_reserved[slot] &&
            !state.read_inflight[slot])
            return slot;
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

bool CSCPartialResultPath::findReadCandidate(
    uint32_t global_bg, uint32_t& selected_slot) const
{
    const auto& state = bg_.at(global_bg);
    uint64_t sequence = std::numeric_limits<uint64_t>::max();
    selected_slot = state.resident.size();
    for (uint32_t slot = 0; slot < state.resident.size(); ++slot) {
        if (!state.resident[slot] || state.read_inflight[slot]) continue;
        const uint64_t candidate =
            state.resident[slot]->writeback_burst_sequence;
        if (candidate < sequence) {
            sequence = candidate;
            selected_slot = slot;
        }
    }
    return selected_slot < state.resident.size() &&
           sequence == state.next_read_burst_sequence;
}

uint64_t CSCPartialResultPath::totalInflightReads() const
{
    uint64_t total = 0;
    for (const auto& channel : inflight_reads_by_channel_)
        total += channel.size();
    return total;
}

void CSCPartialResultPath::completeReads()
{
    std::vector<ReadTransaction> completed;
    for (auto& channel : inflight_reads_by_channel_) {
        for (auto it = channel.begin(); it != channel.end();) {
            if (it->completion_cycle <= cycle_) {
                completed.push_back(*it);
                it = channel.erase(it);
            } else {
                ++it;
            }
        }
    }
    std::sort(completed.begin(), completed.end(),
              [](const ReadTransaction& a, const ReadTransaction& b) {
                  return std::tie(a.completion_cycle, a.channel_id,
                                  a.rank_id, a.local_bg_id,
                                  a.burst.writeback_burst_sequence) <
                         std::tie(b.completion_cycle, b.channel_id,
                                  b.rank_id, b.local_bg_id,
                                  b.burst.writeback_burst_sequence);
              });
    for (const auto& transaction : completed) {
        const uint32_t global_rank =
            transaction.channel_id * config_.ranks_per_channel +
            transaction.rank_id;
        const uint32_t global_bg =
            global_rank * config_.bank_groups_per_rank +
            transaction.local_bg_id;
        if (global_bg >= bg_.size() ||
            transaction.buffer_slot >= bg_[global_bg].resident.size() ||
            !bg_[global_bg].resident[transaction.buffer_slot] ||
            !bg_[global_bg].read_inflight[transaction.buffer_slot] ||
            bg_[global_bg].resident[transaction.buffer_slot]
                    ->writeback_burst_sequence !=
                transaction.burst.writeback_burst_sequence) {
            fail("read completion without in-flight resident burst");
            return;
        }
        if (!reserved_host_return_slots_ ||
            host_return_queue_.size() >=
                config_.host_return_queue_capacity_bursts) {
            fail("read completion without reserved return slot");
            return;
        }
        host_return_queue_.push_back(
            {transaction.burst, transaction.channel_id,
             transaction.rank_id, transaction.local_bg_id, cycle_ + 1});
        --reserved_host_return_slots_;
        bg_[global_bg].resident[transaction.buffer_slot].reset();
        bg_[global_bg].read_inflight[transaction.buffer_slot] = false;

        ++readback_counters_.read_requests_completed;
        ++readback_counters_.returned_bursts_committed;
        readback_counters_.last_read_complete_cycle = cycle_;
        if (!readback_counters_.first_read_complete_cycle)
            readback_counters_.first_read_complete_cycle = cycle_;
        readback_counters_.peak_host_return_queue_occupancy =
            std::max<uint64_t>(
                readback_counters_.peak_host_return_queue_occupancy,
                host_return_queue_.size());
        readback_counters_.readback_transferred_bytes +=
            kCSCPartialResultBurstBytes;
        readback_counters_.readback_records +=
            transaction.burst.valid_record_count;
        readback_counters_.readback_useful_bytes +=
            uint64_t(transaction.burst.valid_record_count) *
            kCSCPartialResultRecordBytes;
        readback_counters_.readback_padding_bytes +=
            kCSCPartialResultBurstBytes -
            uint64_t(transaction.burst.valid_record_count) *
                kCSCPartialResultRecordBytes;
        for (uint32_t record = 0;
             record < transaction.burst.valid_record_count; ++record)
            readback_counters_.readback_contribution_sum +=
                transaction.burst.records[record].contribution_count;
    }
}

void CSCPartialResultPath::issueReads()
{
    const uint32_t bgs_per_channel =
        config_.ranks_per_channel * config_.bank_groups_per_rank;
    for (uint32_t channel = 0; channel < config_.channel_count; ++channel) {
        uint32_t issued = 0;
        uint32_t scanned_without_issue = 0;
        while (issued < config_.read_issue_limit_per_channel_per_cycle &&
               scanned_without_issue < bgs_per_channel) {
            if (inflight_reads_by_channel_[channel].size() >=
                config_.max_inflight_reads_per_channel) {
                ++readback_counters_.read_inflight_backpressure_cycles;
                break;
            }
            if (host_return_queue_.size() +
                    reserved_host_return_slots_ >=
                config_.host_return_queue_capacity_bursts) {
                ++readback_counters_.return_queue_backpressure_cycles;
                break;
            }
            const uint32_t channel_bg =
                channel_read_rr_cursor_[channel];
            channel_read_rr_cursor_[channel] =
                (channel_bg + 1) % bgs_per_channel;
            const uint32_t rank =
                channel_bg / config_.bank_groups_per_rank;
            const uint32_t local_bg =
                channel_bg % config_.bank_groups_per_rank;
            const uint32_t global_bg =
                (channel * config_.ranks_per_channel + rank) *
                    config_.bank_groups_per_rank +
                local_bg;
            uint32_t slot = 0;
            if (!findReadCandidate(global_bg, slot)) {
                ++scanned_without_issue;
                continue;
            }

            auto& state = bg_[global_bg];
            const auto burst = *state.resident[slot];
            if (!burst.valid_record_count ||
                burst.valid_record_count > kCSCPartialRecordsPerBurst) {
                fail("invalid resident burst record count");
                return;
            }
            state.read_inflight[slot] = true;
            ++state.next_read_burst_sequence;
            ++reserved_host_return_slots_;
            const uint64_t completion =
                cycle_ + std::max<uint32_t>(
                             1, config_.read_latency_cycles);
            inflight_reads_by_channel_[channel].push_back(
                {burst, slot, channel, rank, local_bg, cycle_,
                 completion});
            ++readback_counters_.read_requests_issued;
            readback_counters_.last_read_issue_cycle = cycle_;
            if (!readback_counters_.first_read_issue_cycle)
                readback_counters_.first_read_issue_cycle = cycle_;
            readback_counters_.peak_inflight_reads =
                std::max(readback_counters_.peak_inflight_reads,
                         totalInflightReads());
            readback_counters_.peak_inflight_reads_by_channel[channel] =
                std::max<uint64_t>(
                    readback_counters_
                        .peak_inflight_reads_by_channel[channel],
                    inflight_reads_by_channel_[channel].size());
            readback_counters_.peak_reserved_return_slots =
                std::max<uint64_t>(
                    readback_counters_.peak_reserved_return_slots,
                    reserved_host_return_slots_);
            ++issued;
            scanned_without_issue = 0;
        }
    }
}

bool CSCPartialResultPath::readbackConservationInvariant() const
{
    return readback_counters_.read_requests_issued ==
               counters_.write_requests_completed &&
           readback_counters_.read_requests_completed ==
               readback_counters_.read_requests_issued &&
           readback_counters_.returned_bursts_committed ==
               readback_counters_.read_requests_completed &&
           readback_counters_.readback_records ==
               counters_.partial_records_generated &&
           readback_counters_.readback_contribution_sum ==
               counters_.partial_record_contribution_sum &&
           readback_counters_.readback_transferred_bytes ==
               uint64_t(kCSCPartialResultBurstBytes) *
                   readback_counters_.read_requests_completed &&
           readback_counters_.readback_useful_bytes ==
               uint64_t(kCSCPartialResultRecordBytes) *
                   readback_counters_.readback_records &&
           readback_counters_.readback_padding_bytes ==
               readback_counters_.readback_transferred_bytes -
                   readback_counters_.readback_useful_bytes &&
           readback_counters_.readback_transferred_bytes ==
               counters_.writeback_transferred_bytes &&
           readback_counters_.readback_useful_bytes ==
               counters_.writeback_useful_bytes &&
           readback_counters_.readback_padding_bytes ==
               counters_.writeback_padding_bytes;
}

void CSCPartialResultPath::updateReadTransportCompletion()
{
    if (error_ || !readback_counters_.host_readback_start_cycle ||
        readback_counters_.read_transport_complete_cycle)
        return;
    if (totalInflightReads() || reserved_host_return_slots_) return;
    for (const auto& state : bg_)
        for (uint32_t slot = 0; slot < state.resident.size(); ++slot)
            if (state.resident[slot] || state.read_inflight[slot])
                return;
    if (!readbackConservationInvariant()) {
        fail("partial-result readback conservation mismatch");
        return;
    }
    readback_counters_.read_transport_complete_cycle = cycle_;
}

void CSCPartialResultPath::stepHostReadback()
{
    if (!partialWritebackComplete()) return;
    if (!readback_counters_.host_readback_start_cycle) {
        readback_counters_.host_readback_start_cycle = cycle_;
        return;
    }
    completeReads();
    if (error_) return;
    issueReads();
    if (error_) return;
    updateReadTransportCompletion();
}

void CSCPartialResultPath::enqueueReductionRecord(
    const ReductionRecord& record) noexcept
{
    if (!config_.host_reduction_enabled ||
        reduction_queue_count_ >= reduction_queue_.size() ||
        reduction_queue_[reduction_queue_tail_]) {
        fail("host reduction queue reservation mismatch");
        return;
    }
    reduction_queue_[reduction_queue_tail_] = record;
    reduction_queue_tail_ =
        (reduction_queue_tail_ + 1) % reduction_queue_.size();
    ++reduction_queue_count_;
}

CSCPartialResultPath::ReductionRecord
CSCPartialResultPath::dequeueReductionRecord() noexcept
{
    if (!reduction_queue_count_ ||
        !reduction_queue_[reduction_queue_head_]) {
        fail("host reduction dequeue mismatch");
        return {};
    }
    const auto record = *reduction_queue_[reduction_queue_head_];
    reduction_queue_[reduction_queue_head_].reset();
    reduction_queue_head_ =
        (reduction_queue_head_ + 1) % reduction_queue_.size();
    --reduction_queue_count_;
    return record;
}

const CSCPartialResultPath::ReductionRecord&
CSCPartialResultPath::frontReductionRecord() const
{
    if (!reduction_queue_count_ ||
        !reduction_queue_[reduction_queue_head_])
        throw std::logic_error("host reduction queue empty");
    return *reduction_queue_[reduction_queue_head_];
}

void CSCPartialResultPath::completeReductionBatch()
{
    if (!reduction_batch_inflight_ ||
        reduction_batch_completion_cycle_ > cycle_)
        return;
    if (reduction_batch_.empty()) {
        fail("reduction completion without in-flight records");
        return;
    }
    for (const auto& item : reduction_batch_) {
        const auto& envelope = item.envelope;
        const uint32_t row = envelope.record.row_idx;
        if (row >= rows_) {
            fail("host reduction row out of range");
            return;
        }
        if (row_reduced_[row]) {
            if (row_first_global_bg_[row] == envelope.global_bg_id)
                ++reduction_counters_
                      .same_row_repeated_record_merges;
            else
                ++reduction_counters_.same_row_cross_bg_merges;
        } else {
            row_reduced_[row] = true;
            row_first_global_bg_[row] = envelope.global_bg_id;
        }
        final_y_fp32_[row] =
            static_cast<float>(
                final_y_fp32_[row] + envelope.record.value);
        ++reduction_counters_.reduced_partial_records;
        reduction_counters_.reduced_contribution_count +=
            envelope.contribution_count;
        ++reduction_counters_.fp32_host_add_count;
    }
    ++reduction_counters_.reduction_batches_completed;
    if (!reduction_counters_.first_host_reduce_cycle)
        reduction_counters_.first_host_reduce_cycle = cycle_;
    reduction_counters_.last_host_reduce_cycle = cycle_;
    reduction_batch_.clear();
    reduction_batch_inflight_ = false;
    reduction_batch_completion_cycle_ = 0;
}

void CSCPartialResultPath::acceptReturnedBurstForReduction()
{
    if (!hasHostReturnedBurst()) return;
    const auto returned = peekHostReturnedBurst();
    const auto& burst = returned.burst;
    if (!burst.valid_record_count ||
        burst.valid_record_count > kCSCPartialRecordsPerBurst) {
        fail("invalid returned burst record count");
        return;
    }
    if (reduction_queue_.size() - reduction_queue_count_ <
        burst.valid_record_count) {
        ++reduction_counters_.reduction_queue_backpressure_cycles;
        return;
    }

    std::array<ReductionRecord, kCSCPartialRecordsPerBurst> records{};
    for (uint32_t index = 0;
         index < burst.valid_record_count; ++index) {
        const auto& envelope = burst.records[index];
        if (envelope.record.row_idx >= rows_ ||
            !envelope.generation ||
            envelope.generation != generation_ ||
            !envelope.contribution_count ||
            envelope.global_bg_id >= config_.global_bg_count ||
            envelope.global_bg_id != burst.global_bg_id) {
            fail("invalid returned partial-result record");
            return;
        }
        records[index] = {envelope, index, cycle_ + 1};
    }

    try {
        acceptHostReturnedBurst();
    } catch (const std::exception& error) {
        fail(std::string("returned burst ownership transfer failed: ") +
             error.what());
        return;
    }
    for (uint32_t index = 0;
         index < burst.valid_record_count; ++index) {
        enqueueReductionRecord(records[index]);
        if (error_) return;
    }
    reduction_counters_.reduction_records_enqueued +=
        burst.valid_record_count;
    if (!reduction_counters_.first_returned_burst_accept_cycle)
        reduction_counters_.first_returned_burst_accept_cycle = cycle_;
    reduction_counters_.peak_reduction_queue_occupancy =
        std::max<uint64_t>(
            reduction_counters_.peak_reduction_queue_occupancy,
            reduction_queue_count_);
}

void CSCPartialResultPath::issueReductionBatch()
{
    if (reduction_batch_inflight_ || !reduction_queue_count_)
        return;
    if (frontReductionRecord().eligible_cycle > cycle_) return;

    const uint32_t limit =
        std::min(config_.host_reduce_records_per_cycle,
                 reduction_queue_count_);
    for (uint32_t index = 0; index < limit; ++index) {
        if (frontReductionRecord().eligible_cycle > cycle_) break;
        reduction_batch_.push_back(dequeueReductionRecord());
        if (error_) return;
    }
    if (reduction_batch_.empty()) return;
    reduction_batch_inflight_ = true;
    reduction_batch_completion_cycle_ =
        cycle_ + std::max<uint32_t>(
                     1, config_.host_reduce_latency_cycles);
    reduction_counters_.reduction_records_issued +=
        reduction_batch_.size();
    ++reduction_counters_.reduction_batches_issued;
    if (!reduction_counters_.first_reduction_batch_issue_cycle)
        reduction_counters_.first_reduction_batch_issue_cycle = cycle_;
}

bool CSCPartialResultPath::reductionConservationInvariant() const
{
    return reduction_counters_.reduction_records_enqueued ==
               readback_counters_.readback_records &&
           reduction_counters_.reduction_records_issued ==
               reduction_counters_.reduction_records_enqueued &&
           reduction_counters_.reduced_partial_records ==
               reduction_counters_.reduction_records_issued &&
           reduction_counters_.reduced_contribution_count ==
               readback_counters_.readback_contribution_sum &&
           reduction_counters_.fp32_host_add_count ==
               reduction_counters_.reduced_partial_records &&
           reduction_counters_.reduction_batches_issued ==
               reduction_counters_.reduction_batches_completed;
}

void CSCPartialResultPath::updateHostCompletion()
{
    if (error_) return;
    if (!reduction_counters_.host_readback_complete_cycle &&
        hostReadTransportComplete() &&
        host_return_queue_.empty() &&
        !reserved_host_return_slots_ &&
        readback_counters_.returned_bursts_delivered ==
            readback_counters_.returned_bursts_committed)
        reduction_counters_.host_readback_complete_cycle = cycle_;

    if (reduction_counters_.host_reduction_complete_cycle ||
        !hostReadbackComplete() || reduction_queue_count_ ||
        reduction_batch_inflight_)
        return;
    if (!reductionConservationInvariant()) {
        fail("host reduction conservation mismatch");
        return;
    }
    reduction_counters_.host_reduction_complete_cycle = cycle_;
}

void CSCPartialResultPath::stepHostReduction()
{
    if (!config_.host_reduction_enabled) return;
    completeReductionBatch();
    if (error_) return;
    acceptReturnedBurstForReduction();
    if (error_) return;
    issueReductionBatch();
    if (error_) return;
    updateHostCompletion();
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
    if (error_) return;
    stepHostReadback();
    if (error_) return;
    stepHostReduction();
}

bool CSCPartialResultPath::partialWritebackComplete() const
{
    return !error_ && counters_.partial_writeback_complete_cycle != 0;
}

bool CSCPartialResultPath::hostReadTransportComplete() const
{
    return !error_ &&
           readback_counters_.read_transport_complete_cycle != 0;
}

bool CSCPartialResultPath::hostReadbackComplete() const
{
    return config_.host_reduction_enabled && !error_ &&
           reduction_counters_.host_readback_complete_cycle != 0;
}

bool CSCPartialResultPath::hostReductionComplete() const
{
    return config_.host_reduction_enabled && !error_ &&
           reduction_counters_.host_reduction_complete_cycle != 0;
}

const std::vector<float>&
CSCPartialResultPath::hostReducedResult() const
{
    if (!hostReductionComplete())
        throw std::logic_error(
            "host reduced result before reduction completion");
    return final_y_fp32_;
}

bool CSCPartialResultPath::hasHostReturnedBurst() const
{
    return !host_return_queue_.empty() &&
           host_return_queue_.front().host_visible_cycle <= cycle_;
}

const CSCHostReturnedBurst&
CSCPartialResultPath::peekHostReturnedBurst() const
{
    if (!hasHostReturnedBurst())
        throw std::logic_error("no host-visible returned burst");
    return host_return_queue_.front();
}

void CSCPartialResultPath::acceptHostReturnedBurst()
{
    if (!hasHostReturnedBurst())
        throw std::logic_error("no host-visible returned burst");
    host_return_queue_.pop_front();
    ++readback_counters_.returned_bursts_delivered;
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

uint32_t CSCPartialResultPath::readInflightBufferSlots(uint32_t bg) const
{
    uint32_t count = 0;
    for (bool inflight : bg_.at(bg).read_inflight)
        if (inflight) ++count;
    return count;
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
