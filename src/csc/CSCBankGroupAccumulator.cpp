#include "csc/CSCBankGroupAccumulator.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace csc_descriptor {

namespace {
constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
}

CSCBankGroupAccumulator::CSCBankGroupAccumulator(const CSCBGAConfig& config,
                                                 uint32_t generation)
    : config_(config), generation_(generation), streams_(config.input_streams),
      accumulator_(config.accumulator_entries)
{
    validateConfig();
    if (!generation_) throw std::invalid_argument("BGA generation must be nonzero");
    for (uint32_t slot = 0; slot < accumulator_.size(); ++slot)
        accumulator_[slot].slot = slot;
}

void CSCBankGroupAccumulator::validateConfig() const
{
    if (!config_.input_streams || !config_.input_queue_depth ||
        !config_.accumulator_entries || !config_.output_queue_depth || !config_.rows)
        throw std::invalid_argument("BGA capacities and rows must be nonzero");
    if (!config_.compare_width || config_.compare_width < config_.accumulator_entries)
        throw std::invalid_argument(
            "M7A-1 requires compare_width >= accumulator_entries");
    if (!config_.compare_latency || !config_.add_latency)
        throw std::invalid_argument("BGA latencies must be nonzero");
}

uint32_t CSCBankGroupAccumulator::floatBits(float value)
{
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "FP32 size");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool CSCBankGroupAccumulator::sameBatch(const CSCBGABatch& a,
                                        const CSCBGABatch& b) const
{
    if (a.logical_stream_id != b.logical_stream_id || a.generation != b.generation ||
        a.sequence != b.sequence || a.partials.size() != b.partials.size())
        return false;
    for (size_t i = 0; i < a.partials.size(); ++i)
        if (a.partials[i].row_idx != b.partials[i].row_idx ||
            floatBits(a.partials[i].value) != floatBits(b.partials[i].value))
            return false;
    return true;
}

void CSCBankGroupAccumulator::saturatingIncrement(uint64_t& value)
{
    if (value != kMax) ++value;
}

void CSCBankGroupAccumulator::checkedAdd(uint64_t& value, uint64_t delta,
                                         const char* name)
{
    if (delta > kMax - value)
        throw std::overflow_error(std::string("BGA exhausted ") + name);
    value += delta;
}

uint64_t CSCBankGroupAccumulator::checkedSum(uint64_t value, uint64_t delta,
                                             const char* name)
{
    if (delta > kMax - value)
        throw std::overflow_error(std::string("BGA exhausted ") + name);
    return value + delta;
}

void CSCBankGroupAccumulator::protocolError(const std::string& message)
{
    saturatingIncrement(counters_.protocol_errors);
    error_ = message;
}

[[noreturn]] void CSCBankGroupAccumulator::invariantError(
    const std::string& message)
{
    saturatingIncrement(counters_.invariant_errors);
    error_ = message;
    throw std::logic_error(message);
}

CSCBGAInputResult CSCBankGroupAccumulator::acceptBatch(
    StreamState& stream, const CSCBGABatch& batch)
{
    checkedAdd(counters_.accepted_valid_partials, batch.partials.size(),
               "accepted contribution count");
    stream.accepted_pending = batch;
    stream.last_accepted_batch = batch;
    stream.last_accepted_sequence = batch.sequence;
    saturatingIncrement(counters_.accepted_batches);
    return CSCBGAInputResult::ACCEPTED;
}

CSCBGAInputResult CSCBankGroupAccumulator::offerBatch(const CSCBGABatch& batch)
{
    saturatingIncrement(counters_.enqueue_attempts);
    if (batch.logical_stream_id >= streams_.size()) {
        protocolError("BGA logical stream out of range");
        return CSCBGAInputResult::PROTOCOL_ERROR;
    }
    auto& stream = streams_[batch.logical_stream_id];
    if (stream.done || batch.generation != generation_ || !batch.sequence ||
        batch.partials.empty() || batch.partials.size() > config_.input_queue_depth) {
        protocolError("invalid BGA batch identity or payload");
        return CSCBGAInputResult::PROTOCOL_ERROR;
    }
    for (const auto& partial : batch.partials)
        if (partial.row_idx >= config_.rows) {
            protocolError("BGA row index out of range");
            return CSCBGAInputResult::PROTOCOL_ERROR;
        }

    if (stream.accepted_pending) {
        if (sameBatch(*stream.accepted_pending, batch))
            return CSCBGAInputResult::DUPLICATE;
        protocolError("producer exceeded one accepted outstanding BGA batch");
        return CSCBGAInputResult::PROTOCOL_ERROR;
    }
    if (stream.presented_batch) {
        if (!sameBatch(*stream.presented_batch, batch)) {
            protocolError("BGA payload changed while valid and not ready");
            return CSCBGAInputResult::PROTOCOL_ERROR;
        }
        if (stream.queue.size() + batch.partials.size() > config_.input_queue_depth) {
            saturatingIncrement(counters_.backpressured_batches);
            return CSCBGAInputResult::BACKPRESSURE;
        }
        stream.presented_batch.reset();
        return acceptBatch(stream, batch);
    }
    if (stream.last_accepted_batch && batch.sequence == stream.last_accepted_sequence) {
        if (!sameBatch(*stream.last_accepted_batch, batch)) {
            protocolError("BGA duplicate identity has conflicting payload");
            return CSCBGAInputResult::PROTOCOL_ERROR;
        }
        return CSCBGAInputResult::DUPLICATE;
    }
    if (batch.sequence < stream.last_accepted_sequence) {
        protocolError("stale BGA batch sequence");
        return CSCBGAInputResult::PROTOCOL_ERROR;
    }
    if (stream.last_accepted_sequence == kMax ||
        batch.sequence != stream.last_accepted_sequence + 1) {
        protocolError("skipped or exhausted BGA batch sequence");
        return CSCBGAInputResult::PROTOCOL_ERROR;
    }
    if (stream.queue.size() + batch.partials.size() > config_.input_queue_depth) {
        stream.presented_batch = batch;
        saturatingIncrement(counters_.backpressured_batches);
        return CSCBGAInputResult::BACKPRESSURE;
    }
    return acceptBatch(stream, batch);
}

void CSCBankGroupAccumulator::markProducerDone(uint32_t stream)
{
    if (stream >= streams_.size()) throw std::out_of_range("BGA stream");
    if (streams_[stream].presented_batch) {
        protocolError("producer done with unaccepted BGA valid");
        return;
    }
    streams_[stream].done = true;
}

bool CSCBankGroupAccumulator::requestFinalDrain()
{
    if (!allProducersDone()) return false;
    if (!final_drain_requested_) {
        if (final_drain_epoch_ == kMax)
            throw std::overflow_error("BGA exhausted final-drain epoch");
        ++final_drain_epoch_;
        final_drain_requested_ = true;
    }
    return true;
}

void CSCBankGroupAccumulator::abortReset(uint32_t new_generation)
{
    if (!new_generation || new_generation <= generation_)
        throw std::invalid_argument(
            "BGA reset requires a strictly newer nonzero generation");
    const uint64_t live = liveContributionCount();
    checkedAdd(counters_.aborted_contributions, live,
               "aborted contribution count");
    for (auto& stream : streams_) stream = StreamState{};
    for (uint32_t slot = 0; slot < accumulator_.size(); ++slot) {
        accumulator_[slot] = CSCBGAAccumulatorEntry{};
        accumulator_[slot].slot = slot;
    }
    output_queue_.clear();
    operation_.reset();
    pending_miss_.reset();
    output_retirement_pending_ = false;
    final_drain_requested_ = false;
    next_stream_rr_ = 0;
    generation_ = new_generation;
    error_.clear();
}

const CSCBGAOutput& CSCBankGroupAccumulator::peekOutput() const
{
    if (output_queue_.empty()) throw std::logic_error("BGA output queue empty");
    return output_queue_.front().payload;
}

void CSCBankGroupAccumulator::acceptOutput()
{
    if (output_queue_.empty() || output_retirement_pending_)
        throw std::logic_error("invalid BGA output acceptance");
    output_retirement_pending_ = true;
}

void CSCBankGroupAccumulator::retireOutput()
{
    const OutputRecord retiring = output_queue_.front();
    auto& slot = accumulator_.at(retiring.reserved_slot);
    if (!slot.valid || !slot.reserved)
        invariantError("BGA retirement lost reserved slot");
    const uint64_t retired =
        checkedSum(counters_.retired_output_contributions,
                   retiring.payload.contribution_count,
                   "retired contribution count");
    counters_.retired_output_contributions = retired;
    slot = CSCBGAAccumulatorEntry{};
    slot.slot = retiring.reserved_slot;
    output_queue_.pop_front();
    output_retirement_pending_ = false;
    if (pending_miss_ && pending_miss_->awaiting_retirement &&
        pending_miss_->eviction_output_sequence == retiring.payload.output_sequence)
        pending_miss_->awaiting_retirement = false;
}

void CSCBankGroupAccumulator::step()
{
    if (cycle_ == kMax) throw std::overflow_error("BGA exhausted cycle");
    const bool was_busy = hasLiveWork();
    ++cycle_;
    operation_committed_this_cycle_ = false;

    if (output_retirement_pending_) retireOutput();
    if (operation_ && !--operation_->remaining) commitOperation();
    updateEvictionOrDrain();
    startService();
    commitAcceptedBatches();

    if (was_busy || hasLiveWork())
        saturatingIncrement(counters_.cycles_busy);
    else
        saturatingIncrement(counters_.cycles_idle);
    if (!validateAccumulatorInvariant())
        invariantError("BGA duplicate/reservation accumulator invariant");
    if (!conservationInvariant())
        invariantError("BGA contribution conservation invariant");
}

void CSCBankGroupAccumulator::commitOperation()
{
    const Operation completed = *operation_;
    if (completed.kind == OperationKind::MERGE &&
        accumulator_.at(completed.slot).contribution_count == kMax)
        throw std::overflow_error("BGA exhausted entry contribution count");
    if (completed.kind == OperationKind::INSERT && insertion_age_ == kMax)
        throw std::overflow_error("BGA exhausted insertion age");
    operation_.reset();
    operation_committed_this_cycle_ = true;
    if (completed.kind == OperationKind::LOOKUP) {
        uint32_t hit = accumulator_.size();
        uint32_t hits = 0;
        uint32_t free = accumulator_.size();
        for (uint32_t slot = 0; slot < accumulator_.size(); ++slot) {
            const auto& entry = accumulator_[slot];
            if (entry.valid && !entry.reserved) {
                saturatingIncrement(counters_.tag_comparisons);
                if (entry.row_idx == completed.input.row_idx) {
                    hit = slot;
                    ++hits;
                }
            } else if (!entry.valid && free == accumulator_.size()) {
                free = slot;
            }
        }
        if (hits > 1) invariantError("BGA duplicate row tag detected by lookup");
        if (hits == 1) {
            saturatingIncrement(counters_.lookup_hits);
            operation_ = Operation{OperationKind::MERGE, completed.input, hit,
                                   config_.add_latency};
        } else {
            saturatingIncrement(counters_.lookup_misses);
            if (free != accumulator_.size())
                operation_ = Operation{OperationKind::INSERT, completed.input, free,
                                       config_.add_latency};
            else {
                const uint32_t victim = selectVictim();
                if (victim == accumulator_.size())
                    invariantError("BGA full miss has no non-reserved victim");
                pending_miss_ = PendingMiss{completed.input, victim, 0, false};
            }
        }
        return;
    }
    if (completed.kind == OperationKind::MERGE) {
        auto& entry = accumulator_[completed.slot];
        if (!entry.valid || entry.reserved || entry.row_idx != completed.input.row_idx)
            invariantError("BGA merge target changed");
        volatile float rounded = entry.value + completed.input.value;
        entry.value = rounded;
        ++entry.contribution_count;
        saturatingIncrement(counters_.fp32_merges);
        return;
    }
    insertAt(completed.slot, completed.input);
}

void CSCBankGroupAccumulator::updateEvictionOrDrain()
{
    if (pending_miss_) {
        if (!pending_miss_->eviction_output_sequence) {
            if (output_queue_.size() >= config_.output_queue_depth) {
                saturatingIncrement(counters_.output_stalls);
                return;
            }
            if (capacity_epoch_ == kMax)
                throw std::overflow_error("BGA exhausted capacity epoch");
            const uint64_t epoch = capacity_epoch_ + 1;
            const uint64_t sequence = pushOutput(
                pending_miss_->victim_slot,
                CSCBGAOutputReason::CAPACITY_EVICTION, epoch);
            capacity_epoch_ = epoch;
            pending_miss_->eviction_output_sequence = sequence;
            pending_miss_->awaiting_retirement = true;
            saturatingIncrement(counters_.capacity_evictions);
            return;
        }
        if (pending_miss_->awaiting_retirement) return;
        if (!operation_) {
            operation_ = Operation{OperationKind::INSERT, pending_miss_->input,
                                   pending_miss_->victim_slot, config_.add_latency};
            pending_miss_.reset();
        }
        return;
    }

    if (!final_drain_requested_ || operation_ || operation_committed_this_cycle_) return;
    bool input_pending = false;
    for (const auto& stream : streams_)
        input_pending = input_pending || !stream.queue.empty() ||
                        stream.accepted_pending || stream.presented_batch;
    if (input_pending) return;
    const uint32_t slot = selectDrainEntry();
    if (slot == accumulator_.size()) return;
    if (output_queue_.size() >= config_.output_queue_depth) {
        saturatingIncrement(counters_.output_stalls);
        return;
    }
    pushOutput(slot, CSCBGAOutputReason::FINAL_DRAIN, final_drain_epoch_);
    saturatingIncrement(counters_.final_drain_outputs);
}

void CSCBankGroupAccumulator::startService()
{
    if (operation_ || pending_miss_) return;
    for (uint32_t offset = 0; offset < streams_.size(); ++offset) {
        const uint32_t stream = (next_stream_rr_ + offset) % streams_.size();
        if (streams_[stream].queue.empty()) continue;
        InputEntry input = streams_[stream].queue.front();
        streams_[stream].queue.pop_front();
        operation_ =
            Operation{OperationKind::LOOKUP, input, 0, config_.compare_latency};
        next_stream_rr_ = (stream + 1) % streams_.size();
        return;
    }
}

void CSCBankGroupAccumulator::commitAcceptedBatches()
{
    for (auto& stream : streams_) {
        if (!stream.accepted_pending) continue;
        const auto& batch = *stream.accepted_pending;
        if (stream.queue.size() + batch.partials.size() > config_.input_queue_depth)
            invariantError("accepted BGA batch lost reserved FIFO capacity");
        for (const auto& partial : batch.partials)
            stream.queue.push_back({partial.row_idx, partial.value,
                                    batch.logical_stream_id, batch.generation,
                                    batch.sequence});
        stream.accepted_pending.reset();
    }
}

uint32_t CSCBankGroupAccumulator::selectVictim() const
{
    uint32_t victim = accumulator_.size();
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (uint32_t slot = 0; slot < accumulator_.size(); ++slot) {
        const auto& entry = accumulator_[slot];
        if (entry.valid && !entry.reserved &&
            (entry.insertion_age < oldest ||
             (entry.insertion_age == oldest && slot < victim))) {
            oldest = entry.insertion_age;
            victim = slot;
        }
    }
    return victim;
}

uint32_t CSCBankGroupAccumulator::selectDrainEntry() const
{
    uint32_t victim = accumulator_.size();
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (uint32_t slot = 0; slot < accumulator_.size(); ++slot) {
        const auto& entry = accumulator_[slot];
        if (entry.valid && !entry.reserved &&
            (entry.insertion_age < oldest ||
             (entry.insertion_age == oldest && slot < victim))) {
            oldest = entry.insertion_age;
            victim = slot;
        }
    }
    return victim;
}

uint64_t CSCBankGroupAccumulator::pushOutput(uint32_t slot,
                                             CSCBGAOutputReason reason,
                                             uint64_t epoch)
{
    auto& entry = accumulator_.at(slot);
    if (!entry.valid || entry.reserved)
        invariantError("BGA output invalid/reserved accumulator slot");
    if (output_sequence_ == kMax)
        throw std::overflow_error("BGA exhausted output sequence");
    const uint64_t sequence = output_sequence_ + 1;
    OutputRecord record;
    record.payload = {entry.row_idx, entry.value, entry.source_stream_id,
                      entry.generation, epoch, sequence,
                      entry.contribution_count, reason};
    record.reserved_slot = slot;
    output_queue_.push_back(record);
    entry.reserved = true;
    output_sequence_ = sequence;
    return sequence;
}

void CSCBankGroupAccumulator::insertAt(uint32_t slot, const InputEntry& input)
{
    auto& entry = accumulator_.at(slot);
    if (entry.valid || entry.reserved)
        invariantError("BGA insert target occupied/reserved");
    entry.valid = true;
    entry.row_idx = input.row_idx;
    entry.value = input.value;
    entry.insertion_age = insertion_age_ + 1;
    entry.generation = input.generation;
    entry.slot = slot;
    entry.source_stream_id = input.stream;
    entry.contribution_count = 1;
    insertion_age_ = entry.insertion_age;
    saturatingIncrement(counters_.inserts);
}

bool CSCBankGroupAccumulator::allProducersDone() const
{
    return std::all_of(streams_.begin(), streams_.end(),
                       [](const StreamState& stream) { return stream.done; });
}

bool CSCBankGroupAccumulator::hasLiveWork() const
{
    if (operation_ || pending_miss_ || !output_queue_.empty() ||
        output_retirement_pending_)
        return true;
    for (const auto& stream : streams_)
        if (!stream.queue.empty() || stream.presented_batch ||
            stream.accepted_pending)
            return true;
    for (const auto& entry : accumulator_)
        if (entry.valid) return true;
    return false;
}

bool CSCBankGroupAccumulator::quiescent() const
{
    return !hasLiveWork();
}

bool CSCBankGroupAccumulator::finalDrainComplete() const
{
    return final_drain_requested_ && allProducersDone() && quiescent() &&
           counters_.accepted_valid_partials ==
               checkedSum(counters_.retired_output_contributions,
                          counters_.aborted_contributions,
                          "completed contribution sum");
}

size_t CSCBankGroupAccumulator::inputQueueSize(uint32_t stream) const
{
    if (stream >= streams_.size()) throw std::out_of_range("BGA stream");
    return streams_[stream].queue.size();
}

bool CSCBankGroupAccumulator::hasAcceptedPending(uint32_t stream) const
{
    if (stream >= streams_.size()) throw std::out_of_range("BGA stream");
    return streams_[stream].accepted_pending.has_value();
}

uint64_t CSCBankGroupAccumulator::lastAcceptedSequence(uint32_t stream) const
{
    if (stream >= streams_.size()) throw std::out_of_range("BGA stream");
    return streams_[stream].last_accepted_sequence;
}

bool CSCBankGroupAccumulator::validateAccumulatorInvariant() const
{
    for (uint32_t i = 0; i < accumulator_.size(); ++i) {
        const auto& entry = accumulator_[i];
        if (entry.slot != i || (entry.reserved && !entry.valid)) return false;
        if (entry.valid && !entry.reserved)
            for (uint32_t j = i + 1; j < accumulator_.size(); ++j)
                if (accumulator_[j].valid && !accumulator_[j].reserved &&
                    accumulator_[j].row_idx == entry.row_idx)
                    return false;
    }
    for (uint32_t slot = 0; slot < accumulator_.size(); ++slot) {
        uint32_t owners = 0;
        for (const auto& output : output_queue_)
            if (output.reserved_slot == slot) ++owners;
        if (accumulator_[slot].reserved != (owners == 1)) return false;
    }
    return true;
}

uint64_t CSCBankGroupAccumulator::liveContributionCount() const
{
    uint64_t count = 0;
    for (const auto& stream : streams_) {
        count = checkedSum(count, stream.queue.size(), "live contribution sum");
        if (stream.accepted_pending)
            count = checkedSum(count, stream.accepted_pending->partials.size(),
                               "live contribution sum");
    }
    if (operation_) count = checkedSum(count, 1, "live contribution sum");
    if (pending_miss_) count = checkedSum(count, 1, "live contribution sum");
    for (const auto& entry : accumulator_)
        if (entry.valid && !entry.reserved)
            count = checkedSum(count, entry.contribution_count,
                               "live contribution sum");
    for (const auto& output : output_queue_)
        count = checkedSum(count, output.payload.contribution_count,
                           "live contribution sum");
    return count;
}

bool CSCBankGroupAccumulator::conservationInvariant() const
{
    const uint64_t completed =
        checkedSum(counters_.retired_output_contributions,
                   counters_.aborted_contributions,
                   "completed contribution sum");
    return counters_.accepted_valid_partials ==
           checkedSum(liveContributionCount(), completed,
                      "conservation contribution sum");
}

}  // namespace csc_descriptor
