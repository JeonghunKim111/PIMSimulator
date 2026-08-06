#include "csc/CSCFp16BankGroupAccumulator.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace csc_descriptor {
namespace {
void addLE64(uint64_t value, uint64_t& hash)
{
    for (uint32_t byte = 0; byte < 8; ++byte) {
        hash ^= static_cast<uint8_t>(value >> (byte * 8));
        hash *= 1099511628211ULL;
    }
}
}

bool CSCFp16PartialBatch::operator==(const CSCFp16PartialBatch& other) const
{
    if (valid_count != other.valid_count || global_bg_id != other.global_bg_id ||
        descriptor_id != other.descriptor_id || chunk_id != other.chunk_id ||
        batch_id != other.batch_id) return false;
    for (uint32_t i = 0; i < valid_count; ++i)
        if (!(entries[i] == other.entries[i])) return false;
    return true;
}

CSCFp16BGAConfig makeFp16SerialCompatibilityConfig(uint32_t rows)
{
    CSCFp16BGAConfig config;
    config.rows = rows;
    return config;
}

CSCFp16BGAConfig makeFp16IsoStructureProductionConfig(uint32_t rows)
{
    CSCFp16BGAConfig config;
    config.ingress_mode = CSCFp16BGAIngressMode::BATCH8;
    config.ingress_batch_width = 8;
    config.rows = rows;
    return config;
}

CSCFp16BGAConfig makeFp16Q8StressConfig(uint32_t rows)
{
    auto config = makeFp16IsoStructureProductionConfig(rows);
    config.accumulator_entries = 8;
    config.compare_width = 8;
    return config;
}
bool CSCFp16BGAOutputEvent::operator==(const CSCFp16BGAOutputEvent& other) const
{
    return row_idx == other.row_idx && value_bits == other.value_bits &&
           global_bg_id == other.global_bg_id && reason == other.reason &&
           sequence == other.sequence && contribution_count == other.contribution_count;
}

CSCFp16BoundedBGAOutputSink::CSCFp16BoundedBGAOutputSink(std::size_t capacity)
    : capacity_(capacity)
{
    if (!capacity_) throw std::invalid_argument("FP16 BGA output capacity is zero");
}

bool CSCFp16BoundedBGAOutputSink::ready() const
{
    return enabled_ && queue_.size() < capacity_;
}

void CSCFp16BoundedBGAOutputSink::accept(const CSCFp16BGAOutputEvent& event)
{
    if (!ready()) throw std::logic_error("FP16 BGA output accept without ready");
    queue_.push_back(event);
    trace_.push_back(event);
}

CSCFp16BGAOutputEvent CSCFp16BoundedBGAOutputSink::pop()
{
    if (queue_.empty()) throw std::logic_error("FP16 BGA output pop empty");
    auto event = queue_.front();
    queue_.pop_front();
    return event;
}

CSCFp16BankGroupAccumulator::CSCFp16BankGroupAccumulator(
    uint32_t global_bg_id, const CSCFp16BGAConfig& config)
    : global_bg_id_(global_bg_id), config_(config),
      accumulator_(config.accumulator_entries)
{
    if (global_bg_id_ >= 64) throw std::invalid_argument("FP16 BGA BG out of range");
    validateConfig();
}

void CSCFp16BankGroupAccumulator::validateConfig() const
{
    if (!config_.input_queue_depth || !config_.accumulator_entries ||
        !config_.output_queue_depth || !config_.rows || !config_.compare_latency ||
        !config_.add_latency || config_.compare_width < config_.accumulator_entries)
        throw std::invalid_argument("invalid FP16 BGA configuration");
    const uint32_t expected = config_.ingress_mode == CSCFp16BGAIngressMode::BATCH8
                                  ? 8 : 1;
    if (config_.ingress_batch_width != expected)
        throw std::invalid_argument("FP16 BGA ingress mode/width mismatch");
}

bool CSCFp16BankGroupAccumulator::ready() const
{
    return !producer_done_ && !accepted_pending_ &&
           input_queue_.size() < config_.input_queue_depth;
}

void CSCFp16BankGroupAccumulator::accept(const CSCFp16PartialEvent& event)
{
    counters_.ingress_attempts++;
    if (!ready()) {
        counters_.ingress_stalls++;
        throw std::logic_error("FP16 BGA ingress accept without ready");
    }
    if (event.global_bg_id != global_bg_id_ || event.row_idx >= config_.rows)
        throw std::invalid_argument("FP16 BGA ingress identity or row mismatch");
    CSCFp16PartialBatch batch;
    batch.entries[0] = event;
    batch.valid_count = 1;
    batch.global_bg_id = event.global_bg_id;
    batch.descriptor_id = event.descriptor_id;
    batch.chunk_id = event.chunk_id;
    batch.batch_id = event.lane_id >= 8;
    if (!acceptBatch(batch))
        throw std::logic_error("FP16 serial ingress lost ready reservation");
}

bool CSCFp16BankGroupAccumulator::canAcceptBatch(
    const CSCFp16PartialBatch& batch) const
{
    if (producer_done_ || accepted_pending_ || !batch.valid_count ||
        batch.valid_count > config_.ingress_batch_width ||
        input_queue_.size() + batch.valid_count > config_.input_queue_depth)
        return false;
    if (batch.global_bg_id != global_bg_id_ || batch.batch_id > 1) return false;
    for (uint32_t i = 0; i < batch.valid_count; ++i) {
        const auto& event = batch.entries[i];
        if (event.global_bg_id != global_bg_id_ || event.row_idx >= config_.rows ||
            event.descriptor_id != batch.descriptor_id ||
            event.chunk_id != batch.chunk_id ||
            (config_.ingress_mode == CSCFp16BGAIngressMode::BATCH8 &&
             event.lane_id != uint32_t(batch.batch_id) * 8 + i))
            return false;
    }
    return true;
}

bool CSCFp16BankGroupAccumulator::acceptBatch(const CSCFp16PartialBatch& batch)
{
    counters_.batch_attempts++;
    if (!canAcceptBatch(batch)) {
        counters_.batches_stalled++;
        return false;
    }
    accepted_pending_ = batch;
    counters_.batches_accepted++;
    counters_.ingress_accepted += batch.valid_count;
    if (batch.batch_id) counters_.batch1_count++;
    else counters_.batch0_count++;
    return true;
}

void CSCFp16BankGroupAccumulator::markProducerDone()
{
    producer_done_ = true;
}

bool CSCFp16BankGroupAccumulator::requestFinalDrain()
{
    if (!producer_done_) return false;
    final_drain_requested_ = true;
    return true;
}

void CSCFp16BankGroupAccumulator::step()
{
    if (cycle_ == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("FP16 BGA cycle exhausted");
    const bool busy = liveWork();
    ++cycle_;
    operation_committed_this_cycle_ = false;
    if (output_retirement_pending_) retireOutput();
    if (operation_ && !--operation_->remaining) commitOperation();
    updateEvictionOrDrain();
    startService();
    commitIngress();
    if (busy || liveWork()) counters_.cycles_busy++;
    else counters_.cycles_idle++;
    counters_.queue_high_water = std::max<uint64_t>(counters_.queue_high_water,
                                                    occupancy());
    counters_.input_high_water = std::max<uint64_t>(counters_.input_high_water,
                                                    input_queue_.size());
    if (!conservationInvariant()) throw std::logic_error("FP16 BGA conservation failure");
}

void CSCFp16BankGroupAccumulator::commitOperation()
{
    const auto completed = *operation_;
    operation_.reset();
    operation_committed_this_cycle_ = true;
    if (completed.kind == OperationKind::LOOKUP) {
        uint32_t hit = accumulator_.size(), hits = 0, free = accumulator_.size();
        for (uint32_t slot = 0; slot < accumulator_.size(); ++slot) {
            const auto& entry = accumulator_[slot];
            if (entry.valid && !entry.reserved) {
                counters_.tag_comparisons++;
                if (entry.row_idx == completed.input.row_idx) {
                    hit = slot;
                    ++hits;
                }
            } else if (!entry.valid && free == accumulator_.size()) free = slot;
        }
        if (hits > 1) throw std::logic_error("duplicate FP16 BGA row tag");
        if (hits) {
            counters_.lookup_hits++;
            operation_ = Operation{OperationKind::MERGE, completed.input, hit,
                                   config_.add_latency};
        } else {
            counters_.lookup_misses++;
            if (free != accumulator_.size())
                operation_ = Operation{OperationKind::INSERT, completed.input, free,
                                       config_.add_latency};
            else
                pending_miss_ = PendingMiss{completed.input, selectOldest(), 0, false};
        }
        return;
    }
    if (completed.kind == OperationKind::MERGE) {
        auto& entry = accumulator_.at(completed.slot);
        if (!entry.valid || entry.reserved || entry.row_idx != completed.input.row_idx)
            throw std::logic_error("FP16 BGA merge target changed");
        entry.value_bits = cscFp16ToBits(cscFp16Add(
            cscFp16FromBits(entry.value_bits),
            cscFp16FromBits(completed.input.value_bits)));
        entry.contributions++;
        counters_.fp16_adds++;
        counters_.merges++;
        return;
    }
    insertAt(completed.slot, completed.input);
}

void CSCFp16BankGroupAccumulator::updateEvictionOrDrain()
{
    if (pending_miss_) {
        if (!pending_miss_->output_sequence) {
            if (outputs_.size() >= config_.output_queue_depth) {
                counters_.output_stalls++;
                return;
            }
            pending_miss_->output_sequence = pushOutput(
                pending_miss_->victim, CSCFp16BGAOutputReason::CAPACITY_EVICTION);
            pending_miss_->awaiting_retirement = true;
            counters_.capacity_evictions++;
            return;
        }
        if (pending_miss_->awaiting_retirement) return;
        if (!operation_) {
            operation_ = Operation{OperationKind::INSERT, pending_miss_->input,
                                   pending_miss_->victim, config_.add_latency};
            pending_miss_.reset();
        }
        return;
    }
    if (!final_drain_requested_ || operation_ || operation_committed_this_cycle_ ||
        !input_queue_.empty() || accepted_pending_) return;
    const uint32_t slot = selectOldest();
    if (slot == accumulator_.size()) return;
    if (outputs_.size() >= config_.output_queue_depth) {
        counters_.output_stalls++;
        return;
    }
    pushOutput(slot, CSCFp16BGAOutputReason::FINAL_DRAIN);
    counters_.final_drain_outputs++;
}

void CSCFp16BankGroupAccumulator::startService()
{
    if (operation_ || pending_miss_ || input_queue_.empty()) return;
    const auto input = input_queue_.front();
    input_queue_.pop_front();
    counters_.partials_serviced++;
    operation_ = Operation{OperationKind::LOOKUP, input, 0, config_.compare_latency};
}

void CSCFp16BankGroupAccumulator::commitIngress()
{
    if (!accepted_pending_) return;
    if (input_queue_.size() + accepted_pending_->valid_count >
        config_.input_queue_depth)
        throw std::logic_error("FP16 BGA lost accepted ingress reservation");
    for (uint32_t i = 0; i < accepted_pending_->valid_count; ++i)
        input_queue_.push_back(accepted_pending_->entries[i]);
    accepted_pending_.reset();
}

uint32_t CSCFp16BankGroupAccumulator::selectOldest() const
{
    uint32_t victim = accumulator_.size();
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (uint32_t slot = 0; slot < accumulator_.size(); ++slot) {
        const auto& entry = accumulator_[slot];
        if (entry.valid && !entry.reserved &&
            (entry.age < oldest || (entry.age == oldest && slot < victim))) {
            victim = slot;
            oldest = entry.age;
        }
    }
    return victim;
}

uint64_t CSCFp16BankGroupAccumulator::pushOutput(
    uint32_t slot, CSCFp16BGAOutputReason reason)
{
    auto& entry = accumulator_.at(slot);
    if (!entry.valid || entry.reserved) throw std::logic_error("invalid FP16 BGA output");
    OutputRecord output;
    output.slot = slot;
    output.event = {entry.row_idx, entry.value_bits, global_bg_id_, reason,
                    ++next_output_sequence_, entry.contributions};
    outputs_.push_back(output);
    entry.reserved = true;
    return next_output_sequence_;
}

void CSCFp16BankGroupAccumulator::insertAt(
    uint32_t slot, const CSCFp16PartialEvent& input)
{
    auto& entry = accumulator_.at(slot);
    if (entry.valid || entry.reserved) throw std::logic_error("occupied FP16 BGA insert");
    entry = {true, false, input.row_idx, input.value_bits, ++next_age_, 1};
    counters_.inserts++;
}

const CSCFp16BGAOutputEvent& CSCFp16BankGroupAccumulator::peekOutput() const
{
    if (outputs_.empty()) throw std::logic_error("FP16 BGA output empty");
    return outputs_.front().event;
}

void CSCFp16BankGroupAccumulator::acceptOutput()
{
    counters_.output_attempts++;
    if (outputs_.empty() || output_retirement_pending_)
        throw std::logic_error("invalid FP16 BGA output acceptance");
    output_retirement_pending_ = true;
    counters_.output_accepted++;
}

void CSCFp16BankGroupAccumulator::retireOutput()
{
    const auto output = outputs_.front();
    auto& entry = accumulator_.at(output.slot);
    if (!entry.valid || !entry.reserved) throw std::logic_error("lost FP16 BGA reserved output");
    counters_.retired_contributions += output.event.contribution_count;
    entry = {};
    outputs_.pop_front();
    output_retirement_pending_ = false;
    if (pending_miss_ && pending_miss_->awaiting_retirement &&
        pending_miss_->output_sequence == output.event.sequence)
        pending_miss_->awaiting_retirement = false;
}

uint32_t CSCFp16BankGroupAccumulator::occupancy() const
{
    return std::count_if(accumulator_.begin(), accumulator_.end(),
                         [](const Entry& entry) { return entry.valid; });
}

bool CSCFp16BankGroupAccumulator::liveWork() const
{
    return accepted_pending_ || !input_queue_.empty() || operation_ || pending_miss_ ||
           !outputs_.empty() || output_retirement_pending_ || occupancy();
}

bool CSCFp16BankGroupAccumulator::quiescent() const { return !liveWork(); }

bool CSCFp16BankGroupAccumulator::finalDrainComplete() const
{
    return final_drain_requested_ && producer_done_ && quiescent() &&
           counters_.ingress_accepted == counters_.retired_contributions;
}

uint64_t CSCFp16BankGroupAccumulator::liveContributions() const
{
    uint64_t count = input_queue_.size() +
                     (accepted_pending_ ? accepted_pending_->valid_count : 0) +
                     (operation_ ? 1 : 0) + (pending_miss_ ? 1 : 0);
    for (const auto& entry : accumulator_)
        if (entry.valid && !entry.reserved) count += entry.contributions;
    for (const auto& output : outputs_) count += output.event.contribution_count;
    return count;
}

bool CSCFp16BankGroupAccumulator::conservationInvariant() const
{
    return counters_.ingress_accepted ==
           liveContributions() + counters_.retired_contributions;
}

uint64_t cscFp16BGAOutputTraceFnv1a64(
    const std::vector<CSCFp16BGAOutputEvent>& trace)
{
    uint64_t hash = 1469598103934665603ULL;
    for (const auto& event : trace) {
        addLE64(event.row_idx, hash);
        addLE64(event.value_bits, hash);
        addLE64(event.global_bg_id, hash);
        addLE64(static_cast<uint64_t>(event.reason), hash);
        addLE64(event.sequence, hash);
        addLE64(event.contribution_count, hash);
    }
    return hash;
}

}  // namespace csc_descriptor
