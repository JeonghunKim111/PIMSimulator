#include "csc/CSCFp16NativeExecution.h"

#include "Callback.h"
#include "MultiChannelMemorySystem.h"
#include "SystemConfiguration.h"
#include "tests/KernelAddrGen.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace csc_descriptor {
namespace {

std::size_t kindIndex(CSCFp16RequestKind kind)
{
    return static_cast<std::size_t>(kind);
}

DRAMSim::RequestKind tokenKind(CSCFp16RequestKind kind)
{
    switch (kind) {
    case CSCFp16RequestKind::X: return DRAMSim::RequestKind::X_READ;
    case CSCFp16RequestKind::VALUE: return DRAMSim::RequestKind::VALUE_READ;
    case CSCFp16RequestKind::INDEX_LOW:
    case CSCFp16RequestKind::INDEX_HIGH:
        return DRAMSim::RequestKind::INDEX_READ;
    }
    return DRAMSim::RequestKind::NONE;
}

void addLE64(uint64_t value, uint64_t& hash)
{
    for (uint32_t byte = 0; byte < 8; ++byte) {
        hash ^= static_cast<uint8_t>(value >> (byte * 8));
        hash *= 1099511628211ULL;
    }
}

}  // namespace

struct CSCFp16NativeExecution::Impl {
    std::shared_ptr<DRAMSim::MultiChannelMemorySystem> memory;
    std::unique_ptr<PIMAddrManager> addresses;
    DRAMSim::TransactionCompleteCB* read_callback = nullptr;
    DRAMSim::TransactionCompleteCB* write_callback = nullptr;
    DRAMSim::TokenCompleteCB* token_read_callback = nullptr;
    DRAMSim::TokenCompleteCB* token_write_callback = nullptr;
    DRAMSim::BurstType burst{};
};

class CSCFp16NativeExecution::BGAIngressAdapter : public CSCFp16PartialSink {
  public:
    BGAIngressAdapter(CSCFp16BankGroupAccumulator* bga,
                      CSCFp16BGAIngressMode mode)
        : bga_(bga), mode_(mode) {}
    bool ready() const override
    {
        return mode_ == CSCFp16BGAIngressMode::SERIAL_EVENT
                   ? bga_->ready() : !batch_complete_;
    }
    void accept(const CSCFp16PartialEvent& event) override
    {
        if (!ready()) throw std::logic_error("FP16 BGA adapter accept without ready");
        trace_.push_back(event);
        if (mode_ == CSCFp16BGAIngressMode::SERIAL_EVENT) {
            bga_->accept(event);
            return;
        }
        const uint8_t batch_id = event.lane_id >= 8;
        const uint8_t first_lane = batch_id * 8;
        const uint8_t expected = batch_id ? event.chunk_valid_count - 8
                                          : std::min<uint8_t>(event.chunk_valid_count, 8);
        if (!expected || expected > 8 || event.lane_id != first_lane + batch_.valid_count)
            throw std::logic_error("invalid FP16 batch8 lane order");
        if (!batch_.valid_count) {
            batch_.global_bg_id = event.global_bg_id;
            batch_.descriptor_id = event.descriptor_id;
            batch_.chunk_id = event.chunk_id;
            batch_.batch_id = batch_id;
        } else if (batch_.global_bg_id != event.global_bg_id ||
                   batch_.descriptor_id != event.descriptor_id ||
                   batch_.chunk_id != event.chunk_id || batch_.batch_id != batch_id) {
            throw std::logic_error("FP16 batch8 identity changed while staging");
        }
        batch_.entries[batch_.valid_count++] = event;
        if (batch_.valid_count == expected) {
            batch_complete_ = true;
            counters_.generated_batches++;
            counters_.generated_partials += batch_.valid_count;
            if (batch_id) counters_.batch1_count++;
            else counters_.batch0_count++;
        }
    }
    void flush()
    {
        if (mode_ == CSCFp16BGAIngressMode::SERIAL_EVENT || !batch_complete_) return;
        counters_.issue_attempts++;
        if (!bga_->acceptBatch(batch_)) {
            counters_.stalled_attempts++;
            return;
        }
        counters_.accepted_batches++;
        counters_.accepted_partials += batch_.valid_count;
        batch_ = {};
        batch_complete_ = false;
    }
    bool empty() const { return !batch_.valid_count && !batch_complete_; }
    const std::vector<CSCFp16PartialEvent>& trace() const { return trace_; }
    const CSCFp16NativeCounters::BatchAdapter& counters() const { return counters_; }
  private:
    CSCFp16BankGroupAccumulator* bga_;
    CSCFp16BGAIngressMode mode_;
    CSCFp16PartialBatch batch_{};
    bool batch_complete_ = false;
    CSCFp16NativeCounters::BatchAdapter counters_{};
    std::vector<CSCFp16PartialEvent> trace_;
};

CSCFp16NativeExecution::CSCFp16NativeExecution(
    std::shared_ptr<const CSCFp16ExecutionImage> image,
    std::size_t sink_capacity_per_bg, const CSCFp16BGAConfig* bga_config,
    std::size_t bga_output_capacity_per_bg,
    const CSCFp16TransportConfig* transport_config,
    uint32_t validation_output_accepts_per_cycle)
    : image_(std::move(image)), impl_(new Impl), bga_enabled_(bga_config != nullptr),
      validation_output_accepts_per_cycle_(validation_output_accepts_per_cycle)
{
    if (!image_ || !sink_capacity_per_bg ||
        (bga_enabled_ && !bga_output_capacity_per_bg))
        throw std::invalid_argument("invalid FP16 native execution construction");
    if (!validation_output_accepts_per_cycle_ ||
        validation_output_accepts_per_cycle_ > 64)
        throw std::invalid_argument(
            "FP16 validation output accepts per cycle must be in [1,64]");
    if (transport_config && !bga_enabled_)
        throw std::invalid_argument("FP16 transport requires BGA");
    if (transport_config)
        transport_.reset(new CSCFp16PartialResultPath(
            *transport_config, image_->image().matrix.rows));
    impl_->memory = std::make_shared<DRAMSim::MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_csc_fp32.ini", ".",
        "csc_fp16_native_m45", 256 * 16);
    if (getConfigParam(UINT, "NUM_CHANS") != 16 ||
        getConfigParam(UINT, "NUM_RANKS") != 1 ||
        getConfigParam(UINT, "NUM_BANK_GROUPS") != 4 ||
        DRAMSim::PIMConfiguration::getAddressMappingScheme() != DRAMSim::Scheme8)
        throw std::runtime_error("FP16 native execution requires 16ch/1rank/4BG/Scheme8");
    impl_->addresses.reset(new PIMAddrManager(16, 1));
    impl_->read_callback =
        new DRAMSim::Callback<CSCFp16NativeExecution, void, unsigned, uint64_t,
                              uint64_t>(this, &CSCFp16NativeExecution::legacyComplete);
    impl_->write_callback =
        new DRAMSim::Callback<CSCFp16NativeExecution, void, unsigned, uint64_t,
                              uint64_t>(this, &CSCFp16NativeExecution::legacyComplete);
    impl_->token_read_callback =
        new DRAMSim::Callback<CSCFp16NativeExecution, void, unsigned,
                              const DRAMSim::RequestToken&, uint64_t>(
            this, &CSCFp16NativeExecution::tokenComplete);
    impl_->token_write_callback =
        new DRAMSim::Callback<CSCFp16NativeExecution, void, unsigned,
                              const DRAMSim::RequestToken&, uint64_t>(
            this, &CSCFp16NativeExecution::tokenComplete);
    impl_->memory->RegisterCallbacks(impl_->read_callback, impl_->write_callback,
                                     nullptr);
    impl_->memory->RegisterTokenCallbacks(impl_->token_read_callback,
                                          impl_->token_write_callback);

    for (uint32_t bg = 0; bg < 64; ++bg) {
        datapaths_.emplace_back(new DRAMSim::PIMBlock(DRAMSim::FP16));
        CSCFp16PartialSink* ingress = nullptr;
        if (bga_enabled_) {
            bgas_.emplace_back(new CSCFp16BankGroupAccumulator(bg, *bga_config));
            bga_output_sinks_.emplace_back(
                new CSCFp16BoundedBGAOutputSink(bga_output_capacity_per_bg));
            bga_ingress_adapters_.emplace_back(
                new BGAIngressAdapter(bgas_.back().get(), bga_config->ingress_mode));
            ingress = bga_ingress_adapters_.back().get();
        } else {
            sinks_.emplace_back(new CSCFp16BoundedCaptureSink(sink_capacity_per_bg));
            ingress = sinks_.back().get();
        }
        engines_.emplace_back(new CSCFp16DescriptorEngine(
            bg, image_, datapaths_.back().get(),
            [this](const CSCFp16Request& request) { return submit(request); },
            ingress));
    }
}

CSCFp16NativeExecution::~CSCFp16NativeExecution()
{
    delete impl_->read_callback;
    delete impl_->write_callback;
    delete impl_->token_read_callback;
    delete impl_->token_write_callback;
}

void CSCFp16NativeExecution::launch()
{
    if (launched_) throw std::logic_error("FP16 native execution already launched");
    timing_ = {};
    timing_.launch_cycle = cycle_;
    bga_done_signaled_ = {};
    for (auto& engine : engines_) engine->launch();
    launched_ = true;
}

uint64_t CSCFp16NativeExecution::physicalAddress(
    const CSCFp16Request& request) const
{
    if (request.global_bg_id >= 64 || request.stream_offset_bytes % 32)
        throw std::invalid_argument("unaligned or invalid FP16 native request");
    uint32_t bank = 0;
    unsigned row = 0;
    if (request.kind == CSCFp16RequestKind::INDEX_LOW ||
        request.kind == CSCFp16RequestKind::INDEX_HIGH) {
        bank = 1;
        row = 4096;
    } else if (request.kind == CSCFp16RequestKind::X) {
        bank = 2;
        row = 8192;
    }
    unsigned column = request.stream_offset_bytes / 32;
    return impl_->addresses->addrGenSafe(request.global_bg_id / 4, 0,
                                         request.global_bg_id % 4, bank, row,
                                         column);
}

bool CSCFp16NativeExecution::submit(const CSCFp16Request& request)
{
    timing_.request_attempts++;
    if (!timing_.first_request_attempt_cycle)
        timing_.first_request_attempt_cycle = cycle_;
    const auto inserted = first_attempt_cycles_.emplace(request.request_id, cycle_);
    if (!inserted.second) timing_.retry_attempts++;
    if (reject_budget_.at(request.global_bg_id)) {
        reject_budget_[request.global_bg_id]--;
        timing_.rejected_requests++;
        timing_.request_queue_stall_cycles++;
        return false;
    }
    if (outstanding_.count(request.request_id)) {
        latchFailure("duplicate FP16 native request submission");
        return false;
    }
    const uint64_t address = physicalAddress(request);
    DRAMSim::RequestToken token;
    token.request_id = request.request_id;
    token.request_kind = tokenKind(request.kind);
    token.global_bg_id = request.global_bg_id;
    token.worker_id = request.global_bg_id;
    token.sequence_number = request.request_id;
    token.client_id = 2;
    token.generation = 1;
    if (!impl_->memory->addTransaction(false, address, "CSC_FP16_NATIVE_M45",
                                       &impl_->burst, token)) {
        timing_.rejected_requests++;
        timing_.request_queue_stall_cycles++;
        return false;
    }
    Outstanding accepted;
    accepted.request = request;
    accepted.physical_address = address;
    accepted.first_attempt_cycle = first_attempt_cycles_.at(request.request_id);
    accepted.accepted_cycle = cycle_;
    outstanding_.emplace(request.request_id, accepted);
    timing_.accepted_requests++;
    timing_.accepted_by_kind[kindIndex(request.kind)]++;
    if (!timing_.first_accepted_request_cycle)
        timing_.first_accepted_request_cycle = cycle_;
    timing_.last_request_issue_cycle = cycle_;
    timing_.last_request_accept_cycle = cycle_;
    timing_.outstanding_high_water =
        std::max<uint64_t>(timing_.outstanding_high_water, outstanding_.size());
    return true;
}

std::array<uint8_t, 32> CSCFp16NativeExecution::payloadFor(
    const Outstanding& outstanding) const
{
    const auto& request = outstanding.request;
    if (physicalAddress(request) != outstanding.physical_address)
        throw std::runtime_error("FP16 native completion address mismatch");
    std::array<uint8_t, 32> payload{};
    if (request.kind == CSCFp16RequestKind::X) {
        const uint64_t first = request.stream_offset_bytes / 2;
        for (uint32_t lane = 0; lane < 16; ++lane) {
            const uint64_t column = first + lane;
            const uint16_t bits = column < image_->image().x_bits.size()
                                      ? image_->image().x_bits[column] : 0;
            payload[lane * 2] = static_cast<uint8_t>(bits);
            payload[lane * 2 + 1] = static_cast<uint8_t>(bits >> 8);
        }
        return payload;
    }
    const auto& bg = image_->image().bg.at(request.global_bg_id);
    const auto& bytes = request.kind == CSCFp16RequestKind::VALUE
                            ? bg.values : bg.row_indices;
    if (request.stream_offset_bytes > bytes.size() ||
        bytes.size() - request.stream_offset_bytes < payload.size())
        throw std::runtime_error("truncated FP16 native payload");
    std::copy_n(bytes.begin() + request.stream_offset_bytes,
                payload.size(), payload.begin());
    return payload;
}

void CSCFp16NativeExecution::tokenComplete(
    unsigned channel, const DRAMSim::RequestToken& token, uint64_t when)
{
    auto found = outstanding_.find(token.request_id);
    if (found == outstanding_.end()) {
        latchFailure("unknown or duplicate FP16 native completion");
        return;
    }
    const Outstanding outstanding = found->second;
    if (token.global_bg_id != outstanding.request.global_bg_id ||
        channel != outstanding.request.global_bg_id / 4 ||
        token.request_kind != tokenKind(outstanding.request.kind)) {
        latchFailure("FP16 native completion identity mismatch");
        return;
    }
    try {
        const auto payload = payloadFor(outstanding);
        if (!engines_[outstanding.request.global_bg_id]->complete(
                outstanding.request, payload)) {
            latchFailure(engines_[outstanding.request.global_bg_id]->error());
            return;
        }
    } catch (const std::exception& error) {
        latchFailure(error.what());
        return;
    }
    outstanding_.erase(found);
    timing_.completed_requests++;
    timing_.completed_by_kind[kindIndex(outstanding.request.kind)]++;
    if (!timing_.first_response_cycle) timing_.first_response_cycle = when;
    timing_.last_response_cycle = when;
    const uint64_t latency = when - outstanding.accepted_cycle;
    timing_.response_latency_sum += latency;
    if (timing_.completed_requests == 1)
        timing_.response_latency_min = timing_.response_latency_max = latency;
    else {
        timing_.response_latency_min = std::min(timing_.response_latency_min, latency);
        timing_.response_latency_max = std::max(timing_.response_latency_max, latency);
    }
    request_trace_.push_back({outstanding.request, outstanding.physical_address,
                              outstanding.first_attempt_cycle,
                              outstanding.accepted_cycle, when});
}

void CSCFp16NativeExecution::latchFailure(const std::string& message)
{
    if (failed_) return;
    failed_ = true;
    error_ = message;
}

void CSCFp16NativeExecution::tick()
{
    if (!launched_ || done() || failed_) return;
    impl_->memory->update();
    cycle_++;
    if (bga_enabled_) {
        for (uint32_t bg = 0; bg < bgas_.size(); ++bg) {
            bga_ingress_adapters_[bg]->flush();
            const auto before = bgas_[bg]->counters();
            bgas_[bg]->step();
            const auto& after = bgas_[bg]->counters();
            if (after.tag_comparisons > before.tag_comparisons) {
                if (!timing_.first_bga_compare_cycle)
                    timing_.first_bga_compare_cycle = cycle_;
            }
            if (after.fp16_adds > before.fp16_adds) {
                if (!timing_.first_fp16_add_cycle) timing_.first_fp16_add_cycle = cycle_;
            }
            if (after.capacity_evictions + after.final_drain_outputs >
                before.capacity_evictions + before.final_drain_outputs) {
                if (!timing_.first_bga_output_generated_cycle)
                    timing_.first_bga_output_generated_cycle = cycle_;
                timing_.last_bga_output_generated_cycle = cycle_;
            }
        }
    }
    bool any_bga_ingress_stall = false;
    for (uint32_t bg = 0; bg < engines_.size() && !failed_; ++bg) {
        auto& engine = *engines_[bg];
        const auto before = engine.counters();
        const auto before_state = engine.state();
        engine.tick();
        if (engine.failed()) {
            latchFailure(engine.error());
            break;
        }
        const auto& after = engine.counters();
        if (bga_enabled_ && after.sink_backpressure_cycles >
                                before.sink_backpressure_cycles)
            any_bga_ingress_stall = true;
        if (after.descriptor_count > before.descriptor_count) {
            if (!timing_.first_descriptor_cycle)
                timing_.first_descriptor_cycle = cycle_;
            timing_.last_descriptor_cycle = cycle_;
        }
        if (before_state == CSCFp16EngineState::WAIT_OPERANDS &&
            engine.state() == CSCFp16EngineState::SIMD_MUL &&
            !timing_.first_complete_operands_cycle)
            timing_.first_complete_operands_cycle = cycle_;
        if (after.logical_compute_chunks > before.logical_compute_chunks) {
            timing_.pim_execution_cycles++;
            if (!timing_.first_pim_mul_cycle)
                timing_.first_pim_mul_cycle = cycle_;
            timing_.last_pim_mul_cycle = cycle_;
            if (!timing_.first_generated_partial_cycle)
                timing_.first_generated_partial_cycle = cycle_;
            timing_.last_generated_partial_cycle = cycle_;
        }
        if (after.emitted_partials > before.emitted_partials) {
            if (bga_enabled_) {
                if (!timing_.first_bga_ingress_attempt_cycle)
                    timing_.first_bga_ingress_attempt_cycle = cycle_;
                if (!timing_.first_bga_ingress_accept_cycle)
                    timing_.first_bga_ingress_accept_cycle = cycle_;
            }
            if (!timing_.first_accepted_partial_cycle)
                timing_.first_accepted_partial_cycle = cycle_;
            timing_.last_accepted_partial_cycle = cycle_;
        }
        if (bga_enabled_ && engine.done() && !bga_done_signaled_[bg] &&
            bga_ingress_adapters_[bg]->empty()) {
            bgas_[bg]->markProducerDone();
            if (!bgas_[bg]->requestFinalDrain()) {
                latchFailure("FP16 BGA final drain request rejected");
                break;
            }
            bga_done_signaled_[bg] = true;
        }
    }
    if (any_bga_ingress_stall) timing_.bga_ingress_stall_global_cycles++;
    if (bga_enabled_ && !timing_.final_drain_start_cycle &&
        std::all_of(bga_done_signaled_.begin(), bga_done_signaled_.end(),
                    [](bool done) { return done; }))
        timing_.final_drain_start_cycle = cycle_;
    if (allEnginesDone() && outstanding_.empty() && !timing_.compute_complete_cycle) {
        timing_.compute_complete_cycle = cycle_;
        timing_.completion_cycle = cycle_;
        timing_.total_compute_only_cycles = cycle_ - timing_.launch_cycle;
    }
    if (bga_enabled_) {
        uint32_t validation_accepted = 0;
        const uint32_t validation_scan_start = validation_output_next_bg_;
        for (uint32_t scan = 0; scan < bgas_.size(); ++scan) {
            const uint32_t bg = transport_ ? scan :
                (validation_scan_start + scan) % bgas_.size();
            if (!bgas_[bg]->hasOutput()) continue;
            if (transport_) {
                if (!transport_->accept(bgas_[bg]->peekOutput())) {
                    timing_.bga_output_sink_stall_cycles++;
                    continue;
                }
                bgas_[bg]->acceptOutput();
            } else if (!bga_output_sinks_[bg]->ready()) {
                timing_.bga_output_sink_stall_cycles++;
                continue;
            } else {
                if (validation_accepted >= validation_output_accepts_per_cycle_)
                    continue;
                bga_output_sinks_[bg]->accept(bgas_[bg]->peekOutput());
                bgas_[bg]->acceptOutput();
                ++validation_accepted;
                validation_output_next_bg_ = (bg + 1) % bgas_.size();
            }
            if (!timing_.first_bga_output_accepted_cycle)
                timing_.first_bga_output_accepted_cycle = cycle_;
            timing_.last_bga_output_accepted_cycle = cycle_;
        }
        if (transport_) {
            std::vector<bool> drain_complete(bgas_.size());
            for (uint32_t bg=0; bg<bgas_.size(); ++bg)
                drain_complete[bg]=bgas_[bg]->finalDrainComplete();
            transport_->step(cycle_, drain_complete);
            if (transport_->failed()) latchFailure(transport_->error());
        }
        if (allBGAsDone() && !timing_.compute_bga_completion_cycle) {
            timing_.compute_bga_completion_cycle = cycle_;
            timing_.total_compute_bga_cycles = cycle_ - timing_.launch_cycle;
        }
    } else if (allEnginesDone() && outstanding_.empty()) {
        timing_.compute_bga_completion_cycle = timing_.completion_cycle;
    }
}

bool CSCFp16NativeExecution::allEnginesDone() const
{
    for (const auto& engine : engines_)
        if (!engine->done()) return false;
    return true;
}

bool CSCFp16NativeExecution::done() const
{
    return launched_ && !failed_ && timing_.completion_cycle && outstanding_.empty() &&
           allEnginesDone() && (!bga_enabled_ ||
                                (timing_.compute_bga_completion_cycle && allBGAsDone() &&
                                 (!transport_ || transport_->done())));
}

bool CSCFp16NativeExecution::allBGAsDone() const
{
    if (!bga_enabled_) return true;
    for (const auto& bga : bgas_)
        if (!bga->finalDrainComplete()) return false;
    return true;
}

void CSCFp16NativeExecution::setRejectBudget(uint32_t global_bg,
                                             uint64_t attempts)
{
    reject_budget_.at(global_bg) = attempts;
}

void CSCFp16NativeExecution::setSinkEnabled(uint32_t global_bg, bool enabled)
{
    if (bga_enabled_) throw std::logic_error("M5 uses BGA output sink control");
    sinks_.at(global_bg)->setEnabled(enabled);
}

CSCFp16PartialEvent CSCFp16NativeExecution::popCaptured(uint32_t global_bg)
{
    if (bga_enabled_) throw std::logic_error("M5 has no compute capture queue");
    return sinks_.at(global_bg)->pop();
}

const CSCFp16BoundedCaptureSink& CSCFp16NativeExecution::sink(
    uint32_t global_bg) const
{
    if (bga_enabled_) throw std::logic_error("M5 uses BGA ingress trace");
    return *sinks_.at(global_bg);
}

CSCFp16BoundedCaptureSink& CSCFp16NativeExecution::sink(uint32_t global_bg)
{
    if (bga_enabled_) throw std::logic_error("M5 uses BGA ingress trace");
    return *sinks_.at(global_bg);
}

const CSCFp16BankGroupAccumulator& CSCFp16NativeExecution::bga(uint32_t bg) const
{
    if (!bga_enabled_) throw std::logic_error("FP16 native BGA disabled");
    return *bgas_.at(bg);
}

const CSCFp16BoundedBGAOutputSink& CSCFp16NativeExecution::bgaOutputSink(
    uint32_t bg) const
{
    if (!bga_enabled_) throw std::logic_error("FP16 native BGA disabled");
    return *bga_output_sinks_.at(bg);
}

CSCFp16BoundedBGAOutputSink& CSCFp16NativeExecution::bgaOutputSink(uint32_t bg)
{
    if (!bga_enabled_) throw std::logic_error("FP16 native BGA disabled");
    return *bga_output_sinks_.at(bg);
}

CSCFp16BGAOutputEvent CSCFp16NativeExecution::popBGAOutput(uint32_t bg)
{
    return bgaOutputSink(bg).pop();
}

void CSCFp16NativeExecution::setBGAOutputSinkEnabled(uint32_t bg, bool enabled)
{
    bgaOutputSink(bg).setEnabled(enabled);
}

const std::vector<CSCFp16PartialEvent>& CSCFp16NativeExecution::bgaIngressTrace(
    uint32_t bg) const
{
    if (!bga_enabled_) throw std::logic_error("FP16 native BGA disabled");
    return bga_ingress_adapters_.at(bg)->trace();
}

const CSCFp16PartialResultPath& CSCFp16NativeExecution::transport() const
{
    if (!transport_) throw std::logic_error("FP16 transport disabled");
    return *transport_;
}

const std::vector<CSCFp16Bits>& CSCFp16NativeExecution::finalYFp16Bits() const
{
    return transport().finalYBits();
}

CSCFp16NativeCounters CSCFp16NativeExecution::counters() const
{
    CSCFp16NativeCounters result;
    result.timing = timing_;
    for (const auto& engine : engines_) {
        const auto& count = engine->counters();
        result.engine.descriptor_count += count.descriptor_count;
        result.engine.descriptor_nnz_sum += count.descriptor_nnz_sum;
        result.engine.x_requests += count.x_requests;
        result.engine.x_scalar_loads += count.x_scalar_loads;
        result.engine.value_requests += count.value_requests;
        result.engine.index_low_requests += count.index_low_requests;
        result.engine.index_high_requests += count.index_high_requests;
        result.engine.row_index_requests += count.row_index_requests;
        result.engine.logical_compute_chunks += count.logical_compute_chunks;
        result.engine.active_lanes += count.active_lanes;
        result.engine.invalid_lanes += count.invalid_lanes;
        result.engine.generated_partials += count.generated_partials;
        result.engine.emitted_partials += count.emitted_partials;
        result.engine.operand_wait_cycles += count.operand_wait_cycles;
        result.engine.request_retry_cycles += count.request_retry_cycles;
        result.engine.sink_backpressure_cycles += count.sink_backpressure_cycles;
    }
    result.timing.operand_wait_cycles = result.engine.operand_wait_cycles;
    result.timing.capture_sink_stall_cycles =
        result.engine.sink_backpressure_cycles;
    if (bga_enabled_)
        result.timing.bga_ingress_stall_engine_cycles =
            result.engine.sink_backpressure_cycles;
    if (bga_enabled_) {
        for (const auto& adapter : bga_ingress_adapters_) {
            const auto& count = adapter->counters();
            result.batch_adapter.generated_batches += count.generated_batches;
            result.batch_adapter.generated_partials += count.generated_partials;
            result.batch_adapter.batch0_count += count.batch0_count;
            result.batch_adapter.batch1_count += count.batch1_count;
            result.batch_adapter.issue_attempts += count.issue_attempts;
            result.batch_adapter.accepted_batches += count.accepted_batches;
            result.batch_adapter.stalled_attempts += count.stalled_attempts;
            result.batch_adapter.accepted_partials += count.accepted_partials;
        }
        for (const auto& bga : bgas_) {
            const auto& count = bga->counters();
            result.bga.ingress_attempts += count.ingress_attempts;
            result.bga.ingress_accepted += count.ingress_accepted;
            result.bga.ingress_stalls += count.ingress_stalls;
            result.bga.tag_comparisons += count.tag_comparisons;
            result.bga.lookup_hits += count.lookup_hits;
            result.bga.lookup_misses += count.lookup_misses;
            result.bga.fp16_adds += count.fp16_adds;
            result.bga.merges += count.merges;
            result.bga.inserts += count.inserts;
            result.bga.capacity_evictions += count.capacity_evictions;
            result.bga.final_drain_outputs += count.final_drain_outputs;
            result.bga.output_stalls += count.output_stalls;
            result.bga.output_attempts += count.output_attempts;
            result.bga.output_accepted += count.output_accepted;
            result.bga.queue_high_water =
                std::max(result.bga.queue_high_water, count.queue_high_water);
            result.bga.cycles_busy += count.cycles_busy;
            result.bga.cycles_idle += count.cycles_idle;
            result.bga.retired_contributions += count.retired_contributions;
        }
    }
    return result;
}

uint64_t cscFp16PartialTraceFnv1a64(
    const std::vector<CSCFp16PartialEvent>& trace)
{
    uint64_t hash = 1469598103934665603ULL;
    for (const auto& event : trace) {
        addLE64(event.row_idx, hash);
        addLE64(event.value_bits, hash);
        addLE64(event.global_bg_id, hash);
        addLE64(event.descriptor_id, hash);
        addLE64(event.chunk_id, hash);
        addLE64(event.lane_id, hash);
    }
    return hash;
}

}  // namespace csc_descriptor
