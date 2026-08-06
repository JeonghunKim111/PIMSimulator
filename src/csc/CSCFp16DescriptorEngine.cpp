#include "csc/CSCFp16DescriptorEngine.h"

#include <algorithm>
#include <stdexcept>

namespace csc_descriptor {
namespace {

std::size_t requestIndex(CSCFp16RequestKind kind)
{
    return static_cast<std::size_t>(kind);
}

uint32_t readU32LE(const std::array<uint8_t, 32>& bytes, std::size_t offset)
{
    return uint32_t(bytes.at(offset)) |
           (uint32_t(bytes.at(offset + 1)) << 8) |
           (uint32_t(bytes.at(offset + 2)) << 16) |
           (uint32_t(bytes.at(offset + 3)) << 24);
}

uint16_t readU16LE(const std::array<uint8_t, 32>& bytes, std::size_t offset)
{
    return uint16_t(bytes.at(offset)) |
           (uint16_t(bytes.at(offset + 1)) << 8);
}

}  // namespace

std::shared_ptr<const CSCFp16ExecutionImage> CSCFp16ExecutionImage::load(
    const std::string& directory, CSCFp16ExecutionMode mode)
{
    if (mode != CSCFp16ExecutionMode::FP16_IMAGE_V2)
        throw std::invalid_argument("unsupported CSC FP16 execution mode");
    return std::shared_ptr<const CSCFp16ExecutionImage>(
        new CSCFp16ExecutionImage(loadCSCFp16ImageV2(directory)));
}

bool CSCFp16Request::operator==(const CSCFp16Request& other) const
{
    return request_id == other.request_id && kind == other.kind &&
           global_bg_id == other.global_bg_id &&
           descriptor_id == other.descriptor_id && chunk_id == other.chunk_id &&
           stream_offset_bytes == other.stream_offset_bytes;
}

bool CSCFp16PartialEvent::operator==(const CSCFp16PartialEvent& other) const
{
    return row_idx == other.row_idx && value_bits == other.value_bits &&
           global_bg_id == other.global_bg_id &&
           descriptor_id == other.descriptor_id && chunk_id == other.chunk_id &&
           lane_id == other.lane_id;
}

CSCFp16BoundedCaptureSink::CSCFp16BoundedCaptureSink(std::size_t capacity)
    : capacity_(capacity)
{
    if (!capacity_) throw std::invalid_argument("FP16 capture capacity is zero");
}

bool CSCFp16BoundedCaptureSink::ready() const
{
    return enabled_ && queue_.size() < capacity_;
}

void CSCFp16BoundedCaptureSink::accept(const CSCFp16PartialEvent& event)
{
    if (!ready()) throw std::logic_error("FP16 capture accept without ready");
    queue_.push_back(event);
    trace_.push_back(event);
}

CSCFp16PartialEvent CSCFp16BoundedCaptureSink::pop()
{
    if (queue_.empty()) throw std::logic_error("FP16 capture pop empty");
    const auto event = queue_.front();
    queue_.pop_front();
    return event;
}

CSCFp16DescriptorEngine::CSCFp16DescriptorEngine(
    uint32_t global_bg_id,
    std::shared_ptr<const CSCFp16ExecutionImage> image,
    DRAMSim::PIMBlock* datapath, Submit submit, CSCFp16PartialSink* sink)
    : global_bg_id_(global_bg_id), execution_image_(std::move(image)),
      datapath_(datapath), submit_(std::move(submit)), sink_(sink)
{
    if (global_bg_id_ >= kCSCFp16ImageBGCount || !execution_image_ ||
        !datapath_ || !submit_ || !sink_)
        throw std::invalid_argument("invalid CSC FP16 engine construction");
    if (datapath_->precision() != DRAMSim::FP16)
        throw std::invalid_argument("CSC FP16 engine requires FP16 PIMBlock");
}

void CSCFp16DescriptorEngine::launch()
{
    if (state_ != CSCFp16EngineState::IDLE)
        throw std::logic_error("CSC FP16 engine launch state");
    datapath_->resetSIMDCounters();
    state_ = execution_image_->image().bg[global_bg_id_].parsed_descriptors.empty()
                 ? CSCFp16EngineState::DONE
                 : CSCFp16EngineState::FETCH_DESCRIPTOR;
}

void CSCFp16DescriptorEngine::fail(const std::string& message)
{
    if (state_ == CSCFp16EngineState::ERROR) return;
    error_ = message;
    state_ = CSCFp16EngineState::ERROR;
}

void CSCFp16DescriptorEngine::beginDescriptor()
{
    const auto& image = execution_image_->image();
    const auto& bg = image.bg[global_bg_id_];
    if (descriptor_id_ >= bg.parsed_descriptors.size()) {
        fail("FP16 descriptor pointer outside image");
        return;
    }
    descriptor_ = bg.parsed_descriptors[descriptor_id_];
    if (!descriptor_.nnz_count || descriptor_.global_bg_id != global_bg_id_ ||
        descriptor_.original_col >= image.matrix.cols ||
        descriptor_.x_slot >= bg.parsed_descriptors.size() ||
        descriptor_.value_offset_bytes % 32 ||
        descriptor_.row_idx_offset_bytes % 32) {
        fail("malformed FP16 execution descriptor");
        return;
    }
    remaining_nnz_ = descriptor_.nnz_count;
    chunk_id_ = 0;
    value_stream_offset_ = 0;
    index_stream_offset_ = 0;
    counters_.descriptor_count++;
    counters_.descriptor_nnz_sum += descriptor_.nnz_count;
    prepareChunk();
}

void CSCFp16DescriptorEngine::prepareChunk()
{
    valid_count_ = std::min<uint32_t>(remaining_nnz_, 16);
    if (!valid_count_) {
        fail("zero-sized FP16 chunk");
        return;
    }
    requests_ = {};
    const auto make = [&](CSCFp16RequestKind kind, uint64_t offset, bool required) {
        auto& request = requests_[requestIndex(kind)];
        request.required = required;
        if (!required) return;
        request.request.request_id =
            (uint64_t(global_bg_id_ + 1) << 56) | next_request_id_++;
        request.request.kind = kind;
        request.request.global_bg_id = global_bg_id_;
        request.request.descriptor_id = descriptor_id_;
        request.request.chunk_id = chunk_id_;
        request.request.stream_offset_bytes = offset;
    };
    const uint64_t x_offset = uint64_t(descriptor_.original_col / 16) * 32;
    make(CSCFp16RequestKind::X, x_offset, chunk_id_ == 0);
    make(CSCFp16RequestKind::VALUE,
         descriptor_.value_offset_bytes + value_stream_offset_, true);
    make(CSCFp16RequestKind::INDEX_LOW,
         descriptor_.row_idx_offset_bytes + index_stream_offset_, true);
    make(CSCFp16RequestKind::INDEX_HIGH,
         descriptor_.row_idx_offset_bytes + index_stream_offset_ + 32,
         valid_count_ > 8);
    state_ = CSCFp16EngineState::ISSUE_REQUESTS;
}

CSCFp16DescriptorEngine::RequestSlot* CSCFp16DescriptorEngine::slot(
    CSCFp16RequestKind kind)
{
    return &requests_[requestIndex(kind)];
}

bool CSCFp16DescriptorEngine::allOperandsReady() const
{
    for (const auto& request : requests_)
        if (request.required && !request.ready) return false;
    return true;
}

void CSCFp16DescriptorEngine::stage(
    const CSCFp16Request& request, const std::array<uint8_t, 32>& payload)
{
    switch (request.kind) {
    case CSCFp16RequestKind::X: {
        const uint32_t lane = descriptor_.original_col % 16;
        x_scalar_ = cscFp16FromBits(readU16LE(payload, lane * 2));
        counters_.x_scalar_loads++;
        break;
    }
    case CSCFp16RequestKind::VALUE:
        for (uint32_t lane = 0; lane < 16; ++lane)
            matrix_values_[lane] =
                cscFp16FromBits(readU16LE(payload, lane * 2));
        break;
    case CSCFp16RequestKind::INDEX_LOW:
        for (uint32_t lane = 0; lane < 8; ++lane)
            row_indices_[lane] = readU32LE(payload, lane * 4);
        break;
    case CSCFp16RequestKind::INDEX_HIGH:
        for (uint32_t lane = 0; lane < 8; ++lane)
            row_indices_[lane + 8] = readU32LE(payload, lane * 4);
        break;
    }
}

bool CSCFp16DescriptorEngine::complete(
    const CSCFp16Request& request, const std::array<uint8_t, 32>& payload)
{
    if (failed() || done()) return false;
    if (completed_request_ids_.count(request.request_id)) {
        fail("duplicate FP16 request completion");
        return false;
    }
    auto* expected = slot(request.kind);
    if (!expected->required || !expected->accepted || expected->ready ||
        !(expected->request == request)) {
        fail("unknown, stale, or mismatched FP16 request completion");
        return false;
    }
    stage(request, payload);
    expected->ready = true;
    expected->accepted = false;
    completed_request_ids_.insert(request.request_id);
    return true;
}

void CSCFp16DescriptorEngine::executeMultiply()
{
    DRAMSim::BurstType values{};
    DRAMSim::BurstType scalar{};
    for (uint32_t lane = 0; lane < 16; ++lane) {
        values.fp16Data_[lane] = matrix_values_[lane];
        scalar.fp16Data_[lane] = x_scalar_;
        result_.fp16Data_[lane] = cscFp16FromBits(0x7bffU);
    }
    datapath_->mul(result_, values, scalar, valid_count_);
    counters_.logical_compute_chunks++;
    counters_.active_lanes += valid_count_;
    counters_.invalid_lanes += 16 - valid_count_;
    counters_.generated_partials += valid_count_;
    emit_lane_ = 0;
}

void CSCFp16DescriptorEngine::tick()
{
    switch (state_) {
    case CSCFp16EngineState::IDLE:
    case CSCFp16EngineState::DONE:
    case CSCFp16EngineState::ERROR:
        return;
    case CSCFp16EngineState::FETCH_DESCRIPTOR:
        beginDescriptor();
        return;
    case CSCFp16EngineState::ISSUE_REQUESTS:
        for (auto& request : requests_) {
            if (!request.required || request.ready || request.accepted) continue;
            if (!submit_(request.request)) {
                counters_.request_retry_cycles++;
                return;
            }
            request.accepted = true;
            switch (request.request.kind) {
            case CSCFp16RequestKind::X: counters_.x_requests++; break;
            case CSCFp16RequestKind::VALUE: counters_.value_requests++; break;
            case CSCFp16RequestKind::INDEX_LOW:
                counters_.index_low_requests++;
                counters_.row_index_requests++;
                break;
            case CSCFp16RequestKind::INDEX_HIGH:
                counters_.index_high_requests++;
                counters_.row_index_requests++;
                break;
            }
            return;
        }
        state_ = CSCFp16EngineState::WAIT_OPERANDS;
        return;
    case CSCFp16EngineState::WAIT_OPERANDS:
        if (allOperandsReady())
            state_ = CSCFp16EngineState::SIMD_MUL;
        else
            counters_.operand_wait_cycles++;
        return;
    case CSCFp16EngineState::SIMD_MUL:
        executeMultiply();
        state_ = CSCFp16EngineState::EMIT_PARTIALS;
        return;
    case CSCFp16EngineState::EMIT_PARTIALS:
        if (!sink_->ready()) {
            counters_.sink_backpressure_cycles++;
            return;
        }
        sink_->accept({row_indices_[emit_lane_],
                       cscFp16ToBits(result_.fp16Data_[emit_lane_]),
                       global_bg_id_, descriptor_id_, chunk_id_,
                       static_cast<uint8_t>(emit_lane_),
                       static_cast<uint8_t>(valid_count_)});
        counters_.emitted_partials++;
        if (++emit_lane_ == valid_count_)
            state_ = CSCFp16EngineState::ADVANCE_CHUNK;
        return;
    case CSCFp16EngineState::ADVANCE_CHUNK:
        remaining_nnz_ -= valid_count_;
        value_stream_offset_ += 32;
        index_stream_offset_ += valid_count_ > 8 ? 64 : 32;
        chunk_id_++;
        if (remaining_nnz_)
            prepareChunk();
        else
            state_ = CSCFp16EngineState::NEXT_DESCRIPTOR;
        return;
    case CSCFp16EngineState::NEXT_DESCRIPTOR:
        if (++descriptor_id_ ==
            execution_image_->image().bg[global_bg_id_].parsed_descriptors.size())
            state_ = CSCFp16EngineState::DONE;
        else
            state_ = CSCFp16EngineState::FETCH_DESCRIPTOR;
        return;
    }
}

}  // namespace csc_descriptor
