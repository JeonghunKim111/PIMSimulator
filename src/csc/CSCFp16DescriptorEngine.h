#ifndef CSC_FP16_DESCRIPTOR_ENGINE_H
#define CSC_FP16_DESCRIPTOR_ENGINE_H

#include "PIMBlock.h"
#include "csc/CSCFp16Image.h"

#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace csc_descriptor {

enum class CSCFp16ExecutionMode { FP16_IMAGE_V2 };

class CSCFp16ExecutionImage {
  public:
    static std::shared_ptr<const CSCFp16ExecutionImage> load(
        const std::string& directory, CSCFp16ExecutionMode mode);
    const CSCFp16LoadedImage& image() const { return image_; }

  private:
    explicit CSCFp16ExecutionImage(CSCFp16LoadedImage image)
        : image_(std::move(image)) {}
    CSCFp16LoadedImage image_;
};

enum class CSCFp16RequestKind { X, VALUE, INDEX_LOW, INDEX_HIGH };

struct CSCFp16Request {
    uint64_t request_id = 0;
    CSCFp16RequestKind kind = CSCFp16RequestKind::X;
    uint32_t global_bg_id = 0;
    uint32_t descriptor_id = 0;
    uint32_t chunk_id = 0;
    uint64_t stream_offset_bytes = 0;

    bool operator==(const CSCFp16Request& other) const;
};

struct CSCFp16PartialEvent {
    uint32_t row_idx = 0;
    CSCFp16Bits value_bits = 0;
    uint32_t global_bg_id = 0;
    uint32_t descriptor_id = 0;
    uint32_t chunk_id = 0;
    uint8_t lane_id = 0;
    uint8_t chunk_valid_count = 0;

    bool operator==(const CSCFp16PartialEvent& other) const;
};

class CSCFp16PartialSink {
  public:
    virtual ~CSCFp16PartialSink() = default;
    virtual bool ready() const = 0;
    virtual void accept(const CSCFp16PartialEvent& event) = 0;
};

class CSCFp16BoundedCaptureSink : public CSCFp16PartialSink {
  public:
    explicit CSCFp16BoundedCaptureSink(std::size_t capacity);
    bool ready() const override;
    void accept(const CSCFp16PartialEvent& event) override;
    CSCFp16PartialEvent pop();
    void setEnabled(bool enabled) { enabled_ = enabled; }
    std::size_t size() const { return queue_.size(); }
    const std::vector<CSCFp16PartialEvent>& trace() const { return trace_; }

  private:
    std::size_t capacity_;
    bool enabled_ = true;
    std::deque<CSCFp16PartialEvent> queue_;
    std::vector<CSCFp16PartialEvent> trace_;
};

enum class CSCFp16EngineState {
    IDLE,
    FETCH_DESCRIPTOR,
    ISSUE_REQUESTS,
    WAIT_OPERANDS,
    SIMD_MUL,
    EMIT_PARTIALS,
    ADVANCE_CHUNK,
    NEXT_DESCRIPTOR,
    DONE,
    ERROR
};

struct CSCFp16EngineCounters {
    uint64_t descriptor_count = 0;
    uint64_t descriptor_nnz_sum = 0;
    uint64_t x_requests = 0;
    uint64_t x_scalar_loads = 0;
    uint64_t value_requests = 0;
    uint64_t index_low_requests = 0;
    uint64_t index_high_requests = 0;
    uint64_t row_index_requests = 0;
    uint64_t logical_compute_chunks = 0;
    uint64_t active_lanes = 0;
    uint64_t invalid_lanes = 0;
    uint64_t generated_partials = 0;
    uint64_t emitted_partials = 0;
    uint64_t operand_wait_cycles = 0;
    uint64_t request_retry_cycles = 0;
    uint64_t sink_backpressure_cycles = 0;
};

class CSCFp16DescriptorEngine {
  public:
    using Submit = std::function<bool(const CSCFp16Request&)>;

    CSCFp16DescriptorEngine(uint32_t global_bg_id,
                            std::shared_ptr<const CSCFp16ExecutionImage> image,
                            DRAMSim::PIMBlock* datapath, Submit submit,
                            CSCFp16PartialSink* sink);
    void launch();
    void tick();
    bool complete(const CSCFp16Request& request,
                  const std::array<uint8_t, 32>& payload);

    bool done() const { return state_ == CSCFp16EngineState::DONE; }
    bool failed() const { return state_ == CSCFp16EngineState::ERROR; }
    CSCFp16EngineState state() const { return state_; }
    const std::string& error() const { return error_; }
    const CSCFp16EngineCounters& counters() const { return counters_; }
    uint32_t validCount() const { return valid_count_; }
    uint64_t valueStreamOffset() const { return value_stream_offset_; }
    uint64_t rowIndexStreamOffset() const { return index_stream_offset_; }

  private:
    struct RequestSlot {
        CSCFp16Request request{};
        bool required = false;
        bool accepted = false;
        bool ready = false;
    };

    void fail(const std::string& message);
    void beginDescriptor();
    void prepareChunk();
    RequestSlot* slot(CSCFp16RequestKind kind);
    bool allOperandsReady() const;
    void stage(const CSCFp16Request&, const std::array<uint8_t, 32>&);
    void executeMultiply();

    uint32_t global_bg_id_;
    std::shared_ptr<const CSCFp16ExecutionImage> execution_image_;
    DRAMSim::PIMBlock* datapath_;
    Submit submit_;
    CSCFp16PartialSink* sink_;
    CSCFp16EngineState state_ = CSCFp16EngineState::IDLE;
    std::string error_;
    CSCFp16EngineCounters counters_;
    uint32_t descriptor_id_ = 0;
    uint32_t chunk_id_ = 0;
    uint32_t remaining_nnz_ = 0;
    uint32_t valid_count_ = 0;
    uint32_t emit_lane_ = 0;
    uint64_t value_stream_offset_ = 0;
    uint64_t index_stream_offset_ = 0;
    uint64_t next_request_id_ = 1;
    CSCDescriptor descriptor_{};
    CSCFp16 x_scalar_{};
    std::array<CSCFp16, 16> matrix_values_{};
    std::array<uint32_t, 16> row_indices_{};
    DRAMSim::BurstType result_{};
    std::array<RequestSlot, 4> requests_{};
    std::set<uint64_t> completed_request_ids_;
};

}  // namespace csc_descriptor

#endif
