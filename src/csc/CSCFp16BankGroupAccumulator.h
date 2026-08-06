#ifndef CSC_FP16_BANK_GROUP_ACCUMULATOR_H
#define CSC_FP16_BANK_GROUP_ACCUMULATOR_H

#include "csc/CSCFp16DescriptorEngine.h"

#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace csc_descriptor {

enum class CSCFp16BGAIngressMode { SERIAL_EVENT, BATCH8 };

struct CSCFp16PartialBatch {
    std::array<CSCFp16PartialEvent, 8> entries{};
    uint8_t valid_count = 0;
    uint32_t global_bg_id = 0;
    uint32_t descriptor_id = 0;
    uint32_t chunk_id = 0;
    uint8_t batch_id = 0;
    bool operator==(const CSCFp16PartialBatch& other) const;
};

struct CSCFp16BGAConfig {
    CSCFp16BGAIngressMode ingress_mode = CSCFp16BGAIngressMode::SERIAL_EVENT;
    uint32_t ingress_batch_width = 1;
    uint32_t input_queue_depth = 16;
    uint32_t accumulator_entries = 16;
    uint32_t compare_width = 16;
    uint32_t compare_latency = 1;
    uint32_t add_latency = 1;
    uint32_t output_queue_depth = 16;
    uint32_t rows = 1;
};

CSCFp16BGAConfig makeFp16SerialCompatibilityConfig(uint32_t rows);
CSCFp16BGAConfig makeFp16IsoStructureProductionConfig(uint32_t rows);
CSCFp16BGAConfig makeFp16Q8StressConfig(uint32_t rows);

enum class CSCFp16BGAOutputReason { CAPACITY_EVICTION, FINAL_DRAIN };

struct CSCFp16BGAOutputEvent {
    uint32_t row_idx = 0;
    CSCFp16Bits value_bits = 0;
    uint32_t global_bg_id = 0;
    CSCFp16BGAOutputReason reason = CSCFp16BGAOutputReason::CAPACITY_EVICTION;
    uint64_t sequence = 0;
    uint64_t contribution_count = 0;
    bool operator==(const CSCFp16BGAOutputEvent& other) const;
};

class CSCFp16BGAOutputSink {
  public:
    virtual ~CSCFp16BGAOutputSink() = default;
    virtual bool ready() const = 0;
    virtual void accept(const CSCFp16BGAOutputEvent&) = 0;
};

class CSCFp16BoundedBGAOutputSink : public CSCFp16BGAOutputSink {
  public:
    explicit CSCFp16BoundedBGAOutputSink(std::size_t capacity);
    bool ready() const override;
    void accept(const CSCFp16BGAOutputEvent&) override;
    CSCFp16BGAOutputEvent pop();
    void setEnabled(bool enabled) { enabled_ = enabled; }
    std::size_t size() const { return queue_.size(); }
    const std::vector<CSCFp16BGAOutputEvent>& trace() const { return trace_; }
  private:
    std::size_t capacity_;
    bool enabled_ = true;
    std::deque<CSCFp16BGAOutputEvent> queue_;
    std::vector<CSCFp16BGAOutputEvent> trace_;
};

struct CSCFp16BGACounters {
    uint64_t ingress_attempts = 0;
    uint64_t ingress_accepted = 0;
    uint64_t ingress_stalls = 0;
    uint64_t batch_attempts = 0;
    uint64_t batches_accepted = 0;
    uint64_t batches_stalled = 0;
    uint64_t batch0_count = 0;
    uint64_t batch1_count = 0;
    uint64_t partials_serviced = 0;
    uint64_t input_high_water = 0;
    uint64_t tag_comparisons = 0;
    uint64_t lookup_hits = 0;
    uint64_t lookup_misses = 0;
    uint64_t fp16_adds = 0;
    uint64_t merges = 0;
    uint64_t inserts = 0;
    uint64_t capacity_evictions = 0;
    uint64_t final_drain_outputs = 0;
    uint64_t output_stalls = 0;
    uint64_t output_attempts = 0;
    uint64_t output_accepted = 0;
    uint64_t queue_high_water = 0;
    uint64_t cycles_busy = 0;
    uint64_t cycles_idle = 0;
    uint64_t retired_contributions = 0;
};

class CSCFp16BankGroupAccumulator : public CSCFp16PartialSink {
  public:
    CSCFp16BankGroupAccumulator(uint32_t global_bg_id,
                                const CSCFp16BGAConfig& config);
    bool ready() const override;
    void accept(const CSCFp16PartialEvent&) override;
    bool canAcceptBatch(const CSCFp16PartialBatch&) const;
    bool acceptBatch(const CSCFp16PartialBatch&);
    void markProducerDone();
    bool requestFinalDrain();
    void step();
    bool hasOutput() const { return !outputs_.empty(); }
    const CSCFp16BGAOutputEvent& peekOutput() const;
    void acceptOutput();
    bool finalDrainComplete() const;
    bool quiescent() const;
    uint64_t cycle() const { return cycle_; }
    uint32_t occupancy() const;
    const CSCFp16BGAConfig& config() const { return config_; }
    const CSCFp16BGACounters& counters() const { return counters_; }
    const std::string& error() const { return error_; }
    bool conservationInvariant() const;
  private:
    struct Entry {
        bool valid = false;
        bool reserved = false;
        uint32_t row_idx = 0;
        CSCFp16Bits value_bits = 0;
        uint64_t age = 0;
        uint64_t contributions = 0;
    };
    enum class OperationKind { LOOKUP, MERGE, INSERT };
    struct Operation {
        OperationKind kind = OperationKind::LOOKUP;
        CSCFp16PartialEvent input{};
        uint32_t slot = 0;
        uint32_t remaining = 0;
    };
    struct PendingMiss {
        CSCFp16PartialEvent input{};
        uint32_t victim = 0;
        uint64_t output_sequence = 0;
        bool awaiting_retirement = false;
    };
    struct OutputRecord {
        CSCFp16BGAOutputEvent event{};
        uint32_t slot = 0;
    };

    void validateConfig() const;
    bool liveWork() const;
    void commitOperation();
    void updateEvictionOrDrain();
    void startService();
    void commitIngress();
    void retireOutput();
    uint32_t selectOldest() const;
    uint64_t pushOutput(uint32_t, CSCFp16BGAOutputReason);
    void insertAt(uint32_t, const CSCFp16PartialEvent&);
    uint64_t liveContributions() const;

    uint32_t global_bg_id_;
    CSCFp16BGAConfig config_;
    std::deque<CSCFp16PartialEvent> input_queue_;
    std::optional<CSCFp16PartialBatch> accepted_pending_;
    std::vector<Entry> accumulator_;
    std::deque<OutputRecord> outputs_;
    std::optional<Operation> operation_;
    std::optional<PendingMiss> pending_miss_;
    bool output_retirement_pending_ = false;
    bool producer_done_ = false;
    bool final_drain_requested_ = false;
    bool operation_committed_this_cycle_ = false;
    uint64_t next_age_ = 0;
    uint64_t next_output_sequence_ = 0;
    uint64_t cycle_ = 0;
    CSCFp16BGACounters counters_{};
    std::string error_;
};

uint64_t cscFp16BGAOutputTraceFnv1a64(
    const std::vector<CSCFp16BGAOutputEvent>& trace);

}  // namespace csc_descriptor

#endif
