#ifndef CSC_BANK_GROUP_ACCUMULATOR_H
#define CSC_BANK_GROUP_ACCUMULATOR_H

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace csc_descriptor {

struct CSCBGAConfig {
    uint32_t input_streams = 1;
    uint32_t input_queue_depth = 16;
    uint32_t accumulator_entries = 16;
    uint32_t compare_width = 16;
    uint32_t compare_latency = 1;
    uint32_t add_latency = 1;
    uint32_t output_queue_depth = 16;
    uint32_t rows = 1;
};

struct CSCBGAPartial {
    uint32_t row_idx = 0;
    float value = 0;
};

struct CSCBGABatch {
    uint32_t logical_stream_id = 0;
    uint32_t generation = 0;
    uint64_t sequence = 0;
    std::vector<CSCBGAPartial> partials;
};

enum class CSCBGAInputResult {
    ACCEPTED,
    BACKPRESSURE,
    DUPLICATE,
    PROTOCOL_ERROR
};

enum class CSCBGAOutputReason {
    CAPACITY_EVICTION,
    FINAL_DRAIN
};

struct CSCBGAOutput {
    uint32_t row_idx = 0;
    float value = 0;
    uint32_t source_stream_id = 0;
    uint32_t generation = 0;
    uint64_t epoch = 0;
    uint64_t output_sequence = 0;
    uint64_t contribution_count = 0;
    CSCBGAOutputReason reason = CSCBGAOutputReason::CAPACITY_EVICTION;
};

struct CSCBGAAccumulatorEntry {
    bool valid = false;
    bool reserved = false;
    uint32_t row_idx = 0;
    float value = 0;
    uint64_t insertion_age = 0;
    uint32_t generation = 0;
    uint32_t slot = 0;
    uint32_t source_stream_id = 0;
    uint64_t contribution_count = 0;
};

// Event statistics saturate at UINT64_MAX. Contribution counters and all
// ordering/identity state use checked arithmetic and throw before wraparound.
// Counters remain cumulative across abortReset().
struct CSCBGACounters {
    uint64_t enqueue_attempts = 0;
    uint64_t accepted_batches = 0;
    uint64_t backpressured_batches = 0;
    uint64_t accepted_valid_partials = 0;
    // One comparison per valid, non-reserved tag examined; lookup scans all.
    uint64_t tag_comparisons = 0;
    uint64_t lookup_hits = 0;
    uint64_t lookup_misses = 0;
    uint64_t fp32_merges = 0;
    uint64_t inserts = 0;
    uint64_t capacity_evictions = 0;
    uint64_t final_drain_outputs = 0;
    uint64_t output_stalls = 0;
    uint64_t protocol_errors = 0;
    uint64_t invariant_errors = 0;
    uint64_t cycles_busy = 0;
    uint64_t cycles_idle = 0;
    uint64_t retired_output_contributions = 0;
    uint64_t aborted_contributions = 0;
};

class CSCBankGroupAccumulator {
  public:
    explicit CSCBankGroupAccumulator(const CSCBGAConfig&, uint32_t generation = 1);

    // ACCEPTED transfers ownership and updates identity/counters immediately.
    // FIFO visibility begins only after the next step() end phase.
    CSCBGAInputResult offerBatch(const CSCBGABatch&);
    void markProducerDone(uint32_t logical_stream_id);
    bool requestFinalDrain();
    void abortReset(uint32_t new_generation);
    void step();

    bool hasOutput() const { return !output_queue_.empty(); }
    const CSCBGAOutput& peekOutput() const;
    // Schedules stable front-output retirement for the next step() phase 1.
    void acceptOutput();

    bool finalDrainRequested() const { return final_drain_requested_; }
    bool finalDrainComplete() const;
    bool quiescent() const;
    uint64_t cycle() const { return cycle_; }
    uint32_t generation() const { return generation_; }
    const CSCBGAConfig& config() const { return config_; }
    const CSCBGACounters& counters() const { return counters_; }
    const std::vector<CSCBGAAccumulatorEntry>& accumulator() const { return accumulator_; }
    size_t inputQueueSize(uint32_t logical_stream_id) const;
    size_t outputQueueSize() const { return output_queue_.size(); }
    bool hasAcceptedPending(uint32_t logical_stream_id) const;
    uint64_t lastAcceptedSequence(uint32_t logical_stream_id) const;
    bool validateAccumulatorInvariant() const;
    bool conservationInvariant() const;
    uint64_t liveContributionCount() const;
    const std::string& error() const { return error_; }

  private:
    struct InputEntry {
        uint32_t row_idx = 0;
        float value = 0;
        uint32_t stream = 0;
        uint32_t generation = 0;
        uint64_t batch_sequence = 0;
    };
    struct StreamState {
        std::deque<InputEntry> queue;
        std::optional<CSCBGABatch> presented_batch;
        std::optional<CSCBGABatch> accepted_pending;
        std::optional<CSCBGABatch> last_accepted_batch;
        uint64_t last_accepted_sequence = 0;
        bool done = false;
    };
    enum class OperationKind { LOOKUP, MERGE, INSERT };
    struct Operation {
        OperationKind kind = OperationKind::LOOKUP;
        InputEntry input{};
        uint32_t slot = 0;
        uint32_t remaining = 0;
    };
    struct PendingMiss {
        InputEntry input{};
        uint32_t victim_slot = 0;
        uint64_t eviction_output_sequence = 0;
        bool awaiting_retirement = false;
    };
    struct OutputRecord {
        CSCBGAOutput payload{};
        uint32_t reserved_slot = 0;
    };

    void validateConfig() const;
    static uint32_t floatBits(float);
    bool sameBatch(const CSCBGABatch&, const CSCBGABatch&) const;
    void protocolError(const std::string&);
    [[noreturn]] void invariantError(const std::string&);
    CSCBGAInputResult acceptBatch(StreamState&, const CSCBGABatch&);
    bool allProducersDone() const;
    bool hasLiveWork() const;
    void retireOutput();
    void commitOperation();
    void updateEvictionOrDrain();
    void startService();
    void commitAcceptedBatches();
    uint32_t selectVictim() const;
    uint32_t selectDrainEntry() const;
    uint64_t pushOutput(uint32_t slot, CSCBGAOutputReason, uint64_t epoch);
    void insertAt(uint32_t slot, const InputEntry&);
    static void saturatingIncrement(uint64_t&);
    static void checkedAdd(uint64_t&, uint64_t, const char*);
    static uint64_t checkedSum(uint64_t, uint64_t, const char*);

    CSCBGAConfig config_;
    uint32_t generation_;
    std::vector<StreamState> streams_;
    std::vector<CSCBGAAccumulatorEntry> accumulator_;
    std::deque<OutputRecord> output_queue_;
    std::optional<Operation> operation_;
    std::optional<PendingMiss> pending_miss_;
    bool output_retirement_pending_ = false;
    bool final_drain_requested_ = false;
    bool operation_committed_this_cycle_ = false;
    uint32_t next_stream_rr_ = 0;
    uint64_t insertion_age_ = 0;
    uint64_t output_sequence_ = 0;
    uint64_t capacity_epoch_ = 0;
    uint64_t final_drain_epoch_ = 0;
    uint64_t cycle_ = 0;
    CSCBGACounters counters_{};
    std::string error_;
};

}  // namespace csc_descriptor

#endif
