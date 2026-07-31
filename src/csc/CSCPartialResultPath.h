#ifndef CSC_PARTIAL_RESULT_PATH_H
#define CSC_PARTIAL_RESULT_PATH_H

#include "csc/CSCBGAIntegration.h"

#include <array>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace csc_descriptor {

struct CSCM7BHostReadbackTestAccess;
struct CSCM7BHostReductionTestAccess;

constexpr uint32_t kCSCPartialResultRecordBytes = 8;
constexpr uint32_t kCSCPartialResultBurstBytes = 32;
constexpr uint32_t kCSCPartialRecordsPerBurst = 4;

struct CSCPartialResultRecord {
    uint32_t row_idx = 0;
    float value = 0;
};
static_assert(sizeof(CSCPartialResultRecord) == kCSCPartialResultRecordBytes,
              "CSC partial-result record must be 8 bytes");

struct CSCPartialResultRecordEnvelope {
    CSCPartialResultRecord record{};
    uint32_t global_bg_id = 0;
    uint32_t generation = 0;
    uint64_t output_sequence = 0;
    uint32_t contribution_count = 0;
    CSCBGAOutputReason reason = CSCBGAOutputReason::CAPACITY_EVICTION;
};

struct CSCPartialResultBurst {
    uint32_t global_bg_id = 0;
    uint64_t writeback_burst_sequence = 0;
    uint32_t valid_record_count = 0;
    bool tail = false;
    std::array<CSCPartialResultRecordEnvelope,
               kCSCPartialRecordsPerBurst> records{};
    uint64_t formation_cycle = 0;
    uint64_t issue_cycle = 0;
    uint64_t completion_cycle = 0;
};

struct CSCPartialResultPathConfig {
    uint32_t global_bg_count = 0;
    uint32_t channel_count = 0;
    uint32_t ranks_per_channel = 0;
    uint32_t bank_groups_per_rank = 0;
    uint32_t buffer_capacity_bursts_per_bg = 16;
    uint32_t pending_capacity_bursts_per_bg = 4;
    uint32_t write_latency_cycles = 4;
    uint32_t max_inflight_writes_per_bg = 2;
    uint32_t write_issue_limit_per_rank_per_cycle = 1;
    uint32_t read_latency_cycles = 4;
    uint32_t read_issue_limit_per_channel_per_cycle = 1;
    uint32_t max_inflight_reads_per_channel = 2;
    uint32_t host_return_queue_capacity_bursts = 16;
    bool host_reduction_enabled = false;
    uint32_t reduction_queue_capacity_records = 16;
    uint32_t host_reduce_records_per_cycle = 4;
    uint32_t host_reduce_latency_cycles = 2;

    void validate() const;
};

struct CSCPartialResultPathCounters {
    uint64_t partial_records_generated = 0;
    uint64_t partial_record_contribution_sum = 0;
    uint64_t packed_record_contribution_sum = 0;
    uint64_t resident_record_contribution_sum = 0;
    uint64_t full_writeback_bursts = 0;
    uint64_t tail_writeback_bursts = 0;
    uint64_t total_writeback_bursts = 0;
    uint64_t writeback_useful_bytes = 0;
    uint64_t writeback_transferred_bytes = 0;
    uint64_t writeback_padding_bytes = 0;
    uint64_t write_requests_issued = 0;
    uint64_t write_requests_completed = 0;
    uint64_t peak_inflight_writes = 0;
    uint64_t writeback_backpressure_cycles = 0;
    uint64_t packer_backpressure_cycles = 0;
    uint64_t buffer_backpressure_cycles = 0;
    uint64_t inflight_backpressure_cycles = 0;
    uint64_t first_bga_output_accept_cycle = 0;
    uint64_t first_write_issue_cycle = 0;
    uint64_t first_write_complete_cycle = 0;
    uint64_t last_write_issue_cycle = 0;
    uint64_t last_write_complete_cycle = 0;
    uint64_t partial_writeback_complete_cycle = 0;
    std::vector<uint64_t> peak_inflight_writes_by_bg;
};

struct CSCPartialResultWriteTrace {
    uint32_t global_bg_id = 0;
    uint64_t burst_sequence = 0;
    uint64_t issue_cycle = 0;
    uint64_t completion_cycle = 0;
};

struct CSCHostReturnedBurst {
    CSCPartialResultBurst burst{};
    uint32_t channel_id = 0;
    uint32_t rank_id = 0;
    uint32_t local_bg_id = 0;
    uint64_t host_visible_cycle = 0;
};

struct CSCHostReadbackCounters {
    uint64_t host_readback_start_cycle = 0;
    uint64_t first_read_issue_cycle = 0;
    uint64_t first_read_complete_cycle = 0;
    uint64_t last_read_issue_cycle = 0;
    uint64_t last_read_complete_cycle = 0;
    uint64_t read_transport_complete_cycle = 0;
    uint64_t read_requests_issued = 0;
    uint64_t read_requests_completed = 0;
    uint64_t peak_inflight_reads = 0;
    std::vector<uint64_t> peak_inflight_reads_by_channel;
    uint64_t returned_bursts_committed = 0;
    uint64_t returned_bursts_delivered = 0;
    uint64_t peak_host_return_queue_occupancy = 0;
    uint64_t peak_reserved_return_slots = 0;
    uint64_t read_inflight_backpressure_cycles = 0;
    uint64_t return_queue_backpressure_cycles = 0;
    uint64_t readback_transferred_bytes = 0;
    uint64_t readback_useful_bytes = 0;
    uint64_t readback_padding_bytes = 0;
    uint64_t readback_records = 0;
    uint64_t readback_contribution_sum = 0;
};

struct CSCHostReductionCounters {
    uint64_t reduction_records_enqueued = 0;
    uint64_t reduction_records_issued = 0;
    uint64_t reduced_partial_records = 0;
    uint64_t reduced_contribution_count = 0;
    uint64_t fp32_host_add_count = 0;
    uint64_t peak_reduction_queue_occupancy = 0;
    uint64_t reduction_queue_backpressure_cycles = 0;
    uint64_t reduction_batches_issued = 0;
    uint64_t reduction_batches_completed = 0;
    uint64_t first_returned_burst_accept_cycle = 0;
    uint64_t first_reduction_batch_issue_cycle = 0;
    uint64_t first_host_reduce_cycle = 0;
    uint64_t last_host_reduce_cycle = 0;
    uint64_t host_readback_complete_cycle = 0;
    uint64_t host_reduction_complete_cycle = 0;
    uint64_t same_row_cross_bg_merges = 0;
    uint64_t same_row_repeated_record_merges = 0;
};

class CSCPartialResultPath final : public CSCBGAOutputDestination {
  public:
    CSCPartialResultPath(const CSCPartialResultPathConfig&, uint32_t rows,
                         uint32_t generation);

    bool reserve(const CSCBGAOutputPortValue&) override;
    void commitReserved() noexcept override;
    void cancelReserved() noexcept override;

    void step(uint64_t cycle,
              const std::vector<bool>& final_drain_complete);

    bool partialWritebackComplete() const;
    bool hostReadTransportComplete() const;
    bool hostReadbackComplete() const;
    bool hostReductionComplete() const;
    bool hasHostReturnedBurst() const;
    const CSCHostReturnedBurst& peekHostReturnedBurst() const;
    void acceptHostReturnedBurst();
    bool hasError() const { return error_; }
    const std::string& errorMessage() const { return error_message_; }
    const CSCPartialResultPathCounters& counters() const { return counters_; }
    const CSCHostReadbackCounters& readbackCounters() const {
        return readback_counters_;
    }
    const CSCHostReductionCounters& reductionCounters() const {
        return reduction_counters_;
    }
    const std::vector<float>& hostReducedResult() const;
    const CSCPartialResultPathConfig& config() const { return config_; }
    uint32_t globalBGCount() const { return bg_.size(); }
    uint64_t currentCycle() const { return cycle_; }
    uint32_t packerRecordCount(uint32_t) const;
    uint32_t pendingBurstCount(uint32_t) const;
    uint32_t inflightWriteCount(uint32_t) const;
    uint32_t residentBurstCount(uint32_t) const;
    uint32_t reservedBufferSlots(uint32_t) const;
    uint32_t readInflightBufferSlots(uint32_t) const;
    uint32_t reservedHostReturnSlots() const {
        return reserved_host_return_slots_;
    }
    uint32_t hostReturnQueueSize() const { return host_return_queue_.size(); }
    uint32_t reductionQueueSize() const { return reduction_queue_count_; }
    bool reductionBatchInflight() const {
        return reduction_batch_inflight_;
    }
    const CSCPartialResultBurst& residentBurst(uint32_t, uint32_t) const;
    const std::vector<CSCPartialResultWriteTrace>& writeTrace() const {
        return write_trace_;
    }

  private:
    friend struct CSCM7BHostReadbackTestAccess;
    friend struct CSCM7BHostReductionTestAccess;
    struct InflightWrite {
        CSCPartialResultBurst burst{};
        uint32_t buffer_slot = 0;
    };
    struct BGState {
        std::array<CSCPartialResultRecordEnvelope,
                   kCSCPartialRecordsPerBurst> packer{};
        uint32_t packer_count = 0;
        uint64_t next_burst_sequence = 1;
        uint64_t last_output_sequence = 0;
        bool tail_flushed = false;
        bool lifecycle_complete = false;
        std::deque<CSCPartialResultBurst> pending;
        std::deque<InflightWrite> inflight;
        std::vector<std::optional<CSCPartialResultBurst>> resident;
        std::vector<bool> buffer_reserved;
        std::vector<bool> read_inflight;
        uint32_t reserved_buffer_slots = 0;
        uint64_t next_read_burst_sequence = 1;
    };
    struct ReadTransaction {
        CSCPartialResultBurst burst{};
        uint32_t buffer_slot = 0;
        uint32_t channel_id = 0;
        uint32_t rank_id = 0;
        uint32_t local_bg_id = 0;
        uint64_t issue_cycle = 0;
        uint64_t completion_cycle = 0;
    };
    struct ReductionRecord {
        CSCPartialResultRecordEnvelope envelope{};
        uint32_t record_index = 0;
        uint64_t eligible_cycle = 0;
    };

    void fail(const std::string&) noexcept;
    bool validateReservation(const CSCBGAOutputPortValue&);
    void formBurst(uint32_t global_bg, bool tail) noexcept;
    void completeWrites();
    void flushEligibleTails(const std::vector<bool>& final_drain_complete);
    void issueWrites();
    void updateCompletion();
    void stepHostReadback();
    void completeReads();
    void issueReads();
    void updateReadTransportCompletion();
    void stepHostReduction();
    void completeReductionBatch();
    void acceptReturnedBurstForReduction();
    void issueReductionBatch();
    void updateHostCompletion();
    void enqueueReductionRecord(const ReductionRecord&) noexcept;
    ReductionRecord dequeueReductionRecord() noexcept;
    const ReductionRecord& frontReductionRecord() const;
    bool reductionConservationInvariant() const;
    bool findReadCandidate(uint32_t global_bg, uint32_t& slot) const;
    uint64_t totalInflightReads() const;
    bool readbackConservationInvariant() const;
    uint32_t freeBufferSlot(uint32_t) const;
    uint64_t totalInflightWrites() const;
    bool conservationInvariant() const;

    CSCPartialResultPathConfig config_{};
    uint32_t rows_ = 0;
    uint32_t generation_ = 0;
    uint64_t cycle_ = 0;
    std::vector<BGState> bg_;
    std::vector<uint32_t> rank_rr_cursor_;
    std::optional<CSCBGAOutputPortValue> reservation_;
    CSCPartialResultPathCounters counters_{};
    CSCHostReadbackCounters readback_counters_{};
    CSCHostReductionCounters reduction_counters_{};
    std::vector<std::deque<ReadTransaction>> inflight_reads_by_channel_;
    std::vector<uint32_t> channel_read_rr_cursor_;
    std::deque<CSCHostReturnedBurst> host_return_queue_;
    uint32_t reserved_host_return_slots_ = 0;
    std::vector<std::optional<ReductionRecord>> reduction_queue_;
    uint32_t reduction_queue_head_ = 0;
    uint32_t reduction_queue_tail_ = 0;
    uint32_t reduction_queue_count_ = 0;
    std::vector<ReductionRecord> reduction_batch_;
    bool reduction_batch_inflight_ = false;
    uint64_t reduction_batch_completion_cycle_ = 0;
    std::vector<float> final_y_fp32_;
    std::vector<bool> row_reduced_;
    std::vector<uint32_t> row_first_global_bg_;
    std::vector<CSCPartialResultWriteTrace> write_trace_;
    uint64_t write_trace_capacity_ = 0;
    bool error_ = false;
    std::string error_message_;
};

}  // namespace csc_descriptor

#endif
