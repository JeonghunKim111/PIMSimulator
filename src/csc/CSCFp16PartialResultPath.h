#ifndef CSC_FP16_PARTIAL_RESULT_PATH_H
#define CSC_FP16_PARTIAL_RESULT_PATH_H

#include "csc/CSCFp16BankGroupAccumulator.h"

#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace csc_descriptor {

constexpr uint32_t kCSCFp16TransportRecordBytes = 8;
constexpr uint32_t kCSCFp16TransportBurstBytes = 32;
constexpr uint32_t kCSCFp16TransportRecordsPerBurst = 4;

struct CSCFp16TransportRecord {
    uint32_t row_idx = 0;
    CSCFp16Bits value_bits = 0;
    uint16_t reserved = 0;
};

std::array<uint8_t, 8> serializeCSCFp16TransportRecord(
    const CSCFp16TransportRecord&);
CSCFp16TransportRecord deserializeCSCFp16TransportRecord(
    const uint8_t* bytes, std::size_t size);

struct CSCFp16TransportConfig {
    uint32_t global_bg_count = 64;
    uint32_t channel_count = 16;
    uint32_t bank_groups_per_rank = 4;
    uint32_t buffer_capacity_bursts_per_bg = 16;
    uint32_t pending_capacity_bursts_per_bg = 4;
    uint32_t write_latency_cycles = 4;
    uint32_t max_inflight_writes_per_bg = 2;
    uint32_t write_issue_limit_per_rank_per_cycle = 1;
    uint32_t read_latency_cycles = 4;
    uint32_t read_issue_limit_per_channel_per_cycle = 1;
    uint32_t max_inflight_reads_per_channel = 2;
    bool host_reduction_enabled = true;
    uint32_t host_reduce_records_per_cycle = 4;
    uint32_t host_reduce_latency_cycles = 2;
    uint32_t write_reject_attempts = 0;
    uint32_t read_reject_attempts = 0;
    void validate() const;
};

struct CSCFp16TransportBurst {
    uint32_t global_bg_id = 0;
    uint64_t sequence = 0;
    uint8_t valid_record_count = 0;
    bool tail = false;
    std::array<uint8_t, 32> bytes{};
    std::array<uint64_t, 4> contribution_counts{};
    uint64_t formation_cycle = 0;
    uint64_t issue_cycle = 0;
    uint64_t completion_cycle = 0;
};

struct CSCFp16TransportCounters {
    uint64_t outputs_offered = 0, outputs_accepted = 0;
    uint64_t records_packed = 0, full_bursts = 0, tail_bursts = 0;
    uint64_t tail_padding_bytes = 0, transport_stall_cycles = 0;
    uint64_t write_attempts = 0, writes_accepted = 0, write_retries = 0;
    uint64_t write_completions = 0, outstanding_write_high_water = 0;
    uint64_t write_bytes = 0, resident_bursts = 0;
    uint64_t read_attempts = 0, reads_accepted = 0, read_retries = 0;
    uint64_t read_completions = 0, outstanding_read_high_water = 0;
    uint64_t read_bytes = 0, records_decoded = 0, records_reduced = 0;
    uint64_t fp16_host_adds = 0, cross_bg_same_row_adds = 0;
    uint64_t rows_touched = 0;
    uint64_t first_output_accept_cycle = 0, first_write_issue_cycle = 0;
    uint64_t first_write_complete_cycle = 0, last_write_issue_cycle = 0;
    uint64_t last_write_complete_cycle = 0, writeback_complete_cycle = 0;
    uint64_t readback_start_cycle = 0, first_read_issue_cycle = 0;
    uint64_t first_read_complete_cycle = 0, last_read_complete_cycle = 0;
    uint64_t readback_complete_cycle = 0, first_reduce_cycle = 0;
    uint64_t last_reduce_cycle = 0, reduction_complete_cycle = 0;
    uint64_t end_to_end_cycle = 0;
    std::vector<uint64_t> records_per_bg;
    std::vector<uint64_t> bursts_per_bg;
};

class CSCFp16PartialResultPath {
  public:
    CSCFp16PartialResultPath(const CSCFp16TransportConfig&, uint32_t rows);
    bool ready(const CSCFp16BGAOutputEvent&);
    bool accept(const CSCFp16BGAOutputEvent&);
    void step(uint64_t cycle, const std::vector<bool>& final_drain_complete);
    bool done() const { return counters_.end_to_end_cycle != 0 && !error_; }
    bool writebackComplete() const { return counters_.writeback_complete_cycle != 0; }
    bool readbackComplete() const { return counters_.readback_complete_cycle != 0; }
    bool reductionComplete() const { return counters_.reduction_complete_cycle != 0; }
    bool hostReductionEnabled() const { return config_.host_reduction_enabled; }
    bool failed() const { return error_; }
    const std::string& error() const { return error_message_; }
    const CSCFp16TransportCounters& counters() const { return counters_; }
    const std::vector<CSCFp16Bits>& finalYBits() const;
    const std::vector<CSCFp16BGAOutputEvent>& acceptedTrace() const {
        return accepted_trace_;
    }
    const std::vector<CSCFp16TransportBurst>& residentTrace() const {
        return resident_trace_;
    }
  private:
    struct Inflight { CSCFp16TransportBurst burst; uint64_t completion = 0; };
    struct BGState {
        std::array<CSCFp16BGAOutputEvent, 4> packer{};
        uint8_t packer_count = 0;
        uint64_t next_sequence = 1, last_output_sequence = 0;
        bool tail_flushed = false, lifecycle_complete = false;
        std::deque<CSCFp16TransportBurst> pending;
        std::deque<Inflight> writes;
        std::vector<CSCFp16TransportBurst> resident;
    };
    struct ReadInflight { CSCFp16TransportBurst burst; uint64_t completion = 0; };
    void fail(const std::string&);
    void formBurst(uint32_t bg, bool tail);
    void completeWrites();
    void issueWrites();
    void completeReads();
    void issueReads();
    void reduceOrdered();
    bool allWritesComplete() const;
    bool allReadsComplete() const;
    uint64_t totalInflightWrites() const;
    uint64_t totalInflightReads() const;

    CSCFp16TransportConfig config_;
    uint32_t rows_ = 0;
    uint64_t cycle_ = 0;
    std::vector<BGState> bg_;
    std::vector<uint32_t> rank_rr_, channel_rr_;
    std::vector<std::deque<ReadInflight>> reads_by_channel_;
    std::map<std::pair<uint32_t,uint64_t>, CSCFp16TransportBurst> returned_;
    uint32_t reduce_bg_ = 0;
    uint64_t reduce_sequence_ = 1;
    uint8_t reduce_slot_ = 0;
    uint64_t reduction_ready_cycle_ = 0;
    std::vector<CSCFp16Bits> final_y_;
    std::vector<bool> row_touched_;
    std::vector<uint32_t> row_first_bg_;
    CSCFp16TransportCounters counters_;
    std::vector<CSCFp16BGAOutputEvent> accepted_trace_;
    std::vector<CSCFp16TransportBurst> resident_trace_;
    bool readback_started_ = false, error_ = false;
    uint32_t write_reject_budget_ = 0, read_reject_budget_ = 0;
    std::string error_message_;
};

uint64_t cscFp16TransportRecordTraceFnv1a64(
    const std::vector<CSCFp16BGAOutputEvent>&);
uint64_t cscFp16BurstTraceFnv1a64(const std::vector<CSCFp16TransportBurst>&);
uint64_t cscFp16FinalYFnv1a64(const std::vector<CSCFp16Bits>&);

}  // namespace csc_descriptor
#endif
