#ifndef CSC_FP16_NATIVE_EXECUTION_H
#define CSC_FP16_NATIVE_EXECUTION_H

#include "csc/CSCFp16DescriptorEngine.h"
#include "csc/CSCFp16BankGroupAccumulator.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace DRAMSim {
class MultiChannelMemorySystem;
class PIMBlock;
struct RequestToken;
}

namespace csc_descriptor {

struct CSCFp16NativeRequestRecord {
    CSCFp16Request request{};
    uint64_t physical_address = 0;
    uint64_t first_attempt_cycle = 0;
    uint64_t accepted_cycle = 0;
    uint64_t response_cycle = 0;
};

struct CSCFp16NativeTimingCounters {
    uint64_t launch_cycle = 0;
    uint64_t first_descriptor_cycle = 0;
    uint64_t first_request_attempt_cycle = 0;
    uint64_t first_accepted_request_cycle = 0;
    uint64_t first_response_cycle = 0;
    uint64_t first_complete_operands_cycle = 0;
    uint64_t first_pim_mul_cycle = 0;
    uint64_t first_generated_partial_cycle = 0;
    uint64_t first_accepted_partial_cycle = 0;
    uint64_t last_descriptor_cycle = 0;
    uint64_t last_request_issue_cycle = 0;
    uint64_t last_request_accept_cycle = 0;
    uint64_t last_response_cycle = 0;
    uint64_t last_pim_mul_cycle = 0;
    uint64_t last_generated_partial_cycle = 0;
    uint64_t last_accepted_partial_cycle = 0;
    uint64_t completion_cycle = 0;
    uint64_t request_attempts = 0;
    uint64_t accepted_requests = 0;
    uint64_t rejected_requests = 0;
    uint64_t retry_attempts = 0;
    uint64_t completed_requests = 0;
    uint64_t outstanding_high_water = 0;
    std::array<uint64_t, 4> accepted_by_kind{};
    std::array<uint64_t, 4> completed_by_kind{};
    uint64_t response_latency_sum = 0;
    uint64_t response_latency_min = 0;
    uint64_t response_latency_max = 0;
    uint64_t operand_wait_cycles = 0;
    uint64_t request_queue_stall_cycles = 0;
    uint64_t capture_sink_stall_cycles = 0;
    uint64_t pim_execution_cycles = 0;
    uint64_t total_compute_only_cycles = 0;
    uint64_t first_bga_ingress_attempt_cycle = 0;
    uint64_t first_bga_ingress_accept_cycle = 0;
    uint64_t first_bga_compare_cycle = 0;
    uint64_t first_fp16_add_cycle = 0;
    uint64_t first_bga_output_generated_cycle = 0;
    uint64_t first_bga_output_accepted_cycle = 0;
    uint64_t compute_complete_cycle = 0;
    uint64_t final_drain_start_cycle = 0;
    uint64_t last_bga_output_generated_cycle = 0;
    uint64_t last_bga_output_accepted_cycle = 0;
    uint64_t compute_bga_completion_cycle = 0;
    uint64_t bga_output_sink_stall_cycles = 0;
    uint64_t bga_ingress_stall_global_cycles = 0;
    uint64_t bga_ingress_stall_engine_cycles = 0;
    uint64_t total_compute_bga_cycles = 0;
};

struct CSCFp16NativeCounters {
    CSCFp16EngineCounters engine{};
    CSCFp16NativeTimingCounters timing{};
    CSCFp16BGACounters bga{};
};

class CSCFp16NativeExecution {
  public:
    explicit CSCFp16NativeExecution(
        std::shared_ptr<const CSCFp16ExecutionImage> image,
        std::size_t sink_capacity_per_bg = 4096,
        const CSCFp16BGAConfig* bga_config = nullptr,
        std::size_t bga_output_capacity_per_bg = 4096);
    ~CSCFp16NativeExecution();

    void launch();
    void tick();
    bool done() const;
    bool failed() const { return failed_; }
    const std::string& error() const { return error_; }
    uint64_t cycle() const { return cycle_; }
    bool hasOutstanding() const { return !outstanding_.empty(); }
    void setRejectBudget(uint32_t global_bg, uint64_t attempts);
    void setSinkEnabled(uint32_t global_bg, bool enabled);
    CSCFp16PartialEvent popCaptured(uint32_t global_bg);
    const CSCFp16BoundedCaptureSink& sink(uint32_t global_bg) const;
    CSCFp16BoundedCaptureSink& sink(uint32_t global_bg);
    uint64_t physicalAddress(const CSCFp16Request& request) const;
    const std::vector<CSCFp16NativeRequestRecord>& requestTrace() const {
        return request_trace_;
    }
    CSCFp16NativeCounters counters() const;
    bool bgaEnabled() const { return bga_enabled_; }
    const CSCFp16BankGroupAccumulator& bga(uint32_t global_bg) const;
    const CSCFp16BoundedBGAOutputSink& bgaOutputSink(uint32_t global_bg) const;
    CSCFp16BoundedBGAOutputSink& bgaOutputSink(uint32_t global_bg);
    CSCFp16BGAOutputEvent popBGAOutput(uint32_t global_bg);
    void setBGAOutputSinkEnabled(uint32_t global_bg, bool enabled);
    const std::vector<CSCFp16PartialEvent>& bgaIngressTrace(uint32_t global_bg) const;

  private:
    struct Outstanding {
        CSCFp16Request request{};
        uint64_t physical_address = 0;
        uint64_t first_attempt_cycle = 0;
        uint64_t accepted_cycle = 0;
    };
    struct Impl;
    class BGAIngressAdapter;

    bool submit(const CSCFp16Request& request);
    void tokenComplete(unsigned channel, const DRAMSim::RequestToken& token,
                       uint64_t when);
    void legacyComplete(unsigned, uint64_t, uint64_t) {}
    std::array<uint8_t, 32> payloadFor(const Outstanding& request) const;
    void latchFailure(const std::string& message);
    bool allEnginesDone() const;
    bool allBGAsDone() const;

    std::shared_ptr<const CSCFp16ExecutionImage> image_;
    std::unique_ptr<Impl> impl_;
    std::vector<std::unique_ptr<DRAMSim::PIMBlock>> datapaths_;
    std::vector<std::unique_ptr<CSCFp16BoundedCaptureSink>> sinks_;
    std::vector<std::unique_ptr<CSCFp16BankGroupAccumulator>> bgas_;
    std::vector<std::unique_ptr<CSCFp16BoundedBGAOutputSink>> bga_output_sinks_;
    std::vector<std::unique_ptr<BGAIngressAdapter>> bga_ingress_adapters_;
    std::vector<std::unique_ptr<CSCFp16DescriptorEngine>> engines_;
    std::map<uint64_t, Outstanding> outstanding_;
    std::map<uint64_t, uint64_t> first_attempt_cycles_;
    std::array<uint64_t, 64> reject_budget_{};
    std::vector<CSCFp16NativeRequestRecord> request_trace_;
    CSCFp16NativeTimingCounters timing_{};
    uint64_t cycle_ = 0;
    bool bga_enabled_ = false;
    std::array<bool, 64> bga_done_signaled_{};
    bool launched_ = false;
    bool failed_ = false;
    std::string error_;
};

uint64_t cscFp16PartialTraceFnv1a64(
    const std::vector<CSCFp16PartialEvent>& trace);

}  // namespace csc_descriptor

#endif
