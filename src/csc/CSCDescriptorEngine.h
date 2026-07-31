#ifndef CSC_DESCRIPTOR_ENGINE_H
#define CSC_DESCRIPTOR_ENGINE_H

#include "BGTargetedOperation.h"
#include "PIMBlock.h"
#include "csc/CSCBGAIntegration.h"
#include "csc/CSCRequestTracker.h"
#include "csc/CSCTypes.h"

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace DRAMSim { class PIMRank; }

namespace csc_descriptor {

struct CSCFreezeTestAccess;
struct CSCAcceptanceTestAccess;
struct CSCM6CompletionTestAccess;

enum class CSCDescriptorState {
    IDLE, FETCH_DESCRIPTOR, LOAD_X, FETCH_VALUE, FETCH_ROW_INDEX, WAIT_OPERANDS,
    SIMD_MUL, WAIT_TARGET_GRANT, WAIT_TARGET_COMPLETION, EMIT_PARTIALS, ADVANCE_CHUNK, NEXT_DESCRIPTOR, DONE, ERROR, FLUSHING
};
enum class CSCRequestPolicy { SERIALIZED, OVERLAPPED };
enum class CSCSchedulingPolicy { BG_DECOUPLED, BARRIER_LOCKSTEP_REFERENCE };
enum class CSCResultStatus { RUNNING, COMPLETE, INCOMPLETE_FLUSHED, ERROR };

struct CSCPartial {
    uint32_t row_idx, global_bg_id, descriptor_index, chunk_index, lane;
    float value;
};
struct CSCBGImageView {
    const std::vector<uint8_t>* values = nullptr;
    const std::vector<uint8_t>* row_indices = nullptr;
    const std::vector<CSCDescriptor>* descriptors = nullptr;
    const std::vector<float>* packed_x = nullptr;
};
struct CSCEngineCounters {
    uint64_t descriptor_fetches = 0, x_scalar_loads = 0, x_transactions = 0;
    uint64_t value_transactions = 0, index_transactions = 0, logical_mul_events = 0;
    uint64_t full_chunks = 0, tail_chunks = 0, total_chunks = 0;
    uint64_t active_lanes = 0, available_lanes = 0;
    uint64_t invalid_lane_multiplies = 0, invalid_lane_writes = 0;
    uint64_t generated_partials = 0, emitted_partials = 0, host_accumulations = 0;
    uint64_t stall_cycles_request = 0, stall_cycles_backpressure = 0, busy_cycles = 0;
    uint64_t total_execution_cycles = 0;
    uint64_t generated_bga_batches = 0, generated_bga_partials = 0;
    uint64_t accepted_bga_batches = 0, accepted_bga_partials = 0;
    uint64_t bga_backpressure_cycles = 0, bga_duplicate_errors = 0;
    uint64_t bga_protocol_errors = 0;
};

class CSCDescriptorEngine {
  public:
    using Submit = std::function<bool(CSCDescriptorEngine*, CSCRequestKind, uint64_t)>;
    using TokenSubmit = std::function<bool(CSCDescriptorEngine*, const DRAMSim::RequestToken&, uint64_t)>;
    using TokenFactory = std::function<DRAMSim::RequestToken(uint32_t, CSCRequestKind, uint64_t)>;
    using Address = std::function<uint64_t(uint32_t, CSCRequestKind, uint64_t)>;
    using TargetSubmit = std::function<bool(CSCDescriptorEngine*, const DRAMSim::BGTargetedOperation&)>;
    using TargetStatus = std::function<DRAMSim::BGTargetedOperationState(const DRAMSim::BGTargetedIdentity&)>;
    using TargetPoll = std::function<bool(const DRAMSim::BGTargetedIdentity&, DRAMSim::BGTargetedCompletion&)>;

    CSCDescriptorEngine(uint32_t, DRAMSim::PIMBlock*, Submit, Address);
    CSCDescriptorEngine(uint32_t, DRAMSim::PIMBlock*, TokenSubmit, TokenFactory, Address);
    CSCDescriptorEngine(uint32_t, TokenSubmit, TokenFactory, Address, TargetSubmit, TargetStatus,
                        TargetPoll);
    void launch(const CSCBGImageView&, std::vector<CSCPartial>*);
    void tick();
    void flush();
    void flushIncomplete();
    void reset();
    void configureBGAIntegration(const CSCBGAIntegrationConfig&,
                                 CSCBGAProducerCallbacks);
    bool onRequestComplete(unsigned, uint64_t, uint64_t);
    bool onRequestComplete(unsigned, const DRAMSim::RequestToken&, uint64_t);

    bool busy() const { return busy_; }
    bool isTerminal() const { return state_ == CSCDescriptorState::DONE || state_ == CSCDescriptorState::ERROR; }
    bool isDone() const { return state_ == CSCDescriptorState::DONE; }
    bool done() const { return isTerminal(); }
    bool hasFailed() const { return state_ == CSCDescriptorState::ERROR; }
    CSCError errorCode() const { return error_code_; }
    const std::string& errorMessage() const { return error_; }
    bool flushRequested() const { return flush_requested_; }
    bool flushComplete() const { return flush_completed_; }
    CSCDescriptorState state() const { return state_; }
    const std::string& error() const { return error_; }
    const CSCEngineCounters& counters() const { return counters_; }
    const CSCRequestTracker& tracker() const { return tracker_; }
    uint32_t globalBG() const { return global_bg_id_; }
    uint64_t maximumOutstanding() const { return maximum_outstanding_; }
    uint64_t progressEpoch() const { return progress_epoch_; }
    bool bgaIntegrationEnabled() const { return bga_config_.enabled; }
    const CSCBGAIntegrationConfig& bgaIntegrationConfig() const {
        return bga_config_;
    }
    uint64_t nextBGASequence() const { return next_bga_sequence_; }
    const std::optional<CSCBGAPartialBatch>& pendingBGABatch() const {
        return pending_bga_batch_;
    }
    const std::optional<CSCBGAPartialBatch>& lastAcceptedBGABatch() const {
        return last_accepted_bga_batch_;
    }
    uint64_t lastTargetGrantCycle() const { return last_target_grant_cycle_; }
    uint64_t lastTargetCompletionCycle() const { return last_target_completion_cycle_; }
    uint64_t lastTargetPollEngineCycle() const { return last_target_poll_engine_cycle_; }
    uint64_t lastBGAMaterializeEngineCycle() const { return last_bga_materialize_engine_cycle_; }
    uint64_t lastBGAAcceptEngineCycle() const { return last_bga_accept_engine_cycle_; }

  private:
    friend struct CSCFreezeTestAccess;
    friend struct CSCAcceptanceTestAccess;
    struct OperandSlot {
        bool required = false, created = false, accepted = false, ready = false;
        DRAMSim::RequestToken token{};
        uint64_t address = 0;
    };
    void fail(CSCError, const std::string&);
    bool prepare(OperandSlot&, CSCRequestKind, uint64_t);
    void issueWaiting();
    void clearSlot(OperandSlot&);
    void stageCompletion(OperandSlot&);
    void validateLaunch() const;
    bool descriptorValid(const CSCDescriptor&) const;
    void finishWork();
    OperandSlot* slotFor(CSCRequestKind);

    uint32_t global_bg_id_, descriptor_pointer_ = 0, remaining_nnz_ = 0, chunk_offset_ = 0;
    uint32_t valid_count_ = 0, chunk_index_ = 0;
    uint64_t local_sequence_ = 0, maximum_outstanding_ = 0;
    uint64_t progress_epoch_ = 0;
    uint64_t next_bga_sequence_ = 1, descriptor_chunk_ordinal_ = 1;
    uint32_t completed_target_generation_ = 0;
    uint64_t last_target_grant_cycle_ = 0, last_target_completion_cycle_ = 0;
    uint64_t last_target_poll_engine_cycle_ = 0;
    uint64_t last_bga_materialize_engine_cycle_ = 0, last_bga_accept_engine_cycle_ = 0;
    CSCDescriptor current_{};
    float x_j_ = 0;
    std::array<uint8_t, 32> value_staging_{}, index_staging_{};
    DRAMSim::BurstType result_staging_;
    DRAMSim::PIMBlock* datapath_;
    TargetSubmit target_submit_;
    TargetStatus target_status_;
    TargetPoll target_poll_;
    DRAMSim::BGTargetedOperation target_operation_{};
    bool target_operation_valid_ = false;
    uint64_t target_sequence_ = 0;
    Submit legacy_submit_;
    TokenSubmit token_submit_;
    TokenFactory token_factory_;
    Address address_;
    CSCBGImageView image_;
    std::vector<CSCPartial>* sink_ = nullptr;
    OperandSlot x_slot_, value_slot_, index_slot_;
    CSCRequestTracker tracker_;
    CSCDescriptorState state_ = CSCDescriptorState::IDLE;
    bool busy_ = false, done_ = false, flush_requested_ = false, flush_completed_ = false;
    bool incomplete_flush_requested_ = false;
    CSCError error_code_ = CSCError::NONE;
    std::string error_;
    CSCEngineCounters counters_;
    CSCBGAIntegrationConfig bga_config_{};
    CSCBGAProducerCallbacks bga_producer_callbacks_{};
    std::optional<CSCBGAPartialBatch> pending_bga_batch_;
    std::optional<CSCBGAPartialBatch> last_accepted_bga_batch_;
    bool bga_configuration_locked_ = false;
};

struct CSCExecutionCounters : CSCEngineCounters {
    uint64_t created_requests = 0, accepted_requests = 0, completion_count = 0;
    uint64_t abandoned_waiting_requests = 0;
    uint64_t submit_attempts = 0, submit_rejections = 0, retry_count = 0;
    uint64_t unknown_completions = 0, duplicate_completions = 0;
    uint64_t stale_generation_completions = 0, request_kind_mismatches = 0;
    uint64_t owner_mismatches = 0, completion_after_retirement = 0;
    uint64_t maximum_waiting_to_submit_depth = 0, same_address_concurrent_requests = 0;
    uint64_t maximum_global_outstanding = 0, maximum_outstanding_per_engine = 0;
    uint64_t maximum_outstanding_per_channel = 0;
    uint64_t cycles_with_outstanding_ge_1 = 0, cycles_with_outstanding_ge_2 = 0;
    uint64_t outstanding_cycle_integral = 0;
    uint64_t launch_to_done_cycles = 0;
    std::array<uint64_t, kCSCGlobalBGCount> per_bg_ready_cycles{}, per_bg_executing_cycles{};
    std::array<uint64_t, kCSCGlobalBGCount> per_bg_memory_wait_cycles{}, per_bg_grant_wait_cycles{};
    std::array<uint64_t, kCSCGlobalBGCount> per_bg_flush_drain_cycles{};
    std::array<uint64_t, kCSCGlobalBGCount> per_bg_barrier_wait_cycles{};
    std::array<uint64_t, kCSCGlobalBGCount> per_bg_idle_cycles{};
    std::array<uint64_t, kCSCGlobalBGCount> per_bg_completion_cycle{};
    CSCSchedulingPolicy scheduling_policy = CSCSchedulingPolicy::BG_DECOUPLED;
    uint64_t total_cycles = 0, total_barrier_wait_cycles = 0;
    uint64_t memory_requests_accepted = 0, memory_requests_completed = 0;
    uint64_t targeted_ops_accepted = 0, targeted_ops_completed = 0;
    uint64_t partial_results_emitted = 0;
    std::array<uint64_t, 128> per_pimblock_active_cycles{}, per_pimblock_targeted_ops{};
    uint64_t rank_targeted_grants = 0, rank_command_bus_stall_cycles = 0;
    uint64_t rank_resource_conflict_stall_cycles = 0, rank_mode_drain_cycles = 0;
    uint64_t round_robin_skip_count = 0;
    double average_outstanding = 0;
    std::array<uint64_t, kCSCRequestKindCount> issued_by_kind{}, completed_by_kind{};
    std::array<uint64_t, kCSCRequestKindCount> abandoned_by_kind{};
    std::array<uint64_t, kCSCGlobalBGCount> issued_by_bg{}, completed_by_bg{}, abandoned_by_bg{};
};

class CSCNativeExecution {
  public:
    explicit CSCNativeExecution(
        CSCRequestPolicy policy = CSCRequestPolicy::OVERLAPPED,
        CSCSchedulingPolicy scheduling_policy = CSCSchedulingPolicy::BG_DECOUPLED);
    ~CSCNativeExecution();
    void launch(const std::array<CSCBGImageView, 64>&, uint32_t, uint64_t);
    void configureBGAIntegration(const CSCBGAIntegrationConfig&,
                                 CSCBGAProducerCallbacks,
                                 CSCBGAOutputPortCallbacks);
    void enableProductionBGAIntegration(const CSCBGAIntegrationConfig&);
    void tick();
    bool busy() const;
    bool done() const { return isTerminal(); }
    void flush();
    void flushBG(uint32_t);
    bool resetBG(uint32_t);
    void reset();
    bool isTerminal() const;
    bool isDone() const;
    bool hasFailed() const { return failed_; }
    CSCError errorCode() const { return error_code_; }
    const std::string& errorMessage() const { return error_message_; }
    int32_t failedEngine() const { return failed_engine_; }
    bool flushRequested() const { return flush_requested_; }
    bool flushComplete() const { return flush_completed_; }
    CSCResultStatus bgResultStatus(uint32_t bg) const { return result_status_.at(bg); }
    bool resultValid() const;
    bool productionBGAEnabled() const { return production_bga_enabled_; }
    bool bgaComputeSubmitComplete() const { return bga_config_.enabled && isDone(); }
    bool bgaOutputCompletionImplemented() const { return false; }
    const CSCBankGroupAccumulator& bankGroupAccumulator(uint32_t global_bg) const;
    bool hasOutstandingRequest() const { return !outstanding_.empty(); }
    bool hasPendingTransactions() const;
    bool hasUnconsumedCompletion() const;
    std::vector<float> hostAccumulate();
    const std::vector<CSCPartial>& partials() const { return partials_; }
    CSCExecutionCounters counters() const;
    const CSCDescriptorEngine& engine(uint32_t bg) const { return *engines_.at(bg); }
    uint64_t cycle() const { return cycle_; }
    uint64_t addressFor(uint32_t, CSCRequestKind, uint64_t);
    CSCRequestPolicy policy() const { return policy_; }
    CSCSchedulingPolicy schedulingPolicy() const { return scheduling_policy_; }
    void setSubmitRejectBudget(uint32_t bg, uint64_t count) {
        if (bg >= 64) throw std::out_of_range("CSC BG");
        submit_reject_budget_[bg] = count;
    }

  private:
    friend struct CSCFreezeTestAccess;
    friend struct CSCAcceptanceTestAccess;
    friend struct CSCM6CompletionTestAccess;
    bool submit(CSCDescriptorEngine*, const DRAMSim::RequestToken&, uint64_t);
    DRAMSim::RequestToken makeToken(uint32_t, CSCRequestKind, uint64_t);
    void failGlobal(CSCError, const std::string&, int32_t engine = -1);
    void readComplete(unsigned, uint64_t, uint64_t);
    void tokenComplete(unsigned, const DRAMSim::RequestToken&, uint64_t);
    bool submitTarget(CSCDescriptorEngine*, const DRAMSim::BGTargetedOperation&);
    DRAMSim::BGTargetedOperationState targetStatus(const DRAMSim::BGTargetedIdentity&) const;
    bool pollTarget(const DRAMSim::BGTargetedIdentity&, DRAMSim::BGTargetedCompletion&);
    void latchFailure(const CSCDescriptorEngine&);
    void updateMLP();
    DRAMSim::PIMRank& targetRank(uint32_t);
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::vector<std::unique_ptr<CSCDescriptorEngine>> engines_;
    std::vector<CSCPartial> partials_;
    std::map<uint64_t, CSCDescriptorEngine*> outstanding_;
    std::map<uint64_t, uint64_t> outstanding_addresses_;
    uint64_t next_request_id_ = 1, cycle_ = 0, nnz_ = 0;
    uint32_t generation_ = 0, rows_ = 0;
    bool launched_ = false;
    uint64_t host_accumulations_ = 0;
    bool failed_ = false, flush_requested_ = false, flush_completed_ = false;
    CSCError error_code_ = CSCError::NONE;
    int32_t failed_engine_ = -1;
    std::string error_message_;
    CSCRequestPolicy policy_;
    CSCSchedulingPolicy scheduling_policy_;
    CSCExecutionCounters execution_stats_;
    std::array<uint64_t, 64> submit_reject_budget_{};
    std::array<CSCResultStatus, 64> result_status_{};
    CSCBGAIntegrationConfig bga_config_{};
    CSCBGAOutputPortCallbacks bga_output_callbacks_{};
    bool bga_configuration_locked_ = false;
    bool production_bga_enabled_ = false;
};

}  // namespace csc_descriptor
#endif
