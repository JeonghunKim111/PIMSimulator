#ifndef CSC_BGA_INTEGRATION_H
#define CSC_BGA_INTEGRATION_H

#include "csc/CSCBankGroupAccumulator.h"
#include "csc/CSCTypes.h"

#include <array>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace csc_descriptor {

constexpr uint32_t kCSCM7ALogicalStream = 0;

enum class CSCBGAOutputConsumerMode { EXTERNAL, VALIDATION_ROUND_ROBIN };

enum class CSCBGAExecutionState {
    DISABLED, RUNNING_COMPUTE, DRAINING_BGA,
    WAITING_FOR_EXTERNAL_OUTPUT_CONSUMER, COMPLETE, ERROR
};

struct CSCBGAIntegrationConfig {
    bool enabled = false;
    CSCBGAConfig accumulator{};
    uint32_t logical_stream_id = kCSCM7ALogicalStream;
    bool record_comparison_trace = false;
    CSCBGAOutputConsumerMode output_consumer_mode =
        CSCBGAOutputConsumerMode::EXTERNAL;
    uint32_t validation_consumer_accepts_per_cycle = 1;

    void validate() const
    {
        if (logical_stream_id != kCSCM7ALogicalStream)
            throw std::invalid_argument("M7A-2 requires logical stream zero");
        if (!enabled) return;
        if (output_consumer_mode ==
                CSCBGAOutputConsumerMode::VALIDATION_ROUND_ROBIN &&
            !validation_consumer_accepts_per_cycle)
            throw std::invalid_argument("zero validation consumer width");
        if (accumulator.input_streams != 1 || !accumulator.input_queue_depth ||
            !accumulator.accumulator_entries || !accumulator.compare_width ||
            accumulator.compare_width < accumulator.accumulator_entries ||
            !accumulator.compare_latency || !accumulator.add_latency ||
            !accumulator.output_queue_depth || !accumulator.rows)
            throw std::invalid_argument("invalid enabled M7A BGA configuration");
    }
};

// Integration provenance is deliberately outside CSCBGABatch. Only generation
// and sequence participate in the BGA exactly-once identity. Generation must
// come from the completed targeted-operation identity when EMIT_PARTIALS is wired.
struct CSCBGAPartialBatch {
    uint32_t global_bg_id = 0;
    uint32_t descriptor_index = 0;
    uint32_t chunk_index = 0;
    uint32_t valid_count = 0;
    uint64_t descriptor_chunk_ordinal = 0;
    CSCBGABatch payload{};

    void validate(const CSCBGAIntegrationConfig& config,
                  uint64_t next_bga_sequence,
                  uint64_t last_accepted_sequence = 0) const
    {
        config.validate();
        if (!config.enabled)
            throw std::logic_error("BGA batch offered while integration is disabled");
        if (global_bg_id >= kCSCGlobalBGs)
            throw std::out_of_range("M7A global BG provenance");
        if (payload.logical_stream_id != config.logical_stream_id ||
            payload.logical_stream_id != kCSCM7ALogicalStream)
            throw std::invalid_argument("invalid M7A logical stream");
        if (!payload.generation || !payload.sequence || !next_bga_sequence ||
            (payload.sequence != next_bga_sequence &&
             payload.sequence != last_accepted_sequence))
            throw std::invalid_argument("invalid M7A BGA exactly-once identity");
        if (!descriptor_chunk_ordinal ||
            descriptor_chunk_ordinal != payload.sequence)
            throw std::invalid_argument("descriptor/chunk ordinal mismatch");
        if (!valid_count || valid_count > 8 ||
            payload.partials.size() != valid_count)
            throw std::invalid_argument("M7A valid lane count mismatch");
        for (const auto& partial : payload.partials)
            if (partial.row_idx >= config.accumulator.rows)
                throw std::out_of_range("M7A BGA partial row");
    }
};

struct CSCBGAProducerIdentity {
    uint32_t global_bg_id = 0;
    uint32_t logical_stream_id = kCSCM7ALogicalStream;
    uint32_t generation = 0;
};

struct CSCBGAOutputPortValue {
    uint32_t global_bg_id = 0;
    CSCBGAOutput payload{};
};

// std::function objects are copied into their owner. Captures by reference must
// outlive that owner (or the next pre-launch reconfiguration). Producer
// callbacks are owned by CSCDescriptorEngine.
struct CSCBGAProducerCallbacks {
    using Submit =
        std::function<CSCBGAInputResult(const CSCBGAPartialBatch&)>;
    using ProducerDone = std::function<void(const CSCBGAProducerIdentity&)>;
    using RequestFinalDrain =
        std::function<bool(const CSCBGAProducerIdentity&)>;

    Submit submit;
    ProducerDone producer_done;
    RequestFinalDrain request_final_drain;

    bool complete() const {
        return bool(submit) && bool(producer_done) &&
               bool(request_final_drain);
    }
};

// Output-port callbacks are owned once by CSCNativeExecution, never copied to
// descriptor engines. A PeekOutput reference must remain valid until the
// matching AcceptOutput call and BGA retirement step.


// Validation-only collector. The row sums are a CPU semantic oracle, not a
// native PIM final-y or a model of TSV/GA timing.
struct CSCBGAValidationCollector {
    std::array<uint64_t, kCSCGlobalBGs> accepted_outputs_by_bg{};
    std::array<uint64_t, kCSCGlobalBGs> last_output_sequence_by_bg{};
    uint64_t accepted_outputs = 0;
    uint64_t capacity_outputs = 0;
    uint64_t final_drain_outputs = 0;
    uint64_t accepted_contributions = 0;
    std::vector<double> validation_y_double;
    std::vector<double> shadow_accepted_y_double;
    std::vector<uint32_t> accepted_bg_trace_sample;
    bool trace_sample_truncated = false;
};

struct CSCBGAOutputPortCallbacks {
    using HasOutput = std::function<bool(uint32_t global_bg_id)>;
    using PeekOutput =
        std::function<const CSCBGAOutput&(uint32_t global_bg_id)>;
    using AcceptOutput = std::function<void(uint32_t global_bg_id)>;

    HasOutput has_output;
    PeekOutput peek_output;
    AcceptOutput accept_output;

    bool complete() const
    {
        return bool(has_output) && bool(peek_output) && bool(accept_output);
    }
};
}  // namespace csc_descriptor

#endif
