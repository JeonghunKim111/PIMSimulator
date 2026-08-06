#ifndef CSC_FP16_M7_H
#define CSC_FP16_M7_H

#include "csc/CSCFp16NativeExecution.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace csc_descriptor {

enum class CSCExecutionMode { COMPUTE_ONLY, BGA_VALIDATION, END_TO_END_TIMED };

struct CSCFp16ValidationResult {
    std::vector<CSCFp16Bits> final_y_bits;
    uint64_t host_add_count = 0;
    uint64_t rows_touched = 0;
    uint64_t cross_bg_same_row_adds = 0;
    uint64_t final_y_hash = 0;
};

CSCFp16ValidationResult reduceCapturedFp16BGAOutputs(
    uint32_t rows, const std::vector<CSCFp16BGAOutputEvent>& events);

struct CSCFp16ResultRegionBG {
    uint64_t assigned_nnz = 0;
    uint64_t descriptor_count = 0;
    uint64_t upper_bound_output_records = 0;
    uint64_t required_result_bursts = 0;
    uint64_t allocated_result_bursts = 0;
};

struct CSCFp16ResultRegionPreflight {
    std::array<CSCFp16ResultRegionBG, 64> bg{};
    uint64_t total_upper_bound_records = 0;
    uint64_t total_required_result_bytes = 0;
    uint64_t maximum_per_bg_result_bytes = 0;
    bool allocation_success = false;
};

CSCFp16ResultRegionPreflight preflightFp16ResultRegion(
    const CSCFp16ExecutionImage& image);

struct CSCFp16M7Result {
    CSCExecutionMode mode = CSCExecutionMode::COMPUTE_ONLY;
    std::string precision = "FP16";
    std::string configuration_preset;
    uint32_t rows = 0, columns = 0;
    uint64_t nnz = 0, descriptor_count = 0, active_bg_count = 0;
    uint64_t matrix_fingerprint = 0, mapping_fingerprint = 0;
    std::optional<uint64_t> compute_complete_cycle;
    std::optional<uint64_t> bga_complete_cycle;
    std::optional<uint64_t> writeback_complete_cycle;
    std::optional<uint64_t> readback_complete_cycle;
    std::optional<uint64_t> host_reduction_complete_cycle;
    std::optional<uint64_t> end_to_end_cycle;
    std::optional<uint64_t> partial_count, partial_trace_hash;
    std::optional<uint64_t> bga_output_count, bga_output_hash;
    std::optional<uint64_t> transport_record_count, write_bursts, read_bursts;
    std::optional<uint64_t> write_bytes, read_bytes;
    std::optional<uint64_t> final_y_hash, host_add_count, rows_touched;
    std::vector<CSCFp16Bits> final_y_bits;
    CSCFp16ResultRegionPreflight preflight{};
    std::string toJson() const;
};

void publishFp16M7Artifacts(const CSCFp16M7Result&,
                            const std::string& output_directory);

CSCFp16M7Result runFp16M7(
    std::shared_ptr<const CSCFp16ExecutionImage> image,
    CSCExecutionMode mode, uint64_t max_cycles = 100000000);

struct CSCPairedIdentity {
    uint32_t rows = 0, columns = 0;
    uint64_t nnz = 0;
    uint64_t source_matrix_fingerprint = 0;
    uint64_t row_index_fingerprint = 0;
    uint64_t mapping_fingerprint = 0;
    uint64_t x_source_fingerprint = 0;
};
void validatePairedIdentity(const CSCPairedIdentity&, const CSCPairedIdentity&);

struct CSCAccuracyMetrics {
    double absolute_l2 = 0, relative_l2 = 0, rmse = 0, normalized_rmse = 0;
    double maximum_absolute = 0, maximum_finite_relative = 0, mean_absolute = 0;
    uint64_t nan_rows = 0, positive_inf_rows = 0, negative_inf_rows = 0;
    uint64_t finite_to_nonfinite_mismatch = 0, sign_mismatch = 0;
    uint64_t zero_nonzero_mismatch = 0, signed_zero_mismatch = 0;
};
std::vector<double> cscFp64Oracle(const CSCFp16ImageSource& source,
                                  bool quantize_inputs);
CSCAccuracyMetrics compareFp16ToFp64(const std::vector<CSCFp16Bits>&,
                                     const std::vector<double>& reference);
CSCAccuracyMetrics compareFp64ToFp64(const std::vector<double>& values,
                                     const std::vector<double>& reference);
struct CSCFp16AccuracyBreakdown {
    CSCAccuracyMetrics total_error;
    CSCAccuracyMetrics input_quantization_error;
    CSCAccuracyMetrics arithmetic_error;
};
CSCFp16AccuracyBreakdown evaluateFp16Accuracy(
    const CSCFp16ImageSource&, const std::vector<CSCFp16Bits>& architectural);

struct CSCPairedCycleReport {
    uint64_t fp32_compute_cycles = 0, fp16_compute_cycles = 0;
    uint64_t fp32_end_to_end_cycles = 0, fp16_end_to_end_cycles = 0;
    double compute_speedup = 0, end_to_end_speedup = 0;
};
CSCPairedCycleReport makePairedCycleReport(
    const CSCPairedIdentity& fp32_identity,
    const CSCPairedIdentity& fp16_identity,
    uint64_t fp32_compute, uint64_t fp16_compute,
    uint64_t fp32_end_to_end, uint64_t fp16_end_to_end);

}  // namespace csc_descriptor
#endif
