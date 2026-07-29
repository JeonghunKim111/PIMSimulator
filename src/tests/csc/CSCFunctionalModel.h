#ifndef CSC_FUNCTIONAL_MODEL_H
#define CSC_FUNCTIONAL_MODEL_H
#include "tests/csc/CSCLayout.h"
namespace csc_descriptor {
constexpr float kAbsoluteTolerance=1e-5f,kRelativeTolerance=1e-5f;
struct ExecutionStats{uint64_t total_chunks=0,full_chunks=0,tail_chunks=0,active_lanes=0,total_available_lanes=0,total_lockstep_rounds=0,x_scalar_load_count=0,x_burst_read_count=0,value_burst_read_count=0,row_index_burst_read_count=0,logical_csc_mul_events=0,generated_partial_count=0,host_accumulation_count=0;double simd_lane_utilization=0,x_packing_wall_ms=0,multiplication_wall_ms=0,host_accumulation_wall_ms=0;uint32_t critical_bg=0;std::vector<uint64_t> bg_chunk_count,bg_idle_round_count;};
struct FunctionalResult{std::vector<float> y;std::vector<std::vector<float>> x_packed;ExecutionStats stats;};
std::vector<float> cpuReference(const CSCMatrix&,const std::vector<float>&);FunctionalResult executeLockstep(const CSCLayout&,const std::vector<float>&);bool compareResults(const std::vector<float>&,const std::vector<float>&,float* = nullptr,uint32_t* = nullptr);
}
#endif
