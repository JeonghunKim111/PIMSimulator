#include "gtest/gtest.h"
#include "tests/csc/CSCExternalImage.h"
#include "tests/csc/CSCFunctionalModel.h"
#include "tests/csc/CSCTimingModel.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
using namespace csc_descriptor;
TEST(CSCExternalTimingBenchmark,ExplicitRun){
 const char*run=std::getenv("CSC_EXTERNAL_TIMING_RUN"),*path=std::getenv("CSC_EXTERNAL_IMAGE");
 if(!run||std::string(run)!="1"||!path||!*path)GTEST_SKIP()<<"set CSC_EXTERNAL_TIMING_RUN=1 and CSC_EXTERNAL_IMAGE";
 auto loaded=loadExternalPhysicalImage(path);auto&l=loaded.layout;
 std::vector<float>x(l.matrix.cols);for(uint32_t i=0;i<x.size();++i)x[i]=float((i%13)+1)/7;
 auto reference=cpuReference(l.matrix,x);auto f=executeLockstep(l,x);float error=0;uint32_t row=UINT32_MAX;
 ASSERT_TRUE(compareResults(reference,f.y,&error,&row))<<"FP32 functional mismatch max_error="<<error<<" row="<<row;
 uint64_t max_bg_chunks=f.stats.bg_chunk_count.empty()?0:*std::max_element(f.stats.bg_chunk_count.begin(),f.stats.bg_chunk_count.end());
 ASSERT_EQ(f.stats.value_burst_read_count,f.stats.total_chunks);ASSERT_EQ(f.stats.row_index_burst_read_count,f.stats.total_chunks);ASSERT_EQ(f.stats.logical_csc_mul_events,f.stats.total_chunks);ASSERT_EQ(f.stats.active_lanes,l.stats.nnz);ASSERT_EQ(f.stats.generated_partial_count,l.stats.nnz);ASSERT_EQ(f.stats.host_accumulation_count,l.stats.nnz);ASSERT_EQ(f.stats.x_scalar_load_count,l.stats.descriptor_count);ASSERT_EQ(f.stats.invalid_lane_multiplication_count,0);ASSERT_EQ(f.stats.invalid_lane_partial_count,0);ASSERT_EQ(f.stats.invalid_lane_accumulation_count,0);
 CSCTimingModel model;model.validateTopologyAddresses(l);auto t=model.run(l,f.stats,ResultTrafficMode::None);
 ASSERT_EQ(t.value_bursts,f.stats.total_chunks);ASSERT_EQ(t.row_index_bursts,f.stats.total_chunks);ASSERT_EQ(t.x_bursts,f.stats.required_unique_x_bursts);ASSERT_EQ(t.result_cycles,0);ASSERT_EQ(t.kernel_resident_cycles,t.value_read_cycles+t.row_index_read_cycles);ASSERT_EQ(t.kernel_with_x_cycles,t.x_load_cycles+t.kernel_resident_cycles);ASSERT_EQ(t.end_to_end_simulated_cycles,t.matrix_preload_cycles+t.kernel_with_x_cycles);
 std::cout<<"CSC_EXTERNAL_TIMING={"
  <<"\"mapping_policy\":\""<<loaded.metadata.mapping_policy<<"\""
  <<",\"rows\":"<<l.matrix.rows<<",\"columns\":"<<l.matrix.cols<<",\"nnz\":"<<l.stats.nnz
  <<",\"descriptors\":"<<l.stats.descriptor_count<<",\"useful_bytes\":"<<(l.stats.useful_value_bytes+l.stats.useful_index_bytes)
  <<",\"physical_bytes\":"<<(l.stats.physical_value_bytes+l.stats.physical_index_bytes)<<",\"padding_bytes\":"<<l.stats.total_alignment_padding_bytes
  <<",\"total_chunks\":"<<f.stats.total_chunks<<",\"simd_utilization\":"<<f.stats.simd_lane_utilization
  <<",\"max_bg_nnz\":"<<l.stats.max_bg_nnz<<",\"max_bg_chunks\":"<<max_bg_chunks
  <<",\"lockstep_rounds\":"<<f.stats.total_lockstep_rounds<<",\"critical_bg\":"<<f.stats.critical_bg
  <<",\"expected_value_transactions\":"<<f.stats.total_chunks<<",\"expected_index_transactions\":"<<f.stats.total_chunks
  <<",\"expected_x_transactions\":"<<f.stats.required_unique_x_bursts<<",\"expected_logical_mul_events\":"<<f.stats.total_chunks
  <<",\"expected_active_lanes\":"<<l.stats.nnz<<",\"expected_partial_count\":"<<l.stats.nnz
  <<",\"actual_value_transactions\":"<<t.value_bursts<<",\"actual_index_transactions\":"<<t.row_index_bursts
  <<",\"actual_x_transactions\":"<<t.x_bursts<<",\"actual_logical_mul_events\":"<<f.stats.logical_csc_mul_events
  <<",\"actual_active_lanes\":"<<f.stats.active_lanes<<",\"actual_partial_count\":"<<f.stats.generated_partial_count
  <<",\"x_scalar_loads\":"<<f.stats.x_scalar_load_count<<",\"invalid_lane_multiplications\":"<<f.stats.invalid_lane_multiplication_count
  <<",\"invalid_lane_partials\":"<<f.stats.invalid_lane_partial_count<<",\"invalid_lane_accumulations\":"<<f.stats.invalid_lane_accumulation_count
  <<",\"functional_pass\":true,\"address_ownership_pass\":true,\"accounting_pass\":true"
  <<",\"preload_cycles\":"<<t.matrix_preload_cycles<<",\"x_load_cycles\":"<<t.x_load_cycles
  <<",\"value_completion_cycles\":"<<t.value_read_cycles<<",\"index_completion_cycles\":"<<t.row_index_read_cycles
  <<",\"resident_cycles\":"<<t.kernel_resident_cycles<<",\"with_x_cycles\":"<<t.kernel_with_x_cycles
  <<",\"end_to_end_cycles\":"<<t.end_to_end_simulated_cycles
  <<",\"resident_boundary_pass\":true,\"with_x_boundary_pass\":true,\"preload_excluded_from_resident\":true}"
  <<std::endl;
}
