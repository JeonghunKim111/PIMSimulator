#include "csc/CSCFp16M7.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <limits>
#include <unistd.h>

using namespace csc_descriptor;
namespace fs=std::filesystem;
namespace {
struct Dir{fs::path p;Dir(const char*n){p=fs::current_path()/(std::string(".csc_fp16_m7_")+std::to_string(getpid())+n);fs::remove_all(p);}~Dir(){fs::remove_all(p);}};
CSCFp16ImageSource boundary(){CSCFp16ImageSource s;s.rows=64;s.cols=18;const std::array<uint32_t,18> counts={0,1,7,8,9,15,16,17,31,0,0,0,0,0,0,32,33,0};s.col_ptr.push_back(0);uint64_t n=0;for(uint32_t c=0;c<s.cols;++c){for(uint32_t l=0;l<counts[c];++l){s.row_idx.push_back(l==9?3U:uint32_t((n*29+11)%s.rows));static const std::array<double,11>v={0.0,-0.0,-2.5,std::ldexp(1.0,-24),65504.0,512.0,1.0/3.0,-1.25,2.0,7.0,std::numeric_limits<double>::quiet_NaN()};s.values.push_back(v[n++%v.size()]);}s.col_ptr.push_back(s.values.size());}for(uint32_t c=0;c<s.cols;++c){s.x.push_back(c==16?512.0:c==15?-0.5:1.0+c/8.0);s.column_to_bg.push_back(c%2?7:8);}return s;}
}

TEST(CSCFp16M7ReducerTest, CanonicalizesBgAndSequenceAndRejectsGaps)
{
    std::vector<CSCFp16BGAOutputEvent> events={{0,cscFp16ToBits(cscFp16FromFloat(2)),3,CSCFp16BGAOutputReason::FINAL_DRAIN,1,1},{0,cscFp16ToBits(cscFp16FromFloat(1)),1,CSCFp16BGAOutputReason::FINAL_DRAIN,2,1},{0,cscFp16ToBits(cscFp16FromFloat(4)),1,CSCFp16BGAOutputReason::FINAL_DRAIN,1,1}};
    auto result=reduceCapturedFp16BGAOutputs(4,events);
    EXPECT_EQ(result.host_add_count,3U);EXPECT_EQ(result.cross_bg_same_row_adds,1U);
    EXPECT_EQ(result.final_y_bits[0],cscFp16ToBits(cscFp16FromFloat(7)));
    events[1].sequence=3;
    EXPECT_THROW(reduceCapturedFp16BGAOutputs(4,events),std::invalid_argument);
    events[1].sequence=2;events[0].row_idx=9;
    EXPECT_THROW(reduceCapturedFp16BGAOutputs(4,events),std::invalid_argument);
}

TEST(CSCFp16M7ModeTest, ThreeModesPreserveAllExistingGoldens)
{
    Dir d("modes");exportCSCFp16ImageV2(boundary(),d.p.string());auto image=CSCFp16ExecutionImage::load(d.p.string(),CSCFp16ExecutionMode::FP16_IMAGE_V2);
    auto compute=runFp16M7(image,CSCExecutionMode::COMPUTE_ONLY);
    auto validation=runFp16M7(image,CSCExecutionMode::BGA_VALIDATION);
    auto full=runFp16M7(image,CSCExecutionMode::END_TO_END_TIMED);
    auto repeated=runFp16M7(image,CSCExecutionMode::END_TO_END_TIMED);
    ASSERT_EQ(compute.partial_count,169U);EXPECT_EQ(compute.partial_trace_hash,0x2e2867563f3d9cacULL);EXPECT_EQ(compute.compute_complete_cycle,544U);
    EXPECT_EQ(validation.bga_complete_cycle,564U);EXPECT_EQ(validation.bga_output_count,165U);EXPECT_EQ(validation.bga_output_hash,0x96c3f823f3fe5ab1ULL);
    ASSERT_EQ(validation.final_y_bits,full.final_y_bits);EXPECT_EQ(validation.final_y_hash,0x64a5f1109a6f2bd6ULL);EXPECT_EQ(full.final_y_hash,validation.final_y_hash);
    EXPECT_EQ(full.bga_complete_cycle,564U);EXPECT_EQ(full.writeback_complete_cycle,569U);EXPECT_EQ(full.readback_complete_cycle,618U);EXPECT_EQ(full.end_to_end_cycle,655U);
    EXPECT_EQ(full.toJson(),repeated.toJson());EXPECT_EQ(full.final_y_bits,repeated.final_y_bits);
    EXPECT_EQ(full.bga_output_hash,0x96c3f823f3fe5ab1ULL);
    EXPECT_EQ(full.write_bursts,42U);EXPECT_EQ(full.read_bursts,42U);EXPECT_FALSE(compute.bga_complete_cycle);EXPECT_FALSE(validation.writeback_complete_cycle);
    EXPECT_NE(compute.toJson().find("COMPUTE_ONLY"),std::string::npos);
}

TEST(CSCFp16M7PreflightTest, SizesLogicalRegionFromAssignedNnz)
{
    Dir d("preflight");exportCSCFp16ImageV2(boundary(),d.p.string());auto image=CSCFp16ExecutionImage::load(d.p.string(),CSCFp16ExecutionMode::FP16_IMAGE_V2);auto p=preflightFp16ResultRegion(*image);EXPECT_TRUE(p.allocation_success);EXPECT_EQ(p.total_upper_bound_records,169U);EXPECT_EQ(p.bg[7].required_result_bursts,(p.bg[7].assigned_nnz+3)/4);EXPECT_EQ(p.bg[8].required_result_bursts,(p.bg[8].assigned_nnz+3)/4);EXPECT_EQ(p.total_required_result_bytes,(p.bg[7].required_result_bursts+p.bg[8].required_result_bursts)*32U);
}

TEST(CSCFp16M7EvaluationTest, PairedIdentityAndOraclesSeparateInputQuantization)
{
    CSCPairedIdentity a{2,2,2,1,2,3,4},b=a;EXPECT_NO_THROW(validatePairedIdentity(a,b));auto cycles=makePairedCycleReport(a,b,200,100,400,250);EXPECT_DOUBLE_EQ(cycles.compute_speedup,2.0);EXPECT_DOUBLE_EQ(cycles.end_to_end_speedup,1.6);b.mapping_fingerprint++;EXPECT_THROW(validatePairedIdentity(a,b),std::invalid_argument);
    CSCFp16ImageSource s;s.rows=2;s.cols=2;s.col_ptr={0,1,2};s.row_idx={0,1};s.values={1.0001,0.3333};s.x={2.0001,3.0001};s.column_to_bg={0,1};auto original=cscFp64Oracle(s,false),quantized=cscFp64Oracle(s,true);EXPECT_NE(original,quantized);std::vector<CSCFp16Bits> arch={cscFp16ToBits(cscFp16FromDouble(quantized[0])),cscFp16ToBits(cscFp16FromDouble(quantized[1]))};auto metric=compareFp16ToFp64(arch,original);EXPECT_GT(metric.absolute_l2,0);EXPECT_TRUE(std::isfinite(metric.relative_l2));auto breakdown=evaluateFp16Accuracy(s,arch);EXPECT_GT(breakdown.total_error.absolute_l2,0);
}

TEST(CSCFp16M7ArtifactTest, PublishesValidatedFilesWithoutOverwrite)
{
    Dir image_dir("artifact_image");exportCSCFp16ImageV2(boundary(),image_dir.p.string());auto image=CSCFp16ExecutionImage::load(image_dir.p.string(),CSCFp16ExecutionMode::FP16_IMAGE_V2);auto result=runFp16M7(image,CSCExecutionMode::BGA_VALIDATION);Dir holder("artifact_holder");fs::create_directories(holder.p);auto output=holder.p/"run";publishFp16M7Artifacts(result,output.string());EXPECT_TRUE(fs::exists(output/"run_manifest.json"));EXPECT_TRUE(fs::exists(output/"phase_cycles.csv"));EXPECT_TRUE(fs::exists(output/"traffic.csv"));EXPECT_EQ(fs::file_size(output/"final_y_fp16.bin"),result.rows*2U);EXPECT_THROW(publishFp16M7Artifacts(result,output.string()),std::runtime_error);
}

TEST(CSCFp16M7ExternalTest, OptInVerifiedV2ImageRunsAllModes)
{
    const char* path=getenv("CSC_FP16_EXTERNAL_IMAGE");if(!path||!*path)GTEST_SKIP()<<"set CSC_FP16_EXTERNAL_IMAGE to a verified v2 image";auto image=CSCFp16ExecutionImage::load(path,CSCFp16ExecutionMode::FP16_IMAGE_V2);auto validation=runFp16M7(image,CSCExecutionMode::BGA_VALIDATION);auto full=runFp16M7(image,CSCExecutionMode::END_TO_END_TIMED);EXPECT_EQ(validation.final_y_bits,full.final_y_bits);
}
