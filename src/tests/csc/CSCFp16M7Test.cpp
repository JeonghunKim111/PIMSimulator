#include "csc/CSCFp16M7.h"
#include "tests/csc/CSCExternalImage.h"

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

TEST(CSCFp16M7ModeTest, ThreeModesPreserveQ64DefaultGoldens)
{
    Dir d("modes");exportCSCFp16ImageV2(boundary(),d.p.string());auto image=CSCFp16ExecutionImage::load(d.p.string(),CSCFp16ExecutionMode::FP16_IMAGE_V2);
    auto compute=runFp16M7(image,CSCExecutionMode::COMPUTE_ONLY);
    auto validation=runFp16M7(image,CSCExecutionMode::BGA_VALIDATION);
    auto full=runFp16M7(image,CSCExecutionMode::END_TO_END_TIMED);
    auto repeated=runFp16M7(image,CSCExecutionMode::END_TO_END_TIMED);
    ASSERT_EQ(compute.partial_count,169U);EXPECT_EQ(compute.partial_trace_hash,0x2e2867563f3d9cacULL);EXPECT_EQ(compute.compute_complete_cycle,544U);
    EXPECT_EQ(validation.bga_complete_cycle,601U);EXPECT_EQ(validation.bga_output_count,108U);EXPECT_EQ(validation.bga_output_hash,0x15baec2c8d0f6fc6ULL);
    ASSERT_EQ(validation.final_y_bits,full.final_y_bits);EXPECT_EQ(validation.final_y_hash,0x64a5f1109a6f2bd6ULL);EXPECT_EQ(full.final_y_hash,validation.final_y_hash);
    EXPECT_EQ(full.bga_complete_cycle,601U);EXPECT_EQ(full.writeback_complete_cycle,606U);EXPECT_EQ(full.readback_complete_cycle,635U);EXPECT_EQ(full.end_to_end_cycle,664U);
    EXPECT_EQ(full.toJson(),repeated.toJson());EXPECT_EQ(full.final_y_bits,repeated.final_y_bits);
    EXPECT_EQ(full.bga_output_hash,0x15baec2c8d0f6fc6ULL);
    EXPECT_EQ(full.write_bursts,28U);EXPECT_EQ(full.read_bursts,28U);EXPECT_FALSE(compute.bga_complete_cycle);EXPECT_FALSE(validation.writeback_complete_cycle);
    EXPECT_NE(compute.toJson().find("COMPUTE_ONLY"),std::string::npos);
}

TEST(CSCFp16M7PreflightTest, SizesLogicalRegionFromAssignedNnz)
{
    Dir d("preflight");exportCSCFp16ImageV2(boundary(),d.p.string());auto image=CSCFp16ExecutionImage::load(d.p.string(),CSCFp16ExecutionMode::FP16_IMAGE_V2);auto p=preflightFp16ResultRegion(*image);EXPECT_TRUE(p.allocation_success);EXPECT_EQ(p.total_upper_bound_records,169U);EXPECT_EQ(p.bg[7].required_result_bursts,(p.bg[7].assigned_nnz+3)/4);EXPECT_EQ(p.bg[8].required_result_bursts,(p.bg[8].assigned_nnz+3)/4);EXPECT_EQ(p.total_required_result_bytes,(p.bg[7].required_result_bursts+p.bg[8].required_result_bursts)*32U);
}

TEST(CSCFp16M7EvaluationTest, PairedIdentityAndOraclesSeparateInputQuantization)
{
    CSCPairedIdentity a{2,2,2,1,2,3,4},b=a;EXPECT_NO_THROW(validatePairedIdentity(a,b));auto cycles=makePairedCycleReport(a,b,200,100,400,250);EXPECT_DOUBLE_EQ(cycles.compute_speedup,2.0);EXPECT_DOUBLE_EQ(cycles.end_to_end_speedup,1.6);b.mapping_fingerprint++;EXPECT_THROW(validatePairedIdentity(a,b),std::invalid_argument);
    CSCFp16ImageSource s;s.rows=2;s.cols=2;s.col_ptr={0,1,2};s.row_idx={0,1};s.values={1.0001,0.3333};s.x={2.0001,3.0001};s.column_to_bg={0,1};auto original=cscFp64Oracle(s,false),quantized=cscFp64Oracle(s,true);EXPECT_NE(original,quantized);std::vector<CSCFp16Bits> arch={cscFp16ToBits(cscFp16FromDouble(quantized[0])),cscFp16ToBits(cscFp16FromDouble(quantized[1]))};auto metric=compareFp16ToFp64(arch,original);EXPECT_GT(metric.absolute_l2,0);EXPECT_TRUE(std::isfinite(metric.relative_l2));auto breakdown=evaluateFp16Accuracy(s,arch);EXPECT_GT(breakdown.total_error.absolute_l2,0);EXPECT_DOUBLE_EQ(breakdown.input_quantization_error.absolute_l2,compareFp64ToFp64(quantized,original).absolute_l2);
}

TEST(CSCFp16M7ArtifactTest, PublishesValidatedFilesWithoutOverwrite)
{
    Dir image_dir("artifact_image");exportCSCFp16ImageV2(boundary(),image_dir.p.string());auto image=CSCFp16ExecutionImage::load(image_dir.p.string(),CSCFp16ExecutionMode::FP16_IMAGE_V2);auto result=runFp16M7(image,CSCExecutionMode::BGA_VALIDATION);Dir holder("artifact_holder");fs::create_directories(holder.p);auto output=holder.p/"run";publishFp16M7Artifacts(result,output.string());EXPECT_TRUE(fs::exists(output/"run_manifest.json"));EXPECT_TRUE(fs::exists(output/"phase_cycles.csv"));EXPECT_TRUE(fs::exists(output/"traffic.csv"));EXPECT_TRUE(fs::exists(output/"bga_stats.csv"));EXPECT_TRUE(fs::exists(output/"accuracy.csv"));EXPECT_TRUE(fs::exists(output/"oracle_summary.json"));EXPECT_EQ(fs::file_size(output/"final_y_fp16.bin"),result.rows*2U);EXPECT_THROW(publishFp16M7Artifacts(result,output.string()),std::runtime_error);
}

TEST(CSCFp16M7ExternalTest, OptInVerifiedV2ImageRunsAllModes)
{
    const char* path=getenv("CSC_FP16_EXTERNAL_IMAGE");if(!path||!*path)GTEST_SKIP()<<"set CSC_FP16_EXTERNAL_IMAGE to a verified v2 image";auto image=CSCFp16ExecutionImage::load(path,CSCFp16ExecutionMode::FP16_IMAGE_V2);auto validation=runFp16M7(image,CSCExecutionMode::BGA_VALIDATION);auto full=runFp16M7(image,CSCExecutionMode::END_TO_END_TIMED);EXPECT_EQ(validation.final_y_bits,full.final_y_bits);
}

TEST(CSCFp16M7ExternalTest, OptInBGAValidationReportsCycles)
{
    const char* path=getenv("CSC_FP16_EXTERNAL_IMAGE");
    if(!path||!*path)GTEST_SKIP()<<"set CSC_FP16_EXTERNAL_IMAGE to a verified v2 image";
    auto image=CSCFp16ExecutionImage::load(path,CSCFp16ExecutionMode::FP16_IMAGE_V2);
    auto result=runFp16M7(image,CSCExecutionMode::BGA_VALIDATION,2000000000ULL);
    const char* accepts=getenv("CSC_BGA_VALIDATION_ACCEPTS_PER_CYCLE");
    std::cout<<"FP16_BGA_VALIDATION"
             <<" image="<<path
             <<" bga_config="<<result.configuration_preset
             <<" validation_accepts_per_cycle="<<(accepts?accepts:"64")
             <<" compute_complete_cycle="<<*result.compute_complete_cycle
             <<" bga_complete_cycle="<<*result.bga_complete_cycle
             <<" bga_output_records="<<*result.bga_output_count
             <<" bga_output_hash=0x"<<std::hex<<*result.bga_output_hash
             <<" final_y_hash=0x"<<*result.final_y_hash<<std::dec<<'\n';
}

TEST(CSCFp16M7ExternalTest, OptInEndToEndTimedReportsCycles)
{
    const char* path=getenv("CSC_FP16_EXTERNAL_IMAGE");
    if(!path||!*path)GTEST_SKIP()<<"set CSC_FP16_EXTERNAL_IMAGE to a verified v2 image";
    auto image=CSCFp16ExecutionImage::load(path,CSCFp16ExecutionMode::FP16_IMAGE_V2);
    auto result=runFp16M7(image,CSCExecutionMode::END_TO_END_TIMED,2000000000ULL);
    std::cout<<"FP16_END_TO_END_TIMED"
             <<" image="<<path
             <<" bga_config="<<result.configuration_preset
             <<" compute_complete_cycle="<<*result.compute_complete_cycle
             <<" bga_complete_cycle="<<*result.bga_complete_cycle
             <<" writeback_complete_cycle="<<*result.writeback_complete_cycle
             <<" readback_complete_cycle="<<*result.readback_complete_cycle
             <<" host_reduction_complete_cycle="<<*result.host_reduction_complete_cycle
             <<" end_to_end_cycle="<<*result.end_to_end_cycle
             <<" bga_output_records="<<*result.bga_output_count
             <<" write_bursts="<<*result.write_bursts
             <<" read_bursts="<<*result.read_bursts
             <<" write_bytes="<<*result.write_bytes
             <<" read_bytes="<<*result.read_bytes
             <<" bga_output_hash=0x"<<std::hex<<*result.bga_output_hash
             <<" final_y_hash=0x"<<*result.final_y_hash<<std::dec<<'\n';
}

TEST(CSCFp16M7ExternalTest, OptInTransportOnlyReportsCycles)
{
    const char* path=getenv("CSC_FP16_EXTERNAL_IMAGE");
    if(!path||!*path)GTEST_SKIP()<<"set CSC_FP16_EXTERNAL_IMAGE to a verified v2 image";
    auto image=CSCFp16ExecutionImage::load(path,CSCFp16ExecutionMode::FP16_IMAGE_V2);
    auto result=runFp16M7(image,CSCExecutionMode::TRANSPORT_ONLY,2000000000ULL);
    ASSERT_TRUE(result.compute_complete_cycle);ASSERT_TRUE(result.bga_complete_cycle);
    ASSERT_TRUE(result.writeback_complete_cycle);ASSERT_TRUE(result.readback_complete_cycle);
    ASSERT_TRUE(result.bga_output_count);ASSERT_TRUE(result.bga_contribution_count);
    ASSERT_TRUE(result.transport_record_count);ASSERT_TRUE(result.write_bursts);
    ASSERT_TRUE(result.read_bursts);ASSERT_TRUE(result.write_bytes);ASSERT_TRUE(result.read_bytes);
    const bool contribution_conservation=*result.bga_contribution_count==result.nnz;
    const bool record_conservation=*result.bga_output_count==*result.transport_record_count;
    const bool byte_conservation=*result.write_bytes==*result.read_bytes;
    std::cout<<"\n=== FP16 M7 BGA + WRITEBACK + READBACK ONLY ===\n"
             <<"rows: "<<result.rows<<'\n'<<"cols: "<<result.columns<<'\n'<<"nnz: "<<result.nnz<<'\n'
             <<"bga_config: "<<result.configuration_preset<<'\n'
             <<"compute_complete_cycle: "<<*result.compute_complete_cycle<<'\n'
             <<"bga_complete_cycle: "<<*result.bga_complete_cycle<<'\n'
             <<"writeback_complete_cycle: "<<*result.writeback_complete_cycle<<'\n'
             <<"readback_complete_cycle: "<<*result.readback_complete_cycle<<'\n'
             <<"T_scope_matched_total: "<<*result.readback_complete_cycle<<'\n'
             <<"physical_bga_output_records: "<<*result.bga_output_count<<'\n'
             <<"write_requests_issued/completed: "<<*result.write_bursts<<'/'<<*result.write_bursts<<'\n'
             <<"read_requests_issued/completed: "<<*result.read_bursts<<'/'<<*result.read_bursts<<'\n'
             <<"writeback_transferred/padding_bytes: "<<*result.write_bytes<<'/'<<*result.padding_bytes<<'\n'
             <<"readback_transferred_bytes: "<<*result.read_bytes<<'\n'
             <<"transport_backpressure_cycles: "<<*result.transport_stall_cycles<<'\n'
             <<"host_reduction_enabled: false\n"
             <<"contribution_conservation: "<<(contribution_conservation?"PASS":"FAIL")<<'\n'
             <<"record_conservation: "<<(record_conservation?"PASS":"FAIL")<<'\n'
             <<"byte_conservation: "<<(byte_conservation?"PASS":"FAIL")<<'\n'
             <<"=== FP16 M7 TRANSPORT-ONLY RUN COMPLETE ===\n";
    EXPECT_TRUE(contribution_conservation);EXPECT_TRUE(record_conservation);EXPECT_TRUE(byte_conservation);
    EXPECT_FALSE(result.host_reduction_complete_cycle);EXPECT_FALSE(result.end_to_end_cycle);
    EXPECT_FALSE(result.final_y_hash);EXPECT_TRUE(result.final_y_bits.empty());
}

TEST(CSCFp16M7ExternalTest, OptInMaterializeVerifiedV1AsFp16V2)
{
    const char* input=getenv("CSC_FP32_EXTERNAL_IMAGE");
    const char* output=getenv("CSC_FP16_OUTPUT_IMAGE");
    if(!input||!*input||!output||!*output)
        GTEST_SKIP()<<"set CSC_FP32_EXTERNAL_IMAGE and CSC_FP16_OUTPUT_IMAGE";
    auto fp32=loadExternalPhysicalImage(input);
    CSCFp16ImageSource source;
    source.rows=fp32.layout.matrix.rows;
    source.cols=fp32.layout.matrix.cols;
    source.col_ptr=fp32.layout.matrix.col_ptr;
    source.row_idx=fp32.layout.matrix.row_idx;
    source.values.assign(fp32.layout.matrix.values.begin(),fp32.layout.matrix.values.end());
    source.column_to_bg=fp32.layout.column_to_bg;
    source.x.resize(source.cols);
    for(uint32_t col=0;col<source.cols;++col)
        source.x[col]=double((col%13)+1)/7.0;
    exportCSCFp16ImageV2(source,output);
    auto loaded=loadCSCFp16ImageV2(output);
    EXPECT_EQ(loaded.matrix.rows,source.rows);
    EXPECT_EQ(loaded.matrix.cols,source.cols);
    EXPECT_EQ(loaded.matrix.row_idx,source.row_idx);
    EXPECT_EQ(loaded.column_to_bg,source.column_to_bg);
    EXPECT_EQ(loaded.matrix.value_bits.size(),source.values.size());
}

TEST(CSCFp16M7ModeTest, ValidationAcceptWidthIsConfigurable)
{
    Dir d("accept_width");exportCSCFp16ImageV2(boundary(),d.p.string());
    auto image=CSCFp16ExecutionImage::load(d.p.string(),CSCFp16ExecutionMode::FP16_IMAGE_V2);
    setenv("CSC_BGA_VALIDATION_ACCEPTS_PER_CYCLE","1",1);
    auto one=runFp16M7(image,CSCExecutionMode::BGA_VALIDATION);
    setenv("CSC_BGA_VALIDATION_ACCEPTS_PER_CYCLE","64",1);
    auto wide=runFp16M7(image,CSCExecutionMode::BGA_VALIDATION);
    unsetenv("CSC_BGA_VALIDATION_ACCEPTS_PER_CYCLE");
    EXPECT_EQ(one.final_y_bits,wide.final_y_bits);
    EXPECT_EQ(one.bga_output_hash,wide.bga_output_hash);
    EXPECT_GE(*one.bga_complete_cycle,*wide.bga_complete_cycle);
}

TEST(CSCFp16M7ModeTest, BGAQ64IsDefaultAndKeepsBatchWidthEight)
{
    Dir d("q64");exportCSCFp16ImageV2(boundary(),d.p.string());
    auto image=CSCFp16ExecutionImage::load(d.p.string(),CSCFp16ExecutionMode::FP16_IMAGE_V2);
    auto q64=runFp16M7(image,CSCExecutionMode::BGA_VALIDATION);
    EXPECT_EQ(q64.configuration_preset,"FP16_BATCH8_Q64");
    EXPECT_EQ(q64.bga_output_count,108U);
}

TEST(CSCFp16M7ModeTest, BGAQ16RemainsAvailableAsSensitivityOverride)
{
    Dir d("q16");exportCSCFp16ImageV2(boundary(),d.p.string());
    auto image=CSCFp16ExecutionImage::load(d.p.string(),CSCFp16ExecutionMode::FP16_IMAGE_V2);
    setenv("CSC_FP16_BGA_CAPACITY","16",1);
    auto q16=runFp16M7(image,CSCExecutionMode::BGA_VALIDATION);
    unsetenv("CSC_FP16_BGA_CAPACITY");
    EXPECT_EQ(q16.configuration_preset,"FP16_BATCH8_Q16");
}

TEST(CSCFp16M7ModeTest, TransportOnlyStopsAfterReadbackWithoutReduction)
{
    Dir d("transport_only");exportCSCFp16ImageV2(boundary(),d.p.string());
    auto image=CSCFp16ExecutionImage::load(d.p.string(),CSCFp16ExecutionMode::FP16_IMAGE_V2);
    auto transport=runFp16M7(image,CSCExecutionMode::TRANSPORT_ONLY);
    EXPECT_EQ(transport.configuration_preset,"FP16_BATCH8_Q64");
    EXPECT_EQ(transport.bga_contribution_count,169U);
    EXPECT_EQ(transport.bga_output_count,transport.transport_record_count);
    EXPECT_EQ(transport.write_bursts,transport.read_bursts);
    EXPECT_EQ(transport.write_bytes,transport.read_bytes);
    EXPECT_TRUE(transport.readback_complete_cycle);
    EXPECT_FALSE(transport.host_reduction_complete_cycle);
    EXPECT_FALSE(transport.end_to_end_cycle);
    EXPECT_EQ(transport.host_add_count,0U);
    EXPECT_TRUE(transport.final_y_bits.empty());
    EXPECT_NE(transport.toJson().find("\"padding_bytes\""),std::string::npos);
    EXPECT_NE(transport.toJson().find("\"transport_stall_cycles\""),std::string::npos);

    Dir holder("transport_artifact_holder");fs::create_directories(holder.p);
    auto output=holder.p/"run";publishFp16M7Artifacts(transport,output.string());
    EXPECT_TRUE(fs::exists(output/"run_manifest.json"));
    EXPECT_TRUE(fs::exists(output/"traffic.csv"));
    EXPECT_TRUE(fs::exists(output/"bga_stats.csv"));
    EXPECT_FALSE(fs::exists(output/"final_y_fp16.bin"));
}
