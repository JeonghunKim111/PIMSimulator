#include "csc/CSCFp16M7.h"

#include "csc/CSCFp16.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace csc_descriptor {
namespace {
void hashByte(uint8_t v,uint64_t&h){h=(h^v)*1099511628211ULL;}
void hash32(uint32_t v,uint64_t&h){for(unsigned i=0;i<4;++i)hashByte(uint8_t(v>>(8*i)),h);}
void hash64(uint64_t v,uint64_t&h){for(unsigned i=0;i<8;++i)hashByte(uint8_t(v>>(8*i)),h);}
uint64_t matrixFingerprint(const CSCFp16LoadedImage&i){uint64_t h=1469598103934665603ULL;hash32(i.matrix.rows,h);hash32(i.matrix.cols,h);for(auto v:i.matrix.col_ptr)hash64(v,h);for(auto v:i.matrix.row_idx)hash32(v,h);return h;}
uint64_t mappingFingerprint(const CSCFp16LoadedImage&i){uint64_t h=1469598103934665603ULL;for(auto v:i.column_to_bg)hash32(v,h);return h;}
std::vector<CSCFp16BGAOutputEvent> canonicalEvents(const std::vector<CSCFp16BGAOutputEvent>& events){std::array<std::vector<CSCFp16BGAOutputEvent>,64> by_bg;for(const auto&e:events)by_bg.at(e.global_bg_id).push_back(e);std::vector<CSCFp16BGAOutputEvent> out;for(auto&v:by_bg){std::sort(v.begin(),v.end(),[](const auto&a,const auto&b){return a.sequence<b.sequence;});out.insert(out.end(),v.begin(),v.end());}return out;}
const char* modeName(CSCExecutionMode m){switch(m){case CSCExecutionMode::COMPUTE_ONLY:return "COMPUTE_ONLY";case CSCExecutionMode::BGA_VALIDATION:return "BGA_VALIDATION";case CSCExecutionMode::END_TO_END_TIMED:return "END_TO_END_TIMED";}throw std::invalid_argument("invalid CSC execution mode");}
void drive(CSCFp16NativeExecution& execution,uint64_t max_cycles){execution.launch();for(uint64_t n=0;n<max_cycles&&!execution.done()&&!execution.failed();++n)execution.tick();if(execution.failed())throw std::runtime_error(execution.error());if(!execution.done())throw std::runtime_error("FP16 M7 execution timeout");}
}

CSCFp16ValidationResult reduceCapturedFp16BGAOutputs(uint32_t rows,const std::vector<CSCFp16BGAOutputEvent>& events)
{
    CSCFp16ValidationResult result; result.final_y_bits.assign(rows,0);
    std::array<std::vector<CSCFp16BGAOutputEvent>,64> ordered;
    for(const auto&e:events){if(e.global_bg_id>=64||e.row_idx>=rows||!e.sequence)throw std::invalid_argument("invalid captured FP16 BGA output");ordered[e.global_bg_id].push_back(e);}
    std::vector<bool> touched(rows,false);std::vector<uint32_t> first_bg(rows,0);
    for(uint32_t bg=0;bg<64;++bg){auto&v=ordered[bg];std::sort(v.begin(),v.end(),[](const auto&a,const auto&b){return a.sequence<b.sequence;});uint64_t expected=1;for(const auto&e:v){if(e.sequence!=expected++)throw std::invalid_argument("duplicate or missing FP16 BGA sequence");if(touched[e.row_idx]&&first_bg[e.row_idx]!=bg)result.cross_bg_same_row_adds++;if(!touched[e.row_idx]){touched[e.row_idx]=true;first_bg[e.row_idx]=bg;result.rows_touched++;}result.final_y_bits[e.row_idx]=cscFp16ToBits(cscFp16Add(cscFp16FromBits(result.final_y_bits[e.row_idx]),cscFp16FromBits(e.value_bits)));result.host_add_count++;}}
    result.final_y_hash=cscFp16FinalYFnv1a64(result.final_y_bits);return result;
}

CSCFp16ResultRegionPreflight preflightFp16ResultRegion(const CSCFp16ExecutionImage& execution)
{
    CSCFp16ResultRegionPreflight out;const auto&i=execution.image();uint64_t descriptor_nnz=0;
    for(uint32_t bg=0;bg<64;++bg){auto&o=out.bg[bg];o.descriptor_count=i.bg[bg].parsed_descriptors.size();for(const auto&d:i.bg[bg].parsed_descriptors){if(std::numeric_limits<uint64_t>::max()-o.assigned_nnz<d.nnz_count)throw std::overflow_error("FP16 result preflight NNZ overflow");o.assigned_nnz+=d.nnz_count;}o.upper_bound_output_records=o.assigned_nnz;o.required_result_bursts=o.assigned_nnz/4+(o.assigned_nnz%4!=0);o.allocated_result_bursts=o.required_result_bursts;descriptor_nnz+=o.assigned_nnz;out.total_upper_bound_records+=o.assigned_nnz;if(o.required_result_bursts>std::numeric_limits<uint64_t>::max()/32)throw std::overflow_error("FP16 result preflight byte overflow");uint64_t bytes=o.required_result_bursts*32;if(std::numeric_limits<uint64_t>::max()-out.total_required_result_bytes<bytes)throw std::overflow_error("FP16 result preflight total overflow");out.total_required_result_bytes+=bytes;out.maximum_per_bg_result_bytes=std::max(out.maximum_per_bg_result_bytes,bytes);}
    if(descriptor_nnz!=i.matrix.value_bits.size())
        throw std::invalid_argument("FP16 preflight descriptor NNZ mismatch");
    out.allocation_success=true;return out;
}

CSCFp16M7Result runFp16M7(std::shared_ptr<const CSCFp16ExecutionImage> image,CSCExecutionMode mode,uint64_t max_cycles)
{
    if(!image)throw std::invalid_argument("null FP16 M7 image");
    CSCFp16M7Result result;result.mode=mode;result.configuration_preset=mode==CSCExecutionMode::COMPUTE_ONLY?"FP16_COMPUTE_ONLY":"FP16_ISO_STRUCTURE_BATCH8_Q16";const auto&i=image->image();result.rows=i.matrix.rows;result.columns=i.matrix.cols;result.nnz=i.matrix.value_bits.size();result.matrix_fingerprint=matrixFingerprint(i);result.mapping_fingerprint=mappingFingerprint(i);result.preflight=preflightFp16ResultRegion(*image);for(const auto&b:result.preflight.bg){result.descriptor_count+=b.descriptor_count;if(b.descriptor_count)result.active_bg_count++;}
    if(mode==CSCExecutionMode::COMPUTE_ONLY){std::size_t cap=1;for(const auto&b:result.preflight.bg)cap=std::max<std::size_t>(cap,b.assigned_nnz+1);CSCFp16NativeExecution execution(image,cap);drive(execution,max_cycles);const auto c=execution.counters();result.compute_complete_cycle=c.timing.compute_complete_cycle;std::vector<CSCFp16PartialEvent> trace;for(uint32_t bg=0;bg<64;++bg){const auto&t=execution.sink(bg).trace();trace.insert(trace.end(),t.begin(),t.end());}result.partial_count=trace.size();result.partial_trace_hash=cscFp16PartialTraceFnv1a64(trace);return result;}
    const auto bga=makeFp16IsoStructureProductionConfig(i.matrix.rows);std::size_t output_cap=1;for(const auto&b:result.preflight.bg)output_cap=std::max<std::size_t>(output_cap,b.upper_bound_output_records+1);
    if(mode==CSCExecutionMode::BGA_VALIDATION){CSCFp16NativeExecution execution(image,output_cap,&bga,output_cap);drive(execution,max_cycles);const auto c=execution.counters();result.compute_complete_cycle=c.timing.compute_complete_cycle;result.bga_complete_cycle=c.timing.compute_bga_completion_cycle;std::vector<CSCFp16BGAOutputEvent> events;for(uint32_t bg=0;bg<64;++bg){const auto&t=execution.bgaOutputSink(bg).trace();events.insert(events.end(),t.begin(),t.end());}auto reduced=reduceCapturedFp16BGAOutputs(i.matrix.rows,events);result.bga_output_count=events.size();result.bga_output_hash=cscFp16BGAOutputTraceFnv1a64(events);result.host_add_count=reduced.host_add_count;result.rows_touched=reduced.rows_touched;result.final_y_hash=reduced.final_y_hash;result.final_y_bits=std::move(reduced.final_y_bits);return result;}
    if(mode!=CSCExecutionMode::END_TO_END_TIMED)
        throw std::invalid_argument("unsupported FP16 M7 mode");
    CSCFp16TransportConfig transport;uint64_t max_bursts=1;for(const auto&b:result.preflight.bg)max_bursts=std::max(max_bursts,b.allocated_result_bursts);if(max_bursts>std::numeric_limits<uint32_t>::max())throw std::overflow_error("FP16 result region too large");transport.buffer_capacity_bursts_per_bg=uint32_t(max_bursts);CSCFp16NativeExecution execution(image,output_cap,&bga,output_cap,&transport);drive(execution,max_cycles);const auto c=execution.counters();const auto&t=execution.transport();const auto&tc=t.counters();result.compute_complete_cycle=c.timing.compute_complete_cycle;result.bga_complete_cycle=c.timing.compute_bga_completion_cycle;result.writeback_complete_cycle=tc.writeback_complete_cycle;result.readback_complete_cycle=tc.readback_complete_cycle;result.host_reduction_complete_cycle=tc.reduction_complete_cycle;result.end_to_end_cycle=tc.end_to_end_cycle;result.bga_output_count=tc.outputs_accepted;result.bga_output_hash=cscFp16BGAOutputTraceFnv1a64(canonicalEvents(t.acceptedTrace()));result.transport_record_count=tc.records_packed;result.write_bursts=tc.writes_accepted;result.read_bursts=tc.reads_accepted;result.write_bytes=tc.write_bytes;result.read_bytes=tc.read_bytes;result.host_add_count=tc.fp16_host_adds;result.rows_touched=tc.rows_touched;result.final_y_bits=execution.finalYFp16Bits();result.final_y_hash=cscFp16FinalYFnv1a64(result.final_y_bits);return result;
}

std::string CSCFp16M7Result::toJson() const
{
    std::ostringstream s;s<<"{\"precision\":\"FP16\",\"execution_mode\":\""<<modeName(mode)<<"\",\"configuration_preset\":\""<<configuration_preset<<"\",\"rows\":"<<rows<<",\"columns\":"<<columns<<",\"nnz\":"<<nnz<<",\"matrix_fingerprint\":\""<<std::hex<<matrix_fingerprint<<"\",\"mapping_fingerprint\":\""<<mapping_fingerprint<<std::dec<<"\"";auto field=[&](const char*n,const std::optional<uint64_t>&v){if(v)s<<",\""<<n<<"\":"<<*v;};field("compute_complete_cycle",compute_complete_cycle);field("bga_complete_cycle",bga_complete_cycle);field("writeback_complete_cycle",writeback_complete_cycle);field("readback_complete_cycle",readback_complete_cycle);field("host_reduction_complete_cycle",host_reduction_complete_cycle);field("end_to_end_cycle",end_to_end_cycle);field("final_y_hash",final_y_hash);s<<'}';return s.str();
}

void publishFp16M7Artifacts(const CSCFp16M7Result&r,const std::string&directory)
{
    namespace fs=std::filesystem;fs::path out(directory);if(fs::exists(out))throw std::runtime_error("M7 output directory already exists");fs::path tmp=out;tmp+=".tmp";if(fs::exists(tmp))throw std::runtime_error("M7 temporary output exists");fs::create_directories(tmp);
    {std::ofstream f(tmp/"run_manifest.json");f<<r.toJson()<<'\n';}
    {std::ofstream f(tmp/"phase_cycles.csv");f<<"phase,cycle\n";auto row=[&](const char*n,const std::optional<uint64_t>&v){if(v)f<<n<<','<<*v<<'\n';};row("compute_complete",r.compute_complete_cycle);row("bga_complete",r.bga_complete_cycle);row("writeback_complete",r.writeback_complete_cycle);row("readback_complete",r.readback_complete_cycle);row("host_reduction_complete",r.host_reduction_complete_cycle);row("end_to_end",r.end_to_end_cycle);}
    {std::ofstream f(tmp/"traffic.csv");f<<"metric,value\n";auto row=[&](const char*n,const std::optional<uint64_t>&v){if(v)f<<n<<','<<*v<<'\n';};row("transport_records",r.transport_record_count);row("write_bursts",r.write_bursts);row("read_bursts",r.read_bursts);row("write_bytes",r.write_bytes);row("read_bytes",r.read_bytes);}
    {std::ofstream f(tmp/"bga_stats.csv");f<<"metric,value\n";if(r.bga_output_count)f<<"output_records,"<<*r.bga_output_count<<'\n';if(r.bga_output_hash)f<<"output_trace_hash,"<<std::hex<<*r.bga_output_hash<<std::dec<<'\n';}
    {std::ofstream f(tmp/"accuracy.csv");f<<"status,reason\nunavailable,source FP64 oracle not supplied to artifact publisher\n";}
    {std::ofstream f(tmp/"oracle_summary.json");f<<"{\"status\":\"unavailable\",\"reason\":\"source FP64 oracle not supplied to artifact publisher\"}\n";}
    {std::ofstream f(tmp/"final_y_fp16.bin",std::ios::binary);for(auto bits:r.final_y_bits){char b[2]={char(bits),char(bits>>8)};f.write(b,2);}}
    fs::rename(tmp,out);
}

void validatePairedIdentity(const CSCPairedIdentity&a,const CSCPairedIdentity&b)
{if(a.rows!=b.rows||a.columns!=b.columns||a.nnz!=b.nnz||a.source_matrix_fingerprint!=b.source_matrix_fingerprint||a.row_index_fingerprint!=b.row_index_fingerprint||a.mapping_fingerprint!=b.mapping_fingerprint||a.x_source_fingerprint!=b.x_source_fingerprint)throw std::invalid_argument("FP32/FP16 paired identity mismatch");}

std::vector<double> cscFp64Oracle(const CSCFp16ImageSource&s,bool quantize)
{s.validate();std::vector<double>y(s.rows,0);for(uint32_t col=0;col<s.cols;++col){double x=quantize?cscFp16ToDouble(cscFp16FromDouble(s.x[col])):s.x[col];for(uint64_t p=s.col_ptr[col];p<s.col_ptr[col+1];++p){double a=quantize?cscFp16ToDouble(cscFp16FromDouble(s.values[p])):s.values[p];y[s.row_idx[p]]+=a*x;}}return y;}

CSCAccuracyMetrics compareFp64ToFp64(const std::vector<double>&values,const std::vector<double>&ref)
{if(values.size()!=ref.size())throw std::invalid_argument("accuracy vector size mismatch");CSCAccuracyMetrics m;long double sum=0,refsum=0,abssum=0;uint64_t finite=0;for(size_t i=0;i<values.size();++i){double v=values[i],r=ref[i];if(std::isnan(v)){m.nan_rows++;if(std::isfinite(r))m.finite_to_nonfinite_mismatch++;continue;}if(std::isinf(v)){v>0?m.positive_inf_rows++:m.negative_inf_rows++;if(std::isfinite(r))m.finite_to_nonfinite_mismatch++;continue;}if(!std::isfinite(r)){m.finite_to_nonfinite_mismatch++;continue;}double d=v-r,a=std::abs(d);sum+=static_cast<long double>(d)*d;refsum+=static_cast<long double>(r)*r;abssum+=a;finite++;m.maximum_absolute=std::max(m.maximum_absolute,a);if(r!=0)m.maximum_finite_relative=std::max(m.maximum_finite_relative,a/std::abs(r));if(std::signbit(v)!=std::signbit(r)&&v!=r)m.sign_mismatch++;if((v==0)!=(r==0))m.zero_nonzero_mismatch++;if(v==0&&r==0&&std::signbit(v)!=std::signbit(r))m.signed_zero_mismatch++;}m.absolute_l2=std::sqrt(double(sum));m.relative_l2=m.absolute_l2/std::max(std::sqrt(double(refsum)),std::numeric_limits<double>::epsilon());m.rmse=finite?std::sqrt(double(sum/finite)):0;m.normalized_rmse=m.rmse/std::max(std::sqrt(double(refsum/std::max<uint64_t>(1,finite))),std::numeric_limits<double>::epsilon());m.mean_absolute=finite?double(abssum/finite):0;return m;}

CSCAccuracyMetrics compareFp16ToFp64(const std::vector<CSCFp16Bits>&bits,const std::vector<double>&ref)
{std::vector<double> values;values.reserve(bits.size());for(auto v:bits)values.push_back(cscFp16ToDouble(cscFp16FromBits(v)));return compareFp64ToFp64(values,ref);}

CSCFp16AccuracyBreakdown evaluateFp16Accuracy(const CSCFp16ImageSource&s,const std::vector<CSCFp16Bits>&arch)
{auto original=cscFp64Oracle(s,false),quantized=cscFp64Oracle(s,true);return {compareFp16ToFp64(arch,original),compareFp64ToFp64(quantized,original),compareFp16ToFp64(arch,quantized)};}

CSCPairedCycleReport makePairedCycleReport(const CSCPairedIdentity&a,const CSCPairedIdentity&b,uint64_t fc32,uint64_t fc16,uint64_t e32,uint64_t e16)
{validatePairedIdentity(a,b);if(!fc32||!fc16||!e32||!e16)throw std::invalid_argument("paired cycles unavailable");return {fc32,fc16,e32,e16,double(fc32)/fc16,double(e32)/e16};}

}  // namespace csc_descriptor
