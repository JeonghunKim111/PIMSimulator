#ifndef CSC_DESCRIPTOR_ENGINE_H
#define CSC_DESCRIPTOR_ENGINE_H
#include "PIMBlock.h"
#include "csc/CSCRequestTracker.h"
#include "csc/CSCTypes.h"
#include <array>
#include <memory>
#include <functional>
#include <string>
#include <vector>
namespace csc_descriptor {
enum class CSCDescriptorState{IDLE,FETCH_DESCRIPTOR,LOAD_X,FETCH_VALUE,FETCH_ROW_INDEX,WAIT_OPERANDS,SIMD_MUL,EMIT_PARTIALS,ADVANCE_CHUNK,NEXT_DESCRIPTOR,DONE,ERROR,FLUSHING};
struct CSCPartial{uint32_t row_idx,global_bg_id,descriptor_index,chunk_index,lane;float value;};
struct CSCBGImageView{const std::vector<uint8_t>*values=nullptr,*row_indices=nullptr;const std::vector<CSCDescriptor>*descriptors=nullptr;const std::vector<float>*packed_x=nullptr;};
struct CSCEngineCounters{uint64_t descriptor_fetches=0,x_scalar_loads=0,x_transactions=0,value_transactions=0,index_transactions=0,logical_mul_events=0,full_chunks=0,tail_chunks=0,total_chunks=0,active_lanes=0,available_lanes=0,invalid_lane_multiplies=0,invalid_lane_writes=0,generated_partials=0,emitted_partials=0,host_accumulations=0,stall_cycles_request=0,stall_cycles_backpressure=0,busy_cycles=0,total_execution_cycles=0;};
class CSCDescriptorEngine{
public:
 using Submit=std::function<bool(CSCDescriptorEngine*,CSCRequestKind,uint64_t)>;
 using Address=std::function<uint64_t(uint32_t,CSCRequestKind,uint64_t)>;
 CSCDescriptorEngine(uint32_t global_bg_id,DRAMSim::PIMBlock*datapath,Submit,Address);
 void launch(const CSCBGImageView&,std::vector<CSCPartial>*);void tick();void flush();void reset();
 bool onRequestComplete(unsigned,uint64_t,uint64_t);bool busy()const{return busy_;}
 bool isTerminal()const{return state_==CSCDescriptorState::DONE||state_==CSCDescriptorState::ERROR;}
 bool isDone()const{return state_==CSCDescriptorState::DONE;}bool done()const{return isTerminal();}
 bool hasFailed()const{return state_==CSCDescriptorState::ERROR;}CSCError errorCode()const{return error_code_;}
 const std::string&errorMessage()const{return error_;}bool flushRequested()const{return flush_requested_;}
 bool flushComplete()const{return flush_completed_;}
 CSCDescriptorState state()const{return state_;}const std::string&error()const{return error_;}
 const CSCEngineCounters&counters()const{return counters_;}const CSCRequestTracker&tracker()const{return tracker_;}
 uint32_t globalBG()const{return global_bg_id_;}
private:
 void fail(CSCError,const std::string&);bool issue(CSCRequestKind,uint64_t);void validateLaunch()const;bool descriptorValid(const CSCDescriptor&)const;void finishWork();
 uint32_t global_bg_id_,descriptor_pointer_=0,remaining_nnz_=0,chunk_offset_=0,valid_count_=0,chunk_index_=0;
 CSCDescriptor current_{};float x_j_=0;std::array<uint8_t,32>value_staging_{},index_staging_{};DRAMSim::BurstType result_staging_;
 DRAMSim::PIMBlock*datapath_;Submit submit_;Address address_;CSCBGImageView image_;std::vector<CSCPartial>*sink_=nullptr;
 CSCRequestTracker tracker_;CSCDescriptorState state_=CSCDescriptorState::IDLE;bool busy_=false,done_=false,flush_requested_=false,flush_completed_=false;CSCError error_code_=CSCError::NONE;std::string error_;CSCEngineCounters counters_;
};
struct CSCExecutionCounters:CSCEngineCounters{};
class CSCNativeExecution{
public:
 CSCNativeExecution();~CSCNativeExecution();
 void launch(const std::array<CSCBGImageView,64>&,uint32_t rows,uint64_t nnz);
 void tick();bool busy()const;bool done()const{return isTerminal();}void flush();void reset();
 bool isTerminal()const;bool isDone()const;bool hasFailed()const{return failed_;}
 CSCError errorCode()const{return error_code_;}const std::string&errorMessage()const{return error_message_;}
 int32_t failedEngine()const{return failed_engine_;}bool flushRequested()const{return flush_requested_;}
 bool flushComplete()const{return flush_completed_;}bool hasOutstandingRequest()const{return outstanding_!=nullptr;}
 bool hasPendingTransactions()const;bool hasUnconsumedCompletion()const;
 std::vector<float>hostAccumulate();const std::vector<CSCPartial>&partials()const{return partials_;}
 CSCExecutionCounters counters()const;const CSCDescriptorEngine&engine(uint32_t bg)const{return *engines_.at(bg);}
 uint64_t cycle()const{return cycle_;}uint64_t addressFor(uint32_t,CSCRequestKind,uint64_t);
private:
 bool submit(CSCDescriptorEngine*,CSCRequestKind,uint64_t);void readComplete(unsigned,uint64_t,uint64_t);void latchFailure(const CSCDescriptorEngine&);
 struct Impl;std::unique_ptr<Impl>impl_;std::vector<std::unique_ptr<DRAMSim::PIMBlock>>blocks_;
 std::vector<std::unique_ptr<CSCDescriptorEngine>>engines_;std::vector<CSCPartial>partials_;
 CSCDescriptorEngine*outstanding_=nullptr;uint64_t outstanding_address_=0,cycle_=0,nnz_=0;uint32_t rows_=0;bool launched_=false;uint64_t host_accumulations_=0;
 bool failed_=false,flush_requested_=false,flush_completed_=false;CSCError error_code_=CSCError::NONE;
 int32_t failed_engine_=-1;std::string error_message_;
};
}
#endif
