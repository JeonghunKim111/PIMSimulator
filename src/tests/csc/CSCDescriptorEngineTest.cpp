#include "gtest/gtest.h"
#include "AddressMapping.h"
#include "Callback.h"
#include "MultiChannelMemorySystem.h"
#include "csc/CSCDescriptorEngine.h"
#include "tests/csc/CSCExternalImage.h"
#include "tests/csc/CSCFunctionalModel.h"
#include "tests/csc/CSCLayout.h"
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
using namespace csc_descriptor;
using DRAMSim::FP32;
namespace{
struct Views{std::array<std::vector<float>,64>x;std::array<CSCBGImageView,64>v;Views(const CSCLayout&l,const std::vector<float>&original){for(uint32_t g=0;g<64;g++){for(auto c:l.bg[g].x_slot_to_original_col)x[g].push_back(original[c]);v[g]={&l.bg[g].values,&l.bg[g].row_indices,&l.bg[g].descriptors,&x[g]};}}};
void invariants(const CSCLayout&l,const CSCExecutionCounters&c){EXPECT_EQ(c.descriptor_fetches,l.stats.descriptor_count);EXPECT_EQ(c.x_scalar_loads,l.stats.descriptor_count);EXPECT_EQ(c.x_transactions,l.stats.descriptor_count);EXPECT_EQ(c.value_transactions,c.total_chunks);EXPECT_EQ(c.index_transactions,c.total_chunks);EXPECT_EQ(c.logical_mul_events,c.total_chunks);EXPECT_EQ(c.full_chunks+c.tail_chunks,c.total_chunks);EXPECT_EQ(c.active_lanes,l.stats.nnz);EXPECT_EQ(c.generated_partials,l.stats.nnz);EXPECT_EQ(c.emitted_partials,l.stats.nnz);EXPECT_EQ(c.host_accumulations,l.stats.nnz);EXPECT_EQ(c.invalid_lane_multiplies,0);EXPECT_EQ(c.invalid_lane_writes,0);EXPECT_EQ(c.created_requests,c.accepted_requests);EXPECT_EQ(c.abandoned_waiting_requests,0);EXPECT_EQ(c.accepted_requests,c.completion_count);EXPECT_EQ(c.launch_to_done_cycles,c.total_execution_cycles);if(l.stats.nnz){EXPECT_GT(c.total_execution_cycles,0);}}
std::vector<float>run(CSCNativeExecution&native,const CSCLayout&l,const std::vector<float>&x){Views views(l,x);native.launch(views.v,l.matrix.rows,l.stats.nnz);uint64_t guard=0,limit=std::max<uint64_t>(2000000,l.stats.nnz*32);while(!native.done()&&guard++<limit)native.tick();EXPECT_TRUE(native.done())<<"cycle guard="<<limit;if(!native.done())return {};auto y=native.hostAccumulate();invariants(l,native.counters());return y;}
void compare(CSCNativeExecution&native,const CSCMatrix&m,const std::vector<float>&x,MappingPolicy p=MappingPolicy::RoundRobin,const std::vector<uint32_t>&map={}){auto l=buildLayout(m,p,map);auto y=run(native,l,x);auto ref=cpuReference(l.matrix,x);auto old=executeLockstep(l,x);float e;uint32_t r;EXPECT_TRUE(compareResults(ref,y,&e,&r))<<e<<" row "<<r;EXPECT_TRUE(compareResults(old.y,y,&e,&r));}
void expectMalformed(const std::vector<uint8_t>&values,const std::vector<uint8_t>&rows,const std::vector<float>&x,const std::vector<CSCDescriptor>&d){DRAMSim::PIMBlock block(FP32);std::vector<CSCPartial>sink;CSCDescriptorEngine engine(0,&block,[](auto*,auto,auto){return true;},[](auto,auto,auto){return 0;});engine.launch({&values,&rows,&d,&x},&sink);engine.tick();EXPECT_TRUE(engine.isTerminal());EXPECT_TRUE(engine.hasFailed());EXPECT_FALSE(engine.isDone());EXPECT_EQ(engine.errorCode(),CSCError::MALFORMED_DESCRIPTOR);EXPECT_FALSE(engine.errorMessage().empty());}
}
TEST(CSCNativeDescriptorEngineTest,FunctionalBoundariesDescriptorsBGsEmptyAndDuplicates){CSCNativeExecution n;compare(n,makeCSC(0,0,{}),{});for(uint32_t z:{1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,15u,16u,17u}){std::vector<COOEntry>e;for(uint32_t i=0;i<z;i++)e.push_back({i,0,float(i+1)});compare(n,makeCSC(z,1,e),{-2});}auto m=makeCSC(8,6,{{1,0,2},{1,0,3},{2,2,4},{2,3,5},{7,5,6}});compare(n,m,{0,9,-1,2,3,4});std::vector<uint32_t>map(6);map[0]=0;map[2]=0;map[3]=17;map[5]=63;compare(n,m,{0,9,-1,2,3,4},MappingPolicy::External,map);}
TEST(CSCNativeDescriptorEngineTest,FlushResetRelaunchAndTopology){CSCNativeExecution n;auto m=makeCSC(3,2,{{0,0,2},{1,1,3}});auto l=buildLayout(m,MappingPolicy::RoundRobin);Views v(l,{2,4});n.launch(v.v,3,2);n.flush();while(!n.done())n.tick();EXPECT_EQ(n.hostAccumulate(),cpuReference(l.matrix,{2,4}));n.reset();auto y=run(n,l,{2,4});EXPECT_EQ(y,cpuReference(l.matrix,{2,4}));DRAMSim::AddrMapping decode;for(uint32_t g:{0u,1u,17u,63u})for(auto k:{CSCRequestKind::X_READ,CSCRequestKind::VALUE_READ,CSCRequestKind::INDEX_READ}){unsigned ch,r,b,row,col;decode.addressMapping(n.addressFor(g,k,0),ch,r,b,row,col);EXPECT_EQ(ch,g/4);EXPECT_EQ(decode.bankgroupId(b),g%4);}}
TEST(CSCNativeDescriptorEngineTest,BackpressureAndMalformedDescriptors){DRAMSim::PIMBlock block(FP32);std::vector<uint8_t>values(64),rows(64);float one=1;uint32_t row=0;std::memcpy(values.data(),&one,4);std::memcpy(rows.data(),&row,4);std::vector<float>x{2};std::vector<CSCDescriptor>d{{0,0,1,0,0,0}};std::vector<CSCPartial>sink;int reject=3;uint64_t addr=0;CSCDescriptorEngine e(0,&block,[&](auto*,auto,uint64_t a){addr=a;if(reject){reject--;return false;}return true;},[](auto,auto,uint64_t o){return o+32;});e.launch({&values,&rows,&d,&x},&sink);for(int i=0;i<200&&!e.done();i++){e.tick();if(e.tracker().pending())e.onRequestComplete(0,addr,i);}EXPECT_TRUE(e.done());EXPECT_GE(e.counters().stall_cycles_backpressure,3);e.reset();auto bad=d;bad[0].value_offset_bytes=4;expectMalformed(values,rows,x,bad);bad=d;bad[0].global_bg_id=1;expectMalformed(values,rows,x,bad);bad=d;bad[0].x_slot=1;expectMalformed(values,rows,x,bad);bad=d;bad[0].nnz_count=17;expectMalformed(values,rows,x,bad);}
TEST(CSCNativeDescriptorEngineIntegrationTest,ExternalImageMatchesAuthoritativeFP32){const char*p=std::getenv("CSC_EXTERNAL_IMAGE");if(!p||!*p)GTEST_SKIP()<<"set CSC_EXTERNAL_IMAGE";auto l=loadExternalPhysicalImage(p).layout;std::vector<float>x(l.matrix.cols);for(uint32_t i=0;i<x.size();i++)x[i]=float((i%13)+1)/7;CSCNativeExecution n;auto y=run(n,l,x),ref=cpuReference(l.matrix,x);float e;uint32_t r;EXPECT_TRUE(compareResults(ref,y,&e,&r))<<e<<" row "<<r;}
TEST(CSCNativeDescriptorEngineTest,ValidXNaNAndInfUseIEEEPropagation){auto l=buildLayout(makeCSC(1,1,{{0,0,2}}),MappingPolicy::RoundRobin);CSCNativeExecution nan_engine;auto nan_y=run(nan_engine,l,{NAN});ASSERT_EQ(nan_y.size(),1);EXPECT_TRUE(std::isnan(nan_y[0]));CSCNativeExecution inf_engine;auto inf_y=run(inf_engine,l,{INFINITY});ASSERT_EQ(inf_y.size(),1);EXPECT_TRUE(std::isinf(inf_y[0]));}
TEST(CSCNativeDescriptorEngineTest,ErrorIsTerminalAndPropagatesGlobally){
 CSCNativeExecution n;auto l=buildLayout(makeCSC(2,1,{{0,0,2},{1,0,3}}),MappingPolicy::RoundRobin);Views v(l,{4});n.launch(v.v,2,2);l.bg[0].descriptors[0].value_offset_bytes=4;uint64_t guard=0;while(!n.done()&&guard++<10000)n.tick();EXPECT_LT(guard,10000);EXPECT_TRUE(n.isTerminal());EXPECT_TRUE(n.hasFailed());EXPECT_FALSE(n.isDone());EXPECT_EQ(n.failedEngine(),0);EXPECT_EQ(n.errorCode(),CSCError::MALFORMED_DESCRIPTOR);EXPECT_FALSE(n.errorMessage().empty());EXPECT_FALSE(n.hasOutstandingRequest());EXPECT_FALSE(n.hasPendingTransactions());EXPECT_FALSE(n.hasUnconsumedCompletion());EXPECT_THROW(n.hostAccumulate(),std::logic_error);n.flush();EXPECT_TRUE(n.hasFailed());EXPECT_FALSE(n.isDone());n.reset();EXPECT_FALSE(n.isTerminal());l.bg[0].descriptors[0].value_offset_bytes=0;EXPECT_EQ(run(n,l,{4}),cpuReference(l.matrix,{4}));
}
TEST(CSCNativeDescriptorEngineTest,FailureWhileRequestPendingDrainsWithoutNewIssue){
 CSCNativeExecution n;auto l=buildLayout(makeCSC(1,1,{{0,0,2}}),MappingPolicy::RoundRobin);Views v(l,{4});n.launch(v.v,1,1);uint64_t guard=0;while(!n.engine(0).tracker().pending()&&guard++<1000)n.tick();ASSERT_TRUE(n.engine(0).tracker().pending());auto issued=n.engine(0).tracker().stats().issued;v.x[0].clear();while(!n.done()&&guard++<10000)n.tick();ASSERT_TRUE(n.isTerminal());EXPECT_TRUE(n.hasFailed());EXPECT_FALSE(n.hasOutstandingRequest());EXPECT_FALSE(n.hasPendingTransactions());EXPECT_FALSE(n.hasUnconsumedCompletion());EXPECT_GE(n.engine(0).tracker().stats().issued,issued);auto issued_after_failure=n.engine(0).tracker().stats().issued;for(int i=0;i<20;i++)n.tick();EXPECT_EQ(n.engine(0).tracker().stats().issued,issued_after_failure);auto failed_counters=n.counters();EXPECT_EQ(failed_counters.created_requests,failed_counters.accepted_requests+failed_counters.abandoned_waiting_requests);EXPECT_EQ(failed_counters.accepted_requests,failed_counters.completion_count);EXPECT_EQ(failed_counters.launch_to_done_cycles,failed_counters.total_execution_cycles);n.reset();Views valid(l,{4});n.launch(valid.v,1,1);while(!n.done()&&guard++<20000)n.tick();EXPECT_TRUE(n.isDone());
}
TEST(CSCNativeDescriptorEngineTest,GracefulFlushCoversStatesAndRelaunch){
 const std::array<CSCDescriptorState,6> states={CSCDescriptorState::LOAD_X,CSCDescriptorState::FETCH_VALUE,CSCDescriptorState::FETCH_ROW_INDEX,CSCDescriptorState::SIMD_MUL,CSCDescriptorState::EMIT_PARTIALS,CSCDescriptorState::NEXT_DESCRIPTOR};
 for(auto target:states){CSCNativeExecution n;auto l=buildLayout(makeCSC(12,2,{{0,0,2},{1,0,3},{2,0,4},{3,0,5},{4,0,6},{5,0,7},{6,0,8},{7,0,9},{8,0,10},{9,1,11},{10,1,12},{11,1,13}}),MappingPolicy::RoundRobin);Views v(l,{2,3});n.launch(v.v,12,12);uint64_t guard=0;while(n.engine(0).state()!=target&&guard++<100000)n.tick();ASSERT_LT(guard,100000);n.flush();n.flush();EXPECT_TRUE(n.flushRequested());while(!n.done()&&guard++<300000)n.tick();ASSERT_TRUE(n.isDone());EXPECT_TRUE(n.flushComplete());EXPECT_FALSE(n.hasFailed());EXPECT_FALSE(n.hasOutstandingRequest());EXPECT_FALSE(n.hasPendingTransactions());EXPECT_FALSE(n.hasUnconsumedCompletion());EXPECT_EQ(n.counters().generated_partials,n.counters().emitted_partials);EXPECT_EQ(n.hostAccumulate(),cpuReference(l.matrix,{2,3}));n.reset();EXPECT_EQ(run(n,l,{2,3}),cpuReference(l.matrix,{2,3}));}
 CSCNativeExecution idle;idle.flush();EXPECT_TRUE(idle.flushRequested());idle.tick();EXPECT_TRUE(idle.isDone())<<" bg0="<<int(idle.engine(0).state())<<" bg63="<<int(idle.engine(63).state())<<" terminal="<<idle.isTerminal();EXPECT_TRUE(idle.flushComplete());idle.flush();EXPECT_TRUE(idle.isDone());
}

namespace {
struct M5TokenSink {std::vector<DRAMSim::RequestToken> tokens;uint64_t legacy=0;void token(unsigned,const DRAMSim::RequestToken&t,uint64_t){tokens.push_back(t);}void old(unsigned,uint64_t,uint64_t){legacy++;}};
DRAMSim::RequestToken tok(uint64_t id,DRAMSim::RequestKind k,uint32_t bg,uint32_t worker,uint64_t seq,uint32_t gen=7){DRAMSim::RequestToken t;t.request_id=id;t.request_kind=k;t.global_bg_id=bg;t.worker_id=worker;t.sequence_number=seq;t.client_id=99;t.generation=gen;return t;}
}
TEST(CSCRequestIdentityTest,TokenRoundTripSameAddressReadWriteAndLegacy){
 auto mem=std::make_shared<DRAMSim::MultiChannelMemorySystem>("ini/HBM2_samsung_2M_16B_x64.ini","system_hbm_csc_fp32.ini",".","m5_token_test",256*16);M5TokenSink sink;
 auto*tr=new DRAMSim::Callback<M5TokenSink,void,unsigned,const DRAMSim::RequestToken&,uint64_t>(&sink,&M5TokenSink::token);auto*tw=new DRAMSim::Callback<M5TokenSink,void,unsigned,const DRAMSim::RequestToken&,uint64_t>(&sink,&M5TokenSink::token);auto*lr=new DRAMSim::Callback<M5TokenSink,void,unsigned,uint64_t,uint64_t>(&sink,&M5TokenSink::old);auto*lw=new DRAMSim::Callback<M5TokenSink,void,unsigned,uint64_t,uint64_t>(&sink,&M5TokenSink::old);mem->RegisterCallbacks(lr,lw,nullptr);mem->RegisterTokenCallbacks(tr,tw);DRAMSim::BurstType bursts[6];uint64_t a=0;
 std::vector<DRAMSim::RequestToken> issued={tok(101,DRAMSim::RequestKind::VALUE_READ,0,0,(1ULL<<50)+1),tok(102,DRAMSim::RequestKind::INDEX_READ,3,9,(1ULL<<50)+2),tok(103,DRAMSim::RequestKind::X_READ,63,63,(1ULL<<50)+3),tok(105,DRAMSim::RequestKind::DESCRIPTOR_READ,5,7,(1ULL<<50)+5)};
 for(size_t i=0;i<issued.size();i++){ASSERT_TRUE(mem->addTransaction(false,a,"same-address",&bursts[i],issued[i]));}ASSERT_TRUE(mem->addTransaction(true,32,"write-token",&bursts[4],tok(104,DRAMSim::RequestKind::RESULT_TRANSFER,17,2,(1ULL<<50)+4)));ASSERT_TRUE(mem->addTransaction(false,64,"legacy",&bursts[5]));
 uint64_t guard=0;while((sink.tokens.size()<5||sink.legacy<6)&&guard++<200000)mem->update();ASSERT_LT(guard,200000);ASSERT_EQ(sink.tokens.size(),5);std::map<uint64_t,DRAMSim::RequestToken> got;for(auto&t:sink.tokens)got[t.request_id]=t;for(auto&t:issued){ASSERT_TRUE(got.count(t.request_id));EXPECT_TRUE(got[t.request_id]==t);}EXPECT_TRUE(got[104]==tok(104,DRAMSim::RequestKind::RESULT_TRANSFER,17,2,(1ULL<<50)+4));EXPECT_EQ(sink.legacy,6);delete tr;delete tw;delete lr;delete lw;
}
TEST(CSCRequestScoreboardTest,IdentityRetryAndInvariantFailures){CSCRequestTracker q(7);auto a=tok(1,DRAMSim::RequestKind::X_READ,4,4,99);EXPECT_TRUE(q.create(a,128,1,10));EXPECT_FALSE(q.create(a,128,1,10));EXPECT_TRUE(q.recordAttempt(1,10));EXPECT_TRUE(q.recordRejected(1));EXPECT_TRUE(q.recordAttempt(1,11));EXPECT_TRUE(q.recordAccepted(1,1,12));EXPECT_FALSE(q.complete(1,tok(999,DRAMSim::RequestKind::X_READ,4,4,99),20));EXPECT_TRUE(q.complete(1,a,21));EXPECT_FALSE(q.complete(1,a,22));EXPECT_TRUE(q.retire(1));EXPECT_FALSE(q.complete(1,a,23));EXPECT_FALSE(q.create(a,128,1,24));EXPECT_EQ(q.stats().completion_after_retirement,1);EXPECT_EQ(q.stats().retries,1);EXPECT_EQ(q.stats().submit_rejected,1);EXPECT_EQ(q.stats().unknown_completions,1);EXPECT_EQ(q.stats().duplicate_completions,1);CSCRequestTracker q2(7);auto b=tok(2,DRAMSim::RequestKind::VALUE_READ,8,3,100);ASSERT_TRUE(q2.create(b,256,2,1));ASSERT_TRUE(q2.recordAttempt(2,1));ASSERT_TRUE(q2.recordAccepted(2,2,2));auto stale=b;stale.generation=8;EXPECT_FALSE(q2.complete(2,stale,3));EXPECT_EQ(q2.stats().stale_generation_completions,1);auto wrong_kind=b;wrong_kind.request_kind=DRAMSim::RequestKind::INDEX_READ;EXPECT_FALSE(q2.complete(2,wrong_kind,4));EXPECT_EQ(q2.stats().kind_mismatches,1);auto wrong_owner=b;wrong_owner.worker_id=9;EXPECT_FALSE(q2.complete(2,wrong_owner,5));EXPECT_EQ(q2.stats().owner_mismatches,1);}
TEST(CSCM5OverlapTest,ThreeWayAndMultiBGOverlapBeatsSerialized){auto m=makeCSC(8,2,{{0,0,1},{1,0,2},{2,0,3},{3,0,4},{4,1,5},{5,1,6},{6,1,7},{7,1,8}});auto l=buildLayout(m,MappingPolicy::External,{0,17});Views vo(l,{2,3});CSCNativeExecution over(CSCRequestPolicy::OVERLAPPED);over.launch(vo.v,8,8);uint64_t g=0;while(!over.done()&&g++<200000)over.tick();ASSERT_TRUE(over.isDone());auto oy=over.hostAccumulate();auto oc=over.counters();Views vs(l,{2,3});CSCNativeExecution serial(CSCRequestPolicy::SERIALIZED);serial.launch(vs.v,8,8);g=0;while(!serial.done()&&g++<400000)serial.tick();ASSERT_TRUE(serial.isDone());auto sy=serial.hostAccumulate();auto sc=serial.counters();EXPECT_EQ(oy,sy);EXPECT_EQ(oy,cpuReference(l.matrix,{2,3}));EXPECT_GE(oc.maximum_global_outstanding,3);EXPECT_GE(oc.maximum_outstanding_per_engine,3);EXPECT_GT(oc.cycles_with_outstanding_ge_2,0);EXPECT_LT(oc.total_execution_cycles,sc.total_execution_cycles);EXPECT_EQ(oc.accepted_requests,oc.completion_count);EXPECT_EQ(sc.accepted_requests,sc.completion_count);EXPECT_EQ(oc.x_transactions,sc.x_transactions);EXPECT_EQ(oc.value_transactions,sc.value_transactions);EXPECT_EQ(oc.index_transactions,sc.index_transactions);EXPECT_EQ(oc.created_requests,oc.accepted_requests);EXPECT_EQ(sc.created_requests,sc.accepted_requests);EXPECT_EQ(oc.abandoned_waiting_requests,0);EXPECT_EQ(sc.abandoned_waiting_requests,0);EXPECT_EQ(oc.launch_to_done_cycles,oc.total_execution_cycles);EXPECT_EQ(sc.launch_to_done_cycles,sc.total_execution_cycles);RecordProperty("overlap_launch_to_done_cycles",oc.launch_to_done_cycles);RecordProperty("serialized_launch_to_done_cycles",sc.launch_to_done_cycles);RecordProperty("max_global_outstanding",oc.maximum_global_outstanding);RecordProperty("cycles_outstanding_ge_2",oc.cycles_with_outstanding_ge_2);}

TEST(CSCM5OverlapTest,BackpressureOnOneBGDoesNotBlockAnother){auto m=makeCSC(2,2,{{0,0,2},{1,1,3}});auto l=buildLayout(m,MappingPolicy::External,{0,17});Views v(l,{4,5});CSCNativeExecution n(CSCRequestPolicy::OVERLAPPED);n.setSubmitRejectBudget(0,20);n.launch(v.v,2,2);uint64_t guard=0;while(n.engine(17).tracker().stats().issued==0&&guard++<1000)n.tick();ASSERT_LT(guard,1000);EXPECT_EQ(n.engine(0).tracker().stats().issued,0);EXPECT_GT(n.engine(0).tracker().stats().submit_rejected,0);while(!n.done()&&guard++<200000)n.tick();ASSERT_TRUE(n.isDone());EXPECT_GE(n.engine(0).tracker().stats().retries,20);EXPECT_EQ(n.engine(0).tracker().stats().issued,3);EXPECT_EQ(n.hostAccumulate(),cpuReference(l.matrix,{4,5}));EXPECT_EQ(n.counters().accepted_requests,n.counters().completion_count);}


namespace csc_descriptor {
struct CSCFreezeTestAccess {
    static void setNextRequestId(CSCNativeExecution& execution, uint64_t value) {
        execution.next_request_id_ = value;
    }
    static uint64_t nextRequestId(const CSCNativeExecution& execution) {
        return execution.next_request_id_;
    }
    static void setGeneration(CSCNativeExecution& execution, uint32_t value) {
        execution.generation_ = value;
    }
    static uint32_t generation(const CSCNativeExecution& execution) {
        return execution.generation_;
    }
    static DRAMSim::RequestToken operandToken(const CSCNativeExecution& execution, uint32_t bg,
                                              DRAMSim::RequestKind kind) {
        const auto& engine = *execution.engines_.at(bg);
        if (kind == DRAMSim::RequestKind::X_READ) return engine.x_slot_.token;
        if (kind == DRAMSim::RequestKind::VALUE_READ) return engine.value_slot_.token;
        return engine.index_slot_.token;
    }
    static void injectFailure(CSCNativeExecution& execution, uint32_t bg,
                              const std::string& message) {
        auto& engine = *execution.engines_.at(bg);
        engine.fail(CSCError::INTERNAL_INVARIANT, message);
        execution.latchFailure(engine);
    }
};
}  // namespace csc_descriptor

TEST(CSCM5FreezeTest, AbandonedTrackerHistoryKindAndBGAccounting) {
    CSCRequestTracker tracker(9);
    auto x = tok(201, DRAMSim::RequestKind::X_READ, 0, 0, 1, 9);
    auto value = tok(202, DRAMSim::RequestKind::VALUE_READ, 17, 17, 2, 9);
    auto index = tok(203, DRAMSim::RequestKind::INDEX_READ, 63, 63, 3, 9);
    ASSERT_TRUE(tracker.create(x, 0, 0, 10));
    ASSERT_TRUE(tracker.create(value, 32, 4, 10));
    ASSERT_TRUE(tracker.create(index, 64, 15, 10));
    ASSERT_TRUE(tracker.recordAttempt(x.request_id, 11));
    ASSERT_TRUE(tracker.recordRejected(x.request_id));
    ASSERT_TRUE(tracker.recordAttempt(x.request_id, 12));
    EXPECT_EQ(tracker.abandonWaiting(13), 3);
    EXPECT_FALSE(tracker.abandon(x.request_id, 14));
    ASSERT_NE(tracker.find(x.request_id), nullptr);
    EXPECT_EQ(tracker.find(x.request_id)->state, CSCRequestState::ABANDONED);
    EXPECT_EQ(tracker.find(x.request_id)->abandoned_cycle, 13);
    EXPECT_EQ(tracker.find(x.request_id)->submit_attempts, 2);
    const auto& stats = tracker.stats();
    EXPECT_EQ(stats.created_requests, 3);
    EXPECT_EQ(stats.issued, 0);
    EXPECT_EQ(stats.completed, 0);
    EXPECT_EQ(stats.abandoned_waiting_requests, 3);
    EXPECT_EQ(stats.retries, 1);
    EXPECT_EQ(stats.abandoned_by_kind[size_t(DRAMSim::RequestKind::X_READ)], 1);
    EXPECT_EQ(stats.abandoned_by_kind[size_t(DRAMSim::RequestKind::VALUE_READ)], 1);
    EXPECT_EQ(stats.abandoned_by_kind[size_t(DRAMSim::RequestKind::INDEX_READ)], 1);
    EXPECT_EQ(stats.abandoned_by_bg[0], 1);
    EXPECT_EQ(stats.abandoned_by_bg[17], 1);
    EXPECT_EQ(stats.abandoned_by_bg[63], 1);
    EXPECT_TRUE(tracker.terminalAccountingValid());
}

TEST(CSCM5FreezeTest, ErrorBeforeAcceptanceAbandonsMultipleKindsAndBGs) {
    auto layout = buildLayout(makeCSC(2, 2, {{0, 0, 2}, {1, 1, 3}}),
                              MappingPolicy::External, {0, 17});
    Views views(layout, {4, 5});
    CSCNativeExecution execution;
    execution.setSubmitRejectBudget(0, 100);
    execution.setSubmitRejectBudget(17, 100);
    execution.launch(views.v, 2, 2);
    execution.tick();  // create x/value/index for both BGs
    execution.tick();  // first rejected attempt
    execution.tick();  // retry with the same IDs
    CSCFreezeTestAccess::injectFailure(execution, 0, "freeze abandonment test");
    CSCFreezeTestAccess::injectFailure(execution, 17, "secondary failure must not replace first");
    ASSERT_TRUE(execution.isTerminal());
    ASSERT_TRUE(execution.hasFailed());
    EXPECT_EQ(execution.failedEngine(), 0);
    auto counters = execution.counters();
    EXPECT_EQ(counters.created_requests, 6);
    EXPECT_EQ(counters.accepted_requests, 0);
    EXPECT_EQ(counters.completion_count, 0);
    EXPECT_EQ(counters.abandoned_waiting_requests, 6);
    EXPECT_EQ(counters.created_requests,
              counters.accepted_requests + counters.abandoned_waiting_requests);
    EXPECT_EQ(counters.accepted_requests, counters.completion_count);
    EXPECT_EQ(counters.retry_count, 2);
    EXPECT_EQ(counters.abandoned_by_kind[size_t(DRAMSim::RequestKind::X_READ)], 2);
    EXPECT_EQ(counters.abandoned_by_kind[size_t(DRAMSim::RequestKind::VALUE_READ)], 2);
    EXPECT_EQ(counters.abandoned_by_kind[size_t(DRAMSim::RequestKind::INDEX_READ)], 2);
    EXPECT_EQ(counters.abandoned_by_bg[0], 3);
    EXPECT_EQ(counters.abandoned_by_bg[17], 3);
    for (uint64_t id = 1; id <= 6; ++id) {
        const CSCRequestEntry* entry = id <= 3 ? execution.engine(0).tracker().find(id)
                                               : execution.engine(17).tracker().find(id);
        ASSERT_NE(entry, nullptr);
        EXPECT_EQ(entry->state, CSCRequestState::ABANDONED);
    }
    EXPECT_EQ(counters.launch_to_done_cycles, counters.total_execution_cycles);
}

TEST(CSCM5FreezeTest, RequestIdExhaustionDoesNotWrapOrRegisterInvalidId) {
    auto layout = buildLayout(makeCSC(1, 1, {{0, 0, 2}}), MappingPolicy::RoundRobin);
    Views views(layout, {3});
    CSCNativeExecution execution;
    CSCFreezeTestAccess::setNextRequestId(execution,
                                         std::numeric_limits<uint64_t>::max() - 1);
    execution.launch(views.v, 1, 1);
    execution.tick();
    ASSERT_TRUE(execution.isTerminal());
    EXPECT_TRUE(execution.hasFailed());
    EXPECT_NE(execution.errorMessage().find("request ID exhausted"), std::string::npos);
    EXPECT_EQ(CSCFreezeTestAccess::nextRequestId(execution),
              std::numeric_limits<uint64_t>::max());
    auto counters = execution.counters();
    EXPECT_EQ(counters.created_requests, 1);
    EXPECT_EQ(counters.accepted_requests, 0);
    EXPECT_EQ(counters.abandoned_waiting_requests, 1);
    const auto* last = execution.engine(0).tracker().find(
        std::numeric_limits<uint64_t>::max() - 1);
    ASSERT_NE(last, nullptr);
    EXPECT_EQ(last->state, CSCRequestState::ABANDONED);
    EXPECT_EQ(execution.engine(0).tracker().find(0), nullptr);
}

TEST(CSCM5FreezeTest, GenerationExhaustionIsStickyAcrossReset) {
    auto layout = buildLayout(makeCSC(1, 1, {{0, 0, 2}}), MappingPolicy::RoundRobin);
    Views first(layout, {3});
    CSCNativeExecution execution;
    const uint64_t original_next = CSCFreezeTestAccess::nextRequestId(execution);
    CSCFreezeTestAccess::setGeneration(execution, std::numeric_limits<uint32_t>::max());
    execution.launch(first.v, 1, 1);
    ASSERT_TRUE(execution.isTerminal());
    EXPECT_TRUE(execution.hasFailed());
    EXPECT_FALSE(execution.isDone());
    EXPECT_EQ(execution.cycle(), 0);
    EXPECT_NE(execution.errorMessage().find("generation exhausted"), std::string::npos);
    EXPECT_EQ(CSCFreezeTestAccess::generation(execution),
              std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(CSCFreezeTestAccess::nextRequestId(execution), original_next);
    execution.reset();
    Views second(layout, {3});
    execution.launch(second.v, 1, 1);
    EXPECT_TRUE(execution.isTerminal());
    EXPECT_TRUE(execution.hasFailed());
    EXPECT_NE(execution.errorMessage().find("generation exhausted"), std::string::npos);
    EXPECT_EQ(CSCFreezeTestAccess::generation(execution),
              std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(CSCFreezeTestAccess::nextRequestId(execution), original_next);
}

TEST(CSCM5FreezeTest, ResetRelaunchKeepsIdAndGenerationMonotonic) {
    auto layout = buildLayout(makeCSC(1, 1, {{0, 0, 2}}), MappingPolicy::RoundRobin);
    CSCNativeExecution execution;
    Views first(layout, {3});
    execution.launch(first.v, 1, 1);
    execution.tick();
    auto old_token = CSCFreezeTestAccess::operandToken(execution, 0,
                                                       DRAMSim::RequestKind::X_READ);
    while (!execution.done()) execution.tick();
    ASSERT_TRUE(execution.isDone());
    execution.reset();
    Views second(layout, {3});
    execution.launch(second.v, 1, 1);
    execution.tick();
    auto new_token = CSCFreezeTestAccess::operandToken(execution, 0,
                                                       DRAMSim::RequestKind::X_READ);
    EXPECT_GT(new_token.request_id, old_token.request_id);
    EXPECT_GT(new_token.generation, old_token.generation);
    EXPECT_EQ(new_token.sequence_number, old_token.sequence_number);
    auto stale = new_token;
    stale.generation = old_token.generation;
    CSCRequestTracker tracker(new_token.generation);
    ASSERT_TRUE(tracker.create(new_token, 0, 0, 0));
    ASSERT_TRUE(tracker.recordAttempt(new_token.request_id, 0));
    ASSERT_TRUE(tracker.recordAccepted(new_token.request_id, 0, 0));
    EXPECT_FALSE(tracker.complete(0, stale, 1));
    EXPECT_EQ(tracker.stats().stale_generation_completions, 1);
    while (!execution.done()) execution.tick();
    EXPECT_TRUE(execution.isDone());
}
