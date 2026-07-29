#ifndef CSC_TIMING_MODEL_H
#define CSC_TIMING_MODEL_H
#include "MultiChannelMemorySystem.h"
#include "tests/KernelAddrGen.h"
#include "tests/csc/CSCFunctionalModel.h"
#include <memory>
namespace csc_descriptor {
struct TimingStats{uint64_t matrix_preload_cycles=0,x_load_cycles=0,value_read_cycles=0,row_index_read_cycles=0,result_cycles=0,synchronization_drain_cycles=0,kernel_resident_cycles=0,kernel_with_x_cycles=0,end_to_end_simulated_cycles=0,matrix_preload_bursts=0,x_bursts=0,value_bursts=0,row_index_bursts=0,result_bursts=0;};
enum class ResultTrafficMode{None,Analytical};
class CSCTimingModel{public:CSCTimingModel();TimingStats run(const CSCLayout&,const ExecutionStats&,ResultTrafficMode result=ResultTrafficMode::Analytical);uint64_t globalBGCount()const;void validateTopologyAddresses(const CSCLayout&);private:uint64_t address(uint32_t,uint32_t,uint32_t,uint64_t);uint64_t drain();uint64_t image(const CSCLayout&,bool,bool,uint32_t,uint32_t,const char*);std::shared_ptr<DRAMSim::MultiChannelMemorySystem> mem_;std::shared_ptr<PIMAddrManager> am_;unsigned channels_=0,ranks_=0,bgs_=0,banks_=0;uint64_t cycle_=0;DRAMSim::BurstType burst_;};
}
#endif
