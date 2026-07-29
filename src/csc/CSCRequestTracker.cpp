#include "csc/CSCRequestTracker.h"
#include <stdexcept>
namespace csc_descriptor {
void CSCRequestTracker::recordIssue(CSCRequestKind k,uint64_t a,unsigned c,uint32_t bg){if(pending_||completion_ready_)throw std::logic_error("ambiguous CSC outstanding request");kind_=k;address_=a;channel_=c;bg_=bg;pending_=true;stats_.issued++;}
bool CSCRequestTracker::complete(unsigned c,uint64_t a,uint64_t cycle){if(!pending_||c!=channel_||a!=address_)return false;pending_=false;completion_ready_=true;completion_cycle_=cycle;stats_.completed++;return true;}
void CSCRequestTracker::consumeCompletion(){if(!completion_ready_)throw std::logic_error("CSC completion not ready");completion_ready_=false;}
void CSCRequestTracker::reset(){if(pending_)throw std::logic_error("reset with pending CSC request");completion_ready_=false;error_.clear();stats_={};}
}
