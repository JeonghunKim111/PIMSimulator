#ifndef CSC_REQUEST_TRACKER_H
#define CSC_REQUEST_TRACKER_H
#include <cstdint>
#include <string>
namespace csc_descriptor {
enum class CSCRequestKind{X_READ,VALUE_READ,INDEX_READ};
struct CSCRequestStats{uint64_t issued=0,completed=0,submit_rejected=0,ambiguous_completions=0;};
class CSCRequestTracker{
public:
 bool pending()const{return pending_;} bool completionReady()const{return completion_ready_;}
 CSCRequestKind kind()const{return kind_;} uint64_t address()const{return address_;}
 void recordIssue(CSCRequestKind,uint64_t,unsigned,uint32_t);
 bool complete(unsigned,uint64_t,uint64_t); void consumeCompletion(); void reset();
 const CSCRequestStats&stats()const{return stats_;} const std::string&error()const{return error_;}
 void recordSubmitRejected(){stats_.submit_rejected++;}
private:
 bool pending_=false,completion_ready_=false;CSCRequestKind kind_=CSCRequestKind::X_READ;
 uint64_t address_=0,completion_cycle_=0;unsigned channel_=0;uint32_t bg_=0;
 CSCRequestStats stats_;std::string error_;
};
}
#endif
