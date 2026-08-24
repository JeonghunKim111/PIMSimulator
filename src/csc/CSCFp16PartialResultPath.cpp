#include "csc/CSCFp16PartialResultPath.h"

#include "csc/CSCFp16.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace csc_descriptor {
namespace {
void addByte(uint8_t byte, uint64_t& hash) { hash = (hash ^ byte) * 1099511628211ULL; }
void add32(uint32_t value, uint64_t& hash) {
    for (unsigned i=0;i<4;++i) addByte(uint8_t(value>>(8*i)), hash);
}
void add64(uint64_t value, uint64_t& hash) {
    for (unsigned i=0;i<8;++i) addByte(uint8_t(value>>(8*i)), hash);
}
}

std::array<uint8_t, 8> serializeCSCFp16TransportRecord(
    const CSCFp16TransportRecord& record)
{
    if (record.reserved) throw std::invalid_argument("FP16 transport reserved field");
    std::array<uint8_t,8> bytes{};
    for (unsigned i=0;i<4;++i) bytes[i]=uint8_t(record.row_idx>>(8*i));
    bytes[4]=uint8_t(record.value_bits); bytes[5]=uint8_t(record.value_bits>>8);
    return bytes;
}

CSCFp16TransportRecord deserializeCSCFp16TransportRecord(
    const uint8_t* bytes, std::size_t size)
{
    if (!bytes || size < 8) throw std::invalid_argument("truncated FP16 transport record");
    CSCFp16TransportRecord record;
    for (unsigned i=0;i<4;++i) record.row_idx |= uint32_t(bytes[i])<<(8*i);
    record.value_bits = uint16_t(bytes[4]) | uint16_t(bytes[5])<<8;
    record.reserved = uint16_t(bytes[6]) | uint16_t(bytes[7])<<8;
    if (record.reserved) throw std::invalid_argument("nonzero FP16 transport reserved field");
    return record;
}

void CSCFp16TransportConfig::validate() const
{
    if (!global_bg_count || !channel_count || !bank_groups_per_rank ||
        global_bg_count != channel_count*bank_groups_per_rank ||
        !buffer_capacity_bursts_per_bg || !pending_capacity_bursts_per_bg ||
        !write_latency_cycles || !max_inflight_writes_per_bg ||
        !write_issue_limit_per_rank_per_cycle || !read_latency_cycles ||
        !read_issue_limit_per_channel_per_cycle ||
        !max_inflight_reads_per_channel ||
        (host_reduction_enabled &&
         (!host_reduce_records_per_cycle || !host_reduce_latency_cycles)))
        throw std::invalid_argument("invalid FP16 transport configuration");
}

CSCFp16PartialResultPath::CSCFp16PartialResultPath(
    const CSCFp16TransportConfig& config, uint32_t rows)
    : config_(config), rows_(rows)
{
    config_.validate();
    if (!rows_) throw std::invalid_argument("invalid FP16 transport rows");
    bg_.resize(config_.global_bg_count);
    rank_rr_.resize(config_.global_bg_count/config_.bank_groups_per_rank);
    channel_rr_.resize(config_.channel_count);
    reads_by_channel_.resize(config_.channel_count);
    final_y_.assign(rows_, 0x0000);
    row_touched_.assign(rows_, false);
    row_first_bg_.assign(rows_, 0);
    write_reject_budget_=config_.write_reject_attempts;
    read_reject_budget_=config_.read_reject_attempts;
    counters_.records_per_bg.resize(config_.global_bg_count);
    counters_.bursts_per_bg.resize(config_.global_bg_count);
}

void CSCFp16PartialResultPath::fail(const std::string& message)
{ if (!error_) { error_=true; error_message_=message; } }

bool CSCFp16PartialResultPath::ready(const CSCFp16BGAOutputEvent& event)
{
    counters_.outputs_offered++;
    if (error_ || event.global_bg_id>=bg_.size() || event.row_idx>=rows_ ||
        !event.sequence || !event.contribution_count) { fail("invalid FP16 transport output"); return false; }
    const auto& state=bg_[event.global_bg_id];
    if (event.sequence!=state.last_output_sequence+1) { fail("FP16 transport output sequence"); return false; }
    if (state.packer_count==3 && state.pending.size()>=config_.pending_capacity_bursts_per_bg) {
        counters_.transport_stall_cycles++; return false;
    }
    return true;
}

bool CSCFp16PartialResultPath::accept(const CSCFp16BGAOutputEvent& event)
{
    if (!ready(event)) return false;
    auto& state=bg_[event.global_bg_id];
    state.packer[state.packer_count++]=event;
    state.last_output_sequence=event.sequence;
    counters_.outputs_accepted++; counters_.records_packed++;
    counters_.records_per_bg[event.global_bg_id]++;
    if (!counters_.first_output_accept_cycle) counters_.first_output_accept_cycle=cycle_;
    accepted_trace_.push_back(event);
    if (state.packer_count==4) formBurst(event.global_bg_id,false);
    return !error_;
}

void CSCFp16PartialResultPath::formBurst(uint32_t id, bool tail)
{
    auto& state=bg_[id]; const uint8_t count=state.packer_count;
    if ((!tail&&count!=4)||(tail&&(!count||count>=4))||state.pending.size()>=config_.pending_capacity_bursts_per_bg) { fail("invalid FP16 burst formation"); return; }
    CSCFp16TransportBurst burst; burst.global_bg_id=id; burst.sequence=state.next_sequence++;
    burst.valid_record_count=count; burst.tail=tail; burst.formation_cycle=cycle_;
    for (uint8_t i=0;i<count;++i) {
        const auto bytes=serializeCSCFp16TransportRecord({state.packer[i].row_idx,state.packer[i].value_bits,0});
        std::copy(bytes.begin(),bytes.end(),burst.bytes.begin()+i*8);
        burst.contribution_counts[i]=state.packer[i].contribution_count;
    }
    state.pending.push_back(burst); state.packer_count=0;
    counters_.bursts_per_bg[id]++; counters_.write_bytes+=32;
    if (tail) { counters_.tail_bursts++; counters_.tail_padding_bytes+=32-count*8; }
    else counters_.full_bursts++;
}

uint64_t CSCFp16PartialResultPath::totalInflightWrites() const { uint64_t n=0; for(auto&s:bg_)n+=s.writes.size(); return n; }
uint64_t CSCFp16PartialResultPath::totalInflightReads() const { uint64_t n=0; for(auto&s:reads_by_channel_)n+=s.size(); return n; }

void CSCFp16PartialResultPath::completeWrites()
{
    for(auto& state:bg_) while(!state.writes.empty()&&state.writes.front().completion<=cycle_) {
        auto burst=state.writes.front().burst; state.writes.pop_front();
        burst.completion_cycle=cycle_; state.resident.push_back(burst); resident_trace_.push_back(burst);
        counters_.write_completions++; counters_.resident_bursts++;
        counters_.last_write_complete_cycle=cycle_; if(!counters_.first_write_complete_cycle)counters_.first_write_complete_cycle=cycle_;
    }
}

void CSCFp16PartialResultPath::issueWrites()
{
    for(uint32_t rank=0;rank<rank_rr_.size();++rank) for(uint32_t issued=0;issued<config_.write_issue_limit_per_rank_per_cycle;) {
        bool found=false;
        for(uint32_t scan=0;scan<config_.bank_groups_per_rank;++scan) {
            uint32_t local=rank_rr_[rank]++%config_.bank_groups_per_rank, id=rank*config_.bank_groups_per_rank+local;
            auto& state=bg_[id]; counters_.write_attempts++;
            if(state.pending.empty()||state.pending.front().formation_cycle>=cycle_||state.writes.size()>=config_.max_inflight_writes_per_bg||state.resident.size()+state.writes.size()>=config_.buffer_capacity_bursts_per_bg) continue;
            if(write_reject_budget_){--write_reject_budget_;counters_.write_retries++;continue;}
            auto burst=state.pending.front(); state.pending.pop_front(); burst.issue_cycle=cycle_;
            state.writes.push_back({burst,cycle_+config_.write_latency_cycles});
            counters_.writes_accepted++; counters_.last_write_issue_cycle=cycle_; if(!counters_.first_write_issue_cycle)counters_.first_write_issue_cycle=cycle_;
            counters_.outstanding_write_high_water=std::max(counters_.outstanding_write_high_water,totalInflightWrites()); found=true; issued++; break;
        }
        if(!found) break;
    }
}

bool CSCFp16PartialResultPath::allWritesComplete() const
{
    for(const auto&s:bg_) if(!s.lifecycle_complete||!s.tail_flushed||s.packer_count||!s.pending.empty()||!s.writes.empty()) return false;
    return true;
}

void CSCFp16PartialResultPath::completeReads()
{
    for(auto& queue:reads_by_channel_) while(!queue.empty()&&queue.front().completion<=cycle_) {
        auto burst=queue.front().burst; queue.pop_front();
        auto key=std::make_pair(burst.global_bg_id,burst.sequence);
        if(!returned_.emplace(key,burst).second){fail("duplicate FP16 read completion");return;}
        counters_.read_completions++; counters_.read_bytes+=32; counters_.last_read_complete_cycle=cycle_;
        if(!counters_.first_read_complete_cycle)counters_.first_read_complete_cycle=cycle_;
    }
}

void CSCFp16PartialResultPath::issueReads()
{
    for(uint32_t channel=0;channel<config_.channel_count;++channel) {
        auto& inflight=reads_by_channel_[channel];
        for(uint32_t issued=0;issued<config_.read_issue_limit_per_channel_per_cycle&&inflight.size()<config_.max_inflight_reads_per_channel;) {
            bool found=false;
            for(uint32_t scan=0;scan<config_.bank_groups_per_rank;++scan) {
                uint32_t local=channel_rr_[channel]++%config_.bank_groups_per_rank, id=channel*config_.bank_groups_per_rank+local;
                auto& resident=bg_[id].resident; counters_.read_attempts++;
                if(resident.empty()) continue;
                if(read_reject_budget_){--read_reject_budget_;counters_.read_retries++;continue;}
                auto burst=resident.front(); resident.erase(resident.begin());
                inflight.push_back({burst,cycle_+config_.read_latency_cycles}); counters_.reads_accepted++;
                counters_.first_read_issue_cycle=counters_.first_read_issue_cycle?counters_.first_read_issue_cycle:cycle_;
                counters_.outstanding_read_high_water=std::max(counters_.outstanding_read_high_water,totalInflightReads()); found=true; issued++; break;
            }
            if(!found)break;
        }
    }
}

bool CSCFp16PartialResultPath::allReadsComplete() const
{ for(const auto&s:bg_)if(!s.resident.empty())return false; for(const auto&q:reads_by_channel_)if(!q.empty())return false; return counters_.read_completions==counters_.full_bursts+counters_.tail_bursts; }

void CSCFp16PartialResultPath::reduceOrdered()
{
    if(reduction_ready_cycle_>cycle_)return;
    uint32_t issued=0;
    while(issued<config_.host_reduce_records_per_cycle&&reduce_bg_<bg_.size()) {
        if(reduce_sequence_>counters_.bursts_per_bg[reduce_bg_]) { reduce_bg_++; reduce_sequence_=1; reduce_slot_=0; continue; }
        auto it=returned_.find({reduce_bg_,reduce_sequence_}); if(it==returned_.end())return;
        const auto& burst=it->second;
        if(reduce_slot_>=burst.valid_record_count) { returned_.erase(it); reduce_sequence_++; reduce_slot_=0; continue; }
        try {
            const auto record=deserializeCSCFp16TransportRecord(burst.bytes.data()+reduce_slot_*8,8);
            if(record.row_idx>=rows_){fail("FP16 reduction row out of range");return;}
            if(row_touched_[record.row_idx]&&row_first_bg_[record.row_idx]!=reduce_bg_)counters_.cross_bg_same_row_adds++;
            if(!row_touched_[record.row_idx]){row_touched_[record.row_idx]=true;row_first_bg_[record.row_idx]=reduce_bg_;counters_.rows_touched++;}
            final_y_[record.row_idx]=cscFp16ToBits(cscFp16Add(cscFp16FromBits(final_y_[record.row_idx]),cscFp16FromBits(record.value_bits)));
            counters_.records_decoded++; counters_.records_reduced++; counters_.fp16_host_adds++;
            counters_.first_reduce_cycle=counters_.first_reduce_cycle?counters_.first_reduce_cycle:cycle_; counters_.last_reduce_cycle=cycle_;
            reduce_slot_++; issued++;
        } catch(const std::exception&e){fail(e.what());return;}
    }
    if(issued)reduction_ready_cycle_=cycle_+config_.host_reduce_latency_cycles;
}

void CSCFp16PartialResultPath::step(uint64_t cycle,const std::vector<bool>& final_done)
{
    if(error_||done()) return;
    cycle_=cycle;
    completeWrites();
    if(final_done.size()!=bg_.size()){fail("FP16 transport topology mismatch");return;}
    for(uint32_t id=0;id<bg_.size();++id){auto&s=bg_[id]; if(final_done[id])s.lifecycle_complete=true; if(s.lifecycle_complete&&!s.tail_flushed){if(s.packer_count&&s.pending.size()<config_.pending_capacity_bursts_per_bg)formBurst(id,true); if(!s.packer_count)s.tail_flushed=true;}}
    issueWrites();
    if(!counters_.writeback_complete_cycle&&allWritesComplete()){counters_.writeback_complete_cycle=cycle_; readback_started_=true; counters_.readback_start_cycle=cycle_;}
    if(readback_started_){completeReads();issueReads();if(!counters_.readback_complete_cycle&&allReadsComplete())counters_.readback_complete_cycle=cycle_;if(config_.host_reduction_enabled)reduceOrdered();}
    if(counters_.readback_complete_cycle&&!config_.host_reduction_enabled&&!counters_.end_to_end_cycle){
        if(counters_.outputs_accepted!=counters_.records_packed||counters_.writes_accepted!=counters_.write_completions||counters_.writes_accepted!=counters_.reads_accepted||counters_.reads_accepted!=counters_.read_completions||counters_.write_bytes!=counters_.read_bytes){fail("FP16 transport-only conservation mismatch");return;}
        counters_.end_to_end_cycle=cycle_;
    }
    if(config_.host_reduction_enabled&&counters_.readback_complete_cycle&&reduce_bg_==bg_.size()&&returned_.empty()&&!counters_.reduction_complete_cycle){
        if(counters_.outputs_accepted!=counters_.records_reduced||counters_.writes_accepted!=counters_.read_completions||counters_.write_bytes!=counters_.read_bytes){fail("FP16 end-to-end conservation mismatch");return;}
        counters_.reduction_complete_cycle=cycle_; counters_.end_to_end_cycle=cycle_;
    }
}

const std::vector<CSCFp16Bits>& CSCFp16PartialResultPath::finalYBits() const
{ if(!done()||!config_.host_reduction_enabled)throw std::logic_error("FP16 final y unavailable"); return final_y_; }

uint64_t cscFp16TransportRecordTraceFnv1a64(const std::vector<CSCFp16BGAOutputEvent>& trace)
{ uint64_t h=1469598103934665603ULL; for(auto&e:trace){auto b=serializeCSCFp16TransportRecord({e.row_idx,e.value_bits,0});for(auto v:b)addByte(v,h);add32(e.global_bg_id,h);add64(e.sequence,h);}return h; }
uint64_t cscFp16BurstTraceFnv1a64(const std::vector<CSCFp16TransportBurst>& trace)
{ uint64_t h=1469598103934665603ULL;for(auto&b:trace){add32(b.global_bg_id,h);add64(b.sequence,h);addByte(b.valid_record_count,h);for(auto v:b.bytes)addByte(v,h);}return h; }
uint64_t cscFp16FinalYFnv1a64(const std::vector<CSCFp16Bits>& y)
{ uint64_t h=1469598103934665603ULL;for(auto v:y){addByte(uint8_t(v),h);addByte(uint8_t(v>>8),h);}return h; }

}  // namespace csc_descriptor
