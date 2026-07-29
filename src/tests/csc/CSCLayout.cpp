#include "tests/csc/CSCLayout.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
namespace csc_descriptor {
static uint64_t align32(uint64_t n){return(n+31)/32*32;}
template<class T>static void append(std::vector<uint8_t>&dst,const std::vector<T>&src,uint64_t begin,uint64_t count){dst.resize(align32(dst.size()),0);size_t at=dst.size();dst.resize(at+count*sizeof(T));if(count)std::memcpy(dst.data()+at,src.data()+begin,count*sizeof(T));dst.resize(align32(dst.size()),0);}
std::string mappingPolicyName(MappingPolicy p){return p==MappingPolicy::RoundRobin?"round_robin":"external_mapping";}
CSCLayout buildLayout(const CSCMatrix&m,MappingPolicy p,const std::vector<uint32_t>&ext){
 if(m.col_ptr.size()!=size_t(m.cols)+1||m.values.size()!=m.row_idx.size()||(!m.col_ptr.empty()&&m.col_ptr.back()!=m.values.size()))throw std::invalid_argument("invalid CSC arrays");
 if(p==MappingPolicy::External&&ext.size()!=m.cols)throw std::invalid_argument("external mapping size mismatch");
 CSCLayout l;l.matrix=m;l.mapping_policy=p;l.column_to_bg.resize(m.cols);l.bg.resize(kGlobalBGs);
 for(uint32_t c=0;c<m.cols;++c){uint32_t owner=p==MappingPolicy::RoundRobin?c%kGlobalBGs:ext[c];if(owner>=kGlobalBGs)throw std::invalid_argument("BG outside [0,63]");l.column_to_bg[c]=owner;uint64_t begin=m.col_ptr[c],n=m.col_ptr[c+1]-begin;if(!n)continue;auto&b=l.bg[owner];uint32_t slot=b.x_slot_to_original_col.size();b.x_slot_to_original_col.push_back(c);b.descriptors.push_back({align32(b.values.size()),align32(b.row_indices.size()),uint32_t(n),slot,c,owner});append(b.values,m.values,begin,n);append(b.row_indices,m.row_idx,begin,n);}
 auto&s=l.stats;s.num_rows=m.rows;s.num_cols=m.cols;s.nnz=m.values.size();s.bg_nnz.assign(64,0);s.bg_descriptor_count.assign(64,0);
 for(uint32_t g=0;g<64;++g){auto&b=l.bg[g];s.physical_value_bytes+=b.values.size();s.physical_index_bytes+=b.row_indices.size();s.descriptor_count+=b.descriptors.size();s.nonempty_columns+=b.descriptors.size();s.bg_descriptor_count[g]=b.descriptors.size();s.x_useful_bytes+=b.x_slot_to_original_col.size()*4;s.x_padding_bytes+=align32(b.x_slot_to_original_col.size()*4)-b.x_slot_to_original_col.size()*4;for(auto&d:b.descriptors)s.bg_nnz[g]+=d.nnz_count;s.max_bg_nnz=std::max(s.max_bg_nnz,s.bg_nnz[g]);}
 s.useful_value_bytes=s.nnz*4;s.useful_index_bytes=s.nnz*4;s.value_alignment_padding_bytes=s.physical_value_bytes-s.useful_value_bytes;s.index_alignment_padding_bytes=s.physical_index_bytes-s.useful_index_bytes;s.total_alignment_padding_bytes=s.value_alignment_padding_bytes+s.index_alignment_padding_bytes;uint64_t useful=s.useful_value_bytes+s.useful_index_bytes;s.alignment_overhead_ratio=useful?double(s.total_alignment_padding_bytes)/useful:0;s.descriptor_bytes=s.descriptor_count*sizeof(CSCDescriptor);s.average_bg_nnz=double(s.nnz)/64;double var=0;for(auto n:s.bg_nnz)var+=(n-s.average_bg_nnz)*(n-s.average_bg_nnz);s.load_cv=s.average_bg_nnz?std::sqrt(var/64)/s.average_bg_nnz:0;validateLayout(l);return l;
}
void validateLayout(const CSCLayout&l){
 if(l.bg.size()!=64||l.column_to_bg.size()!=l.matrix.cols)throw std::logic_error("topology invariant");
 std::vector<uint8_t> seen(l.matrix.cols);
 uint64_t nnz=0;
 for(uint32_t g=0;g<64;++g){auto&b=l.bg[g];if(b.descriptors.size()!=b.x_slot_to_original_col.size())throw std::logic_error("x mapping invariant");for(auto&d:b.descriptors){if(d.global_bg_id!=g||d.original_col>=l.matrix.cols||l.column_to_bg[d.original_col]!=g||d.value_offset_bytes%32||d.row_idx_offset_bytes%32||!d.nnz_count||d.x_slot>=b.x_slot_to_original_col.size()||b.x_slot_to_original_col[d.x_slot]!=d.original_col)throw std::logic_error("descriptor invariant");if(d.value_offset_bytes+d.nnz_count*4>b.values.size()||d.row_idx_offset_bytes+d.nnz_count*4>b.row_indices.size())throw std::logic_error("payload range");seen[d.original_col]++;nnz+=d.nnz_count;}}
 for(uint32_t c=0;c<l.matrix.cols;++c){
  if(seen[c]!=(l.matrix.col_ptr[c+1]>l.matrix.col_ptr[c]))throw std::logic_error("one descriptor per nonempty column");
 }
 if(nnz!=l.matrix.values.size())throw std::logic_error("NNZ sum invariant");
}
}
