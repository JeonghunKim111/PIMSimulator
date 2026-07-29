#ifndef CSC_LAYOUT_H
#define CSC_LAYOUT_H
#include "csc/CSCTypes.h"
#include "tests/csc/CSCMatrixLoader.h"
#include <string>
namespace csc_descriptor {
constexpr uint32_t kGlobalBGs=64,kBGsPerChannel=4,kChannels=16,kSIMDWidth=8,kBurstBytes=32;
enum class MappingPolicy{RoundRobin,External};
enum class LayoutSource{InternalBuilder,ExternalPhysicalImage};
struct BGImage{std::vector<uint8_t> values,row_indices;std::vector<CSCDescriptor> descriptors;std::vector<uint32_t> x_slot_to_original_col;};
struct LayoutStats{
 uint64_t num_rows=0,num_cols=0,nnz=0,num_global_bg=kGlobalBGs,nonempty_columns=0,descriptor_count=0;
 uint64_t useful_value_bytes=0,value_alignment_padding_bytes=0,physical_value_bytes=0,useful_index_bytes=0,index_alignment_padding_bytes=0,physical_index_bytes=0,total_alignment_padding_bytes=0,x_useful_bytes=0,x_padding_bytes=0,descriptor_bytes=0;
 double alignment_overhead_ratio=0,average_bg_nnz=0,load_cv=0;uint64_t max_bg_nnz=0;std::vector<uint64_t> bg_nnz,bg_descriptor_count;
};
struct CSCLayout{CSCMatrix matrix;MappingPolicy mapping_policy;LayoutSource source=LayoutSource::InternalBuilder;std::vector<uint32_t> column_to_bg;std::vector<BGImage> bg;LayoutStats stats;};
CSCLayout buildLayout(const CSCMatrix&,MappingPolicy,const std::vector<uint32_t>& external={});
void validateLayout(const CSCLayout&);void refreshLayoutStats(CSCLayout&);std::string mappingPolicyName(MappingPolicy);
}
#endif
