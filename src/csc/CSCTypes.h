#ifndef CSC_TYPES_H
#define CSC_TYPES_H
#include <cstdint>
namespace csc_descriptor {
constexpr uint32_t kCSCGlobalBGs=64,kCSCSIMDWidth=8,kCSCBurstBytes=32;
enum class CSCError {
    NONE,
    MALFORMED_DESCRIPTOR,
    REQUEST_COMPLETION,
    INTERNAL_INVARIANT
};
struct CSCDescriptor{uint64_t value_offset_bytes,row_idx_offset_bytes;uint32_t nnz_count,x_slot,original_col,global_bg_id;};
}
#endif
