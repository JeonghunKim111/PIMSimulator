#ifndef CSC_EXTERNAL_IMAGE_H
#define CSC_EXTERNAL_IMAGE_H
#include "tests/csc/CSCLayout.h"
namespace csc_descriptor {
struct ExternalImageMetadata {
 std::string mapping_policy,input_value_type,output_value_type;
 uint64_t fp32_conversion_count=0,nan_count=0,inf_count=0;
 double fp32_conversion_ms=0,aligned_materialization_ms=0,metadata_generation_ms=0,validation_ms=0,checksum_ms=0,serialization_ms=0,file_write_ms=0;
};
struct ExternalImage {CSCLayout layout;ExternalImageMetadata metadata;};
ExternalImage loadExternalPhysicalImage(const std::string& directory);
}
#endif
