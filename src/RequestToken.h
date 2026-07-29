#ifndef REQUEST_TOKEN_H
#define REQUEST_TOKEN_H

#include <cstdint>

namespace DRAMSim {

enum class RequestKind : uint8_t {
    NONE = 0,
    VALUE_READ,
    INDEX_READ,
    X_READ,
    DESCRIPTOR_READ,
    RESULT_TRANSFER
};

struct RequestToken {
    uint64_t request_id = 0;
    RequestKind request_kind = RequestKind::NONE;
    uint32_t global_bg_id = 0;
    uint32_t worker_id = 0;
    uint64_t sequence_number = 0;
    uint32_t client_id = 0;
    uint32_t generation = 0;

    bool valid() const { return request_id != 0; }
    bool operator==(const RequestToken& other) const {
        return request_id == other.request_id && request_kind == other.request_kind &&
               global_bg_id == other.global_bg_id && worker_id == other.worker_id &&
               sequence_number == other.sequence_number && client_id == other.client_id &&
               generation == other.generation;
    }
};

}  // namespace DRAMSim

#endif  // REQUEST_TOKEN_H
