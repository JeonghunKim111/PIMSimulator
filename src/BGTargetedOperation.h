#ifndef BG_TARGETED_OPERATION_H
#define BG_TARGETED_OPERATION_H

#include <array>
#include <cstdint>

#include "Burst.h"

namespace DRAMSim
{
constexpr uint32_t kBGsPerRank = 4;
constexpr uint32_t kPIMBlocksPerBG = 2;
constexpr uint8_t kBGTargetFirstPIMBlock = 0x1;
constexpr uint8_t kBGTargetSecondPIMBlock = 0x2;
constexpr uint8_t kBGTargetBothPIMBlocks = 0x3;

enum class BGTargetedOpcode : uint8_t
{
    MASKED_MUL
};

enum class BGTargetedOperationState : uint8_t
{
    READY,
    WAITING_FOR_GRANT,
    ACCEPTED,
    EXECUTING,
    COMPLETED,
    RETIRED
};

enum class RankExecutionMode : uint8_t
{
    IDLE,
    LEGACY_RANK_WIDE,
    CSC_BG_TARGETED,
    DRAINING,
    ERROR_DRAINING,
    ERROR
};

enum class BGLifecycleState : uint8_t
{
    IDLE,
    RUNNING,
    FLUSH_REQUESTED,
    DRAINING,
    FLUSHED,
    ERROR_DRAINING,
    ERROR,
    RESETTING
};

struct BGTargetedIdentity
{
    uint64_t operation_id = 0;
    uint32_t channel = 0;
    uint32_t rank = 0;
    uint32_t local_bg = 0;
    uint8_t pimblock_mask = 0;
    BGTargetedOpcode opcode = BGTargetedOpcode::MASKED_MUL;
    uint32_t worker_id = 0;
    uint64_t sequence = 0;
    uint32_t generation = 0;

    bool operator==(const BGTargetedIdentity& other) const
    {
        return operation_id == other.operation_id && channel == other.channel &&
               rank == other.rank && local_bg == other.local_bg &&
               pimblock_mask == other.pimblock_mask && opcode == other.opcode &&
               worker_id == other.worker_id && sequence == other.sequence &&
               generation == other.generation;
    }
};

struct BGTargetedOperandContext
{
    BurstType lhs{};
    BurstType rhs{};
    bool valid = false;
};

struct BGTargetedOperation
{
    BGTargetedIdentity identity{};
    uint32_t valid_count = 0;
    std::array<BGTargetedOperandContext, kPIMBlocksPerBG> contexts{};
    BGTargetedOperationState state = BGTargetedOperationState::READY;
};

struct BGTargetedCompletion
{
    BGTargetedIdentity identity{};
    std::array<BurstType, kPIMBlocksPerBG> results{};
    uint64_t grant_cycle = 0;
    uint64_t completion_cycle = 0;
};
}  // namespace DRAMSim

#endif
