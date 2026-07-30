/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system, or translated into any human
 * or computer language in any form by any means,electronic, mechanical, manual or otherwise,
 * or disclosed to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose
 * only)
 **************************************************************************************************/

#ifndef _PIMRANK_H_
#define _PIMRANK_H_

#include <optional>
#include <vector>

#include "AddressMapping.h"
#include "BusPacket.h"
#include "Configuration.h"
#include "BGTargetedOperation.h"
#include "PIMBlock.h"
#include "PIMCmd.h"
#include "Rank.h"
#include "SimulatorObject.h"

using namespace std;
using namespace DRAMSim;

namespace DRAMSim
{
struct PIMRankM6TestAccess;
#define OUTLOG_ALL(msg)                                                                       \
    msg << " ch[" << getChanId() << "] ra[" << getRankId() << "] bg["                         \
        << config.addrMapping.bankgroupId(packet->bank) << "] ba[" << packet->bank << "] ro[" \
        << packet->row << "] co[" << packet->column << "] @" << currentClockCycle
#define OUTLOG_CH_RA(msg) \
    msg << " ch[" << getChanId() << "] ra[" << getRankId() << "] @" << currentClockCycle
#define OUTLOG_PRECHARGE(msg)                                                                 \
    msg << " ch[" << getChanId() << "] ra[" << getRankId() << "] bg["                         \
        << config.addrMapping.bankgroupId(packet->bank) << "] ba[" << packet->bank << "] ro[" \
        << bankStates[packet->bank].openRowAddress << "] @" << currentClockCycle
#define OUTLOG_GRF_A(msg)                                                                 \
    msg << " ch[" << getChanId() << "] ra[" << getRankId() << "] pb[" << packet->bank / 2 \
        << " reg" << packet->column - 0x8 << " @" << currentClockCycle
#define OUTLOG_GRF_B(msg)                                                                 \
    msg << " ch[" << getChanId() << "] ra[" << getRankId() << "] pb[" << packet->bank / 2 \
        << "] reg[" << packet->column - 0x18 << "] @" << currentClockCycle
#define OUTLOG_B_GRF_A(msg)                                                                    \
    msg << " ch[" << getChanId() << "] ra[" << getRankId() << "] reg[" << packet->column - 0x8 \
        << "] @" << currentClockCycle
#define OUTLOG_B_GRF_B(msg)                                                                     \
    msg << " ch[" << getChanId() << "] ra[" << getRankId() << "] reg[" << packet->column - 0x18 \
        << "] @" << currentClockCycle
#define OUTLOG_B_CRF(msg)                                                                      \
    msg << " ch[" << getChanId() << "] ra[" << getRankId() << "] idx[" << packet->column - 0x4 \
        << "] @" << currentClockCycle

class Rank;  // forward declaration

class PIMRank : public SimulatorObject
{
  private:
    struct BGPendingSlot
    {
        bool occupied = false;
        BGTargetedOperation operation{};
    };

    int chanId;
    int rankId;
    ostream& dramsimLog;
    Configuration& config;
    int pimPC_, lastJumpIdx_, numJumpToBeTaken_, lastRepeatIdx_, numRepeatToBeDone_;
    bool pimOpMode_, toggleEvenBank_, toggleOddBank_, toggleRa13h_, crfExit_;
    RankExecutionMode execution_mode_ = RankExecutionMode::IDLE;
    std::array<BGLifecycleState, kBGsPerRank> bg_lifecycle_{};
    std::array<uint32_t, kBGsPerRank> bg_generation_{};
    std::array<BGPendingSlot, kBGsPerRank> bg_pending_{};
    std::array<std::array<uint32_t, kPIMBlocksPerBG>, kBGsPerRank> bg_to_pimblocks_{};
    std::array<std::optional<BGTargetedCompletion>, kBGsPerRank> bg_active_{};
    std::array<std::optional<BGTargetedCompletion>, kBGsPerRank> bg_completion_{};
    std::array<std::optional<BGTargetedIdentity>, kBGsPerRank> bg_last_retired_{};
    uint32_t next_bg_rr_ = 0;
    uint8_t pimblock_busy_mask_ = 0;

  public:
    struct TargetedStatistics
    {
        std::array<uint64_t, kBGsPerRank> per_bg_ready_cycles{};
        std::array<uint64_t, kBGsPerRank> per_bg_executing_cycles{};
        std::array<uint64_t, kBGsPerRank> per_bg_grant_wait_cycles{};
        std::array<uint64_t, kBGsPerRank> per_bg_flush_drain_cycles{};
        std::array<uint64_t, 8> per_pimblock_active_cycles{};
        std::array<uint64_t, 8> per_pimblock_targeted_ops{};
        uint64_t rank_targeted_grants = 0;
        uint64_t rank_command_bus_stall_cycles = 0;
        uint64_t rank_resource_conflict_stall_cycles = 0;
        uint64_t round_robin_skip_count = 0;
        uint64_t rank_mode_drain_cycles = 0;
        uint64_t targeted_completions_retired = 0;
        uint64_t duplicate_completion_rejections = 0;
        uint64_t unknown_completion_rejections = 0;
        uint64_t identity_mismatch_rejections = 0;
    };

  private:
    TargetedStatistics targeted_stats_{};

    void validateTargetedTopology();
    void validateTargetedOperation(const BGTargetedOperation&) const;
    void beginBGError(uint32_t);
    void beginRankFatalError();
    bool validateSharedOwnership();
    BGTargetedCompletionValidation validateAndRetireBGTargetedCompletion(
        uint32_t, const BGTargetedCompletion&, BGTargetedCompletion&);
    friend struct PIMRankM6TestAccess;

  public:
    PIMRank(ostream& simLog, Configuration& configuration);
    ~PIMRank() {}

    void attachRank(Rank* r);
    int getChanId() const;
    void setChanId(int id);
    int getRankId() const;
    void setRankId(int id);
    void update();
    void readHab(BusPacket* packet);
    void writeHab(BusPacket* packet);
    void doPIM(BusPacket* packet);
    void doPIMBlock(BusPacket* packet, PIMCmd curCmd, int pimblock_id);
    void controlPIM(BusPacket* packet);
    void readOpd(int pb, BurstType& bst, PIMOpdType type, BusPacket* packet, int idx, bool is_auto,
                 bool is_mac);
    void writeOpd(int pb, BurstType& bst, PIMOpdType type, BusPacket* packet, int idx, bool is_auto,
                  bool is_mac);
    bool isToggleCond(BusPacket* packet);
    bool submitBGTargetedOperation(const BGTargetedOperation&);
    bool requestTargetedModeExit();
    void flushBG(uint32_t local_bg);
    bool resetBG(uint32_t local_bg);
    void serviceBGTargeted(bool command_bus_busy, bool data_bus_busy);
    bool pollBGTargetedCompletion(uint32_t local_bg, BGTargetedCompletion&);
    BGTargetedOperationState queryBGTargetedOperation(uint32_t local_bg, uint64_t operation_id) const;
    uint8_t targetedPIMBlockBusyMask() const { return pimblock_busy_mask_; }
    uint64_t targetedOutstandingCount() const;
    uint64_t targetedPendingCompletionCount() const;
    uint32_t nextBGRoundRobin() const { return next_bg_rr_; }
    const TargetedStatistics& targetedStatistics() const { return targeted_stats_; }
    RankExecutionMode executionMode() const { return execution_mode_; }
    BGLifecycleState bgLifecycle(uint32_t local_bg) const;
    uint32_t bgGeneration(uint32_t local_bg) const;
    bool bgHasPendingOperation(uint32_t local_bg) const;
    const BGTargetedOperation& bgPendingOperation(uint32_t local_bg) const;
    const std::array<uint32_t, kPIMBlocksPerBG>& physicalPIMBlockPair(uint32_t local_bg) const;

    union crf_t
    {
        uint32_t data[32];
        BurstType bst[4];
        crf_t()
        {
            memset(data, 0, sizeof(uint32_t) * 32);
        }
    } crf;

    unsigned inline getGrfIdx(unsigned idx)
    {
        return idx & 0x7;
    }
    unsigned inline getGrfIdxHigh(unsigned r, unsigned c)
    {
        return ((r & 0x1) << 2 | ((c >> 3) & 0x3));
    }
    unsigned inline isReservedRA(unsigned row)
    {
        return (row & (1 << 13));
    }
    unsigned inline masked2accessibleRA(unsigned row)
    {
        return (row & ((1 << 13) - 1));
    }

    Rank* rank;
    vector<PIMBlock> pimBlocks;
};
}  // namespace DRAMSim
#endif
