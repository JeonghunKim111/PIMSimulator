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

#include "PIMRank.h"

#include <stdexcept>

#include <bitset>
#include <iostream>

#include "AddressMapping.h"
#include "PIMCmd.h"

using namespace std;
using namespace DRAMSim;

PIMRank::PIMRank(ostream& simLog, Configuration& configuration)
    : chanId(-1),
      rankId(-1),
      dramsimLog(simLog),
      pimPC_(0),
      lastJumpIdx_(-1),
      numJumpToBeTaken_(-1),
      lastRepeatIdx_(-1),
      numRepeatToBeDone_(-1),
      crfExit_(false),
      config(configuration),
      pimBlocks(getConfigParam(UINT, "NUM_PIM_BLOCKS"),
                PIMBlock(PIMConfiguration::getPIMPrecision()))
{
    currentClockCycle = 0;
    validateTargetedTopology();
}

void PIMRank::validateTargetedTopology()
{
    if (getConfigParam(UINT, "NUM_BANK_GROUPS") != kBGsPerRank || config.NUM_BANKS != 16 ||
        config.NUM_PIM_BLOCKS != kBGsPerRank * kPIMBlocksPerBG)
        throw std::runtime_error("M6 targeted execution requires 4 BG/16 bank/8 PIMBlock topology");

    std::array<bool, 8> owned{};
    for (uint32_t bg = 0; bg < kBGsPerRank; ++bg)
    {
        bg_lifecycle_[bg] = BGLifecycleState::IDLE;
        bg_generation_[bg] = 0;
        for (uint32_t in_pair = 0; in_pair < kPIMBlocksPerBG; ++in_pair)
        {
            const uint32_t block = bg * kPIMBlocksPerBG + in_pair;
            if (block >= pimBlocks.size() || owned[block] || block * 2 + 1 >= config.NUM_BANKS)
                throw std::runtime_error("M6 BG/PIMBlock mapping corruption");
            owned[block] = true;
            bg_to_pimblocks_[bg][in_pair] = block;
        }
    }
}

void PIMRank::validateTargetedOperation(const BGTargetedOperation& operation) const
{
    const auto& id = operation.identity;
    if (!id.operation_id || id.channel != static_cast<uint32_t>(chanId) ||
        id.rank != static_cast<uint32_t>(rankId) || id.local_bg >= kBGsPerRank)
        throw std::invalid_argument("invalid BG-targeted operation identity");
    if (!id.pimblock_mask || (id.pimblock_mask & ~kBGTargetBothPIMBlocks))
        throw std::invalid_argument("invalid BG-targeted PIMBlock mask");
    if (!operation.valid_count || operation.valid_count > 8)
        throw std::invalid_argument("invalid BG-targeted valid count");
    for (uint32_t bit = 0; bit < kPIMBlocksPerBG; ++bit)
        if ((id.pimblock_mask & (1U << bit)) && !operation.contexts[bit].valid)
            throw std::invalid_argument("selected PIMBlock has no operand context");
}

bool PIMRank::submitBGTargetedOperation(const BGTargetedOperation& operation)
{
    validateTargetedOperation(operation);
    const uint32_t bg = operation.identity.local_bg;
    if (bg_lifecycle_[bg] == BGLifecycleState::FLUSH_REQUESTED ||
        bg_lifecycle_[bg] == BGLifecycleState::DRAINING ||
        bg_lifecycle_[bg] == BGLifecycleState::FLUSHED ||
        bg_lifecycle_[bg] == BGLifecycleState::ERROR_DRAINING ||
        bg_lifecycle_[bg] == BGLifecycleState::ERROR || bg_pending_[bg].occupied)
        return false;
    if (execution_mode_ == RankExecutionMode::IDLE)
        execution_mode_ = RankExecutionMode::CSC_BG_TARGETED;
    if (execution_mode_ != RankExecutionMode::CSC_BG_TARGETED) return false;
    bg_pending_[bg].occupied = true;
    bg_pending_[bg].operation = operation;
    bg_pending_[bg].operation.state = BGTargetedOperationState::WAITING_FOR_GRANT;
    bg_lifecycle_[bg] = BGLifecycleState::RUNNING;
    return true;
}

void PIMRank::serviceBGTargeted(bool command_bus_busy, bool data_bus_busy)
{
    for (uint32_t bg = 0; bg < kBGsPerRank; ++bg)
    {
        if (bg_active_[bg])
        {
            for (uint32_t bit = 0; bit < kPIMBlocksPerBG; ++bit)
                if (bg_active_[bg]->identity.pimblock_mask & (1U << bit))
                {
                    const uint32_t block = bg_to_pimblocks_[bg][bit];
                    targeted_stats_.per_pimblock_active_cycles[block]++;
                }
            targeted_stats_.per_bg_executing_cycles[bg]++;
            if (bg_active_[bg]->completion_cycle <= currentClockCycle)
            {
                if (bg_completion_[bg])
                    throw std::logic_error("duplicate unretired targeted completion");
                const uint8_t mask = bg_active_[bg]->identity.pimblock_mask;
                for (uint32_t bit = 0; bit < kPIMBlocksPerBG; ++bit)
                    if (mask & (1U << bit))
                        pimblock_busy_mask_ &= ~(1U << bg_to_pimblocks_[bg][bit]);
                bg_completion_[bg] = bg_active_[bg];
                bg_active_[bg].reset();
            }
        }
        if (bg_pending_[bg].occupied)
        {
            targeted_stats_.per_bg_ready_cycles[bg]++;
            targeted_stats_.per_bg_grant_wait_cycles[bg]++;
        }
    }

    bool any_pending = false;
    for (const auto& slot : bg_pending_) any_pending = any_pending || slot.occupied;
    if (!any_pending || execution_mode_ != RankExecutionMode::CSC_BG_TARGETED) return;
    if (command_bus_busy || data_bus_busy)
    {
        targeted_stats_.rank_command_bus_stall_cycles++;
        return;
    }

    for (uint32_t offset = 0; offset < kBGsPerRank; ++offset)
    {
        const uint32_t bg = (next_bg_rr_ + offset) % kBGsPerRank;
        if (!bg_pending_[bg].occupied)
        {
            targeted_stats_.round_robin_skip_count++;
            continue;
        }
        if (bg_active_[bg] || bg_completion_[bg] ||
            bg_lifecycle_[bg] != BGLifecycleState::RUNNING)
        {
            targeted_stats_.round_robin_skip_count++;
            continue;
        }
        auto& operation = bg_pending_[bg].operation;
        uint8_t physical_mask = 0;
        for (uint32_t bit = 0; bit < kPIMBlocksPerBG; ++bit)
            if (operation.identity.pimblock_mask & (1U << bit))
                physical_mask |= 1U << bg_to_pimblocks_[bg][bit];
        if (physical_mask & pimblock_busy_mask_)
        {
            targeted_stats_.rank_resource_conflict_stall_cycles++;
            targeted_stats_.round_robin_skip_count++;
            continue;
        }

        BGTargetedCompletion completion;
        completion.identity = operation.identity;
        completion.grant_cycle = currentClockCycle;
        completion.completion_cycle = currentClockCycle + 1;
        operation.state = BGTargetedOperationState::EXECUTING;
        for (uint32_t bit = 0; bit < kPIMBlocksPerBG; ++bit)
            if (operation.identity.pimblock_mask & (1U << bit))
            {
                const uint32_t block = bg_to_pimblocks_[bg][bit];
                pimBlocks[block].mul(completion.results[bit], operation.contexts[bit].lhs,
                                     operation.contexts[bit].rhs, operation.valid_count);
                targeted_stats_.per_pimblock_targeted_ops[block]++;
            }
        pimblock_busy_mask_ |= physical_mask;
        bg_active_[bg] = completion;
        bg_pending_[bg].occupied = false;
        targeted_stats_.rank_targeted_grants++;
        next_bg_rr_ = (bg + 1) % kBGsPerRank;
        return;
    }
}

bool PIMRank::pollBGTargetedCompletion(uint32_t local_bg, BGTargetedCompletion& completion)
{
    if (local_bg >= kBGsPerRank) throw std::out_of_range("local BG");
    if (!bg_completion_[local_bg]) return false;
    completion = *bg_completion_[local_bg];
    bg_completion_[local_bg].reset();
    return true;
}

BGTargetedOperationState PIMRank::queryBGTargetedOperation(
    uint32_t local_bg, uint64_t operation_id) const
{
    if (local_bg >= kBGsPerRank || !operation_id)
        throw std::invalid_argument("targeted operation query identity");
    if (bg_pending_[local_bg].occupied &&
        bg_pending_[local_bg].operation.identity.operation_id == operation_id)
        return BGTargetedOperationState::WAITING_FOR_GRANT;
    if (bg_active_[local_bg] && bg_active_[local_bg]->identity.operation_id == operation_id)
        return BGTargetedOperationState::EXECUTING;
    if (bg_completion_[local_bg] && bg_completion_[local_bg]->identity.operation_id == operation_id)
        return BGTargetedOperationState::COMPLETED;
    throw std::logic_error("unknown targeted operation identity");
}

BGLifecycleState PIMRank::bgLifecycle(uint32_t local_bg) const
{
    if (local_bg >= kBGsPerRank) throw std::out_of_range("local BG");
    return bg_lifecycle_[local_bg];
}

uint32_t PIMRank::bgGeneration(uint32_t local_bg) const
{
    if (local_bg >= kBGsPerRank) throw std::out_of_range("local BG");
    return bg_generation_[local_bg];
}

bool PIMRank::bgHasPendingOperation(uint32_t local_bg) const
{
    if (local_bg >= kBGsPerRank) throw std::out_of_range("local BG");
    return bg_pending_[local_bg].occupied;
}

const BGTargetedOperation& PIMRank::bgPendingOperation(uint32_t local_bg) const
{
    if (!bgHasPendingOperation(local_bg)) throw std::logic_error("BG has no pending operation");
    return bg_pending_[local_bg].operation;
}

const std::array<uint32_t, kPIMBlocksPerBG>& PIMRank::physicalPIMBlockPair(
    uint32_t local_bg) const
{
    if (local_bg >= kBGsPerRank) throw std::out_of_range("local BG");
    return bg_to_pimblocks_[local_bg];
}

void PIMRank::attachRank(Rank* r)
{
    this->rank = r;
}

void PIMRank::setChanId(int id)
{
    this->chanId = id;
}

void PIMRank::setRankId(int id)
{
    this->rankId = id;
}

int PIMRank::getChanId() const
{
    return this->chanId;
}

int PIMRank::getRankId() const
{
    return this->rankId;
}

void PIMRank::update() {}

void PIMRank::controlPIM(BusPacket* packet)
{
    uint8_t grf_a_zeroize = packet->data->u8Data_[20];
    if (grf_a_zeroize)
    {
        if (DEBUG_CMD_TRACE)
        {
            PRINTC(RED, OUTLOG_CH_RA("GRF_A_ZEROIZE"));
        }
        BurstType burst_zero;
        for (int pb = 0; pb < config.NUM_PIM_BLOCKS; pb++)
        {
            for (int i = 0; i < 8; i++) pimBlocks[pb].grfA[i] = burst_zero;
        }
    }
    uint8_t grf_b_zeroize = packet->data->u8Data_[21];
    if (grf_b_zeroize)
    {
        if (DEBUG_CMD_TRACE)
        {
            PRINTC(RED, OUTLOG_CH_RA("GRF_B_ZEROIZE"));
        }
        BurstType burst_zero;
        for (int pb = 0; pb < config.NUM_PIM_BLOCKS; pb++)
        {
            for (int i = 0; i < 8; i++) pimBlocks[pb].grfB[i] = burst_zero;
        }
    }
    pimOpMode_ = packet->data->u8Data_[0] & 1;
    toggleEvenBank_ = !(packet->data->u8Data_[16] & 1);
    toggleOddBank_ = !(packet->data->u8Data_[16] & 2);
    toggleRa13h_ = (packet->data->u8Data_[16] & 4);

    if (pimOpMode_)
    {
        rank->mode_ = dramMode::HAB_PIM;
        pimPC_ = 0;
        lastJumpIdx_ = numJumpToBeTaken_ = lastRepeatIdx_ = numRepeatToBeDone_ = -1;
        crfExit_ = false;
        PRINTC(RED, OUTLOG_CH_RA("HAB_PIM"));
    }
    else
    {
        rank->mode_ = dramMode::HAB;
        PRINTC(RED, OUTLOG_CH_RA("HAB mode"));
    }
}

bool PIMRank::isToggleCond(BusPacket* packet)
{
    if (pimOpMode_ && !crfExit_)
    {
        if (toggleRa13h_)
        {
            if (toggleEvenBank_ && ((packet->bank & 1) == 0))
                return true;
            else if (toggleOddBank_ && ((packet->bank & 1) == 1))
                return true;
            return false;
        }
        else if (!toggleRa13h_ && !isReservedRA(packet->row))
        {
            if (toggleEvenBank_ && ((packet->bank & 1) == 0))
                return true;
            else if (toggleOddBank_ && ((packet->bank & 1) == 1))
                return true;
            return false;
        }
        return false;
    }
    else
    {
        return false;
    }
}

void PIMRank::readHab(BusPacket* packet)
{
    if (isReservedRA(packet->row))  // ignored
    {
        PRINTC(GRAY, OUTLOG_ALL("READ"));
    }
    else
    {
        PRINTC(GRAY, OUTLOG_ALL("BANK_TO_PIM"));
#ifndef NO_STORAGE
        int grf_id = getGrfIdx(packet->column);
        for (int pb = 0; pb < config.NUM_PIM_BLOCKS; pb++)
        {
            rank->banks[pb * 2 + packet->bank].read(packet);
            pimBlocks[pb].grfB[grf_id] = *(packet->data);
        }
#endif
    }
}

void PIMRank::writeHab(BusPacket* packet)
{
    if (packet->row == config.PIM_REG_RA)  // WRIO to PIM Broadcasting
    {
        if (packet->column == 0x00)
            controlPIM(packet);
        if ((0x08 <= packet->column && packet->column <= 0x0f) ||
            (0x18 <= packet->column && packet->column <= 0x1f))
        {
            if (DEBUG_CMD_TRACE)
            {
                if (packet->column - 8 < 8)
                    PRINTC(GREEN, OUTLOG_B_GRF_A("BWRITE_GRF_A"));
                else
                    PRINTC(GREEN, OUTLOG_B_GRF_B("BWRITE_GRF_B"));
            }
#ifndef NO_STORAGE
            for (int pb = 0; pb < config.NUM_PIM_BLOCKS; pb++)
            {
                if (packet->column - 8 < 8)
                    pimBlocks[pb].grfA[packet->column - 0x8] = *(packet->data);
                else
                    pimBlocks[pb].grfB[packet->column - 0x18] = *(packet->data);
            }
#endif
        }
        else if (0x04 <= packet->column && packet->column <= 0x07)
        {
            if (DEBUG_CMD_TRACE)
                PRINTC(GREEN, OUTLOG_B_CRF("BWRITE_CRF"));
            crf.bst[packet->column - 0x04] = *(packet->data);
        }
        else if (packet->column == 0x1)
        {
            if (DEBUG_CMD_TRACE)
                PRINTC(GREEN, OUTLOG_CH_RA("BWRITE_SRF"));
            for (int pb = 0; pb < config.NUM_PIM_BLOCKS; pb++) pimBlocks[pb].srf = *(packet->data);
        }
    }
    else if (isReservedRA(packet->row))
    {
        PRINTC(GRAY, OUTLOG_ALL("WRITE"));
    }
    else  // PIM (only GRF) to Bank Move
    {
        PRINTC(GREEN, OUTLOG_ALL("PIM_TO_BANK"));

#ifndef NO_STORAGE
        int grf_id = getGrfIdx(packet->column);
        for (int pb = 0; pb < config.NUM_PIM_BLOCKS; pb++)
        {
            if (packet->bank == 0)
            {
                *(packet->data) = pimBlocks[pb].grfA[grf_id];
                rank->banks[pb * 2].write(packet);  // basically read from bank;
            }
            else if (packet->bank == 1)
            {
                *(packet->data) = pimBlocks[pb].grfB[grf_id];
                rank->banks[pb * 2 + 1].write(packet);  // basically read from bank.
            }
        }
#endif
    }
}

void PIMRank::readOpd(int pb, BurstType& bst, PIMOpdType type, BusPacket* packet, int idx,
                      bool is_auto, bool is_mac)
{
    idx = getGrfIdx(idx);

    switch (type)
    {
        case PIMOpdType::A_OUT:
            bst = pimBlocks[pb].aOut;
            return;
        case PIMOpdType::M_OUT:
            bst = pimBlocks[pb].mOut;
            return;
        case PIMOpdType::EVEN_BANK:
            if (packet->bank % 2 != 0)
                PRINT("Warning, CRF bank coding and bank id from packet are inconsistent");
            rank->banks[pb * 2].read(packet);  // basically read from bank.
            bst = *(packet->data);
            return;
        case PIMOpdType::ODD_BANK:
            if (packet->bank % 2 == 0)
                PRINT("Warning, CRF bank coding and bank id from packet are inconsistent");
            rank->banks[pb * 2 + 1].read(packet);  // basically read from bank.
            bst = *(packet->data);
            return;
        case PIMOpdType::GRF_A:
            bst = pimBlocks[pb].grfA[(is_auto) ? getGrfIdx(packet->column) : idx];
            return;
        case PIMOpdType::GRF_B:
            if (is_auto)
                bst = pimBlocks[pb].grfB[(is_mac) ? getGrfIdxHigh(packet->row, packet->column)
                                                  : getGrfIdx(packet->column)];
            else
                bst = pimBlocks[pb].grfB[idx];
            return;
        case PIMOpdType::SRF_M:
            if (PIMConfiguration::getPIMPrecision() == FP32)
                bst.set(pimBlocks[pb].srf.fp32Data_[idx & 0x3]);
            else
                bst.set(pimBlocks[pb].srf.fp16Data_[idx]);
            return;
        case PIMOpdType::SRF_A:
            if (PIMConfiguration::getPIMPrecision() == FP32)
                bst.set(pimBlocks[pb].srf.fp32Data_[(idx & 0x3) + 4]);
            else
                bst.set(pimBlocks[pb].srf.fp16Data_[idx + 8]);
            return;
    }
}

void PIMRank::writeOpd(int pb, BurstType& bst, PIMOpdType type, BusPacket* packet, int idx,
                       bool is_auto, bool is_mac)
{
    idx = getGrfIdx(idx);

    switch (type)
    {
        case PIMOpdType::A_OUT:
            pimBlocks[pb].aOut = bst;
            return;
        case PIMOpdType::M_OUT:
            pimBlocks[pb].mOut = bst;
            return;
        case PIMOpdType::EVEN_BANK:
            if (packet->bank % 2 != 0)
            {
                PRINT("CRF bank coding and bank id from packet are inconsistent");
            }
            *(packet->data) = bst;
            rank->banks[pb * 2].write(packet);  // basically read from bank.
            return;
        case PIMOpdType::ODD_BANK:
            if (packet->bank % 2 == 0)
            {
                PRINT("CRF bank coding and bank id from packet are inconsistent");
                exit(-1);
            }
            *(packet->data) = bst;
            rank->banks[pb * 2 + 1].write(packet);  // basically read from bank.
            return;
        case PIMOpdType::GRF_A:
            pimBlocks[pb].grfA[(is_auto) ? getGrfIdx(packet->column) : idx] = bst;
            return;
        case PIMOpdType::GRF_B:
            if (is_auto)
                pimBlocks[pb].grfB[(is_mac) ? getGrfIdxHigh(packet->row, packet->column)
                                            : getGrfIdx(packet->column)] = bst;
            else
                pimBlocks[pb].grfB[idx] = bst;
            return;
        case PIMOpdType::SRF_M:
            pimBlocks[pb].srf = bst;
            return;
        case PIMOpdType::SRF_A:
            pimBlocks[pb].srf = bst;
            return;
    }
}

void PIMRank::doPIM(BusPacket* packet)
{
    PIMCmd cCmd;
    packet->row = masked2accessibleRA(packet->row);
    do
    {
        cCmd.fromInt(crf.data[pimPC_]);
        if (DEBUG_CMD_TRACE)
        {
            PRINTC(CYAN, string((packet->busPacketType == READ) ? "READ ch" : "WRITE ch")
                             << getChanId() << " ra" << getRankId() << " bg"
                             << config.addrMapping.bankgroupId(packet->bank) << " b" << packet->bank
                             << " r" << packet->row << " c" << packet->column << "|| [" << pimPC_
                             << "] " << cCmd.toStr() << " @ " << currentClockCycle);
        }

        if (cCmd.type_ == PIMCmdType::EXIT)
        {
            crfExit_ = true;
            break;
        }
        else if (cCmd.type_ == PIMCmdType::JUMP)
        {
            if (lastJumpIdx_ != pimPC_)
            {
                if (cCmd.loopCounter_ > 0)
                {
                    lastJumpIdx_ = pimPC_;
                    numJumpToBeTaken_ = cCmd.loopCounter_;
                }
            }
            if (numJumpToBeTaken_ > 0)
            {
                pimPC_ -= cCmd.loopOffset_;
                numJumpToBeTaken_--;
            }
        }
        else
        {
            if (cCmd.type_ == PIMCmdType::FILL || cCmd.isAuto_)
            {
                if (lastRepeatIdx_ != pimPC_)
                {
                    lastRepeatIdx_ = pimPC_;
                    numRepeatToBeDone_ = 8 - 1;
                }

                if (numRepeatToBeDone_ > 0)
                {
                    pimPC_ -= 1;
                    numRepeatToBeDone_--;
                }
                else
                    lastRepeatIdx_ = -1;
            }
            else if (cCmd.type_ == PIMCmdType::NOP)
            {
                if (lastRepeatIdx_ != pimPC_)
                {
                    lastRepeatIdx_ = pimPC_;
                    numRepeatToBeDone_ = cCmd.loopCounter_;
                }

                if (numRepeatToBeDone_ > 0)
                {
                    pimPC_ -= 1;
                    numRepeatToBeDone_--;
                }
                else
                    lastRepeatIdx_ = -1;
            }

            for (int pimblock_id = 0; pimblock_id < config.NUM_PIM_BLOCKS; pimblock_id++)
            {
                doPIMBlock(packet, cCmd, pimblock_id);

                if (DEBUG_PIM_BLOCK && pimblock_id == 0)
                {
                    PRINT("[BANK_R]" << packet->data->fp16ToStr());
                    PRINT("[CMD]" << bitset<32>(cCmd.toInt()) << "(" << cCmd.toStr() << ")");
                    PRINT(pimBlocks[pimblock_id].print());
                    PRINT("----------");
                }
            }
        }
        pimPC_++;
        // EXIT check
        PIMCmd next_cmd;
        next_cmd.fromInt(crf.data[pimPC_]);
        if (next_cmd.type_ == PIMCmdType::EXIT)
            crfExit_ = true;
    } while (cCmd.type_ == PIMCmdType::JUMP);
}

void PIMRank::doPIMBlock(BusPacket* packet, PIMCmd cCmd, int pimblock_id)
{
    if (cCmd.type_ == PIMCmdType::FILL || cCmd.type_ == PIMCmdType::MOV)
    {
        BurstType bst;
        bool is_auto = (cCmd.type_ == PIMCmdType::FILL) ? true : false;

        readOpd(pimblock_id, bst, cCmd.src0_, packet, cCmd.src0Idx_, is_auto, false);
        if (cCmd.isRelu_)
        {
            for (int i = 0; i < 16; i++)
                bst.u16Data_[i] = (bst.u16Data_[i] & (1 << 15)) ? 0 : bst.u16Data_[i];
        }
        writeOpd(pimblock_id, bst, cCmd.dst_, packet, cCmd.dstIdx_, is_auto, false);
    }
    else if (cCmd.type_ == PIMCmdType::ADD || cCmd.type_ == PIMCmdType::MUL)
    {
        BurstType dstBst;
        BurstType src0Bst;
        BurstType src1Bst;
        readOpd(pimblock_id, src0Bst, cCmd.src0_, packet, cCmd.src0Idx_, cCmd.isAuto_, false);
        readOpd(pimblock_id, src1Bst, cCmd.src1_, packet, cCmd.src1Idx_, cCmd.isAuto_, false);

        if (cCmd.type_ == PIMCmdType::ADD)
            // dstBst = src0Bst + src1Bst;
            pimBlocks[pimblock_id].add(dstBst, src0Bst, src1Bst);
        else if (cCmd.type_ == PIMCmdType::MUL)
            // dstBst = src0Bst * src1Bst;
            pimBlocks[pimblock_id].mul(dstBst, src0Bst, src1Bst);

        writeOpd(pimblock_id, dstBst, cCmd.dst_, packet, cCmd.dstIdx_, cCmd.isAuto_, false);
    }
    else if (cCmd.type_ == PIMCmdType::MAC || cCmd.type_ == PIMCmdType::MAD)
    {
        BurstType dstBst;
        BurstType src0Bst;
        BurstType src1Bst;
        bool is_mac = (cCmd.type_ == PIMCmdType::MAC) ? true : false;

        readOpd(pimblock_id, src0Bst, cCmd.src0_, packet, cCmd.src0Idx_, cCmd.isAuto_, is_mac);
        readOpd(pimblock_id, src1Bst, cCmd.src1_, packet, cCmd.src1Idx_, cCmd.isAuto_, is_mac);
        if (is_mac)
        {
            readOpd(pimblock_id, dstBst, cCmd.dst_, packet, cCmd.dstIdx_, cCmd.isAuto_, is_mac);
            // dstBst = src0Bst * src1Bst + dstBst;
            pimBlocks[pimblock_id].mac(dstBst, src0Bst, src1Bst);
        }
        else
        {
            BurstType src2Bst;
            readOpd(pimblock_id, src2Bst, cCmd.src2_, packet, cCmd.src1Idx_, cCmd.isAuto_, is_mac);
            // dstBst = src0Bst * src1Bst + src2Bst;
            pimBlocks[pimblock_id].mad(dstBst, src0Bst, src1Bst, src2Bst);
        }

        writeOpd(pimblock_id, dstBst, cCmd.dst_, packet, cCmd.dstIdx_, cCmd.isAuto_, is_mac);
    }
    else if (cCmd.type_ == PIMCmdType::NOP && packet->busPacketType == WRITE)
    {
        int grf_id = getGrfIdx(packet->column);
        if (packet->bank == 0)
        {
            *(packet->data) = pimBlocks[pimblock_id].grfA[grf_id];
            rank->banks[pimblock_id * 2].write(packet);  // basically read from bank;
        }
        else if (packet->bank == 1)
        {
            *(packet->data) = pimBlocks[pimblock_id].grfB[grf_id];
            rank->banks[pimblock_id * 2 + 1].write(packet);  // basically read from bank.
        }
    }
}
