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

#ifndef __PIM_BLOCK_HPP__
#define __PIM_BLOCK_HPP__

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

#include "Burst.h"
#include "SystemConfiguration.h"

using namespace std;

namespace DRAMSim
{
class PIMBlock
{
  public:
    struct SIMDExecutionCounters
    {
        uint64_t issued_operations = 0, active_lanes = 0, masked_lanes = 0;
        uint64_t invalid_lane_operand_reads = 0, invalid_lane_arithmetic_operations = 0;
        uint64_t invalid_lane_destination_writes = 0, invalid_lane_partial_results = 0;
    };
    PIMBlock()
    {
        pimPrecision_ = PIMConfiguration::getPIMPrecision();
    }
    PIMBlock(const PIMPrecision& pimPrecision) : pimPrecision_(pimPrecision) {}

    BurstType srf;
    BurstType grfA[8];  // FIXME: hard coding shcha
    BurstType grfB[8];
    BurstType mOut;
    BurstType aOut;

    void add(BurstType& dstBst, BurstType& src0Bst, BurstType& src1Bst);
    void add(BurstType&, const BurstType&, const BurstType&, uint32_t valid_count);
    void mac(BurstType& dstBst, BurstType& src0Bst, BurstType& src1Bst);
    void mac(BurstType&, const BurstType&, const BurstType&, uint32_t valid_count);
    void mul(BurstType& dstBst, BurstType& src0Bst, BurstType& src1Bst);
    void mul(BurstType&, const BurstType&, const BurstType&, uint32_t valid_count);
    void mad(BurstType& dstBst, BurstType& src0Bst, BurstType& src1Bst, BurstType& src2Bst);
    void mad(BurstType&, const BurstType&, const BurstType&, const BurstType&, uint32_t valid_count);
    const SIMDExecutionCounters& simdCounters() const { return simd_counters_; }
    void resetSIMDCounters() { simd_counters_ = {}; }

    std::string print();

  private:
    uint32_t checkedValidCount(uint32_t valid_count);
    PIMPrecision pimPrecision_;
    SIMDExecutionCounters simd_counters_;
};

}  // namespace DRAMSim
#endif
