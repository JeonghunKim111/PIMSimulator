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

#include "PIMBlock.h"

#include <sstream>
#include <stdexcept>
#include <string>

#include "PrintMacros.h"
#include "SystemConfiguration.h"
#include "half.h"

using namespace DRAMSim;

uint32_t PIMBlock::checkedValidCount(uint32_t valid_count)
{
    const uint32_t lanes = pimPrecision_ == FP16 ? 16 : 8;
    if (valid_count > lanes) throw std::invalid_argument("valid_count exceeds SIMD lane count");
    simd_counters_.issued_operations++; simd_counters_.active_lanes += valid_count;
    simd_counters_.masked_lanes += lanes - valid_count; return valid_count;
}

void PIMBlock::add(BurstType& dstBst, BurstType& src0Bst, BurstType& src1Bst)
{
    add(dstBst,src0Bst,src1Bst,pimPrecision_==FP16?16:8);
}
void PIMBlock::add(BurstType& dstBst,const BurstType& src0Bst,const BurstType& src1Bst,uint32_t valid_count)
{
    valid_count=checkedValidCount(valid_count);
    if (pimPrecision_ == FP16)
    {
        for (uint32_t i = 0; i < valid_count; i++)
        {
            dstBst.fp16Data_[i] = src0Bst.fp16Data_[i] + src1Bst.fp16Data_[i];
        }
    }
    else if (pimPrecision_ == FP32)
    {
        for (uint32_t i = 0; i < valid_count; i++)
        {
            dstBst.fp32Data_[i] = src0Bst.fp32Data_[i] + src1Bst.fp32Data_[i];
        }
    }
    else
        dstBst = src0Bst + src1Bst;
}

void PIMBlock::mul(BurstType& dstBst, BurstType& src0Bst, BurstType& src1Bst)
{
    mul(dstBst,src0Bst,src1Bst,pimPrecision_==FP16?16:8);
}
void PIMBlock::mul(BurstType& dstBst,const BurstType& src0Bst,const BurstType& src1Bst,uint32_t valid_count)
{
    valid_count=checkedValidCount(valid_count);
    if (pimPrecision_ == FP16)
    {
        for (uint32_t i = 0; i < valid_count; i++)
        {
            dstBst.fp16Data_[i] = src0Bst.fp16Data_[i] * src1Bst.fp16Data_[i];
        }
    }
    else if (pimPrecision_ == FP32)
    {
        for (uint32_t i = 0; i < valid_count; i++)
        {
            dstBst.fp32Data_[i] = src0Bst.fp32Data_[i] * src1Bst.fp32Data_[i];
        }
    }
    else
        dstBst = src0Bst * src1Bst;
}

void PIMBlock::mac(BurstType& dstBst, BurstType& src0Bst, BurstType& src1Bst)
{
    mac(dstBst,src0Bst,src1Bst,pimPrecision_==FP16?16:8);
}
void PIMBlock::mac(BurstType& dstBst,const BurstType& src0Bst,const BurstType& src1Bst,uint32_t valid_count)
{
    valid_count=checkedValidCount(valid_count);
    if (pimPrecision_ == FP16)
    {
        for (uint32_t i = 0; i < valid_count; i++)
        {
            dstBst.fp16Data_[i] = src0Bst.fp16Data_[i] * src1Bst.fp16Data_[i] + dstBst.fp16Data_[i];
        }

        DEBUG("MAC " << src0Bst.hexToStr2() << "*+" << src1Bst.hexToStr2() << ""
                     << dstBst.hexToStr2());
    }
    else if (pimPrecision_ == FP32)
    {
        for (uint32_t i = 0; i < valid_count; i++)
        {
            dstBst.fp32Data_[i] = src0Bst.fp32Data_[i] * src1Bst.fp32Data_[i] + dstBst.fp32Data_[i];
        }
    }
    else
        dstBst = src0Bst * src1Bst + dstBst;
}

void PIMBlock::mad(BurstType& dstBst, BurstType& src0Bst, BurstType& src1Bst, BurstType& src2Bst)
{
    mad(dstBst,src0Bst,src1Bst,src2Bst,pimPrecision_==FP16?16:8);
}
void PIMBlock::mad(BurstType& dstBst,const BurstType& src0Bst,const BurstType& src1Bst,const BurstType& src2Bst,uint32_t valid_count)
{
    valid_count=checkedValidCount(valid_count);
    if (pimPrecision_ == FP16)
    {
        for (uint32_t i = 0; i < valid_count; i++)
        {
            dstBst.fp16Data_[i] =
                src0Bst.fp16Data_[i] * src1Bst.fp16Data_[i] + src2Bst.fp16Data_[i];
        }
    }
    else if (pimPrecision_ == FP32)
    {
        for (uint32_t i = 0; i < valid_count; i++)
        {
            dstBst.fp32Data_[i] =
                src0Bst.fp32Data_[i] * src1Bst.fp32Data_[i] + src2Bst.fp32Data_[i];
        }
    }
    else
        dstBst = src0Bst * src1Bst + src2Bst;
}

std::string PIMBlock::print()
{
    stringstream ss;
    ss << "[SRF]" << srf.binToStr() << endl;
    for (int i = 0; i < 8; i++) ss << "[GRF_A][" << i << "]" << grfA[i].fp16ToStr() << endl;
    for (int i = 0; i < 8; i++) ss << "[GRF_B][" << i << "]" << grfB[i].fp16ToStr() << endl;
    ss << "[M_OUT]" << mOut.binToStr() << endl;
    ss << "[A_OUT]" << aOut.binToStr() << endl;

    return ss.str();
}
