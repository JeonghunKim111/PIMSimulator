#include "gtest/gtest.h"
#include "AddressMapping.h"
#include "Configuration.h"
#include "PIMRank.h"
#include "tests/csc/CSCTimingModel.h"
#include <sstream>
using namespace DRAMSim;
using namespace csc_descriptor;
TEST(CSCFP32SrfAuditTest,ScalarReadReplicatesAcrossEightMulLanes){CSCTimingModel initialize_fp32_configuration;AddrMapping mapping;Configuration config(mapping);std::ostringstream log;PIMRank rank(log,config);ASSERT_EQ(PIMConfiguration::getPIMPrecision(),FP32);rank.pimBlocks[0].srf.fp32Data_[2]=3.25f;rank.pimBlocks[0].srf.fp32Data_[5]=-2.0f;BurstType m_scalar,a_scalar;rank.readOpd(0,m_scalar,PIMOpdType::SRF_M,nullptr,2,false,false);rank.readOpd(0,a_scalar,PIMOpdType::SRF_A,nullptr,1,false,false);BurstType input,product;for(uint32_t lane=0;lane<8;++lane){input.fp32Data_[lane]=float(lane+1);EXPECT_FLOAT_EQ(m_scalar.fp32Data_[lane],3.25f);EXPECT_FLOAT_EQ(a_scalar.fp32Data_[lane],-2.0f);}rank.pimBlocks[0].mul(product,input,m_scalar);for(uint32_t lane=0;lane<8;++lane)EXPECT_FLOAT_EQ(product.fp32Data_[lane],float(lane+1)*3.25f);}
