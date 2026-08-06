#include "csc/CSCFp16PartialResultPath.h"

#include <gtest/gtest.h>

using namespace csc_descriptor;

TEST(CSCFp16TransportRecordTest, ExplicitLittleEndianRoundTripAndValidation)
{
    const CSCFp16TransportRecord record{0x78563412U, 0xbc9aU, 0};
    const auto bytes=serializeCSCFp16TransportRecord(record);
    const std::array<uint8_t,8> expected={0x12,0x34,0x56,0x78,0x9a,0xbc,0,0};
    EXPECT_EQ(bytes,expected);
    const auto decoded=deserializeCSCFp16TransportRecord(bytes.data(),bytes.size());
    EXPECT_EQ(decoded.row_idx,record.row_idx); EXPECT_EQ(decoded.value_bits,record.value_bits);
    auto invalid=bytes; invalid[6]=1;
    EXPECT_THROW(deserializeCSCFp16TransportRecord(invalid.data(),invalid.size()),std::invalid_argument);
    EXPECT_THROW(deserializeCSCFp16TransportRecord(bytes.data(),7),std::invalid_argument);
    EXPECT_THROW(serializeCSCFp16TransportRecord({1,2,3}),std::invalid_argument);
}

TEST(CSCFp16TransportRecordTest, SpecialFp16BitsRemainBitExact)
{
    const std::array<uint16_t,9> patterns={0x0000,0x8000,0x0001,0x3c00,0xbc00,0x7bff,0x7c00,0xfc00,0x7e01};
    for(auto bits:patterns){auto bytes=serializeCSCFp16TransportRecord({7,bits,0});auto record=deserializeCSCFp16TransportRecord(bytes.data(),8);EXPECT_EQ(record.value_bits,bits);}
}

TEST(CSCFp16PartialResultPathTest, PacksFullAndTailBurstsAndReducesInBgOrder)
{
    CSCFp16TransportConfig config; config.global_bg_count=4; config.channel_count=1;
    CSCFp16PartialResultPath path(config,8);
    std::vector<bool> done(4,false);
    uint64_t cycle=1;
    for(uint32_t i=0;i<5;++i) ASSERT_TRUE(path.accept({i%2,cscFp16ToBits(cscFp16FromFloat(1.0f)),0,CSCFp16BGAOutputReason::CAPACITY_EVICTION,i+1,1}));
    ASSERT_TRUE(path.accept({0,cscFp16ToBits(cscFp16FromFloat(2.0f)),1,CSCFp16BGAOutputReason::FINAL_DRAIN,1,1}));
    done={true,true,true,true};
    for(;cycle<1000&&!path.done();++cycle)path.step(cycle,done);
    ASSERT_TRUE(path.done())<<path.error();
    const auto& count=path.counters();
    EXPECT_EQ(count.outputs_accepted,6U); EXPECT_EQ(count.full_bursts,1U); EXPECT_EQ(count.tail_bursts,2U);
    EXPECT_EQ(count.write_bytes,96U); EXPECT_EQ(count.read_bytes,96U); EXPECT_EQ(count.records_reduced,6U);
    EXPECT_EQ(count.fp16_host_adds,6U); EXPECT_EQ(count.cross_bg_same_row_adds,1U);
    const auto& y=path.finalYBits();
    EXPECT_EQ(y[0],cscFp16ToBits(cscFp16FromFloat(5.0f)));
    EXPECT_EQ(y[1],cscFp16ToBits(cscFp16FromFloat(2.0f)));
    for(const auto& burst:path.residentTrace()) for(uint32_t i=burst.valid_record_count*8;i<32;++i) EXPECT_EQ(burst.bytes[i],0);
}

TEST(CSCFp16PartialResultPathTest, WriteAndReadRejectionRetryExactlyOnce)
{
    CSCFp16TransportConfig config;
    config.global_bg_count=4; config.channel_count=1;
    config.write_reject_attempts=3; config.read_reject_attempts=2;
    CSCFp16PartialResultPath path(config,2);
    ASSERT_TRUE(path.accept({0,cscFp16ToBits(cscFp16FromFloat(1.0f)),0,
                             CSCFp16BGAOutputReason::FINAL_DRAIN,1,1}));
    std::vector<bool> done(4,true);
    for(uint64_t cycle=1;cycle<1000&&!path.done();++cycle)path.step(cycle,done);
    ASSERT_TRUE(path.done())<<path.error();
    const auto& count=path.counters();
    EXPECT_EQ(count.write_retries,3U); EXPECT_EQ(count.read_retries,2U);
    EXPECT_EQ(count.writes_accepted,1U); EXPECT_EQ(count.write_completions,1U);
    EXPECT_EQ(count.reads_accepted,1U); EXPECT_EQ(count.read_completions,1U);
    EXPECT_EQ(count.records_reduced,1U);
}
