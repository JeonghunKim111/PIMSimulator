#include "csc/CSCBGAIntegration.h"
#include "csc/CSCDescriptorEngine.h"

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

using namespace csc_descriptor;

namespace {

using ConfigureBGA = void (CSCDescriptorEngine::*)(
    const CSCBGAIntegrationConfig&, CSCBGAProducerCallbacks);
using ConfigureNativeBGA = void (CSCNativeExecution::*)(
    const CSCBGAIntegrationConfig&, CSCBGAProducerCallbacks,
    CSCBGAOutputPortCallbacks);
static_assert(std::is_same<decltype(&CSCDescriptorEngine::configureBGAIntegration),
                           ConfigureBGA>::value,
              "CSC engine BGA configuration contract changed");
static_assert(std::is_same<decltype(&CSCNativeExecution::configureBGAIntegration),
                           ConfigureNativeBGA>::value,
              "CSC native BGA configuration contract changed");
static_assert(std::is_same<CSCBGAProducerCallbacks::Submit::result_type,
                           CSCBGAInputResult>::value,
              "BGA submit result contract changed");
static_assert(std::is_same<decltype(std::declval<CSCBGAPartialBatch>().global_bg_id),
                           uint32_t>::value,
              "global BG provenance type changed");

CSCBGAIntegrationConfig enabledConfig()
{
    CSCBGAIntegrationConfig config;
    config.enabled = true;
    config.accumulator.rows = 32;
    return config;
}

CSCBGAPartialBatch validBatch()
{
    CSCBGAPartialBatch batch;
    batch.global_bg_id = 7;
    batch.descriptor_index = 3;
    batch.chunk_index = 2;
    batch.valid_count = 2;
    batch.descriptor_chunk_ordinal = 11;
    batch.payload.logical_stream_id = kCSCM7ALogicalStream;
    batch.payload.generation = 4;
    batch.payload.sequence = 11;
    batch.payload.partials = {{1, 2.0F}, {9, 3.0F}};
    return batch;
}

}  // namespace

TEST(CSCM7A2ContractTest, BatchSeparatesProvenanceFromExactlyOnceIdentity)
{
    const auto config = enabledConfig();
    auto batch = validBatch();
    EXPECT_NO_THROW(batch.validate(config, 11));

    batch.valid_count = 1;
    EXPECT_THROW(batch.validate(config, 11), std::invalid_argument);
    batch = validBatch();
    batch.payload.sequence = 12;
    EXPECT_THROW(batch.validate(config, 11), std::invalid_argument);
    batch = validBatch();
    batch.descriptor_chunk_ordinal = 12;
    EXPECT_THROW(batch.validate(config, 11), std::invalid_argument);
    batch = validBatch();
    batch.global_bg_id = kCSCGlobalBGs;
    EXPECT_THROW(batch.validate(config, 11), std::out_of_range);
}

TEST(CSCM7A2ContractTest, M7A2KeepsOneLogicalStreamWithoutPIMBlockIdentity)
{
    auto config = enabledConfig();
    EXPECT_NO_THROW(config.validate());
    config.logical_stream_id = 1;
    EXPECT_THROW(config.validate(), std::invalid_argument);
}

TEST(CSCM7A2ContractTest, CallbackPortCompilesAgainstStandalonePrimitive)
{
    const auto config = enabledConfig();
    CSCBankGroupAccumulator bga(config.accumulator, 4);
    CSCBGAProducerCallbacks producer;
    CSCBGAOutputPortCallbacks output;
    producer.submit = [&](const CSCBGAPartialBatch& batch) {
        batch.validate(config, bga.lastAcceptedSequence(0) + 1);
        return bga.offerBatch(batch.payload);
    };
    producer.producer_done = [&](const CSCBGAProducerIdentity& identity) {
        EXPECT_EQ(identity.logical_stream_id, kCSCM7ALogicalStream);
        bga.markProducerDone(identity.logical_stream_id);
    };
    producer.request_final_drain = [&](const CSCBGAProducerIdentity&) {
        return bga.requestFinalDrain();
    };
    output.has_output = [&](uint32_t) { return bga.hasOutput(); };
    output.peek_output =
        [&](uint32_t) -> const CSCBGAOutput& { return bga.peekOutput(); };
    output.accept_output = [&](uint32_t) { bga.acceptOutput(); };
    ASSERT_TRUE(producer.complete());
    ASSERT_TRUE(output.complete());

    auto batch = validBatch();
    batch.descriptor_index = 0;
    batch.chunk_index = 0;
    batch.descriptor_chunk_ordinal = 1;
    batch.payload.sequence = 1;
    EXPECT_EQ(producer.submit(batch), CSCBGAInputResult::ACCEPTED);
    producer.producer_done({batch.global_bg_id,
                             batch.payload.logical_stream_id,
                             batch.payload.generation});
    EXPECT_TRUE(producer.request_final_drain({batch.global_bg_id,
                                                batch.payload.logical_stream_id,
                                                batch.payload.generation}));
}


TEST(CSCM7A2ContractTest, ConfigurationIsPreLaunchAndDisabledNeedsNoCallbacks)
{
    DRAMSim::PIMBlock block(DRAMSim::FP32);
    CSCDescriptorEngine engine(
        0, &block, [](auto*, auto, auto) { return true; },
        [](auto, auto, auto) { return uint64_t{0}; });
    CSCBGAIntegrationConfig enabled = enabledConfig();
    EXPECT_THROW(engine.configureBGAIntegration(enabled, {}),
                 std::invalid_argument);
    CSCBGAIntegrationConfig disabled;
    EXPECT_NO_THROW(engine.configureBGAIntegration(disabled, {}));
    EXPECT_FALSE(engine.bgaIntegrationEnabled());

    std::vector<uint8_t> values, rows;
    std::vector<CSCDescriptor> descriptors;
    std::vector<float> x;
    std::vector<CSCPartial> sink;
    engine.launch({&values, &rows, &descriptors, &x}, &sink);
    EXPECT_THROW(engine.configureBGAIntegration(disabled, {}), std::logic_error);
}
