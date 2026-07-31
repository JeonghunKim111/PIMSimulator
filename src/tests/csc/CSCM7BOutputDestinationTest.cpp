#include "csc/CSCBGAIntegration.h"

#include <gtest/gtest.h>

#include <optional>
#include <stdexcept>
#include <type_traits>

using namespace csc_descriptor;

namespace {

class OneSlotDestination final : public CSCBGAOutputDestination {
  public:
    bool reserve(const CSCBGAOutputPortValue& value) override
    {
        if (blocked || reserved_) return false;
        reserved_ = value;
        return true;
    }

    void commitReserved() noexcept override
    {
        committed = reserved_;
        reserved_.reset();
        ++commit_count;
    }

    void cancelReserved() noexcept override
    {
        reserved_.reset();
        ++cancel_count;
    }

    bool blocked = false;
    uint32_t commit_count = 0;
    uint32_t cancel_count = 0;
    std::optional<CSCBGAOutputPortValue> committed;

  private:
    std::optional<CSCBGAOutputPortValue> reserved_;
};

CSCBGAOutput sampleOutput()
{
    CSCBGAOutput output;
    output.row_idx = 7;
    output.value = 3.5F;
    output.generation = 2;
    output.output_sequence = 1;
    output.contribution_count = 4;
    output.reason = CSCBGAOutputReason::FINAL_DRAIN;
    return output;
}

}  // namespace

static_assert(noexcept(std::declval<CSCBGAOutputDestination&>().commitReserved()),
              "destination commit must be no-throw");
static_assert(noexcept(std::declval<CSCBGAOutputDestination&>().cancelReserved()),
              "destination cancel must be no-throw");

TEST(CSCM7BOutputDestinationTest, ModesKeepProductionSeparateFromValidation)
{
    EXPECT_NE(CSCBGAOutputConsumerMode::EXTERNAL,
              CSCBGAOutputConsumerMode::VALIDATION_ROUND_ROBIN);
    EXPECT_NE(CSCBGAOutputConsumerMode::EXTERNAL,
              CSCBGAOutputConsumerMode::PARTIAL_RESULT_WRITEBACK);
    EXPECT_NE(CSCBGAOutputConsumerMode::VALIDATION_ROUND_ROBIN,
              CSCBGAOutputConsumerMode::PARTIAL_RESULT_WRITEBACK);

    CSCBGAIntegrationConfig config;
    config.enabled = true;
    config.accumulator.rows = 8;
    config.output_consumer_mode =
        CSCBGAOutputConsumerMode::PARTIAL_RESULT_WRITEBACK;
    EXPECT_NO_THROW(config.validate());
}

TEST(CSCM7BOutputDestinationTest, BackpressureLeavesSourceOutputUntouched)
{
    const auto output = sampleOutput();
    bool available = true;
    uint32_t accepts = 0;
    CSCBGAOutputPortCallbacks source;
    source.has_output = [&](uint32_t) { return available; };
    source.peek_output = [&](uint32_t) -> const CSCBGAOutput& { return output; };
    source.accept_output = [&](uint32_t) {
        ++accepts;
        available = false;
    };
    OneSlotDestination destination;
    destination.blocked = true;

    EXPECT_EQ(transferCSCBGAOutput(3, source, destination),
              CSCBGAOutputTransferResult::DESTINATION_BACKPRESSURE);
    EXPECT_TRUE(available);
    EXPECT_EQ(accepts, 0U);
    EXPECT_EQ(destination.commit_count, 0U);
    EXPECT_EQ(destination.cancel_count, 0U);
}

TEST(CSCM7BOutputDestinationTest, AcceptFailureCancelsReservation)
{
    const auto output = sampleOutput();
    CSCBGAOutputPortCallbacks source;
    source.has_output = [](uint32_t) { return true; };
    source.peek_output = [&](uint32_t) -> const CSCBGAOutput& { return output; };
    source.accept_output = [](uint32_t) {
        throw std::runtime_error("injected accept failure");
    };
    OneSlotDestination destination;

    EXPECT_THROW(transferCSCBGAOutput(3, source, destination),
                 std::runtime_error);
    EXPECT_EQ(destination.commit_count, 0U);
    EXPECT_EQ(destination.cancel_count, 1U);
    EXPECT_FALSE(destination.committed.has_value());
}

TEST(CSCM7BOutputDestinationTest, SuccessfulAcceptCommitsReservedOwnership)
{
    const auto output = sampleOutput();
    bool available = true;
    CSCBGAOutputPortCallbacks source;
    source.has_output = [&](uint32_t) { return available; };
    source.peek_output = [&](uint32_t) -> const CSCBGAOutput& { return output; };
    source.accept_output = [&](uint32_t) { available = false; };
    OneSlotDestination destination;

    EXPECT_EQ(transferCSCBGAOutput(3, source, destination),
              CSCBGAOutputTransferResult::ACCEPTED);
    ASSERT_TRUE(destination.committed.has_value());
    EXPECT_EQ(destination.committed->global_bg_id, 3U);
    EXPECT_EQ(destination.committed->payload.output_sequence,
              output.output_sequence);
    EXPECT_FALSE(available);
    EXPECT_EQ(destination.commit_count, 1U);
    EXPECT_EQ(destination.cancel_count, 0U);
}
