#include "csc/CSCFp16DescriptorEngine.h"

#include <gtest/gtest.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <unistd.h>

using namespace csc_descriptor;

namespace {
namespace fs = std::filesystem;

std::atomic<uint32_t> directory_sequence{0};

struct TestDirectory {
    fs::path path;
    explicit TestDirectory(const char* label)
    {
        std::ostringstream name;
        name << ".csc_fp16_m34_test_" << ::getpid() << '_'
             << directory_sequence++ << '_' << label;
        path = fs::current_path() / name.str();
    }
    ~TestDirectory()
    {
        if (path.filename().string().find(".csc_fp16_m34_test_") == 0)
            fs::remove_all(path);
    }
};

CSCFp16ImageSource boundarySource()
{
    CSCFp16ImageSource source;
    source.rows = 64;
    source.cols = 18;
    const std::array<uint32_t, 18> counts =
        {0, 1, 7, 8, 9, 15, 16, 17, 31, 0, 0, 0, 0, 0, 0, 32, 33, 0};
    source.col_ptr.push_back(0);
    uint64_t ordinal = 0;
    for (uint32_t column = 0; column < source.cols; ++column) {
        for (uint32_t lane = 0; lane < counts[column]; ++lane) {
            source.row_idx.push_back(lane == 9 ? 3U :
                static_cast<uint32_t>((ordinal * 29 + 11) % source.rows));
            static const std::array<double, 11> values = {
                0.0, -0.0, -2.5, std::ldexp(1.0, -24), 65504.0,
                512.0, 1.0 / 3.0, -1.25, 2.0, 7.0,
                std::numeric_limits<double>::quiet_NaN()};
            source.values.push_back(values[ordinal % values.size()]);
            ++ordinal;
        }
        source.col_ptr.push_back(source.values.size());
    }
    for (uint32_t column = 0; column < source.cols; ++column) {
        source.x.push_back(column == 16 ? 512.0 :
                           column == 15 ? -0.5 : 1.0 + column / 8.0);
        source.column_to_bg.push_back(column % 2 ? 7U : 8U);
    }
    return source;
}

CSCFp16ImageSource wideSource()
{
    CSCFp16ImageSource source;
    source.rows = 32;
    source.cols = 17;
    source.col_ptr.assign(17, 0);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        source.row_idx.push_back((lane * 7) % source.rows);
        source.values.push_back(1.0 + lane / 16.0);
    }
    source.col_ptr.push_back(16);
    source.x.assign(source.cols, 1.0);
    source.x[16] = 1.5;
    source.column_to_bg.assign(source.cols, 7);
    return source;
}

std::array<uint8_t, 32> payloadFor(const CSCFp16ExecutionImage& execution,
                                   const CSCFp16Request& request)
{
    std::array<uint8_t, 32> payload{};
    const auto& image = execution.image();
    if (request.kind == CSCFp16RequestKind::X) {
        const uint64_t first = request.stream_offset_bytes / 2;
        for (uint32_t lane = 0; lane < 16; ++lane) {
            const uint64_t column = first + lane;
            const uint16_t bits = column < image.x_bits.size()
                                      ? image.x_bits[column] : 0;
            payload[lane * 2] = static_cast<uint8_t>(bits);
            payload[lane * 2 + 1] = static_cast<uint8_t>(bits >> 8);
        }
        return payload;
    }
    const auto& bg = image.bg.at(request.global_bg_id);
    const auto& bytes = request.kind == CSCFp16RequestKind::VALUE
                            ? bg.values : bg.row_indices;
    if (request.stream_offset_bytes + payload.size() > bytes.size())
        throw std::out_of_range("test response outside FP16 image");
    std::copy_n(bytes.begin() + request.stream_offset_bytes,
                payload.size(), payload.begin());
    return payload;
}

struct Harness {
    std::vector<CSCFp16Request> issued;
    std::size_t completion_cursor = 0;

    bool submit(const CSCFp16Request& request)
    {
        issued.push_back(request);
        return true;
    }
};

void drainEngine(CSCFp16DescriptorEngine& engine,
                 const CSCFp16ExecutionImage& image,
                 CSCFp16BoundedCaptureSink& sink, Harness& harness,
                 bool pop_sink = true)
{
    for (uint32_t guard = 0; guard < 20000 && !engine.done() && !engine.failed();
         ++guard) {
        engine.tick();
        while (harness.completion_cursor < harness.issued.size()) {
            const auto request = harness.issued[harness.completion_cursor++];
            ASSERT_TRUE(engine.complete(request, payloadFor(image, request)));
        }
        if (pop_sink && sink.size()) sink.pop();
    }
    ASSERT_FALSE(engine.failed()) << engine.error();
    ASSERT_TRUE(engine.done());
}

std::vector<CSCFp16PartialEvent> expectedForBG(
    const CSCFp16LoadedImage& image, uint32_t bg_id)
{
    std::vector<CSCFp16PartialEvent> expected;
    const auto& descriptors = image.bg[bg_id].parsed_descriptors;
    for (uint32_t descriptor_id = 0; descriptor_id < descriptors.size();
         ++descriptor_id) {
        const auto& descriptor = descriptors[descriptor_id];
        const uint64_t begin = image.matrix.col_ptr[descriptor.original_col];
        const auto x = cscFp16FromBits(image.x_bits[descriptor.original_col]);
        for (uint32_t offset = 0, chunk = 0; offset < descriptor.nnz_count;
             offset += 16, ++chunk) {
            const uint32_t valid =
                std::min<uint32_t>(16, descriptor.nnz_count - offset);
            for (uint32_t lane = 0; lane < valid; ++lane) {
                const uint64_t logical = begin + offset + lane;
                expected.push_back({
                    image.matrix.row_idx[logical],
                    cscFp16ToBits(cscFp16Mul(
                        cscFp16FromBits(image.matrix.value_bits[logical]), x)),
                    bg_id, descriptor_id, chunk, static_cast<uint8_t>(lane)});
            }
        }
    }
    return expected;
}

TEST(CSCFp16ProductionGateTest, LoadsOnlyValidatedV2)
{
    TestDirectory directory("gate");
    exportCSCFp16ImageV2(boundarySource(), directory.path.string());
    const auto execution = CSCFp16ExecutionImage::load(
        directory.path.string(), CSCFp16ExecutionMode::FP16_IMAGE_V2);
    ASSERT_NE(execution, nullptr);
    EXPECT_EQ(execution->image().matrix.value_bits.size(), 169U);

    TestDirectory v1("gate_v1");
    fs::create_directory(v1.path);
    std::ofstream(v1.path / "manifest.json")
        << "{\"format_magic\":\"SPCSCIMG\",\"format_version\":1}";
    EXPECT_THROW(CSCFp16ExecutionImage::load(
                     v1.path.string(), CSCFp16ExecutionMode::FP16_IMAGE_V2),
                 std::runtime_error);

    DRAMSim::PIMBlock fp32_datapath(DRAMSim::FP32);
    CSCFp16BoundedCaptureSink sink(1);
    EXPECT_THROW(CSCFp16DescriptorEngine(
                     7, execution, &fp32_datapath,
                     [](const auto&) { return true; }, &sink),
                 std::invalid_argument);
}

TEST(CSCFp16DescriptorEngineTest, BoundaryRequestsOffsetsAndPartialsAreExact)
{
    TestDirectory directory("boundaries");
    exportCSCFp16ImageV2(boundarySource(), directory.path.string());
    const auto execution = CSCFp16ExecutionImage::load(
        directory.path.string(), CSCFp16ExecutionMode::FP16_IMAGE_V2);

    CSCFp16EngineCounters total;
    for (uint32_t bg_id : {7U, 8U}) {
        DRAMSim::PIMBlock datapath(DRAMSim::FP16);
        CSCFp16BoundedCaptureSink sink(2);
        Harness harness;
        CSCFp16DescriptorEngine engine(
            bg_id, execution, &datapath,
            [&](const auto& request) { return harness.submit(request); }, &sink);
        engine.launch();
        drainEngine(engine, *execution, sink, harness);
        EXPECT_EQ(sink.trace(), expectedForBG(execution->image(), bg_id));
        const auto& counters = engine.counters();
        total.descriptor_count += counters.descriptor_count;
        total.descriptor_nnz_sum += counters.descriptor_nnz_sum;
        total.x_requests += counters.x_requests;
        total.value_requests += counters.value_requests;
        total.index_low_requests += counters.index_low_requests;
        total.index_high_requests += counters.index_high_requests;
        total.row_index_requests += counters.row_index_requests;
        total.logical_compute_chunks += counters.logical_compute_chunks;
        total.active_lanes += counters.active_lanes;
        total.invalid_lanes += counters.invalid_lanes;
        total.generated_partials += counters.generated_partials;
        total.emitted_partials += counters.emitted_partials;
        const auto& simd = datapath.simdCounters();
        EXPECT_EQ(simd.active_lanes, counters.active_lanes);
        EXPECT_EQ(simd.masked_lanes, counters.invalid_lanes);
        EXPECT_EQ(simd.invalid_lane_operand_reads, 0U);
        EXPECT_EQ(simd.invalid_lane_destination_writes, 0U);
    }
    EXPECT_EQ(total.descriptor_count, 10U);
    EXPECT_EQ(total.descriptor_nnz_sum, 169U);
    EXPECT_EQ(total.x_requests, 10U);
    EXPECT_EQ(total.value_requests, 15U);
    EXPECT_EQ(total.index_low_requests, 15U);
    EXPECT_EQ(total.index_high_requests, 10U);
    EXPECT_EQ(total.row_index_requests, 25U);
    EXPECT_EQ(total.logical_compute_chunks, 15U);
    EXPECT_EQ(total.active_lanes, 169U);
    EXPECT_EQ(total.invalid_lanes, 71U);
    EXPECT_EQ(total.generated_partials, 169U);
    EXPECT_EQ(total.emitted_partials, 169U);
}

TEST(CSCFp16DescriptorEngineTest, PerColumnRequestCountsMatchBoundaryContract)
{
    TestDirectory directory("request_table");
    const auto source = boundarySource();
    exportCSCFp16ImageV2(source, directory.path.string());
    const auto execution = CSCFp16ExecutionImage::load(
        directory.path.string(), CSCFp16ExecutionMode::FP16_IMAGE_V2);
    const std::array<uint32_t, 18> expected_chunks =
        {0, 1, 1, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 2, 3, 0};
    const std::array<uint32_t, 18> expected_indices =
        {0, 1, 1, 1, 2, 2, 2, 3, 4, 0, 0, 0, 0, 0, 0, 4, 5, 0};
    const std::array<uint32_t, 18> expected_high =
        {0, 0, 0, 0, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0};

    struct Counts { uint32_t x = 0, value = 0, index = 0, high = 0; };
    std::array<Counts, 18> actual{};
    for (uint32_t bg_id : {7U, 8U}) {
        DRAMSim::PIMBlock datapath(DRAMSim::FP16);
        CSCFp16BoundedCaptureSink sink(2);
        Harness harness;
        CSCFp16DescriptorEngine engine(
            bg_id, execution, &datapath,
            [&](const auto& request) { return harness.submit(request); }, &sink);
        engine.launch();
        drainEngine(engine, *execution, sink, harness);
        for (const auto& request : harness.issued) {
            const auto& descriptor = execution->image().bg[bg_id]
                                         .parsed_descriptors[request.descriptor_id];
            auto& count = actual[descriptor.original_col];
            if (request.kind == CSCFp16RequestKind::X) count.x++;
            if (request.kind == CSCFp16RequestKind::VALUE) count.value++;
            if (request.kind == CSCFp16RequestKind::INDEX_LOW ||
                request.kind == CSCFp16RequestKind::INDEX_HIGH) count.index++;
            if (request.kind == CSCFp16RequestKind::INDEX_HIGH) count.high++;
        }
    }
    for (uint32_t column = 0; column < source.cols; ++column) {
        EXPECT_EQ(actual[column].x, expected_chunks[column] ? 1U : 0U)
            << "column " << column;
        EXPECT_EQ(actual[column].value, expected_chunks[column])
            << "column " << column;
        EXPECT_EQ(actual[column].index, expected_indices[column])
            << "column " << column;
        EXPECT_EQ(actual[column].high, expected_high[column])
            << "column " << column;
    }
}

TEST(CSCFp16DescriptorEngineTest, SeventeenNnzUsesIndependentStreamOffsets)
{
    TestDirectory directory("offsets");
    exportCSCFp16ImageV2(boundarySource(), directory.path.string());
    const auto execution = CSCFp16ExecutionImage::load(
        directory.path.string(), CSCFp16ExecutionMode::FP16_IMAGE_V2);
    DRAMSim::PIMBlock datapath(DRAMSim::FP16);
    CSCFp16BoundedCaptureSink sink(2);
    Harness harness;
    CSCFp16DescriptorEngine engine(
        7, execution, &datapath,
        [&](const auto& request) { return harness.submit(request); }, &sink);
    engine.launch();
    drainEngine(engine, *execution, sink, harness);
    const auto& descriptors = execution->image().bg[7].parsed_descriptors;
    uint32_t descriptor_id = 0;
    while (descriptors[descriptor_id].original_col != 7) ++descriptor_id;
    const auto& descriptor = descriptors[descriptor_id];
    std::vector<uint64_t> values, indices;
    for (const auto& request : harness.issued) {
        if (request.descriptor_id != descriptor_id) continue;
        if (request.kind == CSCFp16RequestKind::VALUE)
            values.push_back(request.stream_offset_bytes -
                             descriptor.value_offset_bytes);
        if (request.kind == CSCFp16RequestKind::INDEX_LOW ||
            request.kind == CSCFp16RequestKind::INDEX_HIGH)
            indices.push_back(request.stream_offset_bytes -
                              descriptor.row_idx_offset_bytes);
    }
    EXPECT_EQ(values, (std::vector<uint64_t>{0, 32}));
    EXPECT_EQ(indices, (std::vector<uint64_t>{0, 32, 64}));
}

TEST(CSCFp16DescriptorEngineTest, RejectedRequestRetriesWithoutChangingIdentity)
{
    TestDirectory directory("retry");
    exportCSCFp16ImageV2(wideSource(), directory.path.string());
    const auto execution = CSCFp16ExecutionImage::load(
        directory.path.string(), CSCFp16ExecutionMode::FP16_IMAGE_V2);
    DRAMSim::PIMBlock datapath(DRAMSim::FP16);
    CSCFp16BoundedCaptureSink sink(1);
    std::vector<CSCFp16Request> attempts;
    CSCFp16DescriptorEngine engine(
        7, execution, &datapath,
        [&](const auto& request) {
            attempts.push_back(request);
            return attempts.size() != 1;
        }, &sink);
    engine.launch();
    engine.tick();
    engine.tick();
    engine.tick();
    ASSERT_EQ(attempts.size(), 2U);
    EXPECT_EQ(attempts[0], attempts[1]);
    EXPECT_EQ(engine.counters().request_retry_cycles, 1U);
}

TEST(CSCFp16DescriptorEngineTest, CompletionOrderIsIdentityBased)
{
    TestDirectory directory("completion_order");
    exportCSCFp16ImageV2(wideSource(), directory.path.string());
    const auto execution = CSCFp16ExecutionImage::load(
        directory.path.string(), CSCFp16ExecutionMode::FP16_IMAGE_V2);
    const std::array<std::array<CSCFp16RequestKind, 4>, 4> orders = {{
        {CSCFp16RequestKind::VALUE, CSCFp16RequestKind::INDEX_LOW,
         CSCFp16RequestKind::INDEX_HIGH, CSCFp16RequestKind::X},
        {CSCFp16RequestKind::INDEX_LOW, CSCFp16RequestKind::VALUE,
         CSCFp16RequestKind::INDEX_HIGH, CSCFp16RequestKind::X},
        {CSCFp16RequestKind::INDEX_HIGH, CSCFp16RequestKind::VALUE,
         CSCFp16RequestKind::INDEX_LOW, CSCFp16RequestKind::X},
        {CSCFp16RequestKind::INDEX_LOW, CSCFp16RequestKind::INDEX_HIGH,
         CSCFp16RequestKind::VALUE, CSCFp16RequestKind::X}}};

    for (const auto& order : orders) {
        DRAMSim::PIMBlock datapath(DRAMSim::FP16);
        CSCFp16BoundedCaptureSink sink(32);
        Harness harness;
        CSCFp16DescriptorEngine engine(
            7, execution, &datapath,
            [&](const auto& request) { return harness.submit(request); }, &sink);
        engine.launch();
        for (uint32_t guard = 0; guard < 20 && harness.issued.size() < 4; ++guard)
            engine.tick();
        ASSERT_EQ(harness.issued.size(), 4U);
        std::map<CSCFp16RequestKind, CSCFp16Request> by_kind;
        for (const auto& request : harness.issued) by_kind[request.kind] = request;
        for (const auto kind : order) {
            const auto& request = by_kind.at(kind);
            ASSERT_TRUE(engine.complete(request, payloadFor(*execution, request)));
        }
        EXPECT_EQ(engine.state(), CSCFp16EngineState::ISSUE_REQUESTS);
        engine.tick();
        engine.tick();
        EXPECT_EQ(engine.state(), CSCFp16EngineState::SIMD_MUL);
    }
}

TEST(CSCFp16DescriptorEngineTest, DoesNotComputeBeforeRequiredHighIndex)
{
    TestDirectory directory("missing_high");
    exportCSCFp16ImageV2(wideSource(), directory.path.string());
    const auto execution = CSCFp16ExecutionImage::load(
        directory.path.string(), CSCFp16ExecutionMode::FP16_IMAGE_V2);
    DRAMSim::PIMBlock datapath(DRAMSim::FP16);
    CSCFp16BoundedCaptureSink sink(16);
    Harness harness;
    CSCFp16DescriptorEngine engine(
        7, execution, &datapath,
        [&](const auto& request) { return harness.submit(request); }, &sink);
    engine.launch();
    for (uint32_t guard = 0; guard < 20 && harness.issued.size() < 4; ++guard)
        engine.tick();
    ASSERT_EQ(harness.issued.size(), 4U);
    CSCFp16Request high;
    for (const auto& request : harness.issued) {
        if (request.kind == CSCFp16RequestKind::INDEX_HIGH) {
            high = request;
            continue;
        }
        ASSERT_TRUE(engine.complete(request, payloadFor(*execution, request)));
    }
    engine.tick();
    engine.tick();
    engine.tick();
    EXPECT_EQ(engine.state(), CSCFp16EngineState::WAIT_OPERANDS);
    EXPECT_EQ(datapath.simdCounters().issued_operations, 0U);
    EXPECT_GT(engine.counters().operand_wait_cycles, 0U);
    ASSERT_TRUE(engine.complete(high, payloadFor(*execution, high)));
    engine.tick();
    EXPECT_EQ(engine.state(), CSCFp16EngineState::SIMD_MUL);
}

TEST(CSCFp16DescriptorEngineTest, XAddressCrossesSixteenElementBoundary)
{
    TestDirectory directory("x_boundary");
    exportCSCFp16ImageV2(boundarySource(), directory.path.string());
    const auto execution = CSCFp16ExecutionImage::load(
        directory.path.string(), CSCFp16ExecutionMode::FP16_IMAGE_V2);
    for (const auto& descriptor : execution->image().bg[7].parsed_descriptors) {
        if (descriptor.original_col == 15) {
            EXPECT_EQ(uint64_t(descriptor.original_col / 16) * 32, 0U);
        }
    }
    for (const auto& descriptor : execution->image().bg[8].parsed_descriptors) {
        if (descriptor.original_col == 16) {
            EXPECT_EQ(uint64_t(descriptor.original_col / 16) * 32, 32U);
        }
    }
}

TEST(CSCFp16DescriptorEngineTest, SinkBackpressureHoldsExactlyOneEvent)
{
    TestDirectory directory("sink_stall");
    exportCSCFp16ImageV2(boundarySource(), directory.path.string());
    const auto execution = CSCFp16ExecutionImage::load(
        directory.path.string(), CSCFp16ExecutionMode::FP16_IMAGE_V2);
    DRAMSim::PIMBlock datapath(DRAMSim::FP16);
    CSCFp16BoundedCaptureSink sink(1);
    sink.setEnabled(false);
    Harness harness;
    CSCFp16DescriptorEngine engine(
        7, execution, &datapath,
        [&](const auto& request) { return harness.submit(request); }, &sink);
    engine.launch();
    for (uint32_t guard = 0; guard < 100 &&
         engine.state() != CSCFp16EngineState::EMIT_PARTIALS; ++guard) {
        engine.tick();
        while (harness.completion_cursor < harness.issued.size()) {
            const auto request = harness.issued[harness.completion_cursor++];
            ASSERT_TRUE(engine.complete(request, payloadFor(*execution, request)));
        }
    }
    ASSERT_EQ(engine.state(), CSCFp16EngineState::EMIT_PARTIALS);
    engine.tick();
    engine.tick();
    engine.tick();
    EXPECT_EQ(engine.counters().sink_backpressure_cycles, 3U);
    EXPECT_EQ(engine.counters().emitted_partials, 0U);
    sink.setEnabled(true);
    engine.tick();
    ASSERT_EQ(sink.size(), 1U);
    EXPECT_EQ(engine.counters().emitted_partials, 1U);
    const auto first = sink.pop();
    EXPECT_EQ(first, expectedForBG(execution->image(), 7).front());
}

TEST(CSCFp16DescriptorEngineTest, RejectsDuplicateAndStaleCompletion)
{
    TestDirectory directory("completion_errors");
    exportCSCFp16ImageV2(boundarySource(), directory.path.string());
    const auto execution = CSCFp16ExecutionImage::load(
        directory.path.string(), CSCFp16ExecutionMode::FP16_IMAGE_V2);

    {
        DRAMSim::PIMBlock datapath(DRAMSim::FP16);
        CSCFp16BoundedCaptureSink sink(1);
        Harness harness;
        CSCFp16DescriptorEngine engine(
            7, execution, &datapath,
            [&](const auto& request) { return harness.submit(request); }, &sink);
        engine.launch();
        engine.tick();
        engine.tick();
        ASSERT_EQ(harness.issued.size(), 1U);
        const auto request = harness.issued.front();
        ASSERT_TRUE(engine.complete(request, payloadFor(*execution, request)));
        EXPECT_FALSE(engine.complete(request, payloadFor(*execution, request)));
        EXPECT_TRUE(engine.failed());
    }
    {
        DRAMSim::PIMBlock datapath(DRAMSim::FP16);
        CSCFp16BoundedCaptureSink sink(1);
        Harness harness;
        CSCFp16DescriptorEngine engine(
            7, execution, &datapath,
            [&](const auto& request) { return harness.submit(request); }, &sink);
        engine.launch();
        engine.tick();
        engine.tick();
        ASSERT_EQ(harness.issued.size(), 1U);
        auto stale = harness.issued.front();
        stale.chunk_id++;
        EXPECT_FALSE(engine.complete(stale, payloadFor(*execution, harness.issued.front())));
        EXPECT_TRUE(engine.failed());
    }
}

}  // namespace
