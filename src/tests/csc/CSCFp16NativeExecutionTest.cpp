#include "AddressMapping.h"
#include "csc/CSCFp16NativeExecution.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
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
        name << ".csc_fp16_native_test_" << ::getpid() << '_'
             << directory_sequence++ << '_' << label;
        path = fs::current_path() / name.str();
    }
    ~TestDirectory()
    {
        if (path.filename().string().find(".csc_fp16_native_test_") == 0)
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

std::array<uint8_t, 32> imagePayload(const CSCFp16ExecutionImage& execution,
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
    std::copy_n(bytes.begin() + request.stream_offset_bytes, 32,
                payload.begin());
    return payload;
}

std::array<std::vector<CSCFp16PartialEvent>, 64> runSynthetic(
    std::shared_ptr<const CSCFp16ExecutionImage> execution)
{
    std::array<std::vector<CSCFp16PartialEvent>, 64> traces;
    for (uint32_t bg = 0; bg < 64; ++bg) {
        DRAMSim::PIMBlock datapath(DRAMSim::FP16);
        CSCFp16BoundedCaptureSink sink(256);
        std::vector<CSCFp16Request> pending;
        std::size_t completion = 0;
        CSCFp16DescriptorEngine engine(
            bg, execution, &datapath,
            [&](const CSCFp16Request& request) {
                pending.push_back(request);
                return true;
            }, &sink);
        engine.launch();
        for (uint32_t guard = 0; guard < 20000 && !engine.done(); ++guard) {
            engine.tick();
            while (completion < pending.size()) {
                const auto request = pending[completion++];
                EXPECT_TRUE(engine.complete(request,
                    imagePayload(*execution, request)));
            }
        }
        EXPECT_FALSE(engine.failed()) << engine.error();
        EXPECT_TRUE(engine.done());
        traces[bg] = sink.trace();
    }
    return traces;
}

void runNative(CSCFp16NativeExecution& native, uint32_t max_cycles = 200000)
{
    native.launch();
    for (uint32_t guard = 0; guard < max_cycles && !native.done() &&
         !native.failed(); ++guard)
        native.tick();
    ASSERT_FALSE(native.failed()) << native.error();
    ASSERT_TRUE(native.done());
}

std::vector<CSCFp16PartialEvent> flatten(
    const std::array<std::vector<CSCFp16PartialEvent>, 64>& traces)
{
    std::vector<CSCFp16PartialEvent> result;
    for (const auto& trace : traces)
        result.insert(result.end(), trace.begin(), trace.end());
    return result;
}

std::vector<CSCFp16PartialEvent> flattenNative(
    const CSCFp16NativeExecution& native)
{
    std::vector<CSCFp16PartialEvent> result;
    for (uint32_t bg = 0; bg < 64; ++bg) {
        const auto& trace = native.sink(bg).trace();
        result.insert(result.end(), trace.begin(), trace.end());
    }
    return result;
}

TEST(CSCFp16NativeExecutionTest, SyntheticAndNativeTracesAreBitExact)
{
    TestDirectory directory("equivalence");
    exportCSCFp16ImageV2(boundarySource(), directory.path.string());
    const auto execution = CSCFp16ExecutionImage::load(
        directory.path.string(), CSCFp16ExecutionMode::FP16_IMAGE_V2);
    const auto synthetic = runSynthetic(execution);
    CSCFp16NativeExecution native(execution, 256);
    runNative(native);
    for (uint32_t bg = 0; bg < 64; ++bg)
        EXPECT_EQ(native.sink(bg).trace(), synthetic[bg]) << "BG " << bg;

    const auto native_trace = flattenNative(native);
    const auto synthetic_trace = flatten(synthetic);
    ASSERT_EQ(native_trace, synthetic_trace);
    ASSERT_EQ(native_trace.size(), 169U);
    const uint64_t trace_hash = cscFp16PartialTraceFnv1a64(native_trace);
    EXPECT_EQ(trace_hash, cscFp16PartialTraceFnv1a64(synthetic_trace));
    EXPECT_EQ(trace_hash, 0x2e2867563f3d9cacULL);

    const auto counters = native.counters();
    EXPECT_EQ(counters.engine.descriptor_count, 10U);
    EXPECT_EQ(counters.engine.descriptor_nnz_sum, 169U);
    EXPECT_EQ(counters.engine.x_requests, 10U);
    EXPECT_EQ(counters.engine.value_requests, 15U);
    EXPECT_EQ(counters.engine.index_low_requests, 15U);
    EXPECT_EQ(counters.engine.index_high_requests, 10U);
    EXPECT_EQ(counters.engine.row_index_requests, 25U);
    EXPECT_EQ(counters.engine.logical_compute_chunks, 15U);
    EXPECT_EQ(counters.engine.active_lanes, 169U);
    EXPECT_EQ(counters.engine.invalid_lanes, 71U);
    EXPECT_EQ(counters.engine.generated_partials, 169U);
    EXPECT_EQ(counters.engine.emitted_partials, 169U);
    EXPECT_EQ(counters.timing.accepted_requests, 50U);
    EXPECT_EQ(counters.timing.completed_requests, 50U);
    EXPECT_EQ(counters.timing.request_attempts, 50U);
    EXPECT_EQ(counters.timing.outstanding_high_water, 8U);
    EXPECT_EQ(counters.timing.total_compute_only_cycles, 544U);
    EXPECT_EQ(counters.timing.operand_wait_cycles, 710U);
    EXPECT_GT(counters.timing.response_latency_sum, 0U);
    EXPECT_GT(counters.timing.response_latency_min, 0U);
    EXPECT_GE(counters.timing.response_latency_max,
              counters.timing.response_latency_min);
    EXPECT_LT(counters.timing.first_descriptor_cycle,
              counters.timing.first_accepted_request_cycle);
    EXPECT_LE(counters.timing.first_accepted_request_cycle,
              counters.timing.first_response_cycle);
    EXPECT_LT(counters.timing.first_response_cycle,
              counters.timing.first_pim_mul_cycle);
    EXPECT_LE(counters.timing.last_pim_mul_cycle,
              counters.timing.last_accepted_partial_cycle);
    EXPECT_EQ(counters.timing.last_accepted_partial_cycle,
              counters.timing.completion_cycle - 2);

    std::cout << "FP16_M45_GOLDEN cycles="
              << counters.timing.total_compute_only_cycles
              << " attempts=" << counters.timing.request_attempts
              << " accepted=" << counters.timing.accepted_requests
              << " responses=" << counters.timing.completed_requests
              << " outstanding_hwm=" << counters.timing.outstanding_high_water
              << " operand_wait=" << counters.timing.operand_wait_cycles
              << " trace_records=" << native_trace.size()
              << " trace_fnv1a64=0x" << std::hex
              << trace_hash << std::dec << '\n';
}

TEST(CSCFp16NativeExecutionTest, PhysicalAddressesPreserveScheme8AndBoundaries)
{
    TestDirectory directory("addresses");
    exportCSCFp16ImageV2(boundarySource(), directory.path.string());
    const auto execution = CSCFp16ExecutionImage::load(
        directory.path.string(), CSCFp16ExecutionMode::FP16_IMAGE_V2);
    CSCFp16NativeExecution native(execution, 256);
    runNative(native);

    DRAMSim::AddrMapping mapping;
    bool saw_x0 = false, saw_x1 = false;
    std::vector<uint64_t> value17, index17, value33, index33;
    for (const auto& record : native.requestTrace()) {
        unsigned channel, rank, bank, row, column;
        mapping.addressMapping(record.physical_address, channel, rank, bank,
                               row, column);
        EXPECT_EQ(channel, record.request.global_bg_id / 4);
        EXPECT_EQ(rank, 0U);
        EXPECT_EQ(mapping.bankgroupId(bank), record.request.global_bg_id % 4);
        EXPECT_EQ(record.physical_address,
                  native.physicalAddress(record.request));
        const auto& descriptor = execution->image()
                                     .bg[record.request.global_bg_id]
                                     .parsed_descriptors[record.request.descriptor_id];
        if (record.request.kind == CSCFp16RequestKind::X &&
            descriptor.original_col == 15) {
            EXPECT_EQ(record.request.stream_offset_bytes, 0U);
            saw_x0 = true;
        }
        if (record.request.kind == CSCFp16RequestKind::X &&
            descriptor.original_col == 16) {
            EXPECT_EQ(record.request.stream_offset_bytes, 32U);
            saw_x1 = true;
        }
        auto collect = [&](std::vector<uint64_t>& values,
                           std::vector<uint64_t>& indices) {
            if (record.request.kind == CSCFp16RequestKind::VALUE)
                values.push_back(record.request.stream_offset_bytes -
                                 descriptor.value_offset_bytes);
            if (record.request.kind == CSCFp16RequestKind::INDEX_LOW ||
                record.request.kind == CSCFp16RequestKind::INDEX_HIGH)
                indices.push_back(record.request.stream_offset_bytes -
                                  descriptor.row_idx_offset_bytes);
        };
        if (descriptor.original_col == 7) collect(value17, index17);
        if (descriptor.original_col == 16) collect(value33, index33);
    }
    EXPECT_TRUE(saw_x0);
    EXPECT_TRUE(saw_x1);
    std::sort(value17.begin(), value17.end());
    std::sort(index17.begin(), index17.end());
    std::sort(value33.begin(), value33.end());
    std::sort(index33.begin(), index33.end());
    EXPECT_EQ(value17, (std::vector<uint64_t>{0, 32}));
    EXPECT_EQ(index17, (std::vector<uint64_t>{0, 32, 64}));
    EXPECT_EQ(value33, (std::vector<uint64_t>{0, 32, 64}));
    EXPECT_EQ(index33,
              (std::vector<uint64_t>{0, 32, 64, 96, 128}));
}

TEST(CSCFp16NativeExecutionTest, RejectionRetriesSameIdentityExactlyOnce)
{
    TestDirectory directory("retry");
    exportCSCFp16ImageV2(boundarySource(), directory.path.string());
    const auto execution = CSCFp16ExecutionImage::load(
        directory.path.string(), CSCFp16ExecutionMode::FP16_IMAGE_V2);
    CSCFp16NativeExecution native(execution, 256);
    native.setRejectBudget(7, 3);
    runNative(native);
    const auto counters = native.counters();
    EXPECT_EQ(counters.timing.rejected_requests, 3U);
    EXPECT_EQ(counters.timing.retry_attempts, 3U);
    EXPECT_EQ(counters.timing.accepted_requests, 50U);
    EXPECT_EQ(counters.timing.completed_requests, 50U);
    EXPECT_EQ(native.requestTrace().size(), 50U);
}

TEST(CSCFp16NativeExecutionTest, SinkBackpressureDelaysCompletionWithoutTraceChange)
{
    TestDirectory directory("sink_stall");
    exportCSCFp16ImageV2(boundarySource(), directory.path.string());
    const auto execution = CSCFp16ExecutionImage::load(
        directory.path.string(), CSCFp16ExecutionMode::FP16_IMAGE_V2);

    CSCFp16NativeExecution reference(execution, 256);
    runNative(reference);
    const auto expected = flattenNative(reference);

    CSCFp16NativeExecution stalled(execution, 1);
    stalled.launch();
    uint64_t forced_stall = 0;
    for (uint32_t guard = 0; guard < 200000 && !stalled.done() &&
         !stalled.failed(); ++guard) {
        stalled.tick();
        for (uint32_t bg : {7U, 8U}) {
            if (!stalled.sink(bg).size()) continue;
            if (forced_stall < 8) {
                forced_stall++;
            } else {
                stalled.popCaptured(bg);
            }
        }
    }
    ASSERT_FALSE(stalled.failed()) << stalled.error();
    ASSERT_TRUE(stalled.done());
    EXPECT_EQ(flattenNative(stalled), expected);
    EXPECT_GT(stalled.counters().timing.capture_sink_stall_cycles, 0U);
    EXPECT_GT(stalled.counters().timing.total_compute_only_cycles,
              reference.counters().timing.total_compute_only_cycles);
}

std::vector<CSCFp16BGAOutputEvent> referenceBGA(
    uint32_t bg, const std::vector<CSCFp16PartialEvent>& input, uint32_t capacity)
{
    struct Entry {
        uint32_t row;
        CSCFp16Bits value;
        uint64_t age;
        uint64_t contributions;
    };
    std::vector<Entry> entries;
    std::vector<CSCFp16BGAOutputEvent> output;
    uint64_t age = 0, sequence = 0;
    for (const auto& event : input) {
        auto hit = std::find_if(entries.begin(), entries.end(), [&](const Entry& entry) {
            return entry.row == event.row_idx;
        });
        if (hit != entries.end()) {
            hit->value = cscFp16ToBits(cscFp16Add(
                cscFp16FromBits(hit->value), cscFp16FromBits(event.value_bits)));
            hit->contributions++;
            continue;
        }
        if (entries.size() == capacity) {
            auto victim = std::min_element(entries.begin(), entries.end(),
                [](const Entry& a, const Entry& b) { return a.age < b.age; });
            output.push_back({victim->row, victim->value, bg,
                              CSCFp16BGAOutputReason::CAPACITY_EVICTION,
                              ++sequence, victim->contributions});
            entries.erase(victim);
        }
        entries.push_back({event.row_idx, event.value_bits, ++age, 1});
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) { return a.age < b.age; });
    for (const auto& entry : entries)
        output.push_back({entry.row, entry.value, bg,
                          CSCFp16BGAOutputReason::FINAL_DRAIN,
                          ++sequence, entry.contributions});
    return output;
}

std::vector<CSCFp16BGAOutputEvent> flattenBGAOutput(
    const CSCFp16NativeExecution& native)
{
    std::vector<CSCFp16BGAOutputEvent> result;
    for (uint32_t bg = 0; bg < 64; ++bg) {
        const auto& trace = native.bgaOutputSink(bg).trace();
        result.insert(result.end(), trace.begin(), trace.end());
    }
    return result;
}

TEST(CSCFp16NativeExecutionTest, NativeFp16BGAMatchesIndependentReplayReference)
{
    TestDirectory directory("m5_bga");
    exportCSCFp16ImageV2(boundarySource(), directory.path.string());
    const auto execution = CSCFp16ExecutionImage::load(
        directory.path.string(), CSCFp16ExecutionMode::FP16_IMAGE_V2);
    CSCFp16BGAConfig config;
    config.rows = 64;
    config.accumulator_entries = config.compare_width = 8;
    CSCFp16NativeExecution native(execution, 256, &config, 256);
    runNative(native);

    const auto synthetic = runSynthetic(execution);
    std::vector<CSCFp16BGAOutputEvent> expected;
    for (uint32_t bg = 0; bg < 64; ++bg) {
        EXPECT_EQ(native.bgaIngressTrace(bg), synthetic[bg]);
        const auto reference = referenceBGA(bg, synthetic[bg], 8);
        EXPECT_EQ(native.bgaOutputSink(bg).trace(), reference) << "BG " << bg;
        expected.insert(expected.end(), reference.begin(), reference.end());
    }
    const auto actual = flattenBGAOutput(native);
    ASSERT_EQ(actual, expected);
    const auto partials = flatten(synthetic);
    EXPECT_EQ(cscFp16PartialTraceFnv1a64(partials), 0x2e2867563f3d9cacULL);
    const auto counters = native.counters();
    EXPECT_EQ(counters.engine.generated_partials, 169U);
    EXPECT_EQ(counters.engine.emitted_partials, 169U);
    EXPECT_EQ(counters.bga.ingress_accepted, 169U);
    EXPECT_EQ(counters.bga.retired_contributions, 169U);
    EXPECT_EQ(counters.bga.fp16_adds, counters.bga.merges);
    EXPECT_EQ(counters.bga.output_accepted,
              counters.bga.capacity_evictions + counters.bga.final_drain_outputs);
    EXPECT_GT(counters.bga.capacity_evictions, 0U);
    EXPECT_EQ(counters.bga.queue_high_water, 8U);
    EXPECT_GT(counters.timing.compute_bga_completion_cycle,
              counters.timing.compute_complete_cycle);
    const uint64_t hash = cscFp16BGAOutputTraceFnv1a64(actual);
    std::cout << "FP16_M5_GOLDEN cycles="
              << counters.timing.total_compute_bga_cycles
              << " compute_cycle=" << counters.timing.compute_complete_cycle
              << " drain_start=" << counters.timing.final_drain_start_cycle
              << " ingress=" << counters.bga.ingress_accepted
              << " ingress_stall_engine_cycles="
              << counters.timing.bga_ingress_stall_engine_cycles
              << " merges=" << counters.bga.merges
              << " evictions=" << counters.bga.capacity_evictions
              << " final_drains=" << counters.bga.final_drain_outputs
              << " outputs=" << actual.size()
              << " trace_fnv1a64=0x" << std::hex << hash << std::dec << '\n';
}

TEST(CSCFp16NativeExecutionTest, BGAOutputBackpressureDelaysButPreservesTrace)
{
    TestDirectory directory("m5_output_stall");
    exportCSCFp16ImageV2(boundarySource(), directory.path.string());
    const auto execution = CSCFp16ExecutionImage::load(
        directory.path.string(), CSCFp16ExecutionMode::FP16_IMAGE_V2);
    CSCFp16BGAConfig config;
    config.rows = 64;
    config.accumulator_entries = config.compare_width = 8;
    CSCFp16NativeExecution reference(execution, 256, &config, 256);
    runNative(reference);
    const auto expected = flattenBGAOutput(reference);

    CSCFp16NativeExecution stalled(execution, 256, &config, 256);
    stalled.setBGAOutputSinkEnabled(7, false);
    stalled.setBGAOutputSinkEnabled(8, false);
    stalled.launch();
    for (uint32_t guard = 0; guard < 200000 && !stalled.done() &&
         !stalled.failed(); ++guard) {
        if (guard == 600) {
            stalled.setBGAOutputSinkEnabled(7, true);
            stalled.setBGAOutputSinkEnabled(8, true);
        }
        stalled.tick();
    }
    ASSERT_FALSE(stalled.failed()) << stalled.error();
    ASSERT_TRUE(stalled.done());
    EXPECT_EQ(flattenBGAOutput(stalled), expected);
    EXPECT_GT(stalled.counters().timing.bga_output_sink_stall_cycles, 0U);
    EXPECT_GT(stalled.counters().timing.bga_ingress_stall_engine_cycles, 0U);
    EXPECT_GT(stalled.counters().timing.bga_ingress_stall_global_cycles, 0U);
    EXPECT_GT(stalled.counters().timing.total_compute_bga_cycles,
              reference.counters().timing.total_compute_bga_cycles);
    EXPECT_EQ(stalled.counters().bga.ingress_accepted, 169U);
}

}  // namespace
