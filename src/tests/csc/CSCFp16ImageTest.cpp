#include "csc/CSCFp16Image.h"
#include "tests/csc/CSCExternalImage.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace csc_descriptor;

namespace {

namespace fs = std::filesystem;

std::atomic<uint32_t> next_directory{0};

struct TestDirectory {
    fs::path path;
    explicit TestDirectory(const char* label)
    {
        std::ostringstream name;
        name << ".csc_fp16_v2_test_" << ::getpid() << '_'
             << next_directory++ << '_' << label;
        path = fs::current_path() / name.str();
        if (fs::exists(path)) fs::remove_all(path);
    }
    ~TestDirectory()
    {
        if (path.filename().string().find(".csc_fp16_v2_test_") == 0)
            fs::remove_all(path);
    }
};

CSCFp16ImageSource boundarySource()
{
    CSCFp16ImageSource source;
    source.rows = 64;
    source.cols = 11;
    const std::array<uint32_t, 11> counts =
        {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33};
    source.col_ptr.push_back(0);
    uint64_t ordinal = 0;
    for (uint32_t column = 0; column < source.cols; ++column) {
        for (uint32_t lane = 0; lane < counts[column]; ++lane) {
            source.row_idx.push_back(
                lane == 1 ? 3U : static_cast<uint32_t>((ordinal * 17 + 5) % source.rows));
            static const std::array<double, 13> special = {
                0.0, -0.0, std::ldexp(1.0, -14), std::ldexp(1.0, -24),
                65504.0, 70000.0, -70000.0, std::ldexp(1.0, -30),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity(), -3.25, 1.0 / 3.0};
            source.values.push_back(special[ordinal % special.size()]);
            ++ordinal;
        }
        source.col_ptr.push_back(source.values.size());
        source.column_to_bg.push_back((column * 7) % 64);
    }
    source.x = {1.0, -0.0, std::ldexp(1.0, -24), 70000.0, -70000.0,
                0.5, -2.0, 1.0 / 3.0,
                std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::quiet_NaN(), 4.0};
    source.validate();
    return source;
}

std::vector<uint8_t> readBytes(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("test cannot read file");
    input.seekg(0, std::ios::end);
    std::vector<uint8_t> bytes(static_cast<std::size_t>(input.tellg()));
    input.seekg(0, std::ios::beg);
    if (!bytes.empty())
        input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    return bytes;
}

void writeBytes(const fs::path& path, const std::vector<uint8_t>& bytes)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    if (!output) throw std::runtime_error("test cannot write file");
}

std::string hex64(uint64_t value)
{
    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << value;
    return stream.str();
}

std::string readText(const fs::path& path)
{
    const auto bytes = readBytes(path);
    return std::string(bytes.begin(), bytes.end());
}

void writeText(const fs::path& path, const std::string& text)
{
    writeBytes(path, std::vector<uint8_t>(text.begin(), text.end()));
}

void replaceOnce(std::string& text, const std::string& from,
                 const std::string& to)
{
    const auto position = text.find(from);
    ASSERT_NE(position, std::string::npos) << from;
    text.replace(position, from.size(), to);
}

std::vector<CSCFp16Bits> convertedValues(const CSCFp16ImageSource& source)
{
    std::vector<CSCFp16Bits> bits;
    for (const double value : source.values)
        bits.push_back(cscFp16ToBits(cscFp16FromDouble(value)));
    return bits;
}

std::vector<CSCFp16Bits> convertedX(const CSCFp16ImageSource& source)
{
    std::vector<CSCFp16Bits> bits;
    for (const double value : source.x)
        bits.push_back(cscFp16ToBits(cscFp16FromDouble(value)));
    return bits;
}

}  // namespace

TEST(CSCFp16ImageV2Test, BoundaryFixtureRoundTripsBitExactly)
{
    const auto source = boundarySource();
    TestDirectory directory("roundtrip");
    exportCSCFp16ImageV2(source, directory.path.string());
    const auto loaded = loadCSCFp16ImageV2(directory.path.string());

    EXPECT_EQ(loaded.matrix.rows, source.rows);
    EXPECT_EQ(loaded.matrix.cols, source.cols);
    EXPECT_EQ(loaded.matrix.col_ptr, source.col_ptr);
    EXPECT_EQ(loaded.matrix.row_idx, source.row_idx);
    EXPECT_EQ(loaded.matrix.value_bits, convertedValues(source));
    EXPECT_EQ(loaded.x_bits, convertedX(source));
    EXPECT_EQ(loaded.column_to_bg, source.column_to_bg);
    EXPECT_EQ(loaded.accounting.descriptor_count, 10U);
    EXPECT_EQ(loaded.accounting.logical_value_bytes,
              source.values.size() * 2U);
    EXPECT_EQ(loaded.accounting.logical_index_bytes,
              source.row_idx.size() * 4U);
    EXPECT_EQ(loaded.accounting.logical_x_bytes, source.x.size() * 2U);
    EXPECT_EQ(loaded.accounting.physical_x_bytes, 32U);
    EXPECT_EQ(loaded.accounting.x_padding_bytes, 10U);
    EXPECT_EQ(loaded.conversion.conversion_count,
              source.values.size() + source.x.size());
    EXPECT_GT(loaded.conversion.subnormal_count, 0U);
    EXPECT_GT(loaded.conversion.underflow_to_zero_count, 0U);
    EXPECT_GT(loaded.conversion.positive_infinity_count, 0U);
    EXPECT_GT(loaded.conversion.negative_infinity_count, 0U);
    EXPECT_GT(loaded.conversion.nan_count, 0U);
    EXPECT_GT(loaded.conversion.signed_zero_count, 0U);

    uint64_t descriptors = 0;
    for (uint32_t bg = 0; bg < 64; ++bg) {
        EXPECT_EQ(loaded.bg[bg].values.size() % 32, 0U);
        EXPECT_EQ(loaded.bg[bg].row_indices.size() % 32, 0U);
        for (const auto& descriptor : loaded.bg[bg].parsed_descriptors) {
            EXPECT_EQ(descriptor.global_bg_id, bg);
            EXPECT_EQ(descriptor.value_offset_bytes % 32, 0U);
            EXPECT_EQ(descriptor.row_idx_offset_bytes % 32, 0U);
            ++descriptors;
        }
    }
    EXPECT_EQ(descriptors, 10U);

    CSCFp16LogicalMatrix expected_matrix{
        source.rows, source.cols, source.col_ptr, source.row_idx,
        convertedValues(source)};
    EXPECT_EQ(cscFp16SequentialSpMV(loaded.matrix, loaded.x_bits),
              cscFp16SequentialSpMV(expected_matrix, convertedX(source)));
}

TEST(CSCFp16ImageV2Test, ShortColumnsExposeAlignmentOverhead)
{
    CSCFp16ImageSource source;
    source.rows = 2;
    source.cols = 2;
    source.col_ptr = {0, 1, 2};
    source.row_idx = {0, 1};
    source.values = {1.0, 2.0};
    source.x = {3.0, 4.0};
    source.column_to_bg = {0, 0};
    TestDirectory directory("padding");
    exportCSCFp16ImageV2(source, directory.path.string());
    const auto loaded = loadCSCFp16ImageV2(directory.path.string());
    EXPECT_EQ(loaded.accounting.logical_value_bytes, 4U);
    EXPECT_EQ(loaded.accounting.physical_value_bytes, 64U);
    EXPECT_EQ(loaded.accounting.value_padding_bytes, 60U);
    EXPECT_EQ(loaded.accounting.logical_index_bytes, 8U);
    EXPECT_EQ(loaded.accounting.physical_index_bytes, 64U);
}

TEST(CSCFp16ImageV2Test, RejectsExistingOutputDirectory)
{
    TestDirectory directory("no_overwrite");
    fs::create_directory(directory.path);
    EXPECT_THROW(exportCSCFp16ImageV2(boundarySource(),
                                     directory.path.string()),
                 std::runtime_error);
}

TEST(CSCFp16ImageV2Test, FP16LoaderRejectsV1AndManifestMismatch)
{
    TestDirectory v1("v1");
    fs::create_directory(v1.path);
    writeText(v1.path / "manifest.json",
              "{\"format_magic\":\"SPCSCIMG\",\"format_version\":1}");
    EXPECT_THROW(loadCSCFp16ImageV2(v1.path.string()), std::runtime_error);

    TestDirectory mismatch("precision");
    exportCSCFp16ImageV2(boundarySource(), mismatch.path.string());
    auto manifest = readText(mismatch.path / "manifest.json");
    replaceOnce(manifest, "\"precision\":\"FP16\"",
                          "\"precision\":\"FP32\"");
    writeText(mismatch.path / "manifest.json", manifest);
    EXPECT_THROW(loadCSCFp16ImageV2(mismatch.path.string()),
                 std::runtime_error);

    TestDirectory unsupported("unsupported_version");
    exportCSCFp16ImageV2(boundarySource(), unsupported.path.string());
    manifest = readText(unsupported.path / "manifest.json");
    replaceOnce(manifest, "\"format_version\":2",
                          "\"format_version\":99");
    writeText(unsupported.path / "manifest.json", manifest);
    EXPECT_THROW(loadCSCFp16ImageV2(unsupported.path.string()),
                 std::runtime_error);
}

TEST(CSCFp16ImageV2Test, RejectsExecutionCriticalMetadataMismatch)
{
    struct Mutation { const char* label; const char* from; const char* to; };
    const std::array<Mutation, 3> mutations = {{
        {"value_width", "\"value_bytes\":2", "\"value_bytes\":4"},
        {"simd_width", "\"future_simd_width\":16", "\"future_simd_width\":8"},
        {"index_lanes", "\"row_indices_per_burst\":8",
                        "\"row_indices_per_burst\":16"}}};
    for (const auto& mutation : mutations) {
        TestDirectory directory(mutation.label);
        exportCSCFp16ImageV2(boundarySource(), directory.path.string());
        auto manifest = readText(directory.path / "manifest.json");
        replaceOnce(manifest, mutation.from, mutation.to);
        writeText(directory.path / "manifest.json", manifest);
        EXPECT_THROW(loadCSCFp16ImageV2(directory.path.string()),
                     std::runtime_error) << mutation.label;
    }
}

TEST(CSCFp16ImageV2Test, RejectsTruncatedSecondIndexBurst)
{
    TestDirectory directory("truncated_index_high");
    exportCSCFp16ImageV2(boundarySource(), directory.path.string());
    const fs::path index_file = directory.path / "bg_28_row_idx_u32.bin";
    auto bytes = readBytes(index_file);
    ASSERT_GE(bytes.size(), 64U);
    bytes.pop_back();
    writeBytes(index_file, bytes);
    EXPECT_THROW(loadCSCFp16ImageV2(directory.path.string()),
                 std::runtime_error);
}

TEST(CSCFp16ImageV2Test, ExistingFP32LoaderRejectsV2)
{
    TestDirectory directory("fp32_reject");
    exportCSCFp16ImageV2(boundarySource(), directory.path.string());
    EXPECT_THROW(loadExternalPhysicalImage(directory.path.string()),
                 std::runtime_error);
}

TEST(CSCFp16ImageV2Test, RejectsChecksumAndFileSizeCorruption)
{
    TestDirectory checksum("checksum");
    exportCSCFp16ImageV2(boundarySource(), checksum.path.string());
    const fs::path value_file = checksum.path / "bg_07_values_fp16.bin";
    auto bytes = readBytes(value_file);
    ASSERT_FALSE(bytes.empty());
    bytes[0] ^= 1;
    writeBytes(value_file, bytes);
    EXPECT_THROW(loadCSCFp16ImageV2(checksum.path.string()),
                 std::runtime_error);

    TestDirectory size("size");
    exportCSCFp16ImageV2(boundarySource(), size.path.string());
    const fs::path x_file = size.path / "x_fp16.bin";
    bytes = readBytes(x_file);
    bytes.push_back(0);
    writeBytes(x_file, bytes);
    EXPECT_THROW(loadCSCFp16ImageV2(size.path.string()),
                 std::runtime_error);
}

TEST(CSCFp16ImageV2Test, RejectsMisalignedDescriptorWithValidChecksum)
{
    TestDirectory directory("alignment");
    exportCSCFp16ImageV2(boundarySource(), directory.path.string());
    const fs::path descriptor_file =
        directory.path / "bg_07_descriptors.bin";
    auto bytes = readBytes(descriptor_file);
    ASSERT_GE(bytes.size(), 32U);
    const std::string old_hash = hex64(cscFp16ImageFnv1a64(bytes));
    bytes[0] = 1;
    writeBytes(descriptor_file, bytes);
    const std::string new_hash = hex64(cscFp16ImageFnv1a64(bytes));
    auto manifest = readText(directory.path / "manifest.json");
    replaceOnce(manifest, old_hash, new_hash);
    writeText(directory.path / "manifest.json", manifest);
    EXPECT_THROW(loadCSCFp16ImageV2(directory.path.string()),
                 std::runtime_error);
}

TEST(CSCFp16ImageV2Test, RejectsNonZeroPaddingWithValidChecksum)
{
    TestDirectory directory("zero_padding");
    exportCSCFp16ImageV2(boundarySource(), directory.path.string());
    const fs::path value_file = directory.path / "bg_07_values_fp16.bin";
    auto bytes = readBytes(value_file);
    ASSERT_GE(bytes.size(), 3U);
    const std::string old_hash = hex64(cscFp16ImageFnv1a64(bytes));
    bytes[2] = 1;
    writeBytes(value_file, bytes);
    const std::string new_hash = hex64(cscFp16ImageFnv1a64(bytes));
    auto manifest = readText(directory.path / "manifest.json");
    replaceOnce(manifest, old_hash, new_hash);
    writeText(directory.path / "manifest.json", manifest);
    EXPECT_THROW(loadCSCFp16ImageV2(directory.path.string()),
                 std::runtime_error);
}
