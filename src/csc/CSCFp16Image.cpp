#include "csc/CSCFp16Image.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace csc_descriptor {
namespace {

namespace fs = std::filesystem;

uint64_t align32(uint64_t value)
{
    if (value > std::numeric_limits<uint64_t>::max() - 31)
        throw std::overflow_error("FP16 image alignment overflow");
    return (value + 31) / 32 * 32;
}

void appendU32(std::vector<uint8_t>& bytes, uint32_t value)
{
    for (uint32_t shift = 0; shift < 32; shift += 8)
        bytes.push_back(static_cast<uint8_t>((value >> shift) & 0xffU));
}

void appendU64(std::vector<uint8_t>& bytes, uint64_t value)
{
    for (uint32_t shift = 0; shift < 64; shift += 8)
        bytes.push_back(static_cast<uint8_t>((value >> shift) & 0xffU));
}

uint32_t readU32(const std::vector<uint8_t>& bytes, std::size_t offset)
{
    if (offset > bytes.size() || bytes.size() - offset < 4)
        throw std::runtime_error("FP16 image u32 outside file");
    uint32_t value = 0;
    for (uint32_t byte = 0; byte < 4; ++byte)
        value |= uint32_t(bytes[offset + byte]) << (byte * 8);
    return value;
}

uint64_t readU64(const std::vector<uint8_t>& bytes, std::size_t offset)
{
    if (offset > bytes.size() || bytes.size() - offset < 8)
        throw std::runtime_error("FP16 image u64 outside file");
    uint64_t value = 0;
    for (uint32_t byte = 0; byte < 8; ++byte)
        value |= uint64_t(bytes[offset + byte]) << (byte * 8);
    return value;
}

std::string prefix(uint32_t bg)
{
    std::ostringstream stream;
    stream << "bg_" << std::setw(2) << std::setfill('0') << bg;
    return stream.str();
}

void writeBytes(const fs::path& path, const std::vector<uint8_t>& bytes)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create " + path.string());
    if (!bytes.empty())
        output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    if (!output) throw std::runtime_error("cannot write " + path.string());
}

std::vector<uint8_t> readBytes(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    if (end < 0) throw std::runtime_error("cannot size " + path.string());
    std::vector<uint8_t> bytes(static_cast<std::size_t>(end));
    input.seekg(0, std::ios::beg);
    if (!bytes.empty())
        input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!input) throw std::runtime_error("cannot read " + path.string());
    return bytes;
}

std::string readText(const fs::path& path)
{
    const auto bytes = readBytes(path);
    return std::string(bytes.begin(), bytes.end());
}

std::string hex64(uint64_t value)
{
    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << value;
    return stream.str();
}

std::size_t valueAt(const std::string& json, const std::string& key)
{
    const std::string quoted = "\"" + key + "\"";
    std::size_t position = json.find(quoted);
    if (position == std::string::npos)
        throw std::runtime_error("FP16 manifest missing " + key);
    position = json.find(':', position + quoted.size());
    if (position == std::string::npos)
        throw std::runtime_error("FP16 manifest syntax " + key);
    return position + 1;
}

std::string stringValue(const std::string& json, const std::string& key)
{
    std::size_t begin = json.find('"', valueAt(json, key));
    std::size_t end = begin == std::string::npos
                          ? begin : json.find('"', begin + 1);
    if (begin == std::string::npos || end == std::string::npos)
        throw std::runtime_error("FP16 manifest string " + key);
    return json.substr(begin + 1, end - begin - 1);
}

uint64_t uintValue(const std::string& json, const std::string& key)
{
    const std::size_t begin = valueAt(json, key);
    std::size_t consumed = 0;
    const unsigned long long value =
        std::stoull(json.substr(begin), &consumed, 10);
    if (!consumed) throw std::runtime_error("FP16 manifest integer " + key);
    return static_cast<uint64_t>(value);
}

std::string arrayBody(const std::string& json, const std::string& key)
{
    const std::size_t begin = json.find('[', valueAt(json, key));
    const std::size_t end = begin == std::string::npos
                                ? begin : json.find(']', begin + 1);
    if (begin == std::string::npos || end == std::string::npos)
        throw std::runtime_error("FP16 manifest array " + key);
    return json.substr(begin + 1, end - begin - 1);
}

std::vector<uint64_t> uintArray(const std::string& json,
                                const std::string& key)
{
    std::stringstream stream(arrayBody(json, key));
    std::vector<uint64_t> result;
    while (stream) {
        stream >> std::ws;
        if (stream.peek() == EOF) break;
        uint64_t value = 0;
        if (!(stream >> value))
            throw std::runtime_error("FP16 manifest uint array " + key);
        result.push_back(value);
        stream >> std::ws;
        if (stream.peek() == ',') stream.get();
        else if (stream.peek() != EOF)
            throw std::runtime_error("FP16 manifest uint separator " + key);
    }
    return result;
}

std::vector<std::string> stringArray(const std::string& json,
                                     const std::string& key)
{
    const std::string body = arrayBody(json, key);
    std::vector<std::string> result;
    std::size_t position = 0;
    while ((position = body.find('"', position)) != std::string::npos) {
        const std::size_t end = body.find('"', position + 1);
        if (end == std::string::npos)
            throw std::runtime_error("FP16 manifest string array " + key);
        result.push_back(body.substr(position + 1, end - position - 1));
        position = end + 1;
    }
    return result;
}

template <typename T>
void writeArray(std::ostream& output, const std::vector<T>& values)
{
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) output << ',';
        output << values[index];
    }
    output << ']';
}

void writeStringArray(std::ostream& output,
                      const std::vector<std::string>& values)
{
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) output << ',';
        output << '"' << values[index] << '"';
    }
    output << ']';
}

CSCFp16Bits convertAndCount(double source, CSCFp16ConversionStats& stats)
{
    const CSCFp16Bits bits = cscFp16ToBits(cscFp16FromDouble(source));
    ++stats.conversion_count;
    const uint16_t magnitude = bits & 0x7fffU;
    const uint16_t exponent = bits & 0x7c00U;
    const uint16_t fraction = bits & 0x03ffU;
    if (exponent == 0x7c00U) {
        if (fraction) ++stats.nan_count;
        else if (bits & 0x8000U) ++stats.negative_infinity_count;
        else ++stats.positive_infinity_count;
    } else if (!exponent && fraction) {
        ++stats.subnormal_count;
    }
    if (!magnitude && source != 0.0 && std::isfinite(source))
        ++stats.underflow_to_zero_count;
    if (!magnitude && std::signbit(source)) ++stats.signed_zero_count;
    return bits;
}

void appendDescriptor(std::vector<uint8_t>& bytes,
                      const CSCDescriptor& descriptor)
{
    appendU64(bytes, descriptor.value_offset_bytes);
    appendU64(bytes, descriptor.row_idx_offset_bytes);
    appendU32(bytes, descriptor.nnz_count);
    appendU32(bytes, descriptor.x_slot);
    appendU32(bytes, descriptor.original_col);
    appendU32(bytes, descriptor.global_bg_id);
}

CSCDescriptor readDescriptor(const std::vector<uint8_t>& bytes,
                             std::size_t offset)
{
    return {readU64(bytes, offset), readU64(bytes, offset + 8),
            readU32(bytes, offset + 16), readU32(bytes, offset + 20),
            readU32(bytes, offset + 24), readU32(bytes, offset + 28)};
}

void require64(std::size_t size, const char* label)
{
    if (size != kCSCFp16ImageBGCount)
        throw std::runtime_error(std::string("FP16 manifest requires 64 ") + label);
}

}  // namespace

void CSCFp16ImageSource::validate() const
{
    if (col_ptr.size() != static_cast<std::size_t>(cols) + 1 ||
        col_ptr.empty() || col_ptr.front() != 0 ||
        col_ptr.back() != values.size() ||
        values.size() != row_idx.size() || x.size() != cols ||
        column_to_bg.size() != cols)
        throw std::invalid_argument("invalid FP16 image source arrays");
    for (uint32_t column = 0; column < cols; ++column) {
        if (col_ptr[column] > col_ptr[column + 1])
            throw std::invalid_argument("non-monotonic FP16 source col_ptr");
        if (column_to_bg[column] >= kCSCFp16ImageBGCount)
            throw std::invalid_argument("FP16 source BG outside [0,63]");
    }
    for (const uint32_t row : row_idx)
        if (row >= rows)
            throw std::invalid_argument("FP16 source row outside matrix");
}

uint64_t cscFp16ImageFnv1a64(const std::vector<uint8_t>& bytes)
{
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (const uint8_t byte : bytes) {
        hash ^= byte;
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

void exportCSCFp16ImageV2(const CSCFp16ImageSource& source,
                         const std::string& directory)
{
    source.validate();
    const fs::path root(directory);
    if (fs::exists(root))
        throw std::runtime_error("FP16 image output directory already exists");
    if (!fs::create_directories(root))
        throw std::runtime_error("cannot create FP16 image directory");

    try {
        std::array<CSCFp16ImageBG, kCSCFp16ImageBGCount> bg;
        CSCFp16ConversionStats conversion;
        CSCFp16ImageAccounting accounting;
        std::array<uint32_t, kCSCFp16ImageBGCount> next_x_slot{};

        for (uint32_t column = 0; column < source.cols; ++column) {
            const uint64_t begin = source.col_ptr[column];
            const uint64_t count = source.col_ptr[column + 1] - begin;
            if (!count) continue;
            if (count > UINT32_MAX)
                throw std::runtime_error("FP16 column NNZ exceeds descriptor");
            const uint32_t owner = source.column_to_bg[column];
            auto& image = bg[owner];
            image.values.resize(align32(image.values.size()), 0);
            image.row_indices.resize(align32(image.row_indices.size()), 0);
            const uint64_t value_offset = image.values.size();
            const uint64_t row_offset = image.row_indices.size();
            for (uint64_t index = begin; index < begin + count; ++index) {
                cscAppendU16LE(image.values,
                    convertAndCount(source.values[index], conversion));
                appendU32(image.row_indices, source.row_idx[index]);
            }
            image.values.resize(align32(image.values.size()), 0);
            image.row_indices.resize(align32(image.row_indices.size()), 0);
            const CSCDescriptor descriptor{
                value_offset, row_offset, static_cast<uint32_t>(count),
                next_x_slot[owner]++, column, owner};
            image.parsed_descriptors.push_back(descriptor);
            appendDescriptor(image.descriptors, descriptor);
            appendU32(image.x_permutation, column);
        }

        std::vector<uint8_t> x_bytes;
        x_bytes.reserve(align32(uint64_t(source.cols) * 2));
        for (const double value : source.x)
            cscAppendU16LE(x_bytes, convertAndCount(value, conversion));
        x_bytes.resize(align32(x_bytes.size()), 0);

        accounting.logical_value_bytes = uint64_t(source.values.size()) * 2;
        accounting.logical_index_bytes = uint64_t(source.row_idx.size()) * 4;
        accounting.logical_x_bytes = uint64_t(source.x.size()) * 2;
        accounting.physical_x_bytes = x_bytes.size();
        accounting.x_padding_bytes = accounting.physical_x_bytes -
                                     accounting.logical_x_bytes;
        for (const auto& image : bg) {
            accounting.descriptor_count += image.parsed_descriptors.size();
            accounting.physical_value_bytes += image.values.size();
            accounting.physical_index_bytes += image.row_indices.size();
        }
        accounting.value_padding_bytes = accounting.physical_value_bytes -
                                         accounting.logical_value_bytes;
        accounting.index_padding_bytes = accounting.physical_index_bytes -
                                         accounting.logical_index_bytes;
        accounting.descriptor_bytes = accounting.descriptor_count * 32;

        std::vector<uint64_t> descriptor_counts, value_sizes, index_sizes;
        std::vector<uint64_t> descriptor_sizes, permutation_sizes;
        std::vector<std::string> value_hashes, index_hashes;
        std::vector<std::string> descriptor_hashes, permutation_hashes;
        for (uint32_t owner = 0; owner < kCSCFp16ImageBGCount; ++owner) {
            const std::string name = prefix(owner);
            writeBytes(root / (name + "_values_fp16.bin"), bg[owner].values);
            writeBytes(root / (name + "_row_idx_u32.bin"), bg[owner].row_indices);
            writeBytes(root / (name + "_descriptors.bin"), bg[owner].descriptors);
            writeBytes(root / (name + "_x_permutation.bin"), bg[owner].x_permutation);
            descriptor_counts.push_back(bg[owner].parsed_descriptors.size());
            value_sizes.push_back(bg[owner].values.size());
            index_sizes.push_back(bg[owner].row_indices.size());
            descriptor_sizes.push_back(bg[owner].descriptors.size());
            permutation_sizes.push_back(bg[owner].x_permutation.size());
            value_hashes.push_back(hex64(cscFp16ImageFnv1a64(bg[owner].values)));
            index_hashes.push_back(hex64(cscFp16ImageFnv1a64(bg[owner].row_indices)));
            descriptor_hashes.push_back(hex64(cscFp16ImageFnv1a64(bg[owner].descriptors)));
            permutation_hashes.push_back(hex64(cscFp16ImageFnv1a64(bg[owner].x_permutation)));
        }
        writeBytes(root / "x_fp16.bin", x_bytes);

        // Reopen and verify every file before publishing the manifest.
        const auto verify_file = [&](const fs::path& path,
                                     const std::vector<uint8_t>& expected,
                                     const std::string& expected_hash) {
            const auto actual = readBytes(path);
            if (actual.size() != expected.size() ||
                hex64(cscFp16ImageFnv1a64(actual)) != expected_hash)
                throw std::runtime_error(
                    "FP16 export size/checksum verification failed");
        };
        for (uint32_t owner = 0; owner < kCSCFp16ImageBGCount; ++owner) {
            const std::string name = prefix(owner);
            verify_file(root / (name + "_values_fp16.bin"),
                        bg[owner].values, value_hashes[owner]);
            verify_file(root / (name + "_row_idx_u32.bin"),
                        bg[owner].row_indices, index_hashes[owner]);
            verify_file(root / (name + "_descriptors.bin"),
                        bg[owner].descriptors, descriptor_hashes[owner]);
            verify_file(root / (name + "_x_permutation.bin"),
                        bg[owner].x_permutation, permutation_hashes[owner]);
        }
        const std::string x_hash = hex64(cscFp16ImageFnv1a64(x_bytes));
        verify_file(root / "x_fp16.bin", x_bytes, x_hash);

        std::ofstream manifest(root / "manifest.json", std::ios::trunc);
        if (!manifest) throw std::runtime_error("cannot publish FP16 manifest");
        manifest << "{\n"
                 << "  \"format_magic\":\"SPCSCIMG\",\n"
                 << "  \"format_version\":2,\n"
                 << "  \"endianness\":\"little-endian\",\n"
                 << "  \"precision\":\"FP16\",\n"
                 << "  \"storage_value_type\":\"IEEE-754 binary16\",\n"
                 << "  \"value_bytes\":2,\n"
                 << "  \"row_index_type\":\"uint32\",\n"
                 << "  \"row_index_bytes\":4,\n"
                 << "  \"burst_bytes\":32,\n"
                 << "  \"future_simd_width\":16,\n"
                 << "  \"value_elements_per_burst\":16,\n"
                 << "  \"row_indices_per_burst\":8,\n"
                 << "  \"descriptor_bytes\":32,\n"
                 << "  \"num_global_bgs\":64,\n"
                 << "  \"matrix_rows\":" << source.rows << ",\n"
                 << "  \"matrix_columns\":" << source.cols << ",\n"
                 << "  \"matrix_nnz\":" << source.values.size() << ",\n"
                 << "  \"descriptor_count\":" << accounting.descriptor_count << ",\n"
                 << "  \"logical_value_bytes\":" << accounting.logical_value_bytes << ",\n"
                 << "  \"physical_value_bytes\":" << accounting.physical_value_bytes << ",\n"
                 << "  \"value_padding_bytes\":" << accounting.value_padding_bytes << ",\n"
                 << "  \"logical_index_bytes\":" << accounting.logical_index_bytes << ",\n"
                 << "  \"physical_index_bytes\":" << accounting.physical_index_bytes << ",\n"
                 << "  \"index_padding_bytes\":" << accounting.index_padding_bytes << ",\n"
                 << "  \"logical_x_bytes\":" << accounting.logical_x_bytes << ",\n"
                 << "  \"physical_x_bytes\":" << accounting.physical_x_bytes << ",\n"
                 << "  \"x_padding_bytes\":" << accounting.x_padding_bytes << ",\n"
                 << "  \"descriptor_physical_bytes\":" << accounting.descriptor_bytes << ",\n"
                 << "  \"fp16_conversion_count\":" << conversion.conversion_count << ",\n"
                 << "  \"positive_infinity_count\":" << conversion.positive_infinity_count << ",\n"
                 << "  \"negative_infinity_count\":" << conversion.negative_infinity_count << ",\n"
                 << "  \"nan_count\":" << conversion.nan_count << ",\n"
                 << "  \"subnormal_count\":" << conversion.subnormal_count << ",\n"
                 << "  \"underflow_to_zero_count\":" << conversion.underflow_to_zero_count << ",\n"
                 << "  \"signed_zero_count\":" << conversion.signed_zero_count << ",\n"
                 << "  \"x_file_bytes\":" << x_bytes.size() << ",\n"
                 << "  \"x_fnv1a64\":\"" << x_hash << "\",\n";
        manifest << "  \"column_to_bg\":"; writeArray(manifest, source.column_to_bg); manifest << ",\n";
        manifest << "  \"bg_descriptor_counts\":"; writeArray(manifest, descriptor_counts); manifest << ",\n";
        manifest << "  \"bg_value_stream_bytes\":"; writeArray(manifest, value_sizes); manifest << ",\n";
        manifest << "  \"bg_row_index_stream_bytes\":"; writeArray(manifest, index_sizes); manifest << ",\n";
        manifest << "  \"bg_descriptor_stream_bytes\":"; writeArray(manifest, descriptor_sizes); manifest << ",\n";
        manifest << "  \"bg_x_permutation_stream_bytes\":"; writeArray(manifest, permutation_sizes); manifest << ",\n";
        manifest << "  \"bg_values_fnv1a64\":"; writeStringArray(manifest, value_hashes); manifest << ",\n";
        manifest << "  \"bg_row_idx_fnv1a64\":"; writeStringArray(manifest, index_hashes); manifest << ",\n";
        manifest << "  \"bg_descriptors_fnv1a64\":"; writeStringArray(manifest, descriptor_hashes); manifest << ",\n";
        manifest << "  \"bg_x_permutation_fnv1a64\":"; writeStringArray(manifest, permutation_hashes); manifest << "\n}\n";
        if (!manifest) throw std::runtime_error("failed publishing FP16 manifest");
    } catch (...) {
        fs::remove_all(root);
        throw;
    }
}

CSCFp16LoadedImage loadCSCFp16ImageV2(const std::string& directory)
{
    const fs::path root(directory);
    const std::string json = readText(root / "manifest.json");
    if (stringValue(json, "format_magic") != "SPCSCIMG" ||
        uintValue(json, "format_version") != kCSCFp16ImageVersion ||
        stringValue(json, "endianness") != "little-endian" ||
        stringValue(json, "precision") != "FP16" ||
        stringValue(json, "storage_value_type") != "IEEE-754 binary16" ||
        uintValue(json, "value_bytes") != kCSCFp16ImageValueBytes ||
        stringValue(json, "row_index_type") != "uint32" ||
        uintValue(json, "row_index_bytes") != kCSCFp16ImageIndexBytes ||
        uintValue(json, "burst_bytes") != kCSCFp16ImageBurstBytes ||
        uintValue(json, "future_simd_width") != kCSCFp16ImageFutureSIMDWidth ||
        uintValue(json, "value_elements_per_burst") != 16 ||
        uintValue(json, "row_indices_per_burst") != 8 ||
        uintValue(json, "descriptor_bytes") != kCSCFp16ImageDescriptorBytes ||
        uintValue(json, "num_global_bgs") != kCSCFp16ImageBGCount)
        throw std::runtime_error("unsupported FP16 CSC image contract");

    CSCFp16LoadedImage loaded;
    loaded.matrix.rows = uintValue(json, "matrix_rows");
    loaded.matrix.cols = uintValue(json, "matrix_columns");
    const uint64_t nnz = uintValue(json, "matrix_nnz");
    loaded.accounting.descriptor_count = uintValue(json, "descriptor_count");
    loaded.accounting.logical_value_bytes = uintValue(json, "logical_value_bytes");
    loaded.accounting.physical_value_bytes = uintValue(json, "physical_value_bytes");
    loaded.accounting.value_padding_bytes = uintValue(json, "value_padding_bytes");
    loaded.accounting.logical_index_bytes = uintValue(json, "logical_index_bytes");
    loaded.accounting.physical_index_bytes = uintValue(json, "physical_index_bytes");
    loaded.accounting.index_padding_bytes = uintValue(json, "index_padding_bytes");
    loaded.accounting.logical_x_bytes = uintValue(json, "logical_x_bytes");
    loaded.accounting.physical_x_bytes = uintValue(json, "physical_x_bytes");
    loaded.accounting.x_padding_bytes = uintValue(json, "x_padding_bytes");
    loaded.accounting.descriptor_bytes = uintValue(json, "descriptor_physical_bytes");
    loaded.conversion.conversion_count = uintValue(json, "fp16_conversion_count");
    loaded.conversion.positive_infinity_count = uintValue(json, "positive_infinity_count");
    loaded.conversion.negative_infinity_count = uintValue(json, "negative_infinity_count");
    loaded.conversion.nan_count = uintValue(json, "nan_count");
    loaded.conversion.subnormal_count = uintValue(json, "subnormal_count");
    loaded.conversion.underflow_to_zero_count = uintValue(json, "underflow_to_zero_count");
    loaded.conversion.signed_zero_count = uintValue(json, "signed_zero_count");
    if (loaded.conversion.positive_infinity_count > loaded.conversion.conversion_count ||
        loaded.conversion.negative_infinity_count > loaded.conversion.conversion_count ||
        loaded.conversion.nan_count > loaded.conversion.conversion_count ||
        loaded.conversion.subnormal_count > loaded.conversion.conversion_count ||
        loaded.conversion.underflow_to_zero_count > loaded.conversion.conversion_count ||
        loaded.conversion.signed_zero_count > loaded.conversion.conversion_count)
        throw std::runtime_error("FP16 conversion statistics invariant");

    const auto owners64 = uintArray(json, "column_to_bg");
    const auto descriptor_counts = uintArray(json, "bg_descriptor_counts");
    const auto value_sizes = uintArray(json, "bg_value_stream_bytes");
    const auto index_sizes = uintArray(json, "bg_row_index_stream_bytes");
    const auto descriptor_sizes = uintArray(json, "bg_descriptor_stream_bytes");
    const auto permutation_sizes = uintArray(json, "bg_x_permutation_stream_bytes");
    const auto value_hashes = stringArray(json, "bg_values_fnv1a64");
    const auto index_hashes = stringArray(json, "bg_row_idx_fnv1a64");
    const auto descriptor_hashes = stringArray(json, "bg_descriptors_fnv1a64");
    const auto permutation_hashes = stringArray(json, "bg_x_permutation_fnv1a64");
    require64(descriptor_counts.size(), "descriptor counts");
    require64(value_sizes.size(), "value sizes");
    require64(index_sizes.size(), "index sizes");
    require64(descriptor_sizes.size(), "descriptor sizes");
    require64(permutation_sizes.size(), "permutation sizes");
    require64(value_hashes.size(), "value checksums");
    require64(index_hashes.size(), "index checksums");
    require64(descriptor_hashes.size(), "descriptor checksums");
    require64(permutation_hashes.size(), "permutation checksums");
    if (owners64.size() != loaded.matrix.cols)
        throw std::runtime_error("FP16 column owner count mismatch");
    for (const uint64_t owner : owners64) {
        if (owner >= kCSCFp16ImageBGCount)
            throw std::runtime_error("FP16 manifest BG outside [0,63]");
        loaded.column_to_bg.push_back(owner);
    }

    std::vector<std::vector<CSCFp16Bits>> column_values(loaded.matrix.cols);
    std::vector<std::vector<uint32_t>> column_rows(loaded.matrix.cols);
    std::vector<uint8_t> seen(loaded.matrix.cols, 0);
    uint64_t descriptor_total = 0;
    uint64_t physical_values = 0, physical_indices = 0;
    for (uint32_t owner = 0; owner < kCSCFp16ImageBGCount; ++owner) {
        auto& image = loaded.bg[owner];
        const std::string name = prefix(owner);
        image.values = readBytes(root / (name + "_values_fp16.bin"));
        image.row_indices = readBytes(root / (name + "_row_idx_u32.bin"));
        image.descriptors = readBytes(root / (name + "_descriptors.bin"));
        image.x_permutation = readBytes(root / (name + "_x_permutation.bin"));
        if (image.values.size() != value_sizes[owner] ||
            image.row_indices.size() != index_sizes[owner] ||
            image.descriptors.size() != descriptor_sizes[owner] ||
            image.x_permutation.size() != permutation_sizes[owner] ||
            image.descriptors.size() != descriptor_counts[owner] * 32 ||
            image.x_permutation.size() != descriptor_counts[owner] * 4)
            throw std::runtime_error("FP16 image file size invariant");
        if (hex64(cscFp16ImageFnv1a64(image.values)) != value_hashes[owner] ||
            hex64(cscFp16ImageFnv1a64(image.row_indices)) != index_hashes[owner] ||
            hex64(cscFp16ImageFnv1a64(image.descriptors)) != descriptor_hashes[owner] ||
            hex64(cscFp16ImageFnv1a64(image.x_permutation)) != permutation_hashes[owner])
            throw std::runtime_error("FP16 image checksum invariant");
        if ((!image.values.empty() && image.values.size() % 32) ||
            (!image.row_indices.empty() && image.row_indices.size() % 32))
            throw std::runtime_error("FP16 stream size alignment invariant");

        std::vector<uint8_t> used_values(image.values.size(), 0);
        std::vector<uint8_t> used_indices(image.row_indices.size(), 0);
        uint64_t expected_value_offset = 0;
        uint64_t expected_index_offset = 0;
        for (uint64_t index = 0; index < descriptor_counts[owner]; ++index) {
            const CSCDescriptor descriptor =
                readDescriptor(image.descriptors, index * 32);
            if (descriptor.global_bg_id != owner || !descriptor.nnz_count ||
                descriptor.original_col >= loaded.matrix.cols ||
                loaded.column_to_bg[descriptor.original_col] != owner ||
                descriptor.x_slot != index ||
                readU32(image.x_permutation, index * 4) != descriptor.original_col ||
                descriptor.value_offset_bytes != expected_value_offset ||
                descriptor.row_idx_offset_bytes != expected_index_offset ||
                descriptor.value_offset_bytes % 32 ||
                descriptor.row_idx_offset_bytes % 32 ||
                seen[descriptor.original_col]++)
                throw std::runtime_error("FP16 descriptor invariant");
            const uint64_t value_end = descriptor.value_offset_bytes +
                                       uint64_t(descriptor.nnz_count) * 2;
            const uint64_t index_end = descriptor.row_idx_offset_bytes +
                                       uint64_t(descriptor.nnz_count) * 4;
            if (value_end > image.values.size() ||
                index_end > image.row_indices.size())
                throw std::runtime_error("FP16 descriptor payload range");
            for (uint32_t lane = 0; lane < descriptor.nnz_count; ++lane) {
                const uint64_t value_offset = descriptor.value_offset_bytes +
                                              uint64_t(lane) * 2;
                const uint64_t row_offset = descriptor.row_idx_offset_bytes +
                                            uint64_t(lane) * 4;
                const uint32_t row = readU32(image.row_indices, row_offset);
                if (row >= loaded.matrix.rows)
                    throw std::runtime_error("FP16 row index outside matrix");
                column_values[descriptor.original_col].push_back(
                    cscReadU16LE(image.values, value_offset));
                column_rows[descriptor.original_col].push_back(row);
                used_values[value_offset] = used_values[value_offset + 1] = 1;
                for (uint32_t byte = 0; byte < 4; ++byte)
                    used_indices[row_offset + byte] = 1;
            }
            expected_value_offset = align32(value_end);
            expected_index_offset = align32(index_end);
            image.parsed_descriptors.push_back(descriptor);
            ++descriptor_total;
        }
        if (expected_value_offset != image.values.size() ||
            expected_index_offset != image.row_indices.size())
            throw std::runtime_error("FP16 expected stream size invariant");
        for (std::size_t index = 0; index < image.values.size(); ++index)
            if (!used_values[index] && image.values[index])
                throw std::runtime_error("FP16 value padding is not zero");
        for (std::size_t index = 0; index < image.row_indices.size(); ++index)
            if (!used_indices[index] && image.row_indices[index])
                throw std::runtime_error("FP16 index padding is not zero");
        physical_values += image.values.size();
        physical_indices += image.row_indices.size();
    }

    loaded.matrix.col_ptr.push_back(0);
    for (uint32_t column = 0; column < loaded.matrix.cols; ++column) {
        loaded.matrix.value_bits.insert(loaded.matrix.value_bits.end(),
            column_values[column].begin(), column_values[column].end());
        loaded.matrix.row_idx.insert(loaded.matrix.row_idx.end(),
            column_rows[column].begin(), column_rows[column].end());
        loaded.matrix.col_ptr.push_back(loaded.matrix.value_bits.size());
    }
    loaded.matrix.validate();
    if (loaded.matrix.value_bits.size() != nnz ||
        descriptor_total != loaded.accounting.descriptor_count ||
        physical_values != loaded.accounting.physical_value_bytes ||
        physical_indices != loaded.accounting.physical_index_bytes ||
        loaded.accounting.logical_value_bytes != nnz * 2 ||
        loaded.accounting.logical_index_bytes != nnz * 4 ||
        loaded.accounting.value_padding_bytes !=
            physical_values - loaded.accounting.logical_value_bytes ||
        loaded.accounting.index_padding_bytes !=
            physical_indices - loaded.accounting.logical_index_bytes ||
        loaded.accounting.descriptor_bytes != descriptor_total * 32)
        throw std::runtime_error("FP16 manifest accounting invariant");

    const auto x_bytes = readBytes(root / "x_fp16.bin");
    if (x_bytes.size() != uintValue(json, "x_file_bytes") ||
        x_bytes.size() != loaded.accounting.physical_x_bytes ||
        hex64(cscFp16ImageFnv1a64(x_bytes)) != stringValue(json, "x_fnv1a64") ||
        x_bytes.size() != align32(uint64_t(loaded.matrix.cols) * 2))
        throw std::runtime_error("FP16 x file invariant");
    for (uint32_t column = 0; column < loaded.matrix.cols; ++column)
        loaded.x_bits.push_back(cscReadU16LE(x_bytes, uint64_t(column) * 2));
    for (uint64_t offset = uint64_t(loaded.matrix.cols) * 2;
         offset < x_bytes.size(); ++offset)
        if (x_bytes[offset]) throw std::runtime_error("FP16 x padding is not zero");
    if (loaded.accounting.logical_x_bytes != uint64_t(loaded.matrix.cols) * 2 ||
        loaded.accounting.x_padding_bytes !=
            loaded.accounting.physical_x_bytes - loaded.accounting.logical_x_bytes ||
        loaded.conversion.conversion_count != nnz + loaded.matrix.cols)
        throw std::runtime_error("FP16 x/conversion accounting invariant");

    return loaded;
}

}  // namespace csc_descriptor
