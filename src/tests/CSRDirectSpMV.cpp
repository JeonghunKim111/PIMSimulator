#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "AddressMapping.h"
#include "Burst.h"
#include "MultiChannelMemorySystem.h"
#include "PIMCmd.h"
#include "gtest/gtest.h"
#include "tests/PIMKernel.h"

using namespace DRAMSim;
using namespace std;
namespace fs = std::filesystem;

namespace csr_direct
{
namespace
{
constexpr uint32_t kDescriptorBaseRow = 0;
constexpr uint32_t kValueBaseRow = 1024;
constexpr uint32_t kIndexBaseRow = 4096;
constexpr uint32_t kXBaseRow = 7168;
constexpr uint32_t kOutputBaseRow = 8192;
constexpr uint32_t kValueBytes = 4;
constexpr uint32_t kColumnIndexBytes = 4;
constexpr uint32_t kXBytes = 4;
constexpr uint32_t kOutputBytes = 4;
constexpr uint32_t kRowDescriptorBytes = 16;
constexpr const char* kMappingName = "direct_row_round_robin";

enum class CSRMappingPolicy
{
    DirectRowRoundRobin,
    NNZBalanced, // Reserved for a future preprocessing comparison.
};

struct CSRMatrix
{
    uint32_t rows = 0;
    uint32_t cols = 0;
    vector<uint64_t> row_ptr;
    vector<uint32_t> col_idx;
    vector<float> values;
};

struct RowDescriptor
{
    uint32_t global_row_id = 0;
    uint64_t local_value_start = 0;
    uint32_t nnz = 0;
};

struct CSRBGShard
{
    vector<float> values;
    vector<uint32_t> col_indices;
    vector<RowDescriptor> rows;
};

struct InputResult
{
    CSRMatrix matrix;
    double input_load_us = 0.0;
};

struct MappingResult
{
    vector<uint64_t> row_to_bg;
    vector<CSRBGShard> shards;
    double direct_mapping_us = 0.0;
    double metadata_generation_us = 0.0;
    double bg_sharding_us = 0.0;
};

struct LoadStats
{
    uint64_t nonempty_rows = 0;
    uint64_t empty_rows = 0;
    uint64_t active_bank_groups = 0;
    uint64_t rows_per_bg_min = 0;
    uint64_t rows_per_bg_max = 0;
    double rows_per_bg_avg = 0.0;
    uint64_t nnz_per_bg_min = 0;
    uint64_t nnz_per_bg_max = 0;
    double nnz_per_bg_avg = 0.0;
    double nnz_per_bg_stddev = 0.0;
    double bg_load_imbalance_all = 0.0;
    double bg_load_imbalance_active = 0.0;
    uint64_t critical_bg_id = 0;
    uint64_t critical_bg_nnz = 0;
    uint64_t longest_row_nnz = 0;
    double average_nonempty_row_nnz = 0.0;
};

struct TrafficStats
{
    uint64_t value_read_count = 0;
    uint64_t value_read_bytes = 0;
    uint64_t column_index_read_count = 0;
    uint64_t column_index_read_bytes = 0;
    uint64_t x_gather_count = 0;
    uint64_t x_gather_bytes = 0;
    uint64_t output_write_count = 0;
    uint64_t output_write_bytes = 0;
    uint64_t output_readback_count = 0;
    uint64_t output_readback_bytes = 0;
};

struct TimingResult
{
    uint64_t matrix_programming_cycle = 0;
    uint64_t x_transfer_cycle = 0;
    uint64_t pim_mode_change_cycle = 0;
    uint64_t crf_programming_cycle = 0;
    uint64_t csr_execution_trace_cycle = 0;
    uint64_t output_write_cycle = 0;
    uint64_t output_readback_cycle = 0;
    uint64_t modeled_core_kernel_cycle = 0;
    uint64_t pim_execution_cycle = 0;
    uint64_t retrieval_inclusive_cycle = 0;
    uint64_t resident_iteration_cycle = 0;
    uint64_t total_pim_cycle = 0;
    double modeled_core_kernel_ms = 0.0;
    double resident_iteration_ms = 0.0;
    double pim_time_ms = 0.0;
};

struct Result
{
    string matrix_name;
    string input_path;
    CSRMatrix matrix;
    MappingResult mapping;
    LoadStats load;
    TrafficStats traffic;
    TimingResult timing;
    unsigned iterations = 0;
    bool validation_passed = false;
    string error_message;
    double input_load_us = 0.0;
    double end_to_end_ms = 0.0;
};

uint64_t ceilDiv(uint64_t n, uint64_t d) { return d == 0 ? 0 : (n + d - 1) / d; }

string envString(const char* name, const string& fallback = "")
{
    const char* value = getenv(name);
    return value == nullptr || *value == '\0' ? fallback : string(value);
}

unsigned envUnsigned(const char* name, unsigned fallback)
{
    string value = envString(name);
    return value.empty() ? fallback : static_cast<unsigned>(stoul(value));
}

string csvEscape(const string& text)
{
    string escaped = "\"";
    for (char c : text)
        escaped += c == '"' ? "\"\"" : string(1, c);
    return escaped + "\"";
}

string shellQuote(const string& text)
{
    string quoted = "'";
    for (char c : text)
        quoted += c == '\'' ? "'\\''" : string(1, c);
    return quoted + "'";
}

string matrixName(const fs::path& path)
{
    string name = path.filename().string();
    const string suffix = "_csr.npz";
    if (name.size() >= suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
        name.resize(name.size() - suffix.size());
    return name;
}

template <typename T>
void readExact(ifstream& input, vector<T>& output)
{
    if (!output.empty())
        input.read(reinterpret_cast<char*>(output.data()), output.size() * sizeof(T));
    if (!input)
        throw runtime_error("truncated CSR binary input");
}

InputResult loadCSRNPZ(const fs::path& npz_path)
{
    auto start = chrono::steady_clock::now();
    fs::path sidecar = fs::temp_directory_path() /
        ("csr_direct_" + to_string(hash<string>{}(npz_path.string())) + ".bin");
    string command = "python3 tools/export_csr_npz.py " + shellQuote(npz_path.string()) +
                     " " + shellQuote(sidecar.string());
    int status = system(command.c_str());
    if (status != 0)
        throw runtime_error("CSR NPZ exporter failed with status " + to_string(status));

    InputResult result;
    try
    {
        ifstream input(sidecar, ios::binary);
        if (!input)
            throw runtime_error("failed to open exported CSR binary");
        char magic[8]{};
        uint32_t rows = 0, cols = 0;
        uint64_t nnz = 0;
        input.read(magic, sizeof(magic));
        input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        input.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        input.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));
        if (!input || string(magic, 7) != "CSRDIR1")
            throw runtime_error("invalid CSR binary header");
        if (nnz > numeric_limits<size_t>::max())
            throw runtime_error("CSR nnz exceeds host size_t");
        result.matrix.rows = rows;
        result.matrix.cols = cols;
        result.matrix.row_ptr.resize(static_cast<size_t>(rows) + 1);
        result.matrix.col_idx.resize(static_cast<size_t>(nnz));
        result.matrix.values.resize(static_cast<size_t>(nnz));
        readExact(input, result.matrix.row_ptr);
        readExact(input, result.matrix.col_idx);
        readExact(input, result.matrix.values);
        if (result.matrix.row_ptr.front() != 0 || result.matrix.row_ptr.back() != nnz)
            throw runtime_error("invalid CSR indptr endpoints");
        for (uint32_t row = 0; row < rows; ++row)
            if (result.matrix.row_ptr[row] > result.matrix.row_ptr[row + 1])
                throw runtime_error("CSR indptr is not monotonic");
        for (uint32_t col : result.matrix.col_idx)
            if (col >= cols)
                throw runtime_error("CSR column index exceeds shape");
        fs::remove(sidecar);
    }
    catch (...)
    {
        fs::remove(sidecar);
        throw;
    }
    result.input_load_us = chrono::duration<double, micro>(
        chrono::steady_clock::now() - start).count();
    return result;
}

MappingResult buildMapping(const CSRMatrix& matrix, uint64_t global_bank_groups,
                           CSRMappingPolicy policy = CSRMappingPolicy::DirectRowRoundRobin)
{
    if (policy != CSRMappingPolicy::DirectRowRoundRobin)
        throw invalid_argument("only DirectRowRoundRobin is enabled");
    if (global_bank_groups == 0)
        throw invalid_argument("global bank group count must be nonzero");
    MappingResult result;
    result.shards.resize(global_bank_groups);
    result.row_to_bg.resize(matrix.rows);

    auto mapping_start = chrono::steady_clock::now();
    for (uint32_t row = 0; row < matrix.rows; ++row)
        result.row_to_bg[row] = row % global_bank_groups;
    result.direct_mapping_us = chrono::duration<double, micro>(
        chrono::steady_clock::now() - mapping_start).count();

    auto metadata_start = chrono::steady_clock::now();
    vector<uint64_t> bg_nnz(global_bank_groups, 0);
    for (uint32_t row = 0; row < matrix.rows; ++row)
    {
        uint64_t bg = result.row_to_bg[row];
        uint64_t count = matrix.row_ptr[row + 1] - matrix.row_ptr[row];
        if (count > numeric_limits<uint32_t>::max())
            throw runtime_error("row nnz exceeds RowDescriptor capacity");
        result.shards[bg].rows.push_back(
            {row, bg_nnz[bg], static_cast<uint32_t>(count)});
        bg_nnz[bg] += count;
    }
    for (uint64_t bg = 0; bg < global_bank_groups; ++bg)
    {
        result.shards[bg].values.reserve(bg_nnz[bg]);
        result.shards[bg].col_indices.reserve(bg_nnz[bg]);
    }
    result.metadata_generation_us = chrono::duration<double, micro>(
        chrono::steady_clock::now() - metadata_start).count();

    auto sharding_start = chrono::steady_clock::now();
    for (uint32_t row = 0; row < matrix.rows; ++row)
    {
        CSRBGShard& shard = result.shards[result.row_to_bg[row]];
        for (uint64_t p = matrix.row_ptr[row]; p < matrix.row_ptr[row + 1]; ++p)
        {
            shard.values.push_back(matrix.values[p]);
            shard.col_indices.push_back(matrix.col_idx[p]);
        }
    }
    result.bg_sharding_us = chrono::duration<double, micro>(
        chrono::steady_clock::now() - sharding_start).count();

    uint64_t row_sum = 0, nnz_sum = 0;
    vector<uint8_t> owner_count(matrix.rows, 0);
    for (uint64_t bg = 0; bg < global_bank_groups; ++bg)
    {
        row_sum += result.shards[bg].rows.size();
        nnz_sum += result.shards[bg].values.size();
        for (const RowDescriptor& row : result.shards[bg].rows)
        {
            if (result.row_to_bg[row.global_row_id] != bg)
                throw logic_error("row assigned to wrong BG");
            owner_count[row.global_row_id]++;
        }
    }
    if (row_sum != matrix.rows || nnz_sum != matrix.values.size() ||
        any_of(owner_count.begin(), owner_count.end(), [](uint8_t n) { return n != 1; }))
        throw logic_error("CSR BG sharding invariant failed");
    return result;
}

vector<float> makeX(uint32_t cols)
{
    vector<float> x(cols);
    for (uint32_t col = 0; col < cols; ++col)
        x[col] = static_cast<float>((col % 17) + 1) / 17.0f;
    return x;
}

vector<float> cpuReference(const CSRMatrix& matrix, const vector<float>& x)
{
    vector<float> y(matrix.rows, 0.0f);
    for (uint32_t row = 0; row < matrix.rows; ++row)
    {
        float sum = 0.0f;
        for (uint64_t p = matrix.row_ptr[row]; p < matrix.row_ptr[row + 1]; ++p)
            sum += matrix.values[p] * x[matrix.col_idx[p]];
        y[row] = sum;
    }
    return y;
}

vector<float> functionalSharded(const CSRMatrix& matrix, const MappingResult& mapping,
                                const vector<float>& x)
{
    vector<float> y(matrix.rows, 0.0f);
    for (const CSRBGShard& shard : mapping.shards)
        for (const RowDescriptor& row : shard.rows)
        {
            float sum = 0.0f;
            for (uint64_t p = row.local_value_start;
                 p < row.local_value_start + row.nnz; ++p)
                sum += shard.values[p] * x[shard.col_indices[p]];
            y[row.global_row_id] = sum;
        }
    return y;
}

bool almostEqual(const vector<float>& lhs, const vector<float>& rhs)
{
    if (lhs.size() != rhs.size())
        return false;
    for (size_t i = 0; i < lhs.size(); ++i)
    {
        float diff = fabs(lhs[i] - rhs[i]);
        float scale = max(fabs(lhs[i]), fabs(rhs[i]));
        if (diff > 1.0e-5f + 1.0e-5f * scale)
            return false;
    }
    return true;
}

LoadStats buildLoadStats(const CSRMatrix& matrix, const MappingResult& mapping)
{
    LoadStats stats;
    vector<uint64_t> rows_per_bg, nnz_per_bg;
    rows_per_bg.reserve(mapping.shards.size());
    nnz_per_bg.reserve(mapping.shards.size());
    uint64_t nonempty_nnz = 0;
    for (const CSRBGShard& shard : mapping.shards)
    {
        rows_per_bg.push_back(shard.rows.size());
        nnz_per_bg.push_back(shard.values.size());
        if (!shard.rows.empty())
            stats.active_bank_groups++;
        for (const RowDescriptor& row : shard.rows)
        {
            stats.longest_row_nnz = max<uint64_t>(stats.longest_row_nnz, row.nnz);
            if (row.nnz == 0)
                stats.empty_rows++;
            else
            {
                stats.nonempty_rows++;
                nonempty_nnz += row.nnz;
            }
        }
    }
    stats.rows_per_bg_min = *min_element(rows_per_bg.begin(), rows_per_bg.end());
    stats.rows_per_bg_max = *max_element(rows_per_bg.begin(), rows_per_bg.end());
    stats.rows_per_bg_avg = static_cast<double>(matrix.rows) / mapping.shards.size();
    stats.nnz_per_bg_min = *min_element(nnz_per_bg.begin(), nnz_per_bg.end());
    auto critical = max_element(nnz_per_bg.begin(), nnz_per_bg.end());
    stats.critical_bg_id = distance(nnz_per_bg.begin(), critical);
    stats.critical_bg_nnz = stats.nnz_per_bg_max = *critical;
    stats.nnz_per_bg_avg = static_cast<double>(matrix.values.size()) / mapping.shards.size();
    double variance = 0.0;
    for (uint64_t count : nnz_per_bg)
        variance += pow(static_cast<double>(count) - stats.nnz_per_bg_avg, 2.0);
    stats.nnz_per_bg_stddev = sqrt(variance / mapping.shards.size());
    stats.bg_load_imbalance_all = stats.nnz_per_bg_avg == 0.0
        ? 0.0 : stats.nnz_per_bg_max / stats.nnz_per_bg_avg;
    uint64_t active_nnz_groups = count_if(nnz_per_bg.begin(), nnz_per_bg.end(),
                                          [](uint64_t n) { return n != 0; });
    double active_avg = active_nnz_groups == 0 ? 0.0
        : static_cast<double>(matrix.values.size()) / active_nnz_groups;
    stats.bg_load_imbalance_active = active_avg == 0.0 ? 0.0
        : stats.nnz_per_bg_max / active_avg;
    stats.average_nonempty_row_nnz = stats.nonempty_rows == 0 ? 0.0
        : static_cast<double>(nonempty_nnz) / stats.nonempty_rows;
    return stats;
}

TrafficStats buildTrafficStats(const CSRMatrix& matrix)
{
    TrafficStats stats;
    uint64_t nnz = matrix.values.size();
    stats.value_read_count = nnz;
    stats.value_read_bytes = nnz * kValueBytes;
    stats.column_index_read_count = nnz;
    stats.column_index_read_bytes = nnz * kColumnIndexBytes;
    stats.x_gather_count = nnz;
    stats.x_gather_bytes = nnz * kXBytes;
    stats.output_write_count = matrix.rows;
    stats.output_write_bytes = static_cast<uint64_t>(matrix.rows) * kOutputBytes;
    stats.output_readback_count = matrix.rows;
    stats.output_readback_bytes = static_cast<uint64_t>(matrix.rows) * kOutputBytes;
    return stats;
}

uint64_t medianCycles(vector<uint64_t> samples)
{
    sort(samples.begin(), samples.end());
    size_t mid = samples.size() / 2;
    return samples.size() % 2 == 0 ? (samples[mid - 1] + samples[mid]) / 2
                                   : samples[mid];
}

class TimingSimulator
{
  public:
    TimingSimulator()
    {
        // Model one HBM2 stack: 4 bank groups per pseudo-channel,
        // 4 pseudo-channels per die, and 4 dies per stack. PIMSimulator
        // represents each pseudo-channel as an independent channel, so one
        // stack has 16 channels. Allocate one 256 MiB rank per channel.
        mem_ = make_shared<MultiChannelMemorySystem>("ini/HBM2_samsung_2M_16B_x64.ini",
                                                     "system_hbm.ini", ".",
                                                     "csr_direct", 256 * 16);
        channels_ = getConfigParam(UINT, "NUM_CHANS");
        ranks_ = getConfigParam(UINT, "NUM_RANKS");
        kernel_ = make_shared<PIMKernel>(mem_, channels_, ranks_);
        bank_groups_ = kernel_->pim_addr_mgr_->num_bank_groups_;
        banks_ = kernel_->pim_addr_mgr_->num_banks_;
        transaction_bytes_ = kernel_->transaction_size_;
        global_bank_groups_ = static_cast<uint64_t>(channels_) * ranks_ * bank_groups_;
    }

    uint64_t globalBankGroups() const { return global_bank_groups_; }

    bool validateRepresentativeMappings(uint32_t rows, bool debug = false) const
    {
        if (rows == 0)
            return true;
        set<uint32_t> representatives{0, min<uint32_t>(1, rows - 1),
                                      min<uint32_t>(rows - 1, global_bank_groups_ - 1),
                                      rows - 1};
        for (uint32_t row_id : representatives)
        {
            uint64_t expected_bg = row_id % global_bank_groups_;
            BgAddress expected = decodeBg(expected_bg);
            uint64_t physical = address(expected_bg, 0, kDescriptorBaseRow, row_id /
                                        global_bank_groups_);
            BgAddress decoded = decodePhysical(physical);
            if (debug)
                cout << "CSR_BG_MAP row=" << row_id << " global_bg=" << expected_bg
                     << " channel=" << expected.channel << " rank=" << expected.rank
                     << " bank_group=" << expected.bank_group << " address=" << physical
                     << " decoded_channel=" << decoded.channel
                     << " decoded_rank=" << decoded.rank
                     << " decoded_bank_group=" << decoded.bank_group << endl;
            if (decoded.channel != expected.channel || decoded.rank != expected.rank ||
                decoded.bank_group != expected.bank_group)
                return false;
        }
        return true;
    }

    TimingResult run(const CSRMatrix& matrix, const MappingResult& mapping,
                     unsigned iterations)
    {
        if (mapping.shards.size() != global_bank_groups_)
            throw invalid_argument("mapping BG count differs from physical topology");
        if (!validateRepresentativeMappings(matrix.rows,
                                             envString("CSR_DIRECT_DEBUG_MAPPING") == "1"))
            throw runtime_error("physical BG address mapping validation failed");
        TimingResult timing;
        BurstType burst;

        issueMatrixProgramming(mapping.shards, burst);
        timing.matrix_programming_cycle = drain();

        vector<uint64_t> x_cycles, mode_cycles, crf_cycles, execution_cycles,
            output_cycles, readback_cycles;
        iterations = max(1u, iterations);
        for (unsigned iteration = 0; iteration < iterations + 1; ++iteration)
        {
            uint64_t x_cycle = issueXTransfer(matrix.cols, burst);
            uint64_t mode_cycle = 0;
            kernel_->configurePIMControl();
            kernel_->parkIn();
            kernel_->changePIMMode(dramMode::SB, dramMode::HAB);
            mode_cycle += drain();

            vector<PIMCmd> commands{
                PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A,
                       PIMOpdType::EVEN_BANK, 1),
                PIMCmd(PIMCmdType::NOP, 7),
                PIMCmd(PIMCmdType::EXIT, 0),
            };
            kernel_->programCrf(commands);
            uint64_t crf_cycle = drain();
            // Public PIM APIs cannot express independent variable-length CSR
            // rows or indexed x gather. Return to SB and issue an explicit
            // memory timing trace after programming the real MAC CRF.
            kernel_->changePIMMode(dramMode::HAB, dramMode::SB);
            kernel_->parkOut();
            mode_cycle += drain();

            issueExecutionTrace(mapping.shards, burst);
            uint64_t execution_cycle = drain();
            issueOutputTraffic(mapping.shards, true, burst);
            uint64_t output_cycle = drain();
            issueOutputTraffic(mapping.shards, false, burst);
            uint64_t readback_cycle = drain();

            if (iteration != 0) // warm-up is intentionally discarded
            {
                x_cycles.push_back(x_cycle);
                mode_cycles.push_back(mode_cycle);
                crf_cycles.push_back(crf_cycle);
                execution_cycles.push_back(execution_cycle);
                output_cycles.push_back(output_cycle);
                readback_cycles.push_back(readback_cycle);
            }
        }
        timing.x_transfer_cycle = medianCycles(x_cycles);
        timing.pim_mode_change_cycle = medianCycles(mode_cycles);
        timing.crf_programming_cycle = medianCycles(crf_cycles);
        timing.csr_execution_trace_cycle = medianCycles(execution_cycles);
        timing.output_write_cycle = medianCycles(output_cycles);
        timing.output_readback_cycle = medianCycles(readback_cycles);
        timing.modeled_core_kernel_cycle = timing.crf_programming_cycle +
            timing.csr_execution_trace_cycle + timing.output_write_cycle;
        timing.pim_execution_cycle = timing.pim_mode_change_cycle +
            timing.modeled_core_kernel_cycle;
        timing.retrieval_inclusive_cycle = timing.pim_execution_cycle +
            timing.output_readback_cycle;
        timing.resident_iteration_cycle = timing.x_transfer_cycle +
            timing.retrieval_inclusive_cycle;
        timing.total_pim_cycle = timing.matrix_programming_cycle +
            timing.resident_iteration_cycle;
        double tck_ns = getConfigParam(FLOAT, "tCK");
        timing.modeled_core_kernel_ms = timing.modeled_core_kernel_cycle * tck_ns / 1.0e6;
        timing.resident_iteration_ms = timing.resident_iteration_cycle * tck_ns / 1.0e6;
        timing.pim_time_ms = timing.total_pim_cycle * tck_ns / 1.0e6;
        return timing;
    }

  private:
    struct BgAddress { unsigned channel; unsigned rank; unsigned bank_group; };

    static unsigned integerLog2(unsigned value)
    {
        unsigned bits = 0;
        while ((1u << bits) < value)
            ++bits;
        return bits;
    }

    BgAddress decodeBg(uint64_t global_bg) const
    {
        uint64_t groups_per_channel = static_cast<uint64_t>(ranks_) * bank_groups_;
        return {static_cast<unsigned>(global_bg / groups_per_channel),
                static_cast<unsigned>((global_bg / bank_groups_) % ranks_),
                static_cast<unsigned>(global_bg % bank_groups_)};
    }

    BgAddress decodePhysical(uint64_t physical) const
    {
        unsigned offset_bits = integerLog2(transaction_bytes_);
        unsigned channel_bits = integerLog2(channels_);
        unsigned bank_bits = integerLog2(banks_) - integerLog2(bank_groups_);
        unsigned bg_bits = integerLog2(bank_groups_);
        unsigned col_bits = integerLog2(kernel_->pim_addr_mgr_->num_cols_per_bl_);
        unsigned row_bits = integerLog2(kernel_->pim_addr_mgr_->num_rows_);
        uint64_t shifted = physical >> offset_bits;
        unsigned channel = shifted & (channels_ - 1);
        shifted >>= channel_bits + bank_bits;
        unsigned bg = shifted & (bank_groups_ - 1);
        shifted >>= bg_bits + col_bits + row_bits;
        unsigned rank = shifted & (ranks_ - 1);
        return {channel, rank, bg};
    }

    uint64_t address(uint64_t global_bg, unsigned bank, unsigned base_row,
                     uint64_t burst_index) const
    {
        BgAddress bg = decodeBg(global_bg);
        unsigned banks_per_bg = banks_ / bank_groups_;
        unsigned row = base_row;
        unsigned col = static_cast<unsigned>(burst_index);
        return kernel_->pim_addr_mgr_->addrGenSafe(bg.channel, bg.rank, bg.bank_group,
                                                   bank % banks_per_bg, row, col);
    }

    void addActiveBarriers(const vector<CSRBGShard>& shards)
    {
        vector<char> active(channels_, 0);
        for (uint64_t bg = 0; bg < shards.size(); ++bg)
            if (!shards[bg].rows.empty())
                active[decodeBg(bg).channel] = 1;
        for (unsigned channel = 0; channel < channels_; ++channel)
            if (active[channel])
                mem_->addBarrier(channel);
    }

    void issuePacked(uint64_t global_bg, unsigned bank, unsigned base_row,
                     uint64_t bytes, bool write, const string& tag, BurstType& burst)
    {
        uint64_t bursts = ceilDiv(bytes, transaction_bytes_);
        for (uint64_t i = 0; i < bursts; ++i)
            mem_->addTransaction(write, address(global_bg, bank, base_row, i), tag, &burst);
    }

    void issueMatrixProgramming(const vector<CSRBGShard>& shards, BurstType& burst)
    {
        for (uint64_t bg = 0; bg < shards.size(); ++bg)
        {
            issuePacked(bg, 0, kDescriptorBaseRow,
                        shards[bg].rows.size() * kRowDescriptorBytes, true,
                        "CSR_DESCRIPTOR_PROGRAM", burst);
            issuePacked(bg, 0, kValueBaseRow,
                        shards[bg].values.size() * kValueBytes, true,
                        "CSR_VALUE_PROGRAM", burst);
            issuePacked(bg, 1, kIndexBaseRow,
                        shards[bg].col_indices.size() * kColumnIndexBytes, true,
                        "CSR_INDEX_PROGRAM", burst);
        }
        addActiveBarriers(shards);
    }

    uint64_t issueXTransfer(uint32_t cols, BurstType& burst)
    {
        vector<uint64_t> values_per_bg(global_bank_groups_, 0);
        for (uint32_t col = 0; col < cols; ++col)
            values_per_bg[col % global_bank_groups_]++;
        for (uint64_t bg = 0; bg < global_bank_groups_; ++bg)
            issuePacked(bg, 2, kXBaseRow, values_per_bg[bg] * kXBytes, true,
                        "CSR_X_TRANSFER", burst);
        for (unsigned channel = 0; channel < channels_; ++channel)
            mem_->addBarrier(channel);
        return drain();
    }

    void issueExecutionTrace(const vector<CSRBGShard>& shards, BurstType& burst)
    {
        for (uint64_t bg = 0; bg < shards.size(); ++bg)
        {
            const CSRBGShard& shard = shards[bg];
            issuePacked(bg, 0, kDescriptorBaseRow,
                        shard.rows.size() * kRowDescriptorBytes, false,
                        "CSR_ROW_DESCRIPTOR_READ", burst);
            issuePacked(bg, 0, kValueBaseRow, shard.values.size() * kValueBytes,
                        false, "CSR_VALUE_READ_MAC_TRACE", burst);
            issuePacked(bg, 1, kIndexBaseRow,
                        shard.col_indices.size() * kColumnIndexBytes,
                        false, "CSR_COLUMN_INDEX_READ", burst);
            for (uint32_t col : shard.col_indices)
            {
                uint64_t x_bg = col % global_bank_groups_;
                uint64_t local_index = col / global_bank_groups_;
                uint64_t burst_index = (local_index * kXBytes) / transaction_bytes_;
                mem_->addTransaction(false, address(x_bg, 2, kXBaseRow, burst_index),
                                     "CSR_INDEXED_X_GATHER_MAC_TRACE", &burst);
            }
        }
        addActiveBarriers(shards);
    }

    void issueOutputTraffic(const vector<CSRBGShard>& shards, bool write, BurstType& burst)
    {
        for (uint64_t bg = 0; bg < shards.size(); ++bg)
            issuePacked(bg, 3, kOutputBaseRow,
                        shards[bg].rows.size() * kOutputBytes, write,
                        write ? "CSR_ROW_OUTPUT_WRITE" : "CSR_OUTPUT_READBACK", burst);
        addActiveBarriers(shards);
    }

    uint64_t drain()
    {
        uint64_t before = kernel_->getCycle();
        kernel_->runPIM();
        return kernel_->getCycle() - before;
    }

    shared_ptr<MultiChannelMemorySystem> mem_;
    shared_ptr<PIMKernel> kernel_;
    unsigned channels_ = 0, ranks_ = 0, bank_groups_ = 0, banks_ = 0;
    uint64_t transaction_bytes_ = 0, global_bank_groups_ = 0;
};

Result runOne(const fs::path& input_path, unsigned iterations)
{
    Result result;
    result.matrix_name = matrixName(input_path);
    result.input_path = fs::absolute(input_path).string();
    // Permit a single recorded resident sample for time-constrained sweeps.
    // TimingSimulator still performs one unreported warm-up iteration.
    result.iterations = max(1u, iterations);
    auto loaded = loadCSRNPZ(input_path);
    result.input_load_us = loaded.input_load_us;
    result.matrix = move(loaded.matrix);

    TimingSimulator simulator;
    result.mapping = buildMapping(result.matrix, simulator.globalBankGroups());
    vector<float> x = makeX(result.matrix.cols);
    vector<float> reference = cpuReference(result.matrix, x);
    vector<float> modeled = functionalSharded(result.matrix, result.mapping, x);
    result.validation_passed = almostEqual(reference, modeled) &&
        modeled.size() == result.matrix.rows;
    result.load = buildLoadStats(result.matrix, result.mapping);
    result.traffic = buildTrafficStats(result.matrix);
    if (result.traffic.x_gather_count != result.matrix.values.size())
        throw logic_error("x gather count differs from nnz");
    result.timing = simulator.run(result.matrix, result.mapping, result.iterations);
    if (result.timing.total_pim_cycle != result.timing.matrix_programming_cycle +
                                            result.timing.resident_iteration_cycle)
        throw logic_error("total PIM cycle accounting mismatch");
    result.end_to_end_ms = result.timing.pim_time_ms +
        (result.input_load_us + result.mapping.direct_mapping_us +
         result.mapping.metadata_generation_us + result.mapping.bg_sharding_us) / 1000.0;
    return result;
}

const char* csvHeader()
{
    return "matrix,input_path,rows,cols,nnz,nonempty_rows,empty_rows,mapping_policy,"
           "global_bank_groups,active_bank_groups,min_rows_per_bg,max_rows_per_bg,"
           "avg_rows_per_bg,min_nnz_per_bg,max_nnz_per_bg,avg_nnz_per_bg,"
           "nnz_per_bg_stddev,bg_load_imbalance_all,bg_load_imbalance_active,"
           "critical_bg_id,critical_bg_nnz,longest_row_nnz,average_nonempty_row_nnz,"
           "pim_alu_precision,value_element_bytes,column_index_bytes,x_element_bytes,"
           "output_element_bytes,value_read_count,value_read_bytes,"
           "column_index_read_count,column_index_read_bytes,x_gather_count,x_gather_bytes,"
           "output_write_count,output_write_bytes,output_readback_count,output_readback_bytes,"
           "input_load_us,direct_mapping_us,bg_sharding_us,metadata_generation_us,"
           "matrix_programming_cycle,x_transfer_cycle,pim_mode_change_cycle,"
           "crf_programming_cycle,csr_execution_trace_cycle,output_write_cycle,"
           "output_readback_cycle,modeled_core_kernel_cycle,pim_execution_cycle,"
           "retrieval_inclusive_cycle,resident_iteration_cycle,total_pim_cycle,"
           "modeled_core_kernel_ms,resident_iteration_ms,pim_time_ms,end_to_end_ms,"
           "iterations,validation_passed,error_message\n";
}

void initializeCsv(const fs::path& path)
{
    ofstream output(path, ios::trunc);
    if (!output)
        throw runtime_error("failed to create CSV " + path.string());
    output << csvHeader();
}

void appendCsv(const fs::path& path, const Result& r)
{
    ofstream out(path, ios::app);
    if (!out)
        throw runtime_error("failed to append CSV " + path.string());
    const LoadStats& l = r.load;
    const TrafficStats& x = r.traffic;
    const TimingResult& t = r.timing;
    out << setprecision(12) << csvEscape(r.matrix_name) << ',' << csvEscape(r.input_path) << ','
        << r.matrix.rows << ',' << r.matrix.cols << ',' << r.matrix.values.size() << ','
        << l.nonempty_rows << ',' << l.empty_rows << ',' << kMappingName << ','
        << r.mapping.shards.size() << ',' << l.active_bank_groups << ','
        << l.rows_per_bg_min << ',' << l.rows_per_bg_max << ',' << l.rows_per_bg_avg << ','
        << l.nnz_per_bg_min << ',' << l.nnz_per_bg_max << ',' << l.nnz_per_bg_avg << ','
        << l.nnz_per_bg_stddev << ',' << l.bg_load_imbalance_all << ','
        << l.bg_load_imbalance_active << ',' << l.critical_bg_id << ','
        << l.critical_bg_nnz << ',' << l.longest_row_nnz << ','
        << l.average_nonempty_row_nnz << ','
        << "configured_fp16_mac_crf_trace_logical_fp32" << ','
        << kValueBytes << ',' << kColumnIndexBytes << ',' << kXBytes << ',' << kOutputBytes << ','
        << x.value_read_count << ',' << x.value_read_bytes << ','
        << x.column_index_read_count << ',' << x.column_index_read_bytes << ','
        << x.x_gather_count << ',' << x.x_gather_bytes << ','
        << x.output_write_count << ',' << x.output_write_bytes << ','
        << x.output_readback_count << ',' << x.output_readback_bytes << ','
        << r.input_load_us << ',' << r.mapping.direct_mapping_us << ','
        << r.mapping.bg_sharding_us << ',' << r.mapping.metadata_generation_us << ','
        << t.matrix_programming_cycle << ',' << t.x_transfer_cycle << ','
        << t.pim_mode_change_cycle << ',' << t.crf_programming_cycle << ','
        << t.csr_execution_trace_cycle << ',' << t.output_write_cycle << ','
        << t.output_readback_cycle << ',' << t.modeled_core_kernel_cycle << ','
        << t.pim_execution_cycle << ',' << t.retrieval_inclusive_cycle << ','
        << t.resident_iteration_cycle << ',' << t.total_pim_cycle << ','
        << t.modeled_core_kernel_ms << ',' << t.resident_iteration_ms << ','
        << t.pim_time_ms << ',' << r.end_to_end_ms << ',' << r.iterations << ','
        << (r.validation_passed ? 1 : 0) << ',' << csvEscape(r.error_message) << '\n';
}

CSRMatrix makeCSR(uint32_t rows, uint32_t cols,
                  const vector<vector<pair<uint32_t, float>>>& entries_by_row)
{
    if (entries_by_row.size() != rows)
        throw invalid_argument("synthetic row count mismatch");
    CSRMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.row_ptr.push_back(0);
    for (const auto& row : entries_by_row)
    {
        for (const auto& entry : row)
        {
            if (entry.first >= cols)
                throw invalid_argument("synthetic column outside shape");
            matrix.col_idx.push_back(entry.first);
            matrix.values.push_back(entry.second);
        }
        matrix.row_ptr.push_back(matrix.values.size());
    }
    return matrix;
}

void validateFunctional(const CSRMatrix& matrix, uint64_t bank_groups)
{
    MappingResult mapping = buildMapping(matrix, bank_groups);
    vector<float> x = makeX(matrix.cols);
    EXPECT_TRUE(almostEqual(cpuReference(matrix, x), functionalSharded(matrix, mapping, x)));
    uint64_t rows = 0, nnz = 0;
    for (const auto& shard : mapping.shards)
    {
        rows += shard.rows.size();
        nnz += shard.values.size();
    }
    EXPECT_EQ(rows, matrix.rows);
    EXPECT_EQ(nnz, matrix.values.size());
    EXPECT_EQ(buildTrafficStats(matrix).x_gather_count, matrix.values.size());
}

} // namespace

TEST(CSRDirectSpMVFunctionalTest, RequiredCornerCases)
{
    vector<vector<pair<uint32_t, float>>> rows(600);
    rows[0] = {{7, 1.0f}, {2, 2.0f}, {7, 3.0f}, {1, 4.0f}, {19, 5.0f},
               {0, 6.0f}, {12, 7.0f}, {3, 8.0f}, {5, 9.0f}}; // unsorted + duplicate
    rows[1] = {{4, 2.5f}}; // singleton
    rows[300] = {{0, 1.0f}, {1, 1.0f}, {2, 1.0f}};
    // Empty rows, trailing empty row/column, rectangular shape, row count > BG count,
    // and a non-burst-multiple NNZ count are all intentional.
    validateFunctional(makeCSR(600, 23, rows), 32);

    vector<vector<pair<uint32_t, float>>> small(3);
    small[0] = {{1, 1.0f}};
    validateFunctional(makeCSR(3, 8, small), 16); // rows < BG count
}

TEST(CSRDirectSpMVPhysicalMappingTest, Scheme8RepresentativeRows)
{
    TimingSimulator simulator;
    EXPECT_TRUE(simulator.validateRepresentativeMappings(2000, true));
}

TEST(CSRDirectSpMVSanityTest, ScalingImbalanceShuffleAndHeavyRow)
{
    // Use enough traffic to exceed one DRAM burst and expose the critical BG.
    // The smaller 16-row fixture was entirely parallel across the 512 BGs, so
    // doubling two entries per BG legitimately had the same critical-path time.
    constexpr uint32_t kRows = 512;
    constexpr uint32_t kBasePerRow = 16;
    vector<vector<pair<uint32_t, float>>> base_rows(kRows), doubled_rows(kRows),
        heavy_rows(kRows);
    for (uint32_t row = 0; row < kRows; ++row)
    {
        for (uint32_t i = 0; i < kBasePerRow; ++i)
            base_rows[row].push_back({(row * 17 + i) % 4096, 1.0f});
        doubled_rows[row] = base_rows[row];
        for (uint32_t i = 0; i < kBasePerRow; ++i)
            doubled_rows[row].push_back({(row * 17 + i + 2048) % 4096, 1.0f});
    }
    for (uint32_t i = 0; i < kRows * kBasePerRow; ++i)
        heavy_rows[0].push_back({i % 4096, 1.0f});
    CSRMatrix base = makeCSR(kRows, 4096, base_rows);
    CSRMatrix doubled = makeCSR(kRows, 4096, doubled_rows);
    CSRMatrix heavy = makeCSR(kRows, 4096, heavy_rows);
    vector<vector<pair<uint32_t, float>>> shuffled_rows = base_rows;
    for (auto& row : shuffled_rows)
        reverse(row.begin(), row.end());
    CSRMatrix shuffled = makeCSR(kRows, 4096, shuffled_rows);

    TimingSimulator base_sim, doubled_sim, heavy_sim, shuffled_sim;
    auto base_map = buildMapping(base, base_sim.globalBankGroups());
    auto doubled_map = buildMapping(doubled, doubled_sim.globalBankGroups());
    auto heavy_map = buildMapping(heavy, heavy_sim.globalBankGroups());
    auto shuffled_map = buildMapping(shuffled, shuffled_sim.globalBankGroups());
    TimingResult base_t = base_sim.run(base, base_map, 1);
    TimingResult doubled_t = doubled_sim.run(doubled, doubled_map, 1);
    TimingResult heavy_t = heavy_sim.run(heavy, heavy_map, 1);
    TimingResult shuffled_t = shuffled_sim.run(shuffled, shuffled_map, 1);
    TrafficStats base_x = buildTrafficStats(base), doubled_x = buildTrafficStats(doubled);
    EXPECT_GT(doubled_x.value_read_bytes, base_x.value_read_bytes);
    EXPECT_GT(doubled_x.column_index_read_bytes, base_x.column_index_read_bytes);
    EXPECT_GT(doubled_x.x_gather_bytes, base_x.x_gather_bytes);
    EXPECT_GT(doubled_t.csr_execution_trace_cycle, base_t.csr_execution_trace_cycle);
    EXPECT_GT(buildLoadStats(heavy, heavy_map).nnz_per_bg_max,
              buildLoadStats(base, base_map).nnz_per_bg_max);
    EXPECT_GE(heavy_t.csr_execution_trace_cycle, base_t.csr_execution_trace_cycle);
    EXPECT_EQ(buildTrafficStats(shuffled).x_gather_count, base_x.x_gather_count);
    cout << "CSR_SANITY base_cycle=" << base_t.csr_execution_trace_cycle
         << " doubled_cycle=" << doubled_t.csr_execution_trace_cycle
         << " heavy_cycle=" << heavy_t.csr_execution_trace_cycle
         << " shuffled_cycle=" << shuffled_t.csr_execution_trace_cycle << endl;
}

TEST(CSRDirectSpMVBenchmark, RunFromEnvironment)
{
    string matrix = envString("CSR_DIRECT_MATRIX");
    if (matrix.empty())
        GTEST_SKIP() << "set CSR_DIRECT_MATRIX to a standard CSR NPZ";
    fs::path output = envString("CSR_DIRECT_OUTPUT",
                                "../SparsePIM/csr_direct_round_robin_results.csv");
    unsigned iterations = envUnsigned("CSR_DIRECT_ITERATIONS", 7);
    initializeCsv(output);
    Result result = runOne(matrix, iterations);
    appendCsv(output, result);
    EXPECT_TRUE(result.validation_passed);
    cout << ">>Trace-based direct CSR inner-product baseline\n"
         << "  matrix: " << result.matrix_name << " rows=" << result.matrix.rows
         << " cols=" << result.matrix.cols << " nnz=" << result.matrix.values.size() << '\n'
         << "  mapping_policy: " << kMappingName
         << " global_bank_groups=" << result.mapping.shards.size() << '\n'
         << "  modeled_core_kernel_cycle: " << result.timing.modeled_core_kernel_cycle
         << " ms=" << result.timing.modeled_core_kernel_ms << '\n'
         << "  resident_iteration_cycle: " << result.timing.resident_iteration_cycle
         << " ms=" << result.timing.resident_iteration_ms << '\n'
         << "  validation_passed: " << result.validation_passed
         << " output=" << output << endl;
}

TEST(CSRDirectSpMVBenchmark, RunDirectory)
{
    string directory = envString("CSR_DIRECT_MATRIX_DIR");
    if (directory.empty())
        GTEST_SKIP() << "set CSR_DIRECT_MATRIX_DIR";
    fs::path output = envString("CSR_DIRECT_OUTPUT",
                                "../SparsePIM/csr_direct_all_results.csv");
    unsigned iterations = envUnsigned("CSR_DIRECT_ITERATIONS", 7);
    initializeCsv(output);
    vector<fs::path> inputs;
    for (const auto& entry : fs::directory_iterator(directory))
        if (entry.is_regular_file() && entry.path().filename().string().size() >= 8 &&
            entry.path().filename().string().rfind("_csr.npz") ==
                entry.path().filename().string().size() - 8)
            inputs.push_back(entry.path());
    sort(inputs.begin(), inputs.end());
    ASSERT_FALSE(inputs.empty());
    for (const fs::path& input : inputs)
    {
        try
        {
            Result result = runOne(input, iterations);
            appendCsv(output, result);
            cout << "CSR_DIRECT_DONE " << result.matrix_name
                 << " resident_ms=" << result.timing.resident_iteration_ms << endl;
        }
        catch (const exception& error)
        {
            Result failed;
            failed.matrix_name = matrixName(input);
            failed.input_path = fs::absolute(input).string();
            failed.iterations = iterations;
            failed.error_message = error.what();
            appendCsv(output, failed);
            cerr << "CSR_DIRECT_ERROR " << input << ": " << error.what() << endl;
        }
    }
}

} // namespace csr_direct
