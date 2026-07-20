#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "Burst.h"
#include "MultiChannelMemorySystem.h"
#include "PIMCmd.h"
#include "gtest/gtest.h"
#include "tests/PIMKernel.h"

using namespace DRAMSim;
using namespace std;

namespace csc_partial_stream
{
namespace
{
constexpr uint32_t kPimRegisterRow = 0x3fff;
constexpr uint32_t kMatrixBaseRow = 0;
constexpr uint32_t kXBaseRow = 6144;
constexpr uint32_t kPartialBaseRow = 8192;
constexpr uint32_t kLogicalValueBytes = sizeof(float);

struct CooEntry
{
    uint32_t row = 0;
    uint32_t col = 0;
    float value = 0.0f;
};

struct CscMatrix
{
    uint32_t rows = 0;
    uint32_t cols = 0;
    vector<uint64_t> col_ptr;
    vector<uint32_t> row_idx;
    vector<float> values;
};

struct ColumnDescriptor
{
    uint32_t global_column_id = 0;
    uint64_t local_value_start = 0;
    uint32_t nnz = 0;
};

struct BgShard
{
    vector<float> values;
    vector<uint32_t> row_indices;
    vector<ColumnDescriptor> columns;
};

struct ShardingResult
{
    vector<BgShard> shards;
    double metadata_generation_us = 0.0;
    double csc_sharding_us = 0.0;
};

struct TrafficStats
{
    uint64_t active_bank_groups = 0;
    uint64_t max_nnz_per_bg = 0;
    uint64_t min_nnz_per_active_bg = 0;
    double avg_nnz_per_active_bg = 0.0;
    double bg_load_imbalance = 0.0;
    uint64_t partial_value_count = 0;
    uint64_t partial_value_bytes = 0;
    uint64_t final_y_bytes = 0;
    double partial_output_amplification = 0.0;
    vector<uint64_t> unique_rows_per_bg;
    uint64_t sum_unique_rows_across_bgs = 0;
    double average_row_fanout = 0.0;
    uint64_t maximum_row_fanout = 0;
};

struct TimingResult
{
    uint64_t matrix_programming_cycle = 0;
    uint64_t x_transfer_cycle = 0;
    uint64_t pim_mode_change_cycle = 0;
    uint64_t crf_programming_cycle = 0;
    uint64_t pim_execution_cycle = 0;
    uint64_t partial_write_cycle = 0;
    uint64_t partial_readback_cycle = 0;
    uint64_t total_pim_cycle = 0;
    double pim_time_ms = 0.0;
};

struct BenchmarkResult
{
    string matrix;
    CscMatrix csc;
    ShardingResult sharding;
    TrafficStats traffic;
    TimingResult timing;
    double host_reduction_us = 0.0;
    double steady_state_ms = 0.0;
    double end_to_end_ms = 0.0;
    bool validation_passed = false;
};

uint64_t ceilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

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

string baseName(const string& path)
{
    size_t slash = path.find_last_of("/\\");
    string name = slash == string::npos ? path : path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    if (dot != string::npos)
        name.resize(dot);
    if (name.size() > 4 && name.compare(name.size() - 4, 4, "_csc") == 0)
        name.resize(name.size() - 4);
    return name;
}

CscMatrix makeCsc(uint32_t rows, uint32_t cols, vector<CooEntry> entries)
{
    sort(entries.begin(), entries.end(), [](const CooEntry& lhs, const CooEntry& rhs) {
        if (lhs.col != rhs.col)
            return lhs.col < rhs.col;
        return lhs.row < rhs.row;
    });
    CscMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.col_ptr.assign(static_cast<size_t>(cols) + 1, 0);
    for (const CooEntry& entry : entries)
    {
        if (entry.row >= rows || entry.col >= cols)
            throw runtime_error("matrix entry exceeds declared dimensions");
        matrix.col_ptr[entry.col + 1]++;
    }
    partial_sum(matrix.col_ptr.begin(), matrix.col_ptr.end(), matrix.col_ptr.begin());
    matrix.row_idx.reserve(entries.size());
    matrix.values.reserve(entries.size());
    for (const CooEntry& entry : entries)
    {
        matrix.row_idx.push_back(entry.row);
        matrix.values.push_back(entry.value);
    }
    return matrix;
}

CscMatrix loadMatrixMarket(const string& path)
{
    ifstream in(path);
    if (!in)
        throw runtime_error("failed to open " + path);
    string banner;
    getline(in, banner);
    istringstream header(banner);
    string magic, object, format, field, symmetry;
    header >> magic >> object >> format >> field >> symmetry;
    if (magic != "%%MatrixMarket" || object != "matrix" || format != "coordinate")
        throw runtime_error("only Matrix Market coordinate matrices are supported");
    if (field == "complex")
        throw runtime_error("complex Matrix Market matrices are not supported");
    bool pattern = field == "pattern";
    bool symmetric = symmetry == "symmetric" || symmetry == "hermitian";
    bool skew = symmetry == "skew-symmetric";

    string line;
    do
    {
        if (!getline(in, line))
            throw runtime_error("missing Matrix Market dimensions");
    } while (line.empty() || line[0] == '%');
    uint64_t rows = 0, cols = 0, declared_nnz = 0;
    istringstream dims(line);
    dims >> rows >> cols >> declared_nnz;
    if (rows > numeric_limits<uint32_t>::max() || cols > numeric_limits<uint32_t>::max())
        throw runtime_error("matrix dimensions exceed uint32_t");

    vector<CooEntry> entries;
    entries.reserve(symmetric || skew ? declared_nnz * 2 : declared_nnz);
    uint64_t row = 0, col = 0;
    double value = 1.0;
    while (in >> row >> col)
    {
        if (!pattern)
            in >> value;
        else
            value = 1.0;
        if (row == 0 || col == 0)
            throw runtime_error("Matrix Market indices must be one-based");
        --row;
        --col;
        entries.push_back({static_cast<uint32_t>(row), static_cast<uint32_t>(col),
                           static_cast<float>(value)});
        if ((symmetric || skew) && row != col)
            entries.push_back({static_cast<uint32_t>(col), static_cast<uint32_t>(row),
                               static_cast<float>(skew ? -value : value)});
    }
    return makeCsc(static_cast<uint32_t>(rows), static_cast<uint32_t>(cols), move(entries));
}

CscMatrix loadTripletCsc(const string& path)
{
    ifstream in(path);
    if (!in)
        throw runtime_error("failed to open " + path);
    vector<CooEntry> entries;
    string line;
    uint64_t max_row = 0, max_col = 0;
    while (getline(in, line))
    {
        if (line.empty() || line[0] == '#' || line[0] == '%')
            continue;
        istringstream iss(line);
        uint64_t row = 0, col = 0;
        float value = 1.0f;
        if (!(iss >> row >> col))
            continue;
        iss >> value;
        if (row > numeric_limits<uint32_t>::max() || col > numeric_limits<uint32_t>::max())
            throw runtime_error("triplet index exceeds uint32_t");
        entries.push_back({static_cast<uint32_t>(row), static_cast<uint32_t>(col), value});
        max_row = max(max_row, row);
        max_col = max(max_col, col);
    }
    if (entries.empty())
        throw runtime_error("no entries in " + path);
    return makeCsc(static_cast<uint32_t>(max_row + 1), static_cast<uint32_t>(max_col + 1),
                   move(entries));
}

CscMatrix loadCsc(const string& path)
{
    ifstream probe(path);
    string first;
    getline(probe, first);
    return first.rfind("%%MatrixMarket", 0) == 0 ? loadMatrixMarket(path)
                                                  : loadTripletCsc(path);
}

ShardingResult shardCsc(const CscMatrix& matrix, uint64_t num_global_bank_groups)
{
    if (num_global_bank_groups == 0)
        throw invalid_argument("num_global_bank_groups must be nonzero");
    ShardingResult result;
    result.shards.resize(num_global_bank_groups);

    auto metadata_start = chrono::steady_clock::now();
    vector<uint64_t> counts(num_global_bank_groups, 0);
    for (uint32_t col = 0; col < matrix.cols; ++col)
    {
        uint64_t bg = col % num_global_bank_groups;
        uint64_t count = matrix.col_ptr[col + 1] - matrix.col_ptr[col];
        if (count > numeric_limits<uint32_t>::max())
            throw runtime_error("column nnz exceeds uint32_t");
        result.shards[bg].columns.push_back(
            {col, counts[bg], static_cast<uint32_t>(count)});
        counts[bg] += count;
    }
    for (uint64_t bg = 0; bg < num_global_bank_groups; ++bg)
    {
        result.shards[bg].values.reserve(counts[bg]);
        result.shards[bg].row_indices.reserve(counts[bg]);
    }
    auto metadata_end = chrono::steady_clock::now();

    auto sharding_start = chrono::steady_clock::now();
    for (uint32_t col = 0; col < matrix.cols; ++col)
    {
        BgShard& shard = result.shards[col % num_global_bank_groups];
        for (uint64_t k = matrix.col_ptr[col]; k < matrix.col_ptr[col + 1]; ++k)
        {
            shard.values.push_back(matrix.values[k]);
            shard.row_indices.push_back(matrix.row_idx[k]);
        }
    }
    auto sharding_end = chrono::steady_clock::now();
    result.metadata_generation_us =
        chrono::duration<double, micro>(metadata_end - metadata_start).count();
    result.csc_sharding_us =
        chrono::duration<double, micro>(sharding_end - sharding_start).count();
    return result;
}

vector<float> cpuCscSpmv(const CscMatrix& matrix, const vector<float>& x)
{
    vector<float> y(matrix.rows, 0.0f);
    for (uint32_t col = 0; col < matrix.cols; ++col)
        for (uint64_t k = matrix.col_ptr[col]; k < matrix.col_ptr[col + 1]; ++k)
            y[matrix.row_idx[k]] += matrix.values[k] * x[col];
    return y;
}

vector<vector<float>> generatePartialValues(const vector<BgShard>& shards,
                                            const vector<float>& x)
{
    vector<vector<float>> partials(shards.size());
    for (size_t bg = 0; bg < shards.size(); ++bg)
    {
        partials[bg].resize(shards[bg].values.size());
        for (const ColumnDescriptor& column : shards[bg].columns)
            for (uint64_t local = column.local_value_start;
                 local < column.local_value_start + column.nnz; ++local)
                partials[bg][local] = shards[bg].values[local] * x[column.global_column_id];
    }
    return partials;
}

vector<float> reducePartials(const vector<BgShard>& shards,
                             const vector<vector<float>>& partials, uint32_t rows)
{
    vector<float> y(rows, 0.0f);
    for (size_t bg = 0; bg < shards.size(); ++bg)
        for (size_t k = 0; k < partials[bg].size(); ++k)
            y[shards[bg].row_indices[k]] += partials[bg][k];
    return y;
}

bool almostEqual(const vector<float>& lhs, const vector<float>& rhs,
                 float abs_tol = 1.0e-5f, float rel_tol = 1.0e-5f)
{
    if (lhs.size() != rhs.size())
        return false;
    for (size_t i = 0; i < lhs.size(); ++i)
    {
        float diff = fabs(lhs[i] - rhs[i]);
        float scale = max(fabs(lhs[i]), fabs(rhs[i]));
        if (diff > abs_tol + rel_tol * scale)
            return false;
    }
    return true;
}

TrafficStats buildTrafficStats(const CscMatrix& matrix, const vector<BgShard>& shards)
{
    TrafficStats stats;
    stats.partial_value_count = matrix.values.size();
    stats.partial_value_bytes = stats.partial_value_count * kLogicalValueBytes;
    stats.final_y_bytes = static_cast<uint64_t>(matrix.rows) * sizeof(float);
    stats.partial_output_amplification = stats.final_y_bytes == 0
        ? 0.0 : static_cast<double>(stats.partial_value_bytes) / stats.final_y_bytes;
    stats.unique_rows_per_bg.resize(shards.size(), 0);
    vector<uint16_t> fanout(matrix.rows, 0);
    uint64_t active_total_nnz = 0;
    stats.min_nnz_per_active_bg = numeric_limits<uint64_t>::max();
    for (size_t bg = 0; bg < shards.size(); ++bg)
    {
        uint64_t nnz = shards[bg].values.size();
        if (nnz == 0)
            continue;
        stats.active_bank_groups++;
        active_total_nnz += nnz;
        stats.max_nnz_per_bg = max(stats.max_nnz_per_bg, nnz);
        stats.min_nnz_per_active_bg = min(stats.min_nnz_per_active_bg, nnz);
        unordered_set<uint32_t> rows(shards[bg].row_indices.begin(),
                                     shards[bg].row_indices.end());
        stats.unique_rows_per_bg[bg] = rows.size();
        stats.sum_unique_rows_across_bgs += rows.size();
        for (uint32_t row : rows)
            fanout[row]++;
    }
    if (stats.active_bank_groups == 0)
        stats.min_nnz_per_active_bg = 0;
    else
        stats.avg_nnz_per_active_bg =
            static_cast<double>(active_total_nnz) / stats.active_bank_groups;
    stats.bg_load_imbalance = stats.avg_nnz_per_active_bg == 0.0
        ? 0.0 : static_cast<double>(stats.max_nnz_per_bg) / stats.avg_nnz_per_active_bg;
    uint64_t active_rows = 0;
    uint64_t fanout_sum = 0;
    for (uint16_t count : fanout)
    {
        if (count == 0)
            continue;
        active_rows++;
        fanout_sum += count;
        stats.maximum_row_fanout = max<uint64_t>(stats.maximum_row_fanout, count);
    }
    stats.average_row_fanout = active_rows == 0
        ? 0.0 : static_cast<double>(fanout_sum) / active_rows;
    return stats;
}

double measureHostReductionUs(const vector<BgShard>& shards,
                              const vector<vector<float>>& partials, uint32_t rows,
                              unsigned iterations)
{
    volatile float sink = 0.0f;
    vector<float> warm = reducePartials(shards, partials, rows);
    if (!warm.empty())
        sink += warm[0];
    vector<double> samples;
    samples.reserve(iterations);
    for (unsigned i = 0; i < iterations; ++i)
    {
        auto start = chrono::steady_clock::now();
        vector<float> y = reducePartials(shards, partials, rows);
        auto end = chrono::steady_clock::now();
        if (!y.empty())
            sink += y[i % y.size()];
        samples.push_back(chrono::duration<double, micro>(end - start).count());
    }
    (void)sink;
    sort(samples.begin(), samples.end());
    size_t mid = samples.size() / 2;
    return samples.size() % 2 == 0 ? (samples[mid - 1] + samples[mid]) * 0.5
                                   : samples[mid];
}

class TimingSimulator
{
  public:
    TimingSimulator()
    {
        mem_ = make_shared<MultiChannelMemorySystem>("ini/HBM2_samsung_2M_16B_x64.ini",
                                                     "system_hbm_64ch.ini", ".",
                                                     "csc_partial_stream", 256 * 64 * 2);
        unsigned channels = getConfigParam(UINT, "NUM_CHANS");
        unsigned ranks = getConfigParam(UINT, "NUM_RANKS");
        kernel_ = make_shared<PIMKernel>(mem_, channels, ranks);
        channels_ = kernel_->pim_addr_mgr_->num_chans_;
        ranks_ = kernel_->pim_addr_mgr_->num_ranks_;
        bank_groups_ = kernel_->pim_addr_mgr_->num_bank_groups_;
        global_bank_groups_ = static_cast<uint64_t>(channels_) * ranks_ * bank_groups_;
        transaction_bytes_ = kernel_->transaction_size_;
    }

    uint64_t globalBankGroups() const { return global_bank_groups_; }

    TimingResult run(const CscMatrix& matrix, const vector<BgShard>& shards)
    {
        TimingResult timing;
        BurstType null_burst;
        bool verbose = envString("CSC_PARTIAL_PHASE_LOG") == "1";
        auto phase = [&](const char* name) {
            if (verbose)
                cerr << "CSC_PARTIAL_PHASE " << name << endl;
        };

        phase("matrix_programming_begin");
        issuePayloadWrites(shards, kMatrixBaseRow, 0, null_burst);
        timing.matrix_programming_cycle = drain();
        phase("matrix_programming_end");

        phase("x_transfer_begin");
        issueXWrites(matrix.cols, null_burst);
        timing.x_transfer_cycle = drain();
        phase("x_transfer_end");

        phase("pim_setup_begin");
        kernel_->configurePIMControl();
        kernel_->parkIn();
        kernel_->changePIMMode(dramMode::SB, dramMode::HAB);
        timing.pim_mode_change_cycle += drain();

        vector<PIMCmd> commands{
            PIMCmd(PIMCmdType::MUL, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK,
                   PIMOpdType::SRF_M, 1),
            PIMCmd(PIMCmdType::NOP, 7),
            PIMCmd(PIMCmdType::EXIT, 0),
        };
        kernel_->programCrf(commands);
        timing.crf_programming_cycle = drain();
        kernel_->changePIMMode(dramMode::HAB, dramMode::SB);
        kernel_->parkOut();
        timing.pim_mode_change_cycle += drain();
        phase("pim_setup_end");

        /*
         * The public executeEltwise() API assumes one dense shape shared by every
         * channel and cannot reset SRF/CRF independently for variable-length CSC
         * columns.  We therefore program the real MUL CRF above, but issue the
         * column scalar writes and value-stream trigger reads as a timing trace in
         * HAB mode.  The functional path computes the identical products on CPU.
         * Entering HAB_PIM here would leave the CRF PC at a different repeat state
         * for every short column and is not a valid execution of the public API.
         */
        phase("execution_begin");
        issueScalarProgrammingAndMulTriggers(shards, null_burst);
        timing.pim_execution_cycle = drain();
        phase("execution_end");

        phase("partial_write_begin");
        issuePayloadWrites(shards, kPartialBaseRow, 1, null_burst);
        timing.partial_write_cycle = drain();
        phase("partial_write_end");

        phase("mode_exit_end");

        phase("readback_begin");
        issuePayloadReads(shards, kPartialBaseRow, null_burst);
        timing.partial_readback_cycle = drain();
        phase("readback_end");

        timing.total_pim_cycle = timing.matrix_programming_cycle + timing.x_transfer_cycle +
            timing.pim_mode_change_cycle + timing.crf_programming_cycle +
            timing.pim_execution_cycle + timing.partial_write_cycle +
            timing.partial_readback_cycle;
        double tck_ns = getConfigParam(FLOAT, "tCK");
        timing.pim_time_ms = static_cast<double>(timing.total_pim_cycle) * tck_ns / 1.0e6;
        return timing;
    }

  private:
    struct BgAddress
    {
        unsigned channel;
        unsigned rank;
        unsigned bank_group;
    };

    BgAddress decodeBg(uint64_t global_bg) const
    {
        uint64_t per_channel = static_cast<uint64_t>(ranks_) * bank_groups_;
        return {static_cast<unsigned>(global_bg / per_channel),
                static_cast<unsigned>((global_bg / bank_groups_) % ranks_),
                static_cast<unsigned>(global_bg % bank_groups_)};
    }

    uint64_t address(uint64_t global_bg, unsigned bank, unsigned base_row,
                     uint64_t burst_index) const
    {
        BgAddress bg = decodeBg(global_bg);
        unsigned banks_per_bg = kernel_->pim_addr_mgr_->num_banks_ / bank_groups_;
        unsigned local_bank = bank % banks_per_bg;
        unsigned row = base_row;
        unsigned col = static_cast<unsigned>(burst_index);
        return kernel_->pim_addr_mgr_->addrGenSafe(bg.channel, bg.rank, bg.bank_group,
                                                   local_bank, row, col);
    }

    void addActiveBarriers(const vector<BgShard>& shards)
    {
        vector<char> active(channels_, 0);
        for (uint64_t bg = 0; bg < shards.size(); ++bg)
            if (!shards[bg].values.empty())
                active[decodeBg(bg).channel] = 1;
        for (unsigned channel = 0; channel < channels_; ++channel)
            if (active[channel])
                mem_->addBarrier(channel);
    }

    void issuePayloadWrites(const vector<BgShard>& shards, unsigned base_row, unsigned bank,
                            BurstType& burst)
    {
        for (uint64_t bg = 0; bg < shards.size(); ++bg)
        {
            uint64_t bursts = ceilDiv(shards[bg].values.size() * kLogicalValueBytes,
                                      transaction_bytes_);
            for (uint64_t i = 0; i < bursts; ++i)
                mem_->addTransaction(true, address(bg, bank, base_row, i), &burst);
        }
        addActiveBarriers(shards);
    }

    void issuePayloadReads(const vector<BgShard>& shards, unsigned base_row,
                           BurstType& burst)
    {
        for (uint64_t bg = 0; bg < shards.size(); ++bg)
        {
            uint64_t bursts = ceilDiv(shards[bg].values.size() * kLogicalValueBytes,
                                      transaction_bytes_);
            for (uint64_t i = 0; i < bursts; ++i)
                mem_->addTransaction(false, address(bg, 1, base_row, i), "CSC_PARTIAL_READ",
                                     &burst);
        }
        addActiveBarriers(shards);
    }

    void issueXWrites(uint32_t columns, BurstType& burst)
    {
        for (uint32_t col = 0; col < columns; ++col)
        {
            uint64_t bg = col % global_bank_groups_;
            mem_->addTransaction(true, address(bg, 0, kXBaseRow, col / global_bank_groups_),
                                 &burst);
        }
        for (unsigned channel = 0; channel < channels_; ++channel)
            mem_->addBarrier(channel);
    }

    void issueScalarProgrammingAndMulTriggers(const vector<BgShard>& shards,
                                               BurstType& burst)
    {
        for (uint64_t global_bg = 0; global_bg < shards.size(); ++global_bg)
        {
            uint64_t local_burst = 0;
            for (const ColumnDescriptor& column : shards[global_bg].columns)
            {
                if (column.nnz == 0)
                    continue;
                // A direct WRIO-to-SRF mixed with per-BG reads is not exposed as
                // a safe public scheduling API. Account one scalar-programming
                // transaction at the column's BG-local x location instead.
                uint64_t srf_addr = address(global_bg, 0, kXBaseRow,
                                            column.global_column_id /
                                                global_bank_groups_);
                mem_->addTransaction(true, srf_addr, "CSC_X_TO_SRF", &burst);
                uint64_t bursts = ceilDiv(static_cast<uint64_t>(column.nnz) *
                                              kLogicalValueBytes,
                                          transaction_bytes_);
                for (uint64_t i = 0; i < bursts; ++i, ++local_burst)
                    mem_->addTransaction(false, address(global_bg, 0, kMatrixBaseRow,
                                                        local_burst),
                                         "CSC_MUL_TRIGGER", &burst);
            }
        }
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
    unsigned channels_ = 0;
    unsigned ranks_ = 0;
    unsigned bank_groups_ = 0;
    uint64_t global_bank_groups_ = 0;
    uint64_t transaction_bytes_ = 0;
};

void writeCsv(const string& path, const BenchmarkResult& result)
{
    bool write_header = true;
    {
        ifstream existing(path, ios::binary | ios::ate);
        write_header = !existing || existing.tellg() == 0;
    }
    ofstream out(path, ios::app);
    if (!out)
        throw runtime_error("failed to open " + path);
    if (write_header)
    {
        out << "matrix,rows,cols,nnz,global_bank_groups,active_bank_groups,"
               "max_nnz_per_bg,min_nnz_per_active_bg,avg_nnz_per_bg,bg_load_imbalance,"
               "partial_value_count,partial_value_bytes,final_y_bytes,"
               "partial_output_amplification,sum_unique_rows_across_bgs,"
               "average_row_fanout,maximum_row_fanout,csc_sharding_us,"
               "metadata_generation_us,matrix_programming_cycle,x_transfer_cycle,"
               "pim_mode_change_cycle,crf_programming_cycle,pim_setup_cycle,"
               "pim_execution_cycle,partial_write_cycle,partial_readback_cycle,"
               "total_pim_cycle,pim_time_ms,host_reduction_us,steady_state_ms,"
               "end_to_end_ms,validation_passed\n";
    }
    const TrafficStats& s = result.traffic;
    const TimingResult& t = result.timing;
    uint64_t pim_setup = t.pim_mode_change_cycle + t.crf_programming_cycle;
    out << setprecision(12) << result.matrix << ',' << result.csc.rows << ','
        << result.csc.cols << ',' << result.csc.values.size() << ','
        << result.sharding.shards.size() << ',' << s.active_bank_groups << ','
        << s.max_nnz_per_bg << ',' << s.min_nnz_per_active_bg << ','
        << s.avg_nnz_per_active_bg << ',' << s.bg_load_imbalance << ','
        << s.partial_value_count << ',' << s.partial_value_bytes << ','
        << s.final_y_bytes << ',' << s.partial_output_amplification << ','
        << s.sum_unique_rows_across_bgs << ',' << s.average_row_fanout << ','
        << s.maximum_row_fanout << ',' << result.sharding.csc_sharding_us << ','
        << result.sharding.metadata_generation_us << ',' << t.matrix_programming_cycle << ','
        << t.x_transfer_cycle << ',' << t.pim_mode_change_cycle << ','
        << t.crf_programming_cycle << ',' << pim_setup << ',' << t.pim_execution_cycle << ','
        << t.partial_write_cycle << ',' << t.partial_readback_cycle << ','
        << t.total_pim_cycle << ',' << t.pim_time_ms << ',' << result.host_reduction_us << ','
        << result.steady_state_ms << ',' << result.end_to_end_ms << ','
        << (result.validation_passed ? 1 : 0) << '\n';
}

BenchmarkResult runBenchmark(const string& matrix_path, const string& output_path,
                             unsigned iterations)
{
    BenchmarkResult result;
    result.matrix = baseName(matrix_path);
    result.csc = loadCsc(matrix_path);
    vector<float> x(result.csc.cols);
    for (uint32_t col = 0; col < result.csc.cols; ++col)
        x[col] = 0.25f + static_cast<float>((col * 17) % 23) / 23.0f;

    TimingSimulator simulator;
    result.sharding = shardCsc(result.csc, simulator.globalBankGroups());
    vector<vector<float>> partials = generatePartialValues(result.sharding.shards, x);
    vector<float> reference = cpuCscSpmv(result.csc, x);
    vector<float> reduced = reducePartials(result.sharding.shards, partials, result.csc.rows);
    result.validation_passed = almostEqual(reference, reduced);
    result.traffic = buildTrafficStats(result.csc, result.sharding.shards);
    result.host_reduction_us = measureHostReductionUs(
        result.sharding.shards, partials, result.csc.rows, max(1u, iterations));
    result.timing = simulator.run(result.csc, result.sharding.shards);
    double tck_ns = getConfigParam(FLOAT, "tCK");
    double steady_pim_ms =
        static_cast<double>(result.timing.total_pim_cycle -
                            result.timing.matrix_programming_cycle) * tck_ns / 1.0e6;
    result.steady_state_ms = steady_pim_ms + result.host_reduction_us / 1000.0;
    result.end_to_end_ms = result.timing.pim_time_ms + result.host_reduction_us / 1000.0 +
        (result.sharding.metadata_generation_us + result.sharding.csc_sharding_us) / 1000.0;
    writeCsv(output_path, result);
    return result;
}

void validateSynthetic(const CscMatrix& matrix, uint64_t bank_groups)
{
    vector<float> x(matrix.cols);
    iota(x.begin(), x.end(), 1.0f);
    ShardingResult sharding = shardCsc(matrix, bank_groups);
    vector<vector<float>> partials = generatePartialValues(sharding.shards, x);
    EXPECT_TRUE(almostEqual(cpuCscSpmv(matrix, x),
                            reducePartials(sharding.shards, partials, matrix.rows)));
    uint64_t count = 0;
    for (const auto& shard : partials)
        count += shard.size();
    EXPECT_EQ(count, matrix.values.size());
}

} // namespace

TEST(CSCPartialStreamFunctionalTest, RequiredCornerCases)
{
    // Same row across columns; empty column; singleton column; hot row; tail burst.
    CscMatrix mixed = makeCsc(6, 7, {
        {0, 0, 1.0f}, {2, 0, 2.0f}, {0, 1, 3.0f}, {0, 3, 4.0f},
        {0, 4, 5.0f}, {1, 4, 6.0f}, {3, 5, 7.0f}, {4, 5, 8.0f},
        {5, 5, 9.0f},
    });
    validateSynthetic(mixed, 16); // fewer columns than BGs, non-burst-multiple NNZ.

    vector<CooEntry> many_columns;
    for (uint32_t col = 0; col < 37; ++col)
        many_columns.push_back({col % 5, col, 0.5f + col});
    validateSynthetic(makeCsc(5, 37, move(many_columns)), 8); // more columns than BGs.
}

TEST(CSCPartialStreamFunctionalTest, DuplicateEntriesAccumulate)
{
    CscMatrix matrix = makeCsc(2, 2, {{0, 0, 1.0f}, {0, 0, 2.0f}, {0, 1, 3.0f}});
    validateSynthetic(matrix, 4);
    vector<float> y = cpuCscSpmv(matrix, {2.0f, 4.0f});
    EXPECT_FLOAT_EQ(y[0], 18.0f);
}

TEST(CSCPartialStreamBenchmark, RunFromEnvironment)
{
    string matrix_path = envString("CSC_PARTIAL_MATRIX");
    if (matrix_path.empty())
        GTEST_SKIP() << "set CSC_PARTIAL_MATRIX to a Matrix Market or CSC triplet file";
    string output_path = envString(
        "CSC_PARTIAL_OUTPUT", "../SparsePIM/csc_partial_stream_results.csv");
    unsigned iterations = envUnsigned("CSC_PARTIAL_ITERATIONS", 5);
    BenchmarkResult result = runBenchmark(matrix_path, output_path, iterations);
    EXPECT_TRUE(result.validation_passed);
    EXPECT_EQ(result.traffic.partial_value_count, result.csc.values.size());
    EXPECT_EQ(result.traffic.partial_value_bytes,
              result.csc.values.size() * sizeof(float));
    EXPECT_EQ(result.timing.total_pim_cycle,
              result.timing.matrix_programming_cycle + result.timing.x_transfer_cycle +
                  result.timing.pim_mode_change_cycle +
                  result.timing.crf_programming_cycle +
                  result.timing.pim_execution_cycle +
                  result.timing.partial_write_cycle +
                  result.timing.partial_readback_cycle);
    cout << ">>CSC outer-product partial-stream baseline\n"
         << "  matrix: " << result.matrix << "\n"
         << "  rows: " << result.csc.rows << " cols: " << result.csc.cols
         << " nnz: " << result.csc.values.size() << "\n"
         << "  global_bank_groups: " << result.sharding.shards.size() << "\n"
         << "  partial_value_bytes: " << result.traffic.partial_value_bytes << "\n"
         << "  total_pim_cycle: " << result.timing.total_pim_cycle << "\n"
         << "  pim_time_ms: " << result.timing.pim_time_ms << "\n"
         << "  host_reduction_us_median: " << result.host_reduction_us << "\n"
         << "  steady_state_ms: " << result.steady_state_ms << "\n"
         << "  end_to_end_ms: " << result.end_to_end_ms << "\n"
         << "  validation_passed: " << result.validation_passed << "\n"
         << "  output: " << output_path << endl;
}

} // namespace csc_partial_stream
