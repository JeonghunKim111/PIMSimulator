/***************************************************************************************************
 * SparsePIM-style clustered SpMV cycle-only benchmark.
 **************************************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Burst.h"
#include "MultiChannelMemorySystem.h"
#include "PIMCmd.h"
#include "gtest/gtest.h"
#include "tests/PIMCmdGen.h"
#include "tests/PIMKernel.h"

using namespace DRAMSim;
using namespace std;

namespace
{
constexpr unsigned kPimRegRow = 0x3fff;
constexpr unsigned kMacBaseRow = 128;
constexpr unsigned kResultBaseRow = 8192;
constexpr unsigned kResultReadBaseRow = 12288;
constexpr unsigned kElementsPerBurst = 16;
constexpr unsigned kDrafNzesPerColumnGroup = 16;
constexpr unsigned kDrafColumnGroupsPerRow = 7;
constexpr unsigned kBgaEntriesPerBacc = 8;
constexpr unsigned kBgaQueueDepth = 16;
constexpr unsigned kBgaFlushThreshold = 9;
constexpr unsigned kConservativeBgaFlushPenalty = 4;
constexpr unsigned kConservativeHostReduceWidth = 16;
constexpr unsigned kDefaultV2BgaAccCapacity = 64;

struct ClusterInfo
{
    unsigned id = 0;
    uint64_t nnz = 0;
    uint64_t num_cols = 0;
    uint64_t active_rows = 0;
};

struct SpmvInputs
{
    uint64_t n_rows = 0;
    uint64_t n_cols = 0;
    uint64_t nnz = 0;
    vector<ClusterInfo> clusters;
    vector<int> reordered_col_to_cluster;
    vector<vector<unsigned>> rows_by_col;
    uint64_t total_active_row_memberships = 0;
    uint64_t rows_with_any_partial = 0;
};

struct DrafClusterInfo
{
    unsigned id = 0;
    uint64_t column_groups = 0;
    uint64_t draf_rows = 0;
    uint64_t nnz = 0;
    uint64_t active_rows = 0;
};

struct DrafStats
{
    vector<DrafClusterInfo> clusters;
    uint64_t column_groups = 0;
    uint64_t draf_rows = 0;
    uint64_t packed_nnz_capacity = 0;
    uint64_t nnz = 0;
    uint64_t active_row_memberships = 0;
    uint64_t rows_with_any_partial = 0;
    uint64_t nze_padding = 0;
    uint64_t row_group_padding = 0;
    double nze_padding_ratio = 0.0;
    double row_group_padding_ratio = 0.0;
    double expansion_ratio = 0.0;
};

struct BgaGroupInfo
{
    unsigned channel = 0;
    unsigned bank_group = 0;
    uint64_t partials_before = 0;
    uint64_t partials_after = 0;
    uint64_t bacc_instructions = 0;
    uint64_t estimated_flushes = 0;
    uint64_t simple_capacity_flushes = 0;
    uint64_t stream_capacity_flushes = 0;
};

struct BgaStats
{
    vector<BgaGroupInfo> groups;
    uint64_t partials_before = 0;
    uint64_t partials_after = 0;
    uint64_t bga_reduce_ops = 0;
    uint64_t bacc_instructions = 0;
    uint64_t estimated_flushes = 0;
    uint64_t max_bacc_instructions_per_group = 0;
    uint64_t max_estimated_flushes_per_group = 0;
    uint64_t simple_capacity_flushes = 0;
    uint64_t stream_capacity_flushes = 0;
    uint64_t max_simple_capacity_flushes_per_group = 0;
    uint64_t max_stream_capacity_flushes_per_group = 0;
    uint64_t output_readback_tx = 0;
    uint64_t host_reduce_ops_after_bga = 0;
};

struct SpmvDataset
{
    string name;
    string base;
};

struct V2Params
{
    uint64_t bga_acc_capacity = kDefaultV2BgaAccCapacity;
    double skew_alpha = 0.5;
    double final_reduce_skew_alpha = 0.5;
    double padding_alpha = 0.2;
    double locality_beta = 0.3;
    double min_locality_factor = 0.6;
    uint64_t flush_penalty = kConservativeBgaFlushPenalty;
    bool use_stream_flush = false;
};

struct ShapeStats
{
    double row_nnz_cv = 0.0;
    double col_nnz_cv = 0.0;
    double row_nnz_gini = 0.0;
    double col_nnz_gini = 0.0;
    double max_row_nnz_over_mean = 0.0;
    double max_col_nnz_over_mean = 0.0;
    double hot_top1pct_row_nnz_ratio = 0.0;
    double bga_partials_imbalance = 0.0;
    double bga_unique_rows_imbalance = 0.0;
    double hot_bga_group_ratio = 0.0;
    double skew_score = 0.0;
    double bga_reduction_ratio = 0.0;
    double single_nnz_column_ratio = 0.0;
    double low_nnz_column_ratio = 0.0;
    double cluster_row_alignment_ratio = 0.0;
    double draf_row_alignment_ratio = 0.0;
    double sampled_cluster_jaccard = 0.0;
    double draf_padding_pressure = 0.0;
};

struct PaperTarget
{
    string workload;
    string matrix;
    double speedup = 0.0;
};

struct DrafCriticalPathStats
{
    double ideal_steps = 0.0;
    double actual_steps = 0.0;
    double critical_padding = 0.0;
    double critical_padding_ratio = 0.0;
    double group_steps_mean = 0.0;
    double group_steps_max = 0.0;
    double bg_imbalance = 0.0;
};

uint64_t ceilDiv(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

void addTx(shared_ptr<MultiChannelMemorySystem> mem, PIMAddrManager& addr_mgr, bool is_write,
           unsigned chan, unsigned bg, unsigned bank, unsigned row, unsigned col, BurstType* bst)
{
    uint64_t addr = addr_mgr.addrGenSafe(chan, 0, bg, bank, row, col);
    mem->addTransaction(is_write, addr, bst);
}

uint64_t drain(PIMKernel& kernel)
{
    uint64_t before = kernel.getCycle();
    kernel.runPIM();
    return kernel.getCycle() - before;
}

uint64_t envLimit(const char* name)
{
    const char* value = getenv(name);
    if (value == nullptr || value[0] == '\0')
        return 0;
    return strtoull(value, nullptr, 10);
}

string envString(const char* name)
{
    const char* value = getenv(name);
    return value == nullptr ? string() : string(value);
}

double envDouble(const char* name, double default_value)
{
    const char* value = getenv(name);
    if (value == nullptr || value[0] == '\0')
        return default_value;
    return strtod(value, nullptr);
}

uint64_t envUint64(const char* name, uint64_t default_value)
{
    const char* value = getenv(name);
    if (value == nullptr || value[0] == '\0')
        return default_value;
    return strtoull(value, nullptr, 10);
}

bool envBool(const char* name, bool default_value)
{
    const char* value = getenv(name);
    if (value == nullptr || value[0] == '\0')
        return default_value;
    return string(value) != "0";
}

V2Params loadV2Params()
{
    V2Params params;
    params.bga_acc_capacity = envUint64("SPMV_V2_BGA_ACC_CAPACITY", params.bga_acc_capacity);
    params.skew_alpha = envDouble("SPMV_V2_SKEW_ALPHA", params.skew_alpha);
    params.final_reduce_skew_alpha =
        envDouble("SPMV_V2_FINAL_REDUCE_SKEW_ALPHA", params.final_reduce_skew_alpha);
    params.padding_alpha = envDouble("SPMV_V2_PADDING_ALPHA", params.padding_alpha);
    params.locality_beta = envDouble("SPMV_V2_LOCALITY_BETA", params.locality_beta);
    params.min_locality_factor =
        envDouble("SPMV_V2_MIN_LOCALITY_FACTOR", params.min_locality_factor);
    params.flush_penalty = envUint64("SPMV_V2_FLUSH_PENALTY", params.flush_penalty);
    params.use_stream_flush = envBool("SPMV_V2_USE_STREAM_FLUSH", params.use_stream_flush);
    if (params.bga_acc_capacity == 0)
        params.bga_acc_capacity = 1;
    return params;
}

double paperTargetSpeedup(const string& matrix)
{
    static const vector<PaperTarget> targets{
        {"w1", "cant", 2.2},
        {"w2", "crankseg_2", 3.0},
        {"w3", "lhr71", 1.8},
        {"w4", "pdb1HYS", 5.4},
        {"w5", "rma10", 2.3},
        {"w6", "soc-sign-epinions", 1.2},
        {"w7", "Stanford", 1.3},
        {"w8", "bcsstk32", 2.0},
        {"w9", "consph", 2.3},
        {"w10", "ct20stif", 2.1},
        {"w11", "ohne2", 3.2},
        {"w12", "pwtk", 3.6},
        {"w13", "shipsec1", 2.1},
        {"w14", "ASIC_100k", 1.4},
        {"w15", "xenon2", 5.6},
        {"w16", "webbase-1M", 0.7},
    };
    for (const PaperTarget& target : targets)
    {
        if (target.matrix == matrix)
            return target.speedup;
    }
    return 0.0;
}

double gpuBaselineMs(const string& matrix)
{
    static const unordered_map<string, double> baselines{
        {"ASIC_100k", 0.138408},
        {"Stanford", 0.350585},
        {"bcsstk32", 0.099581},
        {"cant", 0.129579},
        {"consph", 0.170734},
        {"crankseg_2", 0.222282},
        {"ct20stif", 0.106881},
        {"lhr71", 0.117986},
        {"ohne2", 0.279932},
        {"pdb1HYS", 0.103166},
        {"pwtk", 0.361986},
        {"rma10", 0.102458},
        {"shipsec1", 0.208423},
        {"soc-sign-epinions", 0.176878},
        {"webbase-1M", 0.867231},
        {"xenon2", 0.234144},
    };
    auto it = baselines.find(matrix);
    return it == baselines.end() ? 0.0 : it->second;
}

void addPhaseBarriers(shared_ptr<MultiChannelMemorySystem> mem, const vector<char>& active_channels)
{
    for (size_t chan = 0; chan < active_channels.size(); ++chan)
    {
        if (active_channels[chan])
            mem->addBarrier(static_cast<int>(chan));
    }
}

SpmvInputs loadSparsePIMInputs(const string& matrix_path, const string& permutation_path,
                               const string& clusters_path)
{
    SpmvInputs inputs;
    ifstream cluster_file(clusters_path);
    if (!cluster_file)
        throw runtime_error("failed to open " + clusters_path);

    string line;
    while (getline(cluster_file, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        istringstream iss(line);
        ClusterInfo info;
        if (!(iss >> info.id >> info.nnz >> info.num_cols))
            continue;
        if (inputs.clusters.size() <= info.id)
            inputs.clusters.resize(info.id + 1);
        inputs.clusters[info.id] = info;
    }

    ifstream permutation_file(permutation_path);
    if (!permutation_file)
        throw runtime_error("failed to open " + permutation_path);

    while (getline(permutation_file, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        istringstream iss(line);
        uint64_t old_col = 0;
        uint64_t new_col = 0;
        uint64_t cluster = 0;
        if (!(iss >> old_col >> new_col >> cluster))
            continue;
        (void)old_col;
        if (inputs.reordered_col_to_cluster.size() <= new_col)
            inputs.reordered_col_to_cluster.resize(new_col + 1, -1);
        inputs.reordered_col_to_cluster[new_col] = static_cast<int>(cluster);
    }

    ifstream matrix_file(matrix_path);
    if (!matrix_file)
        throw runtime_error("failed to open " + matrix_path);
    matrix_file >> inputs.n_rows >> inputs.n_cols >> inputs.nnz;
    inputs.rows_by_col.resize(inputs.n_cols);

    vector<unordered_set<unsigned>> active_rows_by_cluster(inputs.clusters.size());
    vector<char> any_row(inputs.n_rows, 0);

    uint64_t row = 0;
    uint64_t col = 0;
    string value;
    while (matrix_file >> row >> col >> value)
    {
        if (col >= inputs.reordered_col_to_cluster.size())
            throw runtime_error("matrix column exceeds permutation table");
        int cluster = inputs.reordered_col_to_cluster[col];
        if (cluster < 0 || static_cast<size_t>(cluster) >= active_rows_by_cluster.size())
            throw runtime_error("matrix column has no cluster assignment");
        active_rows_by_cluster[cluster].insert(static_cast<unsigned>(row));
        inputs.rows_by_col[col].push_back(static_cast<unsigned>(row));
        any_row[row] = 1;
    }

    for (size_t i = 0; i < inputs.clusters.size(); ++i)
    {
        inputs.clusters[i].active_rows = active_rows_by_cluster[i].size();
        inputs.total_active_row_memberships += inputs.clusters[i].active_rows;
    }
    inputs.rows_with_any_partial = count(any_row.begin(), any_row.end(), 1);
    return inputs;
}

void applyClusterLimit(SpmvInputs& inputs, uint64_t max_clusters)
{
    if (max_clusters == 0 || max_clusters >= inputs.clusters.size())
        return;

    vector<unordered_set<unsigned>> active_rows_by_cluster(max_clusters);
    vector<char> any_row(inputs.n_rows, 0);
    uint64_t filtered_nnz = 0;

    for (size_t col = 0; col < inputs.rows_by_col.size(); ++col)
    {
        int cluster = col < inputs.reordered_col_to_cluster.size()
                          ? inputs.reordered_col_to_cluster[col]
                          : -1;
        if (cluster < 0 || static_cast<uint64_t>(cluster) >= max_clusters)
            continue;
        for (unsigned row : inputs.rows_by_col[col])
        {
            active_rows_by_cluster[cluster].insert(row);
            any_row[row] = 1;
            filtered_nnz++;
        }
    }

    inputs.clusters.resize(max_clusters);
    inputs.total_active_row_memberships = 0;
    inputs.nnz = filtered_nnz;
    for (size_t cluster = 0; cluster < inputs.clusters.size(); ++cluster)
    {
        inputs.clusters[cluster].active_rows = active_rows_by_cluster[cluster].size();
        inputs.total_active_row_memberships += inputs.clusters[cluster].active_rows;
    }
    inputs.rows_with_any_partial = count(any_row.begin(), any_row.end(), 1);
}

DrafStats buildDrafStats(const SpmvInputs& inputs)
{
    DrafStats stats;
    stats.clusters.resize(inputs.clusters.size());
    vector<unordered_set<unsigned>> active_rows_by_cluster(inputs.clusters.size());
    vector<char> any_row(inputs.n_rows, 0);

    for (size_t i = 0; i < inputs.clusters.size(); ++i)
        stats.clusters[i].id = inputs.clusters[i].id;

    for (size_t col = 0; col < inputs.rows_by_col.size(); ++col)
    {
        int cluster = col < inputs.reordered_col_to_cluster.size()
                          ? inputs.reordered_col_to_cluster[col]
                          : -1;
        if (cluster < 0 || static_cast<size_t>(cluster) >= inputs.clusters.size())
            continue;

        const vector<unsigned>& rows = inputs.rows_by_col[col];
        if (rows.empty())
            continue;

        uint64_t groups = ceilDiv(rows.size(), kDrafNzesPerColumnGroup);
        DrafClusterInfo& cluster_info = stats.clusters[cluster];
        cluster_info.column_groups += groups;
        cluster_info.nnz += rows.size();
        stats.column_groups += groups;
        stats.nnz += rows.size();
        stats.nze_padding += groups * kDrafNzesPerColumnGroup - rows.size();

        for (unsigned row : rows)
        {
            active_rows_by_cluster[cluster].insert(row);
            any_row[row] = 1;
        }
    }

    for (size_t cluster = 0; cluster < stats.clusters.size(); ++cluster)
    {
        DrafClusterInfo& cluster_info = stats.clusters[cluster];
        cluster_info.draf_rows = ceilDiv(cluster_info.column_groups, kDrafColumnGroupsPerRow);
        cluster_info.active_rows = active_rows_by_cluster[cluster].size();
        stats.draf_rows += cluster_info.draf_rows;
        stats.active_row_memberships += cluster_info.active_rows;
        stats.row_group_padding +=
            cluster_info.draf_rows * kDrafColumnGroupsPerRow - cluster_info.column_groups;
    }
    stats.packed_nnz_capacity =
        stats.draf_rows * kDrafColumnGroupsPerRow * kDrafNzesPerColumnGroup;
    stats.rows_with_any_partial = count(any_row.begin(), any_row.end(), 1);
    stats.nze_padding_ratio =
        stats.nnz == 0 ? 0.0 : static_cast<double>(stats.nze_padding) / stats.nnz;
    stats.row_group_padding_ratio =
        stats.column_groups == 0
            ? 0.0
            : static_cast<double>(stats.row_group_padding) / stats.column_groups;
    stats.expansion_ratio =
        stats.nnz == 0 ? 0.0 : static_cast<double>(stats.packed_nnz_capacity) / stats.nnz;
    return stats;
}

uint64_t estimateBgaFlushes(uint64_t bacc_instructions)
{
    if (bacc_instructions <= 2)
        return 0;

    uint64_t flushes = 0;
    uint64_t valid_entries = 0;
    for (uint64_t i = 0; i < bacc_instructions; ++i)
    {
        if (valid_entries > kBgaFlushThreshold)
        {
            flushes++;
            valid_entries = 0;
        }
        valid_entries += kBgaEntriesPerBacc;
        if (valid_entries > kBgaQueueDepth)
            valid_entries = kBgaQueueDepth;
    }
    return flushes;
}

uint64_t estimateSimpleCapacityFlushes(uint64_t unique_rows, uint64_t capacity)
{
    if (unique_rows <= capacity)
        return 0;
    return ceilDiv(unique_rows, capacity) - 1;
}

uint64_t estimateStreamCapacityFlushes(const vector<unsigned>& row_stream, uint64_t capacity)
{
    unordered_set<unsigned> active_rows;
    uint64_t flushes = 0;
    for (unsigned row : row_stream)
    {
        if (active_rows.find(row) != active_rows.end())
            continue;
        if (active_rows.size() >= capacity)
        {
            flushes++;
            active_rows.clear();
        }
        active_rows.insert(row);
    }
    return flushes;
}

BgaStats buildBgaStats(const SpmvInputs& inputs,
                       uint64_t bga_acc_capacity = kDefaultV2BgaAccCapacity)
{
    BgaStats stats;
    constexpr unsigned kLogicalBankGroups = 4;
    uint64_t num_groups = static_cast<uint64_t>(64) * kLogicalBankGroups;
    vector<uint64_t> partials_by_group(num_groups, 0);
    vector<unordered_set<unsigned>> rows_by_group(num_groups);
    vector<vector<unsigned>> row_stream_by_group(num_groups);

    for (size_t col = 0; col < inputs.rows_by_col.size(); ++col)
    {
        int cluster = col < inputs.reordered_col_to_cluster.size()
                          ? inputs.reordered_col_to_cluster[col]
                          : -1;
        if (cluster < 0 || static_cast<size_t>(cluster) >= inputs.clusters.size())
            continue;

        unsigned channel = static_cast<unsigned>(cluster) / kLogicalBankGroups;
        unsigned bank_group = static_cast<unsigned>(cluster) % kLogicalBankGroups;
        uint64_t group_idx = channel * kLogicalBankGroups + bank_group;
        for (unsigned row : inputs.rows_by_col[col])
        {
            partials_by_group[group_idx]++;
            rows_by_group[group_idx].insert(row);
            row_stream_by_group[group_idx].push_back(row);
        }
    }

    vector<uint16_t> row_bga_memberships(inputs.n_rows, 0);
    for (uint64_t group_idx = 0; group_idx < num_groups; ++group_idx)
    {
        if (partials_by_group[group_idx] == 0)
            continue;

        BgaGroupInfo info;
        info.channel = group_idx / kLogicalBankGroups;
        info.bank_group = group_idx % kLogicalBankGroups;
        info.partials_before = partials_by_group[group_idx];
        info.partials_after = rows_by_group[group_idx].size();
        info.bacc_instructions = ceilDiv(info.partials_before, kBgaEntriesPerBacc);
        info.estimated_flushes = estimateBgaFlushes(info.bacc_instructions);
        info.simple_capacity_flushes =
            estimateSimpleCapacityFlushes(info.partials_after, bga_acc_capacity);
        info.stream_capacity_flushes =
            estimateStreamCapacityFlushes(row_stream_by_group[group_idx], bga_acc_capacity);

        stats.partials_before += info.partials_before;
        stats.partials_after += info.partials_after;
        stats.bacc_instructions += info.bacc_instructions;
        stats.estimated_flushes += info.estimated_flushes;
        stats.simple_capacity_flushes += info.simple_capacity_flushes;
        stats.stream_capacity_flushes += info.stream_capacity_flushes;
        stats.max_bacc_instructions_per_group =
            max(stats.max_bacc_instructions_per_group, info.bacc_instructions);
        stats.max_estimated_flushes_per_group =
            max(stats.max_estimated_flushes_per_group, info.estimated_flushes);
        stats.max_simple_capacity_flushes_per_group =
            max(stats.max_simple_capacity_flushes_per_group, info.simple_capacity_flushes);
        stats.max_stream_capacity_flushes_per_group =
            max(stats.max_stream_capacity_flushes_per_group, info.stream_capacity_flushes);
        stats.groups.push_back(info);

        for (unsigned row : rows_by_group[group_idx])
            row_bga_memberships[row]++;
    }

    stats.bga_reduce_ops = stats.partials_before > stats.partials_after
                               ? stats.partials_before - stats.partials_after
                               : 0;
    stats.output_readback_tx = ceilDiv(stats.partials_after, kElementsPerBurst);
    for (uint16_t memberships : row_bga_memberships)
    {
        if (memberships > 1)
            stats.host_reduce_ops_after_bga += memberships - 1;
    }
    return stats;
}

double meanOf(const vector<uint64_t>& values)
{
    if (values.empty())
        return 0.0;
    double sum = 0.0;
    for (uint64_t value : values)
        sum += static_cast<double>(value);
    return sum / values.size();
}

double coefficientOfVariation(const vector<uint64_t>& values)
{
    double mean = meanOf(values);
    if (mean == 0.0)
        return 0.0;
    double sum_sq = 0.0;
    for (uint64_t value : values)
    {
        double diff = static_cast<double>(value) - mean;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / values.size()) / mean;
}

double giniCoefficient(vector<uint64_t> values)
{
    if (values.empty())
        return 0.0;
    sort(values.begin(), values.end());
    double sum = 0.0;
    double weighted_sum = 0.0;
    for (size_t i = 0; i < values.size(); ++i)
    {
        sum += static_cast<double>(values[i]);
        weighted_sum += static_cast<double>(i + 1) * values[i];
    }
    if (sum == 0.0)
        return 0.0;
    double n = static_cast<double>(values.size());
    return (2.0 * weighted_sum) / (n * sum) - (n + 1.0) / n;
}

double maxOverMean(const vector<uint64_t>& values)
{
    double mean = meanOf(values);
    if (mean == 0.0)
        return 0.0;
    uint64_t max_value = 0;
    for (uint64_t value : values)
        max_value = max(max_value, value);
    return static_cast<double>(max_value) / mean;
}

double topPercentRatio(vector<uint64_t> values, double percent)
{
    if (values.empty())
        return 0.0;
    double total = 0.0;
    for (uint64_t value : values)
        total += static_cast<double>(value);
    if (total == 0.0)
        return 0.0;
    sort(values.begin(), values.end(), greater<uint64_t>());
    size_t top_count =
        max<size_t>(1, static_cast<size_t>(ceil(values.size() * percent / 100.0)));
    double top_sum = 0.0;
    for (size_t i = 0; i < top_count && i < values.size(); ++i)
        top_sum += static_cast<double>(values[i]);
    return top_sum / total;
}

double normalizeRatio(double value, double scale)
{
    if (value <= 1.0)
        return 0.0;
    return min(1.0, log(value) / log(scale));
}

double jaccardOfSortedRows(vector<unsigned> lhs, vector<unsigned> rhs)
{
    if (lhs.empty() && rhs.empty())
        return 0.0;
    sort(lhs.begin(), lhs.end());
    sort(rhs.begin(), rhs.end());
    size_t i = 0;
    size_t j = 0;
    uint64_t intersection = 0;
    uint64_t union_count = 0;
    while (i < lhs.size() || j < rhs.size())
    {
        if (j >= rhs.size() || (i < lhs.size() && lhs[i] < rhs[j]))
        {
            union_count++;
            i++;
        }
        else if (i >= lhs.size() || rhs[j] < lhs[i])
        {
            union_count++;
            j++;
        }
        else
        {
            intersection++;
            union_count++;
            i++;
            j++;
        }
    }
    return union_count == 0 ? 0.0 : static_cast<double>(intersection) / union_count;
}

double sampledClusterJaccard(const SpmvInputs& inputs)
{
    constexpr size_t kMaxColumnsPerCluster = 32;
    vector<vector<size_t>> sampled_cols_by_cluster(inputs.clusters.size());
    vector<uint64_t> sampled_nnz_by_cluster(inputs.clusters.size(), 0);

    for (size_t col = 0; col < inputs.rows_by_col.size(); ++col)
    {
        if (inputs.rows_by_col[col].empty())
            continue;
        int cluster = col < inputs.reordered_col_to_cluster.size()
                          ? inputs.reordered_col_to_cluster[col]
                          : -1;
        if (cluster < 0 || static_cast<size_t>(cluster) >= inputs.clusters.size())
            continue;
        vector<size_t>& sampled_cols = sampled_cols_by_cluster[cluster];
        if (sampled_cols.size() < kMaxColumnsPerCluster)
            sampled_cols.push_back(col);
        sampled_nnz_by_cluster[cluster] += inputs.rows_by_col[col].size();
    }

    double weighted_jaccard = 0.0;
    double total_weight = 0.0;
    for (size_t cluster = 0; cluster < sampled_cols_by_cluster.size(); ++cluster)
    {
        const vector<size_t>& sampled_cols = sampled_cols_by_cluster[cluster];
        if (sampled_cols.size() < 2)
            continue;

        double sum = 0.0;
        uint64_t pairs = 0;
        for (size_t i = 0; i < sampled_cols.size(); ++i)
        {
            for (size_t j = i + 1; j < sampled_cols.size(); ++j)
            {
                sum += jaccardOfSortedRows(inputs.rows_by_col[sampled_cols[i]],
                                           inputs.rows_by_col[sampled_cols[j]]);
                pairs++;
            }
        }
        if (pairs == 0)
            continue;
        double weight = static_cast<double>(sampled_nnz_by_cluster[cluster]);
        weighted_jaccard += (sum / pairs) * weight;
        total_weight += weight;
    }
    return total_weight == 0.0 ? 0.0 : weighted_jaccard / total_weight;
}

ShapeStats buildShapeStats(const SpmvInputs& inputs, const DrafStats& draf, const BgaStats& bga)
{
    ShapeStats stats;
    vector<uint64_t> col_nnz(inputs.rows_by_col.size(), 0);
    vector<uint64_t> row_nnz(inputs.n_rows, 0);
    uint64_t single_nnz_columns = 0;
    uint64_t low_nnz_columns = 0;
    uint64_t nonempty_columns = 0;
    for (size_t col = 0; col < inputs.rows_by_col.size(); ++col)
    {
        col_nnz[col] = inputs.rows_by_col[col].size();
        if (col_nnz[col] > 0)
        {
            nonempty_columns++;
            if (col_nnz[col] == 1)
                single_nnz_columns++;
            if (col_nnz[col] < kDrafNzesPerColumnGroup)
                low_nnz_columns++;
        }
        for (unsigned row : inputs.rows_by_col[col])
            row_nnz[row]++;
    }

    stats.row_nnz_cv = coefficientOfVariation(row_nnz);
    stats.col_nnz_cv = coefficientOfVariation(col_nnz);
    stats.row_nnz_gini = giniCoefficient(row_nnz);
    stats.col_nnz_gini = giniCoefficient(col_nnz);
    stats.max_row_nnz_over_mean = maxOverMean(row_nnz);
    stats.max_col_nnz_over_mean = maxOverMean(col_nnz);
    stats.hot_top1pct_row_nnz_ratio = topPercentRatio(row_nnz, 1.0);

    vector<uint64_t> group_partials;
    vector<uint64_t> group_unique_rows;
    for (const BgaGroupInfo& group : bga.groups)
    {
        group_partials.push_back(group.partials_before);
        group_unique_rows.push_back(group.partials_after);
    }
    stats.bga_partials_imbalance = maxOverMean(group_partials);
    stats.bga_unique_rows_imbalance = maxOverMean(group_unique_rows);
    uint64_t max_group_partials = 0;
    for (uint64_t partials : group_partials)
        max_group_partials = max(max_group_partials, partials);
    stats.hot_bga_group_ratio =
        bga.partials_before == 0
            ? 0.0
            : static_cast<double>(max_group_partials) / bga.partials_before;
    stats.bga_reduction_ratio =
        bga.partials_before == 0
            ? 0.0
            : 1.0 - static_cast<double>(bga.partials_after) / bga.partials_before;
    stats.single_nnz_column_ratio =
        nonempty_columns == 0 ? 0.0 : static_cast<double>(single_nnz_columns) / nonempty_columns;
    stats.low_nnz_column_ratio =
        nonempty_columns == 0 ? 0.0 : static_cast<double>(low_nnz_columns) / nonempty_columns;

    uint64_t cluster_row_alignment_padding = 0;
    for (const ClusterInfo& cluster : inputs.clusters)
        cluster_row_alignment_padding += (4 - (cluster.active_rows % 4)) % 4;
    stats.cluster_row_alignment_ratio =
        inputs.total_active_row_memberships == 0
            ? 0.0
            : static_cast<double>(cluster_row_alignment_padding) /
                  inputs.total_active_row_memberships;

    uint64_t draf_row_alignment_padding = 0;
    for (const DrafClusterInfo& cluster : draf.clusters)
        draf_row_alignment_padding += (4 - (cluster.draf_rows % 4)) % 4;
    stats.draf_row_alignment_ratio =
        draf.draf_rows == 0 ? 0.0
                            : static_cast<double>(draf_row_alignment_padding) / draf.draf_rows;
    stats.sampled_cluster_jaccard = sampledClusterJaccard(inputs);
    stats.draf_padding_pressure =
        min(1.0, 0.50 * normalizeRatio(draf.expansion_ratio, 8.0) +
                     0.25 * stats.single_nnz_column_ratio +
                     0.25 * stats.low_nnz_column_ratio);

    double row_skew = 0.5 * stats.row_nnz_gini +
                      0.5 * normalizeRatio(stats.max_row_nnz_over_mean, 128.0);
    double col_skew = 0.5 * stats.col_nnz_gini +
                      0.5 * normalizeRatio(stats.max_col_nnz_over_mean, 128.0);
    double bga_skew = 0.5 * normalizeRatio(stats.bga_partials_imbalance, 16.0) +
                      0.5 * normalizeRatio(stats.bga_unique_rows_imbalance, 16.0);
    double hot_skew = min(1.0, stats.hot_top1pct_row_nnz_ratio * 2.0);
    stats.skew_score = min(1.0, 0.30 * row_skew + 0.20 * col_skew + 0.30 * bga_skew +
                                    0.20 * hot_skew);
    return stats;
}

DrafCriticalPathStats buildDrafCriticalPathStats(const DrafStats& draf)
{
    DrafCriticalPathStats stats;
    constexpr double kLogicalBanksPerBankGroup = 2.0;
    vector<uint64_t> actual_group_steps;

    for (const DrafClusterInfo& cluster : draf.clusters)
    {
        if (cluster.column_groups == 0)
            continue;

        double actual_group_step =
            ceil(static_cast<double>(cluster.column_groups) / kLogicalBanksPerBankGroup);
        double ideal_column_groups =
            static_cast<double>(cluster.nnz) / static_cast<double>(kDrafNzesPerColumnGroup);
        double ideal_group_step = ideal_column_groups / kLogicalBanksPerBankGroup;

        stats.actual_steps += actual_group_step;
        stats.ideal_steps += ideal_group_step;
        stats.critical_padding += max(0.0, actual_group_step - ideal_group_step);
        actual_group_steps.push_back(static_cast<uint64_t>(ceil(actual_group_step)));
    }

    stats.critical_padding_ratio =
        stats.ideal_steps == 0.0 ? 0.0 : stats.critical_padding / stats.ideal_steps;
    stats.group_steps_mean = meanOf(actual_group_steps);
    stats.group_steps_max = actual_group_steps.empty()
                                ? 0.0
                                : static_cast<double>(
                                      *max_element(actual_group_steps.begin(),
                                                   actual_group_steps.end()));
    stats.bg_imbalance =
        stats.group_steps_mean == 0.0 ? 0.0 : stats.group_steps_max / stats.group_steps_mean;
    return stats;
}

class ClusteredSpmvBenchFixture : public testing::Test
{
  protected:
    void SetUp() override
    {
        resetPIMKernel();
    }

    void resetPIMKernel()
    {
        mem_ = make_shared<MultiChannelMemorySystem>("ini/HBM2_samsung_2M_16B_x64.ini",
                                                     "system_hbm_64ch.ini", ".", "spmv_bench",
                                                     256 * 64 * 2);
        kernel_ = make_shared<PIMKernel>(mem_, 64, 1);
    }

    shared_ptr<MultiChannelMemorySystem> mem_;
    shared_ptr<PIMKernel> kernel_;
    void runDrafBgaModel(const string& base, const string& input_name,
                         bool conservative_model = false, bool v2_model = false,
                         bool v21_model = false);
    void runDrafBgaStructuralModel(const string& base, const string& input_name,
                                   const string& matrix_name,
                                   bool critical_padding_model = false);
    void runGuidedKmeansDrafBgaSuite(bool conservative_model = false,
                                     bool v2_model = false, bool v21_model = false);
    void runGuidedKmeansDrafBgaStructuralSuite(bool critical_padding_model = false);
};
}  // namespace

TEST_F(ClusteredSpmvBenchFixture, sparsepim_cluster_cantcoo)
{
    const string base = "../SparsePIM/cluster_cantcoo/";
    SpmvInputs inputs = loadSparsePIMInputs(base + "reordered_matrix.txt",
                                            base + "column_permutation.txt", base + "clusters.txt");
    uint64_t max_clusters = envLimit("SPMV_BENCH_MAX_CLUSTERS");
    applyClusterLimit(inputs, max_clusters);

    BurstType null_bst;
    vector<PIMCmd> mac_cmds{
        PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK, 1),
        PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A, PIMOpdType::ODD_BANK, 1),
        PIMCmd(PIMCmdType::NOP, 7),
        PIMCmd(PIMCmdType::EXIT, 0),
    };

    uint64_t input_load_tx = 0;
    uint64_t mac_tx = 0;
    uint64_t partial_write_tx = 0;
    uint64_t partial_read_tx = 0;
    for (const ClusterInfo& cluster : inputs.clusters)
    {
        input_load_tx += ceilDiv(cluster.num_cols, kElementsPerBurst);
        mac_tx += ceilDiv(cluster.nnz, kElementsPerBurst);
        uint64_t result_bursts = ceilDiv(cluster.active_rows, kElementsPerBurst);
        partial_write_tx += result_bursts;
        partial_read_tx += result_bursts;
    }

    cout << ">>Clustered SpMV DRAMSim transaction plan" << endl;
    cout << "  input: SparsePIM/cluster_cantcoo" << endl;
    cout << "  rows: " << inputs.n_rows << endl;
    cout << "  cols: " << inputs.n_cols << endl;
    cout << "  nnz: " << inputs.nnz << endl;
    cout << "  clusters: " << inputs.clusters.size() << endl;
    if (max_clusters > 0)
        cout << "  SPMV_BENCH_MAX_CLUSTERS: " << max_clusters << endl;
    cout << "  planned_input_load_tx: " << input_load_tx << endl;
    cout << "  planned_mac_tx: " << mac_tx << endl;
    cout << "  planned_partial_write_tx: " << partial_write_tx << endl;
    cout << "  planned_partial_read_tx: " << partial_read_tx << endl;
    cout << "  note: cycle-accurate DRAMSim drain can take several minutes for full input"
         << endl;

    cout << "  phase: setup" << endl;
    kernel_->configurePIMControl();
    kernel_->parkIn();
    kernel_->changePIMMode(dramMode::SB, dramMode::HAB);
    kernel_->programCrf(mac_cmds);
    uint64_t excluded_setup_cycle = drain(*kernel_);

    cout << "  phase: input_load" << endl;
    vector<char> input_channels(kernel_->num_pim_chans_, 0);
    for (const ClusterInfo& cluster : inputs.clusters)
    {
        unsigned chan = cluster.id / 4;
        input_channels[chan] = 1;
        uint64_t bursts = ceilDiv(cluster.num_cols, kElementsPerBurst);
        for (uint64_t i = 0; i < bursts; ++i)
        {
            unsigned col = 0x8 + (i % 8);
            uint64_t addr = kernel_->pim_addr_mgr_->addrGen(chan, 0, 0, 1, kPimRegRow, col);
            mem_->addTransaction(true, addr, &null_bst);
        }
    }
    addPhaseBarriers(mem_, input_channels);
    uint64_t input_load_cycle = drain(*kernel_);

    cout << "  phase: pim_enable" << endl;
    kernel_->changePIMMode(dramMode::HAB, dramMode::HAB_PIM);
    uint64_t pim_enable_cycle = drain(*kernel_);

    cout << "  phase: pim_compute" << endl;
    vector<char> compute_channels(kernel_->num_pim_chans_, 0);
    for (const ClusterInfo& cluster : inputs.clusters)
    {
        unsigned chan = cluster.id / 4;
        uint64_t mac_bursts = ceilDiv(cluster.nnz, kElementsPerBurst);
        compute_channels[chan] = 1;

        for (uint64_t i = 0; i < mac_bursts; ++i)
        {
            unsigned row = kMacBaseRow + cluster.id;
            unsigned col = i;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, chan, 0, i % 2, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, compute_channels);

    for (const ClusterInfo& cluster : inputs.clusters)
    {
        unsigned chan = cluster.id / 4;
        uint64_t result_bursts = ceilDiv(cluster.active_rows, kElementsPerBurst);
        for (uint64_t i = 0; i < result_bursts; ++i)
        {
            unsigned row = kResultBaseRow + cluster.id;
            unsigned col = i;
            addTx(mem_, *kernel_->pim_addr_mgr_, true, chan, 0, 0, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, compute_channels);
    uint64_t pim_compute_cycle = drain(*kernel_);

    cout << "  phase: pim_disable" << endl;
    kernel_->changePIMMode(dramMode::HAB_PIM, dramMode::HAB);
    uint64_t pim_disable_cycle = drain(*kernel_);
    cout << "  phase: pim_to_sb" << endl;
    kernel_->changePIMMode(dramMode::HAB, dramMode::SB);
    uint64_t pim_to_sb_cycle = drain(*kernel_);

    cout << "  phase: partial_readback" << endl;
    vector<char> readback_channels(kernel_->num_pim_chans_, 0);
    for (const ClusterInfo& cluster : inputs.clusters)
    {
        unsigned chan = cluster.id / 4;
        uint64_t result_bursts = ceilDiv(cluster.active_rows, kElementsPerBurst);
        readback_channels[chan] = 1;
        for (uint64_t i = 0; i < result_bursts; ++i)
        {
            unsigned row = kResultReadBaseRow + cluster.id;
            unsigned col = i;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, chan, 0, 0, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, readback_channels);
    uint64_t partial_readback_cycle = drain(*kernel_);

    uint64_t host_reduce_ops =
        inputs.total_active_row_memberships > inputs.rows_with_any_partial
            ? inputs.total_active_row_memberships - inputs.rows_with_any_partial
            : 0;

    cout << ">>Clustered SpMV Per-Run Cycle Model" << endl;
    cout << "  input: SparsePIM/cluster_cantcoo" << endl;
    cout << "  rows: " << inputs.n_rows << endl;
    cout << "  cols: " << inputs.n_cols << endl;
    cout << "  nnz: " << inputs.nnz << endl;
    cout << "  clusters: " << inputs.clusters.size() << endl;
    cout << "  mapping: cluster_id -> channel=floor(id/4), bank_group=id%4" << endl;
    cout << "  simulator_physical_pim_bg: 0" << endl;
    cout << "  excluded_setup_cycle: " << excluded_setup_cycle << endl;
    cout << "> input_load_cycle: " << input_load_cycle << " tx=" << input_load_tx << endl;
    cout << "> pim_enable_cycle: " << pim_enable_cycle << endl;
    cout << "> pim_compute_cycle: " << pim_compute_cycle << " mac_tx=" << mac_tx
         << " partial_write_tx=" << partial_write_tx << endl;
    cout << "> pim_disable_cycle: " << pim_disable_cycle << endl;
    cout << "> pim_to_sb_cycle: " << pim_to_sb_cycle << endl;
    cout << "> partial_readback_cycle: " << partial_readback_cycle
         << " tx=" << partial_read_tx << endl;
    cout << "> host_reduce_ops: " << host_reduce_ops << endl;
    cout << "> per_spmv_simulated_cycle: "
         << input_load_cycle + pim_enable_cycle + pim_compute_cycle + pim_disable_cycle +
                pim_to_sb_cycle + partial_readback_cycle
         << endl;
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_cluster_cantcoo_draf_model)
{
    const string base = "../SparsePIM/cluster_cantcoo/";
    SpmvInputs inputs = loadSparsePIMInputs(base + "reordered_matrix.txt",
                                            base + "column_permutation.txt", base + "clusters.txt");
    uint64_t max_clusters = envLimit("SPMV_BENCH_MAX_CLUSTERS");
    applyClusterLimit(inputs, max_clusters);
    DrafStats draf = buildDrafStats(inputs);

    BurstType null_bst;
    vector<PIMCmd> mac_cmds{
        PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK, 1),
        PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A, PIMOpdType::ODD_BANK, 1),
        PIMCmd(PIMCmdType::NOP, 7),
        PIMCmd(PIMCmdType::EXIT, 0),
    };

    cout << ">>DRAF-aware SpMV synthetic transaction plan" << endl;
    cout << "  model: DRAF-aware synthetic DRAMSim model" << endl;
    cout << "  note: not a bit-accurate DRAF memory image" << endl;
    cout << "  input: SparsePIM/cluster_cantcoo" << endl;
    cout << "  rows: " << inputs.n_rows << endl;
    cout << "  cols: " << inputs.n_cols << endl;
    cout << "  nnz: " << draf.nnz << endl;
    cout << "  clusters: " << inputs.clusters.size() << endl;
    if (max_clusters > 0)
        cout << "  SPMV_BENCH_MAX_CLUSTERS: " << max_clusters << endl;
    cout << "  draf_column_group_capacity_nnz: " << kDrafNzesPerColumnGroup << endl;
    cout << "  draf_column_groups_per_row: " << kDrafColumnGroupsPerRow << endl;
    cout << "  draf_column_groups: " << draf.column_groups << endl;
    cout << "  draf_rows: " << draf.draf_rows << endl;
    cout << "  packed_nnz_capacity: " << draf.packed_nnz_capacity << endl;
    cout << "  planned_draf_row_fetch_tx: " << draf.draf_rows << endl;
    cout << "  planned_draf_compute_trigger_tx: " << draf.draf_rows << endl;
    cout << "  planned_draf_partial_write_tx: " << draf.draf_rows << endl;
    cout << "  planned_draf_partial_read_tx: " << draf.draf_rows << endl;

    cout << "  phase: setup" << endl;
    kernel_->configurePIMControl();
    kernel_->parkIn();
    kernel_->changePIMMode(dramMode::SB, dramMode::HAB);
    kernel_->programCrf(mac_cmds);
    uint64_t excluded_setup_cycle = drain(*kernel_);

    cout << "  phase: draf_row_fetch" << endl;
    vector<char> draf_channels(kernel_->num_pim_chans_, 0);
    uint64_t global_draf_row = 0;
    for (const DrafClusterInfo& cluster : draf.clusters)
    {
        unsigned chan = cluster.id / 4;
        draf_channels[chan] = cluster.draf_rows > 0 ? 1 : draf_channels[chan];
        for (uint64_t i = 0; i < cluster.draf_rows; ++i, ++global_draf_row)
        {
            unsigned row = kMacBaseRow + (global_draf_row / 32);
            unsigned col = global_draf_row % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, chan, 0, 0, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, draf_channels);
    uint64_t draf_row_fetch_cycle = drain(*kernel_);

    cout << "  phase: pim_enable" << endl;
    kernel_->changePIMMode(dramMode::HAB, dramMode::HAB_PIM);
    uint64_t pim_enable_cycle = drain(*kernel_);

    cout << "  phase: draf_compute" << endl;
    global_draf_row = 0;
    for (const DrafClusterInfo& cluster : draf.clusters)
    {
        unsigned chan = cluster.id / 4;
        for (uint64_t i = 0; i < cluster.draf_rows; ++i, ++global_draf_row)
        {
            unsigned row = kMacBaseRow + (global_draf_row / 32);
            unsigned col = global_draf_row % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, chan, 0, i % 2, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, draf_channels);

    global_draf_row = 0;
    for (const DrafClusterInfo& cluster : draf.clusters)
    {
        unsigned chan = cluster.id / 4;
        for (uint64_t i = 0; i < cluster.draf_rows; ++i, ++global_draf_row)
        {
            unsigned row = kResultBaseRow + (global_draf_row / 32);
            unsigned col = global_draf_row % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, true, chan, 0, 0, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, draf_channels);
    uint64_t draf_compute_cycle = drain(*kernel_);

    cout << "  phase: pim_disable" << endl;
    kernel_->changePIMMode(dramMode::HAB_PIM, dramMode::HAB);
    uint64_t pim_disable_cycle = drain(*kernel_);
    cout << "  phase: pim_to_sb" << endl;
    kernel_->changePIMMode(dramMode::HAB, dramMode::SB);
    uint64_t pim_to_sb_cycle = drain(*kernel_);

    cout << "  phase: draf_partial_readback" << endl;
    global_draf_row = 0;
    for (const DrafClusterInfo& cluster : draf.clusters)
    {
        unsigned chan = cluster.id / 4;
        for (uint64_t i = 0; i < cluster.draf_rows; ++i, ++global_draf_row)
        {
            unsigned row = kResultReadBaseRow + (global_draf_row / 32);
            unsigned col = global_draf_row % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, chan, 0, 0, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, draf_channels);
    uint64_t draf_partial_readback_cycle = drain(*kernel_);

    uint64_t host_reduce_ops =
        draf.active_row_memberships > draf.rows_with_any_partial
            ? draf.active_row_memberships - draf.rows_with_any_partial
            : 0;

    cout << ">>DRAF-aware SpMV Per-Run Cycle Model" << endl;
    cout << "  input: SparsePIM/cluster_cantcoo" << endl;
    cout << "  rows: " << inputs.n_rows << endl;
    cout << "  cols: " << inputs.n_cols << endl;
    cout << "  nnz: " << draf.nnz << endl;
    cout << "  clusters: " << inputs.clusters.size() << endl;
    cout << "  draf_column_groups: " << draf.column_groups << endl;
    cout << "  draf_rows: " << draf.draf_rows << endl;
    cout << "  simulator_physical_pim_bg: 0" << endl;
    cout << "  excluded_setup_cycle: " << excluded_setup_cycle << endl;
    cout << "> draf_row_fetch_cycle: " << draf_row_fetch_cycle
         << " tx=" << draf.draf_rows << endl;
    cout << "> pim_enable_cycle: " << pim_enable_cycle << endl;
    cout << "> draf_compute_cycle: " << draf_compute_cycle
         << " compute_trigger_tx=" << draf.draf_rows
         << " partial_write_tx=" << draf.draf_rows << endl;
    cout << "> pim_disable_cycle: " << pim_disable_cycle << endl;
    cout << "> pim_to_sb_cycle: " << pim_to_sb_cycle << endl;
    cout << "> draf_partial_readback_cycle: " << draf_partial_readback_cycle
         << " tx=" << draf.draf_rows << endl;
    cout << "> host_reduce_ops: " << host_reduce_ops << endl;
    cout << "> per_spmv_draf_model_cycle: "
         << draf_row_fetch_cycle + pim_enable_cycle + draf_compute_cycle + pim_disable_cycle +
                pim_to_sb_cycle + draf_partial_readback_cycle
         << endl;
}

void ClusteredSpmvBenchFixture::runDrafBgaModel(const string& base, const string& input_name,
                                                bool conservative_model, bool v2_model,
                                                bool v21_model)
{
    if (v21_model)
        v2_model = true;
    SpmvInputs inputs = loadSparsePIMInputs(base + "reordered_matrix.txt",
                                            base + "column_permutation.txt", base + "clusters.txt");
    uint64_t max_clusters = envLimit("SPMV_BENCH_MAX_CLUSTERS");
    applyClusterLimit(inputs, max_clusters);
    V2Params v2_params = loadV2Params();
    DrafStats draf = buildDrafStats(inputs);
    BgaStats bga = buildBgaStats(inputs, v2_params.bga_acc_capacity);
    ShapeStats shape = buildShapeStats(inputs, draf, bga);

    BurstType null_bst;
    vector<PIMCmd> mac_cmds{
        PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK, 1),
        PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A, PIMOpdType::ODD_BANK, 1),
        PIMCmd(PIMCmdType::NOP, 7),
        PIMCmd(PIMCmdType::EXIT, 0),
    };

    cout << ">>DRAF+BGA-aware SpMV synthetic transaction plan" << endl;
    cout << "  model: "
         << (v21_model ? "DRAF+BGA-aware v2.1 correction synthetic DRAMSim model"
                       : (v2_model ? "DRAF+BGA-aware v2 correction synthetic DRAMSim model"
                      : (conservative_model
                             ? "DRAF+BGA-aware conservative synthetic DRAMSim model"
                             : "DRAF+BGA-aware synthetic DRAMSim model")))
         << endl;
    cout << "  note: not a bit-accurate DRAF/BGA microarchitecture" << endl;
    cout << "  input: " << input_name << endl;
    cout << "  rows: " << inputs.n_rows << endl;
    cout << "  cols: " << inputs.n_cols << endl;
    cout << "  nnz: " << draf.nnz << endl;
    cout << "  clusters: " << inputs.clusters.size() << endl;
    if (max_clusters > 0)
        cout << "  SPMV_BENCH_MAX_CLUSTERS: " << max_clusters << endl;
    cout << "  draf_column_group_capacity_nnz: " << kDrafNzesPerColumnGroup << endl;
    cout << "  draf_column_groups_per_row: " << kDrafColumnGroupsPerRow << endl;
    cout << "  bga_entries_per_bacc: " << kBgaEntriesPerBacc << endl;
    cout << "  bga_queue_depth: " << kBgaQueueDepth << endl;
    cout << "  bga_flush_threshold: " << kBgaFlushThreshold << endl;
    if (conservative_model)
    {
        cout << "  conservative_bga_flush_penalty: " << kConservativeBgaFlushPenalty << endl;
        cout << "  conservative_host_reduce_width: " << kConservativeHostReduceWidth << endl;
    }
    if (v2_model)
    {
        cout << "  v2_bga_acc_capacity: " << v2_params.bga_acc_capacity << endl;
        cout << "  v2_use_stream_flush: " << (v2_params.use_stream_flush ? 1 : 0) << endl;
        cout << "  v2_skew_alpha: " << v2_params.skew_alpha << endl;
        cout << "  v2_final_reduce_skew_alpha: " << v2_params.final_reduce_skew_alpha
             << endl;
        cout << "  v2_padding_alpha: " << v2_params.padding_alpha << endl;
        cout << "  v2_locality_beta: " << v2_params.locality_beta << endl;
        cout << "  v2_min_locality_factor: " << v2_params.min_locality_factor << endl;
        cout << "  v2_flush_penalty: " << v2_params.flush_penalty << endl;
    }
    cout << "  draf_column_groups: " << draf.column_groups << endl;
    cout << "  draf_rows: " << draf.draf_rows << endl;
    cout << "  draf_expansion_ratio: " << draf.expansion_ratio << endl;
    cout << "  draf_nze_padding_ratio: " << draf.nze_padding_ratio << endl;
    cout << "  draf_row_group_padding_ratio: " << draf.row_group_padding_ratio << endl;
    cout << "  active_bga_groups: " << bga.groups.size() << endl;
    cout << "  bga_partials_before: " << bga.partials_before << endl;
    cout << "  bga_partials_after: " << bga.partials_after << endl;
    cout << "  bga_reduce_ops: " << bga.bga_reduce_ops << endl;
    cout << "  bacc_instructions: " << bga.bacc_instructions << endl;
    cout << "  max_bacc_instructions_per_group: " << bga.max_bacc_instructions_per_group
         << endl;
    cout << "  estimated_bga_flushes: " << bga.estimated_flushes << endl;
    cout << "  max_estimated_flushes_per_group: " << bga.max_estimated_flushes_per_group
         << endl;
    cout << "  simple_capacity_flushes: " << bga.simple_capacity_flushes << endl;
    cout << "  max_simple_capacity_flushes_per_group: "
         << bga.max_simple_capacity_flushes_per_group << endl;
    cout << "  stream_capacity_flushes: " << bga.stream_capacity_flushes << endl;
    cout << "  max_stream_capacity_flushes_per_group: "
         << bga.max_stream_capacity_flushes_per_group << endl;
    cout << "  bga_output_readback_tx: " << bga.output_readback_tx << endl;
    cout << "  host_reduce_ops_after_bga: " << bga.host_reduce_ops_after_bga << endl;
    if (v2_model)
    {
        cout << "  row_nnz_cv: " << shape.row_nnz_cv << endl;
        cout << "  col_nnz_cv: " << shape.col_nnz_cv << endl;
        cout << "  row_nnz_gini: " << shape.row_nnz_gini << endl;
        cout << "  col_nnz_gini: " << shape.col_nnz_gini << endl;
        cout << "  max_row_nnz_over_mean: " << shape.max_row_nnz_over_mean << endl;
        cout << "  max_col_nnz_over_mean: " << shape.max_col_nnz_over_mean << endl;
        cout << "  hot_top1pct_row_nnz_ratio: " << shape.hot_top1pct_row_nnz_ratio << endl;
        cout << "  bga_partials_imbalance: " << shape.bga_partials_imbalance << endl;
        cout << "  bga_unique_rows_imbalance: " << shape.bga_unique_rows_imbalance << endl;
        cout << "  hot_bga_group_ratio: " << shape.hot_bga_group_ratio << endl;
        cout << "  bga_reduction_ratio: " << shape.bga_reduction_ratio << endl;
        cout << "  single_nnz_column_ratio: " << shape.single_nnz_column_ratio << endl;
        cout << "  low_nnz_column_ratio: " << shape.low_nnz_column_ratio << endl;
        cout << "  cluster_row_alignment_ratio: " << shape.cluster_row_alignment_ratio << endl;
        cout << "  draf_row_alignment_ratio: " << shape.draf_row_alignment_ratio << endl;
        cout << "  sampled_cluster_jaccard: " << shape.sampled_cluster_jaccard << endl;
        cout << "  draf_padding_pressure: " << shape.draf_padding_pressure << endl;
        cout << "  v2_skew_score: " << shape.skew_score << endl;
    }

    cout << "  phase: setup" << endl;
    kernel_->configurePIMControl();
    kernel_->parkIn();
    kernel_->changePIMMode(dramMode::SB, dramMode::HAB);
    kernel_->programCrf(mac_cmds);
    uint64_t excluded_setup_cycle = drain(*kernel_);

    cout << "  phase: draf_row_fetch" << endl;
    vector<char> draf_channels(kernel_->num_pim_chans_, 0);
    uint64_t global_draf_row = 0;
    for (const DrafClusterInfo& cluster : draf.clusters)
    {
        unsigned chan = cluster.id / 4;
        draf_channels[chan] = cluster.draf_rows > 0 ? 1 : draf_channels[chan];
        for (uint64_t i = 0; i < cluster.draf_rows; ++i, ++global_draf_row)
        {
            unsigned row = kMacBaseRow + (global_draf_row / 32);
            unsigned col = global_draf_row % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, chan, 0, 0, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, draf_channels);
    uint64_t draf_row_fetch_cycle = drain(*kernel_);

    cout << "  phase: pim_enable" << endl;
    kernel_->changePIMMode(dramMode::HAB, dramMode::HAB_PIM);
    uint64_t pim_enable_cycle = drain(*kernel_);

    cout << "  phase: draf_compute" << endl;
    global_draf_row = 0;
    for (const DrafClusterInfo& cluster : draf.clusters)
    {
        unsigned chan = cluster.id / 4;
        for (uint64_t i = 0; i < cluster.draf_rows; ++i, ++global_draf_row)
        {
            unsigned row = kMacBaseRow + (global_draf_row / 32);
            unsigned col = global_draf_row % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, chan, 0, i % 2, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, draf_channels);
    uint64_t draf_compute_cycle = drain(*kernel_);

    cout << "  phase: bga_accumulate" << endl;
    uint64_t bga_accumulate_cycle =
        bga.max_bacc_instructions_per_group +
        bga.max_estimated_flushes_per_group *
            (conservative_model ? kConservativeBgaFlushPenalty : 1);

    /*
     * BGA is modeled as an internal bank-group accumulator. Active BGAs run in parallel and
     * reuse the idle SIMD adders during BACC, so this phase is not emitted as serialized DRAM
     * traffic. The synthetic cycle is the longest per-BGA BACC stream plus its flush penalty.
     */
    vector<char> bga_channels(kernel_->num_pim_chans_, 0);
    for (size_t group_idx = 0; group_idx < bga.groups.size(); ++group_idx)
    {
        const BgaGroupInfo& group = bga.groups[group_idx];
        bga_channels[group.channel] = 1;
    }

    cout << "  phase: pim_disable" << endl;
    kernel_->changePIMMode(dramMode::HAB_PIM, dramMode::HAB);
    uint64_t pim_disable_cycle = drain(*kernel_);
    cout << "  phase: pim_to_sb" << endl;
    kernel_->changePIMMode(dramMode::HAB, dramMode::SB);
    uint64_t pim_to_sb_cycle = drain(*kernel_);

    cout << "  phase: bga_output_readback" << endl;
    for (size_t group_idx = 0; group_idx < bga.groups.size(); ++group_idx)
    {
        const BgaGroupInfo& group = bga.groups[group_idx];
        uint64_t readback_tx = ceilDiv(group.partials_after, kElementsPerBurst);
        for (uint64_t i = 0; i < readback_tx; ++i)
        {
            unsigned row = kResultReadBaseRow + static_cast<unsigned>(group_idx * 16 + (i / 32));
            unsigned col = i % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, group.channel, 0, 0, row, col,
                  &null_bst);
        }
    }
    addPhaseBarriers(mem_, bga_channels);
    uint64_t bga_output_readback_cycle = drain(*kernel_);

    uint64_t pim_kernel_compute_cycle = draf_compute_cycle + bga_accumulate_cycle;
    uint64_t final_reduce_cycle =
        conservative_model ? ceilDiv(bga.host_reduce_ops_after_bga, kConservativeHostReduceWidth)
                           : 0;
    uint64_t per_spmv_draf_bga_model_cycle =
        draf_row_fetch_cycle + pim_enable_cycle + pim_kernel_compute_cycle + pim_disable_cycle +
        pim_to_sb_cycle + bga_output_readback_cycle + final_reduce_cycle;
    double tck_ns = getConfigParam(FLOAT, "tCK");
    double pim_kernel_compute_ms = pim_kernel_compute_cycle * tck_ns / 1000000.0;
    double per_spmv_draf_bga_model_ms = per_spmv_draf_bga_model_cycle * tck_ns / 1000000.0;

    uint64_t v2_capacity_max_flushes =
        v2_params.use_stream_flush ? bga.max_stream_capacity_flushes_per_group
                                   : bga.max_simple_capacity_flushes_per_group;
    uint64_t v2_selected_max_flushes =
        max(bga.max_estimated_flushes_per_group, v2_capacity_max_flushes);
    uint64_t v2_flush_penalty = conservative_model ? v2_params.flush_penalty : 1;
    double v2_padding_factor =
        1.0 + v2_params.padding_alpha *
                  min(1.0, 0.5 * draf.nze_padding_ratio + 0.5 * draf.row_group_padding_ratio);
    double v2_acc_skew_factor = 1.0 + v2_params.skew_alpha * shape.skew_score;
    double v2_readback_skew_factor =
        1.0 + 0.5 * v2_params.skew_alpha *
                  normalizeRatio(shape.bga_unique_rows_imbalance, 16.0);
    double v2_final_skew_factor = 1.0 + v2_params.final_reduce_skew_alpha * shape.skew_score;
    double v2_locality_factor =
        max(v2_params.min_locality_factor,
            1.0 - v2_params.locality_beta * shape.bga_reduction_ratio);
    double v2_final_locality_factor =
        max(0.75, 1.0 - 0.5 * v2_params.locality_beta * shape.bga_reduction_ratio);

    uint64_t v2_draf_row_fetch_cycle =
        static_cast<uint64_t>(ceil(draf_row_fetch_cycle * v2_padding_factor));
    uint64_t v2_draf_compute_cycle =
        static_cast<uint64_t>(ceil(draf_compute_cycle * v2_padding_factor));
    uint64_t v2_bga_accumulate_cycle = static_cast<uint64_t>(
        ceil((bga.max_bacc_instructions_per_group +
              v2_flush_penalty * v2_selected_max_flushes) *
             v2_acc_skew_factor * v2_locality_factor));
    uint64_t v2_bga_output_readback_cycle =
        static_cast<uint64_t>(ceil(bga_output_readback_cycle * v2_readback_skew_factor));
    uint64_t v2_final_reduce_cycle =
        conservative_model
            ? static_cast<uint64_t>(
                  ceil(final_reduce_cycle * v2_final_skew_factor * v2_final_locality_factor))
            : 0;
    uint64_t v2_pim_kernel_compute_cycle = v2_draf_compute_cycle + v2_bga_accumulate_cycle;
    uint64_t v2_per_spmv_draf_bga_model_cycle =
        v2_draf_row_fetch_cycle + pim_enable_cycle + v2_pim_kernel_compute_cycle +
        pim_disable_cycle + pim_to_sb_cycle + v2_bga_output_readback_cycle +
        v2_final_reduce_cycle;
    double v2_pim_kernel_compute_ms = v2_pim_kernel_compute_cycle * tck_ns / 1000000.0;
    double v2_per_spmv_draf_bga_model_ms =
        v2_per_spmv_draf_bga_model_cycle * tck_ns / 1000000.0;

    double v21_padding_factor =
        1.0 + 2.0 * v2_params.padding_alpha * shape.draf_padding_pressure;
    double v21_alignment_factor =
        1.0 + v2_params.skew_alpha *
                  min(1.0, 0.5 * shape.cluster_row_alignment_ratio +
                               0.5 * shape.draf_row_alignment_ratio);
    double v21_effective_locality_beta =
        v2_params.locality_beta *
        max(0.0, 1.0 - 0.75 * shape.draf_padding_pressure);
    double v21_jaccard_boost =
        min(0.15, 0.5 * v21_effective_locality_beta * shape.sampled_cluster_jaccard);
    double v21_locality_factor =
        max(v2_params.min_locality_factor,
            1.0 - v21_effective_locality_beta * shape.bga_reduction_ratio -
                v21_jaccard_boost);
    double v21_final_locality_factor =
        max(0.75,
            1.0 - 0.5 * v21_effective_locality_beta * shape.bga_reduction_ratio -
                0.5 * v21_jaccard_boost);
    double v21_readback_skew_factor =
        v2_readback_skew_factor *
        (1.0 + 0.5 * v2_params.skew_alpha * shape.draf_padding_pressure);
    double v21_final_skew_factor = v2_final_skew_factor * v21_alignment_factor;

    uint64_t v21_draf_row_fetch_cycle =
        static_cast<uint64_t>(ceil(draf_row_fetch_cycle * v21_padding_factor));
    uint64_t v21_draf_compute_cycle =
        static_cast<uint64_t>(ceil(draf_compute_cycle * v21_padding_factor *
                                   v21_alignment_factor));
    uint64_t v21_bga_accumulate_cycle = static_cast<uint64_t>(
        ceil((bga.max_bacc_instructions_per_group +
              v2_flush_penalty * v2_selected_max_flushes) *
             v2_acc_skew_factor * v21_alignment_factor * v21_locality_factor));
    uint64_t v21_bga_output_readback_cycle =
        static_cast<uint64_t>(ceil(bga_output_readback_cycle * v21_readback_skew_factor));
    uint64_t v21_final_reduce_cycle =
        conservative_model
            ? static_cast<uint64_t>(
                  ceil(final_reduce_cycle * v21_final_skew_factor *
                       v21_final_locality_factor))
            : 0;
    uint64_t v21_pim_kernel_compute_cycle =
        v21_draf_compute_cycle + v21_bga_accumulate_cycle;
    uint64_t v21_per_spmv_draf_bga_model_cycle =
        v21_draf_row_fetch_cycle + pim_enable_cycle + v21_pim_kernel_compute_cycle +
        pim_disable_cycle + pim_to_sb_cycle + v21_bga_output_readback_cycle +
        v21_final_reduce_cycle;
    double v21_pim_kernel_compute_ms = v21_pim_kernel_compute_cycle * tck_ns / 1000000.0;
    double v21_per_spmv_draf_bga_model_ms =
        v21_per_spmv_draf_bga_model_cycle * tck_ns / 1000000.0;

    cout << ">>DRAF+BGA-aware SpMV Per-Run Cycle Model" << endl;
    cout << "  input: " << input_name << endl;
    cout << "  model_variant: " << (conservative_model ? "conservative" : "base") << endl;
    cout << "  rows: " << inputs.n_rows << endl;
    cout << "  cols: " << inputs.n_cols << endl;
    cout << "  nnz: " << draf.nnz << endl;
    cout << "  clusters: " << inputs.clusters.size() << endl;
    cout << "  draf_column_groups: " << draf.column_groups << endl;
    cout << "  draf_rows: " << draf.draf_rows << endl;
    cout << "  active_bga_groups: " << bga.groups.size() << endl;
    cout << "  bga_partials_before: " << bga.partials_before << endl;
    cout << "  bga_partials_after: " << bga.partials_after << endl;
    cout << "  simulator_physical_pim_bg: 0" << endl;
    cout << "  excluded_setup_cycle: " << excluded_setup_cycle << endl;
    cout << "> draf_row_fetch_cycle: " << draf_row_fetch_cycle
         << " tx=" << draf.draf_rows << endl;
    cout << "> pim_enable_cycle: " << pim_enable_cycle << endl;
    cout << "> draf_compute_cycle: " << draf_compute_cycle
         << " compute_trigger_tx=" << draf.draf_rows << endl;
    cout << "> bga_accumulate_cycle: " << bga_accumulate_cycle
         << " bacc_instructions=" << bga.bacc_instructions
         << " max_bacc_per_group=" << bga.max_bacc_instructions_per_group
         << " estimated_flushes=" << bga.estimated_flushes
         << " max_flushes_per_group=" << bga.max_estimated_flushes_per_group;
    if (conservative_model)
        cout << " flush_penalty=" << kConservativeBgaFlushPenalty;
    cout << endl;
    cout << "> pim_disable_cycle: " << pim_disable_cycle << endl;
    cout << "> pim_to_sb_cycle: " << pim_to_sb_cycle << endl;
    cout << "> bga_output_readback_cycle: " << bga_output_readback_cycle
         << " tx=" << bga.output_readback_tx << endl;
    cout << "> bga_reduce_ops: " << bga.bga_reduce_ops << endl;
    cout << "> host_reduce_ops_after_bga: " << bga.host_reduce_ops_after_bga << endl;
    if (conservative_model)
    {
        cout << "> final_reduce_cycle: " << final_reduce_cycle
             << " reduce_width=" << kConservativeHostReduceWidth << endl;
    }
    cout << "> pim_kernel_compute_cycle: " << pim_kernel_compute_cycle
         << " ms=" << pim_kernel_compute_ms << endl;
    cout << "> per_spmv_draf_bga_model_cycle: " << per_spmv_draf_bga_model_cycle
         << " ms=" << per_spmv_draf_bga_model_ms << endl;
    if (v2_model)
    {
        cout << ">>DRAF+BGA-aware SpMV "
             << (v21_model ? "V2.1" : "V2") << " Corrected Cycle Model" << endl;
        cout << "  input: " << input_name << endl;
        cout << "  model_variant: "
             << (v21_model ? (conservative_model ? "v2.1_conservative" : "v2.1_base")
                           : (conservative_model ? "v2_conservative" : "v2_base"))
             << endl;
        cout << "  flush_mode: " << (v2_params.use_stream_flush ? "stream" : "simple")
             << endl;
        cout << "  capacity_max_flushes_per_group: " << v2_capacity_max_flushes << endl;
        cout << "  selected_max_flushes_per_group: " << v2_selected_max_flushes << endl;
        cout << "  padding_factor: " << v2_padding_factor << endl;
        cout << "  acc_skew_factor: " << v2_acc_skew_factor << endl;
        cout << "  readback_skew_factor: " << v2_readback_skew_factor << endl;
        cout << "  final_skew_factor: " << v2_final_skew_factor << endl;
        cout << "  locality_factor: " << v2_locality_factor << endl;
        cout << "  final_locality_factor: " << v2_final_locality_factor << endl;
        if (v21_model)
        {
            cout << "  v2.1_padding_factor: " << v21_padding_factor << endl;
            cout << "  v2.1_alignment_factor: " << v21_alignment_factor << endl;
            cout << "  v2.1_effective_locality_beta: " << v21_effective_locality_beta
                 << endl;
            cout << "  v2.1_jaccard_boost: " << v21_jaccard_boost << endl;
            cout << "  v2.1_locality_factor: " << v21_locality_factor << endl;
            cout << "  v2.1_final_locality_factor: " << v21_final_locality_factor << endl;
            cout << "  v2.1_readback_skew_factor: " << v21_readback_skew_factor << endl;
        }
        cout << "> v1_per_spmv_draf_bga_model_cycle: " << per_spmv_draf_bga_model_cycle
             << " ms=" << per_spmv_draf_bga_model_ms << endl;
        uint64_t reported_draf_row_fetch_cycle =
            v21_model ? v21_draf_row_fetch_cycle : v2_draf_row_fetch_cycle;
        uint64_t reported_draf_compute_cycle =
            v21_model ? v21_draf_compute_cycle : v2_draf_compute_cycle;
        uint64_t reported_bga_accumulate_cycle =
            v21_model ? v21_bga_accumulate_cycle : v2_bga_accumulate_cycle;
        uint64_t reported_bga_output_readback_cycle =
            v21_model ? v21_bga_output_readback_cycle : v2_bga_output_readback_cycle;
        uint64_t reported_final_reduce_cycle =
            v21_model ? v21_final_reduce_cycle : v2_final_reduce_cycle;
        uint64_t reported_pim_kernel_compute_cycle =
            v21_model ? v21_pim_kernel_compute_cycle : v2_pim_kernel_compute_cycle;
        uint64_t reported_total_cycle =
            v21_model ? v21_per_spmv_draf_bga_model_cycle
                      : v2_per_spmv_draf_bga_model_cycle;
        double reported_pim_kernel_compute_ms =
            v21_model ? v21_pim_kernel_compute_ms : v2_pim_kernel_compute_ms;
        double reported_total_ms =
            v21_model ? v21_per_spmv_draf_bga_model_ms : v2_per_spmv_draf_bga_model_ms;

        cout << "> v2_draf_row_fetch_cycle: " << reported_draf_row_fetch_cycle
             << " v1=" << draf_row_fetch_cycle << endl;
        cout << "> v2_draf_compute_cycle: " << reported_draf_compute_cycle
             << " v1=" << draf_compute_cycle << endl;
        cout << "> v2_bga_accumulate_cycle: " << reported_bga_accumulate_cycle
             << " v1=" << bga_accumulate_cycle << endl;
        cout << "> v2_bga_output_readback_cycle: " << reported_bga_output_readback_cycle
             << " v1=" << bga_output_readback_cycle << endl;
        if (conservative_model)
            cout << "> v2_final_reduce_cycle: " << reported_final_reduce_cycle
                 << " v1=" << final_reduce_cycle << endl;
        cout << "> v2_pim_kernel_compute_cycle: " << reported_pim_kernel_compute_cycle
             << " ms=" << reported_pim_kernel_compute_ms << endl;
        cout << "> v2_per_spmv_draf_bga_model_cycle: " << reported_total_cycle
             << " ms=" << reported_total_ms << endl;
    }
}

void ClusteredSpmvBenchFixture::runGuidedKmeansDrafBgaSuite(bool conservative_model,
                                                            bool v2_model,
                                                            bool v21_model)
{
    if (v21_model)
        v2_model = true;
    const vector<SpmvDataset> datasets{
        {"ASIC_100k", "../SparsePIM/guided_kmeans_coo_results/ASIC_100k/"},
        {"Stanford", "../SparsePIM/guided_kmeans_coo_results/Stanford/"},
        {"bcsstk32", "../SparsePIM/guided_kmeans_coo_results/bcsstk32/"},
        {"cant", "../SparsePIM/guided_kmeans_coo_results/cant/"},
        {"consph", "../SparsePIM/guided_kmeans_coo_results/consph/"},
        {"crankseg_2", "../SparsePIM/guided_kmeans_coo_results/crankseg_2/"},
        {"ct20stif", "../SparsePIM/guided_kmeans_coo_results/ct20stif/"},
        {"lhr71", "../SparsePIM/guided_kmeans_coo_results/lhr71/"},
        {"ohne2", "../SparsePIM/guided_kmeans_coo_results/ohne2/"},
        {"pdb1HYS", "../SparsePIM/guided_kmeans_coo_results/pdb1HYS/"},
        {"pwtk", "../SparsePIM/guided_kmeans_coo_results/pwtk/"},
        {"rma10", "../SparsePIM/guided_kmeans_coo_results/rma10/"},
        {"shipsec1", "../SparsePIM/guided_kmeans_coo_results/shipsec1/"},
        {"soc-sign-epinions", "../SparsePIM/guided_kmeans_coo_results/soc-sign-epinions/"},
        {"webbase-1M", "../SparsePIM/guided_kmeans_coo_results/webbase-1M/"},
        {"xenon2", "../SparsePIM/guided_kmeans_coo_results/xenon2/"},
    };

    string only_matrix = envString("SPMV_BENCH_MATRIX");
    cout << ">>Guided K-means COO DRAF+BGA suite" << endl;
    cout << "  model_variant: "
         << (v21_model ? (conservative_model ? "v2.1_conservative" : "v2.1_base")
                       : (v2_model ? (conservative_model ? "v2_conservative" : "v2_base")
                                   : (conservative_model ? "conservative" : "base")))
         << endl;
    if (!only_matrix.empty())
        cout << "  SPMV_BENCH_MATRIX: " << only_matrix << endl;

    bool matched = false;
    for (const SpmvDataset& dataset : datasets)
    {
        if (!only_matrix.empty() && dataset.name != only_matrix)
            continue;

        matched = true;
        resetPIMKernel();
        runDrafBgaModel(dataset.base,
                        "SparsePIM/guided_kmeans_coo_results/" + dataset.name,
                        conservative_model, v2_model, v21_model);
    }

    if (!only_matrix.empty())
    {
        ASSERT_TRUE(matched) << "unknown SPMV_BENCH_MATRIX=" << only_matrix;
    }
}

void ClusteredSpmvBenchFixture::runDrafBgaStructuralModel(const string& base,
                                                          const string& input_name,
                                                          const string& matrix_name,
                                                          bool critical_padding_model)
{
    SpmvInputs inputs = loadSparsePIMInputs(base + "reordered_matrix.txt",
                                            base + "column_permutation.txt", base + "clusters.txt");
    uint64_t max_clusters = envLimit("SPMV_BENCH_MAX_CLUSTERS");
    applyClusterLimit(inputs, max_clusters);
    DrafStats draf = buildDrafStats(inputs);
    BgaStats bga = buildBgaStats(inputs, kDefaultV2BgaAccCapacity);
    ShapeStats shape = buildShapeStats(inputs, draf, bga);
    DrafCriticalPathStats critical = buildDrafCriticalPathStats(draf);

    BurstType null_bst;
    vector<PIMCmd> mac_cmds{
        PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK, 1),
        PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A, PIMOpdType::ODD_BANK, 1),
        PIMCmd(PIMCmdType::NOP, 7),
        PIMCmd(PIMCmdType::EXIT, 0),
    };

    cout << ">>DRAF+BGA-aware SpMV "
         << (critical_padding_model ? "V4 Critical-Path Structural Model"
                                    : "V3 Structural Model")
         << endl;
    cout << "  input: " << input_name << endl;
    cout << "  matrix: " << matrix_name << endl;
    cout << "  note: structural model; target speedup is used only for reporting" << endl;
    cout << "  rows: " << inputs.n_rows << endl;
    cout << "  cols: " << inputs.n_cols << endl;
    cout << "  nnz: " << draf.nnz << endl;
    cout << "  clusters: " << inputs.clusters.size() << endl;
    cout << "  draf_column_groups: " << draf.column_groups << endl;
    cout << "  draf_rows: " << draf.draf_rows << endl;
    cout << "  draf_packed_nnz_capacity: " << draf.packed_nnz_capacity << endl;
    cout << "  draf_nze_padding: " << draf.nze_padding << endl;
    cout << "  draf_padding_ratio: " << draf.nze_padding_ratio << endl;
    cout << "  draf_expansion_ratio: " << draf.expansion_ratio << endl;
    cout << "  draf_nze_padding_ratio: " << draf.nze_padding_ratio << endl;
    cout << "  draf_memory_expansion: " << draf.expansion_ratio << endl;
    cout << "  critical_padding: " << critical.critical_padding << endl;
    cout << "  critical_padding_ratio: " << critical.critical_padding_ratio << endl;
    cout << "  group_steps_mean: " << critical.group_steps_mean << endl;
    cout << "  group_steps_max: " << critical.group_steps_max << endl;
    cout << "  bg_imbalance: " << critical.bg_imbalance << endl;
    cout << "  single_nnz_column_ratio: " << shape.single_nnz_column_ratio << endl;
    cout << "  low_nnz_column_ratio: " << shape.low_nnz_column_ratio << endl;
    cout << "  draf_padding_pressure: " << shape.draf_padding_pressure << endl;
    cout << "  bga_reduction_ratio: " << shape.bga_reduction_ratio << endl;

    kernel_->configurePIMControl();
    kernel_->parkIn();
    kernel_->changePIMMode(dramMode::SB, dramMode::HAB);
    kernel_->programCrf(mac_cmds);
    uint64_t setup_cycle = drain(*kernel_);

    vector<char> draf_channels(kernel_->num_pim_chans_, 0);
    uint64_t global_draf_row = 0;
    for (const DrafClusterInfo& cluster : draf.clusters)
    {
        unsigned chan = cluster.id / 4;
        draf_channels[chan] = cluster.draf_rows > 0 ? 1 : draf_channels[chan];
        for (uint64_t i = 0; i < cluster.draf_rows; ++i, ++global_draf_row)
        {
            unsigned row = kMacBaseRow + (global_draf_row / 32);
            unsigned col = global_draf_row % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, chan, 0, 0, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, draf_channels);
    uint64_t draf_row_fetch_cycle = drain(*kernel_);

    kernel_->changePIMMode(dramMode::HAB, dramMode::HAB_PIM);
    uint64_t pim_enable_cycle = drain(*kernel_);

    global_draf_row = 0;
    for (const DrafClusterInfo& cluster : draf.clusters)
    {
        unsigned chan = cluster.id / 4;
        for (uint64_t i = 0; i < cluster.draf_rows; ++i, ++global_draf_row)
        {
            unsigned row = kMacBaseRow + (global_draf_row / 32);
            unsigned col = global_draf_row % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, chan, 0, i % 2, row, col, &null_bst);
        }
    }
    addPhaseBarriers(mem_, draf_channels);
    uint64_t draf_compute_trigger_cycle = drain(*kernel_);

    uint64_t bga_capacity_flushes =
        max(bga.max_estimated_flushes_per_group, bga.max_stream_capacity_flushes_per_group);
    uint64_t bga_accumulate_cycle =
        bga.max_bacc_instructions_per_group + kConservativeBgaFlushPenalty * bga_capacity_flushes;

    kernel_->changePIMMode(dramMode::HAB_PIM, dramMode::HAB);
    uint64_t pim_disable_cycle = drain(*kernel_);
    kernel_->changePIMMode(dramMode::HAB, dramMode::SB);
    uint64_t pim_to_sb_cycle = drain(*kernel_);

    vector<char> bga_channels(kernel_->num_pim_chans_, 0);
    for (const BgaGroupInfo& group : bga.groups)
        bga_channels[group.channel] = 1;
    for (size_t group_idx = 0; group_idx < bga.groups.size(); ++group_idx)
    {
        const BgaGroupInfo& group = bga.groups[group_idx];
        uint64_t readback_tx = ceilDiv(group.partials_after, kElementsPerBurst);
        for (uint64_t i = 0; i < readback_tx; ++i)
        {
            unsigned row = kResultReadBaseRow + static_cast<unsigned>(group_idx * 16 + (i / 32));
            unsigned col = i % 32;
            addTx(mem_, *kernel_->pim_addr_mgr_, false, group.channel, 0, 0, row, col,
                  &null_bst);
        }
    }
    addPhaseBarriers(mem_, bga_channels);
    uint64_t bga_output_readback_cycle = drain(*kernel_);

    /*
     * V3 charges all padded NZE slots. V4 charges only the padding exposed on the synchronous
     * bank-group critical path, so regular padding that is hidden inside shorter banks does not
     * receive the same cost as useful nonzero work.
     */
    uint64_t padded_zero_compute_cycle =
        critical_padding_model
            ? static_cast<uint64_t>(ceil(critical.critical_padding))
            : ceilDiv(draf.nze_padding, kElementsPerBurst);
    uint64_t final_reduce_cycle = ceilDiv(bga.host_reduce_ops_after_bga,
                                          kConservativeHostReduceWidth);
    uint64_t v3_total_cycle =
        setup_cycle + draf_row_fetch_cycle + pim_enable_cycle + draf_compute_trigger_cycle +
        padded_zero_compute_cycle + bga_accumulate_cycle + pim_disable_cycle + pim_to_sb_cycle +
        bga_output_readback_cycle + final_reduce_cycle;

    double tck_ns = getConfigParam(FLOAT, "tCK");
    double v3_total_ms = v3_total_cycle * tck_ns / 1000000.0;
    double gpu_ms = gpuBaselineMs(matrix_name);
    double v3_speedup = v3_total_ms == 0.0 ? 0.0 : gpu_ms / v3_total_ms;
    double target_speedup = paperTargetSpeedup(matrix_name);
    double target_pim_ms =
        target_speedup == 0.0 ? 0.0 : gpu_ms / target_speedup;

    cout << "  latency_scope: setup + DRAF access + "
         << (critical_padding_model ? "critical-path padding" : "padded zero work")
         << " + BGA + result readback" << endl;
    cout << "> setup_cycle: " << setup_cycle << endl;
    cout << "> draf_row_fetch_cycle: " << draf_row_fetch_cycle << endl;
    cout << "> draf_compute_trigger_cycle: " << draf_compute_trigger_cycle << endl;
    cout << "> "
         << (critical_padding_model ? "critical_padding_cycle: "
                                    : "padded_zero_compute_cycle: ")
         << padded_zero_compute_cycle;
    if (critical_padding_model)
    {
        cout << " critical_padding=" << critical.critical_padding
             << " ideal_steps=" << critical.ideal_steps
             << " actual_steps=" << critical.actual_steps;
    }
    else
    {
        cout << " padded_slots=" << draf.nze_padding
             << " simd_width=" << kElementsPerBurst;
    }
    cout << endl;
    cout << "> bga_accumulate_cycle: " << bga_accumulate_cycle
         << " max_bacc_per_group=" << bga.max_bacc_instructions_per_group
         << " selected_flushes_per_group=" << bga_capacity_flushes
         << " flush_penalty=" << kConservativeBgaFlushPenalty << endl;
    cout << "> bga_output_readback_cycle: " << bga_output_readback_cycle << endl;
    cout << "> final_reduce_cycle: " << final_reduce_cycle << endl;
    cout << "> "
         << (critical_padding_model ? "v4_structural_cycle: " : "v3_structural_cycle: ")
         << v3_total_cycle
         << " ms=" << v3_total_ms << endl;
    cout << "> gpu_baseline_ms: " << gpu_ms << endl;
    cout << "> paper_target_speedup: " << target_speedup
         << " target_pim_ms=" << target_pim_ms << endl;
    cout << "> "
         << (critical_padding_model ? "v4_structural_speedup: "
                                    : "v3_structural_speedup: ")
         << v3_speedup
         << " speedup_error_ratio="
         << (target_speedup == 0.0 ? 0.0 : v3_speedup / target_speedup) << endl;
    cout << (critical_padding_model ? "V4_RESULT_CSV," : "V3_RESULT_CSV,")
         << matrix_name << "," << gpu_ms << "," << target_speedup << "," << target_pim_ms
         << "," << v3_total_ms << "," << v3_speedup << ","
         << (target_speedup == 0.0 ? 0.0 : v3_speedup / target_speedup) << ","
         << draf.nze_padding << "," << draf.nze_padding_ratio << ","
         << draf.expansion_ratio << "," << critical.critical_padding << ","
         << critical.critical_padding_ratio << "," << critical.group_steps_mean << ","
         << critical.group_steps_max << "," << critical.bg_imbalance << endl;
}

void ClusteredSpmvBenchFixture::runGuidedKmeansDrafBgaStructuralSuite(
    bool critical_padding_model)
{
    const vector<SpmvDataset> datasets{
        {"ASIC_100k", "../SparsePIM/guided_kmeans_coo_results/ASIC_100k/"},
        {"Stanford", "../SparsePIM/guided_kmeans_coo_results/Stanford/"},
        {"bcsstk32", "../SparsePIM/guided_kmeans_coo_results/bcsstk32/"},
        {"cant", "../SparsePIM/guided_kmeans_coo_results/cant/"},
        {"consph", "../SparsePIM/guided_kmeans_coo_results/consph/"},
        {"crankseg_2", "../SparsePIM/guided_kmeans_coo_results/crankseg_2/"},
        {"ct20stif", "../SparsePIM/guided_kmeans_coo_results/ct20stif/"},
        {"lhr71", "../SparsePIM/guided_kmeans_coo_results/lhr71/"},
        {"ohne2", "../SparsePIM/guided_kmeans_coo_results/ohne2/"},
        {"pdb1HYS", "../SparsePIM/guided_kmeans_coo_results/pdb1HYS/"},
        {"pwtk", "../SparsePIM/guided_kmeans_coo_results/pwtk/"},
        {"rma10", "../SparsePIM/guided_kmeans_coo_results/rma10/"},
        {"shipsec1", "../SparsePIM/guided_kmeans_coo_results/shipsec1/"},
        {"soc-sign-epinions", "../SparsePIM/guided_kmeans_coo_results/soc-sign-epinions/"},
        {"webbase-1M", "../SparsePIM/guided_kmeans_coo_results/webbase-1M/"},
        {"xenon2", "../SparsePIM/guided_kmeans_coo_results/xenon2/"},
    };

    string only_matrix = envString("SPMV_BENCH_MATRIX");
    cout << ">>Guided K-means COO DRAF+BGA "
         << (critical_padding_model ? "V4 critical-path structural suite"
                                    : "V3 structural suite")
         << endl;
    if (!only_matrix.empty())
        cout << "  SPMV_BENCH_MATRIX: " << only_matrix << endl;

    bool matched = false;
    for (const SpmvDataset& dataset : datasets)
    {
        if (!only_matrix.empty() && dataset.name != only_matrix)
            continue;

        matched = true;
        resetPIMKernel();
        runDrafBgaStructuralModel(dataset.base,
                                  "SparsePIM/guided_kmeans_coo_results/" + dataset.name,
                                  dataset.name, critical_padding_model);
    }

    if (!only_matrix.empty())
    {
        ASSERT_TRUE(matched) << "unknown SPMV_BENCH_MATRIX=" << only_matrix;
    }
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_cluster_cantcoo_draf_bga_model)
{
    runDrafBgaModel("../SparsePIM/cluster_cantcoo/", "SparsePIM/cluster_cantcoo");
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_cluster_cantcoo_draf_bga_conservative_model)
{
    runDrafBgaModel("../SparsePIM/cluster_cantcoo/", "SparsePIM/cluster_cantcoo", true);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_hybrid_hash_partition_v2_cantcoo_draf_bga_model)
{
    runDrafBgaModel("../SparsePIM/hybrid_hash_partition_v2_cantcoo/",
                    "SparsePIM/hybrid_hash_partition_v2_cantcoo");
}

TEST_F(ClusteredSpmvBenchFixture,
       sparsepim_hybrid_hash_partition_v2_cantcoo_draf_bga_conservative_model)
{
    runDrafBgaModel("../SparsePIM/hybrid_hash_partition_v2_cantcoo/",
                    "SparsePIM/hybrid_hash_partition_v2_cantcoo", true);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_model)
{
    runGuidedKmeansDrafBgaSuite();
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_conservative_model)
{
    runGuidedKmeansDrafBgaSuite(true);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v2_model)
{
    runGuidedKmeansDrafBgaSuite(false, true);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v2_conservative_model)
{
    runGuidedKmeansDrafBgaSuite(true, true);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v21_model)
{
    runGuidedKmeansDrafBgaSuite(false, true, true);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v21_conservative_model)
{
    runGuidedKmeansDrafBgaSuite(true, true, true);
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v3_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite();
}

TEST_F(ClusteredSpmvBenchFixture, sparsepim_guided_kmeans_coo_draf_bga_v4_structural_model)
{
    runGuidedKmeansDrafBgaStructuralSuite(true);
}
