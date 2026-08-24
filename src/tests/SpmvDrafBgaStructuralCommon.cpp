#include "tests/SpmvDrafBgaStructuralCommon.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace spmv
{
using namespace std;

uint64_t ceilDiv(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1) / divisor;
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

SpmvInputs loadNaiveCooInputs(const string& matrix_path, uint64_t num_clusters)
{
    if (num_clusters == 0)
        throw invalid_argument("num_clusters must be greater than zero");

    ifstream matrix_file(matrix_path);
    if (!matrix_file)
        throw runtime_error("failed to open " + matrix_path);

    SpmvInputs inputs;
    if (!(matrix_file >> inputs.n_rows >> inputs.n_cols >> inputs.nnz))
        throw runtime_error("failed to read COO header from " + matrix_path);

    inputs.rows_by_col.resize(inputs.n_cols);
    inputs.reordered_col_to_cluster.resize(inputs.n_cols, -1);
    inputs.clusters.resize(num_clusters);
    vector<unordered_set<unsigned>> active_rows_by_cluster(num_clusters);
    vector<char> any_row(inputs.n_rows, 0);

    for (uint64_t cluster = 0; cluster < num_clusters; ++cluster)
        inputs.clusters[cluster].id = static_cast<unsigned>(cluster);
    for (uint64_t col = 0; col < inputs.n_cols; ++col)
    {
        uint64_t cluster = col % num_clusters;
        inputs.reordered_col_to_cluster[col] = static_cast<int>(cluster);
        inputs.clusters[cluster].num_cols++;
    }

    uint64_t parsed_nnz = 0;
    uint64_t row = 0;
    uint64_t col = 0;
    string value;
    while (matrix_file >> row >> col >> value)
    {
        if (row >= inputs.n_rows || col >= inputs.n_cols)
            throw runtime_error("COO entry exceeds declared dimensions in " + matrix_path);
        uint64_t cluster = col % num_clusters;
        inputs.rows_by_col[col].push_back(static_cast<unsigned>(row));
        inputs.clusters[cluster].nnz++;
        active_rows_by_cluster[cluster].insert(static_cast<unsigned>(row));
        any_row[row] = 1;
        parsed_nnz++;
    }
    if (parsed_nnz != inputs.nnz)
        throw runtime_error("COO NNZ count does not match header in " + matrix_path);

    for (uint64_t cluster = 0; cluster < num_clusters; ++cluster)
    {
        inputs.clusters[cluster].active_rows = active_rows_by_cluster[cluster].size();
        inputs.total_active_row_memberships += inputs.clusters[cluster].active_rows;
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

struct BgaTemporalReuseStats
{
    uint64_t near_duplicate_partials = 0;
    uint64_t far_duplicate_partials = 0;
};

BgaTemporalReuseStats estimateTemporalBgaReuse(const vector<unsigned>& row_stream,
                                               uint64_t near_window)
{
    BgaTemporalReuseStats stats;
    unordered_map<unsigned, uint64_t> last_seen;
    for (uint64_t idx = 0; idx < row_stream.size(); ++idx)
    {
        unsigned row = row_stream[idx];
        auto it = last_seen.find(row);
        if (it != last_seen.end())
        {
            uint64_t reuse_distance = idx - it->second;
            if (reuse_distance <= near_window)
                stats.near_duplicate_partials++;
            else
                stats.far_duplicate_partials++;
        }
        last_seen[row] = idx;
    }
    return stats;
}

BgaStats buildBgaStats(const SpmvInputs& inputs, uint64_t bga_acc_capacity)
{
    BgaStats stats;
    const uint64_t num_groups = kSparsePimBankGroups;
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

        unsigned channel = static_cast<unsigned>(cluster) / kBankGroupsPerPseudoChannel;
        unsigned bank_group = static_cast<unsigned>(cluster) % kBankGroupsPerPseudoChannel;
        uint64_t group_idx = channel * kBankGroupsPerPseudoChannel + bank_group;
        if (group_idx >= num_groups)
            throw runtime_error("cluster id exceeds SparsePIM bank-group topology");
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
        info.channel = group_idx / kBankGroupsPerPseudoChannel;
        info.bank_group = group_idx % kBankGroupsPerPseudoChannel;
        info.partials_before = partials_by_group[group_idx];
        info.partials_after = rows_by_group[group_idx].size();
        info.bacc_instructions = ceilDiv(info.partials_before, kBgaEntriesPerBacc);
        info.estimated_flushes = estimateBgaFlushes(info.bacc_instructions);
        info.simple_capacity_flushes =
            estimateSimpleCapacityFlushes(info.partials_after, bga_acc_capacity);
        info.stream_capacity_flushes =
            estimateStreamCapacityFlushes(row_stream_by_group[group_idx], bga_acc_capacity);
        BgaTemporalReuseStats temporal_reuse =
            estimateTemporalBgaReuse(row_stream_by_group[group_idx],
                                     bga_acc_capacity * kBgaEntriesPerBacc);
        uint64_t duplicate_partials =
            info.partials_before > info.partials_after
                ? info.partials_before - info.partials_after
                : 0;
        info.near_duplicate_partials = temporal_reuse.near_duplicate_partials;
        info.far_duplicate_partials = temporal_reuse.far_duplicate_partials;
        info.near_duplicate_ratio =
            info.partials_before == 0
                ? 0.0
                : static_cast<double>(info.near_duplicate_partials) /
                      static_cast<double>(info.partials_before);
        info.far_duplicate_ratio =
            info.partials_before == 0
                ? 0.0
                : static_cast<double>(info.far_duplicate_partials) /
                      static_cast<double>(info.partials_before);
        info.row_reuse_factor =
            info.partials_after == 0
                ? 0.0
                : static_cast<double>(info.partials_before) /
                      static_cast<double>(info.partials_after);
        info.effective_partials =
            static_cast<double>(info.partials_after) +
            kBgaDuplicatePartialWeight * static_cast<double>(duplicate_partials);
        info.reuse_aware_bacc_instructions =
            static_cast<uint64_t>(ceil(info.effective_partials /
                                       static_cast<double>(kBgaEntriesPerBacc)));
        info.reuse_aware_accumulate_cycle =
            info.reuse_aware_bacc_instructions +
            kConservativeBgaFlushPenalty * info.stream_capacity_flushes;

        stats.partials_before += info.partials_before;
        stats.partials_after += info.partials_after;
        stats.bacc_instructions += info.bacc_instructions;
        stats.estimated_flushes += info.estimated_flushes;
        stats.simple_capacity_flushes += info.simple_capacity_flushes;
        stats.stream_capacity_flushes += info.stream_capacity_flushes;
        stats.near_duplicate_partials += info.near_duplicate_partials;
        stats.far_duplicate_partials += info.far_duplicate_partials;
        stats.max_bacc_instructions_per_group =
            max(stats.max_bacc_instructions_per_group, info.bacc_instructions);
        stats.max_estimated_flushes_per_group =
            max(stats.max_estimated_flushes_per_group, info.estimated_flushes);
        stats.max_simple_capacity_flushes_per_group =
            max(stats.max_simple_capacity_flushes_per_group, info.simple_capacity_flushes);
        stats.max_stream_capacity_flushes_per_group =
            max(stats.max_stream_capacity_flushes_per_group, info.stream_capacity_flushes);
        stats.max_accumulator_flush_estimate_per_group =
            max(stats.max_accumulator_flush_estimate_per_group, info.stream_capacity_flushes);
        stats.max_reuse_aware_bacc_instructions_per_group =
            max(stats.max_reuse_aware_bacc_instructions_per_group,
                info.reuse_aware_bacc_instructions);
        stats.max_reuse_aware_accumulate_cycle_per_group =
            max(stats.max_reuse_aware_accumulate_cycle_per_group,
                info.reuse_aware_accumulate_cycle);
        stats.max_row_reuse_factor_per_group =
            max(stats.max_row_reuse_factor_per_group, info.row_reuse_factor);
        stats.groups.push_back(info);

        for (unsigned row : rows_by_group[group_idx])
            row_bga_memberships[row]++;
    }

    stats.bga_reduce_ops = stats.partials_before > stats.partials_after
                               ? stats.partials_before - stats.partials_after
                               : 0;
    stats.output_readback_tx = ceilDiv(stats.partials_after, kElementsPerBurst);
    stats.row_reuse_factor =
        stats.partials_after == 0
            ? 0.0
            : static_cast<double>(stats.partials_before) /
                  static_cast<double>(stats.partials_after);
    stats.unique_row_ratio =
        stats.partials_before == 0
            ? 0.0
            : static_cast<double>(stats.partials_after) /
                  static_cast<double>(stats.partials_before);
    stats.duplicate_partial_ratio =
        stats.partials_before == 0
            ? 0.0
            : static_cast<double>(stats.bga_reduce_ops) /
                  static_cast<double>(stats.partials_before);
    stats.near_duplicate_partial_ratio =
        stats.partials_before == 0
            ? 0.0
            : static_cast<double>(stats.near_duplicate_partials) /
                  static_cast<double>(stats.partials_before);
    stats.far_duplicate_partial_ratio =
        stats.partials_before == 0
            ? 0.0
            : static_cast<double>(stats.far_duplicate_partials) /
                  static_cast<double>(stats.partials_before);
    uint64_t temporal_duplicates =
        stats.near_duplicate_partials + stats.far_duplicate_partials;
    stats.near_duplicate_share =
        temporal_duplicates == 0
            ? 0.0
            : static_cast<double>(stats.near_duplicate_partials) /
                  static_cast<double>(temporal_duplicates);
    stats.accumulator_flush_estimate = stats.stream_capacity_flushes;
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

double percentileValue(vector<uint64_t> values, double percentile)
{
    if (values.empty())
        return 0.0;
    sort(values.begin(), values.end());
    double rank = (percentile / 100.0) * static_cast<double>(values.size() - 1);
    size_t lo = static_cast<size_t>(floor(rank));
    size_t hi = static_cast<size_t>(ceil(rank));
    if (lo == hi)
        return static_cast<double>(values[lo]);
    double frac = rank - static_cast<double>(lo);
    return static_cast<double>(values[lo]) * (1.0 - frac) +
           static_cast<double>(values[hi]) * frac;
}

double normalizeRatio(double value, double scale)
{
    if (value <= 1.0)
        return 0.0;
    return min(1.0, log(value) / log(scale));
}

double clamp01(double value)
{
    return min(1.0, max(0.0, value));
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
    stats.bga_row_reuse_factor = bga.row_reuse_factor;
    stats.bga_unique_row_ratio = bga.unique_row_ratio;
    stats.bga_duplicate_partial_ratio = bga.duplicate_partial_ratio;
    stats.bga_capacity_pressure =
        bga.groups.empty()
            ? 0.0
            : static_cast<double>(bga.max_stream_capacity_flushes_per_group) /
                  static_cast<double>(bga.groups.size());
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
    stats.group_steps_p90 = percentileValue(actual_group_steps, 90.0);
    stats.group_steps_p95 = percentileValue(actual_group_steps, 95.0);
    stats.tail_exposure_ratio =
        stats.group_steps_max == 0.0 ? 0.0 : stats.group_steps_p95 / stats.group_steps_max;
    stats.tail_skew_max_over_p90 =
        stats.group_steps_p90 == 0.0 ? 0.0 : stats.group_steps_max / stats.group_steps_p90;
    stats.tail_skew_max_over_p95 =
        stats.group_steps_p95 == 0.0 ? 0.0 : stats.group_steps_max / stats.group_steps_p95;
    stats.p90_to_max_tail_steps = max(0.0, stats.group_steps_max - stats.group_steps_p90);
    stats.p95_to_max_tail_steps = max(0.0, stats.group_steps_max - stats.group_steps_p95);
    stats.bg_imbalance =
        stats.group_steps_mean == 0.0 ? 0.0 : stats.group_steps_max / stats.group_steps_mean;
    return stats;
}

DrafPaddingExposureStats buildDrafPaddingExposureStats(const DrafStats& draf,
                                                       const ShapeStats& shape,
                                                       const DrafCriticalPathStats& critical)
{
    DrafPaddingExposureStats stats;
    stats.total_padding_steps =
        static_cast<double>(draf.nze_padding) / static_cast<double>(kElementsPerBurst);
    stats.critical_padding_steps = critical.critical_padding;
    stats.hidden_padding_steps =
        max(0.0, stats.total_padding_steps - stats.critical_padding_steps);
    stats.fragmentation = max(shape.single_nnz_column_ratio, shape.low_nnz_column_ratio);
    stats.memory_pressure = clamp01((draf.expansion_ratio - 1.0) / 4.0);
    stats.imbalance_pressure = clamp01((critical.bg_imbalance - 1.0) / 3.0);
    stats.exposure_factor =
        clamp01(0.6 * stats.fragmentation + 0.3 * stats.memory_pressure +
                0.1 * stats.imbalance_pressure);
    stats.exposed_hidden_padding = stats.hidden_padding_steps * stats.exposure_factor;
    stats.effective_padding_steps =
        stats.critical_padding_steps + stats.exposed_hidden_padding;
    return stats;
}

BgaOverlapStats buildBgaOverlapStats(uint64_t raw_bga_accumulate_cycle,
                                     uint64_t compute_overlap_window,
                                     const ShapeStats& shape,
                                     const DrafPaddingExposureStats& exposure)
{
    BgaOverlapStats stats;
    stats.raw_accumulate_cycle = raw_bga_accumulate_cycle;
    stats.regularity_score =
        clamp01(1.0 - max(exposure.fragmentation, exposure.memory_pressure));
    stats.reuse_score = clamp01(shape.bga_reduction_ratio);
    stats.padding_guard = clamp01(1.0 - exposure.exposure_factor);

    /*
     * V6 treats BGA accumulation as an internal pipeline that can be partially hidden by
     * DRAF compute slots. The overlap is intentionally guarded by DRAF fragmentation and
     * memory expansion, so padding-dominated workloads do not receive the same benefit as
     * regular/high-reuse matrices.
     */
    stats.overlap_factor =
        clamp01(0.10 + 0.60 * stats.regularity_score * stats.padding_guard +
                0.30 * stats.regularity_score * stats.reuse_score);
    uint64_t candidate_hidden_cycle = static_cast<uint64_t>(
        floor(static_cast<double>(raw_bga_accumulate_cycle) * stats.overlap_factor));
    stats.hidden_accumulate_cycle = min(candidate_hidden_cycle, compute_overlap_window);
    stats.exposed_accumulate_cycle =
        raw_bga_accumulate_cycle > stats.hidden_accumulate_cycle
            ? raw_bga_accumulate_cycle - stats.hidden_accumulate_cycle
            : 0;
    return stats;
}

DrafStreamingOverlapStats buildDrafStreamingOverlapStats(
    uint64_t draf_row_fetch_cycle,
    uint64_t draf_compute_trigger_cycle,
    uint64_t pre_overlap_total_cycle,
    uint64_t compute_overlap_window,
    const DrafPaddingExposureStats& exposure,
    const BgaOverlapStats& bga_overlap,
    bool budget_by_bga_overlap)
{
    DrafStreamingOverlapStats stats;
    uint64_t draf_access_cycle = draf_row_fetch_cycle + draf_compute_trigger_cycle;
    uint64_t streamable_cycle = min(draf_row_fetch_cycle, draf_compute_trigger_cycle);
    stats.regularity_score =
        clamp01(1.0 - max(exposure.fragmentation, exposure.memory_pressure));
    stats.guard_score = clamp01(1.0 - exposure.exposure_factor);
    stats.access_share = pre_overlap_total_cycle == 0
                             ? 0.0
                             : static_cast<double>(draf_access_cycle) /
                                   static_cast<double>(pre_overlap_total_cycle);

    /*
     * V7 keeps the existing bg=0 simulator path but models DRAF fetch and compute-trigger
     * as a staged stream rather than two fully independent traversals. Regular DRAF streams
     * can reuse staged rows, while fragmented or padding-heavy streams expose most of the
     * second traversal.
     */
    stats.overlap_factor =
        clamp01(0.10 + 0.70 * stats.regularity_score * stats.guard_score +
                0.20 * stats.access_share);
    uint64_t raw_hidden_access_cycle = static_cast<uint64_t>(
        floor(static_cast<double>(streamable_cycle) * stats.overlap_factor));

    if (budget_by_bga_overlap)
    {
        stats.bga_window_occupancy =
            compute_overlap_window == 0
                ? 0.0
                : clamp01(static_cast<double>(bga_overlap.hidden_accumulate_cycle) /
                          static_cast<double>(compute_overlap_window));
        stats.bga_contention_score =
            clamp01(bga_overlap.overlap_factor * stats.bga_window_occupancy);

        /*
         * V8 prevents DRAF streaming overlap and BGA overlap from both consuming the same
         * compute/padding slack at full strength. The coefficient is structural rather than
         * target-derived: BGA contention can at most halve the DRAF staging benefit because
         * row staging still overlaps command/address flow even when the accumulator is busy.
         */
        stats.budget_factor = clamp01(1.0 - 0.50 * stats.bga_contention_score);
    }

    stats.hidden_access_cycle = static_cast<uint64_t>(
        floor(static_cast<double>(raw_hidden_access_cycle) * stats.budget_factor));
    return stats;
}

DrafMemoryEfficiencyStats buildDrafMemoryEfficiencyStats(
    const DrafStats& draf,
    const DrafPaddingExposureStats& exposure,
    uint64_t memory_bound_access_cycle)
{
    DrafMemoryEfficiencyStats stats;
    stats.coo_bytes_per_nnz = kSparseValueBytes + kCooIndexBytesPerNnz;
    stats.draf_bytes_per_nnz =
        kSparseValueBytes * draf.expansion_ratio + kDrafCompactMetadataBytesPerNnz;
    stats.draf_vs_coo_ratio =
        stats.coo_bytes_per_nnz == 0.0
            ? 1.0
            : stats.draf_bytes_per_nnz / stats.coo_bytes_per_nnz;
    stats.memory_saving_factor = clamp01(1.0 - stats.draf_vs_coo_ratio);

    /*
     * DRAF's storage advantage is a memory-traffic benefit, not a compute/BGA benefit.
     * Fragmented low-NNZ columns expose more empty DRAF area and metadata movement, so
     * only the non-fragmented portion of the storage saving is allowed to reduce access
     * cycles.
     */
    double fragmentation_guard = clamp01(1.0 - exposure.fragmentation);
    stats.exposed_saving_factor = stats.memory_saving_factor * fragmentation_guard;
    stats.saved_access_cycle = static_cast<uint64_t>(
        floor(static_cast<double>(memory_bound_access_cycle) *
              stats.exposed_saving_factor));
    return stats;
}


} // namespace spmv
