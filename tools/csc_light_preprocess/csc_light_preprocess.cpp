#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

using Clock = std::chrono::steady_clock;
using Sketch = std::array<uint64_t, 4>;

struct Config {
    std::string input, layout = "physical", policy = "load_similarity";
    std::string export_dir;
    uint32_t num_bg = 64, choices = 4, segment_nnz = 0, sketch_bits = 256;
    uint64_t seed = 1;
    double alpha = 1.0, beta = 0.1;
    int threads = 1, warmup = 3, repeat = 20;
};
struct Matrix {
    uint64_t rows = 0, cols = 0, nnz = 0;
    std::vector<uint64_t> col_ptr;
    std::vector<uint32_t> row_idx;
    std::vector<double> values;
};
struct Unit {
    uint32_t col, len;
    uint64_t start;
    Sketch sketch{};
};
struct Descriptor {
    uint32_t original_column_id;
    uint64_t nnz_start;
    uint32_t nnz_length;
    uint8_t global_bg_id;
};
struct Timing {
    double feature = 0, mapping = 0, ordering = 0, materialization = 0, descriptor = 0;
    double total() const { return feature + mapping + ordering + materialization + descriptor; }
};
struct Result {
    std::vector<Unit> units;
    std::vector<uint8_t> assignment;
    std::vector<uint64_t> bg_load, bg_begin;
    std::vector<uint32_t> bg_count, unit_order, permutation, inverse;
    std::vector<Descriptor> descriptors;
    std::vector<uint64_t> reordered_col_ptr;
    std::vector<uint32_t> reordered_row_idx;
    std::vector<double> reordered_values;
    std::vector<uint8_t> reordered_column_to_bg;
    std::vector<Sketch> bg_sketch;
};
struct AlignedDescriptor {
    uint64_t value_offset_bytes = 0, row_idx_offset_bytes = 0;
    uint32_t nnz_count = 0, x_slot = 0, original_col = 0, global_bg_id = 0;
};
struct AlignedBGImage {
    std::vector<uint8_t> values, row_indices;
    std::vector<AlignedDescriptor> descriptors;
    std::vector<uint32_t> x_permutation;
};
struct AlignedStats {
    uint64_t useful_value_bytes = 0, useful_index_bytes = 0;
    uint64_t physical_value_bytes = 0, physical_index_bytes = 0;
    uint64_t value_padding_bytes = 0, index_padding_bytes = 0;
    uint64_t fp32_conversion_count = 0, nan_count = 0, inf_count = 0;
    double fp32_conversion_ms = 0, aligned_materialization_ms = 0;
    double metadata_generation_ms = 0, validation_ms = 0;
    uint64_t descriptor_count = 0, total_chunks = 0, lockstep_rounds = 0;
    uint32_t critical_bg = 0;
    double simd_utilization = 0, bg_nnz_cv = 0, bg_chunk_cv = 0;
};
struct AlignedImage {
    uint64_t rows = 0, cols = 0, nnz = 0;
    std::string mapping_policy;
    std::vector<uint32_t> column_to_bg;
    std::vector<AlignedBGImage> bg;
    AlignedStats stats;
};
struct ExportTiming {
    double checksum_ms = 0, serialization_ms = 0, file_write_ms = 0;
    double export_validation_ms = 0;
};

static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static int popcount(const Sketch &s, int words) {
    int n = 0;
    for (int i = 0; i < words; ++i) n += __builtin_popcountll(s[i]);
    return n;
}
static double containment(const Sketch &a, const Sketch &b, int words) {
    int inter = 0, denom = popcount(a, words);
    for (int i = 0; i < words; ++i) inter += __builtin_popcountll(a[i] & b[i]);
    return denom ? static_cast<double>(inter) / denom : 0.0;
}
static Matrix load_binary(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open binary CSC");
    Matrix m;
    f.read(reinterpret_cast<char *>(&m.rows), 8);
    f.read(reinterpret_cast<char *>(&m.cols), 8);
    f.read(reinterpret_cast<char *>(&m.nnz), 8);
    m.col_ptr.resize(m.cols + 1);
    m.row_idx.resize(m.nnz);
    m.values.resize(m.nnz);
    f.read(reinterpret_cast<char *>(m.col_ptr.data()), 8 * m.col_ptr.size());
    f.read(reinterpret_cast<char *>(m.row_idx.data()), 4 * m.row_idx.size());
    f.read(reinterpret_cast<char *>(m.values.data()), 8 * m.values.size());
    if (!f || m.col_ptr.back() != m.nnz) throw std::runtime_error("invalid binary CSC");
    return m;
}
static Config parse(int argc, char **argv) {
    Config c;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::runtime_error("missing option value");
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--input") c.input = v;
        else if (k == "--layout") c.layout = v;
        else if (k == "--policy") c.policy = v;
        else if (k == "--num-bg") c.num_bg = std::stoul(v);
        else if (k == "--choices") c.choices = std::stoul(v);
        else if (k == "--segment-nnz") c.segment_nnz = std::stoul(v);
        else if (k == "--sketch-bits") c.sketch_bits = std::stoul(v);
        else if (k == "--alpha") c.alpha = std::stod(v);
        else if (k == "--beta") c.beta = std::stod(v);
        else if (k == "--seed") c.seed = std::stoull(v);
        else if (k == "--threads") c.threads = std::stoi(v);
        else if (k == "--warmup") c.warmup = std::stoi(v);
        else if (k == "--repeat") c.repeat = std::stoi(v);
        else if (k == "--export-dir") c.export_dir = v;
        else throw std::runtime_error("unknown option: " + k);
    }
    if (c.input.empty() || c.num_bg == 0 || c.num_bg > 64 ||
        (c.choices != 2 && c.choices != 4) ||
        (c.sketch_bits != 128 && c.sketch_bits != 256))
        throw std::runtime_error("invalid configuration");
    if (c.layout != "logical" && c.layout != "physical" &&
        c.layout != "physical_legacy" && c.layout != "csc_aligned")
        throw std::runtime_error("unknown layout mode");
    if (c.policy != "round_robin" && c.policy != "load_only" &&
        c.policy != "load_similarity")
        throw std::runtime_error("unknown mapping policy");
    if (c.layout == "csc_aligned" &&
        (c.num_bg != 64 || c.segment_nnz != 0))
        throw std::runtime_error(
            "csc_aligned requires num_bg=64 and segment_nnz=0");
    if (!c.export_dir.empty() && c.layout != "csc_aligned")
        throw std::runtime_error("--export-dir requires --layout csc_aligned");
    return c;
}

static Result preprocess(const Matrix &m, const Config &c, Timing &t) {
    Result r;
    const int words = c.sketch_bits / 64;
    auto a = Clock::now();
    std::vector<uint32_t> per_col(m.cols);
    uint64_t unit_count = 0;
    #pragma omp parallel for num_threads(c.threads) reduction(+:unit_count)
    for (uint64_t col = 0; col < m.cols; ++col) {
        uint64_t len = m.col_ptr[col + 1] - m.col_ptr[col];
        uint64_t n = (c.segment_nnz && len > c.segment_nnz)
                   ? (len + c.segment_nnz - 1) / c.segment_nnz : 1;
        per_col[col] = static_cast<uint32_t>(n);
        unit_count += n;
    }
    std::vector<uint64_t> offsets(m.cols + 1);
    for (uint64_t col = 0; col < m.cols; ++col) offsets[col + 1] = offsets[col] + per_col[col];
    r.units.resize(unit_count);
    #pragma omp parallel for num_threads(c.threads) schedule(dynamic, 64)
    for (uint64_t col = 0; col < m.cols; ++col) {
        uint64_t pos = m.col_ptr[col], end = m.col_ptr[col + 1];
        for (uint32_t s = 0; s < per_col[col]; ++s) {
            uint64_t stop = c.segment_nnz ? std::min(end, pos + c.segment_nnz) : end;
            Unit u{static_cast<uint32_t>(col), static_cast<uint32_t>(stop - pos), pos, {}};
            for (uint64_t p = pos; p < stop; ++p) {
                uint64_t h1 = splitmix64(static_cast<uint64_t>(m.row_idx[p]) ^ c.seed);
                uint64_t h2 = splitmix64(h1 ^ 0xd6e8feb86659fd93ULL);
                uint32_t b1 = h1 % c.sketch_bits, b2 = h2 % c.sketch_bits;
                u.sketch[b1 >> 6] |= 1ULL << (b1 & 63);
                u.sketch[b2 >> 6] |= 1ULL << (b2 & 63);
            }
            r.units[offsets[col] + s] = u;
            pos = stop;
        }
    }
    auto b = Clock::now(); t.feature = ms(a, b);

    r.assignment.resize(unit_count);
    r.bg_load.assign(c.num_bg, 0);
    r.bg_count.assign(c.num_bg, 0);
    r.bg_sketch.assign(c.num_bg, {});
    const double target = m.nnz ? static_cast<double>(m.nnz) / c.num_bg : 1.0;
    for (uint64_t i = 0; i < unit_count; ++i) {
        const Unit &u = r.units[i];
        if (c.policy == "round_robin") {
            uint32_t bg = u.col % c.num_bg;
            r.assignment[i] = static_cast<uint8_t>(bg);
            r.bg_load[bg] += u.len;
            r.bg_count[bg]++;
            for (int w = 0; w < words; ++w)
                r.bg_sketch[bg][w] |= u.sketch[w];
            continue;
        }
        std::array<uint32_t, 4> candidates{};
        for (uint32_t q = 0; q < c.choices; ++q) {
            uint64_t key = c.seed ^ (static_cast<uint64_t>(u.col) << 32) ^ u.start ^
                           (0x9e3779b97f4a7c15ULL * (q + 1));
            uint32_t bg = splitmix64(key) % c.num_bg;
            bool duplicate;
            do {
                duplicate = false;
                for (uint32_t z = 0; z < q; ++z) duplicate |= candidates[z] == bg;
                if (duplicate) bg = (bg + 1) % c.num_bg;
            } while (duplicate);
            candidates[q] = bg;
        }
        uint32_t best = candidates[0];
        double best_score = 1e300;
        for (uint32_t q = 0; q < c.choices; ++q) {
            uint32_t bg = candidates[q];
            // Projected load is normalized by the ideal NNZ/BG load.
            double score = c.alpha * ((r.bg_load[bg] + u.len) / target);
            // Similarity is sketch containment: popcount(unit & BG)/popcount(unit).
            if (c.policy == "load_similarity")
                score -= c.beta * containment(u.sketch, r.bg_sketch[bg], words);
            if (score < best_score || (score == best_score && bg < best)) {
                best_score = score; best = bg;
            }
        }
        r.assignment[i] = static_cast<uint8_t>(best);
        r.bg_load[best] += u.len;
        r.bg_count[best]++;
        for (int w = 0; w < words; ++w) r.bg_sketch[best][w] |= u.sketch[w];
    }
    auto d = Clock::now(); t.mapping = ms(b, d);

    r.bg_begin.resize(c.num_bg + 1);
    for (uint32_t bg = 0; bg < c.num_bg; ++bg) r.bg_begin[bg + 1] = r.bg_begin[bg] + r.bg_count[bg];
    r.unit_order.resize(unit_count);
    std::vector<uint64_t> cursor = r.bg_begin;
    for (uint64_t i = 0; i < unit_count; ++i) r.unit_order[cursor[r.assignment[i]]++] = i;
    std::vector<uint8_t> column_bg(m.cols, 0);
    for (uint64_t i = 0; i < unit_count; ++i)
        if (r.units[i].start == m.col_ptr[r.units[i].col]) column_bg[r.units[i].col] = r.assignment[i];
    std::vector<uint32_t> col_counts(c.num_bg);
    for (uint8_t bg : column_bg) col_counts[bg]++;
    std::vector<uint64_t> col_begin(c.num_bg + 1);
    for (uint32_t bg = 0; bg < c.num_bg; ++bg) col_begin[bg + 1] = col_begin[bg] + col_counts[bg];
    cursor = col_begin;
    r.permutation.resize(m.cols); r.inverse.resize(m.cols);
    r.reordered_column_to_bg.resize(m.cols);
    for (uint64_t col = 0; col < m.cols; ++col) {
        uint64_t k = cursor[column_bg[col]]++;
        r.permutation[k] = col; r.inverse[col] = k; r.reordered_column_to_bg[k] = column_bg[col];
    }
    auto e = Clock::now(); t.ordering = ms(d, e);

    if (c.layout == "physical" || c.layout == "physical_legacy") {
        r.reordered_col_ptr.resize(m.cols + 1);
        for (uint64_t k = 0; k < m.cols; ++k)
            r.reordered_col_ptr[k + 1] = r.reordered_col_ptr[k] +
                (m.col_ptr[r.permutation[k] + 1] - m.col_ptr[r.permutation[k]]);
        r.reordered_row_idx.resize(m.nnz); r.reordered_values.resize(m.nnz);
        #pragma omp parallel for num_threads(c.threads) schedule(dynamic, 64)
        for (uint64_t k = 0; k < m.cols; ++k) {
            uint32_t col = r.permutation[k];
            uint64_t src = m.col_ptr[col], len = m.col_ptr[col + 1] - src;
            std::memcpy(r.reordered_row_idx.data() + r.reordered_col_ptr[k],
                        m.row_idx.data() + src, len * sizeof(uint32_t));
            std::memcpy(r.reordered_values.data() + r.reordered_col_ptr[k],
                        m.values.data() + src, len * sizeof(double));
        }
    }
    auto f = Clock::now();
    t.materialization =
        (c.layout == "physical" || c.layout == "physical_legacy") ? ms(e, f) : 0.0;

    r.descriptors.resize(unit_count);
    #pragma omp parallel for num_threads(c.threads)
    for (uint64_t k = 0; k < unit_count; ++k) {
        uint32_t i = r.unit_order[k]; const Unit &u = r.units[i];
        r.descriptors[k] = {u.col, u.start, u.len, r.assignment[i]};
    }
    auto g = Clock::now(); t.descriptor = ms(f, g);
    return r;
}

static bool verify(const Matrix &m, const Config &c, const Result &r) {
    if (r.bg_begin.back() != r.units.size() || r.descriptors.size() != r.units.size()) return false;
    std::vector<uint8_t> covered(m.nnz, 0);
    for (const auto &d : r.descriptors) {
        if (d.original_column_id >= m.cols || d.global_bg_id >= c.num_bg ||
            d.nnz_start < m.col_ptr[d.original_column_id] ||
            d.nnz_start + d.nnz_length > m.col_ptr[d.original_column_id + 1]) return false;
        for (uint64_t p = d.nnz_start; p < d.nnz_start + d.nnz_length; ++p)
            if (++covered[p] != 1) return false;
    }
    if (std::find(covered.begin(), covered.end(), 0) != covered.end()) return false;
    std::vector<double> x(m.cols), y0(m.rows, 0), yd(m.rows, 0);
    for (uint64_t col = 0; col < m.cols; ++col) {
        x[col] = ((splitmix64(col + 7) >> 11) * 0x1.0p-53) - 0.5;
        for (uint64_t p = m.col_ptr[col]; p < m.col_ptr[col + 1]; ++p)
            y0[m.row_idx[p]] += m.values[p] * x[col];
    }
    for (const auto &d : r.descriptors)
        for (uint64_t p = d.nnz_start; p < d.nnz_start + d.nnz_length; ++p)
            yd[m.row_idx[p]] += m.values[p] * x[d.original_column_id];
    for (uint64_t row = 0; row < m.rows; ++row)
        if (std::abs(y0[row] - yd[row]) > 1e-9 * (1 + std::abs(y0[row]))) return false;
    if (c.layout == "physical" || c.layout == "physical_legacy") {
        if (r.reordered_col_ptr.size() != m.cols + 1 || r.reordered_col_ptr.back() != m.nnz ||
            r.reordered_row_idx.size() != m.nnz || r.reordered_values.size() != m.nnz) return false;
        std::vector<uint8_t> seen(m.cols);
        for (uint64_t k = 0; k < m.cols; ++k) {
            uint32_t col = r.permutation[k];
            if (col >= m.cols || seen[col]++ || r.inverse[col] != k) return false;
            uint64_t len = m.col_ptr[col + 1] - m.col_ptr[col];
            if (r.reordered_col_ptr[k + 1] - r.reordered_col_ptr[k] != len) return false;
            for (uint64_t q = 0; q < len; ++q) {
                uint64_t src = m.col_ptr[col] + q, dst = r.reordered_col_ptr[k] + q;
                if (m.row_idx[src] != r.reordered_row_idx[dst] ||
                    m.values[src] != r.reordered_values[dst]) return false;
            }
        }
    }
    return true;
}

static uint64_t align32(uint64_t n) { return (n + 31) / 32 * 32; }
static void put_u32(std::vector<uint8_t> &out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
static void put_u64(std::vector<uint8_t> &out, uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
static uint32_t get_u32(const std::vector<uint8_t> &in, uint64_t at) {
    if (at + 4 > in.size()) throw std::runtime_error("uint32 payload range");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= uint32_t(in[at + i]) << (8 * i);
    return v;
}
static uint64_t fnv1a64(const std::vector<uint8_t> &bytes) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint8_t b : bytes) { h ^= b; h *= 0x100000001b3ULL; }
    return h;
}
static std::string hex64(uint64_t v) {
    std::ostringstream s;
    s << std::hex << std::nouppercase << std::setw(16) << std::setfill('0') << v;
    return s.str();
}

static AlignedImage materialize_aligned(const Matrix &m, const Config &c,
                                        const Result &mapping) {
    if (c.num_bg != 64 || c.segment_nnz != 0)
        throw std::runtime_error("aligned materializer requires 64 BG and no segmentation");
    if (mapping.units.size() != m.cols || mapping.assignment.size() != m.cols ||
        mapping.permutation.size() != m.cols)
        throw std::runtime_error("aligned materializer requires one mapping unit per column");

    AlignedImage image;
    image.rows = m.rows; image.cols = m.cols; image.nnz = m.nnz;
    image.mapping_policy = c.policy;
    image.bg.resize(64); image.column_to_bg.resize(m.cols);
    for (uint64_t col = 0; col < m.cols; ++col)
        image.column_to_bg[col] = mapping.assignment[col];

    auto convert_begin = Clock::now();
    std::vector<float> fp32(m.nnz);
    for (uint64_t p = 0; p < m.nnz; ++p) {
        fp32[p] = static_cast<float>(m.values[p]);
        image.stats.nan_count += std::isnan(fp32[p]);
        image.stats.inf_count += std::isinf(fp32[p]);
    }
    image.stats.fp32_conversion_count = m.nnz;
    image.stats.fp32_conversion_ms = ms(convert_begin, Clock::now());

    struct Pending {
        uint32_t bg, col, nnz;
        uint64_t value_offset, row_offset;
    };
    std::vector<Pending> pending;
    pending.reserve(m.cols);
    auto materialize_begin = Clock::now();
    for (uint32_t ordered_col : mapping.permutation) {
        uint64_t begin = m.col_ptr[ordered_col];
        uint64_t count = m.col_ptr[ordered_col + 1] - begin;
        if (!count) continue;
        uint32_t bg = image.column_to_bg[ordered_col];
        if (bg >= 64 || count > UINT32_MAX)
            throw std::runtime_error("aligned column metadata outside format range");
        auto &dst = image.bg[bg];
        dst.values.resize(align32(dst.values.size()), 0);
        dst.row_indices.resize(align32(dst.row_indices.size()), 0);
        uint64_t value_offset = dst.values.size();
        uint64_t row_offset = dst.row_indices.size();
        for (uint64_t q = 0; q < count; ++q) {
            uint32_t bits = 0;
            std::memcpy(&bits, &fp32[begin + q], sizeof(bits));
            put_u32(dst.values, bits);
            put_u32(dst.row_indices, m.row_idx[begin + q]);
        }
        dst.values.resize(align32(dst.values.size()), 0);
        dst.row_indices.resize(align32(dst.row_indices.size()), 0);
        pending.push_back({bg, ordered_col, static_cast<uint32_t>(count),
                           value_offset, row_offset});
    }
    image.stats.aligned_materialization_ms = ms(materialize_begin, Clock::now());

    auto metadata_begin = Clock::now();
    for (const Pending &p : pending) {
        auto &dst = image.bg[p.bg];
        uint32_t slot = static_cast<uint32_t>(dst.descriptors.size());
        dst.descriptors.push_back({p.value_offset, p.row_offset, p.nnz, slot,
                                   p.col, p.bg});
        dst.x_permutation.push_back(p.col);
    }
    image.stats.metadata_generation_ms = ms(metadata_begin, Clock::now());
    std::vector<uint64_t> bg_nnz(64), bg_chunks(64);
    for (uint32_t g = 0; g < 64; ++g) for (const auto &d : image.bg[g].descriptors) {
        image.stats.descriptor_count++;
        bg_nnz[g] += d.nnz_count;
        bg_chunks[g] += (d.nnz_count + 7) / 8;
    }
    image.stats.total_chunks = std::accumulate(bg_chunks.begin(), bg_chunks.end(), uint64_t(0));
    image.stats.critical_bg = std::distance(bg_chunks.begin(), std::max_element(bg_chunks.begin(), bg_chunks.end()));
    image.stats.lockstep_rounds = bg_chunks[image.stats.critical_bg];
    image.stats.simd_utilization = image.stats.total_chunks ? double(m.nnz) / (image.stats.total_chunks * 8) : 0;
    auto cv = [](const std::vector<uint64_t>&v){double mean=std::accumulate(v.begin(),v.end(),0.0)/v.size();if(!mean)return 0.0;double var=0;for(auto x:v)var+=(x-mean)*(x-mean);return std::sqrt(var/v.size())/mean;};
    image.stats.bg_nnz_cv = cv(bg_nnz); image.stats.bg_chunk_cv = cv(bg_chunks);
    image.stats.useful_value_bytes = m.nnz * 4;
    image.stats.useful_index_bytes = m.nnz * 4;
    for (const auto &bg : image.bg) {
        image.stats.physical_value_bytes += bg.values.size();
        image.stats.physical_index_bytes += bg.row_indices.size();
    }
    image.stats.value_padding_bytes =
        image.stats.physical_value_bytes - image.stats.useful_value_bytes;
    image.stats.index_padding_bytes =
        image.stats.physical_index_bytes - image.stats.useful_index_bytes;
    return image;
}

static void validate_aligned(const Matrix &m, AlignedImage &image) {
    auto begin = Clock::now();
    if (image.bg.size() != 64 || image.column_to_bg.size() != m.cols)
        throw std::runtime_error("aligned topology invariant");
    std::vector<uint8_t> seen(m.cols, 0);
    uint64_t nnz = 0;
    for (uint32_t g = 0; g < 64; ++g) {
        const auto &bg = image.bg[g];
        if (bg.descriptors.size() != bg.x_permutation.size())
            throw std::runtime_error("aligned x permutation size invariant");
        for (uint32_t i = 0; i < bg.descriptors.size(); ++i) {
            const auto &d = bg.descriptors[i];
            if (d.global_bg_id != g || d.original_col >= m.cols ||
                image.column_to_bg[d.original_col] != g || !d.nnz_count ||
                d.value_offset_bytes % 32 || d.row_idx_offset_bytes % 32 ||
                d.x_slot != i || d.x_slot >= bg.x_permutation.size() ||
                bg.x_permutation[d.x_slot] != d.original_col)
                throw std::runtime_error("aligned descriptor invariant");
            if (d.value_offset_bytes + uint64_t(d.nnz_count) * 4 > bg.values.size() ||
                d.row_idx_offset_bytes + uint64_t(d.nnz_count) * 4 > bg.row_indices.size())
                throw std::runtime_error("aligned payload range invariant");
            for (uint32_t q = 0; q < d.nnz_count; ++q)
                if (get_u32(bg.row_indices, d.row_idx_offset_bytes + uint64_t(q) * 4) >= m.rows)
                    throw std::runtime_error("aligned row range invariant");
            if (++seen[d.original_col] != 1)
                throw std::runtime_error("aligned duplicate column owner");
            nnz += d.nnz_count;
        }
        std::vector<uint8_t> useful_value(bg.values.size(), 0), useful_row(bg.row_indices.size(), 0);
        for (const auto &d : bg.descriptors)
            for (uint64_t q = 0; q < uint64_t(d.nnz_count) * 4; ++q) {
                useful_value[d.value_offset_bytes + q] = 1;
                useful_row[d.row_idx_offset_bytes + q] = 1;
            }
        for (uint64_t i = 0; i < bg.values.size(); ++i)
            if (!useful_value[i] && bg.values[i] != 0)
                throw std::runtime_error("nonzero value padding");
        for (uint64_t i = 0; i < bg.row_indices.size(); ++i)
            if (!useful_row[i] && bg.row_indices[i] != 0)
                throw std::runtime_error("nonzero row-index padding");
    }
    for (uint64_t col = 0; col < m.cols; ++col) {
        bool nonempty = m.col_ptr[col + 1] != m.col_ptr[col];
        if (seen[col] != static_cast<uint8_t>(nonempty))
            throw std::runtime_error("aligned one descriptor per nonempty column invariant");
    }
    if (nnz != m.nnz) throw std::runtime_error("aligned descriptor NNZ sum invariant");
    image.stats.validation_ms = ms(begin, Clock::now());
}

static std::vector<uint8_t> serialize_descriptors(const AlignedBGImage &bg) {
    std::vector<uint8_t> bytes; bytes.reserve(bg.descriptors.size() * 32);
    for (const auto &d : bg.descriptors) {
        put_u64(bytes, d.value_offset_bytes); put_u64(bytes, d.row_idx_offset_bytes);
        put_u32(bytes, d.nnz_count); put_u32(bytes, d.x_slot);
        put_u32(bytes, d.original_col); put_u32(bytes, d.global_bg_id);
    }
    return bytes;
}
static std::vector<uint8_t> serialize_x(const AlignedBGImage &bg) {
    std::vector<uint8_t> bytes; bytes.reserve(bg.x_permutation.size() * 4);
    for (uint32_t col : bg.x_permutation) put_u32(bytes, col);
    return bytes;
}
static void write_bytes(const std::filesystem::path &path,
                        const std::vector<uint8_t> &bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot create " + path.string());
    if (!bytes.empty()) f.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    if (!f) throw std::runtime_error("failed writing " + path.string());
}
static std::vector<uint8_t> read_written_bytes(const std::filesystem::path &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot verify " + path.string());
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}
template <typename T>
static void json_array(std::ostream &out, const std::vector<T> &v) {
    out << '[';
    for (size_t i = 0; i < v.size(); ++i) { if (i) out << ','; out << v[i]; }
    out << ']';
}
static void json_string_array(std::ostream &out, const std::vector<std::string> &v) {
    out << '[';
    for (size_t i = 0; i < v.size(); ++i) { if (i) out << ','; out << '"' << v[i] << '"'; }
    out << ']';
}

static ExportTiming export_aligned(const AlignedImage &image, const Config &c) {
    namespace fs = std::filesystem;
    fs::path dir(c.export_dir);
    if (fs::exists(dir) && fs::directory_iterator(dir) != fs::directory_iterator())
        throw std::runtime_error("export directory is not empty: " + dir.string());
    fs::create_directories(dir);
    ExportTiming timing;
    std::vector<uint64_t> desc_counts, value_sizes, row_sizes, x_counts;
    std::vector<std::string> value_hash, row_hash, desc_hash, x_hash;
    std::vector<std::vector<uint8_t>> descriptors(64), xbytes(64);
    auto serialize_begin = Clock::now();
    for (uint32_t g = 0; g < 64; ++g) {
        descriptors[g] = serialize_descriptors(image.bg[g]);
        xbytes[g] = serialize_x(image.bg[g]);
        desc_counts.push_back(image.bg[g].descriptors.size());
        value_sizes.push_back(image.bg[g].values.size());
        row_sizes.push_back(image.bg[g].row_indices.size());
        x_counts.push_back(image.bg[g].x_permutation.size());
    }
    timing.serialization_ms = ms(serialize_begin, Clock::now());
    auto checksum_begin = Clock::now();
    for (uint32_t g = 0; g < 64; ++g) {
        value_hash.push_back(hex64(fnv1a64(image.bg[g].values)));
        row_hash.push_back(hex64(fnv1a64(image.bg[g].row_indices)));
        desc_hash.push_back(hex64(fnv1a64(descriptors[g])));
        x_hash.push_back(hex64(fnv1a64(xbytes[g])));
    }
    timing.checksum_ms = ms(checksum_begin, Clock::now());
    auto write_begin = Clock::now();
    for (uint32_t g = 0; g < 64; ++g) {
        std::ostringstream prefix; prefix << "bg_" << std::setw(2) << std::setfill('0') << g;
        write_bytes(dir / (prefix.str() + "_values.bin"), image.bg[g].values);
        write_bytes(dir / (prefix.str() + "_row_idx.bin"), image.bg[g].row_indices);
        write_bytes(dir / (prefix.str() + "_descriptors.bin"), descriptors[g]);
        write_bytes(dir / (prefix.str() + "_x_permutation.bin"), xbytes[g]);
    }
    timing.file_write_ms = ms(write_begin, Clock::now());
    auto export_validation_begin = Clock::now();
    for (uint32_t g = 0; g < 64; ++g) {
        std::ostringstream prefix; prefix << "bg_" << std::setw(2) << std::setfill('0') << g;
        auto verify_file = [&](const std::string &suffix, uint64_t size,
                               const std::string &checksum) {
            auto bytes = read_written_bytes(dir / (prefix.str() + suffix));
            if (bytes.size() != size || hex64(fnv1a64(bytes)) != checksum)
                throw std::runtime_error("exported file size/checksum invariant");
        };
        verify_file("_values.bin", value_sizes[g], value_hash[g]);
        verify_file("_row_idx.bin", row_sizes[g], row_hash[g]);
        verify_file("_descriptors.bin", desc_counts[g] * 32, desc_hash[g]);
        verify_file("_x_permutation.bin", x_counts[g] * 4, x_hash[g]);
    }
    timing.export_validation_ms = ms(export_validation_begin, Clock::now());

    std::ofstream manifest(dir / "manifest.json");
    if (!manifest) throw std::runtime_error("cannot create manifest.json");
    manifest << std::setprecision(10) << "{\n"
      << "  \"format_magic\":\"SPCSCIMG\",\n  \"format_version\":1,\n"
      << "  \"endianness\":\"little\",\n  \"precision\":\"FP32\",\n"
      << "  \"input_value_type\":\"float64\",\n  \"output_value_type\":\"float32\",\n"
      << "  \"row_index_type\":\"uint32\",\n  \"offset_type\":\"uint64\",\n"
      << "  \"alignment_bytes\":32,\n  \"num_global_bgs\":64,\n"
      << "  \"descriptor_record_bytes\":32,\n  \"segment_nnz\":0,\n"
      << "  \"matrix_rows\":" << image.rows << ",\n  \"matrix_columns\":" << image.cols
      << ",\n  \"matrix_nnz\":" << image.nnz << ",\n"
      << "  \"mapping_policy\":\"" << image.mapping_policy << "\",\n"
      << "  \"fp32_conversion_count\":" << image.stats.fp32_conversion_count << ",\n"
      << "  \"nan_count\":" << image.stats.nan_count << ",\n"
      << "  \"inf_count\":" << image.stats.inf_count << ",\n"
      << "  \"fp32_conversion_ms\":" << image.stats.fp32_conversion_ms << ",\n"
      << "  \"aligned_materialization_ms\":" << image.stats.aligned_materialization_ms << ",\n"
      << "  \"metadata_generation_ms\":" << image.stats.metadata_generation_ms << ",\n"
      << "  \"validation_ms\":" << image.stats.validation_ms << ",\n"
      << "  \"checksum_ms\":" << timing.checksum_ms << ",\n"
      << "  \"serialization_ms\":" << timing.serialization_ms << ",\n"
      << "  \"file_write_ms\":" << timing.file_write_ms << ",\n"
      << "  \"export_validation_ms\":" << timing.export_validation_ms << ",\n"
      << "  \"useful_value_bytes\":" << image.stats.useful_value_bytes << ",\n"
      << "  \"useful_index_bytes\":" << image.stats.useful_index_bytes << ",\n"
      << "  \"physical_value_bytes\":" << image.stats.physical_value_bytes << ",\n"
      << "  \"physical_index_bytes\":" << image.stats.physical_index_bytes << ",\n"
      << "  \"value_padding_bytes\":" << image.stats.value_padding_bytes << ",\n"
      << "  \"index_padding_bytes\":" << image.stats.index_padding_bytes << ",\n";
    manifest << "  \"total_chunks\":" << image.stats.total_chunks << ",\n"
      << "  \"lockstep_rounds\":" << image.stats.lockstep_rounds << ",\n"
      << "  \"critical_bg\":" << image.stats.critical_bg << ",\n"
      << "  \"simd_utilization\":" << image.stats.simd_utilization << ",\n"
      << "  \"bg_nnz_cv\":" << image.stats.bg_nnz_cv << ",\n"
      << "  \"bg_chunk_cv\":" << image.stats.bg_chunk_cv << ",\n";
    manifest << "  \"bg_descriptor_counts\":"; json_array(manifest, desc_counts); manifest << ",\n";
    manifest << "  \"column_to_bg\":"; json_array(manifest, image.column_to_bg); manifest << ",\n";
    manifest << "  \"bg_value_stream_bytes\":"; json_array(manifest, value_sizes); manifest << ",\n";
    manifest << "  \"bg_row_index_stream_bytes\":"; json_array(manifest, row_sizes); manifest << ",\n";
    manifest << "  \"bg_x_permutation_counts\":"; json_array(manifest, x_counts); manifest << ",\n";
    manifest << "  \"bg_values_fnv1a64\":"; json_string_array(manifest, value_hash); manifest << ",\n";
    manifest << "  \"bg_row_idx_fnv1a64\":"; json_string_array(manifest, row_hash); manifest << ",\n";
    manifest << "  \"bg_descriptors_fnv1a64\":"; json_string_array(manifest, desc_hash); manifest << ",\n";
    manifest << "  \"bg_x_permutation_fnv1a64\":"; json_string_array(manifest, x_hash); manifest << "\n}\n";
    if (!manifest) throw std::runtime_error("failed writing manifest.json");
    return timing;
}

struct Quality {
    double avg_load, max_ratio, load_sd, load_cv, avg_count;
    uint64_t max_load, min_load, empty, max_count;
    double bg_sim, unit_sim, avg_fanout, median_fanout, p95_fanout;
    uint64_t max_fanout; double single_fraction, avg_remote;
};
static Quality quality(const Matrix &m, const Config &c, const Result &r) {
    Quality q{};
    q.avg_load = static_cast<double>(m.nnz) / c.num_bg;
    q.max_load = *std::max_element(r.bg_load.begin(), r.bg_load.end());
    q.min_load = *std::min_element(r.bg_load.begin(), r.bg_load.end());
    q.max_ratio = q.avg_load ? q.max_load / q.avg_load : 0;
    for (auto x : r.bg_load) { double z = x - q.avg_load; q.load_sd += z * z; q.empty += x == 0; }
    q.load_sd = std::sqrt(q.load_sd / c.num_bg);
    q.load_cv = q.avg_load ? q.load_sd / q.avg_load : 0;
    q.avg_count = static_cast<double>(r.units.size()) / c.num_bg;
    q.max_count = *std::max_element(r.bg_count.begin(), r.bg_count.end());
    int words = c.sketch_bits / 64; uint64_t pairs = 0;
    for (uint32_t a = 0; a < c.num_bg; ++a) for (uint32_t b = a + 1; b < c.num_bg; ++b) {
        Sketch uni{}, inter{};
        for (int w = 0; w < words; ++w) { uni[w] = r.bg_sketch[a][w] | r.bg_sketch[b][w]; inter[w] = r.bg_sketch[a][w] & r.bg_sketch[b][w]; }
        int den = popcount(uni, words);
        q.bg_sim += den ? static_cast<double>(popcount(inter, words)) / den : 0; pairs++;
    }
    if (pairs) q.bg_sim /= pairs;
    for (uint64_t i = 0; i < r.units.size(); ++i)
        q.unit_sim += containment(r.units[i].sketch, r.bg_sketch[r.assignment[i]], words);
    if (!r.units.empty()) q.unit_sim /= r.units.size();
    std::vector<uint64_t> masks(m.rows, 0);
    for (const auto &d : r.descriptors)
        for (uint64_t p = d.nnz_start; p < d.nnz_start + d.nnz_length; ++p)
            masks[m.row_idx[p]] |= 1ULL << d.global_bg_id;
    std::vector<uint32_t> fan;
    fan.reserve(m.rows);
    for (uint64_t mask : masks) if (mask) fan.push_back(__builtin_popcountll(mask));
    if (!fan.empty()) {
        std::sort(fan.begin(), fan.end());
        q.avg_fanout = std::accumulate(fan.begin(), fan.end(), 0.0) / fan.size();
        q.median_fanout = fan[(fan.size() - 1) / 2];
        q.p95_fanout = fan[static_cast<size_t>(std::ceil(.95 * fan.size())) - 1];
        q.max_fanout = fan.back();
        q.single_fraction = std::count(fan.begin(), fan.end(), 1) / static_cast<double>(fan.size());
        q.avg_remote = q.avg_fanout - 1.0;
    }
    return q;
}
static double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    double x = p * (v.size() - 1), frac = x - std::floor(x);
    size_t i = static_cast<size_t>(x);
    return v[i] * (1 - frac) + v[std::min(i + 1, v.size() - 1)] * frac;
}
static void stat(const std::vector<Timing> &ts, double Timing::*field,
                 double &mn, double &med, double &mean, double &p95, double &sd) {
    std::vector<double> v; v.reserve(ts.size());
    for (auto &t : ts) v.push_back(field ? t.*field : t.total());
    mn = *std::min_element(v.begin(), v.end()); med = percentile(v, .5);
    mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size(); p95 = percentile(v, .95);
    for (double x : v) sd += (x - mean) * (x - mean);
    sd = std::sqrt(sd / v.size());
}
int main(int argc, char **argv) try {
    Config c = parse(argc, argv); omp_set_dynamic(0); omp_set_num_threads(c.threads);
    Matrix m = load_binary(c.input); Timing tmp; Result final;
    for (int i = 0; i < c.warmup; ++i) final = preprocess(m, c, tmp);
    Timing probe; final = preprocess(m, c, probe);
    int repeats = c.repeat;
    if (probe.total() >= 10000) repeats = 1;
    else if (probe.total() >= 1000) repeats = std::min(repeats, 3);
    std::vector<Timing> ts; ts.reserve(repeats); ts.push_back(probe);
    for (int i = 1; i < repeats; ++i) { Timing z; final = preprocess(m, c, z); ts.push_back(z); }
    auto va = Clock::now(); bool ok = verify(m, c, final); auto vb = Clock::now();
    AlignedImage aligned;
    ExportTiming export_timing;
    bool has_aligned = c.layout == "csc_aligned";
    if (has_aligned) {
        aligned = materialize_aligned(m, c, final);
        validate_aligned(m, aligned);
        if (!c.export_dir.empty()) export_timing = export_aligned(aligned, c);
    }
    auto qa = Clock::now(); Quality q = quality(m, c, final); auto qb = Clock::now();
    double fmn=0,fmed=0,fmean=0,fp95=0,fsd=0, mmn=0,mmed=0,mmean=0,mp95=0,msd=0, omn=0,omed=0,omean=0,op95=0,osd=0;
    double pmn=0,pmed=0,pmean=0,pp95=0,psd=0, dmn=0,dmed=0,dmean=0,dp95=0,dsd=0, tmn=0,tmed=0,tmean=0,tp95=0,tsd=0;
    stat(ts,&Timing::feature,fmn,fmed,fmean,fp95,fsd); stat(ts,&Timing::mapping,mmn,mmed,mmean,mp95,msd);
    stat(ts,&Timing::ordering,omn,omed,omean,op95,osd); stat(ts,&Timing::materialization,pmn,pmed,pmean,pp95,psd);
    stat(ts,&Timing::descriptor,dmn,dmed,dmean,dp95,dsd); stat(ts,nullptr,tmn,tmed,tmean,tp95,tsd);
    uint64_t metadata = final.descriptors.size()*sizeof(Descriptor) + final.bg_begin.size()*8 +
        final.permutation.size()*4 + final.inverse.size()*4 + final.reordered_column_to_bg.size();
    bool legacy_physical = c.layout=="physical" || c.layout=="physical_legacy";
    uint64_t read_bytes = legacy_physical ? m.nnz*12 + m.cols*16 : 0;
    uint64_t write_bytes = legacy_physical ? m.nnz*12 + (m.cols+1)*8 : 0;
    double bw = pmed > 0 ? (read_bytes + write_bytes) / pmed / 1e6 : 0;
    // STREAM-like single-socket sustained bandwidth is machine-dependent; 50 GB/s
    // is an explicitly labeled reference, not a measured hardware peak.
    double theoretical = (read_bytes + write_bytes) / 50e6;
    std::cout << std::setprecision(10) << "{"
      << "\"rows\":"<<m.rows<<",\"columns\":"<<m.cols<<",\"nnz\":"<<m.nnz
      <<",\"num_bg\":"<<c.num_bg<<",\"layout\":\""<<c.layout<<"\",\"policy\":\""<<c.policy<<"\""
      <<",\"choices\":"<<c.choices<<",\"segment_nnz\":"<<c.segment_nnz<<",\"sketch_bits\":"<<c.sketch_bits
      <<",\"alpha\":"<<c.alpha<<",\"beta\":"<<c.beta<<",\"seed\":"<<c.seed<<",\"threads\":"<<c.threads
      <<",\"warmup\":"<<c.warmup<<",\"repeat\":"<<repeats<<",\"descriptor_count\":"<<final.descriptors.size()
      <<",\"metadata_bytes\":"<<metadata
      <<",\"feature_extraction_ms_min\":"<<fmn<<",\"feature_extraction_ms_median\":"<<fmed
      <<",\"mapping_ms_min\":"<<mmn<<",\"mapping_ms_median\":"<<mmed
      <<",\"ordering_ms_min\":"<<omn<<",\"ordering_ms_median\":"<<omed
      <<",\"physical_materialization_ms_min\":"<<pmn<<",\"physical_materialization_ms_median\":"<<pmed
      <<",\"descriptor_generation_ms_min\":"<<dmn<<",\"descriptor_generation_ms_median\":"<<dmed
      <<",\"preprocessing_total_ms_min\":"<<tmn<<",\"preprocessing_total_ms_median\":"<<tmed
      <<",\"preprocessing_total_ms_mean\":"<<tmean<<",\"preprocessing_total_ms_p95\":"<<tp95
      <<",\"preprocessing_total_ms_stddev\":"<<tsd<<",\"verification_ms\":"<<ms(va,vb)
      <<",\"quality_analysis_ms\":"<<ms(qa,qb)<<",\"verification_passed\":"<<(ok?"true":"false")
      <<",\"total_assigned_nnz\":"<<std::accumulate(final.bg_load.begin(),final.bg_load.end(),uint64_t(0))
      <<",\"avg_bg_nnz\":"<<q.avg_load<<",\"max_bg_nnz\":"<<q.max_load<<",\"min_bg_nnz\":"<<q.min_load
      <<",\"max_avg_load_ratio\":"<<q.max_ratio<<",\"bg_load_stddev\":"<<q.load_sd<<",\"bg_load_cv\":"<<q.load_cv
      <<",\"empty_bg_count\":"<<q.empty<<",\"avg_descriptors_per_bg\":"<<q.avg_count
      <<",\"max_descriptors_per_bg\":"<<q.max_count<<",\"avg_bg_row_sketch_similarity\":"<<q.bg_sim
      <<",\"avg_unit_bg_sketch_similarity\":"<<q.unit_sim<<",\"sampled_exact_jaccard\":-1"
      <<",\"avg_row_fanout\":"<<q.avg_fanout<<",\"median_row_fanout\":"<<q.median_fanout
      <<",\"p95_row_fanout\":"<<q.p95_fanout<<",\"max_row_fanout\":"<<q.max_fanout
      <<",\"single_bg_row_fraction\":"<<q.single_fraction<<",\"avg_remote_bg_contributions\":"<<q.avg_remote
      <<",\"physical_read_bytes\":"<<read_bytes<<",\"physical_write_bytes\":"<<write_bytes
      <<",\"effective_memory_bandwidth_gbps\":"<<bw
      <<",\"theoretical_min_data_movement_ms\":"<<theoretical;
    if (has_aligned) {
      double aligned_total = tmed + aligned.stats.fp32_conversion_ms +
          aligned.stats.aligned_materialization_ms +
          aligned.stats.metadata_generation_ms + aligned.stats.validation_ms;
      std::cout
        <<",\"input_value_type\":\"float64\",\"output_value_type\":\"float32\""
        <<",\"fp32_conversion_count\":"<<aligned.stats.fp32_conversion_count
        <<",\"aligned_descriptor_count\":";
      std::cout<<aligned.stats.descriptor_count
        <<",\"nan_count\":"<<aligned.stats.nan_count<<",\"inf_count\":"<<aligned.stats.inf_count
        <<",\"fp32_conversion_ms\":"<<aligned.stats.fp32_conversion_ms
        <<",\"aligned_materialization_ms\":"<<aligned.stats.aligned_materialization_ms
        <<",\"metadata_generation_ms\":"<<aligned.stats.metadata_generation_ms
        <<",\"aligned_validation_ms\":"<<aligned.stats.validation_ms
        <<",\"aligned_preprocessing_total_ms\":"<<aligned_total
        <<",\"checksum_ms\":"<<export_timing.checksum_ms
        <<",\"serialization_ms\":"<<export_timing.serialization_ms
        <<",\"file_write_ms\":"<<export_timing.file_write_ms
        <<",\"export_validation_ms\":"<<export_timing.export_validation_ms
        <<",\"aligned_useful_value_bytes\":"<<aligned.stats.useful_value_bytes
        <<",\"aligned_useful_index_bytes\":"<<aligned.stats.useful_index_bytes
        <<",\"aligned_physical_value_bytes\":"<<aligned.stats.physical_value_bytes
        <<",\"aligned_physical_index_bytes\":"<<aligned.stats.physical_index_bytes
        <<",\"aligned_value_padding_bytes\":"<<aligned.stats.value_padding_bytes
        <<",\"aligned_index_padding_bytes\":"<<aligned.stats.index_padding_bytes;
      std::cout<<",\"total_chunks\":"<<aligned.stats.total_chunks
        <<",\"simd_utilization\":"<<aligned.stats.simd_utilization
        <<",\"bg_nnz_cv_aligned\":"<<aligned.stats.bg_nnz_cv
        <<",\"bg_chunk_cv\":"<<aligned.stats.bg_chunk_cv
        <<",\"lockstep_rounds\":"<<aligned.stats.lockstep_rounds
        <<",\"critical_bg\":"<<aligned.stats.critical_bg
        <<",\"value_transactions\":"<<aligned.stats.total_chunks
        <<",\"row_index_transactions\":"<<aligned.stats.total_chunks
        <<",\"logical_mul_events\":"<<aligned.stats.total_chunks;
    }
    std::cout << "}\n";
    return ok ? 0 : 2;
} catch (const std::exception &e) {
    std::cerr << e.what() << "\n"; return 1;
}
