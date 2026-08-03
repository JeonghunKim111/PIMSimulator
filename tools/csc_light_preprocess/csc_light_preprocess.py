#!/usr/bin/env python3
"""CLI and TXT loader for the C++ lightweight CSC preprocessing hot path."""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


CSV_FIELDS = [
    "matrix_name", "rows", "columns", "nnz", "num_bg", "layout", "policy",
    "choices", "segment_nnz", "sketch_bits", "alpha", "beta", "seed",
    "threads", "warmup", "repeat", "descriptor_count", "metadata_bytes",
    "matrix_load_ms", "feature_extraction_ms_min",
    "feature_extraction_ms_median", "mapping_ms_min", "mapping_ms_median",
    "ordering_ms_min", "ordering_ms_median",
    "physical_materialization_ms_min", "physical_materialization_ms_median",
    "descriptor_generation_ms_min", "descriptor_generation_ms_median",
    "preprocessing_total_ms_min", "preprocessing_total_ms_median",
    "preprocessing_total_ms_mean", "preprocessing_total_ms_p95",
    "preprocessing_total_ms_stddev", "verification_ms",
    "quality_analysis_ms", "output_write_ms", "verification_passed",
    "total_assigned_nnz", "avg_bg_nnz", "max_bg_nnz", "min_bg_nnz",
    "max_avg_load_ratio", "bg_load_stddev", "bg_load_cv", "empty_bg_count",
    "avg_descriptors_per_bg", "max_descriptors_per_bg",
    "avg_bg_row_sketch_similarity", "avg_unit_bg_sketch_similarity",
    "sampled_exact_jaccard", "avg_row_fanout", "median_row_fanout",
    "p95_row_fanout", "max_row_fanout", "single_bg_row_fraction",
    "avg_remote_bg_contributions", "physical_read_bytes",
    "physical_write_bytes", "effective_memory_bandwidth_gbps",
    "theoretical_min_data_movement_ms", "latency_class",
    "input_value_type", "output_value_type", "fp32_conversion_count",
    "nan_count", "inf_count", "fp32_conversion_ms",
    "aligned_materialization_ms", "metadata_generation_ms",
    "aligned_validation_ms", "aligned_preprocessing_total_ms",
    "checksum_ms", "serialization_ms", "file_write_ms",
    "export_validation_ms",
    "aligned_useful_value_bytes", "aligned_useful_index_bytes",
    "aligned_physical_value_bytes", "aligned_physical_index_bytes",
    "aligned_value_padding_bytes", "aligned_index_padding_bytes",
    "aligned_descriptor_count",
    "total_chunks", "simd_utilization", "bg_nnz_cv_aligned", "bg_chunk_cv",
    "lockstep_rounds", "critical_bg", "value_transactions",
    "row_index_transactions", "logical_mul_events",
]


def load_txt_csc(path: Path):
    """Load headerless, column-major row/column/value ASCII triplets."""
    start = time.perf_counter_ns()
    data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if data.shape[1] != 3:
        raise ValueError(f"{path}: expected exactly 3 fields per line")
    rows64 = data[:, 0].astype(np.int64)
    cols64 = data[:, 1].astype(np.int64)
    values = np.ascontiguousarray(data[:, 2], dtype=np.float64)
    if np.any(rows64 < 0) or np.any(cols64 < 0):
        raise ValueError(f"{path}: negative row/column index")
    if rows64.size and (rows64.max() > np.iinfo(np.uint32).max or
                        cols64.max() > np.iinfo(np.uint32).max):
        raise ValueError(f"{path}: index exceeds uint32 range")
    if cols64.size > 1 and np.any(cols64[1:] < cols64[:-1]):
        raise ValueError(f"{path}: triplets are not in CSC column order")
    n_rows = int(rows64.max()) + 1 if rows64.size else 0
    n_cols = int(cols64.max()) + 1 if cols64.size else 0
    counts = np.bincount(cols64, minlength=n_cols).astype(np.uint64)
    col_ptr = np.empty(n_cols + 1, dtype=np.uint64)
    col_ptr[0] = 0
    np.cumsum(counts, out=col_ptr[1:])
    row_idx = np.ascontiguousarray(rows64, dtype=np.uint32)
    del data, rows64, cols64, counts
    load_ms = (time.perf_counter_ns() - start) / 1e6
    return n_rows, n_cols, col_ptr, row_idx, values, load_ms


def write_binary_csc(path: Path, n_rows: int, n_cols: int, col_ptr, row_idx, values):
    """Interchange only; its time is part of loading/staging, never preprocessing."""
    with path.open("wb") as f:
        f.write(struct.pack("<QQQ", n_rows, n_cols, len(row_idx)))
        col_ptr.tofile(f)
        row_idx.tofile(f)
        values.tofile(f)


def build_if_needed(binary: Path):
    source = binary.with_name("csc_light_preprocess.cpp")
    if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(["make", "-C", str(binary.parent)], check=True)


def latency_class(ms: float) -> str:
    if ms <= 10:
        return "<=10ms"
    if ms <= 50:
        return "10-50ms"
    if ms <= 100:
        return "50-100ms"
    if ms < 1000:
        return "100ms-1s"
    return ">=1s"


def append_csv(path: Path, record: dict):
    start = time.perf_counter_ns()
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in CSV_FIELDS})
    return (time.perf_counter_ns() - start) / 1e6


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", required=True, type=Path)
    p.add_argument("--layout",
                   choices=("physical", "physical_legacy", "logical", "csc_aligned"),
                   default="physical")
    p.add_argument("--policy", choices=("round_robin", "load_only", "load_similarity"),
                   default="load_similarity")
    p.add_argument("--num-bg", type=int, default=64)
    p.add_argument("--choices", type=int, choices=(2, 4), default=4)
    p.add_argument("--segment-nnz", type=int, default=0)
    p.add_argument("--sketch-bits", type=int, choices=(128, 256), default=256)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--output", type=Path,
                   default=Path(__file__).parent / "results/csc_light_preprocess_results.csv")
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--keep-binary", type=Path)
    p.add_argument("--export-image", type=Path,
                   help="new, empty output directory for csc_aligned image files")
    return p


def run(args) -> dict:
    if args.layout == "csc_aligned" and (args.num_bg != 64 or args.segment_nnz != 0):
        raise ValueError("csc_aligned requires --num-bg 64 and --segment-nnz 0")
    if args.export_image and args.layout != "csc_aligned":
        raise ValueError("--export-image requires --layout csc_aligned")
    n_rows, n_cols, col_ptr, row_idx, values, matrix_load_ms = load_txt_csc(args.matrix)
    binary = Path(__file__).with_name("csc_light_preprocess_bin")
    build_if_needed(binary)
    if args.keep_binary:
        staging = args.keep_binary
        staging.parent.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        fd, name = tempfile.mkstemp(prefix="csc_light_", suffix=".bin")
        os.close(fd)
        staging = Path(name)
        cleanup = True
    try:
        write_binary_csc(staging, n_rows, n_cols, col_ptr, row_idx, values)
        cmd = [
            str(binary), "--input", str(staging), "--layout", args.layout,
            "--policy", args.policy, "--num-bg", str(args.num_bg),
            "--choices", str(args.choices), "--segment-nnz", str(args.segment_nnz),
            "--sketch-bits", str(args.sketch_bits), "--alpha", str(args.alpha),
            "--beta", str(args.beta), "--seed", str(args.seed),
            "--threads", str(args.threads), "--warmup", str(args.warmup),
            "--repeat", str(args.repeat),
        ]
        if args.export_image:
            cmd.extend(["--export-dir", str(args.export_image)])
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
        record = json.loads(proc.stdout)
    finally:
        if cleanup:
            staging.unlink(missing_ok=True)
    record["matrix_name"] = args.matrix.stem.removesuffix("_csc")
    record["matrix_load_ms"] = matrix_load_ms
    record["latency_class"] = latency_class(record["preprocessing_total_ms_median"])
    record["output_write_ms"] = 0.0
    if not args.no_csv:
        record["output_write_ms"] = append_csv(args.output, record)
    return record


def main():
    args = make_parser().parse_args()
    record = run(args)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
