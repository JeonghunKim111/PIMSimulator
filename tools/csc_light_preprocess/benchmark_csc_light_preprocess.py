#!/usr/bin/env python3
"""Batch orchestration with one TXT load per matrix."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from csc_light_preprocess import (
    append_csv, build_if_needed, latency_class, load_txt_csc, write_binary_csc)

def invoke(binary, staged, cfg):
    cmd = [str(binary), "--input", str(staged)]
    for key, value in cfg.items():
        cmd += ["--" + key.replace("_", "-"), str(value)]
    return json.loads(subprocess.run(
        cmd, check=True, text=True, capture_output=True).stdout)

def configs(mode, threads):
    base = dict(num_bg=64, sketch_bits=256, alpha=1.0, beta=0.1, seed=1,
                threads=threads, warmup=3, repeat=20)
    if mode == "cant":
        for segment in (0, 256, 512):
            for layout, policy, choices in (
                ("physical", "load_only", 2), ("physical", "load_only", 4),
                ("physical", "load_similarity", 2),
                ("physical", "load_similarity", 4),
                ("logical", "load_only", 4),
                ("logical", "load_similarity", 4)):
                yield dict(base, layout=layout, policy=policy, choices=choices,
                           segment_nnz=segment)
    else:
        for layout in ("physical", "logical"):
            yield dict(base, layout=layout, policy="load_similarity", choices=4,
                       segment_nnz=0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-dir", type=Path, default=Path("sparse_matrix_csc"))
    p.add_argument("--output", type=Path, default=Path(__file__).parent /
                   "results/csc_light_preprocess_results.csv")
    p.add_argument("--mode", choices=("cant", "all"), default="all")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.overwrite:
        args.output.unlink(missing_ok=True)
    paths = ([args.matrix_dir / "cant_csc.txt"] if args.mode == "cant"
             else sorted(args.matrix_dir.glob("*_csc.txt")))
    binary = Path(__file__).with_name("csc_light_preprocess_bin")
    build_if_needed(binary)
    for path in paths:
        rows, cols, col_ptr, row_idx, values, load_ms = load_txt_csc(path)
        with tempfile.NamedTemporaryFile(prefix="csc_light_", suffix=".bin") as tmp:
            staged = Path(tmp.name)
            write_binary_csc(staged, rows, cols, col_ptr, row_idx, values)
            for cfg in configs(args.mode, args.threads):
                rec = invoke(binary, staged, cfg)
                rec.update(matrix_name=path.stem.removesuffix("_csc"),
                           matrix_load_ms=load_ms, output_write_ms=0.0)
                rec["latency_class"] = latency_class(
                    rec["preprocessing_total_ms_median"])
                rec["output_write_ms"] = append_csv(args.output, rec)
                print(rec["matrix_name"], cfg["layout"], cfg["policy"],
                      f"segment={cfg['segment_nnz']}",
                      f"median={rec['preprocessing_total_ms_median']:.3f} ms",
                      "PASS" if rec["verification_passed"] else "FAIL", flush=True)

if __name__ == "__main__":
    main()
