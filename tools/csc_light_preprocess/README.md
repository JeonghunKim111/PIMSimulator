# CSC image preprocessor

This directory contains the standalone preprocessor used by the CSC M7 SpMV
path. It converts a sparse matrix stored in the repository's text CSC format
into the physical, column-aligned image consumed by PIMSimulator. No checkout
of the separate SparsePIM repository is required.

The input text format is:

1. `num_rows num_columns nnz`
2. `num_columns + 1` integer column pointers
3. `nnz` integer row indices
4. `nnz` floating-point values

See `testdata/toy_csc.txt` for a minimal example. The physical image contract
is documented in [`../../docs/csc/CSC_IMAGE_FORMAT.md`](../../docs/csc/CSC_IMAGE_FORMAT.md).

## Requirements

- Python 3
- A C++17 compiler with OpenMP support (the default is `g++`)

The Python entry point builds `csc_light_preprocess_bin` on demand. It can also
be built explicitly:

```bash
make -C tools/csc_light_preprocess
```

## Generate the toy image

The export directory must not exist or must be empty; the exporter never
overwrites an existing image.

```bash
python3 tools/csc_light_preprocess/csc_light_preprocess.py \
  --matrix tools/csc_light_preprocess/testdata/toy_csc.txt \
  --layout csc_aligned --policy round_robin --segment-nnz 0 \
  --warmup 0 --repeat 1 --no-csv \
  --export-image /tmp/csc_toy_image
```

Then run the M7 end-to-end simulator test:

```bash
CSC_EXTERNAL_IMAGE=/tmp/csc_toy_image \
  ./sim --gtest_filter=CSCM7BFullRunTest.ExternalToyPrintsFullCycleBreakdown
```

For another matrix, replace the `--matrix` argument. The M7 image format
requires `--layout csc_aligned`, 64 bank groups, and `--segment-nnz 0`.

## Additional tools

- `benchmark_csc_light_preprocess.py` runs the general preprocessing benchmark.
- `benchmark_csc_aligned_m2.py` runs an in-memory aligned-layout sweep over a
  directory of matrices in the same text CSC format.

Generated binaries, Python caches, and benchmark output under `results/` are
ignored by Git.
