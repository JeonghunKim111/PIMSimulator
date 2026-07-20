#!/usr/bin/env python3
"""Export a SciPy CSR NPZ without reordering or structural modification."""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

import numpy as np
from scipy import sparse

MAGIC = b"CSRDIR1\0"
HEADER = struct.Struct("<8sIIQ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as archive:
        required = {"data", "indices", "indptr", "shape", "format"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"missing NPZ arrays: {sorted(missing)}")
        fmt = archive["format"].item()
        if isinstance(fmt, bytes):
            fmt = fmt.decode("ascii")
        if fmt != "csr":
            raise ValueError(f"expected CSR NPZ, got format={fmt!r}")

    matrix = sparse.load_npz(args.input)
    if matrix.format != "csr":
        raise ValueError(f"expected CSR matrix, got {matrix.format}")
    rows, cols = matrix.shape
    if rows > np.iinfo(np.uint32).max or cols > np.iinfo(np.uint32).max:
        raise OverflowError("matrix dimensions exceed uint32")
    if matrix.nnz > np.iinfo(np.uint64).max:
        raise OverflowError("matrix nnz exceeds uint64")
    if matrix.indices.size and (
        matrix.indices.min() < 0 or matrix.indices.max() >= cols
    ):
        raise ValueError("column index outside matrix shape")

    # astype preserves stored entry order and duplicates. No sort/sum is used.
    indptr = np.asarray(matrix.indptr, dtype="<u8")
    indices = np.asarray(matrix.indices, dtype="<u4")
    data = np.asarray(matrix.data, dtype="<f4")
    if len(indptr) != rows + 1 or len(indices) != matrix.nnz or len(data) != matrix.nnz:
        raise AssertionError("invalid CSR array lengths")

    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        with temporary.open("wb") as output:
            output.write(HEADER.pack(MAGIC, rows, cols, matrix.nnz))
            indptr.tofile(output)
            indices.tofile(output)
            data.tofile(output)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
