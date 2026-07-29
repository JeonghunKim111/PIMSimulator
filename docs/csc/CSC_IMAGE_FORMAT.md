# Column-aligned BG-partitioned CSC image format

## Scope

This document defines the versioned binary contract between
`SparsePIM/csc_light_preprocess` and PIMSimulator's
`external_physical_image` input. The preprocessor output is the authoritative
BG-local physical byte image. PIMSimulator validates and consumes it without
recomputing mapping, column order, alignment, descriptors, x slots, or stream
bytes.

The format does not contain absolute HBM addresses. It contains global BG IDs
and BG-local byte offsets. PIMSimulator combines those fields with its Scheme8
topology using an address encoder:

```text
encode(channel, rank, bank_group, selected_bank, bg_local_stream_offset)
```

Integer addition to an absolute base address is not a valid encoder because
channel, rank, bank-group, bank, row, and column fields are interleaved by the
configured address mapping.

## Version 1 constants

| Property | Required value |
|---|---|
| Format magic | `SPCSCIMG` |
| Format version | `1` |
| Endianness | little-endian |
| Input value type | float64 |
| Output value type | IEEE-754 float32 |
| Row-index type | uint32 |
| Offset type | uint64 |
| Burst/alignment | 32 bytes |
| Global BG count | 64 |
| SIMD width | 8 |
| Descriptor size | 32 bytes |
| Checksum | FNV-1a 64-bit |

Version 1 forbids segmentation. `segment_nnz` must be zero. Every nonempty
original column has exactly one owner BG, one descriptor, and one x slot.
Empty columns remain part of the matrix dimensions but have no descriptor,
payload allocation, or x slot.

## Preprocessing modes

- `logical`: existing mapping and BG-local ordering only.
- `physical_legacy`: existing BG-major packed FP64 physical arrays, retained
  for reproduction of earlier experiments.
- `csc_aligned`: the opt-in image described here.

The `csc_aligned` materializer consumes the original conventional CSC arrays
and the existing mapping/order output directly. It must not align or convert
an intermediate `physical_legacy` array.

Mapping policy and input mode are separate:

- `round_robin`: internal or preprocessing ownership policy.
- `external_mapping`: original-column-to-global-BG input followed by internal
  PIMSimulator materialization.
- `external_physical_image`: load this authoritative image without
  rematerialization.

## Directory and files

```text
manifest.json
bg_00_values.bin
bg_00_row_idx.bin
bg_00_descriptors.bin
bg_00_x_permutation.bin
...
bg_63_values.bin
bg_63_row_idx.bin
bg_63_descriptors.bin
bg_63_x_permutation.bin
```

All files are mandatory. Empty BGs have zero-length files.

## BG-local streams

Each BG owns independent values and row-index byte streams. Before each column,
each stream cursor is rounded up to 32 bytes. The FP32 values or uint32 row
indices are written in little-endian order, then the allocation is extended
with zero bytes to the next 32-byte boundary.

Padding is physical space, not a logical NNZ. It is never multiplied, never
produces a partial, and is never accumulated. Production padding bytes are
zero for deterministic checksums. A test-only in-memory sentinel mode may use
quiet-NaN value bits and `0xffffffff` row indices; sentinel images are not
valid version 1 exports.

FP64-to-FP32 conversion occurs in the aligned materializer, not the exporter.
Functional reference and simulator execution both use the exported FP32
semantics.

## Descriptor serialization

Writers serialize every field explicitly. Dumping a compiler structure's
memory image is forbidden.

| Offset | Bytes | Type | Field |
|---:|---:|---|---|
| 0 | 8 | uint64 | `value_offset_bytes` |
| 8 | 8 | uint64 | `row_idx_offset_bytes` |
| 16 | 4 | uint32 | `nnz_count` |
| 20 | 4 | uint32 | `x_slot` |
| 24 | 4 | uint32 | `original_col` |
| 28 | 4 | uint32 | `global_bg_id` |

Offsets are relative to the corresponding BG stream. They are not simulator
addresses. Both offsets are multiples of 32.

Descriptor execution-order index equals `x_slot` in version 1.
`bg_NN_x_permutation.bin` is an unpadded sequence of explicitly serialized
little-endian uint32 original-column IDs:

```text
x_permutation[x_slot] = original_col
```

## Manifest

`manifest.json` is UTF-8 JSON. It contains the format constants, matrix rows,
columns and NNZ, mapping policy and parameters, input/output value types,
conversion statistics, phase timings, and these 64-entry arrays:

- `bg_descriptor_counts`
- `bg_value_stream_bytes`
- `bg_row_index_stream_bytes`
- `bg_x_permutation_counts`
- `column_to_bg` (one entry per original column, including empty columns)
- `bg_values_fnv1a64`
- `bg_row_idx_fnv1a64`
- `bg_descriptors_fnv1a64`
- `bg_x_permutation_fnv1a64`

Checksums are lowercase zero-padded 16-digit hexadecimal strings over the
exact bytes of each complete file. FNV-1a uses offset basis
`0xcbf29ce484222325` and prime `0x100000001b3`. The empty-file checksum is the
offset basis.

The manifest records:

- `fp32_conversion_count`, `nan_count`, and `inf_count`
- `fp32_conversion_ms`
- `aligned_materialization_ms`
- `metadata_generation_ms`
- `validation_ms`
- `checksum_ms`
- `serialization_ms`
- `file_write_ms`

Research in-memory preprocessing latency includes feature extraction, mapping,
BG-local ordering, FP32 conversion, aligned materialization, metadata
generation, and required validation. Checksum, serialization, and disk write
are separate export-infrastructure latency.

The manifest is published last. Export to an existing nonempty directory
fails; version 1 has no overwrite protocol.

Before publishing the manifest, the exporter reopens every binary file and
rechecks its exact size and FNV-1a checksum. The loader independently repeats
those checks.

## Required validation

Materializer and loader enforce:

1. Exactly 64 BG images.
2. One owner and one descriptor per nonempty original column.
3. No descriptor or x slot for an empty column.
4. Descriptor NNZ sum equals matrix NNZ.
5. Both descriptor offsets are 32-byte aligned.
6. Payload ranges are inside their declared streams.
7. `global_bg_id` matches the containing BG file.
8. x slots are unique, in range, and equal descriptor execution order.
9. `x_permutation[x_slot] == original_col`.
10. Row indices are smaller than matrix row count.
11. Production padding is zero.
12. Actual file sizes equal the manifest.
13. Recomputed file checksums equal the manifest.

Execution validation additionally requires:

```text
active lanes                       = matrix NNZ
generated partials                 = matrix NNZ
host accumulations                 = matrix NNZ
invalid-lane multiplications       = 0
invalid-lane partials              = 0
invalid-lane accumulations         = 0
value transactions                 = total chunks
row-index transactions             = total chunks
logical CSC MUL events             = total chunks
x scalar loads                     = descriptor count
```

`required_unique_x_bursts` and `actual_issued_x_transactions` are distinct
metrics. Result traffic is explicitly `none` or `analytical` in Milestone 2;
native BGA/GA traffic is not represented.

For every descriptor chunk, topology tests decode both generated value and
row-index addresses and require the decoded channel/rank/BG to match the
descriptor's global BG. Version 1 retains the Milestone 1 bank policy:
values use bank 0/base row 0, row indices use bank 1/base row 4096, packed x
uses bank 2/base row 8192, and analytical results use bank 3/base row 12288.
