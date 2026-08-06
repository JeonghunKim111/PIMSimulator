# CSC M7 FP16 representation and timing contract

## Current and target paths

Current FP32 M7 ends with indexed partial transport and ordered host reduction:

```text
FP32 image -> 8-lane MUL -> FP32 BGA
-> 8-byte indexed partial writeback/readback
-> ordered host FP32 reduction -> final_y_fp32
```

The FP16 target preserves that architecture. There is no current logic-die GA,
GA arbitration, GA accumulator, or dense PIM-side final-y writeback.

## FP32 versus FP16 representation

| Property | FP32 v1 | FP16 v2 / future execution |
|---|---:|---:|
| burst and column alignment | 32 B | 32 B |
| matrix value bytes | 4 | 2 |
| values per burst | 8 | 16 |
| row-index bytes | 4 | 4 |
| indices per burst | 8 | 8 |
| maximum compute chunk | 8 NZE | 16 NZE |
| x elements per burst | 8 | 16 |
| descriptor bytes | 32 | 32 |

M2 implements only the v2 representation. Production execution remains v1.

## Implemented FP16 image v2

Each nonempty column has independently 32-byte-aligned value and row-index
allocations:

```text
value allocation = align32(2 * nnz_column)
index allocation = align32(4 * nnz_column)
```

Per BG files are `bg_NN_values_fp16.bin`, `bg_NN_row_idx_u32.bin`,
`bg_NN_descriptors.bin`, and `bg_NN_x_permutation.bin`. `x_fp16.bin` stores the
complete original x in FP16 and is padded once to 32 bytes. The permutation
stream preserves the existing descriptor `x_slot -> original_col` contract.
All integer and FP16-bit fields are explicitly little-endian.

The descriptor remains the existing 32-byte integer layout. M2 changes no
descriptor field or production parser. Logical, physical, and padding bytes
for values, indices, and x are separate manifest counters. This exposes the
case where short columns consume the same 32-byte physical allocation in FP32
and FP16.

## M3 request asymmetry

A future 16-NZE chunk requires one value request and one or two index requests:

| chunk NNZ | value bursts | index bursts |
|---:|---:|---:|
| 1-8 | 1 | 1 |
| 9-16 | 1 | 2 |

The future tracker needs independent value/index offsets, two index-completion
states, a 16-bit valid mask, and `chunk_nnz=min(remaining_nnz,16)`. None of
these production changes are part of M1/M2.

## Producer and BGA policy

M3/M4 will preserve the current PIM-block scheduling policy initially. One
selected block can produce up to 16 FP16 partials per operation; a later
two-block concurrent policy would raise the peak to 32 and is a separate
decision. BGA capacity remains iso-entry-count with FP32. Increasing producer
width must create modeled backpressure rather than an implicit BGA throughput
increase.

## Indexed partial transport decision

The future FP16 record is fixed at eight bytes:

```text
offset 0: uint32 row_idx
offset 4: uint16 fp16_value_bits
offset 6: uint16 reserved (zero)
```

Four records occupy one 32-byte burst, preserving current packing geometry.
A six-byte packed record is not part of the base implementation and may only
be considered as a later sensitivity study. M1/M2 do not change production
partial serialization or ordered host reduction.

## Timing and traffic principles

- FP16 does not by itself reduce MUL/ADD latency.
- Gains may come from fewer value chunks and denser x storage.
- Row-index and descriptor bytes do not shrink.
- The second index request for 9-16 NZE participates in bus use and readiness.
- Per-column 32-byte alignment can hide logical value-byte savings.
- Wider partial production and BGA backpressure must be modeled explicitly.

The post-M3 invariants remain drafts:

```text
value bursts     = sum_columns ceil(nnz_column/16)
row-index bursts = sum_columns ceil(nnz_column/8)
logical MUL ops  = sum_columns ceil(nnz_column/16)
active lanes = generated partials = BGA enqueues = matrix NNZ
x scalar loads = descriptor count
```

The M2 round-trip fixture already covers column sizes
`0,1,7,8,9,15,16,17,31,32,33`, but no production request count or cycle is
changed until M3.
