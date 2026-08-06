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

M2 implements the v2 representation. M3/M4 adds the explicit
`CSCFp16ExecutionMode::FP16_IMAGE_V2` path through FP16 partial generation;
the existing FP32 engine remains v1-only and unchanged.

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

## M3 production gate and request asymmetry

`CSCFp16ExecutionImage::load()` is the only construction path for an FP16
execution image. It invokes the complete v2 loader validation (manifest type
and widths, checksums, sizes, alignment, descriptor ranges, ownership, x slots
and zero padding) before an engine can retain the image. The FP16 engine also
requires a `PIMBlock` whose configured precision is FP16. A v1 image or FP32
datapath fails before launch.

A future 16-NZE chunk requires one value request and one or two index requests:

| chunk NNZ | value bursts | index bursts |
|---:|---:|---:|
| 1-8 | 1 | 1 |
| 9-16 | 1 | 2 |

The implemented engine keeps independent `value_stream_offset_` and
`index_stream_offset_`. Each chunk owns four identity-bearing request slots:
`X`, `VALUE`, `INDEX_LOW`, and optional `INDEX_HIGH`. Completion compares the
full request identity (request id, BG, descriptor, chunk, kind and offset), so
arrival order is independent of issue order and stale/duplicate completion is
rejected.

The current FP32 policy creates one active chunk and issues at most one request
per tick in X/value/index order. FP16 preserves that policy and adds high index
as the fourth request. It introduces no next-chunk prefetch or extra request
issue bandwidth. Compute waits for all required slots.

The offsets advance after a chunk by 32 value bytes and by 32 or 64 index
bytes. A 17-NNZ descriptor therefore requests value offsets `{0,32}` and index
offsets `{0,32,64}` relative to its independent bases.

The global v2 x file is addressed with
`(original_col/16)*32`, selecting lane `original_col%16`. Descriptor `x_slot`
remains the BG-local permutation slot verified against `original_col`; it is
not incorrectly reused as a global x element number.

## Producer and BGA policy

M3/M4 preserves a single injected PIMBlock per engine, matching the current
one-hot first-PIM-block policy. `PIMBlock::mul(..., valid_count)` executes the
native 16-lane FP16 path and materializes binary16 products. A bounded capture
sink replaces BGA only at the milestone boundary. It accepts one valid-lane
event at a time and backpressures the engine without drop, duplication, or
recomputation. It is not an FP16 BGA timing model.

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

The production boundary fixture covers column sizes
`0,1,7,8,9,15,16,17,31,32,33`, x columns 15/16, multiple BGs, low/high and
cross-chunk duplicate rows, tails, signed zero, subnormal, maximum finite,
overflow and NaN. It produces 169 bit-exact partial events from 15 value and
25 index requests. No FP16 BGA, serialized partial transport, or host FP16
reduction exists yet.

## M4.5 native DRAM timing integration

`CSCFp16NativeExecution` now connects the typed M3 request interface to actual
`MultiChannelMemorySystem::addTransaction` acceptance, Scheme8 arbitration and
token callbacks. It preserves tick-at-most-one request attempt per engine and
the X, VALUE, INDEX_LOW, INDEX_HIGH order. Rejected requests retain identity and
are not entered into the accepted outstanding table.

DRAMSim completion timing gates image-backed 32-byte payload delivery. The
outstanding entry, not the address alone, restores kind, BG, descriptor, chunk
and offset. Native tick order follows FP32: memory update/callback, global cycle
increment, then descriptor-engine tick. Synthetic and native raw partial traces
must be identical before native timing is valid.

The M4.5 compute-only scope begins at launch and ends only after all descriptor
engines are DONE, all accepted reads have completed and all generated events
have passed the bounded capture handshake. It includes actual operand request
queueing, response latency, operand wait and capture stalls. It still excludes
descriptor DRAM fetch, FP16 BGA, partial transport, readback and host reduction.

## M5 native FP16 BGA

M5 can replace the compute capture port with one iso-entry-count FP16 BGA per
global BG. The M4 ordered one-event ready/valid interface is retained and is
not widened to 16. Accepted events become FIFO-visible at the following BGA
step. Lookup scans all valid non-reserved tags, hits immediately materialize a
`cscFp16Add`, and full misses retire the oldest insertion before replacement.
Descriptor boundaries do not flush; producer completion triggers final drain.

Compute complete and compute+BGA complete are distinct global milestones.
Operand wait, ingress backpressure and BGA busy counters summed over BGs are
aggregate engine/BGA cycles. BGA output remains a bounded logical event;
transport, writeback, readback and host FP16 reduction remain outside M5.

M5.1 selects atomic batch8/Q16 as the production structure while retaining
serial/Q16 compatibility and batch8/Q8 stress presets. A 16-lane chunk maps to
two ordered batches; internal service remains one entry. See
`CSC_FP16_M5_1_INGRESS_AND_PRODUCTION_CONFIG.md` for golden and M6 contracts.
