# CSC M7 FP16 representation and timing contract draft

## FP32 versus target FP16

| Property | FP32 M7 now | FP16 target |
|---|---:|---:|
| burst/alignment | 32 B | 32 B |
| matrix value bytes | 4 | 2 |
| values per burst | 8 | 16 |
| row-index bytes | 4 | 4 |
| indices per burst | 8 | 8 |
| maximum compute chunk | 8 NZE | 16 NZE |
| x elements per burst | 8 | 16 |
| descriptor bytes | 32 | 32 |

The key M3 change is asymmetric operand fetch. For a 16-NZE chunk:

```text
value requests = 1
index requests = ceil(chunk_nnz/8) = 1 or 2
```

| chunk NNZ | value bursts | index bursts | valid index halves |
|---:|---:|---:|---|
| 1-8 | 1 | 1 | low only |
| 9-16 | 1 | 2 | low and high |

## Required descriptor/request state

The current single `chunk_offset_` and one `index_slot_` are insufficient.
M3 needs independent logical value and row-index offsets plus readiness for
two index bursts:

```text
remaining_nnz
chunk_nnz=min(remaining_nnz,16)
value_offset
row_index_offset
required_index_bursts=ceil(chunk_nnz/8)
value_ready
index_low_ready
index_high_ready
valid_count
valid_mask[16]
```

Request identity must distinguish both index completions. The engine may issue
the value and required index requests concurrently under the existing
overlapped policy, but compute cannot trigger until every required operand is
ready. Row-index lane order is low burst lanes 0-7 followed by high burst lanes
8-15. Tail lanes must never be decoded, multiplied, or enqueued.

## Producer-rate impact

The current descriptor targets only `kBGTargetFirstPIMBlock` for each masked
MUL, so production M7 does not presently emit 16 FP32 partials per BG cycle.
With target FP16, one selected PIM block can materialize up to 16 partials per
operation. If a later policy uses both PIM blocks in a BG concurrently, the
peak becomes 32 partials/BG event. That dual-block rate is a future policy, not
a current baseline fact.

Keeping BGA input queue depth, comparison width, add latency, and output queue
entry count fixed can increase compute-to-BGA backpressure. Timing must reflect
that pressure; FP16 does not imply a free BGA throughput increase.

## Partial transport choice

Current transport is an 8-byte record:

```text
uint32 row_idx
float value
```

The logical FP16 payload is six bytes, but two physical contracts remain:

- 6-byte packed: five records per 32-B burst with two padding bytes, but
  alignment/serialization and crossing rules become more complex;
- 8-byte aligned: `uint32 row_idx + uint16 value_bits + uint16 reserved`, four
  records per burst, preserving the current packer geometry.

The roadmap currently prefers 8-byte aligned BG-to-GA records. This means
precision alone does not reduce partial-result transport bytes versus current
M7. The choice must be fixed in M6 and versioned; M0 makes no format change.

## Final-y representation

The target dense y uses 2-byte FP16 elements and packs 16 rows per 32-B burst.
Current M7 has no modeled dense-y writeback. M7 must define staging of 16
elements, zero-row initialization, partial-tail behavior, whether a burst is
written once or updated, and overlap with GA before exact cycles can be frozen.

## Timing principles

- Retain current MUL/ADD latency unless a separately justified hardware model
  changes it.
- Do not halve ALU cycles solely because precision is FP16.
- Cycle gains may arise from fewer value chunks, denser x packing, and dense-y
  byte reduction.
- Row-index and descriptor traffic do not shrink.
- The second index request for 9-16 NZE chunks must consume request/bus
  resources and participate in readiness/backpressure.
- Column-local 32-B alignment means short columns may see little or no physical
  value-byte reduction.
- BGA/GA producer-consumer pressure and transport arbitration remain explicit.

## Draft FP16 traffic invariants

```text
value bursts     = sum_columns ceil(nnz_column/16)
row-index bursts = sum_columns ceil(nnz_column/8)
logical MUL ops  = sum_columns ceil(nnz_column/16)
active lanes = generated partials = BGA enqueues = matrix NNZ
x scalar loads = descriptor count
```

Useful logical bytes are `2*NNZ` for matrix values and `4*NNZ` for row indices.
Physical bytes include independent per-column 32-B alignment. Descriptor bytes
remain `32*descriptor_count`. Transport and final-y byte invariants remain
open until M6/M7 formats are approved.

## Boundary cases required by M3

Column NNZ counts `0,1,7,8,9,15,16,17,31,32,33` must verify request counts,
lane association, independent offsets, tails, and absence of invalid accesses.
The M0 golden covers `0,1,7,8,9` under FP32 and freezes the pre-change behavior.

No production lane, request, record, or latency setting was changed in this
draft stage.

