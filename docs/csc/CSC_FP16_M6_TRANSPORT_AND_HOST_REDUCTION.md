# FP16 M6 Transport and Ordered Host Reduction

## Scope

M6 extends `FP16_ISO_STRUCTURE_BATCH8_Q16` from BGA output through indexed
partial writeback, readback, ordered host FP16 reduction, and raw-bit
`final_y_fp16`. It adds no logic-die accumulator, dense PIM-side output,
six-byte packing, or external evaluation.

## FP32 transport audit

`CSCPartialResultPath` is the source of truth. FP32 uses a logical resident-slot
timing model, not byte-addressed payload DRAM: transactions have issue limits,
cycle latency, BG ownership, bounded slots and burst sequence, while payloads
remain in simulator-owned burst objects.

| Property | FP32 | FP16 M6 |
|---|---|---|
| Record | row u32 + FP32, 8 B | row u32 + FP16 bits + zero reserved, 8 B |
| Packing | four records/32 B, BG-local | same control and width |
| Full burst | overlaps compute | unchanged |
| Tail | after that BG final drain | unchanged |
| Storage | BG resident slot + sequence | unchanged abstraction |
| Read barrier | all-BG writeback completion | unchanged |
| Reduction | timed host stage | deterministic BG/burst/slot FP16 replay |

No FP16-only address allocator, queue widening, or transport bandwidth increase
was introduced.

## Record, packing, and ownership

`CSCFp16TransportRecord` is explicitly serialized little endian:

```text
bytes 0..3  uint32 row_idx
bytes 4..5  IEEE binary16 raw value bits
bytes 6..7  uint16 reserved = 0
```

Four records occupy a burst. Valid count is sideband metadata, never inferred
from row/value. Tail bytes are zero; nonzero reserved is rejected. Contribution
count, reason and debug sequence are not serialized.

An output is accepted only when the BG packer has durable capacity. Ownership
then advances exactly once through packer, pending burst, in-flight write,
resident burst, readback and reduction. Full bursts overlap BGA execution;
partial tails become eligible only after BG final drain.

Write issue preserves FP32 rank round-robin, issue width and latency. The M6
golden uses 64 resident slots/BG because its active BGs produce more than 16
bursts and the inherited global read barrier retains all bursts until writeback
completion. This is fixture capacity, not a bandwidth or issue-width change.

## Readback and ordered FP16 reduction

Readback starts after all tails and writes complete. Requests retain FP32
channel/BG round-robin timing, but response order never defines arithmetic.
Completed bursts are keyed by BG and sequence; reduction consumes:

```text
global_bg_id ascending -> burst_sequence ascending -> record slot ascending
```

`final_y_fp16` is `std::vector<uint16_t>` initialized to `0x0000`. Every record,
including the first for a row, executes `cscFp16Add(current, value)` and stores
the immediately rounded binary16 bits. This is the first cross-BG merge point;
no FP32/double accumulator is used.

## Conservation and golden

The boundary fixture satisfies:

```text
BGA outputs = serialized = read = reduced = 165
write bursts = read bursts = 42
write bytes = read bytes = 1344
```

Deterministic timing and hashes:

```text
M5 BGA completion       564
M6 end-to-end           655
full / tail bursts      41 / 1
writeback complete      569
readback complete       618
reduction complete      655
BGA trace hash          96c3f823f3fe5ab1
record trace hash       bcfec10599ee6444
resident burst hash     49383a2ab16bbf67
final_y_fp16 hash       64a5f1109a6f2bd6
```

M4.5 compute-only and M5.1 capture-only remain separate regressions.

## Limitations

M6 preserves the FP32 logical-slot abstraction and is not a new physical
address claim. External workloads, speedup, FP64 accuracy, energy and area are
M7 work.

M7 exposes this unchanged path as `END_TO_END_TIMED`; the untimed direct
`BGA_VALIDATION` reducer must produce identical raw final-y bits. See
`CSC_FP16_M7_EXECUTION_MODES_AND_EVALUATION.md`.
