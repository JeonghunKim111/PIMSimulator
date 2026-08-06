# FP16 M5.1 batch8 ingress and production configuration

## Purpose and configuration boundary

Binary16 requires 16 compute lanes but does not require a 16-entry BGA port.
M5.1 therefore retains the serial event path and adds the FP32-equivalent
eight-entry atomic batch boundary. Internal lookup/ADD remains one entry.

Named presets are the only architectural configuration source:

| Preset | Ingress | Accumulator | Role |
| --- | ---: | ---: | --- |
| `FP16_SERIAL_EVENT_Q16` | 1 event | 16 | compatibility/debug |
| `FP16_BATCH8_Q8_STRESS` | atomic 8 | 8 | eviction/backpressure stress |
| `FP16_ISO_STRUCTURE_BATCH8_Q16` | atomic 8 | 16 | M6/M7 production default |

Factories are `makeFp16SerialCompatibilityConfig`,
`makeFp16Q8StressConfig` and `makeFp16IsoStructureProductionConfig`.
Batch16 and iso-area larger queues are not implemented.

## FP32 batch audit

| Property | FP32 | FP16 batch8 | Same |
| --- | --- | --- | --- |
| Maximum payload | one 8-lane chunk | eight events | yes |
| Validity | vector length/valid count | `valid_count` 1..8 | yes |
| Acceptance | reserves all valid-entry FIFO capacity | same | yes |
| Partial accept | impossible | impossible | yes |
| Retry | stable batch identity and payload | stable adapter batch | yes |
| Visibility | `step()` final phase | `step()` final phase | yes |
| Entry order | vector/lane order | ascending lane order | yes |
| Service | one FIFO entry | one FIFO entry | yes |
| Producer advance | only after atomic accept | adapter blocks next batch | yes |

FP32 `offerBatch` checks `queue.size + partials.size <= input_queue_depth`.
An accepted batch is held in `accepted_pending`; `commitAcceptedBatches` appends
all entries at step end. No prefix can be accepted.

## FP16 batch representation and mapping

`CSCFp16PartialBatch` contains eight event slots, valid count, BG, descriptor,
chunk and batch ID. The descriptor event includes `chunk_valid_count` as
adapter sideband; it is deliberately excluded from the legacy partial hash.

```text
valid 1..8:   batch0 lanes 0..valid-1
valid 9..16:  batch0 lanes 0..7, batch1 lanes 8..valid-1
```

The adapter owns exactly one bounded batch. It accumulates events in ascending
lane order, then stops accepting producer events until the entire batch enters
the BGA. Batch1 cannot be staged before batch0 is accepted, and the next chunk
cannot overwrite batch1. Invalid lanes never enter a batch.

## Backpressure and accounting

Batch acceptance reserves `valid_count` FIFO entries atomically. A failed
attempt retains the same batch object and is retried. Output blockage may fill
the BGA output/input queues and then stalls the completed adapter batch and
descriptor engine.

The boundary fixture has 169 partials and exactly:

```text
generated batches = sum(column ceil(nnz/8)) = 25
batch0 = compute chunks = 15
batch1 = chunks with more than eight lanes = 10
sum(valid_count) = accepted partials = 169
```

Production counters distinguish generated, attempted, accepted and stalled
batches; accepted batch0/batch1; generated/accepted partials; BGA input FIFO
high-water and serviced entries.

## Golden separation

### Serial Q16 compatibility

Serial and production batch8 Q16 produce identical output events and hash.
Timing is recorded independently and is not required to match.

### Batch8 Q8 stress

```text
compute+BGA cycles       556
accepted batches          25
accepted partials        169
merges                     2
capacity evictions       151
final drains              16
outputs                   167
hash        935f4049c56e632f
```

This is a stress result, not the production architecture.

### Batch8 Q16 production

```text
compute complete          544
drain start               544
compute+BGA complete      564
last output accepted      563
accepted batches           25
batch0 / batch1         15 / 10
accepted partials         169
queue high-water           16
merges / FP16 ADDs       4 / 4
capacity evictions        133
final drains               32
outputs                    165
ingress global/engine stall 0 / 0
output sink stalls           0
hash        96c3f823f3fe5ab1
```

The normal fixture is operand-memory limited and does not exert ingress
pressure. Forced output blockage separately verifies batch/producer stall.

## M6 transport and reduction contract

M6 production must select `FP16_ISO_STRUCTURE_BATCH8_Q16`.

The transport record is eight bytes: `uint32 row_idx`, `uint16 value_bits`,
`uint16 reserved`. Reserved is always zero and nonzero is an error. Four
records occupy a 32-byte burst. Tail bytes are zero; explicit valid-record count
is authoritative. Therefore FP16 partial transport remains 8 bytes/record and
no transport-bandwidth reduction is claimed. Six-byte packing is future work.

FP32 accepts a BGA output only after the partial path reserves durable packer
capacity. Full bursts form and may write during compute; final tails form only
after that BG final-drain completion. M6 preserves this overlap and adds none.

FP32 storage is a logical per-BG resident burst region indexed by bounded buffer
slot and monotonically increasing burst sequence; regions never share slots.
The current timing model has no physical byte base address or record-count
metadata address. M6 must preserve per-BG isolation/alignment and cannot invent
a physical mapping without a separately reviewed address contract.

Readback starts only after all BG writeback lifecycles, pending/inflight writes
and tails complete. Reads issue round-robin within each channel while preserving
per-BG burst sequence. Same-cycle completions sort by completion cycle, channel,
rank, local BG and burst sequence. Host reduction consumes host-return FIFO
order, then record-slot order. FP16 reduction must initialize every y row to
`0x0000` and apply `cscFp16Add` after every record; raw uint16 bits are the
architectural final representation.

## Not implemented

M5.1 contains no transport record serializer, physical writeback address,
writeback/readback, host FP16 reduction, cross-BG merge or final-y path.
