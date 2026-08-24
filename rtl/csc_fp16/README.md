# CSC FP16 Batch8/Q64 standalone RTL

This package is a synthesis-oriented model of CSC-specific hardware added per
HBM bank group. It is not a conversion of the full C++ simulator and does not
claim iso-area equivalence with SparsePIM.

## Primary preset

`CSC_FP16_BATCH8_Q64_DEFAULT` is the default and primary synthesis target:

| Parameter | Default |
|---|---:|
| Precision / physical SIMD | FP16, 16 lanes / 256 bits |
| `BATCH_WIDTH` | 8 |
| `INPUT_DEPTH` | 64 |
| `ACC_ENTRIES` | 64 |
| `COMPARE_W` | 64 |
| `OUTPUT_DEPTH` | 64 |
| `ROW_W` / `VALUE_W` | 32 / 16 |
| BGA replication | one/BG, 64 BG/stack |

The five queue/lookup parameters are independently overridable. Q32 and Q16
scripts are sensitivity scaffolding; Q64 remains the primary report order.

## Blocks

- `csc_descriptor_engine`: one active 32-byte descriptor's execution state.
- `csc_agu`: aligned X, VALUE, INDEX_LOW and INDEX_HIGH read requests.
- `csc_request_tracker`: fixed-depth typed scoreboard and completion match.
- `csc_partial_event_gen`: captures one 16-lane result/index vector and emits
  indexed partials.
- `csc_batch_ingress`: atomic Batch8 staging followed by entry service.
- `csc_bga`: Q64 associative tags, external reused-adder interface, bounded
  oldest rank, input and output queues.
- `csc_result_serializer`: `{reserved=0, fp16, row32}` 64-bit record.
- `csc_burst_packer`: four records into one 256-bit/32-byte burst, including
  zero-padded tails.
- `csc_result_transport_ctrl`: pending writes and bounded write/read tags.
- `csc_bg_engine`: one-BG structural integration.
- `csc_added_hw_top`: synthesis top excluding baseline arithmetic and memory.

## Replacement metadata

The C++ model's unbounded 64-bit age is not copied. Each Q64 entry has a 6-bit
bounded insertion rank. Hits do not change rank. A full miss evicts the largest
rank (oldest), decrements the other nonzero ranks, and assigns the new entry the
youngest rank. The smallest rank is selected as the victim. This preserves insertion-oldest semantics without simulator age
or contribution/debug fields.

## Arithmetic and memory boundaries

`pim_mul_*` and `bga_add_*` are external interfaces. The default interpretation
is reuse of the existing 256-bit PIM multiplier and existing FP16 adder, so their
area is excluded. The result memory is also external. RTL emits logical linear
burst addresses only; no physical result bank/row mapping is asserted.

## Functional verification

Verilator tests cover zero/short/full/tail descriptors, one/eight-entry atomic
batches, Q64 occupancy, hit merge, full-miss eviction, output backpressure,
final drain, burst tails/full bursts and an integration smoke test. Build output
should be directed to `/tmp` so the repository remains clean except for this
package.

## Q64 logical raw inventory per BG

These are logical state bits, not cell area. Control and implementation overhead
is shown separately where useful.

| State | Calculation | Bits/BG | Bits/64 BG |
|---|---:|---:|---:|
| Accumulator entries | 64 × (valid 1 + tag 32 + value 16 + rank 6) | 3,520 | 225,280 |
| Input queue payload | 64 × (row 32 + value 16) | 3,072 | 196,608 |
| Output queue payload | 64 × (row 32 + value 16) | 3,072 | 196,608 |
| Batch assembler | 8 × 48 + count/pending 5 | 389 | 24,896 |
| Batch ingress staging | 8 × 48 + count/index/active 9 | 393 | 25,152 |
| MUL result/index event staging | 256 + 512 + mask 16 + lane/active 5 | 789 | 50,496 |
| Descriptor execution registers | active fields and FSM flags | 258 | 16,512 |
| Request tracker | 16 × (valid 1 + tag 8 + type 2) + masks/count | 185 | 11,840 |
| Burst packer | 256 + count 3 + pending 1 | 260 | 16,640 |
| Pending write payload | 4 × (256 + valid-record count 3) | 1,036 | 66,304 |
| Write/read tag scoreboards | 2 × 2 × (valid 1 + tag 8) | 36 | 2,304 |
| Held read response | 256 + valid | 257 | 16,448 |

Queue pointer/count bits, AGU/control state and transport counters add small
amounts beyond payload rows above. The package contains exactly **64 parallel
32-bit equality comparisons** in the Q64 BGA lookup. Equality gates, reduction,
priority encoding, hit/free/victim selection and muxes are logic costs and must
not be approximated as storage bits.

The three dominant explicit BGA payload arrays are 9,664 bits/BG and 618,496
bits under raw 64× replication. Including the listed rank, staging, tracker,
packer and transport payload/control fields increases this inventory, but it is
still not a physical area estimate.

## Feasibility risks

1. Q64 64×32-bit tag comparison and hit reduction at a 1 ns target.
2. Hit/free/victim priority/select and data muxing.
3. Q64 accumulator/input/output storage replicated across 64 BGs.
4. Typed request scoreboard and its integration with the memory controller.
5. Indexed result buffering and output-memory mapping not yet characterized.

## Tool and result status

The four standalone Verilator tests and generic Q64 Yosys synthesis are run by
GitHub Actions. A later audited host run used Yosys 0.68+ and the recorded
NanGate 15 nm typical NLDM Liberty to produce the characterized cell-area
summary under `results/nangate15/area_estimation_20260819_153139/`.

Only curated summaries, small `stat` reports, runtimes, and provenance hashes
are versioned. Generated mapped netlists and synthesis logs are omitted and can
be regenerated with `scripts/run_synth.sh` or `scripts/run_module_synth.sh`.
The Liberty files are not redistributed.

The characterized result is an FF-based Liberty-area estimate. Critical path,
slack at 1 ns, Fmax, placed-and-routed area, and iso-area equivalence remain
unavailable because reviewed SDC, LEF/macro views, and a physical flow were not
completed. See `synth/README.md` for the entry points and `HARDWARE_MAPPING.md`
for the simulator-to-RTL boundary.
