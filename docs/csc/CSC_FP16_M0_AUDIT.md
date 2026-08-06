# CSC M7 FP16 M0 audit

## 1. Audited state

The audit was performed in `/home/gs13022/SparsePIM/.pimsimulator-m7-fp16`
on branch `M7-fp16` at FP32 parent commit
`8fcd093925451b3d09da9529a9f0aa9c91f13d04`. The worktree was clean and had
no FP16 CSC production changes before M0 began. The sibling FP32 worktree was
not modified or used for build artifacts.

Build flags are defined by `Sconstruct::build_flags`: `g++`, C++17, `-O2`,
warnings enabled, no `-ffast-math`, and no explicit F16C/ISA option. The
configuration relevant to the golden fixture is recorded in
`results/csc/fp32_m7_golden/m7_fp32_golden_v1.txt`.

Two requested references were absent from this worktree:
`HBM_PIM_CSC_SpMV_Milestone_1_to_7_Roadmap(1).txt` and
`PIMSIMULATOR_CSC_AUDIT.md`. They were not read from another worktree. The
available `README.md`, `docs/csc/CSC_IMAGE_FORMAT.md`, and source were checked.

## 2. Current FP32 architectural dataflow

| Stage | Source/symbol | Input -> output | Width/layout | Counter/test | FP16 impact |
|---|---|---|---|---|---|
| Host CSC | `CSCMatrixLoader.h::CSCMatrix`, `makeCSC()` | `double`/text input is parsed or test `float` CSC is constructed | `values: vector<float>`, `row_idx: uint32` | layout tests | values must become canonical FP16 bits |
| Physical materializer | `tools/csc_light_preprocess/csc_light_preprocess.cpp::materialize_aligned()` | source values -> aligned BG streams | FP32 value 4 B, uint32 index 4 B, 8 each/32 B | manifest counters | value stream becomes 2 B; index stays 4 B |
| Image v1 loader | `CSCExternalImage.cpp::loadExternalPhysicalImage()` | files -> `CSCLayout` | requires version 1/FP32; reads values/index in 4-B steps | external image tests | separate v2 loader required later |
| Layout/view | `CSCLayout.h::BGImage`, `CSCBGImageView` | byte streams/descriptors/x | values/index are byte vectors; packed x is `vector<float>` | `validateLayout()` | packed x and value decode change |
| Descriptor execution | `CSCDescriptorEngine::tick()` | descriptor -> x/value/index requests | `kCSCSIMDWidth=8`; one 32-B value and one 32-B index slot/chunk | `CSCEngineCounters` | 16-value/8-index asymmetric tracker required |
| Operand staging | `CSCDescriptorEngine::stageCompletion()` | 32-B burst -> staging arrays | `array<uint8_t,32>` for value/index; `float x_j_` | transaction counters | value decode/lane association change |
| Masked multiply | `CSCDescriptorEngine::tick(SIMD_MUL)`, `PIMBlock::mul()` | FP32 value and scalar -> FP32 product | 8 `BurstType::fp32Data_` lanes | masked SIMD tests | use 16 `fp16Data_` lanes later |
| Partial emission | `CSCDescriptorEngine::tick(EMIT_PARTIALS)` | row index + result lane -> batch | `CSCBGAPartial{uint32,float}` | generated/accepted partial counters | value becomes FP16 |
| BG-local BGA | `CSCBankGroupAccumulator` | indexed partial -> merge/insert/output | tags uint32; queue/entry/output values `float` | BGA counters/tests | all stored/add/output values become FP16 |
| Partial packing | `CSCPartialResultPath::commitReserved/formBurst` | BGA output -> BG-local bursts | `{uint32 row,float value}` = 8 B; 4/32 B | writeback counters/tests | 6-B packed vs 8-B aligned is a later decision |
| Readback | `CSCPartialResultPath::issueReads/completeReads` | resident partial bursts -> host return queue | same 8-B records and 32-B bursts | readback counters/tests | transport value semantics change |
| Cross-BG reduction | `CSCPartialResultPath::completeReductionBatch()` | returned indexed records -> final y | ordered `vector<float> final_y_fp32_` ADD | host reduction counters/tests | current implementation is host FP32, not logic-die GA |
| Final result | `CSCNativeExecution::finalResult()` | host-reduced vector -> validator | dense FP32 vector in simulator state; no modeled dense-y writeback | M7B end-to-end tests | logic-die GA and dense FP16 y path are new later milestones |

### Roadmap/source discrepancy

The current source has no logic-die global accumulator and no architectural
dense final-y writeback/readback. The implemented path ends with indexed FP32
host reduction after partial-result readback. Therefore a current `GA output
trace`, `GA receive/merge count`, or dense-y traffic cannot be frozen without
inventing behavior. The minimal later instrumentation is a real GA component
with arrival/merge/drain counters and a modeled dense result path. M0 records
these items as unavailable rather than mapping host reduction to hardware GA.

## 3. FP32 width and coupling audit

### A. FP32 SIMD-width dependencies requiring change

- `CSCTypes.h::kCSCSIMDWidth = 8` and `CSCLayout.h::kSIMDWidth = 8`.
- `CSCDescriptorEngine::descriptorValid()` uses `(nnz+7)/8*32` for both
  streams.
- `FETCH_DESCRIPTOR` creates exactly one value slot and one index slot.
- `SIMD_MUL` uses `min(remaining_nnz,8)`, loops over 8 FP32 lanes, copies
  `valid_count*4`, and counts 8 available lanes.
- `ADVANCE_CHUNK` increments a common value/index `chunk_offset_` by 32.
- `CSCLayout.cpp::append/validateLayout/refreshLayoutStats` treats both value
  and index elements as four bytes.
- `CSCFunctionalModel` and aligned preprocessor use `(nnz+7)/8`, eight
  available lanes, and equal value/index transaction counts.
- packed x uses `current_.x_slot/8*32` and `vector<float>`.

These collectively enforce:

```text
one FP32 chunk = one 32-B value request = one 32-B index request = <=8 NZE
```

They prevent the target FP16 relation of one 16-value request plus one or two
8-index requests.

### B. HBM topology dependencies that remain unchanged

- Literal 8 in `PIMBlock::grfA[8]/grfB[8]` is the architectural GRF count,
  not FP32 vector lanes.
- Eight PIM blocks per rank and mappings such as `channel*8+block` are HBM PIM
  topology.
- The two banks per PIM block relation and Scheme8 address mapping are not
  precision widths.

### C. BGA/transport microarchitecture dependencies

- BGA accumulator/compare/output queue depths such as 4, 16, or 64 are entry
  capacities, not SIMD width.
- `kCSCPartialResultRecordBytes=8` is FP32 transport layout: 4-B row plus 4-B
  value.
- `kCSCPartialRecordsPerBurst=4` follows the 8-B record, not compute lanes.

### D. Descriptor/layout dependencies

- `CSCDescriptor` is 32 bytes and its integer fields remain unchanged.
- All per-column streams start at a 32-B boundary.
- Value and row-index offsets currently advance together only because both
  element types are four bytes.

### E. Unrelated constants

Literal 8 also occurs in hashes, addresses, byte serialization, test data,
and unrelated workloads. Those occurrences must not be mechanically replaced.

## 4. Request and execution order

`CSCDescriptorEngine::tick()` currently performs `FETCH_DESCRIPTOR -> LOAD_X
-> FETCH_VALUE -> FETCH_ROW_INDEX -> WAIT_OPERANDS -> SIMD_MUL ->
WAIT_TARGET_* -> EMIT_PARTIALS -> ADVANCE_CHUNK`. The three operand slots are
x/value/index. Each later chunk clears and recreates only one value and one
index slot at the same `+32` offset.

The BGA cycle order in `CSCBankGroupAccumulator::step()` is:

1. retire an accepted output;
2. commit a completed lookup/merge/insert operation;
3. handle capacity eviction or final drain;
4. start one round-robin input lookup;
5. make newly accepted batches visible to the input FIFO.

A lookup scans all non-reserved tags. A hit schedules a floating-point merge;
a free-slot miss schedules insertion; a full miss selects the oldest entry
with slot-index tie-break, emits a capacity-eviction output, waits for its
retirement, then inserts. Final drain also emits the oldest non-reserved entry.
The FP32 merge explicitly materializes `volatile float`.

## 5. Current transport and final-y contract

`CSCPartialResultRecord` is exactly 8 bytes (`uint32_t row_idx; float value`).
Four records form a 32-B burst. Simulator-only BG/generation/sequence/reason
metadata lives in the envelope and is excluded from transferred byte counts.
Full and tail bursts are written to modeled resident buffers, read back in a
deterministic topology order, then expanded into an ordered host reduction
queue. `completeReductionBatch()` performs one FP32 add per physical BGA output
record into `final_y_fp32_`.

This final vector is authoritative for current M7 correctness, but it is not a
modeled PIM dense-y memory image. The roadmap's FP16 GA and dense-y path must be
implemented as new architectural stages rather than by renaming this host path.

## 6. Existing FP32 invariants

For version 1 and the current native engine:

```text
descriptor NNZ sum = matrix NNZ
active lanes = generated partials = accepted BGA partials = matrix NNZ
value transactions = index transactions = logical MUL events
x scalar loads = descriptor count
BGA merges + physical BGA output contributions = matrix NNZ
physical BGA records = packed records = readback records = reduced records
write useful/transferred/padding bytes = corresponding readback bytes
```

The equality of value and index transactions is intentionally retired in M3.

## 7. Golden regression and artifacts

`CSCFP32GoldenBaselineTest.M7ArchitecturalBehaviorAndAccounting` uses columns
with 0/1/7/8/9 NNZ, duplicate rows, same-BG and cross-BG equal rows, SIMD
tails, BGA merge/eviction/final drain, and zero output rows. It fixes raw FP32
final-y bits, request/chunk/partial counts, BGA accounting, transport bytes,
phase cycles, and the ordered BGA event-stream FNV-1a hash. The deterministic values are mirrored in
`results/csc/fp32_m7_golden/m7_fp32_golden_v1.txt`.

The earlier representative cant transport result remains documented in
`M7_FP32_BASELINE_2026-08-06.md`, but depends on a `/tmp` image and is optional,
not a mandatory regression. The existing external toy test is likewise
optional through `CSC_EXTERNAL_IMAGE`; no personal absolute path was added.

## 8. Planned change sites

| Milestone | Files/symbols |
|---|---|
| M1 | `FP16.h`, `lib/half.h` semantics consumer; new shared CSC arithmetic only after contract approval |
| M2 | `tools/csc_light_preprocess/csc_light_preprocess.cpp`, `CSCExternalImage.*`, `CSC_IMAGE_FORMAT.md` |
| M3 | `CSCTypes.h`, `CSCDescriptorEngine.*`, request slots/offsets/counters, functional reference |
| M4 | `PIMBlock`, `BGTargetedOperation`, descriptor result staging and partial emission |
| M5 | `CSCBankGroupAccumulator.*`, BGA tests and trace reference |
| M6+ | new logic-die GA/transport plus `CSCPartialResultPath` boundary |
| M7+ | new dense FP16 y storage/writeback/readback and end-to-end result API |

## 9. Facts, inferences, and open issues

| Status | Item |
|---|---|
| Confirmed | Current CSC production datapath is FP32 and width 8. |
| Confirmed | Original PIMBlock already has a 16-lane FP16 path using `half_float::half`. |
| Confirmed | Current cross-BG operation is host FP32 reduction, not hardware GA. |
| Confirmed | Current image v1 loader rejects anything other than FP32/version 1. |
| Inference | Reusing the existing half library is the lowest-risk M1 arithmetic basis. |
| [OPEN] | Whether SparsePIM uses separate MUL+ADD everywhere or invokes MAC/MAD where product rounding differs. |
| [OPEN] | Whether logic-die GA is bounded associative storage, sorted-stream reduction, or another policy. |
| [OPEN] | FP16 BGA-to-GA record remains 6-byte packed versus 8-byte aligned. |
| [OPEN] | Whether final-y writeback overlaps GA and how partially filled dense bursts are staged. |

Before M1 production implementation, approve separate FP16 MUL then FP16 ADD
as the architectural contract, confirm the existing half library as the shared
implementation basis, and keep subnormal preservation/RNE independent of host
compiler floating-point environment.

## 10. M0 validation record

Commands executed in this worktree on 2026-08-06:

```bash
scons -c
scons -j4
./sim --gtest_filter=CSC*
./sim --gtest_filter=FP16SemanticsCharacterizationTest.*:PIMKernelFixture.mul:CSRDirectSpMVPhysicalMappingTest.Scheme8RepresentativeRows
```

Results:

- Clean build: PASS.
- CSC suites: 222 tests executed, 208 PASS, 14 SKIP, 0 FAIL. Every skip
  required an opt-in benchmark flag or external image/matrix environment
  variable.
- FP16 characterization, legacy PIM MUL, and Scheme8: 8/8 PASS.
- Golden fixture: compute/BGA/write/read/reduction/end cycles
  `178/206/209/242/261/262`; BGA trace 23 records with FNV-1a
  `f3459a33998d8acd`.

No external CSV benchmark was run and no existing result file was overwritten.
