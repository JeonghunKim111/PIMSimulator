# Column-aligned BG-partitioned CSC SpMV milestone 1

FP16 development documents:

- `CSC_FP16_NUMERICAL_CONTRACT_DRAFT.md`
- `CSC_FP16_REPRESENTATION_TIMING_CONTRACT_DRAFT.md`
- `CSC_IMAGE_FORMAT_V2_FP16.md`
- `CSC_FP16_M3_M4_DESCRIPTOR_COMPUTE.md`
- `CSC_FP16_M4_5_NATIVE_EXECUTION.md`
- `CSC_FP16_M5_BGA.md`
- `CSC_FP16_M5_1_INGRESS_AND_PRODUCTION_CONFIG.md`
- `CSC_FP16_M4_5_NATIVE_EXECUTION.md`

## 1. Project goal

This project validates a conventional-CSC-derived sparse SpMV layout and execution model before native BG-local hardware is added to PIMSimulator. It establishes reproducible layout invariants, descriptor/chunk semantics, FP32 correctness, host indexed accumulation, lockstep scheduling statistics, and DRAM traffic phase measurements.

## 2. Relationship to PIMSimulator

Milestones 1 and 2 retain the test-side functional/timing harness. Milestone 4 additionally provides a simulator-side, cycle-stepped BG-local descriptor engine that issues serialized ordinary DRAM reads and calls the Milestone 3 masked FP32 `PIMBlock` datapath. It is not yet a fully BG-decoupled, overlapping, request-ID-based engine, and it does not perform PIM-side indexed accumulation. Existing `src/tests/CSCPartialStreamSpMV.cpp` remains a baseline.

## 3. Current milestone

Milestone 1 supplies a loader, column-to-BG mapping interface, byte-addressable physical layout, ideal/local descriptors, per-SpMV x packing, a sideband `remaining_nnz/valid_count` worker, host accumulation, CPU reference, a test-side lockstep policy, phase-drained timing traffic, unit tests, and an opt-in non-overwriting CSV benchmark.

## 4. Confirmed architecture policies

- Simulator channel means independent HBM pseudo-channel: 16 channels, four BG/channel, 64 global BG.
- One rank/channel, four banks/BG, one logical worker/BG for this model.
- FP32, SIMD width 8, 32-byte value/index/x bursts.
- One original nonempty column has exactly one descriptor and one BG owner; no segmentation.
- Empty columns preserve matrix dimensions but create no work descriptor.
- `round_robin` and caller-supplied `external_mapping` are separate policies.
- A topology-sensitive timing test must run in its own process because configuration storage is process-global.

`system_hbm_csc_fp32.ini` is separate from the existing FP16 configurations. `CSCTimingModel` fails construction unless all required topology, transaction-size, and precision properties match.

## 5. Column-aligned BG-partitioned CSC layout

Conventional CSC semantics (`col_ptr`, `row_idx`, `values`) are preserved. Mapping assigns each original column to one global BG. Each BG owns independent byte-addressable values and row-index images. Before each nonempty column, each image cursor is rounded up to a 32-byte boundary. The column payload is copied, and its physical allocation is extended to the next 32-byte boundary.

Padding is physical byte space, not a dummy NZE: it is absent from `nnz_count`, never receives a logical row index, and is never multiplied or accumulated. Logical NNZ padding is exactly zero. For 10 FP32 entries the useful payload is 40 bytes, physical allocation is 64 bytes, and 24 bytes are alignment padding. Packed unaligned CSC and burst splitting are intentionally unsupported; column alignment avoids DRAMSim's offset truncation.

Required invariants are checked by `validateLayout()`: one descriptor per nonempty column, exactly one owner, no segmentation, aligned value/index offsets, descriptor NNZ sum equal to original NNZ, valid payload ranges, and one-to-one original-column/x-slot metadata.

## 6. Descriptor format

```cpp
struct CSCDescriptor {
    uint64_t value_offset_bytes;
    uint64_t row_idx_offset_bytes;
    uint32_t nnz_count;
    uint32_t x_slot;
    uint32_t original_col;
    uint32_t global_bg_id;
};
```

Offsets are BG-local, not absolute physical addresses, and are always multiples of 32. The timing model combines a BG-specific physical base row with `offset/32`. Descriptors are assumed preloaded in an **ideal/local control buffer**: descriptor fetch latency and traffic are excluded.

## 7. x packing policy

Matrix layout stores `x_slot -> original_col` per BG. Every SpMV packs runtime input as `x_packed[bg][slot] = x_original[original_col]`. Eight FP32 scalars occupy one logical 32-byte burst; the last burst is padded and unreferenced lanes are ignored. A descriptor loads its scalar once and reuses it for all column chunks. Native SRF broadcast is not used.

## 8. Functional model

`cpuReference()` performs conventional CSC SpMV. `executeLockstep()` consumes the exact same `CSCLayout`, packs x, and produces partial records `(row_idx, value*x_j)`. It initializes `remaining_nnz=nnz_count`, uses `valid_count=min(remaining_nnz,8)`, copies and multiplies only those lanes, then decrements remaining NNZ. Invalid lanes produce no multiply, result write, or accumulation record.

Partials are accumulated on the host as `y[row_idx] += partial`. CPU and descriptor paths use double host accumulators over FP32 products, then cast to FP32, so BG-interleaved accumulation order does not create false failures. This is not native indexed accumulation, BGA, GA, or an atomic PIM operation. Comparison uses
`abs(actual-reference) <= 1e-5 + 1e-5*abs(reference)`.

## 9. Timing model

`CSCTimingModel` submits ordinary aligned transactions in separate drained phases: matrix value/index preload writes, packed-x reads, value reads, row-index reads, and optional result writes. Every logical value/index chunk maps to one aligned 32-byte transaction. No native rank-wide CRF MUL is presented as BG-local work; `logical_csc_mul_events` is a trace/statistic only and no invented ALU cycle is added.

The scheduler is an explicit test-side model that emulates a **rank-wide lockstep descriptor/chunk policy**. Each round lets each active BG process at most one chunk; finished BGs advance descriptors and inactive BGs idle. Pseudo-channels have independent controllers in the timing system. This is not a persistent, overlapping, or BG-decoupled engine.

Because callbacks lack request IDs, phases drain before reuse, addresses are unique within a phase, and the model does not issue multiple same-address outstanding requests.

## 10. Included and excluded latency

- `matrix_preload_cycles`: value plus row-index writes; excluded from kernel metrics.
- `x_load_cycles`: runtime packed-x reads; included only in `kernel_with_x` and end-to-end.
- `value_read_cycles`, `row_index_read_cycles`, `result_cycles`: `kernel_resident_cycles`.
- `kernel_with_x_cycles = x_load + kernel_resident`.
- `end_to_end_simulated_cycles = matrix_preload + kernel_with_x`.
- `synchronization_drain_cycles`: currently zero as an extra charge; drain latency is already inside each phase.
- Layout/preprocessing, CPU reference, functional multiplication, and host accumulation wall time are separately measured and never converted into PIM kernel cycles.
- Descriptor traffic/fetch, preprocessing, unaligned splitting, ALU latency, request overlap, and native indexed accumulation are excluded.

## 11. Source tree and file responsibilities

- `CSCMatrixLoader.*`: Matrix Market and zero-based triplet loaders; conventional CSC construction.
- `CSCLayout.*`: policies, descriptor/BG images, materialization, invariants, layout statistics.
- `CSCFunctionalModel.*`: CPU reference, x packing, valid-count worker, partials, host accumulation, execution statistics.
- `CSCTimingModel.*`: topology assertions, Scheme8 address composition, separated DRAM phases and cycles.
- `CSCDescriptorSpMVTest.cpp`: edge cases and isolated topology/timing test.
- `CSCDescriptorSpMVBenchmark.cpp`: explicit benchmark and unique CSV output.
- `system_hbm_csc_fp32.ini`: milestone topology/precision.
- `Sconstruct`: adds only `src/tests/csc/*.cpp`; no recursive source-tree glob.

No `src/csc/` hardware directory exists in this milestone.

## 12. Build instructions

Use an isolated copy when the source tree is dirty:

```bash
cp -a /home/gs13022/PIMSimulator/. /tmp/pimsimulator_csc_m1_build/
scons -C /tmp/pimsimulator_csc_m1_build -c
scons -C /tmp/pimsimulator_csc_m1_build
```

Normal clean checkout command remains `scons`.

## 13. Test instructions

Run functional and topology tests separately:

```bash
./sim --gtest_filter='CSCDescriptorFunctionalTest.*'
./sim --gtest_filter='CSCDescriptorTimingTest.TopologyAndSeparatedPhases'
./sim --gtest_filter='PIMKernelFixture.mul'
./sim --gtest_filter='CSCPartialStreamFunctionalTest.*'
./sim --gtest_filter='CSRDirectSpMVPhysicalMappingTest.Scheme8RepresentativeRows'
```

GoogleTest does not create CSV unless the benchmark opt-in environment variable is set.

## 14. Benchmark instructions

Toy first:

```bash
env CSC_DESCRIPTOR_BENCHMARK_RUN=1 \
  CSC_DESCRIPTOR_MATRIX=/home/gs13022/SparsePIM/csc_partial_stream_sample.mtx \
  CSC_DESCRIPTOR_RUN_ID=toy_m1 \
  ./sim --gtest_filter=CSCDescriptorBenchmark.ExplicitRun
```

Cant uses the same command with `sparse_matrix_csc/cant_csc.txt` and a new run ID. Output is `results/csc/<matrix>_round_robin_fp32_64bg_<run_id>.csv`. Existing paths cause an assertion failure; files are never overwritten.

## 15. Output metrics

Layout metrics include dimensions/NNZ/BGs, nonempty columns/descriptors, useful/padding/physical bytes for values and indices, total padding and `padding/(useful value+index)` overhead, x useful/padding bytes, descriptor bytes, per-BG NNZ/descriptors, maximum/average BG NNZ, and load CV.

Execution metrics include full/tail/total chunks, active/available lanes and utilization, per-BG chunks/idle rounds, rounds/critical BG, scalar and burst x loads, value/index bursts, logical MUL events, partials, and host accumulations.

Timing metrics include all phase cycles and the three inclusion scopes defined above. Wall metrics separately report CPU reference, layout, x packing, functional multiplication, and host accumulation.

## 16. Test results

2026-07-27 staged clean build: success. New functional suite: 6/6 passed. New isolated topology/timing test: 1/1 passed. Toy benchmark: validation passed, maximum error 0. Required existing regressions passed: PIM MUL 1/1 (1,048,576 values, zero failures), legacy CSC functional 2/2, and Scheme8 mapping 1/1.

Toy statistics: 6x7, 9 NNZ, five descriptors, five tail chunks, 9/40 active lanes (22.5%); useful value/index bytes 36+36, padding 124+124, physical bytes 160+160, alignment overhead ratio 3.44444; 60 preload, 47 x, 47 value, 47 index, 22 result, 116 resident-kernel, 163 kernel-with-x, and 223 end-to-end simulated cycles. Descriptor bytes are 160. The high toy padding ratio is expected from five very short independently aligned columns. The final CSV schema also contains 64-entry BG NNZ, descriptor, chunk, and idle-round arrays.

Cant (`cant_csc.txt`) was run after toy validation. The first attempt failed at row 23 with maximum error 0.001953125 because FP32 host additions occurred in different CPU-column and BG-lockstep orders; layout and NNZ invariants had passed. After unifying both paths on double host accumulators over identical FP32 products, cant passed: 62,452x62,452, 4,007,384 NNZ, 62,452 descriptors, maximum error 1.52588e-05. Value/index useful bytes are 16,029,536 each; padding is 1,222,656 each (2,445,312 total, 7.62752% overhead). It produced 539,131 chunks, 4,007,384/4,313,048 active lanes (92.913%), 8,427 rounds, and critical BG 4. Timing was 398,848 preload, 1,456 x, 179,290 value, 179,120 index, 1,130 result, 359,540 resident-kernel, 360,996 kernel-with-x, and 759,844 end-to-end cycles.

## 17. Known limitations

This is not a native BG-local engine. Descriptors are ideal/local. Indexed accumulation is host code. Scheduling is lockstep. Packed unaligned CSC, splitting/staging, callback IDs, same-address outstanding requests, overlap, BG-decoupling, masked native PIM ISA, lane-aware energy, and separate PIM ALU latency are absent. Native FP32 SRF behavior is not used. FP32/8 lanes is evaluated first; SparsePIM-oriented FP16/16 lanes remains future work.

## 18. Milestone 2C simplified cycle calibration

The Phase 1 calibration deliberately runs a representative subset instead of the full 16-matrix by 3-policy timing sweep. The selector reads the completed analytical sweep, chooses toy, smallest, median, and optional largest workloads, and compares round-robin with the lowest-lockstep alternative (or the second-best policy when round-robin is already best). Cases are resumable, preserve per-case images and stdout/stderr, and record timeout as an incomplete performance observation rather than a correctness failure.

The external timing benchmark defines its boundaries as follows. `preload_cycles` begins before value/index image writes and ends after both write phases drain. It is excluded from all resident metrics. `resident_cycles` begins with the value-read phase and ends after the row-index phase drains; with result traffic set to `none`, it equals `value_completion_cycles + index_completion_cycles`. `with_x_cycles` additionally includes the preceding packed-x read phase and equals `x_load_cycles + resident_cycles`. `end_to_end_cycles` equals preload plus with-x. Assertions check all three identities and explicitly check that preload is excluded from resident cycles.

The 2026-07-29 run selected seven cases. Toy, ASIC_100k round-robin/load-only, and cant round-robin/load-similarity completed. Both optional crankseg_2 cases reached their 60-second timeout and did not invalidate the run. All five completed cases passed external FP32 functional comparison, Scheme8 address ownership, transaction/accounting invariants, and zero invalid-lane multiply/partial/accumulation checks. ASIC_100k analytical rounds and resident cycles both favored load-only (13,471/204,104 versus 12,023/190,601). Cant analytical rounds and resident cycles both slightly favored round-robin (8,427/358,410 versus 8,439/358,887).

These observations support analytical lockstep rounds as a policy-ordering indicator, not a direct linear cycle predictor. Absolute cycles also reflect row-buffer locality, channel/rank/BG contention, serialized request issue, x traffic, and the harness's separately drained phases. The calibration runner is `SparsePIM/csc_light_preprocess/run_csc_cycle_calibration.py`; the preserved result, logs, summary, and enriched timeout accounting are under `csc_light_preprocess/results/cycle_calibration_20260729_v2/`.

## 18. Future hardware milestones

**Milestone 2 – test-side timing refinement:** external light-mapping result loader; physical export from `csc_light_preprocess`; real preprocessing-to-simulator connection; refined lockstep timing; matrix/x/result accounting validation; FP32 SRF audit.

### Milestone 2A implementation status (2026-07-29)

The opt-in `csc_aligned` preprocessor path now reuses the existing feature,
mapping, and BG-local ordering stages but bypasses the legacy packed physical
materializer. It converts the original CSC values to FP32 and directly creates
the authoritative 32-byte column-aligned BG images, descriptors, and x
permutations. Existing `physical` behavior is retained; `physical_legacy` is
an explicit alias.

`CSCExternalImage` loads and validates the versioned image without remapping or
rematerializing it. The binary contract is in `CSC_IMAGE_FORMAT.md`. Toy
round-trip testing compares all bytes, descriptor fields, x permutations,
statistics, chunks, and functional output against the internal round-robin
builder. Cant correctness uses the reconstructed exported-FP32 semantics.

`CSCTimingModel` now validates every value/index chunk address by decoding the
Scheme8 address generated by `addrGenSafe`. Values retain bank 0/base row 0;
indices bank 1/base row 4096; x bank 2/base row 8192; analytical results bank
3/base row 12288. Result traffic is explicitly `None` or `Analytical`.

The FP32 SRF audit found and fixed precision-independent FP16 indexing in
`PIMRank::readOpd()`. A separate test checks both SRF halves, scalar replication
to eight FP32 lanes, and MUL output. Native CSC execution remains disabled.

Reproduce external tests from an isolated build:

```bash
env CSC_EXTERNAL_IMAGE=/tmp/csc_toy_image \
    CSC_EXTERNAL_MATRIX=/home/gs13022/SparsePIM/csc_light_preprocess/testdata/toy_csc.txt \
    ./sim --gtest_filter='CSCExternalImageFunctionalTest.*'
env CSC_EXTERNAL_IMAGE=/tmp/csc_toy_image \
    ./sim --gtest_filter='CSCExternalImageTimingTest.*'
env CSC_EXTERNAL_IMAGE=/tmp/csc_cant_image \
    ./sim --gtest_filter='CSCExternalImageCorrectnessTest.*'
./sim --gtest_filter='CSCFP32SrfAuditTest.*'
```

The 16-matrix x three-policy in-memory preprocessing sweep is recorded in
`SparsePIM/csc_light_preprocess/results/csc_aligned_m2_16_matrix_sweep.csv`.
Full external-image DRAM cycle sweeping remains opt-in because the cant timing
run alone submits more than one million runtime value/index transactions in
separately drained phases.

**Milestone 2 status:** M2A aligned image/materializer/exporter, M2B external loader/round-trip, and M2C accounting plus simplified cycle calibration are complete.

**Milestone 3 – native masked SIMD:** masked PIMBlock ADD/MUL/MAC/MAD uses one `valid_count` source of truth; inactive lanes do not access operands, execute arithmetic, modify destination state, or generate partials. Unit tests cover valid counts 0/1/3/5/7/8, NaN/Inf sentinel safety, full-lane legacy equivalence, invalid-count rejection, counters, and toy external-image native masked MUL. Implementation and validation use an isolated snapshot; native descriptor issue remains outside Milestone 3.

**Milestone 4 – native BG-local descriptor engine:** implemented in `src/csc/CSCDescriptorEngine.*`, `CSCRequestTracker.*`, and `CSCTypes.h`. It provides 64 persistent BG FSMs, local descriptors, runtime packed-x selection/reuse, Scheme8 value/index/x requests, completion-gated staging, M3 masked MUL, valid partial emission, busy/done/flush/reset, and counters. Its original serialized behavior remains available through `CSCRequestPolicy::SERIALIZED`. See `MILESTONE4_NATIVE_ENGINE.md`.

**Milestone 5 – request identity/overlap: FROZEN.** Opaque tokens (`request_id`, kind, BG, worker, sequence, client, generation) survive Transaction, command BusPacket, read DATA return or write completion and are returned by a parallel token-aware callback. A request-ID scoreboard preserves retry identity, tracks accept-before-failure requests as `ABANDONED` by kind and BG, and enforces `created == accepted + abandoned` and `accepted == completed`. Request-ID and generation exhaustion are guarded without wrap or reuse. Independent x/value/index slots and multiple BGs overlap under `OVERLAPPED`; same-address requests are correct by token, not FIFO. The canonical cycle metric is `launch_to_done_cycles`; `total_execution_cycles` remains an equal compatibility mirror. Normal and isolated `NO_STORAGE=1` freeze regressions pass. Tokenized pending reads still use a correct full-token linear scan; a controller-local request-ID map remains M6 preparation. See [`MILESTONE5_REQUEST_IDENTITY_OVERLAP.md`](MILESTONE5_REQUEST_IDENTITY_OVERLAP.md).

**Milestone 4 stabilization:** `ERROR` is terminal failure with preserved BG/code/message and accepted-request drain; successful `DONE` is distinct from failure. Graceful flush is implemented (not cancellation), drains the full launched matrix and all partials through observable `FLUSHING`, and is idempotent. The stale self-extracting sim wrapper was replaced by a normal SCons-linked ELF; the standard GoogleTest runner returns 0 for assertion success and nonzero for assertion failure or fatal startup. M5 adds tokenized overlap while retaining this ERROR/FLUSH behavior.

**Milestone 6 – BG-decoupling:** separate rank-wide CRF/PC constraints, targeted execution, per-BG state/queue, shared bus contention, lockstep comparison, dual-PIM-block scheduling.

**Milestone 7 – BGA/GA:** row-index-aware queues, compare/merge, capacity/backpressure, selective enqueue, flush/TSV/GA, cross-BG accumulation, functional validation, comparison to analytical models.

**Milestone 8 – final evaluation:** FP16/16 lanes, SparsePIM parameters, padding analysis, full preprocessing+x+kernel+result latency, mapping comparisons, and RTL area/power specification.

## 19. Change log

### 2026-07-27 – Codex milestone 1

- Git HEAD before work: `5b3ec1e8de472fa3f9c07a4582ca716e2f74de66` on `dev`.
- Pre-existing status: seven modified files (three Guided_kmeans result texts; `CSCPartialStreamSpMV.cpp`; `SpmvBenchTestCases.cpp`; `SpmvDrafBgaStructuralCommon.{cpp,h}`), 230 insertions/90 deletions. None was changed by this work.
- Created: ten `src/tests/csc` files, this README, FP32 ini, and two explicitly requested benchmark result CSVs (toy and cant).
- Modified: `Sconstruct` only, adding the narrow nested test glob.
- Implemented: loader, aligned physical images, descriptors, two mapping interfaces, x packing, valid-count functional lockstep execution, host accumulation, statistics, timing phases, topology assertion, tests, opt-in benchmark.
- Staged commands: `scons -c; scons`; new functional/timing filters; explicit toy benchmark.
- Final results: clean build passed; 6 functional and 1 topology/timing tests passed; PIM MUL, legacy CSC, and Scheme8 regressions passed; toy and cant correctness passed.
- Cant diagnostic: initial FP32 accumulation-order mismatch was fixed by common double host accumulation; no parser, descriptor, duplicate, or NNZ loss was found.
- Remaining: native hardware and timing refinements listed in section 18.

## 20. Git/build environment

PIMSimulator branch is `dev`; compiler is g++ 13.3.0 and SCons is 4.5.2. The working tree was already dirty, so implementation/build happened in a byte-for-byte staging copy and only explicit new files plus the reviewed SCons change are transferred back. No reset, checkout, clean, branch change, commit, or overwrite of existing CSV was performed.

## FP16 milestone documents

- [M5.1 ingress and production configuration](CSC_FP16_M5_1_INGRESS_AND_PRODUCTION_CONFIG.md)
- [M6 transport and ordered host reduction](CSC_FP16_M6_TRANSPORT_AND_HOST_REDUCTION.md)
- [M7 execution modes and evaluation](CSC_FP16_M7_EXECUTION_MODES_AND_EVALUATION.md)
