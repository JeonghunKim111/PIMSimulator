# Milestone 4: native BG-local CSC descriptor engine

## Scope and ownership

`CSCNativeExecution` is the simulator-side execution owner. It owns 64 persistent
`CSCDescriptorEngine` contexts (one per global BG), one FP32 `PIMBlock` datapath
per context, and a `MultiChannelMemorySystem`. Its `tick()` performs one normal
DRAMSim update and then permits at most one FSM transition per BG. It does not
modify `MemorySystem::update()` or `Rank::update()` ordering and does not call the
host transaction API recursively from `PIMRank::update()`.

The explicit launch API accepts 64 authoritative `CSCBGImageView` objects,
matrix rows/NNZ, runtime BG-local packed x vectors, and a partial sink. Mapping,
descriptor order, offsets, x slots, alignment, and physical byte images are not
recomputed. Launch validates descriptor ownership, nonzero NNZ, 32-byte offsets,
x-slot bounds, complete physical burst ranges, and the global descriptor-NNZ
sum. Empty queues complete normally. `reset()` enables relaunch; it rejects a
pending request. `flush()` has drain semantics: issued work and valid partials
are completed before DONE rather than cancelled.

## FSM and datapath

Each engine explicitly progresses through:

`IDLE -> FETCH_DESCRIPTOR -> LOAD_X -> FETCH_VALUE -> FETCH_ROW_INDEX ->
WAIT_OPERANDS -> SIMD_MUL -> EMIT_PARTIALS -> ADVANCE_CHUNK ->
NEXT_DESCRIPTOR -> DONE`.

Descriptors are ideal/local and generate no descriptor-read traffic. LOAD_X
issues one 32-byte bank-2 read per descriptor and selects `x_slot % 8` from the
authoritative packed-x view after completion; the scalar is reused by all
chunks. Value and row-index requests use Scheme8 `addrGenSafe` with bank/base
row 0/0 and 1/4096. X uses bank/base row 2/8192. BG-local byte offsets are
converted to burst indices before encoding; no absolute-base integer addition
is used.

On completion, the corresponding 32-byte authoritative image range is copied
to separate value/index staging bursts. DRAMSim callbacks currently return only
`(channel,address,cycle)`, not payload. Consequently image bytes are never used
before timing completion, but are sourced from the loaded image after that
completion. SIMD_MUL computes `min(remaining_nnz,8)` and calls the Milestone 3
masked `PIMBlock::mul`; it never performs a full-width multiply and discards
lanes afterward. The current PIM ALU adds no independent latency, so SIMD_MUL is
one FSM transition/cycle after operands become ready.

EMIT_PARTIALS creates only valid `(row,value,bg,descriptor,chunk,lane)` records.
The optional `hostAccumulate()` uses double host sums and is outside native
kernel execution. There is no PIM-side indexed accumulation or y writeback.

## Serialized M4 request tracking

`CSCRequestTracker` records one current X, value, or index request per engine.
The owner additionally permits only one outstanding CSC request globally. This
strict fallback prevents ambiguous same-address callbacks and makes completion
dispatch unambiguous without opaque request IDs. Submission failure produces
backpressure and is retried on a later tick. DONE requires no owner request, no
DRAM pending transaction, and no engine completion token.

This is intentionally not M5: there are no request IDs, worker/sequence tokens,
multiple-outstanding scoreboards, same-address concurrency, or value/index/x
overlap. The 64 FSM contexts are persistent BG-local state, but the single
global request gate is not a claim of fully BG-decoupled M6 execution.

## Counters and tests

Counters cover descriptor/x/value/index traffic, logical MULs, full/tail/total
chunks, active/available lanes, invalid multiply/write activity, generated and
emitted partials, host accumulations, request/backpressure stalls, busy cycles,
and total execution cycles. Tests assert NNZ, transaction, chunk, descriptor,
partial, host accumulation, and invalid-lane invariants.

Reproduce:

```bash
scons
./sim --gtest_filter='CSCFP32MaskedSIMDTest.*'
./sim --gtest_filter='CSCNativeDescriptorEngineTest.*'
env CSC_EXTERNAL_IMAGE=/path/to/toy/image \
  ./sim --gtest_filter='CSCNativeDescriptorEngineIntegrationTest.*'
scons NO_STORAGE=1
./sim --gtest_filter='CSCNativeDescriptorEngineTest.*'
```

With `NO_STORAGE=1`, DRAM timing/state/callback behavior is still exercised.
Functional staging remains available because M4 deliberately obtains payload
from the authoritative external image only after callback completion.

The toy external image completes in both builds. The optional cant run is kept
under an external timeout: it reached 120 seconds without a functional,
accounting, address-ownership, or invalid-lane mismatch being reported. This is
an incomplete observation caused by M4's deliberately global serialized request
fallback, not a completed cant acceptance result.

## Stabilization semantics

ERROR is a failed terminal state. Engine APIs separate isTerminal() (DONE or ERROR), isDone() (successful DONE), and hasFailed() (ERROR); errorCode() and errorMessage() preserve the first cause. CSCNativeExecution also preserves failedEngine(). Legacy done() means terminal, so wait loops stop on failure; callers check isDone() or hasFailed() before consuming results.

The owner latches the first error and rejects new submissions. A request already accepted by DRAMSim is never cancelled: its callback and completion token drain before global failure becomes terminal. Valid earlier partials remain observable, but hostAccumulate() rejects failed runs.

Flush is graceful completion, not cancellation. All launched descriptors finish, accepted requests drain, and valid staged partials emit before observable FLUSHING advances to successful DONE. flushRequested() and flushComplete() expose progress. Flush is idempotent; IDLE is an empty successful execution, DONE drains once, and ERROR never changes to success.

The baseline sim artifact was a sectionless self-extracting wrapper. It re-executed an embedded payload (whose exit status was 0), waited for that child twice, and returned -1 from the parent. Rebuilding after removing that stale wrapper produces the normal raw ELF; the unchanged main returns RUN_ALL_TESTS(), so PASS returns 0 while assertion failure and fatal startup remain nonzero. This was a common build-artifact issue, not M4 teardown.

M5 request IDs, multiple outstanding requests, callback redesign, overlap, and arbitration remain deferred.

## Deferred work

M5 must add opaque request identity, multiple outstanding requests, overlap,
and robust same-address tracking. M6 must define truly independent BG issue and
shared-resource arbitration. M7 must add BGA/GA, indexed accumulation, and
native final-y production. Descriptor DRAM fetch, vector x preload/gather,
dual-PIM-block issue, transaction batching, and a new image format are not
claimed by M4.
