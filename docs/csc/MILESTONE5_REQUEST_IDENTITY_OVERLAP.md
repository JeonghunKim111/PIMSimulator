# Milestone 5 — Request Identity and Memory Overlap

## Overview

M5 adds opaque request identity and a request-ID scoreboard to the CSC descriptor engine, and safely overlaps x/value/index and multi-BG memory transactions through the actual DRAM timing path. It does not implement fully BG-decoupled execution, complete PIM-side SpMV, or native BGA/GA accumulation.

## RequestToken ABI

`DRAMSim::RequestToken` carries `uint64_t request_id`, `RequestKind request_kind`, `uint32_t global_bg_id`, `uint32_t worker_id`, `uint64_t sequence_number`, `uint32_t client_id`, and `uint32_t generation`. Supported CSC kinds are `VALUE_READ`, `INDEX_READ`, `X_READ`, `DESCRIPTOR_READ`, and `RESULT_TRANSFER`; `NONE` is invalid.

Request ID 0 is invalid. `CSCNativeExecution` allocates IDs monotonically. A rejected submission retains its token and retries with the same ID; an accepted ID is never reused. Reset does not rewind either the next request ID or generation. Every successful launch increments generation, so an old completion is rejected even when its request ID is known.

The allocator never silently wraps: `UINT64_MAX - 1` is the last allocatable ID. When the next-ID sentinel reaches `UINT64_MAX`, allocation fails before a token or scoreboard entry is created and execution latches terminal `INTERNAL_INVARIANT` with `request ID exhausted`. Generation similarly never wraps: a launch attempted at `UINT32_MAX` is rejected as terminal failure with `generation exhausted`.

## End-to-end propagation path

The implemented path is:

```text
CSC descriptor engine
→ MultiChannelMemorySystem
→ MemorySystem
→ Transaction
→ MemoryController transaction queue
→ BusPacket
→ read DATA return or write completion
→ token-aware callback
→ CSCNativeExecution request-ID routing
→ CSCRequestTracker and the matching operand slot
```

A tokenized completion is matched using the full token identity, not its address, channel, FIFO position, or string tag. Same-address requests remain distinct. Only token-less legacy requests use the pre-existing address matching path. External-image bytes are staged only after the real memory completion reaches the matching operand slot.

## Callback observer contract

Legacy callback and token-aware callback are separate observer interfaces. A tokenized transaction may notify both observers for backward compatibility, but one logical client must not consume or account the same transaction through both callbacks.

In CSC, the legacy completion handler is a compatibility sink. CSC request accounting and operand readiness are driven only by the token-aware callback, and each tokenized transaction is consumed exactly once by the CSC tracker.

## Scoreboard state machine

The scoreboard states are:

```text
CREATED → WAITING_TO_SUBMIT → ACCEPTED → COMPLETED → RETIRED
CREATED or WAITING_TO_SUBMIT → ABANDONED   (failure before acceptance)
```

`CREATED` denotes allocation before submit eligibility; in the current implementation registration and entry into `WAITING_TO_SUBMIT` are atomic, so it is normally a logical/transient state rather than an externally observed active entry. `WAITING_TO_SUBMIT` records an attempted request that was not accepted. `ACCEPTED` is owned by the memory system and must be drained. `COMPLETED` records the unique matching completion. `RETIRED` is preserved in history after the engine consumes it. `ABANDONED` records an unaccepted request discarded by terminal failure, including token, kind, BG, worker, sequence, address, attempts, first-attempt cycle, and abandonment cycle.

`ACCEPTED/COMPLETED` entries cannot become `ABANDONED`; `ABANDONED` cannot be accepted; retired or abandoned requests cannot complete; completion before acceptance and duplicate ID registration are invariant failures.

## Backpressure and retry

A rejected request retains its logical operand slot and complete token and is retried on a later tick with the same request ID. Only acceptance increments issued/accepted counts; accepted requests are never resubmitted. Retry count describes attempts beyond the first, `submit_rejected` counts rejected attempts, and `abandoned_waiting_requests` counts logical requests abandoned before acceptance.

## Operand overlap

For the first descriptor chunk, `X_READ`, `VALUE_READ`, and `INDEX_READ` are independently created and may all be outstanding. Later chunks reuse x and create only value/index requests. SIMD execution begins only when:

```text
x_ready && value_ready && index_ready
```

A completion marks only the operand slot named by its token ready. Multi-BG requests use deterministic round-robin front-end issue while the memory controller retains its existing timing and arbitration.

## ERROR semantics

The first engine error is latched globally with BG, code, and message. New request creation and submission stop. `CREATED` and `WAITING_TO_SUBMIT` entries transition exactly once to `ABANDONED`; they are not counted as completions. Every `ACCEPTED` entry remains live until its real completion is drained. Accepted requests are neither erased nor virtually completed. Terminal ERROR cannot be overwritten by success.

## FLUSH semantics

Flush is graceful completion, not cancellation. It completes the current descriptor/chunk work, drains accepted transactions and valid partials, transitions through `FLUSHING` to `DONE`, and is idempotent. Flush in ERROR cannot replace failure with DONE.

## Counter invariants

At terminal state:

```text
created_requests == accepted_requests + abandoned_waiting_requests
accepted_requests == completed_requests
abandoned_waiting_requests == sum(abandoned_by_kind)
abandoned_waiting_requests == sum(abandoned_by_bg)
```

Normal DONE additionally requires zero abandoned requests and an empty pending scoreboard. ERROR may have abandoned requests, but all accepted requests must still be completed.

## Cycle metric

The canonical public metric is `launch_to_done_cycles`: the number of global execution ticks from immediately after launch initialization through the terminal DONE transition. It includes memory-system updates, operand completions, SIMD state transitions, synchronous partial emission, descriptor advancement, and terminal-state transition. It excludes host-side accumulation time.

It is not the first-accepted-to-final-completion DRAM interval, a sum of accepted-to-completion latencies, memory-active cycles, or end-to-end wall-clock time including host accumulation. The compatibility field `total_execution_cycles` remains and is a mirror of the canonical field at the single counter aggregation point; tests require equality in success and error paths. Cycle comparisons are therefore reported as `SERIALIZED launch_to_done_cycles` and `OVERLAPPED launch_to_done_cycles`.

## Known limitation

Tokenized read completion currently finds the matching `Transaction` by a full-token linear scan of `MemoryController::pendingReadTransactions`. This is correct, but costs O(N) per completion and may increase simulator host-side cost at larger outstanding depths. A controller-local `request_id → pending Transaction*` map is intentionally not part of M5 correctness or this freeze patch and remains an M6-preparation optimization. The legacy address scan remains supported.

## Test results

Freeze verification on 2026-07-29 used freshly SCons-linked raw ELF binaries (41 ELF sections). Normal `scons` passed. Normal results: M3 5/5, M4 7/7, M5 identity/scoreboard/overlap/freeze 9/9, toy native integration 1/1, M1/M2/FP32/Scheme8 group 11/11, toy external image byte/correctness/timing 3/3, and FP16 MUL/ADD 2/2; every process returned 0. The bounded soc-sign-epinions 5,000-NNZ prefix native integration passed 1/1 with RC 0.

An isolated `/tmp` `scons NO_STORAGE=1` build passed: M3/M4/M5 core 21/21, toy native integration 1/1, and M1/M2/FP32/Scheme8 12/12; every process returned 0.

The crafted comparison produced `SERIALIZED launch_to_done_cycles = 240` and `OVERLAPPED launch_to_done_cycles = 57`, with identical functional results and transaction counts, maximum global outstanding 6, and 51 ticks with outstanding at least two.

The first M2 external invocation supplied only the matrix environment variable and correctly skipped 3/3 tests; it was not counted as a pass. Re-running with both the verified toy image and matrix executed and passed all three tests. Full `cant` was not run.

## M6 boundary

M5 retains ideal/local descriptors, one logical FP32 PIM block per global BG, host partial accumulation, and no final y writeback, indexed accumulation, BGA/GA, descriptor memory fetch, multi-chunk lookahead, or BG-targeted rank command execution. Those are not implied by M5 freeze status.
