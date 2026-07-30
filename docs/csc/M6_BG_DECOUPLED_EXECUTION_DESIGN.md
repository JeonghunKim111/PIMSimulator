# M6 BG-Decoupled Execution Design

## 1. Scope and non-goals

M6 connects the existing CSC descriptor/request finite-state machines to the physical
`PIMBlock` objects owned by `PIMRank`. It adds BG-targeted masked multiplication,
rank-local arbitration, one-cycle completion, and BG-local lifecycle isolation.

M6 does not implement BGA/GA, PIM-side indexed accumulation, M7 behavior, arbitrary
per-BG CRF programs, BG-local opcode streams, dual issue inside one BG, dual-worker
chunk scheduling, descriptor prefetch, a new CSC image format, preprocessing changes,
or structural-model changes.

## 2. Baseline and preserved M5 invariants

The implementation baseline is
`1544327d58ad0b865c80bb2e530310199f04d6d7` (`docs: define M6 BG-local
execution boundary`). M5 retains ownership of descriptor/chunk progress, operand
slots, `RequestToken`, request generation, admission retry, completion routing,
and exactly-once memory-request retirement.

M6 does not change the CSC image or memory request identity. Same-address requests,
out-of-order completions, queue backpressure, and
`tokenized_linear_lookup_count == 0` remain invariants. Legacy CRF broadcast and
rank-wide PIM arithmetic remain unchanged when targeted execution is unused.

## 3. BG-to-PIMBlock binding

The production configuration has 16 channels, one rank per channel, four bank
groups, 16 banks, and eight physical PIM blocks
(`system_hbm_csc_fp32.ini:2`,
`ini/HBM2_samsung_2M_16B_x64.ini:1-5`). `PIMRank` constructs exactly
`NUM_PIM_BLOCKS` physical blocks (`src/PIMRank.cpp:24-36`). Existing HAB operand
access binds physical block `p` to banks `2p` and `2p+1`
(`src/PIMRank.cpp:144-159,230-253,280-309`).

M6 therefore defines a static, topology-checked endpoint mapping:

| local BG | physical PIMBlock pair | physical banks |
|---|---|---|
| 0 | 0, 1 | 0-3 |
| 1 | 2, 3 | 4-7 |
| 2 | 4, 5 | 8-11 |
| 3 | 6, 7 | 12-15 |

Construction fails unless `NUM_BANK_GROUPS == 4`, `NUM_BANKS == 16`,
`NUM_PIM_BLOCKS == 8`, and every pair maps to four distinct in-range banks.
The target carries channel, rank, and local BG. A two-bit mask is relative to the
BG pair: `01` selects its first block, `10` its second, and `11` selects both.
Zero and out-of-range masks are rejected. Production CSC uses a one-hot mask.
The pair mask is accepted only when both selected blocks have valid operation
contexts; it produces one aggregate completion after both selections finish.
It is not used to execute the same CSC chunk twice.

## 4. Rank execution modes

Each `PIMRank` owns:

`IDLE -> LEGACY_RANK_WIDE | CSC_BG_TARGETED -> DRAINING -> IDLE`

and the failure path:

`* -> ERROR_DRAINING -> ERROR`.

`IDLE -> LEGACY_RANK_WIDE` requires zero targeted pending, accepted, executing,
and completion records. `IDLE -> CSC_BG_TARGETED` requires the legacy CRF
command/writeback path to be quiescent. Every exit passes through `DRAINING`.
Legacy packets are rejected while targeted mode owns the rank; targeted
submissions are rejected while legacy owns it. There is no implicit execution in
the other mode. The exclusion scope is one rank, so different ranks may own
different modes concurrently.

Existing `Rank::mode_` continues to represent SB/HAB/HAB_PIM DRAM behavior
(`src/Rank.h:98-111`, `src/Rank.cpp:376-455`). The new execution mode guards
ownership without changing those legacy meanings.

## 5. Targeted operation format and identity

The only M6 opcode is `BG_TARGETED_MASKED_MUL`. An operation contains:

- `operation_id`, channel, rank, local BG, two-bit PIMBlock mask, and opcode;
- worker ID, sequence number, generation, and valid lane count;
- one operand/result context per selected block.

Operation states are `READY`, `WAITING_FOR_GRANT`, `ACCEPTED`, `EXECUTING`,
`COMPLETED`, and `RETIRED`. Identity is immutable while waiting. Completion
repeats the full identity and carries the aggregate selected-mask result.
Unknown, duplicate, stale-generation, BG, worker, opcode, or mask mismatches are
errors. Each accepted operation yields exactly one completion and each completion
is retired exactly once. Targeted operation IDs are separate from memory
`RequestToken` IDs.

## 6. PIMRank arbitration

`CSCDescriptorEngine` determines operand readiness and creates an operation.
`PIMRank` owns one depth-one pending slot per local BG, one active operation per
BG, the physical mapping, eligibility checks, `next_bg_rr`, dispatch, and
completion.

At most one operation is granted per rank per simulator cycle. Arbitration scans
the four BGs starting at `next_bg_rr`, skips non-ready, busy, flushing, errored,
or resource-conflicting BGs, and advances the cursor after a grant. A skipped
request keeps its identity and operand context. An accepted operation is
non-preemptive. This deterministic round-robin policy prevents fixed-priority
starvation and allows an ineligible BG to be bypassed.

## 7. Shared-resource contention

The controller owns the real channel command slot. A command is actually selected
at `MemoryController::update()` through `commandQueue.pop()` and is serialized by
`outgoingCmdPacket` (`src/MemoryController.cpp:482-493,555-564`). Rank read data
uses `outgoingDataPacket` and `dataCyclesLeft`
(`src/Rank.cpp:478-515`). M6 records the rank target of each command selected in
the current controller cycle and exposes rank command/data-bus availability.

`MemorySystem::update()` currently updates ranks before the controller
(`src/MemorySystem.cpp:239-265`). Targeted service therefore occurs in an
explicit post-controller phase: it uses the command-selected-this-cycle signal,
rank data-bus occupancy, BG lifecycle state, and physical PIMBlock busy mask.
A legacy command using the same rank's shared command slot prevents a targeted
grant in that cycle. This is a real shared-resource reservation, not an
always-available port or a post-hoc maximum of independent BG timelines.

## 8. Cycle ordering and completion semantics

For simulator cycle `t`:

1. completions scheduled at `t` become visible and release their block mask;
2. rank/DRAM state updates and memory callbacks run;
3. every CSC engine performs at most one meaningful FSM transition;
4. ready targeted operations are submitted;
5. after actual controller issue, `PIMRank` tests eligibility and grants at most
   one operation;
6. functional masked multiplication runs on the selected physical block(s), with
   its aggregate completion scheduled for `t+1`.

A completion is never visible in its grant cycle. An engine enters partial
emission only after consuming completion. Consequently memory completion cannot
cascade through request, grant, completion, and next chunk in one tick. Tick
ownership uses the existing monotonically increasing simulator cycle and does
not duplicate or rewind it.

## 9. BG lifecycle and flush/reset/error isolation

Each BG has `IDLE`, `RUNNING`, `FLUSH_REQUESTED`, `DRAINING`, `FLUSHED`,
`ERROR_DRAINING`, `ERROR`, and `RESETTING`.

Flush is graceful drain. After request, the BG starts no descriptor, memory
request, targeted operation, or next chunk. A not-yet-accepted request may be
abandoned through the existing tracker rules. Accepted memory requests retire
normally and granted operations complete non-preemptively. The flushed BG's
partials are not committed as successful output and its result is
`INCOMPLETE_FLUSHED`. Other BGs continue.

Reset is allowed only when memory outstanding, targeted outstanding, pending
completion, PIMBlock busy mask, and pending writeback are all zero. Reset during
`RUNNING` requests flush, drains, enters `RESETTING`, increments generation, and
returns to `IDLE`; force deletion of in-flight work is forbidden.

Descriptor bounds, valid-count, BG-local request/operation mismatch, stale or
duplicate completion, operand-slot mismatch, and completion BG/mask mismatch put
only the owning BG through `ERROR_DRAINING -> ERROR`. Mapping corruption,
duplicate physical ownership, simultaneous legacy/targeted ownership,
command-bus ownership corruption, unknown completion owner, or shared-counter
underflow put the rank through `ERROR_DRAINING -> ERROR`, stop all new issue, and
drain accepted work. Rank mode exit drains all participating BGs.

## 10. CSCDescriptorEngine integration

The descriptor engine keeps the M5 descriptor/request FSM and tracker. Its
`SIMD_MUL` transition becomes operation creation followed by
`WAIT_TARGET_GRANT` and `WAIT_TARGET_COMPLETION`. The operation ID and complete
operand context remain stable until accepted. Only a matching completion
provides the physical PIMBlock result and permits `EMIT_PARTIALS`.

`CSCNativeExecution` maps global BG `g` to channel `g / 4`, rank zero, and local
BG `g % 4`; it routes targeted submissions and completions separately from
memory callbacks. Production arithmetic no longer terminates in the 64 logical
test-helper blocks. Descriptor state is not copied into `PIMRank`, and memory
requests continue through the M5 tracker.

## 11. Statistics and evaluation

M6 records `per_bg_ready_cycles`, `per_bg_executing_cycles`,
`per_bg_memory_wait_cycles`, `per_bg_grant_wait_cycles`, and
`per_bg_flush_drain_cycles`; `rank_targeted_grants`,
`rank_command_bus_stall_cycles`, `rank_resource_conflict_stall_cycles`, and
`rank_mode_drain_cycles`; `per_pimblock_active_cycles`,
`per_pimblock_targeted_ops`, and `round_robin_skip_count`.

Balanced and imbalanced synthetic workloads run through the simulator in both
lockstep-reference and BG-decoupled modes. Reports include total and per-BG
completion cycles, idle/wait cycles, command/resource stalls, physical-block
active cycles, operation count, and memory overlap. Correctness never depends on
wall-clock time, and the decoupled result is not computed as `max()` of
independent BG estimates.

## 12. Acceptance tests

Acceptance covers topology and mapping, both one-hot masks, guarded pair-mask
aggregation, untouched non-target blocks, rank-local mode exclusion, independent
rank modes, drain transitions, four-way round-robin order, one grant per rank per
cycle, command/data/block conflicts, starvation freedom, next-cycle completion,
no same-cycle cascade, identity stability, exactly-once retirement, and negative
unknown/duplicate/stale/mismatch cases.

It also covers independent BG FSM states, variable-length columns, memory-wait
overlap, BG-local flush/reset/error, rank-fatal escalation, internal and external
images, FP32 lane counts 1-8, empty/short/multi-chunk columns, multiple BGs,
same-address requests, backpressure, forced out-of-order completion, CPU-reference
results, normal storage, and `NO_STORAGE=1`. Existing M1-M5, M6-preparation,
masked-SIMD/SRF, external-image, descriptor, partial-stream, and legacy MUL/ADD
regressions remain mandatory.
