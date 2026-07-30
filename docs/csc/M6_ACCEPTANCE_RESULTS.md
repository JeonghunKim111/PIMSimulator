# M6 Acceptance Results

## 1. Core baseline

The completion branch starts at remotely preserved `origin/m6-core`, commit
`b17f7ea8387c6d597596204cb3f36319324e9aa1`.  The M6 preparation ancestor is
`1544327d58ad0b865c80bb2e530310199f04d6d7`.

## 2. Lockstep reference definition

The barrier-synchronized lockstep reference shares the production
MemoryController, request identity, PIMRank arbitration, physical PIMBlocks,
masked MUL, completion, partial, and result paths.  Its only difference is a
chunk-boundary progress barrier.  Finished, flushed, and failed BGs are not
active barrier participants; accepted work continues to drain.

## 3. Balanced comparison

Four BGs each execute two chunks.  BG-decoupled completes in 155 cycles and
lockstep in 164 cycles.  Both accept 20 memory requests and 8 targeted
operations, produce identical results and completion counts, and emit the same
partials.  Lockstep records 32 aggregate barrier-wait cycles.

## 4. Imbalanced comparison

The workload uses a delayed one-chunk BG0, a four-chunk BG1, a one-chunk BG2,
and an empty BG3.  BG-decoupled completes in 218 cycles; lockstep completes in
292 cycles.  Lockstep adds 74 cycles and the lockstep/decoupled ratio is
1.33945.  BG0 completes at cycle 139 in both modes.  BG1 completes at cycle
218 decoupled and 292 lockstep.  Lockstep records 146 barrier-wait cycles.
Both modes record 2 command-bus stall cycles, 0 PIMBlock resource-conflict
stall cycles, and one active cycle on PIMBlock 0.  Results and physical request,
operation, completion, and partial counts are identical.

## 5. Completion negative tests

Duplicate, unknown operation ID, opcode mismatch, and PIMBlock-mask mismatch
are injected through the shared production validator.  Rejected identity
mismatches preserve the queued valid completion.  Duplicate delivery does not
increment retirement or emit a second partial.

## 6. BG-local isolation

A corrupted BG0 completion drives BG0 through `ERROR_DRAINING` to `ERROR`.
BG1 remains running, receives its physical completion, and produces the valid
result.  The rank does not escalate for the BG-local mismatch.

## 7. Rank-fatal escalation

An injected busy-mask/active-owner disagreement is detected by the production
shared ownership validator.  The rank enters `ERROR_DRAINING`, blocks new
issue, drains the accepted operation, clears the busy mask and queues, then
enters `ERROR`.

## 8. Flush drain coverage

The acceptance accessor creates a state with a real accepted MemoryController
request and a granted physical targeted operation in the same BG.  BG flush
stops new issue; the memory callback and targeted completion each retire once.
Final memory outstanding, targeted outstanding, pending completion, and
PIMBlock busy mask are zero.  The flushed BG is `INCOMPLETE_FLUSHED`, while a
second BG completes and preserves its result.

## 9. Regression

The focused M6 core and completion run executes 30 tests with 30 passed, zero
failed, and process status 0.  The pre-final normal regression executes 78/78
with zero skipped and status 0, including the fresh external image and legacy
MUL/ADD.  The pre-final `NO_STORAGE=1` regression executes 76/76 with zero
skipped and status 0; storage-dependent legacy functional kernels are excluded.

## 10. Final verdict

M6 ACCEPTANCE COMPLETE.  Core correctness, physical integration, scheduling
comparison, completion identity, error isolation, and joint flush drain pass.
The final detached-worktree rerun is the release verification of this verdict.
