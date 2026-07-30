# M6 Preparation: BG-Local Execution Boundary Audit

## 1. Scope

This audit defines the M6 boundary without implementing BG-targeted commands, BG-local CRF/PC, arbitration, or descriptor-engine integration into `PIMRank`.

The CSC configuration has 16 channels, four bank groups and 16 banks per rank, eight physical PIM blocks per rank, FP32 arithmetic, and queue depth 64 (`system_hbm_csc_fp32.ini:2-11`, `ini/HBM2_samsung_2M_16B_x64.ini:1-8`). CSC uses 64 global BGs (`src/csc/CSCDescriptorEngine.h:137,144,185-190`), equal to 16 channels times four BGs.

## 2. Current rank-wide execution model

`MemorySystem` constructs each `Rank` and its single `PIMRank` (`src/MemorySystem.cpp:134-150`, `src/Rank.cpp:48-60`). The rank owns one mode (`src/Rank.h:98-111`). `Rank::sendToBank()` routes READ/WRITE through normal bank access, HAB movement, or `PIMRank::doPIM()` (`src/Rank.cpp:376-399`).

`PIMRank` owns one CRF, PC, jump/repeat context, toggle set, and exit flag for the whole rank (`src/PIMRank.h:59-98`). Entering PIM mode resets that single context (`src/PIMRank.cpp:96-113`). `doPIM()` decodes `crf.data[pimPC_]`, mutates shared control state, and dispatches every non-control instruction to all PIM blocks (`src/PIMRank.cpp:329-420`). CRF is rank-wide; SRF/GRF broadcast writes update every block (`src/PIMRank.cpp:164-201`). `PIMRank::update()` is empty (`src/PIMRank.cpp:66`), so instruction progress is packet-triggered.

Therefore one `pimPC_` applies to all blocks and each decoded CRF instruction is broadcast. The topology has eight PIM blocks but four BGs per rank. HAB access indexes banks as `pb * 2 + packet->bank` (`src/PIMRank.cpp:144-159`), so a block represents a selected bank pair, not one BG. The exact contention relation is not first-class and remains **UNRESOLVED**.

## 3. Current CSC M5 execution model

Each `CSCDescriptorEngine` owns a fixed global BG ID, descriptor/chunk progress, `remaining_nnz`, `valid_count`, three operand slots, request tracker, flush/error state, and a direct `PIMBlock*` datapath (`src/csc/CSCDescriptorEngine.h:55-118`). It prepares X, VALUE and INDEX requests, waits for all slots, calls masked SIMD directly, and emits partials (`src/csc/CSCDescriptorEngine.cpp:25-37`).

`CSCNativeExecution` owns 64 separately allocated logical PIM blocks and engines, global request-ID/generation state, request-to-engine routing maps, and partial results (`src/csc/CSCDescriptorEngine.h:140-200`). These logical BG datapaths are not the eight physical `PIMRank::pimBlocks`. They bypass rank mode/toggle qualification, CRF/PC/loop/repeat, rank broadcast, physical block selection, and command-bus arbitration. M5 nevertheless models DRAM admission/completion, while `CSCRequestTracker` owns the CREATED-to-RETIRED lifecycle and generation checks (`src/csc/CSCRequestTracker.h:20-37,60-109`).

## 4. State ownership matrix

| State | Current owner/scope | Update/reset evidence | M6 scope |
|---|---|---|---|
| CRF contents | `PIMRank`, rank-wide | `PIMRank.cpp:190-195`; `PIMRank.h:90-98` | RANK_SHARED |
| `pimPC` | `PIMRank`, rank-wide | `PIMRank.cpp:101-106,335-420` | BG_LOCAL |
| loop counter | `PIMRank`, rank-wide | `PIMRank.h:66`; `PIMRank.cpp:350-364` | BG_LOCAL |
| repeat counter | `PIMRank`, rank-wide | `PIMRank.h:66`; `PIMRank.cpp:368-398` | BG_LOCAL |
| mode state | `Rank` plus `PIMRank` | `Rank.h:98`; `PIMRank.cpp:96-113` | UNRESOLVED |
| toggle state | `PIMRank`, rank-wide | `PIMRank.h:67`; `PIMRank.cpp:97-140` | BG_LOCAL |
| exit state | `PIMRank`, rank-wide | `PIMRank.cpp:106,345-348,417-419` | BG_LOCAL |
| operand registers | each `PIMBlock` | `PIMBlock.h:28-47`; `PIMRank.cpp:181-187` | PIM_BLOCK_LOCAL |
| SRF | per block, broadcast writes | `PIMBlock.h:28-47`; `PIMRank.cpp:196-201` | PIM_BLOCK_LOCAL |
| GRF | per block, broadcast writes | `PIMBlock.h:28-47`; `PIMRank.cpp:164-187` | PIM_BLOCK_LOCAL |
| PIMBlock state | physical rank vector/logical CSC block | `PIMRank.h:117-118`; `CSCDescriptorEngine.h:105,185` | PIM_BLOCK_LOCAL |
| descriptor queue/image | `CSCDescriptorEngine` view | `CSCDescriptorEngine.h:57,110` | BG_LOCAL |
| descriptor index | `CSCDescriptorEngine` | `CSCDescriptorEngine.h:98`; `CSCDescriptorEngine.cpp:30-37` | BG_LOCAL |
| `remaining_nnz` | `CSCDescriptorEngine` | `CSCDescriptorEngine.h:98`; `CSCDescriptorEngine.cpp:31,36-37` | BG_LOCAL |
| `valid_count` | `CSCDescriptorEngine` | `CSCDescriptorEngine.h:99`; `CSCDescriptorEngine.cpp:36` | BG_LOCAL |
| X/VALUE/INDEX slots | `CSCDescriptorEngine` | `CSCDescriptorEngine.h:83-87,112`; `CSCDescriptorEngine.cpp:25-28` | BG_LOCAL |
| request tracker | one per engine | `CSCDescriptorEngine.h:113`; `CSCRequestTracker.h:60-109` | BG_LOCAL |
| request generation | execution factory/tracker | `CSCDescriptorEngine.h:190-191`; `CSCRequestTracker.h:79-83,102` | REQUEST_LOCAL |
| partial result | execution vector/engine sink | `CSCDescriptorEngine.h:111,161-162,187` | BG_LOCAL |
| flush/error | engine-local plus aggregate | `CSCDescriptorEngine.h:68-74,114-117,148-157,194-197` | BG_LOCAL |
| command issue | packet rank path/CSC FSM | `Rank.cpp:376-399`; `CSCDescriptorEngine.h:114` | BG_LOCAL |
| command bus | controller/rank buses, no BG arbiter | `MemoryController.cpp:477-561`; `Rank.cpp:478-516` | UNRESOLVED |
| bank/BG resource | per rank/bank, BG decoded from bank | `MemoryController.h:92`; `PIMRank.cpp:339-340` | BANK_LOCAL |

## 5. Physical BG/PIMBlock/resource mapping

The physical topology exposes 4 BGs, 16 banks and 8 PIM blocks per rank. Address mapping derives BG from bank (`PIMRank.cpp:339-340`), while operands use block index and packet bank. No authoritative `global_bg_id -> {channel, rank, BG, PIMBlock}` table exists.

M5 is logical: `global_bg_id / 4` selects a channel and every global BG owns a logical engine/block (`CSCDescriptorEngine.cpp:25-28`, `CSCDescriptorEngine.h:185-190`). M6 must decide whether both blocks associated with a BG execute together, whether two BGs contend for one block, how bank parity/BG bits select targets, and whether logical blocks remain proxies. These are **UNRESOLVED**.

## 6. Shared-resource and command-bus conflicts

`MultiChannelMemorySystem::actual_update()` ticks channels in index order (`src/MultiChannelMemorySystem.cpp:386-412`). A controller sends one command packet at a time and a rank handles data returns (`src/MemoryController.cpp:477-561`, `src/Rank.cpp:478-516`). There is no scheduler for multiple BG contexts.

Independent progress needs deterministic arbitration at the shared resource, not in the request tracker: round-robin over ready BGs per channel/rank/resource, no port overbooking, stable retry under backpressure, and bounded service. DRAM completion only makes the same BG/generation's operand ready; it cannot advance another BG.

Reset, flush and error must be target-BG scoped. Accepted requests either drain to their original `(BG, generation)` or are explicitly diagnosed; one BG cannot clear another BG's tracker, registers, progress, or partials.

## 7. Candidate architecture boundaries

| Criterion | A: BG context + shared CRF | B: BG CRF + context | C: CSC context + targeted port |
|---|---|---|---|
| legacy compatibility | Medium | Low | High |
| implementation complexity | High | Very high | Medium |
| state duplication | context only | CRF and context | existing CSC state plus arbiter |
| command-bus arbitration | required | required | required at port |
| physical mapping risk | high | high | isolated behind port |
| BG independence | high | highest | high for CSC |
| flush/reset isolation | new rules | natural but broad | engine-local plus port drain |
| testability | medium | medium | high via M5 fixtures |
| M5 tracker reuse | possible | possible | direct |
| M6 implementation scope | broad refactor | largest | smallest |
| M7 extensibility | good | maximal | good if port is generic |

Candidate A fits a future requirement for arbitrary shared CRF programs. Candidate B is not justified by M6 and risks diverging legacy CRF loading. Candidate C retains the proven descriptor/request owner and exposes only the missing physical operation.

## 8. Recommended M6 boundary

Recommend **Candidate C**:

1. Keep descriptor FSM, tracker, tokens, generation, and partials in `CSCDescriptorEngine`/`CSCNativeExecution`.
2. Add an immutable target resolving global BG to channel/rank/BG/physical resources.
3. Add a narrow targeted masked-primitive request/result port at `PIMRank`.
4. Arbitrate targeted requests per shared command/PIMBlock resource.
5. Preserve the rank-wide CRF path byte-for-byte for legacy mode.

This does not require BG-local CRF initially. If later work requires programmed CRF execution, Candidate A can add BG-local PC/context behind a distinct mode.

Legacy SB/HAB/HAB_PIM remains rank-wide. Whether targeted CSC may coexist with legacy HAB_PIM on one rank is **UNRESOLVED** and should initially be rejected rather than implicitly interleaved.

The preferred tick point is after controller/rank delivery makes completions visible and before the next issue decision. `PIMRank::update()` is plausible because it is empty, but exact placement is **UNRESOLVED** until an update-order test specifies visibility.

## 9. Required source changes for M6

### Must change

| File | Class/function | Expected change/new state | Invariant/test | Risk |
|---|---|---|---|---|
| `src/BusPacket.h` or new header | target type | immutable channel/rank/BG/resource target | target preserves BG identity | MEDIUM |
| `src/PIMRank.h/.cpp` | targeted primitive port | dispatch masked op to resolved blocks | legacy `doPIM()` unchanged | HIGH |
| `src/Rank.cpp` | update/dispatch | service targeted port; arbitration cursor | deterministic/starvation-free | HIGH |
| `src/csc/CSCDescriptorEngine.h/.cpp` | SIMD issue state | target request instead of direct block call | advance only after result | HIGH |
| same, `CSCNativeExecution` | physical binding | authoritative 64-BG binding table | one mapping source | HIGH |
| `src/MemorySystem.cpp`, `src/MultiChannelMemorySystem.cpp` | construction/update | connect rank port; define tick order | DRAM timing unchanged | MEDIUM |

### Prefer not to change

- `RequestToken`, `CSCRequestTracker`, and MemoryController map/legacy lookup.
- CSC image/descriptor format and masked SIMD arithmetic.
- Legacy rank-wide CRF load/decode and transaction/command queue policy.

### Unresolved

- exact BG-to-PIMBlock/bank-pair binding;
- targeted/legacy HAB_PIM coexistence;
- arbiter ownership in `Rank`, `PIMRank`, or channel-local object;
- explicit command-bus reservation versus equivalent timing reservation;
- whether M7 requires arbitrary per-BG CRF programs.

## 10. Invariants that M6 must preserve

- Request ID, full token, BG, worker and generation remain bound through one retirement.
- Backpressure does not allocate a new token or count an unaccepted request outstanding.
- Operand readiness, progress, partials, flush, reset and error are BG-local.
- A BG executes only when its descriptor generation's operands are ready.
- Physical PIMBlock and command-bus capacity are not double-booked.
- Arbitration is deterministic and starvation-free.
- Masked SIMD/invalid-lane behavior remains unchanged.
- Legacy CRF broadcast, mode transitions, callbacks, DRAM timing, and both storage builds remain unchanged.

## 11. Required M6 tests

| Test | Setup/event sequence | Expected result | Failure signature |
|---|---|---|---|
| `TwoBGsHoldDifferentExecutionStates` | different descriptors; stall one operand | distinct progress | cross-write/equalized state |
| `TwoBGsAdvanceIndependently` | complete A before B | A runs, B waits | lockstep/early B |
| `BGTargetDoesNotAffectOtherBG` | seed registers; target A | only A changes | B changes |
| `SharedPIMBlockContentionIsDeterministic` | two BGs request one resource | fixed winner, loser ready | double/nondeterministic issue |
| `CommandBusArbitrationIsStarvationFree` | continuous contenders | bounded service | bound exceeded |
| `OneBGFlushDoesNotDrainOtherBG` | flush A while B has reads | B completes | B cleared |
| `OneBGErrorDoesNotCorruptOtherBG` | malformed A descriptor | A errors, B correct | global corruption |
| `OutstandingRequestsRemainBoundToBGGeneration` | relaunch A around delayed completion | stale A diagnosed, B unaffected | route to new generation |
| `LegacyRankWideCRFBehaviorUnchanged` | existing MUL/ADD program | identical all-block result | broadcast changed |
| `M5RequestIdentityRegression` | reverse same-address completion | correct exactly-once owner | address misroute |
| `M5ActualBackpressureRegression` | saturate real queue | same token retries/completes | regeneration/drop |
| `NormalStorageAndNoStorageRegression` | repeat both builds | same logical result | build divergence |

## 12. Explicit non-goals

This preparation does not implement BG-targeted commands, BG-local CRF/PC, M6 execution semantics, descriptor-engine migration into `PIMRank`, command-bus arbitration, next-descriptor prefetch, CSC format changes, or queue-policy changes. It does not guess unresolved physical mapping.
