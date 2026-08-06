# FP16 M4.5 native execution and DRAM timing

## Scope

M4.5 connects the validated M3/M4 compute engine to the same DRAMSim transaction
and Scheme8 timing machinery used by FP32 native execution:

```text
FP16 image v2 -> verified execution context
-> CSCFp16NativeExecution -> DRAMSim read/arbitration/token completion
-> image-backed 32-byte payload -> CSCFp16DescriptorEngine
-> PIMBlock(FP16) MUL -> bounded partial capture
```

The milestone stops at accepted FP16 partial events. It contains no FP16 BGA,
ADD/merge, serialized partial transport, readback, host reduction or final y.

## FP32 native-path audit

`CSCNativeExecution` in `CSCDescriptorEngine.cpp` constructs the 16-channel
DRAMSim memory system, uses `PIMAddrManager::addrGenSafe` for Scheme8 addresses,
submits tokenized reads with `MultiChannelMemorySystem::addTransaction`, and
registers token callbacks. An engine creates X/value/index requests together
for one active chunk and attempts at most one request per engine tick. Only an
accepted transaction enters the outstanding maps. Token completion identifies
the engine/request independently of return order.

Its tick order is:

```text
memory.update() and callbacks
-> native cycle increment
-> partial-result path service
-> descriptor-engine ticks
-> BGA/output service and termination checks
```

Descriptors are validated local-control metadata rather than DRAM reads. The
FP16 controller preserves that policy. FP32 BGA backpressure is serviced after
engine ticks; M4.5 substitutes only the bounded FP16 capture port.

## Explicit FP16 native mode

`CSCFp16NativeExecution` is a separate compute-only production controller. This
avoids turning the mature FP32 `CSCNativeExecution`/BGA/transport path into a
large runtime-generic framework. Its constructor accepts only a
`CSCFp16ExecutionImage`, which can only be created through full v2 validation,
and creates one FP16 PIMBlock, descriptor engine and bounded sink per BG.

DRAM timing uses the established `system_hbm_csc_fp32.ini` topology identity
(16 channels, one rank, four BGs, Scheme8); arithmetic precision is an explicit
FP16 controller/PIMBlock property rather than a mutable global precision flag.
No FP32 data or BGA object is reused as an FP16 fallback.

## Physical streams and Scheme8

The existing FP32 placement is retained:

| Typed stream | local bank | base row |
|---|---:|---:|
| VALUE | 0 | 0 |
| INDEX_LOW/HIGH | 1 | 4096 |
| X | 2 | 8192 |
| descriptor metadata | local verified control | not requested |

The physical address combines channel=`global_bg/4`, rank 0,
bank-group=`global_bg%4`, the table's bank/base row and
`stream_offset/32`. `addrGenSafe` carries excess columns into rows. Every typed
request must be 32-byte aligned. Tests decode every completed address with
`AddrMapping` and verify channel, rank and BG ownership.

FP16 stream sizes and offsets come from the verified v2 image. X is logically
replicated at each requesting BG for timing while its payload comes from the
validated global original-column x image. Address boundaries are fixed for
columns 15/16 and for the independent value/index offsets of NNZ 17 and 33.

## Request adapter and outstanding table

The adapter converts `X`, `VALUE`, `INDEX_LOW`, and `INDEX_HIGH` to 32-byte
DRAM reads. Low/high share DRAMSim's INDEX token kind, while their unique typed
request remains in the outstanding entry. An entry contains request id, kind,
BG, descriptor/chunk identity, stream offset, physical address, first attempt
and acceptance cycles.

The issue policy remains X, VALUE, INDEX_LOW, optional INDEX_HIGH, with at most
one attempt per engine tick. Rejection does not create an outstanding entry and
the engine retries the identical request. No chunk prefetch, descriptor
interleaving, coalescing or issue-width increase was added.

Token completion checks request id, DRAM token kind, channel and BG, recomputes
the expected physical address, and only then obtains a 32-byte payload from the
validated image. DRAMSim supplies acceptance/arbitration/completion timing;
image-backed storage supplies deterministic data because DRAMSim does not
return file payload bytes. The payload is invisible to the descriptor engine
until the completion callback.

## Native tick, compute and capture

M4.5 preserves FP32 causal ordering: completions delivered by `memory.update()`
can make operands ready, but the engine state advances only in the engine tick
after the native cycle increment. The engine still uses one active chunk and
one FP16 PIMBlock. MUL calls the simulator's masked 16-lane FP16 operation.

The bounded capture sink is wired directly to the engine's ready/accept port.
When disabled or full, the current lane remains pending; no later chunk issues
until every valid event is accepted. The native controller completes only when
all engines are DONE and the accepted-request table is empty. Engine DONE
already implies no pending retry, MUL or partial event.

## Counters and deterministic golden

Native timing records first/last descriptor, request attempt/accept/response,
complete operands, PIM MUL, generated/accepted partial and final completion
cycles. It also records attempts, rejection/retry, kind-specific
accept/completion, outstanding high-water, response-latency sum/min/max,
operand waits, request stalls, PIM cycles and sink stalls.

For the checked 16ch/1rank/4BG/Scheme8 boundary fixture:

```text
NNZ / partial records        169 / 169
descriptors / chunks          10 / 15
X / value requests            10 / 15
index low / high              15 / 10
accepted / completed reads    50 / 50
outstanding high-water         8
operand-wait cycles          710 (sum across 64 BG engines)
compute-only cycles          544
partial trace FNV-1a-64      2e2867563f3d9cac
```

Synthetic and native paths compare every raw product bit and complete event
identity in BG/descriptor/chunk/lane order before timing is accepted.

## Limitations and M5 boundary

Descriptors remain ideal/local, matching current FP32 native execution. The
topology system file still names FP32 because global simulator configuration is
process-wide; FP16 arithmetic is locally and explicitly enforced by the new
controller. This M4.5 timing therefore covers operand memory requests through
partial capture, not descriptor fetch, BGA or result traffic.

M5 must replace the capture sink with an FP16 BGA implementing the same
ready/accept and exactly-once guarantees, then define FP16 ADD rounding,
capacity, merge/eviction/drain order and timing. No performance conclusion or
full-workload evaluation is made at M4.5.
