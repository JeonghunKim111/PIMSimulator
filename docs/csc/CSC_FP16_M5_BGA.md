# FP16 M5 bank-group accumulator

## Scope

M5 replaces the M4.5 compute capture boundary with one FP16 BGA per global BG:

```text
v2 image -> DRAMSim -> FP16 descriptor/PIM MUL -> FP16 BGA
         -> bounded BGA-output capture
```

The output is an internal event. There is no serialized partial record,
writeback, readback, host reduction, final-y or cross-BG merge.

## FP32 source audit

| Property | FP32 implementation | FP16 preservation/change |
| --- | --- | --- |
| Input | atomic batch, committed to FIFO at `step()` end | event ready/valid, committed at `step()` end |
| Service | one FIFO entry starts lookup at a time | preserved |
| Lookup | scans all valid non-reserved entries | preserved |
| Latency | explicit compare then ADD latency | preserved |
| ADD | volatile FP32 add | `cscFp16Add`, immediate binary16 materialization |
| Capacity | configured entry count | iso-entry-count; default 16 |
| Victim | oldest insertion age, lower slot tie break | preserved |
| Full miss | emit, wait for retirement, then insert | preserved |
| Descriptor boundary | no flush | preserved |
| Final drain | producer done + explicit request; oldest first | preserved |
| Output | stable until accepted and next-step retirement | preserved |

M5 originally retained the M4 serial event boundary. M5.1 preserves it as the
Q16 compatibility mode and adds an atomic FP32-equivalent batch8 adapter. The
production preset is batch8/Q16; lookup/ADD service remains one entry.

## FP16 arithmetic and ordering

`CSCFp16BankGroupAccumulator` stores raw binary16 accumulator bits. Every hit
performs `cscFp16Add` and immediately writes its binary16 result. There is no
FP32 accumulator, multiplication or FMA. Signed zero, subnormal, overflow,
infinity and NaN follow the shared M1 half implementation.

The phase order mirrors FP32: output retirement, operation completion,
eviction/drain update, service start, ingress commit. A full miss cannot reuse
the oldest insertion-age victim until its output is accepted and retired.
There is no descriptor flush. Producer DONE triggers oldest-first final drain.

## Native integration and completion

```text
DRAMSim update/completion
-> global cycle increment
-> BGA step
-> descriptor/PIM tick and BGA ingress handshake
-> BGA-output sink handshake
-> completion checks
```

Completion requires all engines done, no DRAM requests, no pending compute
partial, all BGA queues/operations empty and every output accepted. Output
blockage propagates to the descriptor engine without dropping or duplicating
the current partial.

`CSCFp16BGAOutputEvent` contains row, raw FP16 bits, global BG,
capacity/final reason, BG-local sequence and contribution count. It is not the
future M6 8-byte record.

## Counters and conservation

`compute_complete_cycle` and `compute_bga_completion_cycle` are global elapsed
milestones. Operand wait, descriptor ingress stall and BGA busy counters summed
over 64 BGs are aggregate engine/BGA-cycle counts.

Each BGA step checks:

```text
accepted = live contributions + retired contributions
generated partials = accepted BGA ingress
outputs = capacity evictions + final-drain outputs
```

No mathematical-value conservation is claimed because merges round to FP16.

## Legacy stress golden validation

An independent replay implements queue hit, FP16 ADD, FIFO victim selection and
oldest-first drain without sharing production state transitions. With eight
entries per BG, the M4.5 boundary fixture produces:

```text
generated/accepted partials  169 / 169
FP16 merges                  2
capacity evictions           151
final-drain outputs          16
total outputs                167
compute complete cycle       544
compute+BGA cycle            556
partial FNV-1a-64            2e2867563f3d9cac
BGA-output FNV-1a-64         935f4049c56e632f
```

This eight-entry result is the batch8/Q8 stress golden, not production. The
batch8/Q16 production golden is documented in
`CSC_FP16_M5_1_INGRESS_AND_PRODUCTION_CONFIG.md`.

The two active BGs remain independent; equal row indices across BGs never
merge.

## M6 boundary and limitations

M6 replaces `CSCFp16BoundedBGAOutputSink` with a transport consumer using the
same stable ready/accept contract. It must define the 8-byte indexed record,
packing, writeback/readback and ordered host FP16 reduction.

Descriptor fetch remains ideal/local. The output sink is a milestone capture,
not a DRAM transport model. No full-workload performance claim is made.
