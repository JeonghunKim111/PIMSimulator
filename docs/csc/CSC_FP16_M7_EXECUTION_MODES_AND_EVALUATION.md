# FP16 M7 Execution Modes and Evaluation

## Modes

M7 exposes existing boundaries without new hardware:

| Mode | Boundary |
|---|---|
| `COMPUTE_ONLY` | native operand timing, 16-lane MUL, partial capture |
| `BGA_VALIDATION` | batch8/Q16 BGA plus untimed direct replay |
| `END_TO_END_TIMED` | M6 write/read/reduction and raw final y |

Mode and precision are separate. BGA modes use
`FP16_ISO_STRUCTURE_BATCH8_Q16`; serial Q16 and Q8 remain compatibility/stress.
Unavailable phase cycles are optional fields, not zero.

## Fast validation and equivalence

`reduceCapturedFp16BGAOutputs` sorts by global BG then BG-local output sequence,
rejects gaps/duplicates and applies `cscFp16Add` from positive zero for every
event. Replay has no simulated cycles. Packing preserves this order as
BG/burst/slot, so validation and full mode must have identical raw vectors and
hash `64a5f1109a6f2bd6`.

## External image and result-region preflight

`runFp16M7` accepts a verified `CSCFp16ExecutionImage`; v2 manifest, precision,
checksums and streams are therefore validated before launch. External smoke:

```bash
CSC_FP16_EXTERNAL_IMAGE=/path/to/v2/image \
  ./sim --gtest_filter=CSCFp16M7ExternalTest.OptInVerifiedV2ImageRunsAllModes
```

Per BG, preflight computes `records <= assigned_nnz` and
`result_bursts = ceil(assigned_nnz/4)` with overflow and descriptor-NNZ checks.
Packer, pending depth, issue width, latency and in-flight limits remain fixed;
matrix-sized resident capacity is logical HBM result-region allocation, not
transport bandwidth.

Operand requests use native DRAMSim addresses. Partial results retain the
existing logical resident-slot abstraction, so physical output bank conflict,
row locality and output Scheme8 mapping are not modeled.

## Schema, artifacts and determinism

`CSCFp16M7Result` holds common identity and mode-specific optionals.
`publishFp16M7Artifacts` publishes through a temporary directory without
overwrite:

```text
run_manifest.json
phase_cycles.csv
traffic.csv
bga_stats.csv
accuracy.csv
final_y_fp16.bin
oracle_summary.json
```

The binary is row-major little-endian uint16. Repeated synthetic full runs must
match JSON, cycles, hashes and bits. When the publisher is called without the
source FP64 inputs, accuracy/oracle artifacts explicitly report `unavailable`;
they never invent reference results.

## Paired evaluation and accuracy

Paired identity requires dimensions, NNZ, source, row-index, mapping and x
fingerprints. Only then is speedup computed as `FP32 cycles / FP16 cycles`.
FP32 is not reimplemented. Both partial records remain 8 bytes; expected FP16
traffic savings are matrix/x-side.

Three accuracy views are provided:

1. architectural FP16 versus original-input FP64: total error;
2. quantized-input FP64 versus original FP64: input quantization error;
3. architectural FP16 versus quantized-input FP64: arithmetic/order error.

Metrics include L2/relative L2, RMSE/normalized RMSE, maximum and mean absolute,
finite relative maximum, NaN/Inf, sign, zero and signed-zero counts. Nonfinite
rows are separated from finite norms.

## Golden and limits

```text
partial hash       2e2867563f3d9cac (544)
BGA hash           96c3f823f3fe5ab1 (564)
record hash        bcfec10599ee6444
resident hash      49383a2ab16bbf67
validation/full y  64a5f1109a6f2bd6 (full 655)
```

Descriptor-fetch timing, physical result addresses, logic-die GA, dense PIM
output, area/energy and CPU wall-clock are excluded. M7 adds no widening,
batch16, compression, overlap, prefetch or second-PIM-block scheduling.
