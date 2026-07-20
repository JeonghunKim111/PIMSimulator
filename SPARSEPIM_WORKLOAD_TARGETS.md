# SparsePIM Workload Targets

This file records the SparsePIM paper workload mapping and target speedups used
to calibrate the DRAF+BGA analytical model.

## Latency Scope

The model target is the same latency scope as the SparsePIM paper:

- GPU baseline: host-to-device input transfer, cuSPARSE SpMV kernel execution,
  and device-to-host result retrieval per iteration.
- SparsePIM: kernel programming/init, PIM computation, and result retrieval per
  iteration.

The analytical model should therefore compare end-to-end per-iteration latency,
not only the PIM compute phase.

## Workload Mapping

| Workload | Matrix | GPU Baseline (ms) | Paper GPU Baseline Speedup |
| --- | --- | ---: | ---: |
| w1 | cant | 2.388768 | 2.2x |
| w2 | crankseg_2 | 9.807726 | 3.0x |
| w3 | lhr71 | 1.90591 | 1.8x |
| w4 | pdb1HYS | 5.907898 | 5.4x |
| w5 | rma10 | 2.682394 | 2.3x |
| w6 | soc-sign-epinions | 1.187932 | 1.2x |
| w7 | Stanford | 3.311456 | 1.3x |
| w8 | bcsstk32 | 1.1594 | 2.0x |
| w9 | consph | 3.705472 | 2.3x |
| w10 | ct20stif | 1.673094 | 2.1x |
| w11 | ohne2 | 18.24784 | 3.2x |
| w12 | pwtk | 11.52014 | 3.6x |
| w13 | shipsec1 | 4.43507 | 2.1x |
| w14 | ASIC_100k | 1.264486 | 1.4x |
| w15 | xenon2 | 11.67992 | 5.6x |
| w16 | webbase-1M | 5.887198 | 0.7x |
| GMean | Geometric Mean | - | 2.16x |

## Immediate Modeling Concern

w7/Stanford is a key outlier for the current analytical model. The paper reports
only about 1.3x speedup, while the current model predicts a much higher speedup.
This suggests that the model is likely missing an overhead that is especially
large for Stanford, or is over-crediting DRAF/BGA locality or parallelism for
that matrix.
