# SparsePIM V3 Structural Model

This document records the first structural replacement model for the
DRAF+BGA-aware SparsePIM analytical model. It is intentionally kept separate
from the existing v1/v2/v2.1 correction models.

## Goal

The v3 model should explain SparsePIM latency from architectural work terms
rather than fitting coefficients to the paper speedups. The paper target
speedups in `SPARSEPIM_WORKLOAD_TARGETS.md` are used only as validation output.

## Current V3 Components

The current v3 total latency is:

```text
T_v3 =
  T_setup
+ T_draf_row_fetch
+ T_pim_enable
+ T_draf_compute_trigger
+ T_padded_zero_compute
+ T_bga_accumulate
+ T_pim_disable
+ T_pim_to_sb
+ T_bga_output_readback
+ T_final_reduce
```

The first structural change is:

```text
T_padded_zero_compute = ceil(DRAF padded NZE slots / 16)
```

This follows the SparsePIM paper's explanation that padded zero elements in
DRAF increase operation count and memory latency because SparsePIM performs
synchronous column access across banks.

## First-Pass Findings

The padded-zero work term strongly improves the previously overestimated sparse
graph/web workloads:

```text
Stanford:   v3 speedup 1.23x, paper target 1.3x
webbase-1M: v3 speedup 0.75x, paper target 0.7x
```

This suggests that the original model undercounted DRAF padded work for highly
sparse, low-NNZ-per-column workloads.

However, the first-pass v3 model over-penalizes many denser or more regular
workloads, such as `pdb1HYS`, `xenon2`, `crankseg_2`, and `pwtk`. The next
iteration should separate padded slots that truly cause synchronous useless work
from padded slots that are amortized by regular DRAF/BGA execution.

## Next Structural Questions

- Should padded-zero work be gated by low-NNZ-column or single-NNZ-column
  fragmentation instead of applying uniformly to all DRAF padding?
- Should dense/regular workloads receive a structural BGA/locality overlap term
  rather than a tuned speedup factor?
- Is the current DRAMSim-trigger cycle for DRAF row fetch/compute too pessimistic
  for dense workloads compared with the SparsePIM paper's modeled PIM pipeline?

## V4 Critical-Path Padding Model

The v4 model keeps the v3 model intact and adds a separate critical-path padding
variant. Instead of charging all padded NZE slots, it estimates padding exposed
on the synchronous bank-group execution path:

```text
actual_group_step = ceil(DRAF column groups in a logical bank group / 2 banks)
ideal_group_step  = (nonzeros in the same logical bank group / 16 NZEs) / 2 banks
critical_padding  = sum(max(0, actual_group_step - ideal_group_step))
```

The v4 padding latency term is:

```text
T_critical_padding = ceil(critical_padding)
```

This is intentionally separate from total DRAF padding:

```text
total_padding_groups = draf_nze_padding / 16
```

The v4 output adds:

```text
draf_padding_ratio
critical_padding
critical_padding_ratio
draf_memory_expansion
group_steps_mean
group_steps_max
bg_imbalance
```

First-pass v4 results:

```text
GMean target: 2.197x using rounded per-workload paper targets
GMean v3:     1.256x
GMean v4:     1.608x
```

The critical-path model improves the over-penalty from v3, especially for more
regular workloads. However, it hides too much padding for highly fragmented
workloads such as Stanford, soc-sign-epinions, and webbase-1M. A follow-up model
should combine critical padding with a fragmentation or memory-expansion exposure
term so that low-NNZ-column workloads still pay more of the DRAF padding cost.
