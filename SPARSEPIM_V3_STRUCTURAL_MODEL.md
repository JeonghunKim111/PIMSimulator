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

## V5 Exposure Padding Model

The v5 model keeps the v4 critical-path padding term and exposes a fraction of
the hidden padding when the DRAF layout is fragmented or memory-expansive:

```text
total_padding_steps   = draf_nze_padding / 16
critical_padding      = v4 critical-path padding
hidden_padding_steps  = max(0, total_padding_steps - critical_padding)

fragmentation      = max(single_nnz_column_ratio, low_nnz_column_ratio)
memory_pressure    = clamp((draf_memory_expansion - 1) / 4)
imbalance_pressure = clamp((bg_imbalance - 1) / 3)

exposure_factor =
  clamp(0.6 * fragmentation
      + 0.3 * memory_pressure
      + 0.1 * imbalance_pressure)

effective_padding_steps =
  critical_padding + hidden_padding_steps * exposure_factor
```

This is intended to keep v4's amortization for regular workloads while restoring
more of the padding cost for graph/web workloads with many low-NNZ columns.

First-pass v5 results:

```text
GMean target: 2.197x using rounded per-workload paper targets
GMean v3:     1.256x
GMean v4:     1.608x
GMean v5:     1.477x
```

The model improves the v4 overestimate for Stanford and webbase-1M:

```text
Stanford:   v4 1.86x -> v5 1.40x, paper target 1.3x
webbase-1M: v4 1.24x -> v5 0.78x, paper target 0.7x
```

However, v5 still underestimates high-speedup regular workloads such as
pdb1HYS, xenon2, crankseg_2, and pwtk. This suggests that the remaining major
gap is no longer only DRAF padding. A subsequent model should inspect whether
the DRAF row fetch, DRAF compute trigger, BGA accumulate, and final reduction
terms are being serialized too conservatively for regular high-locality
matrices.

## Phase Diagnostics

Full structural-suite runs now write a per-version result file and a phase
diagnostic file:

```text
spmv_guided_kmeans_draf_bga_v{3,4,5,6}_structural_results.txt
spmv_guided_kmeans_draf_bga_v{3,4,5,6}_phase_diagnostics.txt
```

The diagnostics report phase percentages plus several alternative totals that
are used only for analysis:

```text
compute_bga_overlap:
  setup + fetch + max(compute + padding, BGA) + readback + final + mode_switch

aggressive_overlap:
  setup + max(fetch, compute + padding, BGA) + readback + final + mode_switch

wide_final:
  serial total with final_reduce_cycle / 4

compute_bga_overlap_wide_final:
  compute_bga_overlap with final_reduce_cycle / 4
```

The first v5 diagnostic pass shows:

```text
Stanford and webbase-1M:
  padding dominates total latency, and v5 is close to the paper targets.

pdb1HYS, xenon2, crankseg_2, pwtk:
  padding exposure is low, but the model remains much slower than the paper
  target. Compute/BGA overlap improves the estimates, yet still does not reach
  the high target speedups. This points to additional over-serialization or
  over-costing in DRAF row fetch, DRAF compute trigger, BGA accumulate, or
  result reduction for regular high-locality matrices.
```

## V6 BGA Overlap Model

V6 keeps the v5 DRAF padding exposure model and splits BGA accumulation into
raw, hidden, and exposed components:

```text
raw_bga_cycle =
  max_bacc_instructions_per_group
  + conservative_flush_penalty * selected_flushes_per_group

regularity_score = 1 - max(fragmentation, memory_pressure)
reuse_score      = bga_reduction_ratio
padding_guard    = 1 - exposure_factor

bga_overlap_factor =
  clamp(0.10
      + 0.60 * regularity_score * padding_guard
      + 0.30 * regularity_score * reuse_score)

hidden_bga_cycle  = min(raw_bga_cycle * bga_overlap_factor,
                        draf_compute_trigger_cycle + padding_cycle)
exposed_bga_cycle = raw_bga_cycle - hidden_bga_cycle
```

This lets regular/high-reuse matrices hide much of the accumulator stream
behind DRAF compute slots while preserving most BGA cost for padding-dominated
graph/web workloads.

First-pass v6 results:

```text
GMean target: 2.197x using rounded per-workload paper targets
GMean v5:     1.477x
GMean v6:     1.809x
```

V6 improves regular workloads such as cant, ct20stif, lhr71, rma10, and
shipsec1, but pdb1HYS, crankseg_2, pwtk, and xenon2 are still substantially
below target even after most BGA work is hidden. This suggests the remaining
gap is likely in DRAF row fetch, compute-trigger cost, result readback/reduce,
or in the GPU baseline latency normalization rather than BGA serialization
alone.
