# CSC FP16 additional-hardware area estimation

## Scope and provenance

This report estimates digital support logic added for CSC descriptor processing, address generation, request tracking, row-indexed partial-sum management and result transport. Existing FP16 PIM multiplier/adder, DRAM/SRAM macros, result memory, PHY, host reduction and physical interconnect are excluded.

Tools and hashes are recorded under `provenance/`. The historical synthesis
corner is the typical NanGate 15 nm NLDM Liberty identified by SHA-256 in
`provenance/liberty_sha256.txt`; the library is not redistributed.

The applied primary preset is `BATCH_WIDTH=8`, `INPUT_DEPTH=64`, `ACC_ENTRIES=64`, `COMPARE_W=64`, `OUTPUT_DEPTH=64`; nested defaults include 16 FP16 lanes, 32-bit rows, 16-bit values, 16 request entries and 8-bit tags.

Reference flat run:

- directory: `rtl/csc_fp16/results/nangate15/20260819_151723`
- netlist: `csc_added_hw_top_mapped.v`
- report: `csc_added_hw_top_stat.rpt`
- log: `csc_added_hw_top_synth.log`
- 76,577 cells; 34,647.588864 area; 13,447 FFs; 17,184.620544 sequential area.

## Main conclusions

1. **Largest block:** hierarchy-preserved `csc_bga` contributes 26,370.539520 units, 76.11% of the flat top. Its Q64 associative tags, ranks, queues, partial sums, wide muxes and comparison/priority logic dominate.
2. **FF/storage fraction:** all FFs occupy 17,184.620544 units (49.60%). Explicit BGA/request/transport arrays account for at least 13,929.676800 units (40.20%) as direct FF area, excluding their mux/decoder/compare logic.
3. **Per-BG replicated area:** the documented current RTL is one complete `csc_bg_engine` per BG, so the supported estimate is 34,647.588864 units/BG. Storage is included.
4. **Shareable area:** no shared block is implemented or unambiguously specified. A 3,587.457024-unit result-transport block is a plausible channel/rank-shared candidate, but this **needs architecture confirmation** and omits required arbitration/interconnect.
5. **64 BG:** simple/documented upper bound is 2,217,445.687296 units. The non-implemented transport-shared-once sensitivity is 1,991,435.894784 units.
6. **Excluded costs:** FP16 arithmetic, PIM/DRAM/SRAM macros, row buffers, register-file macros, memory controller arbitration, cross-BG/channel interconnect, clock tree, routing, placement whitespace, power grid, pads/PHY and physical timing closure.
7. **Next inputs needed:** exact BG/channel/rank ownership of control and transport, number of sharing domains, arbitration topology, real SRAM/register-file macro views, NanGate LEF/site interpretation, clock/generated-clock definitions, I/O delays, loads/drives, exceptions and physical parasitics.

## Correctness and consistency checks

- All ten individual-top syntheses completed and produced non-empty mapped netlists.
- No design netlist contains a `$_..._` primitive or non-parameterized `$...` primitive.
- `$paramod...` hierarchy module names are elaborated RTL modules, not unmapped cells.
- No inferred latch or mapped latch cell remains.
- No module was optimized away. The full `csc_added_hw_top` has zero constant output bits. Isolated AGU/transport tops expose expected constant upper address bits from their configured base/counter widths; details are in `module_area.md` and `module_area.csv`.
- Individual CSV rows agree with their `stat.rpt` totals. The leaf-module area sum differs from flat top by 2.5096%, explained in `module_area.md`.
- Yosys reported 13 unique frontend warnings, all register-list conversion warnings; no errors/fatals occurred.

## Physical and timing status

All figures are Liberty area units. NanGate LEF is absent, so **LEF cross-check required** before claiming square micrometres. Utilization/margin sensitivity is in `scaling_model.md` and is heuristic, not place-and-route.

No OpenSTA result was produced because the real SDC is not fixed. `timing_sweep/` provides a guarded exploratory synthesis/STA sweep for 2.0, 1.25, 1.0 and 0.8 ns; STA runs only when reviewed period-specific SDC files are explicitly provided.

The large generated mapped netlists and logs are intentionally omitted from
Git. The retained `stat` reports, runtimes, summaries and provenance are the
audit record; use `scripts/run_synth.sh` and `scripts/run_module_synth.sh` to
regenerate full outputs with a matching Liberty file.
