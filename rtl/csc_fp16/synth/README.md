# Synthesis

Run scripts from the repository root. The primary target is Q64:

```bash
yosys -s rtl/csc_fp16/synth/synth_bga_q64.ys
yosys -s rtl/csc_fp16/synth/synth_added_hw_q64.ys
```

Q32 and Q16 are sensitivity presets only. `synth_control.ys` and
`synth_transport.ys` provide the requested block-level decomposition.

The scripts perform generic synthesis and `stat`; they make no area or timing
claim. For characterized synthesis, set the environment at the invoking flow:

```bash
STD_CELL_LIB=/path/to/library.lib TARGET_PERIOD_NS=1.0 your_flow ...
```

The target flow must read `$STD_CELL_LIB`, constrain the clock to
`$TARGET_PERIOD_NS`, and report critical path, slack and Fmax. No Liberty file
is bundled or downloaded. Yosys alone cannot establish 1 ns feasibility.

`run_synthesis.sh` accepts `q64`, `q32`, `q16`, `bga-q64`, `control`, or
`transport`. Without `STD_CELL_LIB` it runs generic synthesis. With a readable
library it performs Yosys/ABC mapping using `TARGET_PERIOD_NS` (default 1.0 ns)
as the ABC delay target. Authoritative slack and Fmax still require STA.

For a preserved mapped netlist and `stat` report, use the portable scripts:

```bash
export STD_CELL_LIB=/path/to/typical.lib
rtl/csc_fp16/scripts/run_synth.sh
rtl/csc_fp16/scripts/run_module_synth.sh csc_added_hw_top
```

`YOSYS_BIN`, `RUN_TAG`, and `OUT_DIR` may be overridden. The scripts reject an
existing output directory and do not assume a server username or home path.
Use `scripts/sta_max.tcl` and `scripts/sta_min.tcl` only after supplying reviewed
`SDC`, `NETLIST`, `TOP`, and corner-specific `LIB_MAX`/`LIB_MIN` variables.
