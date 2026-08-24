#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 MODULE" >&2
  exit 2
fi

MODULE=$1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
YOSYS_BIN=${YOSYS_BIN:-$(command -v yosys || true)}
LIB=${STD_CELL_LIB:-}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT_DIR:-$ROOT/results/nangate15/$RUN_TAG/module_runs/$MODULE}

if [[ -z $YOSYS_BIN || ! -x $YOSYS_BIN ]]; then
  echo "error: Yosys executable not found: $YOSYS_BIN" >&2
  exit 2
fi
if [[ -z $LIB || ! -f $LIB ]]; then
  echo "error: set STD_CELL_LIB to a readable Liberty file" >&2
  exit 2
fi
if [[ -e $OUT ]]; then
  echo "error: output directory already exists: $OUT" >&2
  exit 2
fi
for path in "$ROOT" "$LIB" "$OUT"; do
  if [[ $path == *[[:space:]]* ]]; then
    echo "error: whitespace in paths is not supported by this Yosys command" >&2
    exit 2
  fi
done

SOURCES="$ROOT/csc_descriptor_engine.sv $ROOT/csc_agu.sv $ROOT/csc_request_tracker.sv $ROOT/csc_partial_event_gen.sv $ROOT/csc_batch_ingress.sv $ROOT/csc_bga.sv $ROOT/csc_result_serializer.sv $ROOT/csc_burst_packer.sv $ROOT/csc_result_transport_ctrl.sv $ROOT/csc_control_top.sv $ROOT/csc_transport_top.sv $ROOT/csc_bg_engine.sv $ROOT/csc_added_hw_top.sv"

case "$MODULE" in
  csc_descriptor_engine) PARAMS="chparam -set LANES 16 $MODULE;" ;;
  csc_agu) PARAMS="chparam -set ADDR_W 64 -set TAG_W 8 -set VALUE_BASE 0 -set INDEX_BASE 131072 -set X_BASE 262144 $MODULE;" ;;
  csc_request_tracker) PARAMS="chparam -set DEPTH 16 -set TAG_W 8 $MODULE;" ;;
  csc_batch_ingress) PARAMS="chparam -set BATCH_WIDTH 8 -set ROW_W 32 -set VALUE_W 16 $MODULE;" ;;
  csc_bga) PARAMS="chparam -set ACC_ENTRIES 64 -set COMPARE_W 64 -set INPUT_DEPTH 64 -set OUTPUT_DEPTH 64 -set ROW_W 32 -set VALUE_W 16 -set BATCH_WIDTH 8 $MODULE;" ;;
  csc_partial_event_gen|csc_burst_packer|csc_result_transport_ctrl) PARAMS="" ;;
  csc_bg_engine|csc_added_hw_top) PARAMS="chparam -set BATCH_WIDTH 8 -set INPUT_DEPTH 64 -set ACC_ENTRIES 64 -set COMPARE_W 64 -set OUTPUT_DEPTH 64 $MODULE;" ;;
  *) echo "error: unsupported module: $MODULE" >&2; exit 2 ;;
esac

mkdir -p "$OUT"
CMD="read_verilog -sv $SOURCES; $PARAMS hierarchy -check -top $MODULE; proc; opt; memory; opt; techmap; opt; dfflibmap -liberty $LIB; abc -liberty $LIB; clean; tee -o $OUT/stat.rpt stat -liberty $LIB; write_verilog -noattr -noexpr $OUT/mapped.v"

SECONDS=0
"$YOSYS_BIN" -l "$OUT/synth.log" -p "$CMD"
printf '%s\n' "$SECONDS" > "$OUT/runtime_seconds.txt"
echo "statistics: $OUT/stat.rpt"
echo "mapped netlist: $OUT/mapped.v"
