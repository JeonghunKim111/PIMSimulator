#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
RTL_DIR=$PROJECT_DIR

YOSYS_BIN=${YOSYS_BIN:-$(command -v yosys || true)}
TOP=${TOP:-csc_added_hw_top}
LIB_TYP=${STD_CELL_LIB:-}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-$PROJECT_DIR/results/nangate15/$RUN_TAG}

for path in "$PROJECT_DIR" "$RTL_DIR" "$OUT_DIR"; do
  if [[ $path == *[[:space:]]* ]]; then
    echo "error: whitespace in path is not supported by this Yosys template: $path" >&2
    exit 2
  fi
done

if [[ -z $YOSYS_BIN || ! -x $YOSYS_BIN ]]; then
  echo "error: Yosys executable not found: $YOSYS_BIN" >&2
  exit 2
fi
if [[ -z $LIB_TYP || ! -f $LIB_TYP ]]; then
  echo "error: set STD_CELL_LIB to a readable Liberty file" >&2
  echo "The NanGate file is intentionally not redistributed." >&2
  exit 2
fi
if [[ -e $OUT_DIR ]]; then
  echo "error: output directory already exists; choose a new RUN_TAG or OUT_DIR: $OUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
GENERATED_YS=$OUT_DIR/synth_resolved.ys
NETLIST=$OUT_DIR/${TOP}_mapped.v
STAT_REPORT=$OUT_DIR/${TOP}_stat.rpt
LOG=$OUT_DIR/${TOP}_synth.log

sed \
  -e "s|@RTL_DIR@|$RTL_DIR|g" \
  -e "s|@TOP@|$TOP|g" \
  -e "s|@LIB_TYP@|$LIB_TYP|g" \
  -e "s|@STAT_REPORT@|$STAT_REPORT|g" \
  -e "s|@NETLIST@|$NETLIST|g" \
  "$SCRIPT_DIR/synth_nangate15.ys" > "$GENERATED_YS"

"$YOSYS_BIN" -l "$LOG" "$GENERATED_YS"
echo "mapped netlist: $NETLIST"
echo "statistics:     $STAT_REPORT"
echo "log:            $LOG"
