#!/usr/bin/env bash
set -euo pipefail

PRESET="${1:-q64}"
TARGET_PERIOD_NS="${TARGET_PERIOD_NS:-1.0}"

case "$PRESET" in
  q64) SCRIPT=rtl/csc_fp16/synth/synth_added_hw_q64.ys ;;
  q32) SCRIPT=rtl/csc_fp16/synth/synth_q32.ys ;;
  q16) SCRIPT=rtl/csc_fp16/synth/synth_q16.ys ;;
  bga-q64) SCRIPT=rtl/csc_fp16/synth/synth_bga_q64.ys ;;
  control) SCRIPT=rtl/csc_fp16/synth/synth_control.ys ;;
  transport) SCRIPT=rtl/csc_fp16/synth/synth_transport.ys ;;
  *) echo "unknown preset: $PRESET" >&2; exit 2 ;;
esac

command -v yosys >/dev/null || { echo "yosys is not installed" >&2; exit 127; }

if [[ -z "${STD_CELL_LIB:-}" ]]; then
  echo "Generic synthesis: $SCRIPT"
  exec yosys -s "$SCRIPT"
fi

[[ -r "$STD_CELL_LIB" ]] || { echo "cannot read STD_CELL_LIB=$STD_CELL_LIB" >&2; exit 2; }

# ABC accepts the delay target in picoseconds. The conversion is performed by
# awk to avoid requiring Python or a project-specific environment.
TARGET_PERIOD_PS="$(awk -v ns="$TARGET_PERIOD_NS" 'BEGIN { printf "%d", ns * 1000.0 + 0.5 }')"

echo "Characterized mapping requires a synthesized design from $SCRIPT"
echo "Library: $STD_CELL_LIB"
echo "ABC delay target: ${TARGET_PERIOD_PS} ps"

# Run the selected structural synthesis first, then map sequential and
# combinational cells to the supplied library. This reports mapped area; a
# signoff STA tool is still required for authoritative slack/Fmax.
yosys -p "script $SCRIPT; dfflibmap -liberty $STD_CELL_LIB; abc -liberty $STD_CELL_LIB -D $TARGET_PERIOD_PS; clean; stat -liberty $STD_CELL_LIB"
