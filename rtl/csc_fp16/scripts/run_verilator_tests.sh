#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
RTL_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
VERILATOR_BIN=${VERILATOR_BIN:-$(command -v verilator || true)}

if [[ -z $VERILATOR_BIN || ! -x $VERILATOR_BIN ]]; then
  echo "error: Verilator executable not found: $VERILATOR_BIN" >&2
  exit 2
fi

BUILD_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/csc-fp16-verilator.XXXXXX")
trap 'rm -rf -- "$BUILD_ROOT"' EXIT

run_test() {
  local top=$1
  shift
  local obj_dir=$BUILD_ROOT/$top
  "$VERILATOR_BIN" --binary --timing -Wall -Wno-fatal \
    --top-module "$top" --Mdir "$obj_dir" "$@"
  "$obj_dir/V$top"
}

run_test tb_csc_descriptor_engine \
  "$RTL_DIR/csc_descriptor_engine.sv" \
  "$RTL_DIR/tb/tb_csc_descriptor_engine.sv"

run_test tb_csc_bga \
  "$RTL_DIR/csc_batch_ingress.sv" \
  "$RTL_DIR/csc_bga.sv" \
  "$RTL_DIR/tb/tb_csc_bga.sv"

run_test tb_csc_burst_packer \
  "$RTL_DIR/csc_burst_packer.sv" \
  "$RTL_DIR/tb/tb_csc_burst_packer.sv"

run_test tb_csc_bg_engine \
  "$RTL_DIR/csc_descriptor_engine.sv" \
  "$RTL_DIR/csc_agu.sv" \
  "$RTL_DIR/csc_request_tracker.sv" \
  "$RTL_DIR/csc_partial_event_gen.sv" \
  "$RTL_DIR/csc_batch_ingress.sv" \
  "$RTL_DIR/csc_bga.sv" \
  "$RTL_DIR/csc_result_serializer.sv" \
  "$RTL_DIR/csc_burst_packer.sv" \
  "$RTL_DIR/csc_result_transport_ctrl.sv" \
  "$RTL_DIR/csc_control_top.sv" \
  "$RTL_DIR/csc_transport_top.sv" \
  "$RTL_DIR/csc_bg_engine.sv" \
  "$RTL_DIR/tb/tb_csc_bg_engine.sv"

echo "All CSC FP16 RTL tests passed."
