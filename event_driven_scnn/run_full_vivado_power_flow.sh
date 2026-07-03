#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$ROOT_DIR/event_driven_scnn"
HLS_DIR="$WORK_DIR/hls"
RTL_DIR="$HLS_DIR/impl/verilog"
VIVADO_WORK="$HLS_DIR/impl/vivado_power_work"
VIVADO_PROJECT="vivado_power"
VIVADO_XPR="$VIVADO_WORK/$VIVADO_PROJECT.xpr"
SYNTH_DCP="$VIVADO_WORK/$VIVADO_PROJECT.runs/synth_1/scnn.dcp"
SYNTH_RTL_HASH="$VIVADO_WORK/synth_rtl.sha256"
FORCE_SYNTH="${FORCE_SYNTH:-0}"

hash_rtl() {
  local files=()
  if [[ ! -d "$RTL_DIR" ]]; then
    return 1
  fi

  mapfile -t files < <(find "$RTL_DIR" -type f \( -name '*.v' -o -name '*.sv' -o -name '*.vh' -o -name '*.svh' -o -name '*.dat' \) | LC_ALL=C sort)
  if (( ${#files[@]} == 0 )); then
    return 1
  fi

  sha256sum "${files[@]}" | sha256sum | awk '{print $1}'
}

echo "[1/4] Running Cosim Setup via Vitis (testbench generation only)..."
nice -n 15 vitis-run --mode hls --cosim --config "$ROOT_DIR/hls_config.cfg" --work_dir event_driven_scnn --hls.cosim.setup 1

echo "[2/4] Exporting component to Vivado..."
nice -n 15 vitis-run --mode hls --package --config "$ROOT_DIR/hls_config.cfg" --work_dir event_driven_scnn

CURRENT_RTL_HASH="$(hash_rtl)"
VIVADO_ARGS=()
if [[ "$FORCE_SYNTH" != "0" ]]; then
  echo "[3/4] Running Vivado synthesis and generating SAIF (FORCE_SYNTH=$FORCE_SYNTH)..."
elif [[ -f "$VIVADO_XPR" && -f "$SYNTH_DCP" && -f "$SYNTH_RTL_HASH" && "$(cat "$SYNTH_RTL_HASH")" == "$CURRENT_RTL_HASH" ]]; then
  echo "[3/4] Reusing existing Vivado synthesis; regenerating SAIF only..."
  VIVADO_ARGS=(-tclargs sim_only)
else
  echo "[3/4] Running Vivado synthesis and generating SAIF..."
fi

nice -n 15 vivado -mode batch -source "$ROOT_DIR/02_run_vivado_post_synth.tcl" "${VIVADO_ARGS[@]}"

if (( ${#VIVADO_ARGS[@]} == 0 )); then
  mkdir -p "$VIVADO_WORK"
  printf '%s\n' "$CURRENT_RTL_HASH" > "$SYNTH_RTL_HASH"
fi

echo "[4/4] Generating Vivado power report from checkpoint and SAIF..."
nice -n 15 vivado -mode batch -source "$ROOT_DIR/03_power_report_from_saif.tcl"

echo "Flow completed."
