#!/usr/bin/env bash
# An airfoil run: the free stream from Mach, Reynolds number, temperature and
# angle of attack (scripts/airfoil/freestream.py) over a case's config, steady
# (STEPS=0, ITERATIONS pseudo-time iterations) or dual time (STEPS physical
# steps of DT), optionally from a solution (RESTART); then the wall's state
# (nssolver_op2_benchmark_postprocess_helper wall) and, for a dual-time run,
# the forces (scripts/airfoil/forces.py).
#
#   CASE=oat15a ALPHA=3.5 DT=6.3e-4 STEPS=2000 RESTART=... BIN=... scripts/airfoil/run.sh [--option value ...]
#   CASE=circular_arc MACH=0.76 ALPHA=0 DT=1.9e-4 STEPS=2000 ...
# SPAN is the mesh's extrusion (lz, 0.01 by default).
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$APP_DIR"
CASE=${CASE:-oat15a}
case $CASE in
  oat15a) MACH=${MACH:-0.73}; RE=${RE:-3e6}; TEMP=${TEMP:-271}; MESH=${MESH:-meshes-op2/oat15a/deck.h5} ;;
  circular_arc) MACH=${MACH:-0.76}; RE=${RE:-11e6}; TEMP=${TEMP:-300}; MESH=${MESH:-meshes-op2/arc/arc18.h5} ;;
  *) echo "unknown CASE $CASE" >&2; exit 1 ;;
esac
ALPHA=${ALPHA:-0}
STEPS=${STEPS:-0}
DT=${DT:-0}
BIN=${BIN:-build/nssolver_op2_seq}
LAUNCH=${LAUNCH:-}
SUFFIX=""
[ "$STEPS" -gt 0 ] && SUFFIX="_dt$DT"
TAG=${TAG:-${CASE}_m${MACH}_a${ALPHA}$SUFFIX}
OUT=outputs-op2/$CASE
mkdir -p "$OUT"

FS=$(python3 scripts/airfoil/freestream.py "$MACH" "$RE" "$TEMP" "$ALPHA")
args=(--config "configs/$CASE.cfg" --mesh-file "$MESH" --output-file "$OUT/$TAG.h5" $FS --timing 0)
[ -n "${RESTART:-}" ] && args+=(--restart-file "$RESTART")
[ "$STEPS" -gt 0 ] && args+=(--time-step "$DT" --physical-steps "$STEPS")
[ -n "${ITERATIONS:-}" ] && args+=(--iterations "$ITERATIONS")
echo "$TAG: $FS"
$LAUNCH "$BIN" "${args[@]}" "$@" > "$OUT/$TAG.log" 2>&1
.helpers/bin/nssolver_op2_benchmark_postprocess_helper wall "$MESH" "$OUT/$TAG.h5" "$OUT/$TAG" 0 > /dev/null
if [ "$STEPS" -gt 0 ]; then
  python3 scripts/airfoil/forces.py "$OUT/$TAG.history.csv" "$MACH" "$RE" "$TEMP" "$ALPHA" "${SPAN:-0.01}"
else
  grep progress "$OUT/$TAG.log" | tail -1
fi
