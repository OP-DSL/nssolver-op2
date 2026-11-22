#!/usr/bin/env bash
# Laminar vortex shedding behind a circular cylinder at Re 100, M 0.2
# (configs/cylinder_re100.cfg), by dual time stepping: a mesh, a start with a
# 1% cross flow that breaks the symmetry for TRIGGER steps, then STEPS steps
# from it in the plain free stream, and the Strouhal number, mean drag and
# lift amplitude against Williamson (1989) and Park et al. (1998).
#
#   BIN=build/nssolver_op2_openmp GRID="192 128" DT=1.75e-3 scripts/cylinder/run.sh [--option value ...]
#
# GRID is "ntheta nr" (points around, points out to 30 diameters, the first
# off the wall at FIRST); DT the time step (the period is about 0.0873).
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$APP_DIR"
BIN=${BIN:-build/nssolver_op2_seq}
LAUNCH=${LAUNCH:-}
read -r NTHETA NR <<< "${GRID:-96 64}"
FIRST=${FIRST:-0.005}
DT=${DT:-1.75e-3}
TRIGGER=${TRIGGER:-300}
STEPS=${STEPS:-600}
TAG=${TAG:-cyl_${NTHETA}x${NR}_dt${DT}}
OUT=outputs-op2/cyl
mkdir -p meshes-op2/cyl "$OUT"

mesh=meshes-op2/cyl/cyl_${NTHETA}x${NR}.h5
[ -s "$mesh" ] || .helpers/bin/nssolver_preprocess_op2_helper cylinder "$mesh" ntheta="$NTHETA" nr="$NR" \
                     first_spacing="$FIRST" farfield_radius=30 lz=0.1 > /dev/null

RHO=$(sed -n 's/^rho *= *//p' configs/cylinder_re100.cfg)
U=$(sed -n 's/^u *= *//p' configs/cylinder_re100.cfg)
common=(--config configs/cylinder_re100.cfg --mesh-file "$mesh" --time-step "$DT" --timing 0)

if [ ! -s "$OUT/${TAG}_trigger.h5" ]; then
  $LAUNCH "$BIN" "${common[@]}" --v "$(python3 -c "print(0.01 * $U)")" --physical-steps "$TRIGGER" \
    --output-file "$OUT/${TAG}_trigger.h5" "$@" > "$OUT/${TAG}_trigger.log" 2>&1
fi
$LAUNCH "$BIN" "${common[@]}" --restart-file "$OUT/${TAG}_trigger.h5" --physical-steps "$STEPS" \
  --output-file "$OUT/$TAG.h5" "$@" > "$OUT/$TAG.log" 2>&1
echo -n "$TAG: "
python3 scripts/cylinder/forces.py "$OUT/$TAG.history.csv" "$RHO" "$U" 1.0 0.1 5
