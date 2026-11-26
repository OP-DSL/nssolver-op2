#!/usr/bin/env bash
# ONERA M6, SU2's inviscid tutorial mesh: docs/ONERA_M6_VALIDATION.md.
#
#   BIN=build/nssolver_op2_seq scripts/run_onera_m6_validation.sh [--option value ...]
#
# Extra arguments go to the run.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"
BIN=${BIN:-build/nssolver_op2_seq}
LAUNCH=${LAUNCH:-}

SU2_MESH="public-data/onera_m6/mesh_ONERAM6_inv_ffd.su2"
OP2_MESH="meshes-op2/onera_m6.h5"
CFG="configs/onera_m6.cfg"
SOLUTION="outputs-op2/onera_m6_solution.h5"
SUMMARY="outputs-op2/onera_m6_summary.json"
SURFACE_CP="outputs-op2/onera_m6_surface_cp.csv"
VTK="outputs-op2/onera_m6_solution.vtk"
SECTIONS="outputs-op2/onera_m6_sections.csv"
EXPERIMENT="public-data/onera_m6/ONERAb114.tec"

RHO_INF="1.2250122659906946"
U_INF="285.26805348028574"
V_INF="0.0"
W_INF="15.249834230540356"
P_INF="101325.0"
WALL_GROUPS="1,2,3"

scripts/download_onera_m6_su2_case.sh
mkdir -p meshes-op2 outputs-op2
# Always afresh (a second or so): a mesh from an older importer may lack datasets or be wrong.
.helpers/bin/nssolver_preprocess_op2_helper onera_m6 "$OP2_MESH" "$SU2_MESH"

$LAUNCH "$BIN" --config "$CFG" "$@"

.helpers/bin/nssolver_onera_m6_validation_helper \
  "$OP2_MESH" \
  "$SOLUTION" \
  "$SUMMARY" \
  "$SURFACE_CP" \
  "$RHO_INF" \
  "$U_INF" \
  "$V_INF" \
  "$W_INF" \
  "$P_INF" \
  "$WALL_GROUPS" \
  "$SECTIONS"
python3 scripts/onera_m6_sections.py "$SECTIONS" "$EXPERIMENT" outputs-op2/onera_m6_sections.png

.helpers/bin/nssolver_hdf5_to_vtk_helper "$OP2_MESH" "$SOLUTION" "$VTK"

echo "validation summary: $SUMMARY"
echo "surface Cp CSV: $SURFACE_CP"
echo "sections: $SECTIONS"
echo "VTK output: $VTK"
