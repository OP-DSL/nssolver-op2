#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SU2_MESH="$APP_DIR/public-data/onera_m6/mesh_ONERAM6_inv_ffd.su2"
OP2_MESH="$APP_DIR/meshes-op2/onera_m6.h5"
CFG="$APP_DIR/configs/onera_m6.cfg"
SOLUTION="$APP_DIR/outputs-op2/onera_m6_solution.h5"
SUMMARY="$APP_DIR/outputs-op2/onera_m6_summary.json"
SURFACE_CP="$APP_DIR/outputs-op2/onera_m6_surface_cp.csv"
VTK="$APP_DIR/outputs-op2/onera_m6_solution.vtk"

RHO_INF="1.2250122659906946"
U_INF="285.26805348028574"
V_INF="0.0"
W_INF="15.249834230540356"
P_INF="101325.0"
WALL_GROUPS="1,2,3"

"$APP_DIR/scripts/download_onera_m6_su2_case.sh"

if [ ! -f "$OP2_MESH" ]; then
  "$APP_DIR/scripts/preprocess_mesh.sh" onera_m6 "$OP2_MESH" "$SU2_MESH"
fi

make -C "$APP_DIR" seq
"$APP_DIR/nssolver_op2_seq" --config "$CFG"

"$APP_DIR/scripts/build_helpers.sh"
"$APP_DIR/.helpers/bin/nssolver_onera_m6_validation_helper" \
  "$OP2_MESH" \
  "$SOLUTION" \
  "$SUMMARY" \
  "$SURFACE_CP" \
  "$RHO_INF" \
  "$U_INF" \
  "$V_INF" \
  "$W_INF" \
  "$P_INF" \
  "$WALL_GROUPS"

"$APP_DIR/scripts/hdf5_to_vtk.sh" "$OP2_MESH" "$SOLUTION" "$VTK"

echo "validation summary: $SUMMARY"
echo "surface Cp CSV: $SURFACE_CP"
echo "VTK output: $VTK"
