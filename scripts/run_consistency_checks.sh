#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$APP_DIR"

cd "$ROOT_DIR"

"$APP_DIR/scripts/build_helpers.sh"
"$APP_DIR/.helpers/bin/nssolver_demo_local" box
"$APP_DIR/.helpers/bin/nssolver_demo_local" bump
"$APP_DIR/scripts/preprocess_mesh.sh" box meshes-op2/box.h5
"$APP_DIR/scripts/preprocess_mesh.sh" bump meshes-op2/bump.h5

(
  cd "$APP_DIR"
  make seq
  ./nssolver_op2_seq --config configs/box.cfg
  ./nssolver_op2_seq --config configs/bump.cfg
)

python3 "$APP_DIR/scripts/check_residual_csv.py" "$ROOT_DIR/box_residual.csv" l2_rho 1.0e-12
python3 "$APP_DIR/scripts/check_residual_csv.py" "$ROOT_DIR/outputs-op2/box_solution.residual.csv" l2_rho 1.0e-12
python3 "$APP_DIR/scripts/compare_final_residual.py" \
  "$ROOT_DIR/bump_residual.csv" \
  "$ROOT_DIR/outputs-op2/bump_solution.residual.csv" \
  l2_rho \
  2.0e-01

echo "[ok] OP2 consistency checks passed"
