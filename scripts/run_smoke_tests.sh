#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$APP_DIR"
source "$APP_DIR/scripts/op2_backend.sh"

cd "$ROOT_DIR"

"$APP_DIR/scripts/build_helpers.sh"
"$APP_DIR/scripts/preprocess_mesh.sh" box meshes-op2/box.h5
"$APP_DIR/scripts/preprocess_mesh.sh" bump meshes-op2/bump.h5

op2_ensure_backend_built "$APP_DIR"
OP2_BINARY="./$(op2_binary_name)"

(
  cd "$APP_DIR"
  "$OP2_BINARY" --config configs/box.cfg
  "$OP2_BINARY" --config configs/bump.cfg
)

python3 "$APP_DIR/scripts/check_residual_csv.py" "$ROOT_DIR/outputs-op2/box_solution.residual.csv" l2_rho 1.0e-12
python3 "$APP_DIR/scripts/check_residual_csv.py" "$ROOT_DIR/outputs-op2/bump_solution.residual.csv" l2_rho 8.0e-03

echo "[ok] OP2 smoke tests passed"
