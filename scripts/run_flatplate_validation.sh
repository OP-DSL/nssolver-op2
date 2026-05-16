#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$APP_DIR"
source "$APP_DIR/scripts/op2_backend.sh"

cd "$ROOT_DIR"

"$APP_DIR/scripts/build_helpers.sh"

"$APP_DIR/scripts/preprocess_mesh.sh" flatplate meshes-op2/flatplate.h5

op2_ensure_backend_built "$APP_DIR"
OP2_BINARY="./$(op2_binary_name)"

(
  cd "$APP_DIR"
  "$OP2_BINARY" --config configs/flatplate_develop.cfg
)

python3 "$APP_DIR/scripts/check_residual_csv.py" "$ROOT_DIR/outputs-op2/flatplate_develop_solution.residual.csv" l2_rho 2.4e-03 2.3e-03

"$APP_DIR/scripts/postprocess_flatplate.sh" \
  flatplate_develop \
  meshes-op2/flatplate.h5 \
  outputs-op2/flatplate_develop_solution.h5 \
  outputs-op2/flatplate_develop

for csv in \
  "$ROOT_DIR/outputs-op2/flatplate_develop_wall.csv" \
  "$ROOT_DIR/outputs-op2/flatplate_develop_profile_20.csv" \
  "$ROOT_DIR/outputs-op2/flatplate_develop_profile_50.csv" \
  "$ROOT_DIR/outputs-op2/flatplate_develop_profile_80.csv"; do
  test -s "$csv"
done

echo "[ok] OP2 flat-plate validation passed"
