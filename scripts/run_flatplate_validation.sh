#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$APP_DIR"

cd "$ROOT_DIR"

"$APP_DIR/scripts/build_helpers.sh"
"$APP_DIR/.helpers/bin/nssolver_demo_local" flatplate_develop

"$APP_DIR/scripts/preprocess_mesh.sh" flatplate meshes-op2/flatplate.h5

(
  cd "$APP_DIR"
  make seq
  ./nssolver_op2_seq --config configs/flatplate_develop.cfg
)

"$APP_DIR/scripts/postprocess_flatplate.sh" \
  flatplate_develop \
  meshes-op2/flatplate.h5 \
  outputs-op2/flatplate_develop_solution.h5 \
  outputs-op2/flatplate_develop

python3 "$APP_DIR/scripts/compare_flatplate_benchmark.py" \
  "$ROOT_DIR/flatplate_develop" \
  "$ROOT_DIR/outputs-op2/flatplate_develop"
