#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$APP_DIR/meshes-op2"

bash "$APP_DIR/scripts/preprocess_mesh.sh" bump3d "$APP_DIR/meshes-op2/bump3d.h5"
bash "$APP_DIR/scripts/preprocess_mesh.sh" axisymmetric_body "$APP_DIR/meshes-op2/axisymmetric_body.h5"
bash "$APP_DIR/scripts/preprocess_mesh.sh" naca_wing "$APP_DIR/meshes-op2/naca_wing.h5"

test -s "$APP_DIR/meshes-op2/bump3d.h5"
test -s "$APP_DIR/meshes-op2/axisymmetric_body.h5"
test -s "$APP_DIR/meshes-op2/naca_wing.h5"

echo "[ok] formula-driven 3D preprocessing meshes generated"
