#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <case> <output_mesh.h5> [input_mesh.hdf] [key=value ...]" >&2
  exit 1
fi

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$APP_DIR/scripts/build_helpers.sh"
"$APP_DIR/.helpers/bin/nssolver_preprocess_op2_helper" "$@"
