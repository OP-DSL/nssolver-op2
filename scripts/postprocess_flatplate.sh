#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 <case> <mesh.h5> <solution.h5> <output_prefix>" >&2
  exit 1
fi

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$APP_DIR/scripts/build_helpers.sh"
"$APP_DIR/.helpers/bin/nssolver_op2_benchmark_postprocess_helper" "$1" "$2" "$3" "$4" 0.3
