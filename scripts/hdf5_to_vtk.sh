#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <mesh.h5> <solution.h5> <output.vtk>" >&2
  exit 1
fi

app_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$app_dir/scripts/build_helpers.sh"
"$app_dir/.helpers/bin/nssolver_hdf5_to_vtk_helper" "$1" "$2" "$3"
