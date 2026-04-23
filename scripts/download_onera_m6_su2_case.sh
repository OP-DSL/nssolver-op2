#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$APP_DIR/public-data/onera_m6"
CFG_PATH="$OUT_DIR/inv_ONERAM6.cfg"
MESH_PATH="$OUT_DIR/mesh_ONERAM6_inv_ffd.su2"
ZIP_PATH="$OUT_DIR/Tutorials-master.zip"
TMP_EXTRACT="$OUT_DIR/.tmp_extract"

CFG_URL="https://raw.githubusercontent.com/su2code/Tutorials/master/compressible_flow/Inviscid_ONERAM6/inv_ONERAM6.cfg"
ZIP_URL="https://codeload.github.com/su2code/Tutorials/zip/refs/heads/master"

mkdir -p "$OUT_DIR"

if [ -f "$CFG_PATH" ] && [ -f "$MESH_PATH" ]; then
  echo "ONERA M6 tutorial assets already present in $OUT_DIR"
  exit 0
fi

echo "Downloading ONERA M6 tutorial config"
curl -L --fail "$CFG_URL" -o "$CFG_PATH"

echo "Downloading SU2 Tutorials archive"
curl -L --fail "$ZIP_URL" -o "$ZIP_PATH"

rm -rf "$TMP_EXTRACT"
mkdir -p "$TMP_EXTRACT"

python3 - <<'PY' "$ZIP_PATH" "$TMP_EXTRACT"
import pathlib
import sys
import zipfile

zip_path = pathlib.Path(sys.argv[1])
extract_root = pathlib.Path(sys.argv[2])
target_rel = pathlib.Path("Tutorials-master/compressible_flow/Inviscid_ONERAM6/mesh_ONERAM6_inv_ffd.su2")

with zipfile.ZipFile(zip_path) as zf:
    zf.extract(str(target_rel), path=extract_root)
PY

cp "$TMP_EXTRACT/Tutorials-master/compressible_flow/Inviscid_ONERAM6/mesh_ONERAM6_inv_ffd.su2" "$MESH_PATH"
rm -rf "$TMP_EXTRACT"

echo "Downloaded:"
echo "  $CFG_PATH"
echo "  $MESH_PATH"
