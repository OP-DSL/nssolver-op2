#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[helpers] no configuration step required"
echo "[helpers] build helper tools with: make -C \"$APP_DIR\" helpers-build"
