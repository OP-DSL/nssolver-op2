#!/usr/bin/env bash

set -euo pipefail

op2_target_name() {
  local target="${OP2_TARGET:-seq}"
  case "$target" in
    seq|genseq|openmp|cuda)
      printf '%s\n' "$target"
      ;;
    *)
      echo "[error] unsupported OP2_TARGET='$target' (expected seq, genseq, openmp, or cuda)" >&2
      return 1
      ;;
  esac
}

op2_binary_name() {
  local target
  target="$(op2_target_name)"
  printf 'nssolver_op2_%s\n' "$target"
}

op2_build_target() {
  op2_target_name
}

op2_ensure_backend_built() {
  local app_dir="$1"
  local build_target
  build_target="$(op2_build_target)"
  make -C "$app_dir" "$build_target"
}
