#!/usr/bin/env bash
# Solves the linear systems nssolver_op2 dumped (--dump-interval N --dump-prefix P)
# again with OP2's solve_hdf5 driver, under each of a set of solver settings,
# and tabulates the iterations and residuals.
#
#   scripts/replay_systems.sh [-v VARIANTS] [-s SOLVER] FILE.h5 ...
#   DRIVER=.../solve_hdf5_6_mpi_cuda NP=2 scripts/replay_systems.sh outputs-op2/system_0*.h5
#
# VARIANTS holds one setting per line, "label<TAB>OP_SOLVE_CONFIG json", which
# overrides configs/solve.json (-config) as OP_SOLVE_CONFIG does in a run:
#   petsc	{"library":"petsc"}
#   restart60	{"library":"petsc","solvers":{"nssolver":{"petsc":{"ksp_gmres_restart":60}}}}
# Without it: the libraries built, as configs/solve.json has them. Replaying the
# run's own setting (first line) on the run's rank count reproduces its solve;
# other rank counts change the block-Jacobi blocks, hence the iterations.
set -uo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVER=${DRIVER:-$HOME/opt/build/op2-solvers/apps/c/solve_hdf5/solve_hdf5_6_mpi_genseq}
NP=${NP:-2}
CONFIG=${CONFIG:-$APP_DIR/configs/solve.json}
SOLVER=nssolver
VARIANTS=""
while getopts "v:s:" o; do
  case $o in v) VARIANTS=$OPTARG ;; s) SOLVER=$OPTARG ;; *) exit 2 ;; esac
done
shift $((OPTIND - 1))
[ $# -gt 0 ] || { sed -n '2,15p' "$0"; exit 2; }

if [ -n "$VARIANTS" ]; then
  mapfile -t lines < <(grep -v '^[[:space:]]*\(#\|$\)' "$VARIANTS")
else
  lines=($'petsc\t{"library":"petsc"}' $'hypre\t{"library":"hypre"}')
fi

printf '%-24s %-14s %5s %11s %11s %9s\n' system variant its residual true_res solve_s
for f in "$@"; do
  for line in "${lines[@]}"; do
    label=${line%%$'\t'*}; json=${line#*$'\t'}
    out=$(OP_SOLVE_CONFIG="$json" mpirun -n "$NP" "$DRIVER" -file "$f" -config "$CONFIG" \
          -solver "$SOLVER" 2>&1)
    s=$(grep -E '^solve:' <<<"$out" | head -1)
    its=$(sed -E 's/^solve: ([0-9]+) iterations.*/\1/' <<<"$s")
    res=$(sed -E 's/.*residual ([^,]+),.*/\1/' <<<"$s")
    tm=$(sed -E 's/.*solve ([0-9.]+) s.*/\1/' <<<"$s")
    tru=$(grep -E '^true relative residual' <<<"$out" | awk '{print $4}')
    flag=""; grep -q 'NOT converged' <<<"$s" && flag=" *"
    [ -n "$s" ] || { its=-; res=-; tm=-; flag=" (failed)"; }
    printf '%-24s %-14s %5s %11s %11s %9s%s\n' "$(basename "$f" .h5)" "$label" "$its" "$res" \
      "${tru:--}" "$tm" "$flag"
  done
done
echo "(* = solve did not converge)"
