#!/usr/bin/env bash
# The implicit scheme across solver libraries, app variants and the linear
# solve's options, on the bump with a growing pseudo-time CFL.
#
#   BUILD_DIR=build LIBS="petsc hypre" tests/test_implicit.sh
#
# Every run must finish with L2(rho) fallen below MAX_RATIO of its first
# iteration's. Runs of one algorithm must agree to TOL (round-off grows
# through the Krylov iterations as the residual falls): each library's
# single-rank variants with its seq, and its MPI variants (MPI_NP ranks) with
# its mpi_seq, as the libraries' default preconditioners follow the partition.
# BUILD_DIR holds the CMake build's nssolver_op2_<variant> binaries; variants
# missing there are skipped. OP_SOLVE_CONFIG selects each library over
# configs/solve.json.
set -uo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"
BUILD_DIR=${BUILD_DIR:-build}
LIBS=${LIBS:-"petsc hypre"}
VARIANTS=${VARIANTS:-"seq genseq openmp cuda mpi_seq mpi_openmp mpi_cuda"}
MPI_NP=${MPI_NP:-2}
TOL=${TOL:-1e-2}
MAX_RATIO=${MAX_RATIO:-0.1}
OUT=outputs-op2/test_implicit
mkdir -p "$OUT"

COMMON=(--config configs/bump.cfg --time-scheme implicit --implicit-cfl 10 --implicit-cfl-growth 1.5
        --implicit-cfl-max 1e5 --iterations 12
        --progress-interval 12 --solve-config configs/solve.json --timing 0)

declare -A reference
failures=0
runs=0

# run <label> <group> <library> <variant> [options...]: runs of a group agree.
run() {
  local label=$1 group=$2 lib=$3 variant=$4
  shift 4
  local bin="$BUILD_DIR/nssolver_op2_$variant"
  [ -x "$bin" ] || return 0
  local launch=()
  [[ $variant == mpi_* ]] && launch=(mpirun -n "$MPI_NP")
  local log="$OUT/$label.log"
  OP_SOLVE_CONFIG="{\"library\": \"$lib\"}" OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} \
    "${launch[@]}" "$bin" "${COMMON[@]}" --output-file "$OUT/$label.h5" "$@" > "$log" 2>&1
  local status=$?
  local final ratio rejected
  final=$(grep -oE 'iter 12/12 \| L2\(rho\)=[^ ]+' "$log" | sed 's/.*=//')
  ratio=$(grep -oE 'iter 12/12 .*L2/L2_0=[^ ]+' "$log" | sed 's/.*=//')
  rejected=$(grep -oE '[0-9]+ rejected node updates' "$log" | cut -d' ' -f1)
  runs=$((runs + 1))
  local verdict=ok
  if [ $status -ne 0 ] || [ -z "$final" ]; then
    verdict="FAILED (exit $status)"
  elif ! python3 -c "import sys; sys.exit(not float(sys.argv[1]) <= float(sys.argv[2]))" "$ratio" "$MAX_RATIO"; then
    verdict="FAILED (L2/L2_0 $ratio)"
  elif [ -z "${reference[$group]:-}" ]; then
    reference[$group]=$final
  elif ! python3 -c "import sys; a, b, t = map(float, sys.argv[1:]); sys.exit(abs(a - b) > t * abs(b))" \
         "$final" "${reference[$group]}" "$TOL"; then
    verdict="FAILED (L2 $final, $group ${reference[$group]})"
  fi
  [[ $verdict == ok ]] || failures=$((failures + 1))
  printf '%-36s L2(rho) %-11s L2/L2_0 %-9s rejected %-4s %s\n' "$label" "${final:--}" "${ratio:--}" \
    "${rejected:--}" "$verdict"
}

first_lib=${LIBS%% *}
for lib in $LIBS; do
  for variant in $VARIANTS; do
    group="$lib"
    [[ $variant == mpi_* ]] && group="${lib}_mpi"
    run "${lib}_${variant}" "$group" "$lib" "$variant"
  done
done

# The options, each its own group.
for opts in "--pc bjacobi" "--pc neumann --terms 2" "--pmat decoupled" "--jacobian-interval 3" \
            "--op matfree --pc bjacobi"; do
  name=$(echo "$opts" | tr -d '-' | tr ' ' '_')
  run "${first_lib}_seq_$name" "seq_$name" "$first_lib" seq $opts
  run "${first_lib}_cuda_$name" "seq_$name" "$first_lib" cuda $opts
  run "${first_lib}_mpi_seq_$name" "mpi_$name" "$first_lib" mpi_seq $opts
done
for opts in "--partition random" "--partition inertial" "--reorder 1"; do
  name=$(echo "$opts" | tr -d '-' | tr ' ' '_')
  run "${first_lib}_mpi_seq_$name" "mpi_$name" "$first_lib" mpi_seq $opts
  run "${first_lib}_mpi_cuda_$name" "mpi_$name" "$first_lib" mpi_cuda $opts
done

echo "$((runs - failures))/$runs passed"
[ $failures -eq 0 ]
