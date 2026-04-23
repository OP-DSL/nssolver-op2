#!/usr/bin/env bash

# Test all nssolver_op2 variants (seq, genseq, openmp, cuda, c_cuda, and MPI equivalents)
# against all public mesh configs: box, bump, flatplate_develop.
#
# Usage:
#   COMPILE_TESTS=TRUE RUN_TESTS=TRUE ./tests/nssolver_op2_test.sh
#
# Individual phases can be enabled independently:
#   COMPILE_TESTS=TRUE  - build helper tools, preprocess meshes, and compile all app variants
#   RUN_TESTS=TRUE      - run each binary against all public configs and check for success output

set -e

export TEST_APP="nssolver_op2"

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_RUN_LOC="$APP_DIR/tests"
LOG="$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

COMPILE_TESTS=${COMPILE_TESTS:-FALSE}
RUN_TESTS=${RUN_TESTS:-FALSE}

# MPI process counts (tune for available hardware)
MPI_NP_SEQ=${MPI_NP_SEQ:-4}
MPI_NP_OMP=${MPI_NP_OMP:-2}
OMP_THREADS=${OMP_THREADS:-4}
MPI_NP_GPU=${MPI_NP_GPU:-2}

# Detect GPU availability (requires nvidia-smi; set GPU_AVAILABLE=TRUE to force-enable)
if [[ "${GPU_AVAILABLE:-}" != "TRUE" ]]; then
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        GPU_AVAILABLE=TRUE
    else
        GPU_AVAILABLE=FALSE
    fi
fi

# -------------------------------------------------------------------------
# Test harness (mirrors OP2-Common/tests/bash/test_core.sh)
# -------------------------------------------------------------------------

[ -f "$LOG" ] && rm "$LOG"

function is_file_available {
    local file="$1"
    if [ -f "$file" ]; then
        echo "$file exists" >> "$LOG"
        return 0
    else
        echo "TEST FAILED : $file is missing" >> "$LOG"
        return 1
    fi
}

# validate "<prepend>" "<binary>" "<args>" "<grep_word>"
function validate {
    local prepend="$1"
    local bin="$2"
    local args="$3"
    local grep_word="$4"

    if ! is_file_available "$bin"; then
        echo "SKIPPING: $bin does not exist" | tee -a "$LOG"
        echo "" | tee -a "$LOG"
        return
    fi

    local cmd
    if [[ -n "$prepend" ]]; then
        cmd="$prepend ./$bin $args"
    else
        cmd="./$bin $args"
    fi

    echo "Running: $cmd" | tee -a "$LOG"

    local file_tail="$bin"
    file_tail="${file_tail//./}"
    file_tail="${file_tail//\//_}"

    set +e
    eval "$cmd" > "$SCRIPT_RUN_LOC/perf_out_$file_tail" 2>&1
    local run_rc=$?
    set -e

    grep -q "$grep_word" "$SCRIPT_RUN_LOC/perf_out_$file_tail" 2>/dev/null
    local grep_rc=$?

    if [[ $run_rc != 0 ]] || [[ $grep_rc != 0 ]]; then
        echo "$bin xxxxxxxxxxxxxxxxxxx TEST FAILED" | tee -a "$LOG"
        # Preserve output for diagnosis on failure
        cat "$SCRIPT_RUN_LOC/perf_out_$file_tail" | tee -a "$LOG"
    else
        echo "$bin +++++++++++++++++++ TEST PASSED" | tee -a "$LOG"
    fi

    rm -f "$SCRIPT_RUN_LOC/perf_out_$file_tail"
    echo "" | tee -a "$LOG"
}

# validate_gpu — same as validate but skips when no GPU is detected.
function validate_gpu {
    if [[ "$GPU_AVAILABLE" != "TRUE" ]]; then
        echo "SKIPPING (no GPU): $2" | tee -a "$LOG"
        echo "" | tee -a "$LOG"
        return
    fi
    validate "$@"
}

function check_all_tests {
    set +e
    grep -q "FAILED" "$LOG"
    local rc=$?
    set -e
    if [[ $rc != 0 ]]; then
        echo "All ${TEST_APP} Tests Passed" | tee -a "$LOG"
        return 0
    else
        echo "Some of ${TEST_APP} Tests Failed, check ${TEST_APP}_test.log" | tee -a "$LOG"
        return 1
    fi
}

# -------------------------------------------------------------------------
# Build phase
# -------------------------------------------------------------------------

if [[ "$COMPILE_TESTS" = "TRUE" ]]; then
    echo "Building helper tools..." | tee -a "$LOG"
    "$APP_DIR/scripts/build_helpers.sh"

    echo "Preprocessing meshes..." | tee -a "$LOG"
    "$APP_DIR/scripts/preprocess_mesh.sh" box      meshes-op2/box.h5
    "$APP_DIR/scripts/preprocess_mesh.sh" bump     meshes-op2/bump.h5
    "$APP_DIR/scripts/preprocess_mesh.sh" flatplate meshes-op2/flatplate.h5

    cd "$APP_DIR"
    echo "Building all app variants..." | tee -a "$LOG"
    make all
fi

# -------------------------------------------------------------------------
# Run phase
# -------------------------------------------------------------------------

# run_all_variants "<config_path>"  — runs every variant against the given config.
function run_all_variants {
    local cfg="$1"

    echo "--- config: $cfg ---" | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    # Non-MPI variants
    validate "" \
        "nssolver_op2_seq" \
        "--config $cfg" \
        "Wrote solution"

    validate "" \
        "nssolver_op2_genseq" \
        "--config $cfg" \
        "Wrote solution"

    validate "OMP_NUM_THREADS=${OMP_THREADS}" \
        "nssolver_op2_openmp" \
        "--config $cfg" \
        "Wrote solution"

    validate_gpu "" \
        "nssolver_op2_cuda" \
        "--config $cfg" \
        "Wrote solution"

    validate_gpu "" \
        "nssolver_op2_c_cuda" \
        "--config $cfg" \
        "Wrote solution"

    # MPI variants
    validate "mpirun -np ${MPI_NP_SEQ}" \
        "nssolver_op2_mpi_seq" \
        "--config $cfg" \
        "Wrote solution"

    validate "mpirun -np ${MPI_NP_SEQ}" \
        "nssolver_op2_mpi_genseq" \
        "--config $cfg" \
        "Wrote solution"

    validate "OMP_NUM_THREADS=${OMP_THREADS} mpirun -np ${MPI_NP_OMP}" \
        "nssolver_op2_mpi_openmp" \
        "--config $cfg" \
        "Wrote solution"

    validate_gpu "mpirun -np ${MPI_NP_GPU}" \
        "nssolver_op2_mpi_cuda" \
        "--config $cfg" \
        "Wrote solution"

    validate_gpu "mpirun -np ${MPI_NP_GPU}" \
        "nssolver_op2_mpi_c_cuda" \
        "--config $cfg" \
        "Wrote solution"
}

if [[ "$RUN_TESTS" = "TRUE" ]]; then
    cd "$APP_DIR"

    echo "Running ${TEST_APP} tests" | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    run_all_variants "configs/box.cfg"
    run_all_variants "configs/bump.cfg"
    run_all_variants "configs/flatplate_develop.cfg"

    # Internal placeholder — uncomment when internal mesh files are present:
    # run_all_variants "configs/internal/hydra.cfg"
    # run_all_variants "configs/internal/hydra_benchmark.cfg"

    check_all_tests
fi
