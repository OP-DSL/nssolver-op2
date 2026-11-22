#!/bin/bash -l
#SBATCH -p gpu --gres=gpu:4 -n 4 -c 16 -t 00:20:00 -J ns-diag
#SBATCH -o /scratch/c_ppkeitk/fv2k/op2sol/logs/diag-%j.out
# Does 4-rank nssolver on one node's GPUs start, under each Cray MPICH intra-node
# single-copy setting (the default failed in MPI_Allgather's lmt path)?
source /scratch/c_ppkeitk/fv2k/op2sol/scripts/env.sh
export MPICH_GPU_SUPPORT_ENABLED=1
S=$ROOT/src/nssolver
for mode in default NONE CMA XPMEM; do
  [ $mode != default ] && export MPICH_SMP_SINGLE_COPY_MODE=$mode
  echo "== single copy: $mode"
  OP_SOLVE_CONFIG='{"library":"petsc"}' timeout 300 srun -n 4 --gpus-per-task=1 $ROOT/build/nssolver/nssolver_op2_mpi_cuda \
    --config $S/configs/naca_wing.cfg --mesh-file $ROOT/data/meshes/wing1m.h5 --output-file $ROOT/runs/diag.h5 \
    --include-viscous 1 --include-sa 1 --nu-tilde 1e-4 --time-scheme implicit --implicit-cfl 5 --iterations 2 \
    --progress-interval 1 --solve-config $S/configs/solve.json --timing 0 2>&1 | grep -aE "progress|Invalid|Abort|rror" | head -4
  unset MPICH_SMP_SINGLE_COPY_MODE
done
