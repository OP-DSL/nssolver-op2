#!/bin/bash -l
#SBATCH -p gpu --gres=gpu:4 -n 4 -c 16 --mem=0 -t 01:00:00 -J ns-mem
#SBATCH -o /scratch/c_ppkeitk/fv2k/op2sol/logs/mem-%j.out
# Host and device memory and time of the 10M wing's implicit step per library
# (two iterations, no dumps): which storage the solver layer can set up on one node.
source /scratch/c_ppkeitk/fv2k/op2sol/scripts/env.sh
export MPICH_GPU_SUPPORT_ENABLED=1 OP_SOLVE_MEMORY=1
S=$ROOT/src/nssolver
for lib in ${LIBS:-hypre ginkgo petsc}; do
  echo "== $lib"
  OP_SOLVE_CONFIG="{\"library\":\"$lib\"}" timeout 900 srun -n 4 --gpus-per-task=1 $ROOT/build/nssolver/nssolver_op2_mpi_cuda \
    --config $S/configs/naca_wing.cfg --mesh-file $ROOT/data/meshes/${MESH:-wing10m}.h5 --output-file $ROOT/runs/diag.h5 \
    --include-viscous 1 --include-sa 1 --nu-tilde 1e-4 --time-scheme implicit --implicit-cfl 5 --iterations 2 \
    --progress-interval 1 --solve-config $S/configs/solve.json --timing 0 2>&1 | grep -aE "progress|implicit:|time:|Out Of Memory|oom|matrix|solver .* solved|rror" | head -12 | cut -c1-220
done
sacct -j $SLURM_JOB_ID -n -o JobID%16,MaxRSS,Elapsed,State%12 | grep -v "batch\|extern"
