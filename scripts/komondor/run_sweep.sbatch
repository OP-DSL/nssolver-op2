#!/bin/bash -l
#SBATCH -p gpu --gres=gpu:4 -n 4 -c 16 --mem=0 -t 04:00:00 -J ns-sweep
#SBATCH -o /scratch/c_ppkeitk/fv2k/op2sol/logs/sweep-%j.out
# Solves the systems nssolver dumped for MESH (data/dumps/MESH/system_*.h5) under
# a sweep of libraries, preconditioners and settings, 4 GPUs, one driver launch
# per system (scripts/sweep/run_sweep.sh); results in runs/MESH/sweep-TAG.csv.
#   sbatch --export=ALL,MESH=wing1m,TAG=basic scripts/run_sweep.sbatch
#   more nodes: sbatch -N 2 -n 8 --export=... (a rank per GPU, 4 GPUs a node)
#   env: MESH, TAG (names the sweep), RTOL (1e-3), MAXIT (300), SET (basic|full),
#        LEAN (default 1: PETSc ILU on borrowed device storage; empty for the default format), LIBS (petsc,hypre,ginkgo), FILES (default: all the mesh's dumps), REPEAT (3)
source /scratch/c_ppkeitk/fv2k/op2sol/scripts/env.sh
export MPICH_GPU_SUPPORT_ENABLED=1
MESH=${MESH:-wing1m}; TAG=${TAG:-basic}
W=$ROOT/src/nssolver/scripts/sweep
SW=$ROOT/runs/$MESH/sweep-$TAG; mkdir -p $SW
LEAN=${LEAN-1}   # PETSc ILU/ASM/GAMG on borrowed device storage unless LEAN= is set empty
python3 $W/make_sweep.py $SW --rtol ${RTOL:-1e-3} --maxit ${MAXIT:-300} --set ${SET:-basic} --libs ${LIBS:-petsc,hypre,ginkgo} ${LEAN:+--lean}
export DRIVER=$ROOT/build/op2/apps/c/solve_hdf5/solve_hdf5_6_mpi_cuda
export LAUNCH="srun -n ${SLURM_NTASKS:-4} --gpus-per-task=1" REPEAT=${REPEAT:-3}
FILES=${FILES:-$(ls $ROOT/data/dumps/$MESH/system_*.h5)}
bash $W/run_sweep.sh $SW $ROOT/runs/$MESH/sweep-$TAG.csv $FILES
python3 $W/summarize.py --tol $(python3 -c "print(2 * ${RTOL:-1e-3})") $ROOT/runs/$MESH/sweep-$TAG.csv > $ROOT/runs/$MESH/sweep-$TAG.md
cat $ROOT/runs/$MESH/sweep-$TAG.md
