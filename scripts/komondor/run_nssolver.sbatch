#!/bin/bash -l
#SBATCH -p gpu --gres=gpu:4 -n 4 -c 16 --mem=0 -t 02:00:00 -J ns-wing
#SBATCH -o /scratch/c_ppkeitk/fv2k/op2sol/logs/nswing-%j.out
# The viscous Spalart-Allmaras wing on one node's four GPUs, implicit with a CFL
# ramp, dumping every DUMP-th iteration's linear system (matrix, right-hand
# side, solution) into the striped data/dumps/<mesh>/: the solves get harder as
# the CFL grows, until the last ones the library's default no longer converges.
#   sbatch --export=ALL,MESH=wing1m scripts/run_nssolver.sbatch   (MESH, ITERS, DUMP, KEEP)
# KEEP="4 10 16" removes every other dump once the index lists it (written):
# the group's /scratch quota is 1 TiB and a 10M system takes 22 GB.
source /scratch/c_ppkeitk/fv2k/op2sol/scripts/env.sh
export MPICH_GPU_SUPPORT_ENABLED=1
MESH=${MESH:-wing1m}; ITERS=${ITERS:-32}; DUMP=${DUMP:-2}
S=$ROOT/src/nssolver
D=$ROOT/data/dumps/$MESH
mkdir -p $D $ROOT/runs/$MESH
prune() {
  tail -n +2 $D/system_index.csv 2>/dev/null | while IFS=, read -r it _ _ _ f; do
    case " $KEEP " in *" $it "*) ;; *) rm -f "$f" ;; esac
  done
}
[ -n "${KEEP:-}" ] && { while sleep 10; do prune; done & PRUNER=$!; }
# PETSc on the device keeps the matrix twice in its default format; the borrowed storage
# (owned) holds it once, which is what leaves room for ILU at 10M nodes.
CFG="{\"library\":\"${LIB:-petsc}\"}"
[ "${LIB:-petsc}" = petsc ] && [ "${OWNED:-1}" = 1 ] && CFG='{"library":"petsc","matrices":{"A":{"library":"petsc","format":"aij","owned":true}}}'
OP_SOLVE_CONFIG="$CFG" srun -n ${SLURM_NTASKS:-4} --gpus-per-task=1 $ROOT/build/nssolver/nssolver_op2_mpi_cuda \
  --config $S/configs/naca_wing.cfg --mesh-file $ROOT/data/meshes/$MESH.h5 \
  --output-file $ROOT/runs/$MESH/solution.h5 --include-viscous 1 --include-sa 1 --nu-tilde 1e-4 \
  --time-scheme implicit --implicit-cfl 5 --implicit-cfl-growth 1.5 --implicit-cfl-max 1e5 \
  --iterations $ITERS --progress-interval 1 --solve-config $S/configs/solve.json --timing 0 \
  --dump-interval $DUMP --dump-prefix $D/system
[ -n "${KEEP:-}" ] && { kill $PRUNER; prune; }
