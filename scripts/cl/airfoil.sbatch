#!/bin/bash
# An airfoil run (scripts/airfoil/run.sh) on one A100 of renyi; the meshes
# first if missing (the OAT15A grid from ~/op2sol/data/oat15a, ONERA's
# geometry, not in this repository):
#   CASE=oat15a ALPHA=2.5 ITERATIONS=600 sbatch scripts/cl/airfoil.sbatch
#   CASE=oat15a ALPHA=3.5 DT=6.3e-4 STEPS=2000 RESTART=outputs-op2/oat15a/... sbatch ...
# TAG names the run, EXTRA holds further nssolver options.
#SBATCH -p gpu -w renyi -N1 -c8 --gres=gpu:1 -t 12:00:00 -J airfoil
#SBATCH -o /home/reguly/op2sol/logs/airfoil-%j.out
# ARCH=70 with -w neumann|erdos|wald runs the V100 build (scripts/cl/build.sbatch).
# neumann has GCC 12.4.0 rather than 12.3.0: its libstdc++ serves the build.
module load OP2-env 2>&1
module load gnu12/12.3.0 2>/dev/null || module load gnu12 2>&1
ARCH=${ARCH:-80}
SUFFIX=$([ "$ARCH" = 80 ] || echo "$ARCH")
export LD_LIBRARY_PATH=$HOME/op2sol/install$SUFFIX/petsc/lib:$HOME/op2sol/install$SUFFIX/op2/lib:$HOME/op2sol/install$SUFFIX/op2/lib64:$HOME/hdf5-cmake-par/lib:$LD_LIBRARY_PATH
cd $HOME/op2sol/src/nssolver
H=.helpers/bin/nssolver_preprocess_op2_helper
mkdir -p meshes-op2/arc meshes-op2/oat15a
[ -s meshes-op2/arc/arc18.h5 ] || $H circular_arc meshes-op2/arc/arc18.h5 surface_points=201 normal_points=129
[ -s meshes-op2/oat15a/deck.h5 ] || $H plot3d_blocks meshes-op2/oat15a/deck.h5 $HOME/op2sol/data/oat15a/grid_OAT15A_Deck.x \
  scale=4.3478260869565215 wall_x0=-0.01 wall_x1=1.01 wall_y=0.2
# Every other line where the blocks' shared lines allow it: a quarter of the points.
[ -s meshes-op2/oat15a/deck_s2.h5 ] || $H plot3d_blocks meshes-op2/oat15a/deck_s2.h5 $HOME/op2sol/data/oat15a/grid_OAT15A_Deck.x \
  scale=4.3478260869565215 wall_x0=-0.01 wall_x1=1.01 wall_y=0.2 "strides=2,2;1,2;2,1;1,2"
echo "CASE=$CASE ALPHA=$ALPHA MACH=$MACH DT=$DT STEPS=$STEPS ITERATIONS=$ITERATIONS RESTART=$RESTART TAG=$TAG EXTRA=$EXTRA"
time BIN=$HOME/op2sol/build$SUFFIX/ns/nssolver_op2_cuda scripts/airfoil/run.sh --solve-config configs/solve.json $EXTRA
echo AIRFOIL-DONE
