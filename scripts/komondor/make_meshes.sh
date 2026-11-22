#!/bin/bash -l
#SBATCH -p gpu --gres=gpu:1 -c 16 -t 00:30:00 -J ns-meshes   # the wrapper links libcuda, absent on cpu nodes
#SBATCH -o /scratch/c_ppkeitk/fv2k/op2sol/logs/meshes-%j.out
# The wing ladder (procedural swept NACA wing in a graded farfield block) written
# into a Lustre directory striped over every OST: parallel HDF5 reads of the
# mesh (and of the matrices dumped later) then spread over the storage.
#   sbatch scripts/make_meshes.sh [ladder...]   default: wing1m wing10m
source /scratch/c_ppkeitk/fv2k/op2sol/scripts/env.sh
set -eu
D=$ROOT/data
for d in meshes dumps; do
  mkdir -p $D/$d
  lfs setstripe -c -1 -S 4M $D/$d     # all OSTs, 4 MiB stripes; applies to new files
done
lfs getstripe -d $D/meshes
S=$ROOT/src/nssolver
H=$S/.helpers/bin/nssolver_preprocess_op2_helper
if [ ! -x $H ]; then   # not through the Makefile, which wants an OP2 checkout's makefiles
  mkdir -p $S/.helpers/bin
  (cd $S && $CXX -O2 -std=c++20 -DNSSOLVER_HAVE_HDF5=1 -Iinclude -I$PREFIX/hdf5/include \
     src/mesh.cpp src/physics.cpp src/validation.cpp src/vtk_writer.cpp src/hydra_reader.cpp \
     src/su2_reader.cpp src/hydra_benchmark.cpp src/preprocess_op2.cpp \
     -L$PREFIX/hdf5/lib -lhdf5 -ldl -lm -o $H) >$ROOT/logs/helpers-build.txt 2>&1
fi
declare -A SIZE=([wing1m]="nx=161 ny=61 nz=101" [wing10m]="nx=351 ny=131 nz=221")
for m in ${@:-wing1m wing10m}; do
  [ -f $D/meshes/$m.h5 ] && continue
  /usr/bin/time -f "$m: %e s, %M kB" $H naca_wing $D/meshes/$m.h5 ${SIZE[$m]}
done
ls -la $D/meshes; lfs getstripe $D/meshes/wing1m.h5 | head -4
