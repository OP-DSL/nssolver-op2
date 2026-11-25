#!/bin/bash
# TMR flat plate, the finest grids, on one A100 of renyi.
#SBATCH -p gpu -w renyi -N1 -c8 --gres=gpu:1 -t 03:00:00 -J tmr
#SBATCH -o /home/reguly/op2sol/logs/tmr-%j.out
# ARCH=70 with -w erdos|wald runs the V100 build (scripts/cl/build.sbatch).
module load OP2-env 2>&1
module load gnu12/12.3.0 2>/dev/null || module load gnu12 2>&1
ARCH=${ARCH:-80}
SUFFIX=$([ "$ARCH" = 80 ] || echo "$ARCH")
export LD_LIBRARY_PATH=$HOME/op2sol/install$SUFFIX/petsc/lib:$HOME/op2sol/install$SUFFIX/op2/lib:$HOME/op2sol/install$SUFFIX/op2/lib64:$HOME/hdf5-cmake-par/lib:$LD_LIBRARY_PATH
cd $HOME/op2sol/src/nssolver
time GRIDS="${GRIDS:-273x193 545x385}" BIN=$HOME/op2sol/build$SUFFIX/ns/nssolver_op2_cuda scripts/tmr/flatplate.sh \
  --solve-config configs/solve.json --timing 0 --iterations 250
echo TMR-DONE
