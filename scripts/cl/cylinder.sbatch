#!/bin/bash
# A cylinder run (scripts/cylinder/run.sh) on one A100 of renyi:
#   GRID="384 256" FIRST=0.00125 DT=1.75e-3 TRIGGER=300 STEPS=1500 sbatch scripts/cl/cylinder.sbatch
#SBATCH -p gpu -w renyi -N1 -c8 --gres=gpu:1 -t 06:00:00 -J cyl
#SBATCH -o /home/reguly/op2sol/logs/cyl-%j.out
module load OP2-env gnu12/12.3.0 2>&1
export LD_LIBRARY_PATH=$HOME/op2sol/install/petsc/lib:$HOME/op2sol/install/op2/lib:$HOME/op2sol/install/op2/lib64:$HOME/hdf5-cmake-par/lib:$LD_LIBRARY_PATH
cd $HOME/op2sol/src/nssolver
echo "GRID=$GRID FIRST=$FIRST DT=$DT TRIGGER=$TRIGGER STEPS=$STEPS"
BIN=$HOME/op2sol/build/ns/nssolver_op2_cuda scripts/cylinder/run.sh --solve-config configs/solve.json \
  --inner-iterations 30 --progress-interval 100
grep -E "step\] .*(/|0/)" outputs-op2/cyl/*${DT}*.log | tail -3
echo CYL-DONE
