#!/bin/bash
# The wall state of each snapshot TAG_NNNNNN.h5 of a dual-time run (snapshot_interval)
# as wall/TAG_NNNNNN_wall.csv beside it, for scripts/airfoil/cpstats.py; snapshots
# already converted are skipped. A CPU node, about 2 s a snapshot on the OAT15A grid:
#   CASE=oat15a TAG=oat15a_a3.5 sbatch scripts/cl/wall_series.sbatch
#SBATCH -p cpu -N1 -c4 -t 01:00:00 -J wallseries
#SBATCH -o /home/reguly/op2sol/logs/wallseries-%j.out
module load OP2-env 2>&1
module load gnu12/12.3.0 2>/dev/null || module load gnu12 2>&1
export LD_LIBRARY_PATH=$HOME/hdf5-cmake-par/lib:$LD_LIBRARY_PATH
cd $HOME/op2sol/src/nssolver
CASE=${CASE:-oat15a}
case $CASE in
  oat15a) MESH=${MESH:-meshes-op2/oat15a/deck.h5} ;;
  circular_arc) MESH=${MESH:-meshes-op2/arc/arc18.h5} ;;
esac
O=outputs-op2/$CASE
mkdir -p $O/wall
n=0
for f in $O/${TAG}_[0-9][0-9][0-9][0-9][0-9][0-9].h5; do
  p=$O/wall/$(basename $f .h5)
  [ -s ${p}_wall.csv ] && continue
  .helpers/bin/nssolver_op2_benchmark_postprocess_helper wall $MESH $f $p 0 > /dev/null || { echo "FAILED $f"; exit 1; }
  n=$((n + 1))
done
echo "converted $n, total $(ls $O/wall/${TAG}_*_wall.csv | wc -l)"
