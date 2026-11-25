#!/bin/bash
# A video of a dual-time airfoil run from its snapshots (scripts/airfoil/video.py):
# the snapshots' sanity check, then the video, outputs-op2/$CASE/video/TAG_FIELD.mp4.
# A CPU node renders a few frames a second:
#   CASE=oat15a TAG=oat15a_a3.5_video ALPHA=3.5 sbatch scripts/cl/video.sbatch
# FIELD (mach, schlieren, cp), FPS, TITLE and EXTRA (further video.py options) are optional.
#SBATCH -p cpu -N1 -c16 -t 01:00:00 -J video
#SBATCH -o /home/reguly/op2sol/logs/video-%j.out
module load OP2-env 2>&1
cd $HOME/op2sol/src/nssolver
CASE=${CASE:-oat15a}
case $CASE in
  oat15a) MACH=${MACH:-0.73}; RE=${RE:-3e6}; TEMP=${TEMP:-271}; MESH=${MESH:-meshes-op2/oat15a/deck.h5} ;;
  circular_arc) MACH=${MACH:-0.76}; RE=${RE:-11e6}; TEMP=${TEMP:-300}; MESH=${MESH:-meshes-op2/arc/arc18.h5} ;;
  *) echo "unknown CASE $CASE"; exit 1 ;;
esac
O=outputs-op2/$CASE
FIELD=${FIELD:-mach}
args=(--field "$FIELD" --fps "${FPS:-30}")
[ -n "${TITLE:-}" ] && args+=(--title "$TITLE")
mkdir -p $O/video
python3 scripts/airfoil/video.py check $MESH $O $TAG $O/$TAG.history.csv $MACH $RE $TEMP $ALPHA || exit 1
python3 scripts/airfoil/video.py render $MESH $O $TAG $O/$TAG.history.csv $MACH $RE $TEMP $ALPHA \
  $O/video/${TAG}_$FIELD.mp4 "${args[@]}" $EXTRA
