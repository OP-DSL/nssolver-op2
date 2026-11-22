#!/usr/bin/env bash
# NASA TMR 2D zero-pressure-gradient flat plate, SA verification: fetches the
# TMR PLOT3D grids, preprocesses, runs the implicit scheme and compares the
# skin friction at x = 0.97 and the drag with the TMR reference values.
#
#   BIN=build/nssolver_op2_cuda GRIDS="35x25 69x49 137x97" scripts/tmr/flatplate.sh [--option value ...]
#
# Extra arguments go to every run. Boundaries as TMR's: total pressure and
# temperature inflow at x = -1/3, symmetry ahead of the plate (x < 0), the
# plate from x = 0 to 2, pressure outflow at x = 2 and a Riemann far field at
# the top (y = 1).
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$APP_DIR"
BIN=${BIN:-build/nssolver_op2_seq}
GRIDS=${GRIDS:-"35x25 69x49 137x97"}
LAUNCH=${LAUNCH:-}
MESHES=meshes-op2/tmr
OUT=outputs-op2/tmr
mkdir -p "$MESHES" "$OUT"

declare -A FILE=([545x385]=flatplate_clust2.p2dfmt.gz
                 [273x193]=flatplate_clust2_1leveldown_273x193.p2dfmt.gz
                 [137x97]=flatplate_clust2_2levelsdown_137x97.p2dfmt
                 [69x49]=flatplate_clust2_3levelsdown_69x49.p2dfmt
                 [35x25]=flatplate_clust2_4levelsdown_35x25.p2dfmt)
# All TMR flat-plate grids in one archive (the finer two are only there).
ZIP_URL=https://www.nasa.gov/wp-content/uploads/2026/02/flatplategrids-grids.zip

# rho and u of configs/tmr_flatplate.cfg, for the coefficients.
RHO=$(sed -n 's/^rho *= *//p' configs/tmr_flatplate.cfg)
U=$(sed -n 's/^u *= *//p' configs/tmr_flatplate.cfg)

for grid in $GRIDS; do
  file=${FILE[$grid]}
  p2d=$MESHES/flatplate_$grid.p2dfmt
  if [ ! -s "$p2d" ]; then
    [ -s "$MESHES/flatplategrids.zip" ] || curl -sSfL --max-time 600 -o "$MESHES/flatplategrids.zip" "$ZIP_URL"
    python3 -c 'import sys, zipfile, gzip
z = zipfile.ZipFile(sys.argv[1])
name = next(n for n in z.namelist() if n.endswith("/" + sys.argv[2]))
data = z.read(name)
open(sys.argv[3], "wb").write(gzip.decompress(data) if name.endswith(".gz") else data)' \
      "$MESHES/flatplategrids.zip" "$file" "$p2d"
  fi
  mesh=$MESHES/flatplate_$grid.h5
  [ -s "$mesh" ] || .helpers/bin/nssolver_preprocess_op2_helper plot3d2d "$mesh" "$p2d" \
                       wall_x0=0 wall_x1=2 inlet=total outlet=outlet upper=riemann lower_other=slip > /dev/null
  $LAUNCH "$BIN" --config configs/tmr_flatplate.cfg --mesh-file "$mesh" --output-file "$OUT/flatplate_$grid.h5" \
    "$@" > "$OUT/flatplate_$grid.log" 2>&1 || { echo "$grid: run failed, see $OUT/flatplate_$grid.log"; continue; }
  .helpers/bin/nssolver_op2_benchmark_postprocess_helper flatplate "$mesh" "$OUT/flatplate_$grid.h5" \
    "$OUT/flatplate_$grid" 0 > /dev/null
  python3 scripts/tmr/flatplate_forces.py "$OUT/flatplate_${grid}_wall.csv" "$RHO" "$U" "$grid $(grep -oE 'L2/L2_0=[^ ]+' "$OUT/flatplate_$grid.log" | tail -1)"
done
