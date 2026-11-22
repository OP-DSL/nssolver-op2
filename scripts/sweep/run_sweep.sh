#!/usr/bin/env bash
# Solves each system file with a sweep (make_sweep.py), one driver launch per
# system and the cases inside it; a case that ends the run (a setting the
# library rejects, a crash, a hang past CASE_TIMEOUT seconds) is recorded as failed and the launch resumed at the
# next. Appends to CSV (summarize.py reads it); the logs go beside it.
#
#   LAUNCH="srun -n 4 --gpus-per-task=1" DRIVER=.../solve_hdf5_6_mpi_cuda \
#     run_sweep.sh SWEEPDIR OUT.csv FILE.h5...        (REPEAT=3 by default)
set -uo pipefail
sweep=$1 csv=$2; shift 2
: "${DRIVER:?DRIVER names the solve_hdf5_<n>_<variant> binary}"
LAUNCH=${LAUNCH:-mpirun -n 2}
REPEAT=${REPEAT:-3}
CASE_TIMEOUT=${CASE_TIMEOUT:-900}   # seconds without a new line of output: the case hung
for f in "$@"; do
  name=$(basename "$f" .h5)
  log="${csv%.csv}.$name.log"; : > "$log"
  # one launch per matrix: its values are then freed from the device after the build
  for cases in "$sweep"/cases.*.txt; do
    total=$(grep -vc '^[[:space:]]*\(#\|$\)' "$cases")
    from=0
    while [ "$from" -lt "$total" ]; do
      out=$(mktemp "${csv%.csv}.run.XXXXXX")   # not TMPDIR: a job's may be missing
      $LAUNCH "$DRIVER" -file "$f" -config "$sweep/config.json" -cases "$cases" \
          -repeat "$REPEAT" -from "$from" -csv "$csv" > "$out" 2>&1 &
      pid=$!; timed=0
      while kill -0 $pid 2>/dev/null; do
        sleep 5
        if [ $(( $(date +%s) - $(stat -c %Y "$out") )) -gt "$CASE_TIMEOUT" ]; then
          timed=1; kill $pid 2>/dev/null; sleep 5; kill -9 $pid 2>/dev/null; break
        fi
      done
      wait $pid 2>/dev/null; rc=$?
      [ $timed -eq 1 ] && rc=124
      cat "$out" >> "$log"
      last=$(grep -aE '^\[[0-9]+/[0-9]+\] ' "$out" | tail -1)
      if [ $rc -eq 0 ] || [ -z "$last" ]; then
        rm -f "$out"; [ $rc -ne 0 ] && echo "$name: driver failed before a case (see $log)"; break
      fi
      k=${last#\[}; k=${k%%/*}
      if ! awk -v n="$k" 'f && /^  -> /{ok=1} $0 ~ "^\\[" n "/"{f=1} END{exit !ok}' "$out"; then
        label=$(sed -E 's/^\[[0-9]+\/[0-9]+\] ([^:]*):.*/\1/' <<<"$last")
        mat=$(sed -E 's/.*matrix ([^,]*), solver.*/\1/' <<<"$last"); sol=${last##*solver }
        code=-1; [ $rc -eq 124 ] && code=-2     # -2: timed out
        echo "$name.h5,$label,$mat,$sol,$code,0,,,,,,,," >> "$csv"
        echo "$name: case $k ($label) $([ $code = -2 ] && echo timed out || echo failed); resuming at $((k + 1))"
      fi
      rm -f "$out"
      from=$k
    done
  done
done
