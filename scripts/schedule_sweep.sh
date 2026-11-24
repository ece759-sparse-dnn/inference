#!/usr/bin/env bash
# Sweep OpenMP schedules and thread counts for the openmp backend.
#
# Usage:
#   scripts/schedule_sweep.sh [SIZE] [LAYERS] [OUT_CSV]
#
# Override the sweep with environment variables:
#   THREADS="1 2 4 8 16"   SCHEDULES="static dynamic,64 guided"   REPS=3
#   SDNN_RUN=./build/release/apps/sdnn_run
#
# Requires a release build. Stops on the first FAIL (sdnn_run exits nonzero).
set -euo pipefail

size=${1:-1024}
layers=${2:-120}
out=${3:-schedule_sweep.csv}

bin=${SDNN_RUN:-./build/release/apps/sdnn_run}
threads=${THREADS:-"1 2 4 8 16"}
schedules=${SCHEDULES:-"static static,64 dynamic,1 dynamic,16 dynamic,64 dynamic,256 guided guided,16"}
reps=${REPS:-3}

# Pin threads to cores so repeated runs are comparable.
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}

if [[ ! -x $bin ]]; then
  echo "error: $bin not found; build the release preset first" >&2
  exit 1
fi

echo "schedule,rep,backend,neurons,layers,images,threads,seconds,edges_per_second,pass" > "$out"
for t in $threads; do
  for s in $schedules; do
    for r in $(seq "$reps"); do
      row=$(OMP_NUM_THREADS=$t OMP_SCHEDULE=$s \
        "$bin" --backend openmp --size "$size" --layers "$layers" | tail -1)
      # The schedule can contain a comma ("dynamic,64"), so quote it.
      echo "\"$s\",$r,$row" | tee -a "$out"
    done
  done
done

echo "wrote $out" >&2
