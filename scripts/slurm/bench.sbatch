#!/bin/bash -l
#SBATCH --job-name=sdnn-bench
#SBATCH --partition=instruction        # check the exact name with `sinfo`
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=sdnn-bench-%j.out
##SBATCH --exclusive                   # uncomment if the partition allows it

# CPU benchmark: unit tests, truth checks for serial and openmp, then an
# OpenMP schedule sweep with a summary table.
#
# Usage, from the repo root (build first with scripts/slurm/build.sh):
#   sbatch scripts/slurm/bench.sbatch
#
# Reads data from SDNN_DATA_DIR (default data/ in the repo; populate it with
# scripts/slurm/data.sh). Optional overrides (export before sbatch):
#   SIZE=1024  DEPTHS="120 480"  THREADS="1 2 4 8 16 32 64"  REPS=3
#   SCHEDULES="static dynamic,64 guided"  RESULTS_DIR=results/myrun
#   SDNN_BUILD_DIR=build/cuda-euler
set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }
section() { printf '\n==== %s ====\n' "$*"; }

cd "${SLURM_SUBMIT_DIR:?submit this with sbatch from the repo root}"
export SDNN_DATA_DIR=${SDNN_DATA_DIR:-$PWD/data}

cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}
size=${SIZE:-1024}
depths=${DEPTHS:-120}
reps=${REPS:-3}
results=${RESULTS_DIR:-results/${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}}
build=${SDNN_BUILD_DIR:-build/cuda-euler}
bin=$build/apps/sdnn_run
tests=$build/tests/sdnn_tests

# Default thread list: powers of two below the allocation, plus the allocation itself.
if [[ -z ${THREADS:-} ]]; then
  THREADS=""
  for ((t = 1; t < cpus; t *= 2)); do THREADS+="$t "; done
  THREADS+="$cpus"
fi
export THREADS REPS="$reps" SDNN_RUN="$bin" SDNN_DATA_DIR

[[ -f $SDNN_DATA_DIR/bin/neuron$size/.done ]] || die "no converted dataset at $SDNN_DATA_DIR/bin/neuron$size; run scripts/slurm/data.sh $size on the login node"
source scripts/slurm/env.sh
[[ -x $bin && -x $tests ]] || die "$build not built; run scripts/slurm/build.sh first"

built_on=$(cat "$build/build-cpu.txt" 2>/dev/null || true)
here=$(lscpu | sed -n 's/^Model name: *//p')
if [[ -z $built_on ]]; then
  echo "warning: $build has no build-cpu.txt; build it with scripts/slurm/build.sh so -march=native matches the compute nodes" >&2
elif [[ $built_on != "$here" ]]; then
  echo "warning: built on '$built_on' but running on '$here'; -march=native code may be slower or crash" >&2
fi
mkdir -p "$results"

section "node"
echo "host:    $(hostname)"
echo "job:     ${SLURM_JOB_ID:-none}   cpus: $cpus"
lscpu | grep -E 'Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|NUMA node\(s\)' || true
g++ --version | head -1
echo "build:   $build"
echo "data:    $SDNN_DATA_DIR/bin/neuron$size"
echo "results: $results"

section "unit tests (CPU backends only; this job has no GPU)"
export OMP_NUM_THREADS=$cpus
"$tests" --gtest_filter='-*cuda*'

section "correctness (sdnn_run vs truth)"
for depth in $depths; do
  "$bin" --backend serial --size "$size" --layers "$depth" | tail -1
  OMP_SCHEDULE=dynamic,64 "$bin" --backend openmp --size "$size" --layers "$depth" | tail -1
done

section "schedule sweep (size $size, 120 layers, threads: $THREADS, reps: $reps)"
sweep=$results/sweep.csv
scripts/schedule_sweep.sh "$size" 120 "$sweep"

# Serial baseline rows, same CSV shape, for the speedup tables.
for r in $(seq "$reps"); do
  row=$("$bin" --backend serial --size "$size" --layers 120 | tail -1)
  echo "\"serial\",$r,$row" | tee -a "$sweep"
done

section "summary"
python3 scripts/summarize_sweep.py "$sweep" | tee "$results/summary.txt"

section "done"
echo "raw data: $sweep"
echo "summary:  $results/summary.txt"
