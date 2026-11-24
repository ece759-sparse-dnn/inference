#!/bin/bash -l
#SBATCH --job-name=sdnn-run
#SBATCH --partition=instruction        # check the exact name with `sinfo`
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1              # override at submit time: sbatch -c 16 ...
#SBATCH --time=00:30:00
#SBATCH --output=sdnn-run-%j.out

# Run one backend once and append its result to a CSV.
#
# Usage, from the repo root (build first with scripts/slurm/build.sh):
#   sbatch [-c CPUS] [--gres=gpu:1] scripts/slurm/run.sbatch BACKEND [SIZE] [LAYERS] [THREADS] [CSV]
#
#   BACKEND  serial | openmp | cuda
#   SIZE     neurons, default 1024
#   LAYERS   120 | 480 | 1920, default 120
#   THREADS  OMP_NUM_THREADS, default = the CPUs allocated with -c (must not exceed them)
#   CSV      default results/euler.csv (created with a header if new)
#
# Examples:
#   sbatch          scripts/slurm/run.sbatch serial 1024 120
#   sbatch -c 16    scripts/slurm/run.sbatch openmp 1024 120 16
#   sbatch --gres=gpu:1 scripts/slurm/run.sbatch cuda 1024 120
#
# Reads data from SDNN_DATA_DIR (default data/ in the repo; populate it with
# scripts/slurm/data.sh). sbatch passes your environment through.
# Optional: OMP_SCHEDULE (e.g. dynamic,64), SDNN_BUILD_DIR (default build/cuda-euler).
set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR:?submit this with sbatch from the repo root}"
export SDNN_DATA_DIR=${SDNN_DATA_DIR:-$PWD/data}

backend=${1:?usage: sbatch [-c CPUS] scripts/slurm/run.sbatch BACKEND [SIZE] [LAYERS] [THREADS] [CSV]}
size=${2:-1024}
layers=${3:-120}
cpus=${SLURM_CPUS_PER_TASK:-1}
threads=${4:-$cpus}
csv=${5:-results/euler.csv}
build=${SDNN_BUILD_DIR:-build/cuda-euler}
bin=$build/apps/sdnn_run

(( threads <= cpus )) || die "$threads threads requested but only $cpus CPUs allocated; submit with: sbatch -c $threads ..."
[[ -f $SDNN_DATA_DIR/bin/neuron$size/.done ]] || die "no converted dataset at $SDNN_DATA_DIR/bin/neuron$size; run scripts/slurm/data.sh $size on the login node"

source scripts/slurm/env.sh
[[ -x $bin ]] || die "$bin not found; run scripts/slurm/build.sh first"

built_on=$(cat "$build/build-cpu.txt" 2>/dev/null || true)
here=$(lscpu | sed -n 's/^Model name: *//p')
if [[ -z $built_on ]]; then
  echo "warning: $build has no build-cpu.txt; build it with scripts/slurm/build.sh so -march=native matches the compute nodes" >&2
elif [[ $built_on != "$here" ]]; then
  echo "warning: built on '$built_on' but running on '$here'; -march=native code may be slower or crash" >&2
fi

if [[ $backend == cuda && -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  echo "warning: no GPU visible; submit cuda runs with --gres=gpu:1" >&2
fi

export OMP_NUM_THREADS=$threads
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}
mkdir -p "$(dirname "$csv")"

echo "host:     $(hostname)"
echo "cpu:      $here"
echo "job:      ${SLURM_JOB_ID:-none}   cpus: $cpus   threads: $threads"
echo "schedule: ${OMP_SCHEDULE:-<implementation default>}"
echo "run:      $backend size=$size layers=$layers -> $csv"
echo

"$bin" --backend "$backend" --size "$size" --layers "$layers" \
  --data-dir "$SDNN_DATA_DIR" --csv "$csv"
