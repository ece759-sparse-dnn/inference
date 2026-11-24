#!/bin/bash -l
#SBATCH --job-name=sdnn-build
#SBATCH --partition=instruction        # keep this the same as the run jobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=sdnn-build-%j.out

# Compile an already-configured build directory on a compute node.
#
# Usage: submitted by scripts/slurm/build.sh, which configures first. To
# rebuild an existing configuration directly, from the repo root:
#   sbatch scripts/slurm/build.sbatch
#
# Records the node's CPU model in <build>/build-cpu.txt; run.sbatch and
# bench.sbatch warn if they land on a different CPU, since -march=native
# code is tuned for (and may only run on) the CPU it was compiled on.
# Optional: SDNN_BUILD_DIR (default build/cuda-euler).
set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR:?submit this with sbatch from the repo root}"

build=${SDNN_BUILD_DIR:-build/cuda-euler}
cpus=${SLURM_CPUS_PER_TASK:-1}
cpu_model=$(lscpu | sed -n 's/^Model name: *//p')

[[ -f $build/build.ninja ]] || die "$build is not configured; run scripts/slurm/build.sh on the login node"

source scripts/slurm/env.sh

echo "host:  $(hostname)"
echo "cpu:   $cpu_model"
echo "job:   ${SLURM_JOB_ID:-none}   cpus: $cpus"
echo "build: $build"
echo

cmake --build "$build" -j "$cpus"

echo "$cpu_model" > "$build/build-cpu.txt"
echo
echo "built $build on $(hostname)"
