#!/bin/bash -l
# Configure the cuda-euler preset on the login node, then compile it in a
# Slurm job.
#
# Usage, from the repo root:
#   scripts/slurm/build.sh [SBATCH_OPTIONS...]
#
# Configuring downloads GoogleTest and compute nodes may not have internet
# access, so it runs here; it only takes seconds. Compiling is the long part
# and runs on a compute node (scripts/slurm/build.sbatch), which keeps it off
# the login node and lets -march=native target the compute nodes' CPU.
# SBATCH_OPTIONS are passed to sbatch, e.g. --constraint=... or -c 16.
#
# Prints the build job's ID; queue runs behind it with
#   sbatch --dependency=afterok:<jobid> scripts/slurm/run.sbatch ...
#
# Output: build/cuda-euler/apps/sdnn_run and build/cuda-euler/tests/sdnn_tests
set -euo pipefail
cd "$(dirname "$0")/../.."

source scripts/slurm/env.sh

cmake --preset cuda-euler

jobid=$(sbatch --parsable "$@" scripts/slurm/build.sbatch)
jobid=${jobid%%;*}

echo
echo "submitted build job $jobid (log: sdnn-build-$jobid.out)"
echo "queue a run behind it with:"
echo "  sbatch --dependency=afterok:$jobid [--gres=gpu:1] scripts/slurm/run.sbatch BACKEND ..."
