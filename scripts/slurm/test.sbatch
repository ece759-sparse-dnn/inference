#!/bin/bash -l
#SBATCH --job-name=sdnn-test
#SBATCH --partition=instruction        # check the exact name with `sinfo`
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=sdnn-test-%j.out

# Run the full unit test suite, CUDA included, on a GPU node. Every backend
# preset is checked against the truth data by BackendTest.MatchesTruth.
#
# Usage, from the repo root (build first with scripts/slurm/build.sh):
#   sbatch scripts/slurm/test.sbatch
#   sbatch --dependency=afterok:<build jobid> scripts/slurm/test.sbatch
#
# Reads data from SDNN_DATA_DIR (default data/ in the repo); truth checks are
# skipped if bin/neuron1024 isn't converted.
# Optional: GTEST_FILTER (e.g. '*cuda*:*Cuda*'), SDNN_BUILD_DIR (default build/cuda-euler).
set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR:?submit this with sbatch from the repo root}"
export SDNN_DATA_DIR=${SDNN_DATA_DIR:-$PWD/data}

build=${SDNN_BUILD_DIR:-build/cuda-euler}
tests=$build/tests/sdnn_tests
cpus=${SLURM_CPUS_PER_TASK:-1}

source scripts/slurm/env.sh
[[ -x $tests ]] || die "$tests not found; run scripts/slurm/build.sh first"

built_on=$(cat "$build/build-cpu.txt" 2>/dev/null || true)
here=$(lscpu | sed -n 's/^Model name: *//p')
if [[ -z $built_on ]]; then
  echo "warning: $build has no build-cpu.txt; build it with scripts/slurm/build.sh so -march=native matches the compute nodes" >&2
elif [[ $built_on != "$here" ]]; then
  echo "warning: built on '$built_on' but running on '$here'; -march=native code may be slower or crash" >&2
fi

if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  echo "warning: no GPU visible; CUDA tests will fail" >&2
fi
if [[ ! -f $SDNN_DATA_DIR/bin/neuron1024/.done ]]; then
  echo "warning: no converted dataset at $SDNN_DATA_DIR/bin/neuron1024; truth checks will be skipped (run scripts/slurm/data.sh)" >&2
fi

export OMP_NUM_THREADS=$cpus

echo "host:   $(hostname)"
echo "cpu:    $here"
echo "gpu:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
echo "job:    ${SLURM_JOB_ID:-none}   cpus: $cpus"
echo "filter: ${GTEST_FILTER:-<all>}"
echo

"$tests" ${GTEST_FILTER:+--gtest_filter="$GTEST_FILTER"}
