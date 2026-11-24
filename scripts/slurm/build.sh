#!/bin/bash -l
# Build the cuda-euler preset with the modules from the README.
#
# Usage, on the Euler login node (configuring downloads GoogleTest, and
# compute nodes may not have internet access):
#   scripts/slurm/build.sh
#
# Output: build/cuda-euler/apps/sdnn_run and build/cuda-euler/tests/sdnn_tests
set -euo pipefail
cd "$(dirname "$0")/../.."

source scripts/slurm/env.sh

cmake --preset cuda-euler
cmake --build --preset cuda-euler

echo "built build/cuda-euler/apps/sdnn_run"
