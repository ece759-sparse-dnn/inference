#!/bin/bash -l
# Download and convert datasets with sdnn-tools.
#
# Usage, on the Euler login node (compute nodes may not have internet access):
#   scripts/slurm/data.sh [SIZE...]
#
#   SIZE     1024 | 4096 | 16384 | 65536, default 1024
#
# Euler is for small-dataset testing; 1024 is the intended size. Larger sizes
# work but count against your home quota (4096 is ~2 GB converted).
#
# Installs uv into ~/.local/bin if it isn't already on PATH; uv then fetches
# its own Python, so no Python module is needed. Nothing is added to your
# shell rc files.
#
# Writes to SDNN_DATA_DIR (default data/ in the repo).
# Optional: UV_VERSION (default 0.13.1), UV_INSTALL_DIR (default ~/.local/bin),
# PRUNE_RAW=1 to delete each size's raw TSVs once it is converted.
set -euo pipefail
cd "$(dirname "$0")/../.."

die() { echo "error: $*" >&2; exit 1; }

export SDNN_DATA_DIR=${SDNN_DATA_DIR:-$PWD/data}
(( $# )) || set -- 1024

uv_version=${UV_VERSION:-0.13.1}
uv_dir=${UV_INSTALL_DIR:-$HOME/.local/bin}

todo=()
for size in "$@"; do
  if [[ -f $SDNN_DATA_DIR/bin/neuron$size/.done ]]; then
    echo "neuron$size: already converted"
  else
    todo+=("$size")
  fi
done
(( ${#todo[@]} )) || exit 0

if ! command -v uv >/dev/null 2>&1; then
  if [[ ! -x $uv_dir/uv ]]; then
    command -v curl >/dev/null 2>&1 || die "curl not found; can't install uv"
    echo "installing uv $uv_version into $uv_dir"
    curl -LsSf "https://astral.sh/uv/$uv_version/install.sh" |
      env UV_INSTALL_DIR="$uv_dir" UV_NO_MODIFY_PATH=1 sh
  fi
  export PATH=$uv_dir:$PATH
fi

mkdir -p "$SDNN_DATA_DIR"

sdnn_tools() {
  uv run --locked --project python sdnn-tools --data-dir "$SDNN_DATA_DIR" "$@"
}

echo "data:  $SDNN_DATA_DIR"
echo "sizes: ${todo[*]}"
echo

sdnn_tools download -s "${todo[@]}"
sdnn_tools convert -s "${todo[@]}"

if [[ ${PRUNE_RAW:-0} == 1 ]]; then
  for size in "${todo[@]}"; do
    [[ -f $SDNN_DATA_DIR/bin/neuron$size/.done ]] || die "neuron$size not converted; keeping raw files"
    rm -rf "$SDNN_DATA_DIR/raw/neuron$size" "$SDNN_DATA_DIR/raw/.neuron$size.done"
  done
fi

echo
echo "converted into $SDNN_DATA_DIR/bin"
