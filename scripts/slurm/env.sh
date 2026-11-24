# Euler environment for this project: the modules listed in the README.
# Source this file (`source scripts/slurm/env.sh`); don't run it.

_sdnn_opts=$-
set +eu   # module init scripts often trip `set -u`

if ! command -v module >/dev/null 2>&1; then
  # Batch and non-interactive shells may not have the module command yet.
  for _init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do
    [[ -f $_init ]] && source "$_init" && break
  done
fi

_sdnn_ok=1
if ! command -v module >/dev/null 2>&1; then
  echo "error: 'module' command not found; are you on Euler?" >&2
  _sdnn_ok=0
else
  module purge
  module load gnu15/15.2.0 nvidia/cuda/13.0.0 || {
    echo "error: module load failed" >&2
    _sdnn_ok=0
  }
fi

[[ $_sdnn_opts == *e* ]] && set -e
[[ $_sdnn_opts == *u* ]] && set -u
unset _sdnn_opts _init
if [[ $_sdnn_ok == 0 ]]; then
  unset _sdnn_ok
  return 1 2>/dev/null || exit 1
fi
unset _sdnn_ok
