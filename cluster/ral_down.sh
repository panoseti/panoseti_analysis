#!/usr/bin/env bash
# Tear down the RAL Ray cluster: `ray stop` on every node in cluster/ral_nodes.conf.
# (Stops Ray only; the machines stay up.)  Run from the head node.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="$SCRIPT_DIR/ral_nodes.conf"
[[ -f "$CONF" ]] || { echo "missing config: $CONF" >&2; exit 1; }

CONDA_SH='~/miniconda3/etc/profile.d/conda.sh'
declare -a IPS ENVS
while read -r f1 f2 f3 f4 f5 _; do
  [[ -z "${f1:-}" || "${f1:0:1}" == "#" ]] && continue
  if [[ "$f1" == *=* && -z "${f2:-}" ]]; then
    [[ "${f1%%=*}" == CONDA_SH ]] && CONDA_SH="${f1#*=}"
    continue
  fi
  IPS+=("$f2"); ENVS+=("$f4")
done < "$CONF"

is_local() { [[ " $(hostname -I 2>/dev/null) " == *" $1 "* ]]; }

for i in "${!IPS[@]}"; do
  ip="${IPS[$i]}"; env="${ENVS[$i]}"
  echo ">> ray stop on $ip"
  cmd="source $CONDA_SH && conda activate $env && ray stop"
  if is_local "$ip"; then bash -lc "$cmd" || true
  else ssh -o BatchMode=yes -o ConnectTimeout=10 "nico@$ip" "$cmd" || true; fi
done
