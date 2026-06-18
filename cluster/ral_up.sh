#!/usr/bin/env bash
# Deterministic bring-up of the fixed RAL Ray cluster (no autoscaler).
#
#   bash cluster/ral_up.sh [--no-sync] [--reinstall] [--allow-large]
#
#   --no-sync      Restart Ray only; skip the source rsync to workers.
#   --reinstall    Re-run `pip install --no-deps -e <SRC>` on workers (deps changed).
#   --allow-large  Proceed even if the payload guard finds files > THRESHOLD_MB.
#
# Node specifics + tunables live in cluster/ral_nodes.conf.  Run from the head node.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="$SCRIPT_DIR/ral_nodes.conf"
[[ -f "$CONF" ]] || { echo "missing config: $CONF" >&2; exit 1; }

# Defaults (overridable via KEY=value lines in the conf)
PORT=6379
SRC=/home/nico/panoseti_analysis
CONDA_SH='~/miniconda3/etc/profile.d/conda.sh'
THRESHOLD_MB=50

NO_SYNC=0; REINSTALL=0; ALLOW_LARGE=0
for arg in "$@"; do
  case "$arg" in
    --no-sync) NO_SYNC=1 ;;
    --reinstall) REINSTALL=1 ;;
    --allow-large) ALLOW_LARGE=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

# ── Parse the conf into parallel arrays + tunables ───────────────────────────────
declare -a ROLES IPS GPUS ENVS LABELS
while read -r f1 f2 f3 f4 f5 _; do
  [[ -z "${f1:-}" || "${f1:0:1}" == "#" ]] && continue
  if [[ "$f1" == *=* && -z "${f2:-}" ]]; then        # tunable KEY=value
    key="${f1%%=*}"; val="${f1#*=}"
    case "$key" in
      PORT) PORT="$val" ;; SRC) SRC="$val" ;;
      CONDA_SH) CONDA_SH="$val" ;; THRESHOLD_MB) THRESHOLD_MB="$val" ;;
    esac
    continue
  fi
  ROLES+=("$f1"); IPS+=("$f2"); GPUS+=("$f3"); ENVS+=("$f4"); LABELS+=("$f5")
done < "$CONF"

HEAD_IP=""
for i in "${!ROLES[@]}"; do [[ "${ROLES[$i]}" == "head" ]] && HEAD_IP="${IPS[$i]}"; done
[[ -n "$HEAD_IP" ]] || { echo "no 'head' row in $CONF" >&2; exit 1; }

# Single source of truth for both rsync and the payload guard.  The dir-merge filter
# makes rsync honor every .gitignore (so gitignored data/caches never ship — the bulk of
# the tree); the explicit excludes are belt-and-suspenders for anything not gitignored.
EXCLUDES=(
  --filter=':- .gitignore'
  --exclude='.git' --exclude='work' --exclude='.venv' --exclude='*.zarr'
  --exclude='results' --exclude='results_*' --exclude='.mypy_cache'
  --exclude='.ruff_cache' --exclude='.claude' --exclude='__pycache__'
  --exclude='.ipynb_checkpoints' --exclude='assets' --exclude='ml/*/data'
  --exclude='ml/*/cache' --exclude='ml/*/models'
)

SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"

is_local() { [[ " $(hostname -I 2>/dev/null) " == *" $1 "* ]]; }

# Run a command string on a node: locally if it is this host, else over SSH.
run_on() {
  local ip="$1"; shift
  if is_local "$ip"; then bash -lc "$*"
  else ssh $SSH_OPTS "nico@$ip" "$*"; fi
}

# ── Payload guard: abort if rsync would ship oversized files ─────────────────────
payload_guard() {
  local tmp big
  tmp="$(mktemp -d)"
  big="$(rsync -rn --out-format='%l %n' "${EXCLUDES[@]}" "$SRC/" "$tmp/" 2>/dev/null \
        | awk -v t=$((THRESHOLD_MB*1024*1024)) '$1+0>t {printf "  %d MB\t%s\n",$1/1048576,$2}' || true)"
  rmdir "$tmp" 2>/dev/null || true
  if [[ -n "$big" ]]; then
    echo "!! Large files in the rsync payload (> ${THRESHOLD_MB} MB):" >&2
    echo "$big" >&2
    echo "   Add them to EXCLUDES (cluster/ral_up.sh) + .rayignore, or pass --allow-large." >&2
    if [[ "$ALLOW_LARGE" != 1 ]]; then echo "   Aborting." >&2; exit 3; fi
    echo "   --allow-large set; continuing." >&2
  fi
}

start_node() {
  local i="$1"
  local role="${ROLES[$i]}" ip="${IPS[$i]}" gpus="${GPUS[$i]}" env="${ENVS[$i]}" label="${LABELS[$i]}"
  # NCCL/RoCE v2 config for the 400G ConnectX-7 NICs (mlx5_0 on both GPU nodes).
  # Each var is overridable by setting it in the caller's environment before running ral_up.sh.
  # NCCL_IB_GID_INDEX=0: selects the link-local IPv6 GID (fe80::..., always present).
  #   GID 3 (IPv4 RoCEv2) is all-zero on this cluster because ens7np0 has no configured
  #   IPv4 address; ibv_modify_qp(RTR) fails with errno 61 when given a zero GID.
  # RDMAV_FORK_SAFE=1: required because Ray forks worker processes; without it ibverbs can
  #   deadlock or corrupt state on fork.
  # ulimit -l unlimited: IB memory registration (ibv_reg_mr) pins buffers in RAM.  The Linux
  #   default locked-memory limit (64 KB) is far too small; NCCL fails with
  #   NCCL_ERROR_SYSTEM_ERROR if it cannot register its CPU bounce buffers.
  local nccl_env="export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0} NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0} NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-0} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eno2} NCCL_DEBUG=${NCCL_DEBUG:-INFO} RDMAV_FORK_SAFE=${RDMAV_FORK_SAFE:-1}"
  local res=""
  local pre="source $CONDA_SH && conda activate $env && $nccl_env && ulimit -l unlimited"
  # JSON is escaped so it survives the local shell -> ssh -> remote shell hops.
  [[ "$label" != "-" ]] && res="--resources={\\\"accelerator_type:$label\\\":$gpus}"
  if [[ "$role" == "head" ]]; then
    echo ">> head   $ip  (env=$env gpus=$gpus label=$label)"
    run_on "$ip" "$pre && ray stop >/dev/null 2>&1; ray start --head --port=$PORT --dashboard-host=0.0.0.0 --num-gpus=$gpus $res"
  else
    echo ">> worker $ip  (env=$env gpus=$gpus label=$label)"
    if [[ "$NO_SYNC" == 0 ]] && ! is_local "$ip"; then
      rsync -a --delete -e "ssh $SSH_OPTS" "${EXCLUDES[@]}" "$SRC/" "nico@$ip:$SRC/"
      [[ "$REINSTALL" == 1 ]] && run_on "$ip" "$pre && pip install -q --no-deps -e $SRC"
    fi
    run_on "$ip" "$pre && ray stop >/dev/null 2>&1; ray start --address=$HEAD_IP:$PORT --num-gpus=$gpus $res"
  fi
}

[[ "$NO_SYNC" == 0 ]] && payload_guard

for i in "${!ROLES[@]}"; do [[ "${ROLES[$i]}" == "head" ]] && start_node "$i"; done
sleep 3   # let the head's GCS come up before workers attach
for i in "${!ROLES[@]}"; do [[ "${ROLES[$i]}" != "head" ]] && start_node "$i"; done

# Wait for every node to register before printing status (raylets attach a few seconds
# after `ray start` returns) so the summary reliably shows the full cluster.
want=${#ROLES[@]}
echo; echo "Waiting for $want nodes to register..."
python - "$want" <<'PY' 2>/dev/null || true
import ray, sys, time
ray.init(address="auto", logging_level="ERROR")
want = int(sys.argv[1])
for _ in range(15):
    if sum(1 for n in ray.nodes() if n.get("Alive")) >= want:
        break
    time.sleep(2)
PY

echo; echo "== ray status =="; ray status || true
