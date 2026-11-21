#!/usr/bin/env bash
# Submit whole-body policy training (scripts/train_wbt.sh) as a Slurm GPU job.
# Usage: bash scripts/submit_wbt.sh NAME /abs/path/qpos.npy [wbt.prepare options, e.g. --lowpass-hz 6]
# Uses configs/hpc.env for SLURM_PARTITION / SLURM_ACCOUNT / SLURM_GRES if present.
# Env knobs: WBT_PYTHON (python of the unitree_rl_mjlab env), WBT_TIME (12:00:00), WBT_MEMORY (32G),
#            WBT_CPUS (8), and train_wbt.sh's NUM_ENVS / MAX_ITERS / SAVE_EVERY / RESUME / FORCE.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
config_file=${VIDEO2REF_HPC_CONFIG:-$project_root/configs/hpc.env}
[[ -f "$config_file" ]] && source "$config_file"
name=${1:?Usage: bash scripts/submit_wbt.sh NAME /abs/path/qpos.npy [prepare options]}
qpos=${2:?qpos.npy required}
shift 2
[[ "$name" =~ ^[A-Za-z0-9_]+$ ]] || { echo "NAME: letters, digits, underscores" >&2; exit 2; }
[[ -f "$qpos" ]] || { echo "Missing $qpos" >&2; exit 2; }
command -v sbatch >/dev/null || { echo "sbatch not found: no Slurm here. Run directly instead (see wbt/README.md)." >&2; exit 1; }
if [[ -z "${WBT_PYTHON:-}" ]]; then
  WBT_PYTHON=$(conda run -n unitree_rl_mjlab which python 2>/dev/null || true)
fi
[[ -x "${WBT_PYTHON:-}" ]] || { echo "Set WBT_PYTHON to the unitree_rl_mjlab env's python" >&2; exit 1; }
[[ -d "$project_root/third_party/unitree_rl_mjlab" ]] || { echo "Run scripts/setup_wbt.sh --env first" >&2; exit 1; }
logs="$project_root/runs/slurm"
mkdir -p "$logs"
args=(--job-name="wbt-$name" --gres="${SLURM_GRES:-gpu:1}" --cpus-per-task="${WBT_CPUS:-8}"
      --mem="${WBT_MEMORY:-32G}" --time="${WBT_TIME:-12:00:00}"
      --output="$logs/%x-%j.out" --error="$logs/%x-%j.err")
[[ -z "${SLURM_PARTITION:-}" ]] || args+=(--partition="$SLURM_PARTITION")
[[ -z "${SLURM_ACCOUNT:-}" ]] || args+=(--account="$SLURM_ACCOUNT")
export PYTHON="$WBT_PYTHON" WBT_PROJECT_ROOT="$project_root"
sbatch --export=ALL "${args[@]}" "$project_root/scripts/job_wbt.sh" "$name" "$(realpath "$qpos")" "$@"
echo "logs: $logs/wbt-$name-<jobid>.out   (squeue -u \$USER; tail -f the .out file)"
