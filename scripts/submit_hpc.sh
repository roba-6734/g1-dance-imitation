#!/usr/bin/env bash
# Submit the video -> GVHMR -> GMR -> 50 Hz reference pipeline as a Slurm GPU job.
# Usage: bash scripts/submit_hpc.sh /abs/path/video.mp4 NAME [python -m video2ref options]
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
config_file=${VIDEO2REF_HPC_CONFIG:-$project_root/configs/hpc.env}
[[ -f "$config_file" ]] || { echo "Missing $config_file; copy configs/hpc.env.example" >&2; exit 1; }
source "$config_file"
video=${1:?Usage: bash scripts/submit_hpc.sh /abs/path/video.mp4 NAME [options]}
name=${2:?NAME required}
shift 2
[[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "Invalid NAME" >&2; exit 2; }
[[ -f "$video" ]] || { echo "Video is missing on the HPC filesystem: $video" >&2; exit 2; }
: "${VIDEO2REF_WORKDIR:?}" "${GVHMR_DIR:?}" "${GMR_DIR:?}" "${GVHMR_PYTHON:?}" "${GMR_PYTHON:?}" "${SLURM_PARTITION:?}"
[[ -x "$GVHMR_PYTHON" && -x "$GMR_PYTHON" ]] || { echo "Python environment missing" >&2; exit 1; }
[[ -f "$GVHMR_DIR/tools/demo/demo.py" && -f "$GMR_DIR/general_motion_retargeting/utils/smpl.py" ]] || {
  echo "GVHMR or GMR checkout missing" >&2; exit 1; }
[[ -f "$GMR_DIR/assets/body_models/smplx/SMPLX_NEUTRAL.npz" ]] || {
  echo "Licensed GMR SMPLX_NEUTRAL.npz is missing" >&2; exit 1; }
mkdir -p "$VIDEO2REF_WORKDIR/slurm" "$VIDEO2REF_WORKDIR/runs"
args=(--job-name="video2ref-$name" --partition="$SLURM_PARTITION"
      --gres="${SLURM_GRES:-gpu:1}" --cpus-per-task="${VIDEO2REF_CPUS:-8}"
      --mem="${VIDEO2REF_MEMORY:-48G}" --time="${VIDEO2REF_TIME:-04:00:00}"
      --output="$VIDEO2REF_WORKDIR/slurm/%x-%j.out" --error="$VIDEO2REF_WORKDIR/slurm/%x-%j.err")
[[ -z "${SLURM_ACCOUNT:-}" ]] || args+=(--account="$SLURM_ACCOUNT")
export VIDEO2REF_HPC_CONFIG="$config_file" VIDEO2REF_PROJECT_ROOT="$project_root"
sbatch "${args[@]}" "$project_root/scripts/job_hpc.sh" "$video" "$name" "$@"
