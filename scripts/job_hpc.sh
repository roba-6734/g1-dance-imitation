#!/usr/bin/env bash
set -euo pipefail
: "${VIDEO2REF_HPC_CONFIG:?Submit through scripts/submit_hpc.sh}"
source "$VIDEO2REF_HPC_CONFIG"
project_root=${VIDEO2REF_PROJECT_ROOT:?}
video=${1:?}
name=${2:?}
shift 2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MUJOCO_GL=${MUJOCO_GL:-egl}
output="$VIDEO2REF_WORKDIR/runs/${name}_${SLURM_JOB_ID:?}"
cd "$project_root"
echo "Video: $video"; echo "Output: $output"
nvidia-smi
exec "$GMR_PYTHON" -m video2ref --video "$video" --output-dir "$output" \
  --gvhmr-repo "$GVHMR_DIR" --gmr-repo "$GMR_DIR" --gvhmr-python "$GVHMR_PYTHON" "$@"
