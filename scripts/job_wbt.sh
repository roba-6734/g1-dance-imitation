#!/usr/bin/env bash
# Slurm job body for scripts/submit_wbt.sh (do not call directly).
set -euo pipefail
cd "${WBT_PROJECT_ROOT:?Submit through scripts/submit_wbt.sh}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MUJOCO_GL=${MUJOCO_GL:-egl}
echo "job ${SLURM_JOB_ID:-?} on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
exec bash scripts/train_wbt.sh "$@"
