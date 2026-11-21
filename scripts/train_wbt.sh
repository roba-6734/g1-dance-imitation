#!/usr/bin/env bash
# Prepare -> screen -> convert -> train one whole-body tracking policy (run inside the unitree_rl_mjlab env).
#
#   ./scripts/train_wbt.sh NAME QPOS_NPY [prepare options...]
#   ./scripts/train_wbt.sh ayyala_gmr outputs/ayyala/qpos.npy --lowpass-hz 6
#   ./scripts/train_wbt.sh ayyala_nmr outputs/ayyala_nmr/qpos.npy
#
# Environment knobs: NUM_ENVS (default 4096; use 2048 if the A5000 runs out of memory),
# MAX_ITERS (default 20000), SAVE_EVERY (default 500), LOGGER (tensorboard|wandb, default tensorboard),
# FORCE=1 to train despite a feasibility FAIL (do not take such a policy to hardware),
# RESUME=1 to continue the latest run of NAME from its last checkpoint (e.g. after a time limit),
# PYTHON (default: python on PATH) = the unitree_rl_mjlab env's interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?NAME}; QPOS=${2:?QPOS_NPY}; shift 2
ROOT=$(pwd)
URL=$ROOT/third_party/unitree_rl_mjlab
[ -d "$URL" ] || { echo "run ./scripts/setup_wbt.sh --env first"; exit 1; }
NUM_ENVS=${NUM_ENVS:-4096}; MAX_ITERS=${MAX_ITERS:-20000}; SAVE_EVERY=${SAVE_EVERY:-500}; LOGGER=${LOGGER:-tensorboard}
CSV=$ROOT/wbt_motions/$NAME.csv
PY=${PYTHON:-python}
"$PY" -c "import mjlab, mujoco_warp, onnxruntime" || { echo "$PY lacks mjlab/onnxruntime: activate the unitree_rl_mjlab env or set PYTHON"; exit 1; }
RESUME_ARGS=()
if [ "${RESUME:-0}" = "1" ]; then
  RESUME_ARGS=(--agent.resume=True "--agent.load-run=.*_${NAME}")
fi

echo "== 1/4 prepare $QPOS -> $CSV"
"$PY" -m wbt.prepare --qpos "$QPOS" --out "$CSV" "$@"

echo "== 2/4 feasibility screen"
if ! "$PY" -m wbt.feasibility --csv "$CSV"; then
  if [ "${FORCE:-0}" != "1" ]; then
    echo "Feasibility FAIL: fix the reference (trim, --lowpass-hz, --root-xy lock, re-retarget) or FORCE=1 for a sim-only experiment."
    exit 1
  fi
  echo "FORCE=1: training a motion that failed feasibility. Sim-only; not for hardware."
fi

echo "== 3/4 csv_to_npz (mjlab FK, 30 -> 50 Hz)"
(cd "$URL" && "$PY" scripts/csv_to_npz.py --input-file "$CSV" --output-name "$NAME.npz" \
   --input-fps 30 --output-fps 50 --robot g1)
NPZ=$URL/src/assets/motions/g1/$NAME.npz
[ -f "$NPZ" ] || { echo "expected $NPZ"; exit 1; }

echo "== 4/4 train (No-State-Estimation task: the only observation set State_Mimic implements)"
# Flags follow unitree_rl_mjlab's README; if tyro rejects one, check: python scripts/train.py <task> --help
(cd "$URL" && "$PY" scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
   --motion_file="$NPZ" --env.scene.num-envs="$NUM_ENVS" \
   --agent.max-iterations="$MAX_ITERS" --agent.save-interval="$SAVE_EVERY" \
   --agent.logger="$LOGGER" --agent.run-name="$NAME" ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"})

RUN=$(ls -dt "$URL"/logs/rsl_rl/g1_tracking/*_"$NAME" | head -1)
cat <<EOF

Trained: $RUN
Next (CPU is fine):
  python -m wbt.deploy_pkg --run-dir "$RUN" --motion-npz "$NPZ" --name $NAME --no-install
  # PASS -> rerun without --no-install to copy into the deploy tree; FAIL -> read wbt_deploy/$NAME/sim2sim/report.md
EOF
