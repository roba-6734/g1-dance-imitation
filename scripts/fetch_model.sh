#!/usr/bin/env bash
# Fetch the MuJoCo Menagerie Unitree G1 model (used for simulation and rendering only).
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=370c43965323c0944b433b202aac5058b5f907a9
mkdir -p third_party
if [ ! -d third_party/mujoco_menagerie ]; then
  git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
fi
cd third_party/mujoco_menagerie
git sparse-checkout set unitree_g1
git fetch --depth 1 origin "$COMMIT" 2>/dev/null || true
git checkout -q "$COMMIT"
echo "Menagerie G1 at $(git rev-parse --short HEAD)"
