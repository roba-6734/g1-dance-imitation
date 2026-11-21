#!/usr/bin/env bash
# Set up whole-body tracking training (unitree_rl_mjlab, BeyondMimic re-implementation) on a CUDA box.
#   ./scripts/setup_wbt.sh            clone + check
#   ./scripts/setup_wbt.sh --env      also create the conda env "unitree_rl_mjlab" (python 3.11) and pip install -e
# Tested target: one RTX A5000 (24 GB), Ubuntu 22.04, NVIDIA driver >= 550 (Unitree's stated requirements).
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=1425b15f73bd4095f0df53709d7c389c3eb9e790   # unitree_rl_mjlab, pins mjlab==1.2.0 / mujoco-warp==3.5.0
DIR=third_party/unitree_rl_mjlab
mkdir -p third_party
if [ ! -d "$DIR/.git" ]; then
  git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git "$DIR"
fi
git -C "$DIR" fetch -q origin "$COMMIT" 2>/dev/null || true
git -C "$DIR" checkout -q "$COMMIT"
echo "unitree_rl_mjlab at $(git -C "$DIR" rev-parse --short HEAD)"

# The exact G1 model training uses (mjlab 1.2.0 asset_zoo), for our CPU tools outside the training env.
if [ ! -f third_party/mjlab_g1/g1.xml ]; then
  TMP=$(mktemp -d)
  python3 -m pip download -q mjlab==1.2.0 --no-deps -d "$TMP"
  python3 - "$TMP" <<'PY'
import sys, zipfile, glob, os
whl = glob.glob(os.path.join(sys.argv[1], "mjlab-1.2.0-*.whl"))[0]
prefix = "mjlab/asset_zoo/robots/unitree_g1/xmls/"
with zipfile.ZipFile(whl) as z:
    for n in z.namelist():
        if n.startswith(prefix) and not n.endswith("/"):
            out = os.path.join("third_party/mjlab_g1", n[len(prefix):])
            os.makedirs(os.path.dirname(out), exist_ok=True)
            open(out, "wb").write(z.read(n))
PY
  rm -rf "$TMP"
  echo "mjlab 1.2.0 G1 model -> third_party/mjlab_g1/"
fi

if [ "${1:-}" = "--env" ]; then
  command -v conda >/dev/null || { echo "conda not found (see $DIR/doc/setup_en.md)"; exit 1; }
  conda env list | grep -q '^unitree_rl_mjlab ' || conda create -y -n unitree_rl_mjlab python=3.11
  conda run -n unitree_rl_mjlab pip install -e "$DIR"
  # our evaluation tools run in the same env (CPU MuJoCo + onnxruntime)
  conda run -n unitree_rl_mjlab pip install onnxruntime pyyaml scipy pillow matplotlib
fi

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "WARNING: nvidia-smi not found; training needs a CUDA GPU"
fi
cat <<EOF

Deploy side (robot PC), not needed for training:
  sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
  + cyclonedds and unitree_sdk2, then apply our safety patch before building g1_ctrl:
  git -C $DIR apply ../../wbt/patches/unitree_rl_mjlab_mimic_safety.patch
EOF
