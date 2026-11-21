#!/usr/bin/env bash
# Set up whole-body tracking training (unitree_rl_mjlab, BeyondMimic re-implementation) on a CUDA box.
#   ./scripts/setup_wbt.sh            clone + check
#   ./scripts/setup_wbt.sh --env      also create the conda env "unitree_rl_mjlab" (python 3.11) and pip install -e
# Target: one CUDA GPU (A5000 24 GB / RTX 50xx), Ubuntu 22.04, NVIDIA driver >= 550 (Unitree's stated requirements;
# RTX 50xx needs a CUDA 12.8+ PyTorch build). Re-running is safe: finished steps are skipped.
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

if [ "${1:-}" = "--env" ]; then
  command -v conda >/dev/null || { echo "conda not found (see $DIR/doc/setup_en.md)"; exit 1; }
  conda env list | grep -q '^unitree_rl_mjlab ' || conda create -y -n unitree_rl_mjlab python=3.11
  conda run --no-capture-output -n unitree_rl_mjlab python -m pip install -e "$DIR"
  # mujoco-warp 3.5.0 only asks for mujoco>=3.4, but MuJoCo 4.0 removed mjENBL_MULTICCD, which it uses at import.
  # warp-lang >= 1.13 dropped warp.context, which mjlab 1.2.0 reads (sim.py _should_use_cuda_graph); 1.12.1 is
  # the newest release that has it, built with CUDA 12.9 (supports RTX 50xx / sm_120).
  conda run --no-capture-output -n unitree_rl_mjlab python -m pip install "mujoco==3.5.0" "warp-lang==1.12.1"
  # our evaluation tools run in the same env (CPU MuJoCo + onnxruntime)
  conda run --no-capture-output -n unitree_rl_mjlab python -m pip install onnxruntime pyyaml scipy pillow matplotlib
fi

# The exact G1 model training uses (mjlab 1.2.0 asset_zoo), copied for our CPU tools outside the training env.
# Inside the env the tools find it in the installed mjlab package, so failure here is only a warning.
if [ ! -f third_party/mjlab_g1/g1.xml ]; then
  SRC=$(conda run -n unitree_rl_mjlab python -c "import mjlab, pathlib; print(pathlib.Path(mjlab.__file__).parent / 'asset_zoo/robots/unitree_g1/xmls')" 2>/dev/null || true)
  if [ -n "$SRC" ] && [ -f "$SRC/g1.xml" ]; then
    mkdir -p third_party/mjlab_g1 && cp -r "$SRC"/. third_party/mjlab_g1/
    echo "mjlab G1 model -> third_party/mjlab_g1/ (from the env)"
  else
    echo "NOTE: mjlab G1 model not copied (no env yet). Tools inside the unitree_rl_mjlab env do not need it."
  fi
fi

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "WARNING: nvidia-smi not found; training needs a CUDA GPU"
fi
if conda env list 2>/dev/null | grep -q '^unitree_rl_mjlab '; then
  echo "GPU check (torch + warp must both see the card):"
  conda run --no-capture-output -n unitree_rl_mjlab python -c "
import torch
print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
    print('kernel test', (torch.ones(4, device='cuda') * 2).sum().item())
import mujoco, mujoco_warp, mjlab
print('mujoco', mujoco.__version__, '+ mujoco_warp + mjlab import ok')
import warp as wp
wp.init()
print('warp', wp.config.version, 'devices', wp.get_cuda_devices())
" || echo "GPU check FAILED: send this output (RTX 50xx needs a CUDA 12.8+ torch build)"
fi
cat <<EOF

Deploy side (robot PC), not needed for training:
  sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
  + cyclonedds and unitree_sdk2, then apply our safety patch before building g1_ctrl:
  git -C $DIR apply ../../wbt/patches/unitree_rl_mjlab_mimic_safety.patch
EOF
