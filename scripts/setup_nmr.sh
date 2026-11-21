#!/usr/bin/env bash
# Set up NMR (Make Tracking Easy: pretrained neural motion retargeting, SMPL-X -> G1 29-DoF).
#   ./scripts/setup_nmr.sh          clone the pinned commit into third_party/NMR
#   ./scripts/setup_nmr.sh --env    also create conda env "nmr" (python 3.10) and download the checkpoint
# Needs network access to github.com, pypi.org, huggingface.co and its file CDN (us.aws.cdn.hf.co);
# download.pytorch.org (+ download-r2.pytorch.org) only for CUDA_TAG wheels. CUDA_TAG=pypi uses PyPI's torch.
# The NMR repository has no LICENSE file: use it for research evaluation only.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=ba87ad6541f6512df9a370acdd2c2d95da7ae419
DIR=third_party/NMR
mkdir -p third_party
if [ ! -d "$DIR/.git" ]; then
  git clone https://github.com/NJU3DV-HumanoidGroup/MakeTrackingEasy.git "$DIR"
fi
git -C "$DIR" fetch -q origin "$COMMIT" 2>/dev/null || true
git -C "$DIR" checkout -q "$COMMIT"
echo "NMR at $(git -C "$DIR" rev-parse --short HEAD)"

# SMPL-X is licensed: reuse the copy you already registered for GMR instead of NMR's mirror.
SMPLX=third_party/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz
if [ -f "$SMPLX" ] && [ ! -e "$DIR/assets/SMPLX_NEUTRAL.npz" ]; then
  ln -s "$(pwd)/$SMPLX" "$DIR/assets/SMPLX_NEUTRAL.npz"
  echo "linked your licensed SMPLX_NEUTRAL.npz into $DIR/assets/"
fi

if [ "${1:-}" = "--env" ]; then
  command -v conda >/dev/null || { echo "conda not found"; exit 1; }
  conda env list | grep -q '^nmr ' || conda create -y -n nmr python=3.10
  CUDA_TAG=${CUDA_TAG:-cu118}   # match your driver: cu118 / cu121 / cu124, or cpu
  if [ "$CUDA_TAG" = "pypi" ]; then conda run -n nmr pip install torch
  else conda run -n nmr pip install torch --index-url "https://download.pytorch.org/whl/$CUDA_TAG"; fi
  conda run -n nmr pip install -r "$DIR/requirements.txt"
  (cd "$DIR" && conda run -n nmr huggingface-cli download RayZhao/NMR weights/epoch_30.pth --local-dir .)
  [ -e "$DIR/assets/SMPLX_NEUTRAL.npz" ] || echo "NOTE: no SMPLX_NEUTRAL.npz yet; NMR would fetch its own copy. Prefer your licensed one: $SMPLX"
  echo "Smoke test: (cd $DIR && conda run -n nmr python inference.py --src examples/sample_motion.npz --output-dir /tmp/nmr_smoke)"
fi
