#!/usr/bin/env bash
# Clone GVHMR and GMR at the commits this project used, into third_party/ (git-ignored),
# and create their two conda environments. Licensed SMPL/SMPL-X body models and the
# GVHMR checkpoints are NOT downloaded: see README "From your own video".
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
GVHMR_COMMIT=ee960bb6e2ea2d381aa97f08e9b71ef320b624b1
GMR_COMMIT=25f56db79ac411258b377b6d4ccba0c8ff3372ac
mkdir -p "$root/third_party"
clone() {  # url dir commit
  [[ -d "$2/.git" ]] || git clone "$1" "$2"
  git -C "$2" fetch --quiet origin
  git -C "$2" checkout --quiet "$3"
  echo "$(basename "$2") at $(git -C "$2" rev-parse --short HEAD)"
}
clone https://github.com/zju3dv/GVHMR.git "$root/third_party/GVHMR" "$GVHMR_COMMIT"
clone https://github.com/YanjieZe/GMR.git "$root/third_party/GMR" "$GMR_COMMIT"
if [[ "${1:-}" == "--envs" ]]; then
  command -v conda >/dev/null || { echo "conda not found" >&2; exit 1; }
  conda create -y -n gvhmr python=3.10
  conda run -n gvhmr python -m pip install -r "$root/third_party/GVHMR/requirements.txt"
  conda run -n gvhmr python -m pip install -e "$root/third_party/GVHMR"
  conda create -y -n gmr python=3.10
  conda run -n gmr python -m pip install -e "$root/third_party/GMR"
  conda run -n gmr python -m pip install imageio imageio-ffmpeg
fi
cat <<MSG
Next (manual, licensed downloads):
  * GVHMR checkpoints + SMPL/SMPL-X models -> third_party/GVHMR/inputs/checkpoints/ (GVHMR docs/INSTALL.md)
  * SMPLX_NEUTRAL.npz                       -> third_party/GMR/assets/body_models/smplx/
MSG
