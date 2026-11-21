"""Run NMR (Make Tracking Easy, pretrained) on a GVHMR prediction. Executes INSIDE the NMR env.

    <nmr-python> video2ref/nmr_driver.py --nmr-repo third_party/NMR --gvhmr-pred hmr4d_results.pt \
        --out raw_nmr.npz [--no-filter] [--device cuda|cpu]

Standalone on purpose (the NMR env has torch/mmengine/smplx but not this project). Writes the network
output before any conversion, in NMR's own conventions:
  dof (T,29)            joint angles, NMR/Isaac-Lab breadth-first joint order (NOT SDK order)
  root_trans (T,3)      pelvis position, z-up
  root_rot_quat (T,4)   pelvis orientation, wxyz, z-up
  fps                   30
GVHMR's ``smpl_params_global`` (y-up, gravity aligned, 30 fps) is exactly NMR's "standard" input
(transl / global_orient / body_pose), so nothing is re-fitted.
"""
import argparse
import os
import subprocess
import sys
import tempfile

# Do not let this file's directory (video2ref/) shadow third-party modules (e.g. "video", "validate").
sys.path[:] = [x for x in sys.path if os.path.abspath(x or ".") != os.path.dirname(os.path.abspath(__file__))]

import numpy as np  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nmr-repo", required=True)
    p.add_argument("--gvhmr-pred", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--no-filter", action="store_true", help="skip NMR's 5 Hz Butterworth post-filter")
    p.add_argument("--device", choices=["cuda", "cpu"])
    a = p.parse_args()
    repo = os.path.abspath(a.nmr_repo)
    import torch

    try:  # the user's own GVHMR output: trusted local file (torch >= 2.6 defaults to weights_only=True)
        pred = torch.load(a.gvhmr_pred, map_location="cpu", weights_only=False)
    except TypeError:
        pred = torch.load(a.gvhmr_pred, map_location="cpu")
    g = pred["smpl_params_global"]
    arr = {k: g[k].detach().cpu().numpy().astype(np.float32) for k in ("transl", "global_orient", "body_pose")}
    if arr["body_pose"].shape[1] != 63:
        raise ValueError(f"expected 21-joint body_pose (T,63), got {arr['body_pose'].shape}")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "gvhmr_smplx.npz")
        np.savez(src, **arr, mocap_frame_rate=np.array(30.0))  # GVHMR demo output is 30 fps
        sys.path.insert(0, repo)
        os.chdir(repo)  # NMR resolves weights/ and assets/ relative to its directory
        import inference as nmr  # noqa: E402

        device = torch.device(a.device) if a.device else None
        model, smplx_model, betas, sm, ss, gm, gs, device = nmr.load_all(device=device)
        result, timing = nmr.infer_single(src, model, smplx_model, betas, sm, ss, gm, gs, device,
                                          apply_filter=not a.no_filter)
    if result is None:
        raise RuntimeError("sequence too short for NMR (< 4 frames)")
    try:
        commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    np.savez(a.out, dof=result["dof"], root_trans=result["root_trans"], root_rot_quat=result["root_rot_quat"],
             fps=np.array(30.0), filtered=np.array(not a.no_filter), nmr_commit=np.array(commit),
             infer_seconds=np.array(timing["total"]))
    print(f"NMR: {len(result['dof'])} frames in {timing['total']:.2f}s on {device} -> {a.out}")


if __name__ == "__main__":
    main()
