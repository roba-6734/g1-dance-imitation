"""GVHMR -> NMR (pretrained neural retargeter) -> the same outputs as the GMR pipeline, for A/B tests.

    python -m video2ref.nmr --from-run outputs/my_dance --output-dir outputs/my_dance_nmr \
        --nmr-python "$(conda run -n nmr which python)" [--nmr-repo third_party/NMR] [--no-filter]

``--from-run`` is a finished GMR run directory: its GVHMR prediction (gvhmr/input/hmr4d_results.pt)
and normalised video are reused, so the two retargeters see the identical human motion.
The output directory has the GMR pipeline's layout (qpos.npy, motion.pkl, validation.json,
reference_upper_50hz.npz, g1dance_assets/), so ``g1dance preview/trial --assets`` and
``wbt.prepare`` work unchanged. Compare with ``python -m video2ref.compare``.

NMR: Make Tracking Easy, Zhao et al. 2026, arXiv 2603.22201; code
github.com/NJU3DV-HumanoidGroup/MakeTrackingEasy, weights huggingface.co/RayZhao/NMR.
The repository has no LICENSE file: research evaluation only until the authors state terms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from g1dance.joints import G1_JOINT_NAMES
from .motion import check_qpos, save_formats
from .upper import export as export_upper

ROOT = Path(__file__).resolve().parents[1]
NMR_COMMIT = "ba87ad6541f6512df9a370acdd2c2d95da7ae419"

# NMR's network DoF order: breadth-first over the G1 kinematic tree (Isaac Lab order), as listed by
# G1_BMIMIC_BODY_NAMES in NMR's convert_bmimic.py (body k+1 <-> joint k).
NMR_JOINT_ORDER = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)
# sdk[NMR_TO_SDK[i]] = nmr[i]; equals JOINT_MAPPING in NMR's convert_bmimic.py (asserted in tests).
NMR_TO_SDK = tuple(G1_JOINT_NAMES.index(n) for n in NMR_JOINT_ORDER)


def nmr_to_qpos(raw: dict) -> np.ndarray:
    """NMR network output -> (T,36) qpos: root xyz, quat wxyz, 29 joints in SDK/MJCF order."""
    dof = np.asarray(raw["dof"], float)
    if dof.ndim != 2 or dof.shape[1] != 29:
        raise ValueError(f"NMR dof must be (T,29), got {dof.shape}")
    sdk = np.zeros_like(dof)
    sdk[:, list(NMR_TO_SDK)] = dof
    quat = np.asarray(raw["root_rot_quat"], float)
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    return check_qpos(np.concatenate([np.asarray(raw["root_trans"], float), quat, sdk], axis=1))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_and_limits(xml=None):
    """G1 for validation/rendering: mjlab's 29-DoF model with actuators (wbt.g1model).
    NMR's own assets/g1_mocap_29dof.xml references meshes its repository does not ship."""
    from wbt.g1model import build_model, find_g1_xml

    from .retarget import model_joints

    model = build_model(xml)
    names, limits = model_joints(model)
    if tuple(names) != G1_JOINT_NAMES:
        raise ValueError("validation model joint order differs from the SDK table")
    return model, names, limits, find_g1_xml(xml)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-run", type=Path, help="finished GMR pipeline output dir (reuses its GVHMR result)")
    src.add_argument("--gvhmr-pred", type=Path, help="hmr4d_results.pt (then --video is needed for g1dance assets)")
    p.add_argument("--video", type=Path, help="30 fps video for g1dance_assets (default: the run's preprocess/input.mp4)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--nmr-repo", type=Path, default=ROOT / "third_party/NMR")
    p.add_argument("--nmr-python", default="python", help="Python of the NMR env (torch, mmengine, smplx)")
    p.add_argument("--device", choices=["cuda", "cpu"])
    p.add_argument("--no-filter", action="store_true", help="raw network output (NMR applies 5 Hz low-pass by default)")
    p.add_argument("--upper-smooth-window", type=int, default=3,
                   help="same default as the GMR pipeline, so the A/B differs only in the retargeter")
    p.add_argument("--xml", help="mjlab g1.xml for validation (default: installed mjlab or third_party/mjlab_g1)")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    pred = (a.from_run / "gvhmr/input/hmr4d_results.pt") if a.from_run else a.gvhmr_pred
    video = a.video or (a.from_run / "preprocess/input.mp4" if a.from_run else None)
    if not pred.is_file():
        p.error(f"GVHMR prediction not found: {pred}")
    if not (a.nmr_repo / "inference.py").is_file():
        p.error(f"NMR checkout not found at {a.nmr_repo}; run scripts/setup_nmr.sh")
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw" / "raw_nmr.npz"
    raw_path.parent.mkdir(exist_ok=True)
    source = {"gvhmr_prediction": str(pred.resolve()), "prediction_sha256": digest(pred), "filtered": not a.no_filter}
    manifest = out / "source.json"
    if raw_path.exists() and not a.force and manifest.exists() and json.loads(manifest.read_text()) == source:
        print(f"reusing {raw_path}")
    else:
        cmd = [a.nmr_python, str(Path(__file__).with_name("nmr_driver.py")), "--nmr-repo", str(a.nmr_repo.resolve()),
               "--gvhmr-pred", str(pred.resolve()), "--out", str(raw_path)]
        cmd += ["--no-filter"] if a.no_filter else []
        cmd += ["--device", a.device] if a.device else []
        subprocess.run(cmd, check=True)
        manifest.write_text(json.dumps(source, indent=2))
    with np.load(raw_path) as z:
        raw = {k: z[k] for k in z.files}
    qpos = nmr_to_qpos(raw)
    save_formats(qpos, 30, list(G1_JOINT_NAMES), out)
    model, names, limits, xml = model_and_limits(a.xml)
    from .validate import validate

    report = validate(qpos, 30, names, limits, model)
    report.update({"retargeter": "NMR", "nmr_commit": str(raw.get("nmr_commit", "")),
                   "nmr_post_filter_5hz": bool(raw.get("filtered", True)), "gvhmr_prediction": str(pred),
                   "robot_model": str(xml), "qpos_root_quaternion_order": "wxyz", "joint_order": "SDK/MJCF"})
    (out / "validation.json").write_text(json.dumps(report, indent=2))
    upper = out / "reference_upper_50hz.npz"
    summary = export_upper(out / "qpos.npy", upper, a.upper_smooth_window, overwrite=True)
    print(f"NMR qpos {qpos.shape} -> {out / 'qpos.npy'}; upper reference {summary['frames']} frames at 50 Hz")
    if video and Path(video).is_file():
        from .pipeline import write_g1dance_assets

        assets = write_g1dance_assets(out, upper, Path(video))
        print(f"g1dance assets: {assets}")
    else:
        print("no video given: g1dance_assets/ not written (pass --video)")
    if a.visualize:
        from .visualize import render_robot

        render_robot(qpos, 30, model, out / "g1.mp4")
    if report["warnings"]:
        print("validation warnings:", *report["warnings"], sep="\n  - ")


if __name__ == "__main__":
    main()
