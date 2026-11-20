"""One-command video -> GVHMR -> GMR -> G1 reference pipeline.

Stages: (1) normalise the video to upright 30 fps H.264, (2) GVHMR monocular
SMPL-X recovery (CUDA, separate env), (3) GMR SMPL-X -> Unitree G1 29-DoF
retargeting (30 Hz full-body qpos), (4) 50 Hz upper-body reference for g1dance.
Run inside the GMR environment from the repository root: python -m video2ref ...
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

from .motion import check_qpos, save_formats, smooth_qpos
from .upper import export as export_upper
from .video import normalize
from .validate import validate

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_gvhmr(video, output, repo, python, static, visualize):
    required = [repo / "tools/demo/demo.py", repo / "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
                repo / "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
                repo / "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
                repo / "inputs/checkpoints/yolo/yolov8x.pt"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("GVHMR files missing: " + ", ".join(missing) + ". Follow upstream docs/INSTALL.md")
    command = [str(python), "tools/demo/demo.py", "--video", str(video), "--output_root", str(output)]
    if static:
        command.append("-s")
    # Upstream demo always renders reconstruction; keep it as supplied.
    try:
        subprocess.run(command, cwd=repo, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("GVHMR inference failed. Check tracking visibility, checkpoints, CUDA memory, and its log above") from exc


def write_g1dance_assets(output: Path, upper_path: Path, video: Path) -> Path:
    """Copy the three outputs under the file names g1dance.reference expects."""
    assets = output / "g1dance_assets"
    assets.mkdir(exist_ok=True)
    shutil.copyfile(upper_path, assets / "gmr_reference_full_960_2804_upper_50hz.npz")
    shutil.copyfile(output / "qpos.npy", assets / "retargeted_g1_qpos.npy")
    shutil.copyfile(video, assets / "human_motion.mov")  # 30 fps MP4 data; ffmpeg reads it by content
    return assets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--robot", choices=["unitree_g1"], default="unitree_g1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gvhmr-repo", type=Path, default=ROOT / "third_party/GVHMR")
    parser.add_argument("--gmr-repo", type=Path, default=ROOT / "third_party/GMR")
    parser.add_argument("--gvhmr-python", default="python", help="Python executable in the GVHMR CUDA environment")
    parser.add_argument("--camera", choices=["static", "moving"], default="moving")
    parser.add_argument("--target-fps", type=int, default=30)
    parser.add_argument("--max-dimension", type=int, default=1280,
                        help="Maximum displayed video side after auto-rotation; default 1280")
    parser.add_argument("--clip-seconds", type=float,
                        help="Process only this many seconds from the beginning (useful for a GPU smoke test)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--smooth-window", type=int, default=1,
                        help="Odd moving-average window on the full-body qpos; 1 disables (our reference used 1)")
    parser.add_argument("--upper-smooth-window", type=int, default=3,
                        help="Odd moving-average window (30 Hz frames) before the 50 Hz upper-body export")
    parser.add_argument("--no-upper", action="store_true", help="Skip the 50 Hz upper-body export")
    parser.add_argument("--visualize", action="store_true", help="Render offscreen G1 and comparison MP4")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--skip-gvhmr", action="store_true")
    parser.add_argument("--skip-gmr", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute all stages")
    args = parser.parse_args(argv)
    if args.target_fps != 30:
        parser.error("GVHMR's demo writes and renders at 30 fps; --target-fps must be 30")
    if args.smooth_window < 1 or args.smooth_window % 2 != 1:
        parser.error("--smooth-window must be a positive odd integer")
    if args.device == "cpu" and not args.skip_gvhmr:
        parser.error("GVHMR demo requires CUDA; --device cpu is valid with --skip-gvhmr")
    if args.force and (args.skip_gvhmr or args.skip_gmr):
        parser.error("--force cannot be combined with a skip option")
    video = args.video.resolve()
    if not video.is_file() or video.suffix.lower() not in (".mp4", ".mov", ".m4v"):
        parser.error("--video must be an existing .mp4, .mov, or .m4v file")
    if args.max_dimension < 256 or args.max_dimension % 2:
        parser.error("--max-dimension must be even and at least 256")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    gvh = args.gvhmr_repo.resolve()
    gmr = args.gmr_repo.resolve()
    if not (gvh / "tools/demo/demo.py").is_file() or not (gmr / "scripts/gvhmr_to_robot.py").is_file():
        parser.error("Supply checked-out official repositories with --gvhmr-repo and --gmr-repo")
    # The source manifest prevents silently reusing another video's stage artifacts.
    manifest_path = output / "source.json"
    source = {"video_sha256": digest(video), "camera": args.camera, "target_fps": 30,
              "max_dimension": args.max_dimension, "clip_seconds": args.clip_seconds}
    if manifest_path.exists() and not args.force:
        old = json.loads(manifest_path.read_text())
        if old != source:
            parser.error("Output directory belongs to a different video or camera setting; choose a new directory or --force")
    work_video = output / "preprocess" / "input.mp4"
    pred = output / "gvhmr" / "input" / "hmr4d_results.pt"
    raw_path = output / "raw" / "raw_qpos.npy"
    raw_source = output / "raw" / "source.json"
    report_path = output / "validation.json"
    try:
        if args.force:
            for target in (output / "gvhmr" / "input", output / "raw"):
                if target.exists():
                    if not target.resolve().is_relative_to(output):
                        raise ValueError(f"Refusing to remove path outside output directory: {target}")
                    shutil.rmtree(target)
        metadata = normalize(video, work_video, overwrite=args.force,
                             max_dimension=args.max_dimension, clip_seconds=args.clip_seconds)
        manifest_path.write_text(json.dumps(source, indent=2))
        if args.force or not pred.exists():
            if args.skip_gvhmr:
                raise FileNotFoundError(f"--skip-gvhmr requires {pred}")
            run_gvhmr(work_video, output / "gvhmr", gvh, args.gvhmr_python, args.camera == "static", args.visualize)
        if not pred.is_file():
            raise FileNotFoundError(f"GVHMR did not create {pred}")
        sys.path.insert(0, str(gmr))
        from general_motion_retargeting.params import ROBOT_XML_DICT
        from .retarget import model_joints, retarget_gvhmr
        import mujoco as mj

        model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
        names, limits = model_joints(model)
        prediction_sha = digest(pred)
        if raw_path.exists() and not args.force:
            if not raw_source.exists() or json.loads(raw_source.read_text()).get("prediction_sha256") != prediction_sha:
                parser.error("Cached raw motion does not match GVHMR prediction; use --force or a new output directory")
        if args.force or not raw_path.exists():
            if args.skip_gmr:
                raise FileNotFoundError(f"--skip-gmr requires {raw_path}")
            raw, returned_names, _, _, _ = retarget_gvhmr(pred, gmr, raw_path.parent)
            if names != returned_names:
                raise ValueError("GMR and MuJoCo joint ordering differ")
            raw_source.write_text(json.dumps({"prediction_sha256": prediction_sha}, indent=2))
        else:
            raw = np.load(raw_path)
        raw = check_qpos(raw)
        processed = smooth_qpos(raw, args.smooth_window)
        save_formats(processed, 30, names, output, args.csv)
        report = validate(processed, 30, names, limits, model)
        report["raw_validation"] = validate(raw, 30, names, limits, model)
        report.update({"input_video": str(video), "source_video": metadata["original"],
                       "normalized_video": metadata["normalized"], "gvhmr_prediction": str(pred),
                       "raw_qpos": str(raw_path), "root_rot_pickle_order": "xyzw",
                       "qpos_root_quaternion_order": "wxyz", "smoothing_window_frames": args.smooth_window,
                       "gmr_ik_config": str(gmr / "general_motion_retargeting/ik_configs/smplx_to_g1.json"),
                       "robot_model": str(ROBOT_XML_DICT["unitree_g1"])})
        report_path.write_text(json.dumps(report, indent=2))
        if args.visualize:
            from .visualize import render_robot
            robot_video = output / "g1.mp4"
            render_robot(processed, 30, model, robot_video)
            human_video = output / "gvhmr" / "input" / "1_incam.mp4"
            if human_video.exists() and shutil.which("ffmpeg"):
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(human_video),
                                "-i", str(robot_video), "-filter_complex", "[0:v]scale=-2:480[human];[human][1:v]hstack=inputs=2[v]",
                                "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                str(output / "comparison.mp4")], check=True)
        print(f"Completed {len(processed)} frames; qpos={output / 'qpos.npy'}; report={report_path}")
        if not args.no_upper:
            upper_path = output / "reference_upper_50hz.npz"
            summary = export_upper(output / "qpos.npy", upper_path, args.upper_smooth_window, overwrite=True)
            print(f"Upper-body reference: {summary['frames']} frames at 50 Hz -> {upper_path}")
            assets = write_g1dance_assets(output, upper_path, work_video)
            print(f"g1dance asset folder: {assets}  (use: python -m g1dance preview --assets {assets})")
            if report["warnings"]:
                print("Full-body validation warnings (read validation.json before any robot use):")
                for w in report["warnings"]:
                    print("  -", w)
    except Exception as exc:
        (output / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
