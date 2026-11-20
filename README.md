# g1dance: upper-body dance playback for the Unitree G1

This project plays the supplied retargeted dance on a Unitree G1 29-DoF through **arm_sdk** (`rt/arm_sdk`). It commands only motors 12..28 (waist yaw/roll/pitch and both 7-DoF arms). Unitree's onboard locomotion controller keeps the legs the whole time. Leg and root data from the full-body retarget never enter the command path.

The sequence for every run:

1. **acquire**: read the measured pose and ramp the arm_sdk weight 0 → 1 while commanding exactly that pose.
2. **entry**: min-jerk move to frame 0 of the 50 Hz upper-body reference, then settle. The **gate** then requires the *commanded* pose to equal the frame-0 target and the *measured* pose to be within tolerance. Playback does not start until it passes.
3. **playback**: the full 1,137-frame, 50 Hz reference from frame 0. Speed is a time-warp, independent of amplitude.
4. **return + release**: min-jerk back to the measured start pose, then weight 1 → 0 (handback).

Amplitude per group (`arms`, `waist_yaw`, `waist_roll`, `waist_pitch`) follows `target = measured_start + scale × (recorded − measured_start)`. Scale 0 holds the measured start; scale 1 requests the recorded value. `--hold-waist` fixes all three waist joints, which isolates the arm entry.

Results, numbers and caveats (`docs/FINDINGS.md`), the hardware procedure (`docs/HARDWARE.md`) and sources (`docs/SOURCES.md`) are kept locally and are not part of this repository.

## Setup

Requires Python 3.11+, ffmpeg, and an EGL-capable GPU/driver for offscreen rendering (set `MUJOCO_GL=osmesa` otherwise).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
./scripts/fetch_model.sh
```

The motion assets are not tracked in git. Put the three assets in `motion_assets/`, or pass `--assets DIR` to any command.

For hardware only, install Unitree's Python SDK. The script builds CycloneDDS 0.10.x from source (`cyclonedds==0.10.2` has no Python 3.12 wheel), then installs the SDK into `.venv`. Both go in `third_party/`.

```bash
./scripts/install_unitree_sdk.sh
```

## From your own video: GVHMR → GMR → reference motion

`video2ref/` is the pipeline that produced the dance used here. Give it a monocular video of one person and it writes a G1 reference that `g1dance` can simulate and play:

```
video (.mp4/.mov) ──ffmpeg──▶ upright 30 fps H.264
                  ──GVHMR───▶ SMPL-X body motion (hmr4d_results.pt)      [CUDA]
                  ──GMR─────▶ Unitree G1 29-DoF qpos, 30 Hz (qpos.npy)
                  ──upper───▶ motors 12..28, 3-frame mean, 50 Hz (reference_upper_50hz.npz)
                  ──────────▶ g1dance_assets/  →  python -m g1dance preview --assets …
```

| Stage | What happens | Output |
| --- | --- | --- |
| 1. Video | ffprobe/ffmpeg: auto-rotate, constant 30 fps, long side ≤ `--max-dimension` (default 1280), optional `--clip-seconds` | `preprocess/input.mp4` |
| 2. GVHMR | Official `tools/demo/demo.py` (YOLO + ViTPose + GVHMR). `--camera static` adds `-s`; `moving` uses SimpleVO | `gvhmr/input/hmr4d_results.pt`, human renders |
| 3. GMR | `load_gvhmr_pred_file` → `get_gvhmr_data_offline_fast` (rotates GVHMR's y-up world to z-up with `[[1,0,0],[0,0,-1],[0,1,0]]` and slerp-resamples to 30 Hz) → `GeneralMotionRetargeting(src_human="smplx", tgt_robot="unitree_g1")`, every frame kept, including frame 0 | `raw/raw_qpos.npy`, `qpos.npy` `(T,36)`: root xyz, quaternion **wxyz**, 29 joints in rad; `motion.pkl` (root quaternion **xyzw**); `validation.json` |
| 4. Upper body | Motors 12..28 only (root and legs dropped), centred 3-frame mean, linear 30 → 50 Hz, `joint_vel = gradient`, refuses soft-limit or release-screen breaches | `reference_upper_50hz.npz` + `.json`, `g1dance_assets/` |

Upstream code is used at the commits this project was built with: GVHMR `ee960bb6`, GMR `25f56db7`. Neither is vendored.

### Install

1. Clone both repositories into `third_party/`, which git ignores. With `--envs`, the script also creates the `gvhmr` (CUDA) and `gmr` conda environments.

   ```bash
   ./scripts/setup_video2ref.sh --envs
   ```

2. Download the licensed files yourself. Nothing licensed is redistributed here.
   - SMPL/SMPL-X models and the GVHMR, HMR2, ViTPose and YOLO checkpoints go under `third_party/GVHMR/inputs/checkpoints/`. See GVHMR's `docs/INSTALL.md`.
   - `SMPLX_NEUTRAL.npz` goes under `third_party/GMR/assets/body_models/smplx/`.

GVHMR needs an NVIDIA GPU. GMR and stage 4 run on the CPU.

### Run

Run from the repository root, inside the `gmr` environment:

```bash
export MUJOCO_GL=egl   # headless rendering; omit on a desktop
python -m video2ref --video my_dance.mp4 --output-dir outputs/my_dance \
  --gvhmr-python "$(conda run -n gvhmr which python)" --camera moving --visualize
```

The reference in this repository came from a 22.8 s portrait clip run with `--max-dimension 960 --camera moving` and the default options, which means no full-body smoothing. Its upper-body file was exported with the default 3-frame window.

- Re-running the same command reuses the stages that already finished.
- `--skip-gvhmr` and `--skip-gmr` reuse cached outputs.
- `--force` recomputes everything.
- `--clip-seconds 3` gives a quick smoke test.

On a Slurm cluster, copy `configs/hpc.env.example` to `configs/hpc.env` and submit:

```bash
bash scripts/submit_hpc.sh /abs/path/my_dance.mp4 my_dance --max-dimension 960 --visualize
```

Stage 4 on its own is numpy-only and runs in the g1dance venv. Use it to re-export an existing `qpos.npy` with a different window:

```bash
.venv/bin/python -m video2ref.upper --qpos outputs/my_dance/qpos.npy --output outputs/my_dance/ref_w5.npz --smoothing-window 5
```

### Then

```bash
.venv/bin/python -m g1dance inspect --assets outputs/my_dance/g1dance_assets
```

```bash
.venv/bin/python -m g1dance preview --assets outputs/my_dance/g1dance_assets
```

Before using the output on a robot, check it:

- Review the GVHMR human render for identity swaps, missing limbs or a wrong ground plane.
- Review the G1 preview (`g1.mp4`, `comparison.mp4`) and the warnings in `validation.json`.

All of these are kinematic checks only. They do not show balance, contact or hardware safety. Hardware still goes through the staged, opt-in `g1dance trial` procedure below.

## Offline (no robot)

```bash
.venv/bin/python -m pytest -q
```

```bash
# asset report: shapes, durations, units, joint names/IDs, frame-0 mapping
.venv/bin/python -m g1dance inspect
```

```bash
# plan + MuJoCo simulation + comparison video (runs/<time>_default/comparison.mp4)
.venv/bin/python -m g1dance preview --tag default
```

```bash
# Unitree example gains (waist kp=60): reproduces the waist-pitch overshoot and stops at entry
.venv/bin/python -m g1dance preview --config configs/sdk_example_gains.toml --tag sdk_example_gains
```

```bash
# slow one passage without shrinking it; or scale amplitudes explicitly
.venv/bin/python -m g1dance preview --slow 12:16:0.5 --tag slow_12_16
```

```bash
.venv/bin/python -m g1dance preview --arms 1 --waist-yaw 1 --waist-roll 1 --waist-pitch 0 --tag pitch_held
```

Every option is also a config key (`configs/default.toml`) and can be overridden with `--set section.key=value`. The effective settings are printed and written to `settings.json` for every run.

## Hardware

```bash
# read-only: never publishes
.venv/bin/python -m g1dance check --iface enp3s0 --assets motion_assets --save-stance runs/stance.json
```

```bash
# staged trial, rehearsed on the simulator first (identical code path)
.venv/bin/python -m g1dance trial --stage entry --hold-waist
```

```bash
# live: needs --enable-motion AND typing the stage name at the prompt
.venv/bin/python -m g1dance trial --stage entry --hold-waist --backend unitree --iface enp3s0 --enable-motion
```

Stages: `acquire`, `entry`, `playback --ref-end S`, `full`. Read the operator checklist in `docs/HARDWARE.md` (local, not in the repository) before the first live stage.

## Layout

```
g1dance/joints.py     SDK joint table, upper-body ids, URDF limits, variant table
g1dance/reference.py  asset loaders + validation (full-body file: visualisation only)
g1dance/config.py     every run setting (scales, speed, gains, limits, gates)
g1dance/planner.py    acquire / entry / settle / playback / return / release targets
g1dance/monitor.py    stop checks (named joint, measurement, threshold) + entry gate
g1dance/runner.py     closed-loop executor shared by sim and hardware; abort + handback
g1dance/sim.py        MuJoCo fixed-pelvis backend (arm_sdk-style PD)
g1dance/metrics.py    opening-pose error, worst joint/time, limit events, lag/correlation
g1dance/render.py     4-column comparison video; plots.py joint traces
g1dance/hw/           unitree.py (SDK I/O), check.py (read-only), trial.py (staged)
video2ref/pipeline.py video -> GVHMR -> GMR -> qpos -> 50 Hz reference driver (python -m video2ref)
video2ref/video.py    ffprobe/ffmpeg normalisation; retarget.py GMR call; validate.py screens
video2ref/upper.py    30 Hz full body -> 50 Hz upper-body npz (python -m video2ref.upper)
```
