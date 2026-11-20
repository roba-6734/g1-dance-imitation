# g1dance: upper-body dance playback for the Unitree G1

This project plays the supplied retargeted dance on a Unitree G1 29-DoF through **arm_sdk** (`rt/arm_sdk`). It commands only motors 12..28 (waist yaw/roll/pitch and both 7-DoF arms). Unitree's onboard locomotion controller keeps the legs the whole time. Leg and root data from the full-body retarget never enter the command path.

The sequence for every run:

1. **acquire**: read the measured pose and ramp the arm_sdk weight 0 → 1 while commanding exactly that pose.
2. **entry**: min-jerk move to frame 0 of the 50 Hz upper-body reference, then settle. The **gate** then requires the *commanded* pose to equal the frame-0 target and the *measured* pose to be within tolerance. Playback does not start until it passes.
3. **playback**: the full 1,137-frame, 50 Hz reference from frame 0. Speed is a time-warp, independent of amplitude.
4. **return + release**: min-jerk back to the measured start pose, then weight 1 → 0 (handback).

Amplitude per group (`arms`, `waist_yaw`, `waist_roll`, `waist_pitch`) follows `target = measured_start + scale × (recorded − measured_start)`. Scale 0 holds the measured start; scale 1 requests the recorded value. `--hold-waist` fixes all three waist joints, which isolates the arm entry.

Results, numbers and caveats: **[docs/FINDINGS.md](docs/FINDINGS.md)**. Hardware procedure: **[docs/HARDWARE.md](docs/HARDWARE.md)**. Sources: [docs/SOURCES.md](docs/SOURCES.md).

## Setup

Requires Python 3.11+, ffmpeg, and an EGL-capable GPU/driver for offscreen rendering (set `MUJOCO_GL=osmesa` otherwise).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
./scripts/fetch_model.sh
```

Put the three assets in `motion_assets/`, or pass `--assets DIR` to any command.

For hardware only, install Unitree's Python SDK by following its README. It needs CycloneDDS 0.10.x.

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python && .venv/bin/pip install -e unitree_sdk2_python
```

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

Stages: `acquire`, `entry`, `playback --ref-end S`, `full`. Read the operator checklist in [docs/HARDWARE.md](docs/HARDWARE.md) before the first live stage.

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
```
