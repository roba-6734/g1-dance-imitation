# Whole-body tracking and NMR A/B: from reference to a gantry-ready policy

This covers two experiments that run alongside the existing upper-body `g1dance` path (arm_sdk with Unitree's locomotion keeping the legs):

1. **NMR vs GMR.** Same GVHMR prediction, two retargeters, compared kinematically and then on the robot through the existing `g1dance trial` ladder.
2. **Whole-body policy.** A BeyondMimic-style tracker trained with `unitree_rl_mjlab` on an RTX A5000. It is tested in an independent MuJoCo sim-to-sim harness that copies Unitree's real-robot C++ controller, and packaged only if every gate passes.

Nothing here commands the robot. Robot motion goes through `g1dance trial` (part 1) or Unitree's `g1_ctrl` (part 2).

## Tested so far, and what is still open

| Verified (CPU container, 2026-11-21) | Still open |
| --- | --- |
| The sim-to-sim port of `State_Mimic.cpp` runs **Unitree's own shipped dance policy** stably in every deterministic scenario and in 10/10 randomised runs (first 15 s). Body error 2.8 cm, joint RMS 0.098 rad. | Training on the A5000. Nothing has been trained in this repo yet. |
| Negative control: a wrong default pose or action scale keeps the shipped policy upright, but tracking error rises 2–3×. The joint RMS gate and the metadata check both catch it. | NMR on **your Ayyala video**. So far it has only run on NMR's own AMASS examples, packaged as GVHMR output. |
| **NMR end to end (CPU, real checkpoint):** `nmr_driver.py` output is identical to NMR's own `inference.py`. MuJoCo FK of our SDK-order conversion matches NMR's body positions to **0.0 mm**; skipping the remap gives 393 mm. CPU inference takes ~0.16 s for 86 frames. | |
| NMR output is **not joint-limit bounded**: on its run-turn example, left ankle roll is 0.083 rad past its stop on 7% of frames. `video2ref.nmr` clamps to the hard range and records the clamp in `validation.json`. GMR's IK already respects limits. | |
| Model gains equal Unitree's `deploy.yaml` (kp 40.2/99.1/28.5/14.3/16.8). The motion `.npz` FK matches to 7e-7 m. `wbt.npz` matches mjlab's `csv_to_npz` to p99 0.1 mm. | Compiling the C++ safety patch. It applies cleanly to `1425b15f` but needs the Unitree SDK toolchain. |
| Tools work with mujoco 4.0.1 and 3.5.0 (the mjlab pin). | Any hardware run. |

## 0. Setup

```bash
./scripts/setup_wbt.sh --env      # unitree_rl_mjlab @1425b15f (mjlab 1.2.0) + conda env, on the A5000 box
./scripts/setup_nmr.sh --env      # NMR @ba87ad65 + conda env "nmr" + checkpoint (518 MB, Hugging Face)
```

Our evaluation tools are CPU-only: `numpy`, `scipy`, `mujoco>=3.5`, `onnxruntime`, `pyyaml`. They run in the `unitree_rl_mjlab` env or in `.venv`.

## 1. NMR vs GMR

NMR is *Make Tracking Easy* (arXiv 2603.22201): a 70M-parameter transformer between two VQ-VAEs, SMPL-X to G1 29-DoF, with a 5 Hz zero-phase low-pass on its output. Pretrained weights are public. **Its repository has no LICENSE file**, so treat it as research evaluation only.

```bash
# existing GMR run (has gvhmr/input/hmr4d_results.pt)
python -m video2ref --video ayyala.mp4 --output-dir outputs/ayyala --gvhmr-python ... --camera moving
# same GVHMR prediction through NMR -> identical output layout (qpos.npy, g1dance_assets/, ...)
python -m video2ref.nmr --from-run outputs/ayyala --output-dir outputs/ayyala_nmr \
    --nmr-python "$(conda run -n nmr which python)"
# kinematic A/B: jerk, acceleration, range of motion, agreement, whole-body feasibility
python -m video2ref.compare refs outputs/ayyala outputs/ayyala_nmr --names GMR NMR --out runs/ab_ref
```

Then on the robot, use the existing upper-body ladder with identical flags for both asset folders, and in a randomised order:

```bash
python -m g1dance preview --assets outputs/ayyala_nmr/g1dance_assets --tag nmr          # sim first
python -m g1dance trial --stage entry --hold-waist --assets outputs/ayyala_nmr/g1dance_assets ...   # then the usual stages
python -m video2ref.compare runs runs/<gmr_run> runs/<nmr_run> --names GMR NMR --out runs/ab_robot
```

`compare runs` reads `log.npz` and `summary.json`. It reports measured joint jerk and acceleration during playback, IMU roll/pitch-rate RMS (body shake), tracking RMS and lag. Lower jerk at the **same range of motion** means smoother. Lower jerk with lower range of motion just means less dancing.

Notes:
- `--upper-smooth-window` defaults to 3 for both, so only the retargeter differs.
- NMR ignores the GVHMR body shape and uses its own fixed betas. It outputs the robot directly, so this does not matter for the robot.
- NMR's DoF order is the Isaac-Lab breadth-first order. `video2ref/nmr.py` remaps it to SDK order, and `tests/test_nmr.py` pins the remap against NMR's own `JOINT_MAPPING`.

## 2. Whole-body policy on the A5000

```bash
# inside the unitree_rl_mjlab env, from this repo's root
./scripts/train_wbt.sh ayyala_gmr outputs/ayyala/qpos.npy --lowpass-hz 6   # GMR: add filtering
./scripts/train_wbt.sh ayyala_nmr outputs/ayyala_nmr/qpos.npy              # NMR: already 5 Hz filtered
```

The script runs four steps:

1. **`wbt.prepare`.** Optional trim and low-pass, ground alignment, then a **lead-in** from the Velocity-policy stance (the pose the robot is in when `g1_ctrl` switches to mimic) and a **lead-out** back to it. Our clip starts mid-motion, so without the lead-in the policy would have to jump on entry. Writes the CSV and a `.json` sidecar.
2. **`wbt.feasibility`.** Joint limits and velocity, foot penetration/float/skate, a CoM-over-support proxy, and leg-leg collisions. It stops on FAIL, because a motion the robot cannot do won't train into a safe policy. The thresholds are calibrated so that Unitree's shipped dance (which runs on hardware) is WARN, not FAIL.
3. **mjlab `csv_to_npz.py`.**
4. **Training** with `Unitree-G1-Tracking-No-State-Estimation`. This is the only observation set `State_Mimic` implements. Defaults are `NUM_ENVS=4096` (use 2048 if the 24 GB A5000 runs out of memory), `MAX_ITERS=20000` and `SAVE_EVERY=500`, logged to TensorBoard. Time the first 100 iterations and extrapolate; I have no A5000 timing to quote.

mjlab re-exports `policy.onnx` and `<run>.onnx` (the one with metadata) at every save, so you can evaluate mid-training checkpoints:

```bash
python -m wbt.deploy_pkg --run-dir <logs/rsl_rl/g1_tracking/DATE_ayyala_gmr> \
    --motion-npz third_party/unitree_rl_mjlab/src/assets/motions/g1/ayyala_gmr.npz --name ayyala_gmr --no-install
```

`deploy_pkg` does four things:
1. Writes `deploy.yaml` from the training metadata: gains, default pose, action scale and observation order. Copying these by hand is the most common way a policy silently breaks.
2. Adds a per-joint clip of targets to the hard joint range.
3. Runs `wbt.sim2sim` on exactly the staged files, recording their hashes.
4. Installs into `deploy/robots/g1/config/policy/mimic/<name>/` **only on PASS**. It writes, but does not apply, the `config.yaml` FSM snippet.

### What `wbt.sim2sim` checks

The controller in `deploy_policy.py` copies `State_Mimic.cpp` term by term: observation order and scaling, frame timing, the `init_quat` yaw alignment, the torso anchor built from IMU plus waist joints, action scale/offset/clip, and the gains. It runs in plain MuJoCo, not the mujoco_warp training simulator.

| Scenario | Why |
| --- | --- |
| `nominal` | Start on the reference. Must be near perfect: joint RMS ≤ 0.15 rad, body error ≤ 8 cm. |
| `stand_entry` | The real entry path: robot at rest in the Velocity stance, then switched to mimic. |
| `latency_20ms`, `stand_entry_latency` | Command latency. |
| `motor_pd_1khz` | PD computed at 1 kHz with torque clipping, emulating the motor boards. |
| `imu_yaw_offset` | Real IMU yaw is arbitrary. `init_quat` must cancel it. |
| `random_*` (50 in full) | Friction 0.3–1.2, kp/kd ±20 %, motor strength 85–100 %, torso mass −1..+2 kg and COM ±5 cm, pushes up to 60 % of training's, sensor noise, latency 0–20 ms, random yaw, random entry type. |

**Gates:**
- No fall in any deterministic run.
- ≥ 95 % success over the randomised runs.
- Peak joint speed ≤ 90 % of the motor limit.
- Torque saturation ≤ 5 % of steps per joint.
- No contact with hard joint stops.
- No self-collision over 10 N.
- `deploy.yaml` matches the training metadata.
- Motion `.npz` FK is consistent with its joint angles (this catches joint-order bugs).

A PASS means **ready for a gantry test**, not for free standing.

## 3. Deploy-side safety findings (unitree_rl_mjlab @1425b15f)

Apply `wbt/patches/unitree_rl_mjlab_mimic_safety.patch` before building `g1_ctrl`:

1. **`bad_orientation()` always returns `false`.** It lives in `deploy/include/isaaclab/envs/mdp/terminations.h`, and the computation is commented out. As shipped, `State_Mimic`'s "tilt > 1 rad → Passive" check can never fire, so **there is no automatic fall cut-out.** The patch restores it, with the cosine clamped.
2. **First-entry command race.** `State_Mimic::run()` publishes `processed_actions` continuously, but those are all zeros until the policy thread's first step. That means q = 0 on every joint at mimic gains for up to one tick (stale targets on re-entry). The patch holds the previous state's command until this entry's first policy step.

Both changes are small. I could not compile them here, so build them and test in `unitree_mujoco` first.

## 4. Hardware trial ladder (whole-body policy)

Before you start, all of these must be in place:
- A `deploy_pkg` PASS (full suite).
- The patch applied and `g1_ctrl` rebuilt.
- A gantry with a rope that catches the robot before its knees reach the floor.
- An operator holding the remote, with `LT + B` (→ Passive, damping) rehearsed.
- A 3 m clear radius.
- `python -m g1dance check` showing a clean stance.

| Step | Setup | Move on when |
| --- | --- | --- |
| 1 | `unitree_mujoco` + `g1_ctrl --network=lo` with the installed package. This runs the real C++ code and config. | Full clip, entry and exit, no errors. |
| 2 | Gantry, feet **off** the floor. `time_end` = lead-in end + 2 s. Checks FSM transitions, joint directions, and no twitch at entry. Leg motion is meaningless in the air. | Clean entry and exit ×2. |
| 3 | Gantry, feet on the floor, rope slack but catching. `time_end` = lead-in end + 2 s. | No stepping or slipping beyond the reference ×3. |
| 4 | As step 3, but `time_end` at 25 % → 50 % → 100 % of the dance. Edit `time_end` in `config.yaml`. A policy is trained at one speed, so do not time-warp it. | Each length clean ×3. |
| 5 | Free standing, spotters, remote in hand. | Full clip clean ×3. |

**Abort with `LT + B` on:** any catch by the rope, a foot slide or an unplanned step, audible buzzing or oscillation, a twitch at entry, or any deviation you can't explain.

After an abort, go back one step and re-run sim-to-sim with the condition you observed (for example latency or friction) before trying again.
