# Resources for smoother Ayyala dance on the G1

Research notes from 2026-11-20. These are leads to evaluate. None of them has been integrated or tested in this repo yet.

## Where the roughness comes from now

```
video → GVHMR → GMR (per-frame IK) → 3-frame mean → linear 30→50 Hz → arm_sdk PD (12..28)
```

1. **GVHMR → GMR jitter.** GMR solves IK one frame at a time and does not smooth anything. BeyondRetarget (arXiv 2609.29850) measures GVHMR→GMR jitter at **7.94**, against **1.63** for a neural retargeter (NMR).
2. **Weak filtering.** `video2ref/upper.py` uses a centred 3-frame box filter followed by *linear* interpolation. The linear interpolation leaves velocity kinks at every 30 Hz sample. Those kinks show up as acceleration spikes in `joint_vel = gradient`, and that velocity is fed forward through `dq_feedforward`.
3. **Open-loop playback.** The waist lag (~163 ms at kp=400) is tracking delay of the reference. Higher gains will not remove it.

## Tier 1: cheap, fits the current arm_sdk architecture

| Change | Why | Resource |
| --- | --- | --- |
| Replace the 3-frame mean with **Savitzky–Golay** (e.g. window 9–15, order 3) or a zero-phase **Butterworth** (`filtfilt`, 4–6 Hz cut-off) | Keeps peak amplitude better than a box filter and adds no phase lag offline | `scipy.signal.savgol_filter`, `scipy.signal.filtfilt` |
| Resample 30→50 Hz with a **cubic / quintic spline** (or a smoothing B-spline), not `np.interp` | Gives continuous velocity and acceleration, so the dq feed-forward stays clean | `scipy.interpolate.make_smoothing_spline`, `CubicSpline` |
| Add a **time-shift (lead) compensation** equal to the measured lag per group | Removes the systematic 160–210 ms waist lag without raising gains | measured lag is already in `g1dance/metrics.py` |
| Jitter cleanup used on GVHMR→GMR in published work: downsample, then causal moving average, then median filter | Same input pipeline as this repo | arXiv [2604.21351](https://arxiv.org/pdf/2604.21351) |
| Alternative: smooth the **SMPL-X output before GMR** (rotation-space slerp filtering) | Lets GMR's IK see clean input, which avoids filtering joint angles after limits are applied | GMR wiki: [YanjieZe/GMR](https://github.com/YanjieZe/GMR/wiki) |

## Tier 2: better human-motion front end

| Option | Notes |
| --- | --- |
| **NMR (neural motion retargeting)**, arXiv [2603.22201](https://arxiv.org/pdf/2603.22201) | Learned retargeter that smooths implicitly; jitter 1.63 vs GMR 7.94 in the BeyondRetarget comparison. Check for code. |
| **BeyondRetarget**, arXiv [2609.29850](https://arxiv.org/pdf/2609.29850) | End-to-end video → executable G1 motion, so there is no separate HMR+retarget step. No public code found yet. |
| **DuoMo**, arXiv [2603.03265](https://arxiv.org/html/2603.03265) | Diffusion world-space HMR; reports a 30 % world-error reduction on RICH and low foot skating. A possible GVHMR replacement. |
| **SAM 3D Body world-coordinate retargeting**, arXiv [2512.21573](https://arxiv.org/abs/2512.21573v1) | Another monocular → humanoid route. |
| **VideoMimic**, arXiv [2505.03729](https://arxiv.org/html/2505.03729v5) | Real-to-sim-to-real from video on G1 (whole body, contextual). |

## Tier 3: whole-body RL tracker (largest gain in smoothness and in the legs/waist)

The current design leaves the legs to Unitree's locomotion controller. That controller fights large waist motion, which is why the code has step detection, stance gates and waist limits. A tracking policy trained on the dance clip itself would balance *through* the motion, and its actions are smoothed by training regularisers (action-rate penalties).

| Framework | Fit for us |
| --- | --- |
| **unitree_rl_mjlab** ([repo](https://github.com/unitreerobotics/unitree_rl_mjlab)) | **Most practical.** Official Unitree. Task `Unitree-G1-Tracking-No-State-Estimation`; a dance example ships (`dance1_subject2.csv`). Takes CSV → `csv_to_npz.py --input-fps 30 --output-fps 50`, which matches our GMR `qpos.npy` (root xyz + quat + 29 joints) after a small converter. Exports ONNX. sim2sim via unitree_mujoco and sim2real via `g1_ctrl` over Ethernet. Needs one NVIDIA GPU and trains one policy per clip. |
| **BeyondMimic** ([paper](https://arxiv.org/abs/2508.08241), [repo, archived](https://github.com/HybridRobotics/whole_body_tracking)) | The method behind the above. The repo is retired and lives on in [mjlab](https://github.com/mujocolab/mjlab). Deploy code: [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller). |
| **NVIDIA GEAR-SONIC** ([repo](https://github.com/NVlabs/GR00T-WholeBodyControl), [weights](https://huggingface.co/nvidia/GEAR-SONIC), [paper](https://arxiv.org/html/2511.07820v2)) | A universal G1 tracker (training code released 2026-04, BONES-SEED 142K motions). Accepts "G1 motion reference" and SMPL inputs, so it **may** track our clip without per-clip training (not confirmed; test in sim first). C++/TensorRT deploy. Fine-tuning needs a large GPU budget. |
| **ProtoMotions G1 workflow** ([docs](https://nvlabs.github.io/ProtoMotions/tutorials/workflows/g1_deployment.html)) | NVIDIA's BeyondMimic-style pipeline from data to the real robot. |
| **HoloMotion** ([repo](https://github.com/HorizonRobotics/HoloMotion)), **GMT** ([page](https://gmt-humanoid.github.io)), **TWIST** ([repo](https://github.com/YanjieZe/TWIST)) | Alternative general trackers. TWIST already consumes GMR output natively. |

**Recommended path:** do Tier 1 now (one afternoon, no hardware risk). In parallel, train a unitree_rl_mjlab tracking policy on the existing `qpos.npy` and compare it in sim2sim against the current arm_sdk playback. Try SONIC zero-shot as a cheap side experiment.

## Better source video for Ayyala

Al-Ayyala is UNESCO-listed (2014): two facing rows of men with thin bamboo canes, swaying in unison to drums, with the cane raised or rotated. For pose estimation, the ideal clip is **one performer, full body in frame, static camera, front or 3/4 view, no occlusion by other dancers, ≥ 30 fps, 1080p**.

| Source | Notes |
| --- | --- |
| UNESCO ICH video "Al-Ayyala, a traditional performing art" ([ich.unesco.org/en/video/33501](https://ich.unesco.org/en/video/33501), embedded YouTube `V6roOGoW07k`; [archive item](https://www.unesco.org/archives/multimedia/document-3658)) | Authoritative footage from UNESCO and Abu Dhabi TCA. Group shots, so look for single-dancer segments. Rights are reserved. |
| Arabsstock licensed clips: [479768](https://arabsstock.com/en/videos/clip-479768-al-ayala-is-a-folk-dance-performed), [479770](https://arabsstock.com/en/videos/clip-479770-specific-signals-and-movements-for-folk-dance), [479813](https://arabsstock.com/en/videos/clip-479813-hand-movements-and-dances-folkloric-performance-troupe) | Paid stock with model releases, so it is the cleanest licensing for a demo. Clip 479770 ("specific signals and movements") may show isolated movements. |
| [Abu Dhabi Culture: Ayyala Dance](https://abudhabiculture.ae/en/Cultural-Resources/Publications/Ayyala-Dance), [Dubai Culture: Al Yowla](https://dubaiculture.gov.ae/en/about-us/Discover-Our-Heritage/Traditional-Performances/Al-Yowla), [Google Arts & Culture: Al Yowla](https://artsandculture.google.com/story/LQVBBtOzrgV0oQ) | Reference for the correct movement vocabulary. Contact these bodies for footage or a performer. |
| **Record your own** | The best option for quality. Ask one performer (a heritage centre or a university cultural club) to dance in front of a tripod-mounted phone at 60 fps, full body, plain background, with a cane. A static camera lets us run `--camera static`, which removes SimpleVO drift. |

YouTube search terms: "Al Ayyala solo", "Ayyala dance tutorial", "Yowla cane performance", "العيالة".

**Cane note:** GVHMR/SMPL-X does not track the cane. Hand pose drives the wrist only, so the cane's angle has to be added by hand (wrist yaw/pitch offset) or left out.
