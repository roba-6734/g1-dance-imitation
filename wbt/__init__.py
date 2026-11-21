"""wbt: whole-body tracking (BeyondMimic / unitree_rl_mjlab) preparation and pre-hardware testing.

Pipeline (see wbt/README.md):
  qpos.npy (GMR or NMR, 30 Hz) --prepare--> Unitree CSV --csv_to_npz (mjlab)--> motion.npz
  --train (A5000)--> policy.onnx --sim2sim (this package)--> readiness report --> deploy package
"""
