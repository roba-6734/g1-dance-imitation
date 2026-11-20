"""Unitree G1 (29-DoF) joint table, upper-body command set, and limits.

Sources (checked 2026-11-20, see docs/SOURCES.md):
  * Joint indices 0..28 and weight slot 29: ``G1JointIndex`` in
    unitree_sdk2_python/example/g1/high_level/g1_arm7_sdk_dds_example.py
    (same values as ``JointIndex`` in unitree_sdk2 g1_arm7_sdk_dds_example.cpp).
  * Position / velocity / effort limits: unitree_ros
    robots/g1_description/g1_29dof_rev_1_0.urdf (mode_machine 5).
  * Waist-lock variants: unitree_ros robots/g1_description/README.md table.

Everything in this package commands ONLY ``UPPER_IDS`` (12..28). Leg ids 0..11
are listed here so that code can assert they are never written.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",  # 0
    "left_hip_roll_joint",  # 1
    "left_hip_yaw_joint",  # 2
    "left_knee_joint",  # 3
    "left_ankle_pitch_joint",  # 4
    "left_ankle_roll_joint",  # 5
    "right_hip_pitch_joint",  # 6
    "right_hip_roll_joint",  # 7
    "right_hip_yaw_joint",  # 8
    "right_knee_joint",  # 9
    "right_ankle_pitch_joint",  # 10
    "right_ankle_roll_joint",  # 11
    "waist_yaw_joint",  # 12
    "waist_roll_joint",  # 13  INVALID on waist-locked variants
    "waist_pitch_joint",  # 14 INVALID on waist-locked variants
    "left_shoulder_pitch_joint",  # 15
    "left_shoulder_roll_joint",  # 16
    "left_shoulder_yaw_joint",  # 17
    "left_elbow_joint",  # 18
    "left_wrist_roll_joint",  # 19
    "left_wrist_pitch_joint",  # 20 INVALID on 23-DoF
    "left_wrist_yaw_joint",  # 21   INVALID on 23-DoF
    "right_shoulder_pitch_joint",  # 22
    "right_shoulder_roll_joint",  # 23
    "right_shoulder_yaw_joint",  # 24
    "right_elbow_joint",  # 25
    "right_wrist_roll_joint",  # 26
    "right_wrist_pitch_joint",  # 27
    "right_wrist_yaw_joint",  # 28
)
NUM_MOTORS = 29
WEIGHT_INDEX = 29  # kNotUsedJoint: motor_cmd[29].q is the arm_sdk blend weight (0..1)
LEG_IDS: tuple[int, ...] = tuple(range(0, 12))
UPPER_IDS: tuple[int, ...] = tuple(range(12, 29))
UPPER_NAMES: tuple[str, ...] = tuple(G1_JOINT_NAMES[i] for i in UPPER_IDS)
N_UPPER = len(UPPER_IDS)

ARM_STATE_TOPIC = "rt/lowstate"
ARM_SDK_TOPIC = "rt/arm_sdk"

# Column indices inside the 17-wide upper-body vectors (order = motor id 12..28).
WAIST_YAW, WAIST_ROLL, WAIST_PITCH = 0, 1, 2
ARM_COLS: tuple[int, ...] = tuple(range(3, 17))
GROUPS: dict[str, tuple[int, ...]] = {
    "waist_yaw": (WAIST_YAW,),
    "waist_roll": (WAIST_ROLL,),
    "waist_pitch": (WAIST_PITCH,),
    "arms": ARM_COLS,
}

# (lower, upper, velocity rad/s, effort Nm) from g1_29dof_rev_1_0.urdf
_URDF_LIMITS: dict[str, tuple[float, float, float, float]] = {
    "waist_yaw_joint": (-2.618, 2.618, 32.0, 88.0),
    "waist_roll_joint": (-0.52, 0.52, 30.0, 35.0),
    "waist_pitch_joint": (-0.52, 0.52, 30.0, 35.0),
    "left_shoulder_pitch_joint": (-3.0892, 2.6704, 37.0, 25.0),
    "left_shoulder_roll_joint": (-1.5882, 2.2515, 37.0, 25.0),
    "left_shoulder_yaw_joint": (-2.618, 2.618, 37.0, 25.0),
    "left_elbow_joint": (-1.0472, 2.0944, 37.0, 25.0),
    "left_wrist_roll_joint": (-1.972222054, 1.972222054, 37.0, 25.0),
    "left_wrist_pitch_joint": (-1.614429558, 1.614429558, 22.0, 5.0),
    "left_wrist_yaw_joint": (-1.614429558, 1.614429558, 22.0, 5.0),
    "right_shoulder_pitch_joint": (-3.0892, 2.6704, 37.0, 25.0),
    "right_shoulder_roll_joint": (-2.2515, 1.5882, 37.0, 25.0),
    "right_shoulder_yaw_joint": (-2.618, 2.618, 37.0, 25.0),
    "right_elbow_joint": (-1.0472, 2.0944, 37.0, 25.0),
    "right_wrist_roll_joint": (-1.972222054, 1.972222054, 37.0, 25.0),
    "right_wrist_pitch_joint": (-1.614429558, 1.614429558, 22.0, 5.0),
    "right_wrist_yaw_joint": (-1.614429558, 1.614429558, 22.0, 5.0),
}
POS_LOWER = np.array([_URDF_LIMITS[n][0] for n in UPPER_NAMES])
POS_UPPER = np.array([_URDF_LIMITS[n][1] for n in UPPER_NAMES])
URDF_VEL_LIMIT = np.array([_URDF_LIMITS[n][2] for n in UPPER_NAMES])
URDF_EFFORT_LIMIT = np.array([_URDF_LIMITS[n][3] for n in UPPER_NAMES])

# mode_machine -> variant, from unitree_ros g1_description/README.md
_WAIST_LOCKED = {3: "g1_29dof_lock_waist (deprecated)", 6: "g1_29dof_lock_waist_rev_1_0",
                 12: "g1_29dof_mode_12", 14: "g1_29dof_mode_14", 16: "g1_29dof_mode_16"}
_WAIST_FREE = {2: "g1_29dof (deprecated)", 5: "g1_29dof_rev_1_0", 11: "g1_29dof_mode_11",
               13: "g1_29dof_mode_13", 15: "g1_29dof_mode_15", 18: "g1_29dof_mode_18"}
_DOF23 = {1: "g1_23dof (deprecated)", 4: "g1_23dof_rev_1_0", 10: "g1_23dof_mode_10"}


@dataclass(frozen=True)
class Variant:
    mode_machine: int | None
    name: str
    supported: bool  # 29 DoF with 7-DoF arms
    waist_roll_pitch_actuated: bool | None  # None = unknown

    def describe(self) -> str:
        w = {True: "waist roll/pitch actuated", False: "waist roll/pitch LOCKED",
             None: "waist lock UNKNOWN"}[self.waist_roll_pitch_actuated]
        return f"mode_machine={self.mode_machine} ({self.name}); supported={self.supported}; {w}"


def classify_variant(mode_machine: int | None) -> Variant:
    if mode_machine in _WAIST_FREE:
        return Variant(mode_machine, _WAIST_FREE[mode_machine], True, True)
    if mode_machine in _WAIST_LOCKED:
        return Variant(mode_machine, _WAIST_LOCKED[mode_machine], True, False)
    if mode_machine in _DOF23:
        return Variant(mode_machine, _DOF23[mode_machine], False, False)
    return Variant(mode_machine, "unknown", False, None)


def upper_index_of(name: str) -> int:
    """Column of ``name`` in a 17-wide upper-body vector."""
    return UPPER_NAMES.index(name)


def assert_upper_only(motor_ids) -> None:
    ids = [int(i) for i in motor_ids]
    bad = [i for i in ids if i not in UPPER_IDS]
    if bad:
        raise ValueError(f"refusing to address non-upper-body motor ids {bad}; legs are Unitree's")


def deg(x):
    return np.degrees(x)
