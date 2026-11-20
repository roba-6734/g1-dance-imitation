import numpy as np

from .motion import check_qpos


# Project data-derived thresholds, not certified G1 actuator safety limits.
GROUP_LIMITS = {
    "legs": {"release": (2.4, 53.), "validation": (2.7, 61.)},
    "waist": {"release": (1.1, 30.), "validation": (1.2, 34.)},
    "arms": {"release": (5.1, 83.), "validation": (5.7, 96.)},
}


def per_joint_motion_limits(joint, fps, names):
    """Report absolute per-frame differences; never average joints together."""
    joint = np.asarray(joint, dtype=np.float64)
    if joint.ndim != 2 or joint.shape[1] != 29 or len(names) != 29 or fps <= 0:
        raise ValueError("Expected (frames,29) joint positions and positive FPS")
    speed = np.abs(np.diff(joint, axis=0)) * fps
    acceleration = np.abs(np.diff(joint, n=2, axis=0)) * fps * fps
    result = {}
    for index, name in enumerate(names):
        group = "legs" if index < 12 else "waist" if index < 15 else "arms"
        velocity = speed[:, index]
        accel = acceleration[:, index]
        entry = {"group": group,
                 "peak_abs_velocity_rad_s": float(velocity.max()) if velocity.size else 0.,
                 "peak_abs_acceleration_rad_s2": float(accel.max()) if accel.size else 0.,
                 "p99_7_abs_velocity_rad_s": float(np.percentile(velocity, 99.7)) if velocity.size else 0.,
                 "p99_7_abs_acceleration_rad_s2": float(np.percentile(accel, 99.7)) if accel.size else 0.,
                 "p99_8_abs_velocity_rad_s": float(np.percentile(velocity, 99.8)) if velocity.size else 0.,
                 "p99_8_abs_acceleration_rad_s2": float(np.percentile(accel, 99.8)) if accel.size else 0.}
        for stage, (velocity_limit, acceleration_limit) in GROUP_LIMITS[group].items():
            entry[stage] = {
                "velocity_limit_rad_s": velocity_limit,
                "acceleration_limit_rad_s2": acceleration_limit,
                "velocity_exceedance_frames": int(np.count_nonzero(velocity > velocity_limit)),
                "acceleration_exceedance_frames": int(np.count_nonzero(accel > acceleration_limit)),
            }
        result[name] = entry
    return result


def validate(qpos, fps, names, limits, model=None, max_velocity=3 * np.pi):
    qpos = check_qpos(qpos)
    if fps <= 0 or len(names) != 29 or np.asarray(limits).shape != (29, 2):
        raise ValueError("Invalid FPS, G1 joint mapping, or joint limits")
    warnings = []
    joint = qpos[:, 7:]
    low, high = np.asarray(limits).T
    limit_count = int(np.count_nonzero((joint < low - 1e-4) | (joint > high + 1e-4)))
    velocity = np.diff(joint, axis=0) * fps
    velocity_count = int(np.count_nonzero(np.abs(velocity) > max_velocity))
    root_speed = np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1) * fps
    jump_count = int(np.count_nonzero(root_speed > 3.0))
    quat_dot = np.abs(np.sum(qpos[:-1, 3:7] * qpos[1:, 3:7], axis=1))
    root_angular_speed = 2 * np.arccos(np.clip(quat_dot, -1, 1)) * fps
    orientation_jump_count = int(np.count_nonzero(root_angular_speed > 10.0))
    acceleration = np.diff(joint, n=2, axis=0) * fps * fps
    project_limits = per_joint_motion_limits(joint, fps, names)
    release_exceedance = any(
        item["release"]["velocity_exceedance_frames"] or item["release"]["acceleration_exceedance_frames"]
        for item in project_limits.values())
    validation_exceedance = any(
        item["validation"]["velocity_exceedance_frames"] or item["validation"]["acceleration_exceedance_frames"]
        for item in project_limits.values())
    oscillation_count = int(np.count_nonzero(np.abs(acceleration) > 100))
    drift = float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2]))
    if limit_count:
        warnings.append(f"{limit_count} joint-limit violations")
    if velocity_count:
        warnings.append(f"{velocity_count} joint velocities exceed {max_velocity:.3f} rad/s (screening threshold)")
    if jump_count:
        warnings.append(f"{jump_count} root steps exceed 3 m/s (screening threshold)")
    if orientation_jump_count:
        warnings.append(f"{orientation_jump_count} root orientation steps exceed 10 rad/s (screening threshold)")
    if oscillation_count:
        warnings.append(f"{oscillation_count} joint accelerations exceed 100 rad/s^2 (screening threshold)")
    if release_exceedance:
        warnings.append("At least one joint exceeds a project release velocity or acceleration threshold; see per_joint_motion_limits")
    if validation_exceedance:
        warnings.append("At least one joint exceeds a project validation velocity or acceleration threshold; see per_joint_motion_limits")
    if drift > 2:
        warnings.append(f"Root XY displacement is {drift:.2f} m; inspect whether intended")
    foot = {"evaluated": False, "reason": "MuJoCo model unavailable"}
    if model is not None:
        import mujoco as mj

        data = mj.MjData(model)
        foot_ids = [i for i in range(model.nbody) if any(s in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or "").lower()
                    for s in ("left_ankle_roll_link", "right_ankle_roll_link"))]
        if len(foot_ids) == 2:
            positions = []
            for row in qpos:
                data.qpos[:] = row
                mj.mj_forward(model, data)
                positions.append(data.xpos[foot_ids].copy())
            positions = np.asarray(positions)
            # Body origins are proxies for soles. Report diagnostics, not a contact verdict.
            min_z = float(positions[:, :, 2].min())
            near_ground = positions[:-1, :, 2] < 0.1
            speed = np.linalg.norm(np.diff(positions[:, :, :2], axis=0), axis=2) * fps
            slide_count = int(np.count_nonzero(near_ground & (speed > 0.1)))
            foot = {"evaluated": True, "method": "ankle body-origin proxy; contact not inferred",
                    "minimum_ankle_z_m": min_z, "near_ground_fast_xy_steps": slide_count}
            if min_z < -0.02:
                warnings.append("Ankle body origin falls below ground; inspect full foot geometry")
            if slide_count:
                warnings.append(f"{slide_count} near-ground ankle XY steps exceed 0.1 m/s")
        else:
            foot = {"evaluated": False, "reason": "Expected ankle bodies not found in model"}
    return {"passed_structural_checks": True, "frames": int(len(qpos)), "fps": float(fps),
            "duration_seconds": float(len(qpos) / fps), "qpos_shape": list(qpos.shape),
            "joint_names": list(names), "coordinate_convention": "MuJoCo world, meters, radians, root quaternion wxyz",
            "max_joint_velocity_rad_s": float(np.max(np.abs(velocity))) if velocity.size else 0.0,
            "joint_limit_violations": limit_count, "joint_velocity_exceedances": velocity_count,
            "per_joint_motion_limits": project_limits,
            "project_release_exceedance": bool(release_exceedance),
            "project_validation_exceedance": bool(validation_exceedance),
            "project_limits_note": "Data-derived pipeline thresholds checked at every frame; not manufacturer-certified hardware limits.",
            "root_speed_exceedances": jump_count, "root_orientation_exceedances": orientation_jump_count,
            "oscillation_exceedances": oscillation_count,
            "root_xy_displacement_m": drift, "foot_checks": foot, "warnings": warnings,
            "physical_execution_safe": False}
