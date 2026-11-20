import json
from pathlib import Path

import numpy as np

from .motion import check_qpos


def model_joints(model):
    import mujoco as mj

    if model.nq != 36 or model.nv != 35 or model.nu != 29:
        raise ValueError(f"G1 model mismatch: nq={model.nq}, nv={model.nv}, nu={model.nu}; expected 36/35/29")
    if model.jnt_type[0] != mj.mjtJoint.mjJNT_FREE or model.jnt_qposadr[0] != 0:
        raise ValueError("G1 model does not start with a free root joint")
    names, limits = [], []
    for jid in range(1, model.njnt):
        if model.jnt_type[jid] != mj.mjtJoint.mjJNT_HINGE or model.jnt_qposadr[jid] != jid + 6:
            raise ValueError(f"Unexpected joint type or qpos address at joint {jid}")
        names.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid))
        limits.append(model.jnt_range[jid].copy())
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("G1 model must have 29 unique hinge joints")
    actuator_joint_ids = [int(model.actuator_trnid[i, 0]) for i in range(model.nu)]
    if actuator_joint_ids != list(range(1, 30)):
        raise ValueError("G1 actuator order differs from the 29 qpos hinge joints")
    return names, np.asarray(limits)


def retarget_gvhmr(pred_file: Path, gmr_repo: Path, output: Path, fps=30):
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT
    from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

    body_models = gmr_repo / "assets" / "body_models"
    if not body_models.exists() or not list(body_models.rglob("SMPLX_NEUTRAL.npz")):
        raise FileNotFoundError(f"GMR needs licensed SMPLX_NEUTRAL.npz under {body_models}; see README for .pkl setup")
    smplx_data, body_model, smplx_output, height = load_gvhmr_pred_file(str(pred_file), body_models)
    frames, actual_fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=fps)
    if abs(actual_fps - fps) > 1e-6:
        raise RuntimeError(f"GMR aligned FPS {actual_fps} differs from requested {fps}")
    config_path = IK_CONFIG_DICT["smplx"]["unitree_g1"]
    config = json.loads(Path(config_path).read_text())
    required = {value[0] for table in ("ik_match_table1", "ik_match_table2") for value in config[table].values()}
    missing = required - set(frames[0])
    if missing:
        raise ValueError(f"GVHMR SMPLX output lacks IK bodies: {sorted(missing)}")
    retargeter = GeneralMotionRetargeting(src_human="smplx", tgt_robot="unitree_g1",
                                           actual_human_height=height, verbose=False)
    names, limits = model_joints(retargeter.model)
    qpos = check_qpos(np.stack([retargeter.retarget(frame) for frame in frames]))
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "raw_qpos.npy", qpos)
    return qpos, names, limits, str(ROBOT_XML_DICT["unitree_g1"]), str(config_path)
