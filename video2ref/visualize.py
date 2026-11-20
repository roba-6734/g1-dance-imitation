from pathlib import Path

import numpy as np


def render_robot(qpos, fps, model, path: Path, width=640, height=480):
    import imageio.v2 as imageio
    import mujoco as mj

    data = mj.MjData(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mj.Renderer(model, width=width, height=height) as renderer:
        with imageio.get_writer(str(path), format="FFMPEG", fps=fps, codec="libx264", macro_block_size=16) as writer:
            for row in qpos:
                data.qpos[:] = row
                mj.mj_forward(model, data)
                camera = mj.MjvCamera()
                mj.mjv_defaultCamera(camera)
                camera.type = mj.mjtCamera.mjCAMERA_FREE
                camera.lookat[:] = data.qpos[:3] + np.array([0, 0, 0.2])
                camera.distance = 3.0
                camera.elevation = -15
                camera.azimuth = 140
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
