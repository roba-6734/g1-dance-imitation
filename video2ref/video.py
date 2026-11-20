import json
import shutil
import subprocess
from pathlib import Path


def probe(path: Path) -> dict:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,avg_frame_rate,nb_frames,duration:stream_tags=rotate:stream_side_data=rotation:format=duration",
             "-of", "json", str(path)],
            check=True, capture_output=True, text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot inspect video {path}; install ffmpeg/ffprobe and check the video: {exc}") from exc
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError(f"No video stream in {path}")
    stream = data["streams"][0]
    n, d = stream.get("avg_frame_rate", "0/1").split("/")
    fps = float(n) / float(d)
    duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0)
    if fps <= 0 or duration <= 0 or int(stream.get("width", 0)) < 64 or int(stream.get("height", 0)) < 64:
        raise ValueError("Video needs positive FPS and duration and at least 64x64 resolution")
    rotation = next((entry["rotation"] for entry in stream.get("side_data_list", [])
                     if "rotation" in entry), stream.get("tags", {}).get("rotate", 0))
    rotation = int(rotation) % 360
    display_width, display_height = int(stream["width"]), int(stream["height"])
    if rotation in (90, 270):
        display_width, display_height = display_height, display_width
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "display_width": display_width, "display_height": display_height,
            "rotation_degrees": rotation,
            "fps": fps, "duration": duration,
            "frame_count": int(stream["nb_frames"]) if stream.get("nb_frames", "").isdigit() else None}


def normalize(source: Path, dest: Path, overwrite: bool = False,
              max_dimension: int = 1280, clip_seconds: float | None = None) -> dict:
    metadata = probe(source)
    if max_dimension < 256 or max_dimension % 2:
        raise ValueError("max_dimension must be an even integer of at least 256")
    if clip_seconds is not None and not 0 < clip_seconds <= metadata["duration"]:
        raise ValueError("clip_seconds must be positive and no longer than the video")
    scale = min(1., max_dimension / max(metadata["display_width"], metadata["display_height"]))
    width = max(2, int(metadata["display_width"] * scale) // 2 * 2)
    height = max(2, int(metadata["display_height"] * scale) // 2 * 2)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not dest.exists():
        if not shutil.which("ffmpeg"):
            raise FileNotFoundError("ffmpeg is required for 30 fps video normalization")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
        if clip_seconds is not None:
            command += ["-t", str(clip_seconds)]
        command += ["-map", "0:v:0", "-an", "-vf", f"fps=30,scale={width}:{height}",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(dest)]
        subprocess.run(command, check=True)
    normalized = probe(dest)
    if abs(normalized["fps"] - 30) > 0.01:
        raise RuntimeError(f"FPS normalization failed: {normalized['fps']}")
    if (normalized["display_width"], normalized["display_height"]) != (width, height):
        raise RuntimeError("Video orientation or scaling did not match the expected display dimensions")
    return {"original": metadata, "normalized": normalized,
            "max_dimension": max_dimension, "clip_seconds": clip_seconds}
