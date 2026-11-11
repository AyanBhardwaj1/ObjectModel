"""Write rendered frames to H.264 MP4 through the ffmpeg binary (no Python video dependency)."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np


def write_mp4(frames: list[np.ndarray], path: str, fps: int = 30, crf: int = 24) -> None:
    if not frames:
        raise ValueError("no frames to write; run the task with render=True")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is needed to write video")
    h, w, _ = frames[0].shape
    cmd = [ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-movflags", "+faststart", path]
    proc = subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[:400]}")
