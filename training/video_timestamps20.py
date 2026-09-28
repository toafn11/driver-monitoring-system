"""Read decoded presentation timestamps without an output muxer or FPS conversion."""
import json
import shutil
import subprocess
import numpy as np

TIMESTAMP_VERSION = "ffprobe-best-effort-v1"


def decoded_timestamps(path):
    executable = shutil.which("ffprobe")
    if not executable:
        raise RuntimeError("ffprobe is required for timestamp validation; install FFmpeg or use the Kaggle image containing it")
    result = subprocess.run([
        executable, "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)
    ], capture_output=True, text=True, timeout=1800)
    if result.returncode or result.stderr.strip():
        raise ValueError(f"FFprobe decode errors for {path}: {result.stderr[-3000:]}")
    frames = json.loads(result.stdout).get("frames", [])
    if len(frames) < 2 or any("best_effort_timestamp_time" not in f for f in frames):
        raise ValueError(f"Missing decoded frames/timestamps: {path}")
    times = np.asarray([float(f["best_effort_timestamp_time"]) for f in frames], dtype=np.float64)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"Non-finite, duplicate or backward source timestamps: {path}")
    return times - times[0]
