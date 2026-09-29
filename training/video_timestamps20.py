"""Read decoded presentation timestamps without an output muxer or FPS conversion."""
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np

TIMESTAMP_VERSION = "ffprobe-best-effort-v1"


def timestamp_exception(path):
    if Path(path).as_posix().lower().endswith('/fold1_part2/fold1_part2/11/0.mp4'):
        return "rldd-11-alert-duplicate-15027-v1"
    return None


def validate_timestamps(times, path):
    delta = np.diff(times)
    duplicate = np.flatnonzero(delta == 0)
    if not np.isfinite(times).all() or np.any(delta < 0):
        raise ValueError(f"Non-finite or backward source timestamps: {path}")
    if len(duplicate):
        verified_exception = (timestamp_exception(path) is not None and len(times) == 18818
                              and duplicate.tolist() == [15026]
                              and abs(times[15026] - 501.9471) < 1e-7)
        if not verified_exception:
            raise ValueError(f"Unverified duplicate source timestamps: {path}")
        print(f"Verified timestamp exception: {path}; skip decoded frame 15027 during sampling; keep original PTS", flush=True)
    # Keep one timestamp per decoded frame so decoder alignment remains checkable.
    return times - times[0]


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
    return validate_timestamps(times, path)
