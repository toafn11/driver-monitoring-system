"""Versioned, NumPy-only features shared by video preparation and inference.

This is a NEW schema. Do not feed it to the legacy 12-feature checkpoints.
The last feature is a detection mask, not MediaPipe confidence.
"""
from collections import deque
import numpy as np

SCHEMA = "geometric20-v2"
FEATURE_NAMES = [
    "EAR_L", "EAR_R", "EAR_avg", "MAR", "Yaw_div90", "Pitch_div90",
    "Roll_div90", "Gaze_X", "Gaze_Y", "PERCLOS_window",
    "EyesClosed_seconds_div5", "MouthOpen_seconds_div3",
    "Yaw_velocity_per_second", "Pitch_velocity_per_second", "Gaze_mag",
    "EAR_window_ratio", "Closure_onsets_per_second", "Head_stability",
    "MAR_velocity_per_second", "Face_valid",
]


def make_features(raw, timestamps, valid, min_valid=0.8, max_missing_seconds=0.2):
    """Build (T,20) from ordered (T,9) geometry; reject unusable windows.

    EAR baseline is the window's 80th percentile, NOT personal calibration.
    PERCLOS uses valid observations within this window. Durations are lower
    bounds when an event starts before the window, and reset on missing faces.
    No interpolation across missing faces; no fabricated closed eyes.
    """
    raw = np.asarray(raw, dtype=np.float32)
    times = np.asarray(timestamps, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if raw.ndim != 2 or raw.shape[1] != 9 or len(raw) < 2:
        raise ValueError("Expected at least two frames with 9 geometric features")
    if times.shape != (len(raw),) or valid.shape != times.shape:
        raise ValueError("Frame/timestamp/mask lengths differ")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    if not np.isfinite(raw[valid]).all():
        raise ValueError("Non-finite detected geometry")
    dt = np.r_[np.median(np.diff(times)), np.diff(times)]
    if np.max(dt) > 2.5 * np.median(dt):
        return None  # dropped source frames must not be compressed into time
    if valid.mean() < min_valid:
        return None
    missing_time = 0.0
    for step, detected in zip(dt, valid):
        missing_time = 0.0 if detected else missing_time + step
        if missing_time > max_missing_seconds + 1e-8:
            return None
    out = np.zeros((len(raw), 20), dtype=np.float32)
    out[valid, :9] = raw[valid]
    closed = (raw[:, 2] < 0.22) & valid
    mouth = (raw[:, 3] > 0.55) & valid
    elapsed_closed = elapsed_mouth = 0.0
    for i in range(len(raw)):
        elapsed_closed = elapsed_closed + dt[i] if closed[i] else 0.0
        elapsed_mouth = elapsed_mouth + dt[i] if mouth[i] else 0.0
        out[i, 10] = min(elapsed_closed / 5.0, 1.0)
        out[i, 11] = min(elapsed_mouth / 3.0, 1.0)
    out[valid, 9] = np.sum(dt * closed) / np.sum(dt * valid)
    pairs = valid[1:] & valid[:-1]
    for src, dst in [(4, 12), (5, 13), (3, 18)]:
        velocities = np.diff(out[:, src]) / np.diff(times)
        out[1:, dst] = np.where(pairs, np.clip(velocities, -5, 5), 0)
    out[valid, 14] = np.clip(np.linalg.norm(raw[valid, 7:9], axis=1), 0, 1)
    baseline = max(float(np.percentile(raw[valid, 2], 80)), 0.05)
    out[valid, 15] = np.clip(raw[valid, 2] / baseline, 0, 2)
    onsets = np.sum(pairs & ~closed[:-1] & closed[1:])
    out[valid, 16] = onsets / np.sum(dt * valid)
    out[valid, 17] = np.clip(1 - 5 * raw[valid, 4:6].std(axis=0).sum(), 0, 1)
    out[~valid] = 0
    out[:, 19] = valid
    return out


class FeatureWindow:
    """Same preprocessing for a timestamped live stream or recorded frames."""
    def __init__(self, seq_len=60):
        self.frames = deque(maxlen=seq_len)
        self.seq_len = seq_len

    def reset(self):
        self.frames.clear()

    def push(self, raw9, timestamp):
        if self.frames and timestamp <= self.frames[-1][1]:
            raise ValueError("Non-increasing stream timestamp")
        self.frames.append((np.zeros(9) if raw9 is None else raw9,
                            timestamp, raw9 is not None))
        if len(self.frames) < self.seq_len:
            return None
        raw, times, valid = zip(*self.frames)
        return make_features(raw, times, valid)
