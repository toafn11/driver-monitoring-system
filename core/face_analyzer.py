"""
core/face_analyzer.py
=====================
Face analysis core using MediaPipe FaceLandmarker (Tasks API v1.0).
Computes EAR, MAR, PERCLOS, Head Pose (yaw/pitch/roll), Gaze Direction,
and temporal features (blink rate, velocity, stability).

ANTI-OVERFITTING DESIGN:
- No raw pixels returned — only geometric features
- Features are invariant to background, skin color, clothing
- Normalized relative to each individual's face geometry
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.spatial.distance import euclidean
from dataclasses import dataclass, field
from typing import Optional, Tuple
from collections import deque
from pathlib import Path

# ─── MediaPipe Face Mesh Landmark Indices ─────────────────────────────────────
# Reference: https://developers.google.com/mediapipe/solutions/vision/face_landmarker

# Left eye (from camera perspective = user's RIGHT eye)
LEFT_EYE_IDX   = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_IDX  = [474, 475, 476, 477]

# Right eye (from camera perspective = user's LEFT eye)
RIGHT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_IRIS_IDX = [469, 470, 471, 472]

# Mouth outer corners and midpoints
MOUTH_OUTER    = [61, 291, 13, 14]   # left_corner, right_corner, top_mid, bot_mid

# Head pose: 6 standard 3D reference points
HEAD_POSE_IDX  = [1, 152, 33, 263, 61, 291]

# Resolve model file path
_MODEL_PATH = Path(__file__).parent.parent / "models" / "face_landmarker.task"
if not _MODEL_PATH.exists():
    _MODEL_PATH = Path("models/face_landmarker.task")


# ─── Output feature names (must match FEATURE_DIM order) ─────────────────────
FEATURE_NAMES = [
    # Core biometric features (12)
    "EAR_L", "EAR_R", "EAR_avg", "MAR",
    "Yaw", "Pitch", "Roll",
    "Gaze_X", "Gaze_Y",
    "PERCLOS", "EyesClosed", "MouthOpen",
    # Temporal dynamics features (8) — computed over sequence window
    "Yaw_vel", "Pitch_vel",       # head rotation velocity (deg/frame)
    "Gaze_mag",                    # gaze displacement magnitude from center
    "EAR_ratio",                   # relative EAR vs personal baseline
    "Blink_events",                # normalized blink count in window
    "Head_stability",              # std of head rotation in window
    "MAR_vel",                     # mouth opening velocity
    "Face_conf",                   # face detection confidence (0-1)
]
FEATURE_DIM = len(FEATURE_NAMES)  # 20


@dataclass
class FaceFeatures:
    """
    All geometric features extracted from a single frame.
    No pixel/color information — ensures anti-bias properties.
    """
    ear_left:  float = 0.0
    ear_right: float = 0.0
    ear_avg:   float = 0.0
    mar:       float = 0.0
    yaw:       float = 0.0   # + = looking right, - = looking left (degrees)
    pitch:     float = 0.0   # + = looking up,   - = looking down (degrees)
    roll:      float = 0.0
    gaze_x:    float = 0.0   # + = looking right (normalized -1..1)
    gaze_y:    float = 0.0   # + = looking down
    perclos:   float = 0.0
    blink_rate:            float = 0.0
    eyes_closed_duration:  float = 0.0
    mouth_open_duration:   float = 0.0
    detection_confidence:  float = 0.0
    gaze_zone:             str   = "UNKNOWN"


GAZE_ZONE_LABELS = {
    "FORWARD":      ("Looking forward",           1.00),
    "RIGHT_MIRROR": ("Checking right mirror",     0.75),
    "LEFT_MIRROR":  ("Checking left mirror",      0.75),
    "LOOKING_DOWN": ("Looking down",              0.45),
    "LOOKING_UP":   ("Looking up",                0.70),
    "FAR_RIGHT":    ("Distracted (right)",        0.20),
    "FAR_LEFT":     ("Distracted (left)",         0.20),
    "EYES_CLOSED":  ("Eyes closed / drowsy",      0.00),
    "YAWNING":      ("Yawning",                   0.50),
    "UNKNOWN":      ("Unknown",                   0.50),
}


class FaceAnalyzer:
    """
    Real-time face analysis from OpenCV BGR frames.
    Uses MediaPipe FaceLandmarker Tasks API v1.0+.

    Extracts 20 geometric features per frame including:
    - Eye Aspect Ratio (EAR) — drowsiness indicator
    - Mouth Aspect Ratio (MAR) — yawn indicator
    - Head pose (yaw/pitch/roll) — attention direction
    - Iris gaze vector — gaze zone classification
    - Temporal features: PERCLOS, blink rate, motion velocity
    """

    EAR_THRESHOLD  = 0.22   # Eyes considered closed below this
    MAR_THRESHOLD  = 0.55   # Mouth considered open (yawn) above this
    PERCLOS_WINDOW = 90     # Frames for PERCLOS calculation (~3s @ 30fps)
    BLINK_WINDOW   = 300    # Frames for blink rate calculation (~10s)

    YAW_MIRROR_THRESHOLD  = 28   # degrees: mirror check zone
    YAW_AWAY_THRESHOLD    = 45   # degrees: distracted zone
    PITCH_DOWN_THRESHOLD  = -18  # degrees: looking down
    PITCH_UP_THRESHOLD    = 15   # degrees: looking up

    # 3D face model reference points for solvePnP
    MODEL_POINTS_3D = np.array([
        (0.0,    0.0,    0.0),
        (0.0,   -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0,  170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0,  -150.0, -125.0),
    ], dtype=np.float64)

    def __init__(self, model_path: Optional[str] = None, use_image_mode: bool = False,
                 pose_convention: str = "legacy"):
        """
        Args:
            model_path: Path to face_landmarker.task file.
            use_image_mode: If True, use RunningMode.IMAGE (no timestamp needed).
                            If False, use RunningMode.VIDEO (needs sequential frames).
        """
        if pose_convention not in ("legacy", "camera-v2"):
            raise ValueError("Unknown pose convention")
        self.pose_convention = pose_convention
        path = model_path or str(_MODEL_PATH)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Download with:\n"
                "  python -c \"import urllib.request; "
                "urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task', "
                "'models/face_landmarker.task')\""
            )

        base_options = mp_python.BaseOptions(model_asset_path=path)
        running_mode = (mp_vision.RunningMode.IMAGE if use_image_mode
                        else mp_vision.RunningMode.VIDEO)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._use_image_mode = use_image_mode
        self._frame_ms = 0

        # Temporal state buffers
        self._ear_history   = deque(maxlen=self.PERCLOS_WINDOW)
        self._blink_history = deque(maxlen=self.BLINK_WINDOW)
        self._in_blink      = False
        self._blink_count   = 0
        self._eyes_closed_start: Optional[float] = None
        self._mouth_open_start:  Optional[float] = None
        self._fps = 30.0

    def set_fps(self, fps: float):
        """Update FPS for accurate temporal calculations."""
        self._fps = max(1.0, fps)

    def analyze(self, frame: np.ndarray) -> Optional[FaceFeatures]:
        """
        Analyze one BGR frame → FaceFeatures or None if no face detected.
        """
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._use_image_mode:
            result = self.landmarker.detect(mp_image)
        else:
            self._frame_ms += int(1000 / self._fps)
            result = self.landmarker.detect_for_video(mp_image, self._frame_ms)

        if not result.face_landmarks:
            return None

        lm_list = result.face_landmarks[0]
        pts = np.array([[lm.x * w, lm.y * h, lm.z * w]
                        for lm in lm_list], dtype=np.float64)

        feats = FaceFeatures()
        feats.ear_left  = self._calc_ear(pts, LEFT_EYE_IDX)
        feats.ear_right = self._calc_ear(pts, RIGHT_EYE_IDX)
        feats.ear_avg   = (feats.ear_left + feats.ear_right) / 2.0
        feats.mar       = self._calc_mar(pts)
        feats.yaw, feats.pitch, feats.roll = self._calc_head_pose(pts, w, h)

        # Iris gaze available with full FaceLandmarker model (478+ points)
        if len(pts) >= 478:
            feats.gaze_x, feats.gaze_y = self._calc_gaze(pts)

        feats.perclos, feats.blink_rate = self._update_temporal(feats.ear_avg)
        feats.eyes_closed_duration = self._get_eyes_closed_duration(feats.ear_avg)
        feats.mouth_open_duration  = self._get_mouth_open_duration(feats.mar)
        feats.gaze_zone = self._classify_gaze_zone(feats)
        feats.detection_confidence = 1.0

        return feats

    def to_feature_vector(self, feats: FaceFeatures) -> np.ndarray:
        """
        Convert FaceFeatures to normalized 12-dim vector (core features only).
        Temporal dynamics (dims 12-19) are computed at sequence level.
        """
        return np.array([
            feats.ear_left,
            feats.ear_right,
            feats.ear_avg,
            feats.mar,
            feats.yaw   / 90.0,
            feats.pitch / 90.0,
            feats.roll  / 90.0,
            feats.gaze_x,
            feats.gaze_y,
            feats.perclos,
            min(feats.eyes_closed_duration / 5.0, 1.0),
            min(feats.mouth_open_duration  / 3.0, 1.0),
        ], dtype=np.float32)

    # ─── EAR ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _calc_ear(pts: np.ndarray, eye_idx: list) -> float:
        """Eye Aspect Ratio: measures how open the eye is (0=closed, ~0.3=open)."""
        p1, p2, p3, p4, p5, p6 = [pts[i][:2] for i in eye_idx]
        vertical   = euclidean(p2, p6) + euclidean(p3, p5)
        horizontal = euclidean(p1, p4)
        return vertical / (2.0 * horizontal) if horizontal > 1e-6 else 0.0

    # ─── MAR ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _calc_mar(pts: np.ndarray) -> float:
        """Mouth Aspect Ratio: measures how open the mouth is (0=closed, >0.55=yawning)."""
        left, right, top, bot = [pts[i][:2] for i in MOUTH_OUTER]
        vertical   = euclidean(top, bot)
        horizontal = euclidean(left, right)
        return vertical / horizontal if horizontal > 1e-6 else 0.0

    # ─── Head Pose ────────────────────────────────────────────────────────────
    def _calc_head_pose(self, pts: np.ndarray, w: int, h: int) -> Tuple[float, float, float]:
        """
        Estimate head pose angles using PnP solver.
        Returns (yaw, pitch, roll) in degrees.
        """
        img_points = pts[HEAD_POSE_IDX, :2].astype(np.float64)
        cam_matrix = np.array([
            [w, 0, w / 2],
            [0, w, h / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        ok, rvec, _ = cv2.solvePnP(
            self.MODEL_POINTS_3D, img_points, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return 0.0, 0.0, 0.0

        rot_mat, _ = cv2.Rodrigues(rvec)
        if self.pose_convention == "camera-v2":
            # The reference face has +Y up, whereas image coordinates have +Y
            # down. Remove the neutral 180-degree X rotation before extracting
            # camera-axis angles: X=pitch, Y=yaw, Z=roll. Legacy weights retain
            # their original convention by default.
            rot_mat = rot_mat @ np.diag([1.0, -1.0, -1.0])
            sy = np.hypot(rot_mat[0, 0], rot_mat[1, 0])
            pitch = np.degrees(np.arctan2(rot_mat[2, 1], rot_mat[2, 2]))
            yaw = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
            roll = np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0]))
            return float(yaw), float(pitch), float(roll)
        sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
        if sy > 1e-6:
            roll  = np.degrees(np.arctan2(rot_mat[2, 1], rot_mat[2, 2]))
            pitch = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
            yaw   = np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0]))
        else:
            roll  = np.degrees(np.arctan2(-rot_mat[1, 2], rot_mat[1, 1]))
            pitch = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
            yaw   = 0.0
        return float(yaw), float(pitch), float(roll)

    # ─── Gaze ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _calc_gaze(pts: np.ndarray) -> Tuple[float, float]:
        """
        Compute normalized gaze vector from iris position within eye bounding box.
        Returns (gaze_x, gaze_y) in range [-1, 1] where 0=center.
        """
        if len(pts) < 478:
            return 0.0, 0.0
        try:
            l_iris = pts[LEFT_IRIS_IDX, :2].mean(axis=0)
            r_iris = pts[RIGHT_IRIS_IDX, :2].mean(axis=0)

            l_eye_pts = pts[LEFT_EYE_IDX, :2]
            l_min, l_max = l_eye_pts.min(axis=0), l_eye_pts.max(axis=0)
            l_size = l_max - l_min

            r_eye_pts = pts[RIGHT_EYE_IDX, :2]
            r_min, r_max = r_eye_pts.min(axis=0), r_eye_pts.max(axis=0)
            r_size = r_max - r_min

            if l_size[0] < 1 or r_size[0] < 1:
                return 0.0, 0.0

            l_gaze = (l_iris - l_min) / (l_size + 1e-6)
            r_gaze = (r_iris - r_min) / (r_size + 1e-6)
            avg    = (l_gaze + r_gaze) / 2.0
            return (float(np.clip((avg[0] - 0.5) * 2, -1, 1)),
                    float(np.clip((avg[1] - 0.5) * 2, -1, 1)))
        except Exception:
            return 0.0, 0.0

    # ─── Temporal State ───────────────────────────────────────────────────────
    def _update_temporal(self, ear: float) -> Tuple[float, float]:
        """Update PERCLOS and blink rate buffers."""
        is_closed = ear < self.EAR_THRESHOLD
        self._ear_history.append(1 if is_closed else 0)

        if is_closed:
            self._in_blink = True
        elif self._in_blink:
            self._blink_count += 1
            self._in_blink = False
            self._blink_history.append(1)
        else:
            self._blink_history.append(0)

        perclos = (sum(self._ear_history) / len(self._ear_history)
                   if self._ear_history else 0.0)
        blink_rate = (sum(self._blink_history) / (len(self._blink_history) / self._fps) * 60.0
                      if len(self._blink_history) > 10 else 0.0)
        return perclos, blink_rate

    def _get_eyes_closed_duration(self, ear: float) -> float:
        """Returns cumulative seconds eyes have been continuously closed."""
        if ear < self.EAR_THRESHOLD:
            if self._eyes_closed_start is None:
                self._eyes_closed_start = 0.0
            self._eyes_closed_start += 1.0 / self._fps
            return self._eyes_closed_start
        self._eyes_closed_start = None
        return 0.0

    def _get_mouth_open_duration(self, mar: float) -> float:
        """Returns cumulative seconds mouth has been continuously open (yawning)."""
        if mar > self.MAR_THRESHOLD:
            if self._mouth_open_start is None:
                self._mouth_open_start = 0.0
            self._mouth_open_start += 1.0 / self._fps
            return self._mouth_open_start
        self._mouth_open_start = None
        return 0.0

    # ─── Gaze Zone Classifier ─────────────────────────────────────────────────
    def _classify_gaze_zone(self, f: FaceFeatures) -> str:
        """
        Rule-based gaze zone classification based on head pose and eye state.
        Zones: FORWARD, RIGHT_MIRROR, LEFT_MIRROR, FAR_RIGHT, FAR_LEFT,
               LOOKING_DOWN, LOOKING_UP, EYES_CLOSED, YAWNING
        """
        if f.ear_avg < self.EAR_THRESHOLD and f.eyes_closed_duration > 0.5:
            return "EYES_CLOSED"
        if f.mar > self.MAR_THRESHOLD:
            return "YAWNING"

        yaw, pitch = f.yaw, f.pitch
        if abs(yaw) < self.YAW_MIRROR_THRESHOLD and abs(pitch) < abs(self.PITCH_DOWN_THRESHOLD):
            return "FORWARD"
        if yaw > self.YAW_AWAY_THRESHOLD:
            return "FAR_RIGHT"
        if yaw > self.YAW_MIRROR_THRESHOLD:
            return "RIGHT_MIRROR"
        if yaw < -self.YAW_AWAY_THRESHOLD:
            return "FAR_LEFT"
        if yaw < -self.YAW_MIRROR_THRESHOLD:
            return "LEFT_MIRROR"
        if pitch < self.PITCH_DOWN_THRESHOLD:
            return "LOOKING_DOWN"
        if pitch > self.PITCH_UP_THRESHOLD:
            return "LOOKING_UP"
        return "FORWARD"

    def reset(self):
        """Reset all temporal state (call when switching to a new video/session)."""
        self._ear_history.clear()
        self._blink_history.clear()
        self._in_blink = False
        self._blink_count = 0
        self._eyes_closed_start = None
        self._mouth_open_start  = None
        self._frame_ms = 0

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()


# ─── Sequence-Level Feature Enrichment ────────────────────────────────────────
def enrich_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Enrich a (seq_len, 12) core feature array with 8 temporal dynamic features
    to produce a (seq_len, 20) enriched array.

    New features (dims 12-19):
        12: Yaw_vel        — head yaw velocity (delta yaw per frame / 90)
        13: Pitch_vel      — head pitch velocity
        14: Gaze_mag       — gaze displacement magnitude from center
        15: EAR_ratio      — EAR normalized to personal 80th percentile baseline
        16: Blink_events   — normalized blink event count in window
        17: Head_stability — inverse of head rotation std (1 = very stable)
        18: MAR_vel        — mouth opening velocity
        19: Face_conf      — face confidence (1.0 = detected, 0.0 = not)
    """
    T = seq.shape[0]
    enriched = np.zeros((T, FEATURE_DIM), dtype=np.float32)
    enriched[:, :12] = seq[:, :12]

    # 12: Yaw velocity
    yaw_col = seq[:, 4]   # normalized yaw (already /90)
    yaw_vel = np.diff(yaw_col, prepend=yaw_col[0])
    enriched[:, 12] = np.clip(yaw_vel, -0.5, 0.5)

    # 13: Pitch velocity
    pitch_col = seq[:, 5]
    pitch_vel = np.diff(pitch_col, prepend=pitch_col[0])
    enriched[:, 13] = np.clip(pitch_vel, -0.5, 0.5)

    # 14: Gaze magnitude (distance from center gaze)
    gaze_mag = np.sqrt(seq[:, 7] ** 2 + seq[:, 8] ** 2)
    enriched[:, 14] = np.clip(gaze_mag, 0.0, 1.0)

    # 15: EAR ratio relative to personal baseline (80th percentile of this window)
    ear_col    = seq[:, 2]
    ear_base   = np.percentile(ear_col, 80)
    ear_ratio  = (ear_col / ear_base) if ear_base > 0.05 else np.ones(T)
    enriched[:, 15] = np.clip(ear_ratio, 0.0, 2.0)

    # 16: Blink events (transitions from open to closed)
    is_closed    = (ear_col < 0.22).astype(float)
    blink_events = np.diff(is_closed, prepend=0)
    blink_events = np.clip(blink_events, 0, 1)    # only count openings→closings
    # Spread blink event flag across the sequence
    blink_count  = blink_events.sum() / max(T, 1)
    enriched[:, 16] = blink_count   # same value broadcast — model learns count pattern

    # 17: Head stability (low value = erratic head movement)
    head_std = np.std(seq[:, 4]) + np.std(seq[:, 5])  # yaw + pitch std
    stability = float(np.clip(1.0 - head_std * 5.0, 0.0, 1.0))
    enriched[:, 17] = stability

    # 18: MAR velocity (mouth opening speed)
    mar_col = seq[:, 3]
    mar_vel = np.diff(mar_col, prepend=mar_col[0])
    enriched[:, 18] = np.clip(mar_vel, -0.5, 0.5)

    # 19: Face confidence (from column 11 if available, else 1.0)
    enriched[:, 19] = 1.0

    return enriched
