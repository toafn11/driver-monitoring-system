"""
core/face_analyzer.py
=====================
Lõi phân tích khuôn mặt dùng MediaPipe FaceLandmarker (Tasks API v1.0).
Tính EAR, MAR, PERCLOS, Head Pose (yaw/pitch/roll), Gaze Direction.

THIẾT KẾ ĐỂ TRÁNH HỌC VẸT:
- Không trả về raw pixel → chỉ trả về geometric features
- Features này không phụ thuộc background, màu da, quần áo
- Normalized theo kích thước khuôn mặt từng người
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.spatial.distance import euclidean
from dataclasses import dataclass
from typing import Optional
from collections import deque
from pathlib import Path

# ─── MediaPipe Face Mesh Landmark Indices ─────────────────────────────────────
# Tham khảo: https://developers.google.com/mediapipe/solutions/vision/face_landmarker

# Mắt trái (từ góc nhìn camera, là bên PHẢI người dùng)
LEFT_EYE_IDX   = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_IDX  = [474, 475, 476, 477]

# Mắt phải
RIGHT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_IRIS_IDX = [469, 470, 471, 472]

# Miệng
MOUTH_OUTER    = [61, 291, 13, 14]   # left_corner, right_corner, top_mid, bot_mid

# Head pose: 6 điểm chuẩn 3D
HEAD_POSE_IDX  = [1, 152, 33, 263, 61, 291]

# Tìm model file
_MODEL_PATH = Path(__file__).parent.parent / "models" / "face_landmarker.task"
if not _MODEL_PATH.exists():
    _MODEL_PATH = Path("models/face_landmarker.task")


@dataclass
class FaceFeatures:
    """
    Tất cả geometric features từ 1 frame.
    Không có thông tin pixel/màu sắc → anti-bias.
    """
    ear_left:  float = 0.0
    ear_right: float = 0.0
    ear_avg:   float = 0.0
    mar:       float = 0.0
    yaw:       float = 0.0   # + = nhìn phải, - = nhìn trái (degrees)
    pitch:     float = 0.0   # + = ngước, - = cúi (degrees)
    roll:      float = 0.0
    gaze_x:    float = 0.0   # + = nhìn phải (normalized -1..1)
    gaze_y:    float = 0.0   # + = nhìn xuống
    perclos:   float = 0.0
    blink_rate:            float = 0.0
    eyes_closed_duration:  float = 0.0
    mouth_open_duration:   float = 0.0
    detection_confidence:  float = 0.0
    gaze_zone:             str   = "UNKNOWN"


GAZE_ZONE_LABELS = {
    "FORWARD":      ("Nhìn thẳng",           1.00),
    "RIGHT_MIRROR": ("Nhìn gương phải",       0.75),
    "LEFT_MIRROR":  ("Nhìn gương trái",       0.75),
    "LOOKING_DOWN": ("Nhìn xuống",            0.45),
    "LOOKING_UP":   ("Nhìn lên",              0.70),
    "FAR_RIGHT":    ("Mất tập trung (phải)",  0.20),
    "FAR_LEFT":     ("Mất tập trung (trái)",  0.20),
    "EYES_CLOSED":  ("Mắt nhắm / buồn ngủ",  0.00),
    "YAWNING":      ("Đang ngáp",             0.50),
    "UNKNOWN":      ("Không xác định",        0.50),
}


class FaceAnalyzer:
    """
    Phân tích khuôn mặt real-time từ frame OpenCV.
    Dùng MediaPipe FaceLandmarker Tasks API v1.0+.
    """

    EAR_THRESHOLD  = 0.22
    MAR_THRESHOLD  = 0.55
    PERCLOS_WINDOW = 90
    BLINK_WINDOW   = 300

    YAW_MIRROR_THRESHOLD  = 28
    YAW_AWAY_THRESHOLD    = 45
    PITCH_DOWN_THRESHOLD  = -18
    PITCH_UP_THRESHOLD    = 15

    MODEL_POINTS_3D = np.array([
        (0.0,    0.0,    0.0),
        (0.0,   -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0,  170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0,  -150.0, -125.0),
    ], dtype=np.float64)

    def __init__(self, model_path: Optional[str] = None):
        path = model_path or str(_MODEL_PATH)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Không tìm thấy model: {path}\n"
                "Chạy: python -c \"import urllib.request; "
                "urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task', "
                "'models/face_landmarker.task')\""
            )

        base_options = mp_python.BaseOptions(model_asset_path=path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._frame_ms = 0

        self._ear_history   = deque(maxlen=self.PERCLOS_WINDOW)
        self._blink_history = deque(maxlen=self.BLINK_WINDOW)
        self._in_blink      = False
        self._blink_count   = 0
        self._eyes_closed_start: Optional[float] = None
        self._mouth_open_start:  Optional[float] = None
        self._fps = 30.0

    def set_fps(self, fps: float):
        self._fps = max(1.0, fps)

    def analyze(self, frame: np.ndarray) -> Optional[FaceFeatures]:
        """Phân tích 1 frame BGR → FaceFeatures hoặc None."""
        h, w = frame.shape[:2]
        self._frame_ms += int(1000 / self._fps)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

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

        # Iris landmarks có trong FaceLandmarker (478 + iris)
        if len(pts) >= 478:
            feats.gaze_x, feats.gaze_y = self._calc_gaze(pts)

        feats.perclos, feats.blink_rate = self._update_temporal(feats.ear_avg)
        feats.eyes_closed_duration = self._get_eyes_closed_duration(feats.ear_avg)
        feats.mouth_open_duration  = self._get_mouth_open_duration(feats.mar)
        feats.gaze_zone = self._classify_gaze_zone(feats)
        feats.detection_confidence = 1.0

        return feats

    # ─── EAR ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _calc_ear(pts: np.ndarray, eye_idx: list) -> float:
        p1, p2, p3, p4, p5, p6 = [pts[i][:2] for i in eye_idx]
        vertical   = euclidean(p2, p6) + euclidean(p3, p5)
        horizontal = euclidean(p1, p4)
        if horizontal < 1e-6:
            return 0.0
        return vertical / (2.0 * horizontal)

    # ─── MAR ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _calc_mar(pts: np.ndarray) -> float:
        left, right, top, bot = [pts[i][:2] for i in MOUTH_OUTER]
        vertical   = euclidean(top, bot)
        horizontal = euclidean(left, right)
        if horizontal < 1e-6:
            return 0.0
        return vertical / horizontal

    # ─── Head Pose ────────────────────────────────────────────────────────────
    def _calc_head_pose(self, pts: np.ndarray, w: int, h: int):
        img_points = pts[HEAD_POSE_IDX, :2].astype(np.float64)
        focal_len  = w
        cam_matrix = np.array([
            [focal_len, 0, w / 2],
            [0, focal_len, h / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        ok, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS_3D, img_points, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return 0.0, 0.0, 0.0

        rot_mat, _ = cv2.Rodrigues(rvec)
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
    def _calc_gaze(pts: np.ndarray):
        if len(pts) < 478:
            return 0.0, 0.0
        try:
            l_iris = pts[LEFT_IRIS_IDX, :2].mean(axis=0)
            r_iris = pts[RIGHT_IRIS_IDX, :2].mean(axis=0)

            l_eye_pts = pts[LEFT_EYE_IDX, :2]
            l_min = l_eye_pts.min(axis=0); l_max = l_eye_pts.max(axis=0)
            l_size = l_max - l_min

            r_eye_pts = pts[RIGHT_EYE_IDX, :2]
            r_min = r_eye_pts.min(axis=0); r_max = r_eye_pts.max(axis=0)
            r_size = r_max - r_min

            if l_size[0] < 1 or r_size[0] < 1:
                return 0.0, 0.0

            l_gaze = (l_iris - l_min) / (l_size + 1e-6)
            r_gaze = (r_iris - r_min) / (r_size + 1e-6)
            avg    = (l_gaze + r_gaze) / 2.0
            return float(np.clip((avg[0] - 0.5) * 2, -1, 1)), \
                   float(np.clip((avg[1] - 0.5) * 2, -1, 1))
        except Exception:
            return 0.0, 0.0

    # ─── Temporal ────────────────────────────────────────────────────────────
    def _update_temporal(self, ear: float):
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

        perclos    = sum(self._ear_history) / len(self._ear_history) if self._ear_history else 0.0
        blink_rate = (sum(self._blink_history) / (len(self._blink_history) / self._fps) * 60.0
                      if len(self._blink_history) > 10 else 0.0)
        return perclos, blink_rate

    def _get_eyes_closed_duration(self, ear: float) -> float:
        if ear < self.EAR_THRESHOLD:
            if self._eyes_closed_start is None:
                self._eyes_closed_start = 0.0
            self._eyes_closed_start += 1.0 / self._fps
            return self._eyes_closed_start
        self._eyes_closed_start = None
        return 0.0

    def _get_mouth_open_duration(self, mar: float) -> float:
        if mar > self.MAR_THRESHOLD:
            if self._mouth_open_start is None:
                self._mouth_open_start = 0.0
            self._mouth_open_start += 1.0 / self._fps
            return self._mouth_open_start
        self._mouth_open_start = None
        return 0.0

    # ─── Gaze Zone ────────────────────────────────────────────────────────────
    def _classify_gaze_zone(self, f: FaceFeatures) -> str:
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
        self._ear_history.clear()
        self._blink_history.clear()
        self._in_blink = False
        self._blink_count = 0
        self._eyes_closed_start = None
        self._mouth_open_start  = None
        self._frame_ms = 0

    def close(self):
        self.landmarker.close()
