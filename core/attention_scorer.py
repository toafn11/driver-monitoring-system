"""
core/attention_scorer.py
========================
Tính điểm tập trung (Attention Score 0-100%) từ FaceFeatures.
Kết hợp nhiều signal bằng weighted average + temporal smoothing.
"""

import numpy as np
from collections import deque
from core.face_analyzer import FaceFeatures


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

# Trạng thái tổng hợp cuối
ATTENTION_STATES = {
    "FOCUSED":      ("TẬP TRUNG",        (0, 200, 80)),
    "MILD":         ("PHÂN TÂM NHẸ",    (50, 180, 255)),
    "DISTRACTED":   ("MẤT TẬP TRUNG",  (0, 120, 255)),
    "DROWSY":       ("BUỒN NGỦ",        (0, 0, 220)),
    "DANGER":       ("NGUY HIỂM!",      (0, 0, 180)),
}


class AttentionScorer:
    """
    Tính Attention Score và trạng thái tổng hợp từ các FaceFeatures.

    Weights cho từng thành phần (tổng = 1.0):
    - EAR score:     30% (mắt có mở không)
    - PERCLOS:       25% (% thời gian nhắm mắt trong 3s)
    - Gaze zone:     25% (đang nhìn đâu)
    - MAR score:     10% (ngáp)
    - Head velocity: 10% (đầu có đang chuyển động bất thường không)
    """

    W_EAR      = 0.30
    W_PERCLOS  = 0.25
    W_GAZE     = 0.25
    W_MAR      = 0.10
    W_HEADMOV  = 0.10

    SMOOTH_WINDOW = 15  # frames để smooth score

    def __init__(self):
        self._score_history = deque(maxlen=self.SMOOTH_WINDOW)
        self._prev_yaw   = 0.0
        self._prev_pitch = 0.0
        self._frame_count = 0

    def update(self, feats: FaceFeatures) -> tuple[float, str, str]:
        """
        Trả về: (score 0-100, state_key, state_label_vn)
        """
        self._frame_count += 1

        # 1. EAR score: 0 nếu nhắm hoàn toàn, 1 nếu mở bình thường
        ear_score = np.clip((feats.ear_avg - 0.10) / (0.32 - 0.10), 0.0, 1.0)

        # 2. PERCLOS score: 0 nếu nhắm 100% thời gian, 1 nếu không nhắm
        perclos_score = 1.0 - np.clip(feats.perclos / 0.35, 0.0, 1.0)

        # 3. Gaze zone score
        gaze_score = GAZE_ZONE_LABELS.get(feats.gaze_zone, ("", 0.5))[1]

        # 4. MAR score: 1 nếu miệng đóng, giảm khi ngáp
        mar_score = 1.0 - np.clip((feats.mar - 0.3) / (0.7 - 0.3), 0.0, 1.0)

        # 5. Head movement score: penalize nếu đầu đang xoay nhanh
        d_yaw   = abs(feats.yaw   - self._prev_yaw)
        d_pitch = abs(feats.pitch - self._prev_pitch)
        head_velocity = (d_yaw + d_pitch)
        head_score = 1.0 - np.clip(head_velocity / 15.0, 0.0, 0.5)
        self._prev_yaw   = feats.yaw
        self._prev_pitch = feats.pitch

        # Tổng hợp
        raw_score = (
            self.W_EAR     * ear_score +
            self.W_PERCLOS * perclos_score +
            self.W_GAZE    * gaze_score +
            self.W_MAR     * mar_score +
            self.W_HEADMOV * head_score
        )
        raw_score = float(np.clip(raw_score, 0.0, 1.0))
        self._score_history.append(raw_score)

        # Exponential smoothing (ưu tiên frames gần đây)
        weights = np.exp(np.linspace(-1, 0, len(self._score_history)))
        weights /= weights.sum()
        smooth_score = float(np.dot(list(self._score_history), weights) * 100)

        # Xác định trạng thái
        state = self._classify_state(smooth_score, feats)
        label = ATTENTION_STATES[state][0]
        return round(smooth_score, 1), state, label

    def _classify_state(self, score: float, feats: FaceFeatures) -> str:
        # Override ngay lập tức nếu có dấu hiệu rõ ràng
        if feats.eyes_closed_duration > 2.0 or feats.perclos > 0.4:
            return "DANGER"
        if feats.gaze_zone == "EYES_CLOSED" and feats.eyes_closed_duration > 1.0:
            return "DROWSY"

        if score >= 75:
            return "FOCUSED"
        elif score >= 55:
            return "MILD"
        elif score >= 35:
            return "DISTRACTED"
        elif score >= 15:
            return "DROWSY"
        else:
            return "DANGER"

    def get_component_scores(self, feats: FaceFeatures) -> dict:
        """Trả về điểm từng thành phần để debug/visualization."""
        return {
            "EAR":      round(np.clip((feats.ear_avg - 0.10) / 0.22, 0, 1) * 100, 1),
            "PERCLOS":  round((1 - np.clip(feats.perclos / 0.35, 0, 1)) * 100, 1),
            "Gaze":     round(GAZE_ZONE_LABELS.get(feats.gaze_zone, ("", 0.5))[1] * 100, 1),
            "MAR":      round((1 - np.clip((feats.mar - 0.3) / 0.4, 0, 1)) * 100, 1),
        }
