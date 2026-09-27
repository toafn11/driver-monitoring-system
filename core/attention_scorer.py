"""
core/attention_scorer.py
========================
Computes the Attention Score (0-100%) from FaceFeatures.
Combines multiple signals via weighted average + temporal smoothing.
"""

import numpy as np
from collections import deque
from core.face_analyzer import FaceFeatures


GAZE_ZONE_LABELS = {
    "FORWARD":      ("Looking straight ahead",     1.00),
    "RIGHT_MIRROR": ("Checking right mirror",       0.75),
    "LEFT_MIRROR":  ("Checking left mirror",        0.75),
    "LOOKING_DOWN": ("Looking down",                0.45),
    "LOOKING_UP":   ("Looking up",                  0.70),
    "FAR_RIGHT":    ("Distracted (right)",          0.20),
    "FAR_LEFT":     ("Distracted (left)",           0.20),
    "EYES_CLOSED":  ("Eyes closed / drowsy",        0.00),
    "YAWNING":      ("Yawning",                     0.50),
    "UNKNOWN":      ("Unknown",                     0.50),
}

# Final aggregated attention state
ATTENTION_STATES = {
    "FOCUSED":      ("FOCUSED",      (0, 200, 80)),
    "MILD":         ("MILDLY DISTRACTED", (50, 180, 255)),
    "DISTRACTED":   ("DISTRACTED",   (0, 120, 255)),
    "DROWSY":       ("DROWSY",       (0, 0, 220)),
    "DANGER":       ("DANGER!",      (0, 0, 180)),
}


class AttentionScorer:
    """
    Computes the Attention Score and aggregated state from FaceFeatures.

    Weights per component (sum = 1.0):
    - EAR score:     30% (whether eyes are open)
    - PERCLOS:       25% (% of time eyes are closed within 3s)
    - Gaze zone:     25% (where the driver is looking)
    - MAR score:     10% (yawning)
    - Head velocity: 10% (abnormal head movement)
    """

    W_EAR      = 0.30
    W_PERCLOS  = 0.25
    W_GAZE     = 0.25
    W_MAR      = 0.10
    W_HEADMOV  = 0.10

    SMOOTH_WINDOW = 15  # Number of frames used to smooth the score

    def __init__(self):
        self._score_history = deque(maxlen=self.SMOOTH_WINDOW)
        self._prev_yaw   = 0.0
        self._prev_pitch = 0.0
        self._frame_count = 0

    def update(self, feats: FaceFeatures) -> tuple[float, str, str]:
        """
        Returns: (score 0-100, state_key, state_label)
        """
        self._frame_count += 1

        # 1. EAR score: 0 if fully closed, 1 if normally open
        ear_score = np.clip((feats.ear_avg - 0.10) / (0.32 - 0.10), 0.0, 1.0)

        # 2. PERCLOS score: 0 if eyes closed 100% of the time, 1 if not closed
        perclos_score = 1.0 - np.clip(feats.perclos / 0.35, 0.0, 1.0)

        # 3. Gaze zone score
        gaze_score = GAZE_ZONE_LABELS.get(feats.gaze_zone, ("", 0.5))[1]

        # 4. MAR score: 1 if mouth closed, decreases when yawning
        mar_score = 1.0 - np.clip((feats.mar - 0.3) / (0.7 - 0.3), 0.0, 1.0)

        # 5. Head movement score: penalize if head is rotating rapidly
        d_yaw   = abs(feats.yaw   - self._prev_yaw)
        d_pitch = abs(feats.pitch - self._prev_pitch)
        head_velocity = (d_yaw + d_pitch)
        head_score = 1.0 - np.clip(head_velocity / 15.0, 0.0, 0.5)
        self._prev_yaw   = feats.yaw
        self._prev_pitch = feats.pitch

        # Aggregate
        raw_score = (
            self.W_EAR     * ear_score +
            self.W_PERCLOS * perclos_score +
            self.W_GAZE    * gaze_score +
            self.W_MAR     * mar_score +
            self.W_HEADMOV * head_score
        )
        raw_score = float(np.clip(raw_score, 0.0, 1.0))
        self._score_history.append(raw_score)

        # Exponential smoothing (prioritizes recent frames)
        weights = np.exp(np.linspace(-1, 0, len(self._score_history)))
        weights /= weights.sum()
        smooth_score = float(np.dot(list(self._score_history), weights) * 100)

        # Determine state
        state = self._classify_state(smooth_score, feats)
        label = ATTENTION_STATES[state][0]
        return round(smooth_score, 1), state, label

    def _classify_state(self, score: float, feats: FaceFeatures) -> str:
        # Immediate override if there are clear warning signs
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
        """Returns individual component scores for debugging/visualization."""
        return {
            "EAR":      round(np.clip((feats.ear_avg - 0.10) / 0.22, 0, 1) * 100, 1),
            "PERCLOS":  round((1 - np.clip(feats.perclos / 0.35, 0, 1)) * 100, 1),
            "Gaze":     round(GAZE_ZONE_LABELS.get(feats.gaze_zone, ("", 0.5))[1] * 100, 1),
            "MAR":      round((1 - np.clip((feats.mar - 0.3) / 0.4, 0, 1)) * 100, 1),
        }
