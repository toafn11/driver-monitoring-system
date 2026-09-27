"""
demo/live_demo.py
==================
Demo real-time từ webcam với overlay trực quan.
Chạy được ngay mà không cần LSTM (rule-based mode).
Nếu có model đã train → kết hợp cả hai.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import time
from collections import deque
from typing import Optional

from core.face_analyzer import FaceAnalyzer, FaceFeatures, GAZE_ZONE_LABELS
from core.attention_scorer import AttentionScorer, ATTENTION_STATES
from models.lstm_classifier import ModelManager, CLASSES

import torch


# ─── Config ───────────────────────────────────────────────────────────────────
WINDOW_NAME  = "Driver Monitoring System - THS2026-77"
FONT         = cv2.FONT_HERSHEY_SIMPLEX
SEQ_LEN      = 30
FEATURE_DIM  = 12

# Màu sắc (BGR)
COLOR_GREEN  = (0, 210, 80)
COLOR_YELLOW = (0, 185, 255)
COLOR_ORANGE = (0, 130, 255)
COLOR_RED    = (0, 50, 220)
COLOR_WHITE  = (255, 255, 255)
COLOR_DARK   = (20, 20, 30)
COLOR_BG     = (15, 17, 25)

STATE_COLORS = {
    "FOCUSED":    COLOR_GREEN,
    "MILD":       COLOR_YELLOW,
    "DISTRACTED": COLOR_ORANGE,
    "DROWSY":     COLOR_RED,
    "DANGER":     (0, 0, 180),
}


class LiveDemo:
    def __init__(self, camera_id: int = 0, model_name: str = "driver_lstm_v1",
                 use_lstm: bool = True, show_debug: bool = True):
        self.analyzer      = FaceAnalyzer()
        self.scorer        = AttentionScorer()
        self.show_debug    = show_debug
        self.use_lstm      = use_lstm

        # LSTM model (optional)
        self.model: Optional[object] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if use_lstm:
            self.model = ModelManager.load(model_name, device=self.device)
            if self.model is None:
                print("[WARN] Không tìm thấy LSTM model → chạy rule-based only")
                self.use_lstm = False

        # Feature buffer cho LSTM
        self.feature_buffer = deque(maxlen=SEQ_LEN)

        # Webcam
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.prev_time   = time.time()

        # Alert state
        self.alert_active = False
        self.alert_start  = None

        # Score history cho trend arrow
        self.score_history = deque(maxlen=10)

        # LSTM result cache
        self.lstm_result: Optional[tuple] = None
        self.lstm_frame_count = 0
        self.LSTM_INTERVAL = 5  # Chạy LSTM mỗi 5 frames

    def run(self):
        print(f"\n{'='*50}")
        print(f"  DRIVER MONITORING SYSTEM")
        print(f"  Mode: {'Hybrid (Rule + LSTM)' if self.use_lstm else 'Rule-Based Only'}")
        print(f"  Device: {self.device}")
        print(f"  Press 'q' to quit | 'r' to reset | 'd' debug toggle")
        print(f"{'='*50}\n")

        if not self.cap.isOpened():
            print("[ERROR] Không mở được camera!")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror effect
            display = frame.copy()
            h, w = display.shape[:2]

            # ── FPS ──────────────────────────────────────────────────────────
            now = time.time()
            dt = now - self.prev_time
            self.prev_time = now
            fps = 1.0 / (dt + 1e-9)
            self.fps_history.append(fps)
            avg_fps = np.mean(self.fps_history)
            self.analyzer.set_fps(avg_fps)

            # ── Phân tích khuôn mặt ──────────────────────────────────────────
            feats = self.analyzer.analyze(frame)

            if feats is None:
                self._draw_no_face(display, w, h)
            else:
                # Attention score (rule-based)
                score, state, state_label = self.scorer.update(feats)
                self.score_history.append(score)

                # LSTM prediction (mỗi LSTM_INTERVAL frames)
                lstm_pred = None
                if self.use_lstm and self.model is not None:
                    self.feature_buffer.append(self._feats_to_array(feats))
                    self.lstm_frame_count += 1
                    if (len(self.feature_buffer) == SEQ_LEN and
                            self.lstm_frame_count % self.LSTM_INTERVAL == 0):
                        lstm_pred = self._run_lstm()

                # Vẽ overlay
                self._draw_main_overlay(display, feats, score, state, state_label,
                                        lstm_pred, avg_fps, w, h)

                # Alert nếu nguy hiểm
                if state in ("DANGER", "DROWSY") and not self.alert_active:
                    self._trigger_alert()

                if state == "FOCUSED":
                    self.alert_active = False

            # Vẽ watermark
            self._draw_watermark(display, w, h)

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.analyzer.reset()
                self.scorer = AttentionScorer()
                self.feature_buffer.clear()
                self.score_history.clear()
                print("[RESET] Đã reset trạng thái")
            elif key == ord('d'):
                self.show_debug = not self.show_debug

        self.cap.release()
        cv2.destroyAllWindows()

    # ─── LSTM Inference ───────────────────────────────────────────────────────
    def _run_lstm(self) -> Optional[tuple]:
        try:
            seq = np.array(list(self.feature_buffer), dtype=np.float32)
            x   = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
            results = self.model.predict(x, confidence_threshold=0.55)
            if results:
                cls_id, cls_name, conf, attn = results[0]
                return cls_name, conf
        except Exception as e:
            pass
        return None

    @staticmethod
    def _feats_to_array(f: FaceFeatures) -> np.ndarray:
        return np.array([
            f.ear_left, f.ear_right, f.ear_avg, f.mar,
            f.yaw / 90.0, f.pitch / 90.0, f.roll / 90.0,
            f.gaze_x, f.gaze_y, f.perclos,
            min(f.eyes_closed_duration / 5.0, 1.0),
            min(f.mouth_open_duration / 3.0, 1.0),
        ], dtype=np.float32)

    # ─── Drawing Functions ────────────────────────────────────────────────────
    def _draw_main_overlay(self, frame, feats: FaceFeatures, score: float,
                            state: str, state_label: str, lstm_pred,
                            fps: float, w: int, h: int):
        state_color = STATE_COLORS.get(state, COLOR_WHITE)

        # ── Panel trên cùng (status bar) ────────────────────────────────────
        self._draw_panel(frame, 0, 0, w, 70, alpha=0.85)

        # Trạng thái lớn
        cv2.putText(frame, state_label, (20, 48), FONT, 1.3,
                    state_color, 3, cv2.LINE_AA)

        # LSTM prediction nếu có
        if lstm_pred:
            lstm_text = f"AI: {lstm_pred[0]}  ({lstm_pred[1]*100:.0f}%)"
            cv2.putText(frame, lstm_text, (w - 300, 40), FONT, 0.6,
                        COLOR_YELLOW, 2, cv2.LINE_AA)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 55), FONT, 0.6,
                    COLOR_WHITE, 1, cv2.LINE_AA)

        # ── Attention Score meter ────────────────────────────────────────────
        self._draw_attention_meter(frame, score, state_color, w, h)

        # ── Info panel bên phải ──────────────────────────────────────────────
        if self.show_debug:
            self._draw_debug_panel(frame, feats, w, h)

        # ── Gaze zone label ─────────────────────────────────────────────────
        zone_vn, _ = GAZE_ZONE_LABELS.get(feats.gaze_zone, ("?", 0.5))
        self._draw_pill(frame, f"  {zone_vn}  ", 20, h - 60, state_color)

        # ── Flash effect khi nguy hiểm ───────────────────────────────────────
        if state == "DANGER":
            alpha = 0.15 * abs(np.sin(time.time() * 4))
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_attention_meter(self, frame, score: float, color: tuple, w: int, h: int):
        """Vẽ thanh điểm tập trung hình vòng cung."""
        cx, cy = w // 2, h - 80
        radius = 55

        # Background arc
        cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, 180,
                    (50, 50, 60), 10, cv2.LINE_AA)

        # Score arc
        angle = int(score * 1.8)  # 0-100 → 0-180 degrees
        cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, angle,
                    color, 10, cv2.LINE_AA)

        # Score text
        text = f"{score:.0f}%"
        size = cv2.getTextSize(text, FONT, 0.8, 2)[0]
        cv2.putText(frame, text, (cx - size[0]//2, cy + 10), FONT, 0.8,
                    color, 2, cv2.LINE_AA)

        # Label
        cv2.putText(frame, "TẬP TRUNG", (cx - 40, cy + 30), FONT, 0.4,
                    (150, 150, 170), 1, cv2.LINE_AA)

        # Trend arrow
        if len(self.score_history) >= 5:
            trend = self.score_history[-1] - self.score_history[-5]
            arrow = "▲" if trend > 2 else ("▼" if trend < -2 else "─")
            arr_color = COLOR_GREEN if trend > 2 else (COLOR_RED if trend < -2 else COLOR_WHITE)
            cv2.putText(frame, arrow, (cx + radius + 10, cy + 5), FONT, 0.8,
                        arr_color, 2, cv2.LINE_AA)

    def _draw_debug_panel(self, frame, feats: FaceFeatures, w: int, h: int):
        """Panel debug bên phải màn hình."""
        px, py = w - 230, 90
        self._draw_panel(frame, px - 10, py - 10, w, py + 200, alpha=0.7)

        items = [
            ("EAR",      f"{feats.ear_avg:.3f}",             feats.ear_avg > 0.22),
            ("MAR",      f"{feats.mar:.3f}",                  feats.mar < 0.55),
            ("Yaw",      f"{feats.yaw:+.1f}°",               abs(feats.yaw) < 28),
            ("Pitch",    f"{feats.pitch:+.1f}°",             feats.pitch > -18),
            ("PERCLOS",  f"{feats.perclos*100:.1f}%",        feats.perclos < 0.20),
            ("Gaze X",   f"{feats.gaze_x:+.2f}",            abs(feats.gaze_x) < 0.4),
            ("Blink/min",f"{feats.blink_rate:.1f}",          10 < feats.blink_rate < 25),
        ]
        cv2.putText(frame, "DEBUG", (px, py), FONT, 0.5, (150,150,200), 1)
        for i, (name, val, ok) in enumerate(items):
            y = py + 22 + i * 24
            col = COLOR_GREEN if ok else COLOR_ORANGE
            cv2.putText(frame, f"{name}:", (px, y), FONT, 0.42, (180,180,190), 1)
            cv2.putText(frame, val, (px + 95, y), FONT, 0.42, col, 1)

    @staticmethod
    def _draw_panel(frame, x1: int, y1: int, x2: int, y2: int, alpha: float = 0.8):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_DARK, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    @staticmethod
    def _draw_pill(frame, text: str, x: int, y: int, color: tuple):
        size, baseline = cv2.getTextSize(text, FONT, 0.55, 1)
        pad = 8
        cv2.rectangle(frame, (x - pad, y - size[1] - pad),
                      (x + size[0] + pad, y + pad), color, -1, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), FONT, 0.55, (10, 10, 10), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_no_face(frame, w: int, h: int):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), COLOR_DARK, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, "Không phát hiện khuôn mặt...", (20, 45),
                    FONT, 1.0, COLOR_ORANGE, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_watermark(frame, w: int, h: int):
        cv2.putText(frame, "THS2026-77 | ĐH Cần Thơ", (w - 260, h - 10),
                    FONT, 0.35, (80, 80, 100), 1, cv2.LINE_AA)

    def _trigger_alert(self):
        self.alert_active = True
        self.alert_start  = time.time()
        print(f"[ALERT] {time.strftime('%H:%M:%S')} - PHÁT HIỆN TRẠNG THÁI NGUY HIỂM!")
        # TODO: pygame.mixer.Sound("assets/alert.wav").play()


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Driver Monitoring Live Demo")
    parser.add_argument("--camera",  type=int, default=0, help="Camera ID")
    parser.add_argument("--model",   type=str, default="driver_lstm_v1")
    parser.add_argument("--no-lstm", action="store_true", help="Chỉ rule-based")
    parser.add_argument("--no-debug",action="store_true", help="Ẩn debug panel")
    args = parser.parse_args()

    demo = LiveDemo(
        camera_id=args.camera,
        model_name=args.model,
        use_lstm=not args.no_lstm,
        show_debug=not args.no_debug,
    )
    demo.run()
