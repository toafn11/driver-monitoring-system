"""
collect_data.py
================
Tool thu thập dữ liệu tự (custom dataset).
Người ngồi trước camera → bấm phím để gán nhãn → lưu features.

Dùng để:
1. Thu thập data người Việt Nam (phù hợp đặc điểm nhân trắc học)
2. Calibrate ngưỡng EAR/MAR cá nhân
3. Thu thập negative samples (giả lập nhìn gương, điện thoại)

CHỐNG HỌC VẸT:
- Chỉ lưu GEOMETRIC FEATURES, không lưu pixel ảnh
- Hướng dẫn thực hiện đa dạng điều kiện (ánh sáng, góc máy)
- Thu thập từ nhiều người khác nhau
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import time
import json
from collections import deque
from core.face_analyzer import FaceAnalyzer
from training.dataset_builder import DatasetBuilder, SEQ_LEN, FEATURE_DIM


FONT = cv2.FONT_HERSHEY_SIMPLEX

COLLECTION_CLASSES = {
    ord('0'): ("ALERT",       0, "Nhìn thẳng, mắt mở, tập trung"),
    ord('1'): ("DROWSY",      1, "Giả vờ buồn ngủ, chớp mắt chậm"),
    ord('2'): ("YAWNING",     2, "Ngáp (tự nhiên hoặc giả vờ)"),
    ord('3'): ("DISTRACTED",  3, "Quay đầu sang trái/phải > 30°"),
    ord('4'): ("LEFT_MIRROR", 4, "Nhìn gương chiếu hậu trái (~25°)"),
    ord('5'): ("RIGHT_MIRROR",5, "Nhìn gương chiếu hậu phải (~25°)"),
    ord('6'): ("LOOKING_DOWN",6, "Cúi đầu nhìn xuống"),
}

INSTRUCTIONS = {
    "ALERT":        "→ Nhìn thẳng vào camera, giữ tự nhiên",
    "DROWSY":       "→ Chớp mắt chậm 1-2 giây/lần, mắt lim dim",
    "YAWNING":      "→ Ngáp thật sự hoặc mở miệng rộng 2-3 giây",
    "DISTRACTED":   "→ Quay đầu hẳn sang 1 bên 3-5 giây",
    "LEFT_MIRROR":  "→ Liếc nhìn sang trái ~25 độ (như nhìn gương)",
    "RIGHT_MIRROR": "→ Liếc nhìn sang phải ~25 độ",
    "LOOKING_DOWN": "→ Cúi đầu xuống (như nhìn điện thoại)",
}

class DataCollector:
    def __init__(self, camera_id: int = 0, subject_id: str = "s001",
                 output_dir: str = "data/custom"):
        self.analyzer   = FaceAnalyzer()
        self.subject_id = subject_id
        self.output_dir = Path(output_dir) / subject_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.collecting = False
        self.current_label = None
        self.current_class_id = None
        self.buffer = []
        self.all_data: list[dict] = []  # {features, label, class_id, timestamp}
        self.fps_history = deque(maxlen=20)
        self.prev_time = time.time()
        self.session_counts = {name: 0 for _, (name, _, _) in COLLECTION_CLASSES.items()}

    def run(self):
        print("\n╔══════════════════════════════════════════╗")
        print("║     CÔNG CỤ THU THẬP DỮ LIỆU TÀI XẾ     ║")
        print("╠══════════════════════════════════════════╣")
        for key, (name, cls_id, desc) in COLLECTION_CLASSES.items():
            print(f"║  [{chr(key)}] {name:<12} - {desc:<22}║")
        print("╠══════════════════════════════════════════╣")
        print("║  [SPACE] Bắt đầu/dừng ghi  [S] Lưu      ║")
        print("║  [Q] Thoát               [C] Xem stats  ║")
        print("╚══════════════════════════════════════════╝\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            display = frame.copy()

            # FPS
            now = time.time()
            self.fps_history.append(1.0 / (now - self.prev_time + 1e-9))
            self.prev_time = now
            avg_fps = np.mean(self.fps_history)
            self.analyzer.set_fps(avg_fps)

            # Phân tích
            feats = self.analyzer.analyze(frame)

            # Thu thập nếu đang record
            if self.collecting and feats is not None:
                arr = DatasetBuilder._feats_to_array(feats)
                self.buffer.append(arr)

                # Lưu sequence khi đủ SEQ_LEN frames
                if len(self.buffer) >= SEQ_LEN:
                    seq = np.array(self.buffer[-SEQ_LEN:], dtype=np.float32)
                    self.all_data.append({
                        "features":  seq.tolist(),
                        "label":     self.current_label,
                        "class_id":  self.current_class_id,
                        "timestamp": time.time(),
                    })
                    self.session_counts[self.current_label] += 1

            # Vẽ UI
            self._draw_ui(display, feats, w, h, avg_fps)

            cv2.imshow("Data Collector - Driver Monitoring", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save()
            elif key == ord('c'):
                self._print_stats()
            elif key == ord(' '):
                self.collecting = not self.collecting
                status = "BẮT ĐẦU" if self.collecting else "DỪNG"
                print(f"[{status}] Thu thập: {self.current_label or 'Chưa chọn label'}")
            elif key in COLLECTION_CLASSES:
                info = COLLECTION_CLASSES[key]
                self.current_label    = info[0]
                self.current_class_id = info[1]
                self.buffer.clear()
                self.collecting = True
                print(f"[LABEL] {self.current_label} | {INSTRUCTIONS.get(self.current_label, '')}")

        self._save()
        self.cap.release()
        cv2.destroyAllWindows()

    def _draw_ui(self, frame, feats, w, h, fps):
        # Header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Status
        status_text = f"[GHI] {self.current_label}" if self.collecting else "[DỪNG] Bấm phím để chọn nhãn"
        status_color = (0, 200, 80) if self.collecting else (150, 150, 170)
        cv2.putText(frame, status_text, (20, 45), FONT, 1.0, status_color, 2)

        # Sample count và instruction
        if self.current_label:
            count_text = f"Sequences: {self.session_counts[self.current_label]}"
            cv2.putText(frame, count_text, (20, 70), FONT, 0.55, (180, 180, 200), 1)
            instr = INSTRUCTIONS.get(self.current_label, "")
            cv2.putText(frame, instr, (w - 500, 45), FONT, 0.5, (200, 200, 100), 1)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 70), FONT, 0.5, (150,150,170), 1)

        # Feature bar nếu detect được
        if feats is not None:
            self._draw_feature_bars(frame, feats, w, h)

            # Pulse effect khi đang thu thập
            if self.collecting:
                pulse = abs(np.sin(time.time() * 4))
                r = int(8 + pulse * 6)
                cv2.circle(frame, (w - 30, 30), r, (0, 200, 80), -1)

    def _draw_feature_bars(self, frame, feats, w, h):
        y_start = h - 150
        items = [
            ("EAR",    feats.ear_avg,  0, 0.4),
            ("MAR",    feats.mar,      0, 0.8),
            ("|Yaw|",  abs(feats.yaw)/90, 0, 1),
        ]
        for i, (name, val, vmin, vmax) in enumerate(items):
            norm = np.clip((val - vmin) / (vmax - vmin + 1e-6), 0, 1)
            bx   = 20 + i * 120
            bar_h = 60
            cv2.rectangle(frame, (bx, y_start), (bx + 80, y_start + bar_h),
                          (40, 40, 50), -1)
            fill = int(norm * bar_h)
            color = (0, 200, 80) if norm < 0.7 else (0, 130, 255)
            cv2.rectangle(frame, (bx, y_start + bar_h - fill),
                          (bx + 80, y_start + bar_h), color, -1)
            cv2.putText(frame, name, (bx + 5, y_start - 5), FONT, 0.4, (150,150,170), 1)
            cv2.putText(frame, f"{val:.3f}", (bx + 5, y_start + bar_h + 15),
                        FONT, 0.38, (180,180,200), 1)

    def _save(self):
        if not self.all_data:
            print("[SAVE] Không có dữ liệu để lưu!")
            return
        out_path = self.output_dir / f"session_{int(time.time())}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "subject_id": self.subject_id,
                "class_names": {str(v[1]): v[0] for _, v in COLLECTION_CLASSES.items()},
                "samples": self.all_data,
            }, f, ensure_ascii=False)
        print(f"[SAVE] Đã lưu {len(self.all_data)} sequences → {out_path}")

    def _print_stats(self):
        print(f"\n[STATS] Subject: {self.subject_id}")
        for name, count in self.session_counts.items():
            bar = "█" * min(count, 30)
            print(f"  {name:<14}: {count:4d}  {bar}")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Thu thập dữ liệu tài xế")
    parser.add_argument("--subject", type=str, default="s001", help="ID người thu thập")
    parser.add_argument("--camera",  type=int, default=0)
    parser.add_argument("--output",  type=str, default="data/custom")
    args = parser.parse_args()

    print(f"\n[INFO] Subject: {args.subject}")
    print(f"[INFO] GỢI Ý: Thực hiện từng nhãn trong các điều kiện sau:")
    print("  - Ánh sáng ban ngày (cửa sổ)")
    print("  - Ánh sáng đèn điện (tối)")
    print("  - Đeo kính và không đeo kính")
    print("  - Ngồi thẳng và ngồi hơi nghiêng")
    print()

    collector = DataCollector(
        camera_id=args.camera,
        subject_id=args.subject,
        output_dir=args.output
    )
    collector.run()
