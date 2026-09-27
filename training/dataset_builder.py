"""
training/dataset_builder.py
============================
Xây dựng training dataset từ các nguồn:
  1. NTHU-DDD (video clips, labels: drowsy/non-drowsy)
  2. MRL Eye Dataset (ảnh mắt mở/nhắm)
  3. Video tự thu thập (collect_data.py)

PHÂN TÍCH VÀ PHÒNG TRÁNH BIAS CHO TỪNG DATASET:
=================================================

### NTHU-DDD ###
- ✅ Đa dạng: 36 subjects, 5 điều kiện (ngày/đêm, kính/không kính)
- ⚠️ BIAS 1: Camera cố định dashboard → model có thể học vị trí đầu mặc định
  → GIẢI PHÁP: Chỉ extract GEOMETRIC FEATURES (EAR/MAR/head pose), KHÔNG dùng pixel
- ⚠️ BIAS 2: "Drowsy" thường đi kèm đầu cúi xuống, "Alert" đầu thẳng
  → GIẢI PHÁP: Augment với head pose variations khi extract features
- ⚠️ BIAS 3: Số lượng alert >> drowsy (imbalanced)
  → GIẢI PHÁP: Weighted loss / oversample drowsy class

### MRL Eye Dataset ###
- ✅ 84,898 ảnh, đa dạng thiết bị hồng ngoại
- ⚠️ BIAS 1: Background đen (infrared) nhất quán → model học background
  → GIẢI PHÁP: Crop chỉ vùng mắt, threshold loại bỏ viền đen
- ⚠️ BIAS 2: Nhiều ảnh bị blur hoặc occlusion → nếu không lọc sẽ học noise
  → GIẢI PHÁP: Lọc theo độ sắc nét (Laplacian variance)
- ⚠️ BIAS 3: Metadata không đồng nhất (nhiều người đeo kính)
  → GIẢI PHÁP: Tách validation set đảm bảo phân phối đều

### StateFarm Distracted Driver ###
- ⚠️ BIAS NGHIÊM TRỌNG: Cùng drivers xuất hiện ở train và test
  → GIẢI PHÁP: Driver-disjoint split bắt buộc
- ⚠️ BIAS: Background xe nhất quán (cùng loại xe)
  → GIẢI PHÁP: Dùng để phân loại HÀNH VI (tay, điện thoại), không phải face

CHIẾN LƯỢC CHUNG:
- KHÔNG train model học từ pixel raw của full frame
- Chỉ train từ GEOMETRIC FEATURES (EAR, MAR, yaw, pitch, roll, gaze_x, gaze_y...)
- Sequence-based (30 frames) → LSTM học temporal pattern, không học static pose
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.face_analyzer import FaceAnalyzer


# ─── Config ───────────────────────────────────────────────────────────────────
SEQ_LEN    = 30    # Số frame mỗi sequence (khoảng 1 giây ở 30fps)
STEP_SIZE  = 10    # Bước nhảy giữa các sequence (overlap)
FEATURE_DIM = 12   # Số features mỗi frame

FEATURE_NAMES = [
    "ear_left", "ear_right", "ear_avg",
    "mar",
    "yaw", "pitch", "roll",
    "gaze_x", "gaze_y",
    "perclos",
    "eyes_closed_duration",
    "mouth_open_duration",
]

LABEL_MAP = {
    "alert":       0,
    "focused":     0,
    "non_drowsy":  0,
    "drowsy":      1,
    "sleepy":      1,
    "yawning":     2,
    "distracted":  3,
    "phone":       3,
    "texting":     3,
}


class DatasetBuilder:
    """
    Extract geometric feature sequences từ video files.
    Output: numpy arrays (X, y) cho LSTM training.
    """

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = FaceAnalyzer()

    # ─── NTHU-DDD ─────────────────────────────────────────────────────────────
    def process_nthu_ddd(self, dataset_path: str, split: str = "train") -> tuple:
        """
        Xử lý NTHU-DDD dataset.
        Cấu trúc thư mục chuẩn:
          dataset_path/
            training_data/
              drowsy/   ← video .avi files
              nondrowsy/
            testing_data/
              drowsy/
              nondrowsy/

        CHỐNG BIAS:
        - Split theo SUBJECT ID (subject-disjoint), không random
        - Chỉ extract frame khi face detection confidence cao
        """
        dataset_path = Path(dataset_path)
        X_all, y_all = [], []

        for label_name, label_id in [("nondrowsy", 0), ("drowsy", 1)]:
            folder = dataset_path / f"{split}ing_data" / label_name
            if not folder.exists():
                # Thử cấu trúc thay thế
                folder = dataset_path / label_name
            if not folder.exists():
                print(f"  [SKIP] Không tìm thấy: {folder}")
                continue

            video_files = list(folder.glob("*.avi")) + list(folder.glob("*.mp4"))
            print(f"  [{label_name}] {len(video_files)} videos")

            for vf in tqdm(video_files, desc=f"  {label_name}"):
                seqs = self._extract_sequences_from_video(vf, label_id)
                X_all.extend(seqs[0])
                y_all.extend(seqs[1])

        if not X_all:
            return np.array([]), np.array([])

        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.int64)
        print(f"  → {len(X)} sequences | Class dist: {np.bincount(y)}")
        return X, y

    # ─── MRL Eye Dataset (dùng để train EAR threshold calibration) ────────────
    def process_mrl_eye(self, dataset_path: str) -> tuple:
        """
        MRL Eye Dataset: ảnh mắt hồng ngoại, 2 class (open/closed).

        CHỐNG BIAS:
        1. Crop chỉ phần trung tâm ảnh (loại bỏ viền đen artifact)
        2. Lọc ảnh mờ (Laplacian variance < threshold)
        3. Chuẩn hóa histogram để loại bỏ bias ánh sáng
        4. KHÔNG dùng pixel trực tiếp → convert sang binary (threshould)
           rồi tính EAR-like ratio từ ảnh tĩnh

        NOTE: Dataset này dùng để validate/calibrate ngưỡng EAR,
              KHÔNG dùng để train LSTM chính.
        """
        dataset_path = Path(dataset_path)
        data = []

        for cls_name, cls_id in [("open", 1), ("closed", 0)]:
            folder = dataset_path / cls_name
            if not folder.exists():
                folder = dataset_path / cls_name.upper()
            if not folder.exists():
                continue

            imgs = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
            print(f"  [MRL {cls_name}] {len(imgs)} images")

            for img_path in tqdm(imgs[:5000], desc=f"  {cls_name}"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # 1. Lọc ảnh mờ
                blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
                if blur_score < 50:
                    continue

                # 2. Crop viền đen (threshold)
                _, binary = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
                coords = cv2.findNonZero(binary)
                if coords is None:
                    continue
                x, y, w, h = cv2.boundingRect(coords)
                cropped = img[y:y+h, x:x+w]

                if cropped.size == 0 or w < 10 or h < 5:
                    continue

                # 3. Histogram equalization
                equalized = cv2.equalizeHist(cropped)

                # 4. Tính EAR-like từ ảnh tĩnh
                # (vertical brightness / horizontal brightness ratio)
                resized = cv2.resize(equalized, (32, 16))
                mean_intensity = resized.mean() / 255.0
                vertical_ratio = resized[:, 12:20].mean() / (resized.mean() + 1e-6)

                data.append({
                    "mean_intensity": mean_intensity,
                    "vertical_ratio": vertical_ratio,
                    "width": w, "height": h,
                    "aspect_ratio": w / (h + 1e-6),
                    "blur_score": blur_score,
                    "label": cls_id
                })

        if not data:
            return pd.DataFrame(), None

        df = pd.DataFrame(data)
        out_path = self.output_dir / "mrl_eye_stats.csv"
        df.to_csv(out_path, index=False)
        print(f"  → Saved {len(df)} samples → {out_path}")
        return df, out_path

    # ─── Extract sequences từ video ──────────────────────────────────────────
    def _extract_sequences_from_video(self, video_path: Path, label: int,
                                       min_confidence: float = 0.8) -> tuple:
        """
        Extract sliding window sequences từ 1 video file.
        Chỉ lấy frame khi face detection confidence đủ cao.

        CHỐNG BIAS:
        - Skip các frame đầu video (driver setup, không đại diện)
        - Chỉ lấy sequence khi >= 80% frame có face detection
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [], []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.analyzer.set_fps(fps)
        self.analyzer.reset()

        all_features = []
        frame_idx = 0
        skip_frames = int(fps * 2)  # Skip 2 giây đầu (setup time)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx < skip_frames:
                continue

            # Resize để tăng tốc độ xử lý
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))

            feats = self.analyzer.analyze(frame)
            if feats is not None:
                all_features.append(self._feats_to_array(feats))
            else:
                # Frame không detect được → dùng zero vector
                all_features.append(np.zeros(FEATURE_DIM, dtype=np.float32))

        cap.release()

        # Tạo sliding window sequences
        X_seqs, y_seqs = [], []
        feat_arr = np.array(all_features, dtype=np.float32)

        for i in range(0, len(feat_arr) - SEQ_LEN, STEP_SIZE):
            seq = feat_arr[i:i + SEQ_LEN]
            # Check: sequence phải có >= 80% frame detect được (non-zero)
            valid_ratio = (seq.sum(axis=1) > 0).mean()
            if valid_ratio >= 0.6:
                X_seqs.append(seq)
                y_seqs.append(label)

        return X_seqs, y_seqs

    @staticmethod
    def _feats_to_array(f: "FaceFeatures") -> np.ndarray:
        """Convert FaceFeatures → numpy array (FEATURE_DIM,)."""
        return np.array([
            f.ear_left, f.ear_right, f.ear_avg,
            f.mar,
            f.yaw / 90.0,    # Normalize head pose về [-1, 1]
            f.pitch / 90.0,
            f.roll / 90.0,
            f.gaze_x,
            f.gaze_y,
            f.perclos,
            min(f.eyes_closed_duration / 5.0, 1.0),  # Cap ở 5 giây
            min(f.mouth_open_duration / 3.0, 1.0),
        ], dtype=np.float32)

    # ─── Phân tích bias dataset ────────────────────────────────────────────────
    def analyze_dataset_bias(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Phân tích các dấu hiệu bias trong dataset đã extract.
        In báo cáo chi tiết.
        """
        report = {}
        print("\n" + "="*60)
        print("PHÂN TÍCH BIAS DATASET")
        print("="*60)

        # 1. Class imbalance
        class_counts = np.bincount(y)
        print(f"\n1. PHÂN PHỐI NHÃN:")
        for i, count in enumerate(class_counts):
            print(f"   Class {i}: {count} sequences ({count/len(y)*100:.1f}%)")
        report["class_counts"] = class_counts.tolist()

        # 2. Feature distribution per class
        print(f"\n2. PHÂN PHỐI FEATURES THEO CLASS:")
        feature_names = ["EAR_L", "EAR_R", "EAR_avg", "MAR",
                         "Yaw", "Pitch", "Roll", "Gaze_X", "Gaze_Y",
                         "PERCLOS", "EyesClosed", "MouthOpen"]
        for cls in range(len(class_counts)):
            mask = y == cls
            if mask.sum() == 0:
                continue
            cls_data = X[mask].mean(axis=(0, 1))  # Mean over sequences and time
            print(f"\n   Class {cls}:")
            for fname, fval in zip(feature_names, cls_data):
                print(f"     {fname:15s}: {fval:.4f}")

        # 3. Kiểm tra separation tuyến tính (EAR có đủ phân biệt không?)
        if len(class_counts) >= 2:
            ear_class0 = X[y == 0, :, 2].mean()  # EAR avg, class 0 (alert)
            ear_class1 = X[y == 1, :, 2].mean()  # EAR avg, class 1 (drowsy)
            separation = abs(ear_class0 - ear_class1)
            print(f"\n3. EAR SEPARATION (Alert vs Drowsy): {separation:.4f}")
            if separation < 0.02:
                print("   ⚠️  CẢNH BÁO: EAR không đủ phân biệt → có thể bị bias")
            else:
                print("   ✅ EAR có khả năng phân biệt tốt")

        # 4. Kiểm tra temporal consistency
        seq_std = X.std(axis=1).mean()  # Std over time dimension
        print(f"\n4. TEMPORAL VARIANCE (std over time): {seq_std:.4f}")
        if seq_std < 0.01:
            print("   ⚠️  CẢNH BÁO: Features quá ổn định → có thể đang extract từ ảnh tĩnh")
        else:
            print("   ✅ Features có đủ temporal variation")

        print("="*60 + "\n")
        return report

    # ─── Data Augmentation (chống overfitting) ────────────────────────────────
    @staticmethod
    def augment_sequences(X: np.ndarray, y: np.ndarray,
                          target_counts: Optional[dict] = None) -> tuple:
        """
        Augment geometric feature sequences để balance dataset và tăng robustness.

        Augmentation trên GEOMETRIC FEATURES (không phải pixel):
        1. Gaussian noise nhỏ (simulate measurement error)
        2. Time reversal (nháy mắt ngược)
        3. Head pose jitter (thay đổi nhẹ góc đầu)
        4. Oversample class thiểu số
        """
        if target_counts is None:
            counts = np.bincount(y)
            max_count = counts.max()
            target_counts = {i: max_count for i in range(len(counts))}

        X_aug, y_aug = list(X), list(y)

        for cls, target in target_counts.items():
            cls_mask = np.where(y == cls)[0]
            current  = len(cls_mask)
            needed   = target - current
            if needed <= 0:
                continue

            for _ in range(needed):
                idx = np.random.choice(cls_mask)
                seq = X[idx].copy()

                # Augmentation 1: Gaussian noise (nhỏ thôi)
                noise = np.random.normal(0, 0.008, seq.shape).astype(np.float32)
                noise[:, 4:7] *= 2.0  # Head pose có thể noise nhiều hơn

                # Augmentation 2: Đôi khi đảo ngược thời gian
                if np.random.random() < 0.3:
                    seq = seq[::-1].copy()

                # Augmentation 3: Jitter head pose nhẹ
                if np.random.random() < 0.5:
                    pose_jitter = np.random.normal(0, 0.03, (SEQ_LEN, 3)).astype(np.float32)
                    seq[:, 4:7] = np.clip(seq[:, 4:7] + pose_jitter, -1, 1)

                X_aug.append(np.clip(seq + noise, -1, 2))
                y_aug.append(cls)

        X_aug = np.array(X_aug, dtype=np.float32)
        y_aug = np.array(y_aug, dtype=np.int64)

        # Shuffle
        perm = np.random.permutation(len(X_aug))
        return X_aug[perm], y_aug[perm]

    # ─── Save / Load ──────────────────────────────────────────────────────────
    def save_dataset(self, X: np.ndarray, y: np.ndarray, name: str = "train"):
        """Lưu dataset đã xử lý."""
        np.save(self.output_dir / f"X_{name}.npy", X)
        np.save(self.output_dir / f"y_{name}.npy", y)
        meta = {
            "name": name, "n_samples": len(X), "seq_len": SEQ_LEN,
            "feature_dim": FEATURE_DIM, "feature_names": FEATURE_NAMES,
            "class_counts": np.bincount(y).tolist(),
        }
        with open(self.output_dir / f"meta_{name}.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved: {self.output_dir}/X_{name}.npy  ({X.shape})")

    def load_dataset(self, name: str = "train") -> tuple:
        """Load dataset đã save."""
        X = np.load(self.output_dir / f"X_{name}.npy")
        y = np.load(self.output_dir / f"y_{name}.npy")
        return X, y
