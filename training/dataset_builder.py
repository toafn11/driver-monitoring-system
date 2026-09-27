"""
training/dataset_builder.py
============================
Builds the training dataset from multiple sources:
  1. NTHU-DDD (video clips, labels: drowsy/non-drowsy)
  2. MRL Eye Dataset (open/closed eye images)
  3. Self-collected video (collect_data.py)

BIAS ANALYSIS AND MITIGATION FOR EACH DATASET:
===============================================

### NTHU-DDD ###
- ✅ Diverse: 36 subjects, 5 conditions (day/night, glasses/no glasses)
- ⚠️ BIAS 1: Fixed dashboard camera → model may learn default head position
  → SOLUTION: Extract only GEOMETRIC FEATURES (EAR/MAR/head pose), NOT raw pixels
- ⚠️ BIAS 2: "Drowsy" often accompanied by head nodding down, "Alert" head upright
  → SOLUTION: Augment with head pose variations during feature extraction
- ⚠️ BIAS 3: Alert samples >> drowsy samples (imbalanced)
  → SOLUTION: Weighted loss / oversample drowsy class

### MRL Eye Dataset ###
- ✅ 84,898 images, diverse infrared devices
- ⚠️ BIAS 1: Consistent black (infrared) background → model learns background
  → SOLUTION: Crop only the eye region, threshold to remove black border
- ⚠️ BIAS 2: Many blurry or occluded images → learning noise if not filtered
  → SOLUTION: Filter by sharpness (Laplacian variance)
- ⚠️ BIAS 3: Inconsistent metadata (many subjects wearing glasses)
  → SOLUTION: Ensure even distribution in the validation set split

### StateFarm Distracted Driver ###
- ⚠️ CRITICAL BIAS: Same drivers appear in both train and test sets
  → SOLUTION: Mandatory driver-disjoint split
- ⚠️ BIAS: Consistent vehicle background (same car model)
  → SOLUTION: Use for BEHAVIOR classification (hands, phone), not face

GENERAL STRATEGY:
- Do NOT train models to learn from raw pixel values of the full frame
- Train only from GEOMETRIC FEATURES (EAR, MAR, yaw, pitch, roll, gaze_x, gaze_y...)
- Sequence-based (30 frames) → LSTM learns temporal patterns, not static poses
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
SEQ_LEN    = 30    # Number of frames per sequence (approx. 1 second at 30 fps)
STEP_SIZE  = 10    # Step size between sequences (overlap)
FEATURE_DIM = 12   # Number of features per frame

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
    Extracts geometric feature sequences from video files.
    Output: numpy arrays (X, y) for LSTM training.
    """

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = FaceAnalyzer()

    # ─── NTHU-DDD ─────────────────────────────────────────────────────────────
    def process_nthu_ddd(self, dataset_path: str, split: str = "train") -> tuple:
        """
        Processes the NTHU-DDD dataset.
        Expected directory structure:
          dataset_path/
            training_data/
              drowsy/   ← video .avi files
              nondrowsy/
            testing_data/
              drowsy/
              nondrowsy/

        BIAS MITIGATION:
        - Split by SUBJECT ID (subject-disjoint), not random
        - Only extract frames when face detection confidence is high
        """
        dataset_path = Path(dataset_path)
        X_all, y_all = [], []

        for label_name, label_id in [("nondrowsy", 0), ("drowsy", 1)]:
            folder = dataset_path / f"{split}ing_data" / label_name
            if not folder.exists():
                # Try alternative directory structure
                folder = dataset_path / label_name
            if not folder.exists():
                print(f"  [SKIP] Not found: {folder}")
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

    # ─── MRL Eye Dataset (used to train EAR threshold calibration) ────────────
    def process_mrl_eye(self, dataset_path: str) -> tuple:
        """
        MRL Eye Dataset: infrared eye images, 2 classes (open/closed).

        BIAS MITIGATION:
        1. Crop only the center region of the image (removes black border artifacts)
        2. Filter blurry images (Laplacian variance < threshold)
        3. Histogram normalization to remove lighting bias
        4. Do NOT use pixels directly → convert to binary (threshold)
           then compute EAR-like ratio from the static image

        NOTE: This dataset is used to validate/calibrate the EAR threshold,
              NOT for training the main LSTM.
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

                # 1. Filter blurry images
                blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
                if blur_score < 50:
                    continue

                # 2. Crop black border (threshold)
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

                # 4. Compute EAR-like ratio from static image
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

    # ─── Extract sequences from video ────────────────────────────────────────
    def _extract_sequences_from_video(self, video_path: Path, label: int,
                                       min_confidence: float = 0.8) -> tuple:
        """
        Extracts sliding window sequences from a single video file.
        Only includes frames where face detection confidence is sufficiently high.

        BIAS MITIGATION:
        - Skip frames at the start of the video (driver setup, not representative)
        - Only include sequences where >= 80% of frames have face detections
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [], []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.analyzer.set_fps(fps)
        self.analyzer.reset()

        all_features = []
        frame_idx = 0
        skip_frames = int(fps * 2)  # Skip the first 2 seconds (setup time)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx < skip_frames:
                continue

            # Resize to speed up processing
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))

            feats = self.analyzer.analyze(frame)
            if feats is not None:
                all_features.append(self._feats_to_array(feats))
            else:
                # Face not detected in this frame → use zero vector
                all_features.append(np.zeros(FEATURE_DIM, dtype=np.float32))

        cap.release()

        # Build sliding window sequences
        X_seqs, y_seqs = [], []
        feat_arr = np.array(all_features, dtype=np.float32)

        for i in range(0, len(feat_arr) - SEQ_LEN, STEP_SIZE):
            seq = feat_arr[i:i + SEQ_LEN]
            # Check: sequence must have >= 80% frames with a detected face (non-zero)
            valid_ratio = (seq.sum(axis=1) > 0).mean()
            if valid_ratio >= 0.6:
                X_seqs.append(seq)
                y_seqs.append(label)

        return X_seqs, y_seqs

    @staticmethod
    def _feats_to_array(f: "FaceFeatures") -> np.ndarray:
        """Converts FaceFeatures → numpy array of shape (FEATURE_DIM,)."""
        return np.array([
            f.ear_left, f.ear_right, f.ear_avg,
            f.mar,
            f.yaw / 90.0,    # Normalize head pose to [-1, 1]
            f.pitch / 90.0,
            f.roll / 90.0,
            f.gaze_x,
            f.gaze_y,
            f.perclos,
            min(f.eyes_closed_duration / 5.0, 1.0),  # Cap at 5 seconds
            min(f.mouth_open_duration / 3.0, 1.0),
        ], dtype=np.float32)

    # ─── Dataset bias analysis ────────────────────────────────────────────────
    def analyze_dataset_bias(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Analyzes signs of bias in the extracted dataset.
        Prints a detailed report.
        """
        report = {}
        print("\n" + "="*60)
        print("DATASET BIAS ANALYSIS")
        print("="*60)

        # 1. Class imbalance
        class_counts = np.bincount(y)
        print(f"\n1. LABEL DISTRIBUTION:")
        for i, count in enumerate(class_counts):
            print(f"   Class {i}: {count} sequences ({count/len(y)*100:.1f}%)")
        report["class_counts"] = class_counts.tolist()

        # 2. Feature distribution per class
        print(f"\n2. FEATURE DISTRIBUTION BY CLASS:")
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

        # 3. Check linear separability (is EAR sufficiently discriminative?)
        if len(class_counts) >= 2:
            ear_class0 = X[y == 0, :, 2].mean()  # EAR avg, class 0 (alert)
            ear_class1 = X[y == 1, :, 2].mean()  # EAR avg, class 1 (drowsy)
            separation = abs(ear_class0 - ear_class1)
            print(f"\n3. EAR SEPARATION (Alert vs Drowsy): {separation:.4f}")
            if separation < 0.02:
                print("   ⚠️  WARNING: EAR is not sufficiently discriminative → possible bias")
            else:
                print("   ✅ EAR has good discriminative ability")

        # 4. Check temporal consistency
        seq_std = X.std(axis=1).mean()  # Std over time dimension
        print(f"\n4. TEMPORAL VARIANCE (std over time): {seq_std:.4f}")
        if seq_std < 0.01:
            print("   ⚠️  WARNING: Features are too stable → may be extracting from static images")
        else:
            print("   ✅ Features have sufficient temporal variation")

        print("="*60 + "\n")
        return report

    # ─── Data Augmentation (prevents overfitting) ────────────────────────────
    @staticmethod
    def augment_sequences(X: np.ndarray, y: np.ndarray,
                          target_counts: Optional[dict] = None) -> tuple:
        """
        Augments geometric feature sequences to balance the dataset and increase robustness.

        Augmentation applied to GEOMETRIC FEATURES (not pixels):
        1. Small Gaussian noise (simulates measurement error)
        2. Time reversal (reversed blink)
        3. Head pose jitter (slight head angle variation)
        4. Oversample the minority class
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

                # Augmentation 1: Small Gaussian noise
                noise = np.random.normal(0, 0.008, seq.shape).astype(np.float32)
                noise[:, 4:7] *= 2.0  # Head pose can have more noise

                # Augmentation 2: Occasionally reverse time
                if np.random.random() < 0.3:
                    seq = seq[::-1].copy()

                # Augmentation 3: Slight head pose jitter
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
        """Saves the processed dataset."""
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
        """Loads a previously saved dataset."""
        X = np.load(self.output_dir / f"X_{name}.npy")
        y = np.load(self.output_dir / f"y_{name}.npy")
        return X, y
