"""
training/download_pretrained.py
================================
Tải các pre-trained models có sẵn từ HuggingFace để tích hợp.

Pre-trained models sẽ dùng:
1. mosesb/drowsiness-detection-mobileViT-v2  → Eye-level classifier
2. mosesb/drowsiness-detection-yolo-cls      → Full-face drowsiness
3. MediaPipe Face Mesh                        → Landmarks (đã có sẵn qua pip)

Chiến lược tích hợp:
- KHÔNG dùng các model này như "oracle" cuối cùng
- Dùng như "ensemble member": vote cùng với rule-based + LSTM của mình
- Nếu model HF predict ngược với rule-based → weight down
- Mục tiêu: tăng robustness, không phụ thuộc 1 model duy nhất
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from pathlib import Path

MODELS_DIR = Path("models/pretrained")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_yolo_drowsiness():
    """
    Download mosesb/drowsiness-detection-yolo-cls từ HuggingFace.
    Model YOLO phân loại drowsy/non-drowsy từ crop khuôn mặt.
    """
    try:
        from huggingface_hub import hf_hub_download
        print("[HF] Downloading YOLO drowsiness classifier...")
        path = hf_hub_download(
            repo_id="mosesb/drowsiness-detection-yolo-cls",
            filename="best.pt",
            local_dir=str(MODELS_DIR / "yolo_drowsy"),
        )
        print(f"[HF] Saved → {path}")
        return path
    except Exception as e:
        print(f"[HF] Lỗi download YOLO: {e}")
        return None


def download_mobilevit_drowsiness():
    """
    Download mosesb/drowsiness-detection-mobileViT-v2 từ HuggingFace.
    Model MobileViT nhỏ, nhanh, phân loại Drowsy/Non-Drowsy.
    """
    try:
        from huggingface_hub import snapshot_download
        print("[HF] Downloading MobileViT-v2 drowsiness model...")
        path = snapshot_download(
            repo_id="mosesb/drowsiness-detection-mobileViT-v2",
            local_dir=str(MODELS_DIR / "mobilevit_drowsy"),
        )
        print(f"[HF] Saved → {path}")
        return path
    except Exception as e:
        print(f"[HF] Lỗi download MobileViT: {e}")
        return None


class PretrainedEnsemble:
    """
    Kết hợp các pre-trained model HF với rule-based system.

    Ensemble voting:
      - Rule-based score (EAR/MAR/pose)    → weight 0.45
      - LSTM (tự train)                    → weight 0.35
      - HF YOLO/MobileViT (pre-trained)    → weight 0.20

    Tại sao weight HF thấp hơn?
    - Model HF train trên dataset nước ngoài → có thể bias với khuôn mặt
      châu Âu/châu Á khác nhau
    - KHÔNG biết dataset HF có driver-disjoint split hay không
    - Dùng như "tham khảo bổ sung", không phải quyết định chính
    """

    WEIGHTS = {
        "rule_based": 0.45,
        "lstm":       0.35,
        "pretrained": 0.20,
    }

    def __init__(self):
        self.yolo_model = None
        self.vit_model  = None
        self._load_pretrained()

    def _load_pretrained(self):
        """Load YOLO model nếu đã download."""
        yolo_path = MODELS_DIR / "yolo_drowsy" / "best.pt"
        if yolo_path.exists():
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(str(yolo_path))
                print(f"[ENSEMBLE] YOLO loaded ✓")
            except ImportError:
                print("[ENSEMBLE] ultralytics chưa install: pip install ultralytics")
            except Exception as e:
                print(f"[ENSEMBLE] YOLO load error: {e}")

    def predict_pretrained(self, face_crop: np.ndarray) -> float:
        """
        Chạy inference trên crop khuôn mặt.
        Returns: P(drowsy) 0-1

        face_crop: BGR numpy array đã crop quanh khuôn mặt
        """
        if self.yolo_model is None:
            return 0.5  # Neutral nếu không có model

        try:
            results = self.yolo_model(face_crop, verbose=False)
            if results and len(results[0].probs):
                probs = results[0].probs.data.cpu().numpy()
                # Class 0 = drowsy (thường), class 1 = non_drowsy
                # Kiểm tra thứ tự class trong model
                names = results[0].names
                drowsy_idx = next((k for k, v in names.items()
                                   if "drowsy" in v.lower() and "non" not in v.lower()), 0)
                return float(probs[drowsy_idx])
        except Exception as e:
            pass
        return 0.5

    def ensemble_score(self, rule_score: float, lstm_prob_drowsy: float,
                       pretrained_prob: float) -> float:
        """
        Tổng hợp điểm từ 3 nguồn → xác suất buồn ngủ cuối cùng.
        Tất cả đầu vào là P(drowsy) 0-1.
        """
        # Chuyển rule_score (0-100% attention) → P(drowsy)
        p_drowsy_rule = 1.0 - (rule_score / 100.0)

        final = (
            self.WEIGHTS["rule_based"] * p_drowsy_rule +
            self.WEIGHTS["lstm"]       * lstm_prob_drowsy +
            self.WEIGHTS["pretrained"] * pretrained_prob
        )
        return float(np.clip(final, 0, 1))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download pre-trained models")
    parser.add_argument("--yolo", action="store_true", help="Download YOLO model")
    parser.add_argument("--vit",  action="store_true", help="Download MobileViT model")
    parser.add_argument("--all",  action="store_true", help="Download tất cả")
    args = parser.parse_args()

    if args.all or args.yolo:
        download_yolo_drowsiness()
    if args.all or args.vit:
        download_mobilevit_drowsiness()

    if not any([args.all, args.yolo, args.vit]):
        print("Dùng: python training/download_pretrained.py --all")
        print("      python training/download_pretrained.py --yolo")
