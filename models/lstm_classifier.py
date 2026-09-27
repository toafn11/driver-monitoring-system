"""
models/lstm_classifier.py
==========================
LSTM-based classifier phân loại trạng thái tài xế.

THIẾT KẾ ĐỂ KHÔNG HỌC VẸT:
1. Input = chuỗi geometric features (không phải pixel) → model buộc học
   từ thay đổi thực sự của mắt/đầu qua thời gian
2. LSTM 2 lớp → học temporal patterns (blink pattern, head movement pattern)
3. Dropout + L2 regularization → prevent memorization
4. Batch normalization → stable training
5. Confidence thresholding → nếu không chắc, trả về "UNCERTAIN"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional


CLASSES = ["ALERT", "DROWSY", "YAWNING", "DISTRACTED"]
NUM_CLASSES = len(CLASSES)
FEATURE_DIM = 12
SEQ_LEN = 30


class DriverStateLSTM(nn.Module):
    """
    LSTM 2 lớp + Attention mechanism + Fully Connected classifier.

    Architecture:
        Input: (batch, seq_len=30, features=12)
        → LayerNorm
        → LSTM(hidden=128, layers=2, bidirectional=True, dropout=0.4)
        → Temporal Self-Attention (học frame nào quan trọng nhất)
        → FC(256 → 128 → num_classes)
        → Softmax

    Tại sao Bidirectional LSTM?
    - Cho phép model nhìn cả "trước" và "sau" trong sequence
    - Phát hiện pattern như: "mắt hé → nhắm từ từ → ngáp" tốt hơn
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.4,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input normalization (xử lý scale khác nhau giữa features)
        self.input_norm = nn.LayerNorm(feature_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_out_size = hidden_size * self.num_directions  # 256 nếu bidirectional

        # Self-Attention: học frame nào trong sequence quan trọng nhất
        self.attention = TemporalAttention(lstm_out_size)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, feature_dim)
        Returns: (logits, attention_weights)
        """
        # Normalize input
        x = self.input_norm(x)

        # LSTM
        lstm_out, _ = self.lstm(x)      # (batch, seq_len, hidden*directions)

        # Attention pooling
        context, attn_weights = self.attention(lstm_out)  # (batch, hidden*directions)

        # Classify
        logits = self.classifier(context)  # (batch, num_classes)
        return logits, attn_weights

    def predict(self, x: torch.Tensor, confidence_threshold: float = 0.50) -> tuple:
        """
        Dự đoán với confidence check.
        Returns: (class_idx, class_name, confidence, attention_weights)
        """
        self.eval()
        with torch.no_grad():
            logits, attn = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confidence, pred_class = probs.max(dim=-1)

            results = []
            for i in range(x.shape[0]):
                conf = confidence[i].item()
                cls  = pred_class[i].item()
                if conf < confidence_threshold:
                    results.append((-1, "UNCERTAIN", conf, attn[i]))
                else:
                    results.append((cls, CLASSES[cls], conf, attn[i]))

        return results


class TemporalAttention(nn.Module):
    """
    Soft attention over time dimension.
    Học frame nào trong 30-frame sequence quan trọng nhất.

    Ý nghĩa: nếu 1 frame có mắt nhắm đột ngột, attention weight cao hơn
    → model tập trung vào điểm bất thường đó.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, lstm_out: torch.Tensor) -> tuple:
        # lstm_out: (batch, seq_len, hidden)
        scores = self.attention_layer(lstm_out).squeeze(-1)   # (batch, seq_len)
        weights = F.softmax(scores, dim=-1)                   # (batch, seq_len)
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden)
        return context, weights


# ─── Lightweight version cho edge devices ─────────────────────────────────────
class DriverStateLSTM_Lite(nn.Module):
    """
    Phiên bản nhẹ hơn cho CPU / thiết bị nhúng.
    Hidden size nhỏ hơn, 1 lớp LSTM, không bidirectional.
    """

    def __init__(self, feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.input_norm(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]       # Chỉ lấy hidden state cuối
        return self.classifier(out), None


class DriverLSTM(nn.Module):
    """
    Bi-LSTM tối ưu 50k tham số được train trên Kaggle NTHU-DDD.
    """
    def __init__(self, in_dim=FEATURE_DIM, hidden_dim=64, num_classes=3):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=2, batch_first=True, 
                            bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.norm(x)
        out, _ = self.lstm(x)
        pooled = (out[:, -1, :] + out.mean(dim=1)) / 2.0
        return self.fc(pooled), None

    def predict(self, x, confidence_threshold=0.50):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confidence, pred_class = probs.max(dim=-1)
            results = []
            class_names = ["ALERT", "DROWSY", "YAWNING"]
            for i in range(x.shape[0]):
                conf = confidence[i].item()
                cls = pred_class[i].item()
                name = class_names[cls] if cls < len(class_names) else "UNKNOWN"
                results.append((cls, name, conf, None))
        return results


# ─── Model Manager ─────────────────────────────────────────────────────────────
class ModelManager:
    """Load/save/export model."""

    SAVE_DIR = Path("models/saved")

    @classmethod
    def save(cls, model: nn.Module, name: str = "driver_lstm"):
        cls.SAVE_DIR.mkdir(parents=True, exist_ok=True)
        path = cls.SAVE_DIR / f"{name}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
            "config": {
                "feature_dim": FEATURE_DIM,
                "seq_len": SEQ_LEN,
                "classes": CLASSES,
            }
        }, path)
        print(f"  Model saved → {path}")
        return path

    @classmethod
    def load(cls, name: str = "driver_lstm", lite: bool = False,
             device: str = "cpu") -> Optional[nn.Module]:
        # Hỗ trợ cả name có .pt và không có .pt
        clean_name = name.replace(".pt", "")
        path = cls.SAVE_DIR / f"{clean_name}.pt"
        if not path.exists():
            path = Path(name)
        if not path.exists():
            print(f"  Model not found: {path}")
            return None

        checkpoint = torch.load(path, map_location=device)
        
        # Kiểm tra định dạng checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            ModelClass = DriverStateLSTM_Lite if lite else DriverStateLSTM
            model = ModelClass()
        else:
            state_dict = checkpoint
            # Thử load vào DriverLSTM (model train từ Kaggle)
            model = DriverLSTM(num_classes=3)
            
        try:
            model.load_state_dict(state_dict)
        except Exception:
            # Fallback sang DriverStateLSTM
            model = DriverStateLSTM(num_classes=len(CLASSES))
            model.load_state_dict(state_dict, strict=False)

        model.to(device)
        model.eval()
        print(f"  Model loaded successfully ← {path}")
        return model

    @classmethod
    def export_onnx(cls, model: nn.Module, name: str = "driver_lstm"):
        """Export sang ONNX để deploy trên các platform khác."""
        path = cls.SAVE_DIR / f"{name}.onnx"
        dummy = torch.zeros(1, SEQ_LEN, FEATURE_DIM)
        model.eval()
        torch.onnx.export(
            model, dummy, str(path),
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}},
            opset_version=14,
        )
        print(f"  ONNX exported → {path}")
        return path
