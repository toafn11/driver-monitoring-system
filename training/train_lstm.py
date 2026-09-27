"""
training/train_lstm.py
=======================
Script huấn luyện LSTM classifier.
Hỗ trợ:
  - Train từ đầu trên NTHU-DDD
  - Fine-tune từ pre-extracted dataset
  - Class-weighted loss (chống imbalanced data)
  - Early stopping + LR scheduling
  - Training curves + confusion matrix
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from typing import Optional
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json, time
from tqdm import tqdm
from collections import Counter

from models.lstm_classifier import DriverStateLSTM, DriverStateLSTM_Lite, ModelManager, CLASSES
from training.dataset_builder import DatasetBuilder


# ─── Training Config ──────────────────────────────────────────────────────────
CONFIG = {
    "epochs":          60,
    "batch_size":      64,
    "learning_rate":   1e-3,
    "weight_decay":    1e-4,
    "patience":        12,       # Early stopping patience
    "val_split":       0.15,
    "test_split":      0.10,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "lite_mode":       False,    # True = dùng model nhỏ hơn
    "model_name":      "driver_lstm_v1",
    "use_weighted_sampler": True,  # Quan trọng: chống class imbalance
}

RESULTS_DIR = Path("training/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_or_build_dataset(data_dir: str = "data/processed",
                           nthu_path: Optional[str] = None) -> tuple:
    """
    Load dataset đã xử lý, hoặc build từ NTHU-DDD nếu chưa có.
    """
    builder = DatasetBuilder(data_dir)
    X_path = Path(data_dir) / "X_train.npy"

    if X_path.exists():
        print(f"[DATA] Loading pre-built dataset từ {data_dir}")
        X, y = builder.load_dataset("train")
    elif nthu_path and Path(nthu_path).exists():
        print(f"[DATA] Building từ NTHU-DDD: {nthu_path}")
        X, y = builder.process_nthu_ddd(nthu_path)
        if len(X) > 0:
            # Phân tích bias
            builder.analyze_dataset_bias(X, y)
            # Augment
            X, y = builder.augment_sequences(X, y)
            builder.save_dataset(X, y, "train")
    else:
        print("[DATA] Không có dataset thực → tạo SYNTHETIC dataset để demo")
        X, y = generate_synthetic_dataset()

    return X, y


def generate_synthetic_dataset(n_per_class: int = 800) -> tuple:
    """
    Tạo dataset tổng hợp có đặc tính thực tế để test pipeline.
    Giả lập EAR, MAR, head pose cho từng trạng thái.
    """
    print("[SYNTH] Generating synthetic dataset...")
    X_all, y_all = [], []
    seq_len, feat_dim = 30, 12

    np.random.seed(42)

    for cls_idx, cls_name in enumerate(CLASSES):
        for _ in range(n_per_class):
            seq = np.zeros((seq_len, feat_dim), dtype=np.float32)
            t = np.linspace(0, 1, seq_len)

            if cls_name == "ALERT":
                # Mắt mở bình thường, đầu thẳng, nhìn thẳng
                seq[:, 0] = 0.32 + np.random.normal(0, 0.02, seq_len)  # EAR_L
                seq[:, 1] = 0.31 + np.random.normal(0, 0.02, seq_len)  # EAR_R
                seq[:, 2] = 0.315 + np.random.normal(0, 0.015, seq_len) # EAR_avg
                seq[:, 3] = 0.10 + np.random.normal(0, 0.03, seq_len)  # MAR
                seq[:, 4] = np.random.normal(0, 0.05, seq_len)          # Yaw ≈ 0
                seq[:, 5] = np.random.normal(0, 0.05, seq_len)          # Pitch ≈ 0
                seq[:, 9] = 0.02                                          # Low PERCLOS

            elif cls_name == "DROWSY":
                # Mắt dần nhắm, chớp chậm, đầu gật
                blink = np.sin(t * np.pi * 2) * 0.08
                seq[:, 0] = np.clip(0.22 + blink + np.random.normal(0, 0.02, seq_len), 0.05, 0.4)
                seq[:, 2] = seq[:, 0]
                seq[:, 1] = seq[:, 0] + np.random.normal(0, 0.01, seq_len)
                seq[:, 3] = 0.15 + np.random.normal(0, 0.05, seq_len)
                seq[:, 5] = -0.1 * t + np.random.normal(0, 0.03, seq_len)  # Đầu dần cúi
                seq[:, 9] = 0.20 + t * 0.15                                   # PERCLOS tăng
                seq[:, 10] = t * 0.3                                           # Eyes closed duration

            elif cls_name == "YAWNING":
                # Miệng mở rộng đột ngột
                yawn_profile = np.zeros(seq_len)
                start = seq_len // 3
                end = 2 * seq_len // 3
                yawn_profile[start:end] = np.sin(np.linspace(0, np.pi, end - start)) * 0.5
                seq[:, 0] = 0.28 + np.random.normal(0, 0.02, seq_len)
                seq[:, 2] = seq[:, 0]
                seq[:, 3] = 0.10 + yawn_profile + np.random.normal(0, 0.03, seq_len)
                seq[:, 11] = yawn_profile / 0.5 * 0.7                    # Mouth open duration

            elif cls_name == "DISTRACTED":
                # Đầu quay sang một bên, gaze lệch
                direction = np.random.choice([-1, 1])
                seq[:, 0] = 0.30 + np.random.normal(0, 0.02, seq_len)
                seq[:, 2] = seq[:, 0]
                seq[:, 4] = direction * (0.3 + t * 0.1) + np.random.normal(0, 0.03, seq_len)  # Yaw lệch
                seq[:, 7] = direction * 0.4 + np.random.normal(0, 0.05, seq_len)               # Gaze_x lệch
                seq[:, 9] = 0.05

            # Clip tất cả về range hợp lệ
            seq = np.clip(seq, -1.0, 2.0)
            X_all.append(seq)
            y_all.append(cls_idx)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int64)
    perm = np.random.permutation(len(X))
    print(f"[SYNTH] Generated {len(X)} sequences | Classes: {Counter(y.tolist())}")
    return X[perm], y[perm]


# ─── Training Loop ─────────────────────────────────────────────────────────────
def train(config: dict = CONFIG, nthu_path: str = None):
    device = torch.device(config["device"])
    print(f"\n{'='*60}")
    print(f"TRAINING DRIVER STATE LSTM")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # 1. Load data
    X, y = load_or_build_dataset(nthu_path=nthu_path)
    print(f"[DATA] Total: {len(X)} sequences, Shape: {X.shape}")

    # 2. Subject-disjoint split
    # Dùng stratified split để đảm bảo phân phối nhãn đồng đều
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=config["val_split"] + config["test_split"],
        stratify=y, random_state=42
    )
    val_ratio = config["val_split"] / (config["val_split"] + config["test_split"])
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=1 - val_ratio, stratify=y_tmp, random_state=42
    )
    print(f"[DATA] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 3. Tạo DataLoaders
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t   = torch.FloatTensor(X_val).to(device)
    y_val_t   = torch.LongTensor(y_val).to(device)
    X_test_t  = torch.FloatTensor(X_test).to(device)
    y_test_t  = torch.LongTensor(y_test).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)

    # Weighted sampler để chống class imbalance
    if config["use_weighted_sampler"]:
        counts = np.bincount(y_train)
        weights_per_class = 1.0 / (counts + 1e-6)
        sample_weights = torch.FloatTensor([weights_per_class[y] for y in y_train])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)

    val_ds   = TensorDataset(X_val_t, y_val_t)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])

    # 4. Model
    ModelClass = DriverStateLSTM_Lite if config["lite_mode"] else DriverStateLSTM
    model = ModelClass().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] {ModelClass.__name__} | Parameters: {n_params:,}")

    # 5. Loss với class weights
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(
        [len(y_train) / (len(CLASSES) * c + 1e-6) for c in class_counts]
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # 6. Optimizer + Scheduler
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"],
                      weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-5)

    # 7. Training loop
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, config["epochs"] + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += len(y_batch)

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits, _ = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)

        scheduler.step()

        t_loss = train_loss / train_total
        v_loss = val_loss / val_total
        t_acc  = train_correct / train_total * 100
        v_acc  = val_correct / val_total * 100
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{config['epochs']} | "
                  f"Train Loss: {t_loss:.4f} Acc: {t_acc:.1f}% | "
                  f"Val Loss: {v_loss:.4f} Acc: {v_acc:.1f}%")

        # Early stopping
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            patience_counter = 0
            ModelManager.save(model, config["model_name"])
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"  Early stopping tại epoch {epoch}")
                break

    # 8. Test evaluation
    print("\n[TEST] Đánh giá trên test set...")
    best_model = ModelManager.load(config["model_name"], device=config["device"])
    best_model.eval()
    with torch.no_grad():
        logits, _ = best_model(X_test_t)
        y_pred = logits.argmax(dim=1).cpu().numpy()

    print(classification_report(y_test, y_pred, target_names=CLASSES))

    # 9. Plot
    _plot_training_curves(history, config["model_name"])
    _plot_confusion_matrix(y_test, y_pred, CLASSES, config["model_name"])

    print(f"\n✅ Best Val Accuracy: {best_val_acc:.1f}%")
    return best_model, history


def _plot_training_curves(history: dict, name: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train", color="#4A90D9")
    ax1.plot(history["val_loss"],   label="Val",   color="#E74C3C")
    ax1.set_title("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history["train_acc"], label="Train", color="#4A90D9")
    ax2.plot(history["val_acc"],   label="Val",   color="#E74C3C")
    ax2.set_title("Accuracy (%)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle(f"Training: {name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = RESULTS_DIR / f"{name}_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {path}")


def _plot_confusion_matrix(y_true, y_pred, class_names: list, name: str):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues", ax=ax, vmin=0, vmax=1)
    ax.set_title(f"Confusion Matrix (normalized) - {name}")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    path = RESULTS_DIR / f"{name}_confusion.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix → {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Driver State LSTM")
    parser.add_argument("--nthu",  type=str, default=None, help="Path đến NTHU-DDD dataset")
    parser.add_argument("--data",  type=str, default="data/processed")
    parser.add_argument("--lite",  action="store_true", help="Dùng model nhỏ hơn")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    CONFIG["lite_mode"] = args.lite
    CONFIG["epochs"]    = args.epochs
    train(CONFIG, nthu_path=args.nthu)
