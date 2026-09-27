"""
training/evaluate.py
=====================
Đánh giá mô hình trên test set.
In báo cáo chi tiết: Accuracy, Precision, Recall, F1, Latency, FPS.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, precision_recall_curve)
from models.lstm_classifier import ModelManager, CLASSES, FEATURE_DIM, SEQ_LEN
from training.train_lstm import load_or_build_dataset
from sklearn.model_selection import train_test_split


def evaluate(model_name: str = "driver_lstm_v1", device: str = "cpu"):
    print(f"\n{'='*60}")
    print(f"ĐÁNH GIÁ MÔ HÌNH: {model_name}")
    print(f"{'='*60}\n")

    # Load model
    model = ModelManager.load(model_name, device=device)
    if model is None:
        print("[ERROR] Không tìm thấy model!")
        return

    # Load data
    X, y = load_or_build_dataset()
    _, X_tmp, _, y_tmp = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    _, X_test, _, y_test = train_test_split(X_tmp, y_tmp, test_size=0.4, stratify=y_tmp, random_state=42)

    X_t = torch.FloatTensor(X_test).to(device)

    # ── Inference speed benchmark ────────────────────────────────────────────
    model.eval()
    print("[BENCHMARK] Đo tốc độ inference...")

    # Warm-up
    dummy = torch.zeros(1, SEQ_LEN, FEATURE_DIM).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(dummy)

    # Thực đo
    N_RUNS = 200
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_RUNS):
            model(dummy)
    elapsed = time.perf_counter() - start
    latency_ms = elapsed / N_RUNS * 1000
    fps = 1000 / latency_ms

    print(f"  Latency per frame: {latency_ms:.2f} ms")
    print(f"  Max FPS possible:  {fps:.0f}")
    print(f"  Target (30fps):    {'✅ OK' if fps > 30 else '⚠️ Cần tối ưu'}")

    # ── Accuracy / Classification Report ─────────────────────────────────────
    with torch.no_grad():
        logits, _ = model(X_t)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        y_pred = logits.argmax(dim=1).cpu().numpy()

    print(f"\n[REPORT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=3))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES,
                cmap="Blues", ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_ylabel("Actual"); axes[0].set_xlabel("Predicted")

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=CLASSES, yticklabels=CLASSES,
                cmap="Blues", ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (normalized)")
    axes[1].set_ylabel("Actual"); axes[1].set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig(f"training/results/{model_name}_eval_confusion.png", dpi=150)
    plt.close()

    # ── Per-class accuracy barchart ───────────────────────────────────────────
    per_class_acc = cm_norm.diagonal()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(CLASSES, per_class_acc * 100,
                  color=["#4CAF50", "#F44336", "#FF9800", "#2196F3"])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Per-Class Accuracy — {model_name}")
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val*100:.1f}%", ha="center", fontsize=11)
    ax.axhline(80, color="red", linestyle="--", alpha=0.5, label="80% target")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"training/results/{model_name}_per_class.png", dpi=150)
    plt.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    overall_acc = (y_pred == y_test).mean() * 100
    print(f"\n{'='*60}")
    print(f"  TỔNG KẾT:")
    print(f"  Overall Accuracy: {overall_acc:.1f}%")
    print(f"  Latency:          {latency_ms:.1f} ms/frame")
    print(f"  FPS:              {fps:.0f}")
    for cls, acc in zip(CLASSES, per_class_acc):
        status = "✅" if acc > 0.80 else "⚠️"
        print(f"  {status} {cls:<12}: {acc*100:.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default="driver_lstm_v1")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    evaluate(args.model, args.device)
