"""
training/sanity_check.py
=========================
KIỂM TRA THẬT SỰ XEM MODEL CÓ HỌC VẸT KHÔNG.

3 bài test:
1. Cross-distribution test: Train trên Gaussian, test trên pattern khác → nếu 100% thì học vẹt
2. Permutation test: Shuffle nhãn ngẫu nhiên → accuracy phải rơi về 25% (random)
3. Adversarial test: ALERT với EAR thấp → model có đoán đúng không?
4. Feature ablation: Xóa lần lượt từng feature → xem feature nào thực sự quan trọng
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from models.lstm_classifier import DriverStateLSTM, ModelManager, CLASSES, FEATURE_DIM, SEQ_LEN

RESULTS = {}


def load_model(name="driver_lstm_v1"):
    model = ModelManager.load(name, device="cpu")
    if model is None:
        raise RuntimeError("Chưa train model! Chạy: python training/train_lstm.py")
    return model


def generate_ood_data(n=200):
    """
    Out-of-Distribution data: Dùng distributions KHÁC với lúc train.
    Nếu model vẫn 100% → nó đang học quy tắc thật sự.
    Nếu accuracy rơi mạnh → nó đã học vẹt phân phối training.
    """
    X, y = [], []
    np.random.seed(99)  # Seed khác với seed=42 lúc train

    for cls_idx, cls_name in enumerate(CLASSES):
        for _ in range(n):
            seq = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)
            t = np.linspace(0, 1, SEQ_LEN)

            if cls_name == "ALERT":
                # Phân phối khác: EAR hơi thấp hơn, noise nhiều hơn
                seq[:, 2] = 0.29 + np.random.normal(0, 0.04, SEQ_LEN)   # mean thấp hơn
                seq[:, 3] = 0.08 + np.random.normal(0, 0.05, SEQ_LEN)
                seq[:, 4] = np.random.normal(0, 0.08, SEQ_LEN)           # noise nhiều hơn
                seq[:, 9] = np.random.uniform(0, 0.06, SEQ_LEN)          # PERCLOS thay đổi

            elif cls_name == "DROWSY":
                # Pattern khác: không có sine wave, chỉ EAR thấp dần
                seq[:, 2] = 0.28 - t * 0.10 + np.random.normal(0, 0.03, SEQ_LEN)
                seq[:, 0] = seq[:, 2]
                seq[:, 9] = np.random.uniform(0.15, 0.40, SEQ_LEN)
                seq[:, 10] = np.random.uniform(0, 0.5, SEQ_LEN)

            elif cls_name == "YAWNING":
                # Ngáp ở vị trí ngẫu nhiên (không phải 1/3 → 2/3 như lúc train)
                start = np.random.randint(0, 20)
                end = min(start + np.random.randint(5, 12), SEQ_LEN)
                seq[:, 3] = 0.08 + np.random.normal(0, 0.03, SEQ_LEN)
                yawn = np.zeros(SEQ_LEN)
                yawn[start:end] = np.random.uniform(0.4, 0.7, end - start)
                seq[:, 3] += yawn
                seq[:, 2] = 0.27 + np.random.normal(0, 0.03, SEQ_LEN)

            elif cls_name == "DISTRACTED":
                # Quay đầu theo pattern khác (không tăng dần)
                direction = np.random.choice([-1, 1])
                angle = np.random.uniform(0.3, 0.7)
                seq[:, 4] = direction * angle + np.random.normal(0, 0.04, SEQ_LEN)
                seq[:, 7] = direction * 0.5 + np.random.normal(0, 0.08, SEQ_LEN)
                seq[:, 2] = 0.30 + np.random.normal(0, 0.03, SEQ_LEN)

            seq = np.clip(seq, -1, 2).astype(np.float32)
            X.append(seq); y.append(cls_idx)

    return np.array(X), np.array(y)


def test1_ood(model):
    """Test 1: Out-of-Distribution generalization."""
    print("\n" + "="*55)
    print("TEST 1: Out-of-Distribution (phân phối khác lúc train)")
    print("="*55)
    print("Nếu acc < 60% → model ĐÃ học vẹt phân phối training")
    print("Nếu acc > 80% → model học được đặc trưng thật sự")

    X, y = generate_ood_data(200)
    X_t = torch.FloatTensor(X)
    with torch.no_grad():
        logits, _ = model(X_t)
        preds = logits.argmax(dim=1).numpy()

    acc = accuracy_score(y, preds)
    per_class = {}
    for cls_idx, cls_name in enumerate(CLASSES):
        mask = y == cls_idx
        if mask.sum() > 0:
            per_class[cls_name] = accuracy_score(y[mask], preds[mask])

    print(f"\n  Overall OOD Accuracy: {acc*100:.1f}%")
    for cls, a in per_class.items():
        flag = "✅" if a > 0.7 else ("⚠️" if a > 0.4 else "❌")
        print(f"  {flag} {cls:<12}: {a*100:.1f}%")

    if acc > 0.80:
        print("\n  ✅ PASS: Model generalize được sang phân phối mới")
    elif acc > 0.55:
        print("\n  ⚠️  PARTIAL: Model generalize được 1 phần — cần thêm data thật")
    else:
        print("\n  ❌ FAIL: Model học vẹt phân phối training!")
        print("     → Cần train trên NTHU-DDD thật, không dùng synthetic")

    RESULTS["ood_acc"] = acc
    return acc


def test2_permutation(model):
    """Test 2: Permutation test — nhãn ngẫu nhiên → phải về 25%."""
    print("\n" + "="*55)
    print("TEST 2: Permutation Test (shuffle nhãn ngẫu nhiên)")
    print("="*55)
    print("Acc phải ≈ 25% (random chance) → model không đoán theo nhãn")

    X, y = generate_ood_data(100)
    y_random = np.random.permutation(y)  # Shuffle nhãn
    X_t = torch.FloatTensor(X)
    with torch.no_grad():
        logits, _ = model(X_t)
        preds = logits.argmax(dim=1).numpy()

    acc = accuracy_score(y_random, preds)
    print(f"\n  Accuracy với nhãn ngẫu nhiên: {acc*100:.1f}%")
    if acc < 0.35:
        print("  ✅ PASS: Model không đoán theo nhãn random (như kỳ vọng)")
    else:
        print("  ❌ FAIL: Accuracy quá cao với nhãn random → có data leakage!")

    RESULTS["permutation_acc"] = acc
    return acc


def test3_adversarial(model):
    """
    Test 3: Adversarial cases — tình huống khó.
    Ví dụ: người đang chớp mắt (ALERT) vs người buồn ngủ (DROWSY)
    — cả 2 đều có EAR thấp nhưng khác nhau về temporal pattern.
    """
    print("\n" + "="*55)
    print("TEST 3: Adversarial — Các tình huống mơ hồ")
    print("="*55)

    cases = []

    # Case 1: Chớp mắt nhanh (ALERT) — EAR thấp trong 2-3 frames
    seq = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)
    seq[:, 2] = 0.33  # Mắt mở
    seq[14:17, 2] = 0.12  # Chớp 3 frames
    seq[:, 9] = 0.03  # PERCLOS thấp
    cases.append(("Chớp mắt nhanh (→ ALERT)", seq, 0))

    # Case 2: Mắt nhắm kéo dài (DROWSY) — EAR thấp trong 10+ frames
    seq2 = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)
    seq2[:, 2] = 0.33
    seq2[10:25, 2] = 0.14  # Nhắm 15 frames = 0.5 giây
    seq2[:, 9] = 0.30  # PERCLOS cao
    seq2[:, 10] = 0.5
    cases.append(("Mắt nhắm dài (→ DROWSY)", seq2, 1))

    # Case 3: Vừa ngáp vừa nhìn nghiêng (YAWNING hay DISTRACTED?)
    seq3 = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)
    seq3[:, 2] = 0.27
    seq3[:, 3] = 0.40  # Ngáp vừa
    seq3[:, 4] = 0.25  # Hơi quay đầu
    cases.append(("Vừa ngáp vừa nhìn nghiêng (mơ hồ)", seq3, 2))

    # Case 4: ALERT nhưng EAR thấp do kính râm
    seq4 = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)
    seq4[:, 2] = 0.24  # EAR thấp (kính)
    seq4[:, 9] = 0.05  # PERCLOS thấp (vẫn thức)
    seq4[:, 4] = 0.03  # Nhìn thẳng
    seq4[:, 5] = 0.01
    cases.append(("ALERT + EAR thấp (kính râm) → khó", seq4, 0))

    print(f"  {'Tình huống':<40} {'Đoán':<12} {'Đúng?'}")
    print("  " + "-"*60)
    correct = 0
    for desc, seq, true_cls in cases:
        x = torch.FloatTensor(seq).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_cls = logits.argmax(dim=1).item()
            conf = probs.max().item()

        flag = "✅" if pred_cls == true_cls else "❌"
        if pred_cls == true_cls:
            correct += 1
        print(f"  {desc:<40} {CLASSES[pred_cls]:<12} {flag} ({conf*100:.0f}%)")

    print(f"\n  Adversarial Accuracy: {correct}/{len(cases)} = {correct/len(cases)*100:.0f}%")
    RESULTS["adversarial"] = correct / len(cases)


def test4_feature_ablation(model):
    """
    Test 4: Feature Ablation — xem feature nào thực sự quan trọng.
    Nếu xóa EAR mà accuracy không đổi → model KHÔNG dùng EAR → học vẹt feature khác.
    """
    print("\n" + "="*55)
    print("TEST 4: Feature Ablation (xóa từng feature xem tác động)")
    print("="*55)
    print("Feature quan trọng → khi xóa, accuracy giảm mạnh")

    feature_names = ["EAR_L", "EAR_R", "EAR_avg", "MAR",
                     "Yaw", "Pitch", "Roll", "Gaze_X", "Gaze_Y",
                     "PERCLOS", "EyesClosed", "MouthOpen"]

    X, y = generate_ood_data(100)
    X_t = torch.FloatTensor(X)

    # Baseline
    with torch.no_grad():
        base_preds = model(X_t)[0].argmax(1).numpy()
    base_acc = accuracy_score(y, base_preds)

    print(f"  Baseline Accuracy: {base_acc*100:.1f}%\n")
    print(f"  {'Feature':<14} {'Acc khi xóa':>12} {'Giảm':>8} {'Quan trọng?'}")
    print("  " + "-"*50)

    importances = {}
    for feat_idx, fname in enumerate(feature_names):
        X_ablated = X.copy()
        X_ablated[:, :, feat_idx] = 0.0  # Zero-out feature
        X_abl_t = torch.FloatTensor(X_ablated)
        with torch.no_grad():
            abl_preds = model(X_abl_t)[0].argmax(1).numpy()
        abl_acc = accuracy_score(y, abl_preds)
        drop = base_acc - abl_acc
        importances[fname] = drop
        flag = "🔴 Quan trọng!" if drop > 0.15 else ("🟡 Vừa" if drop > 0.05 else "⚪ Ít")
        print(f"  {fname:<14} {abl_acc*100:>11.1f}% {drop*100:>+7.1f}%  {flag}")

    top_feat = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"\n  Top 3 features quan trọng nhất: {[f[0] for f in top_feat]}")
    RESULTS["ablation"] = importances


def test5_confidence_calibration(model):
    """
    Test 5: Kiểm tra confidence calibration.
    Model tốt: khi nói 90% chắc → đúng ~90% lần.
    Model học vẹt: luôn nói 99% → overconfident.
    """
    print("\n" + "="*55)
    print("TEST 5: Confidence Calibration")
    print("="*55)
    print("Model tốt: confident ↔ accuracy cao, uncertain ↔ accuracy thấp")

    X, y = generate_ood_data(400)
    X_t = torch.FloatTensor(X)
    with torch.no_grad():
        logits, _ = model(X_t)
        probs = torch.softmax(logits, dim=-1).numpy()
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    corrects = (preds == y).astype(float)

    bins = [(0.25, 0.50), (0.50, 0.70), (0.70, 0.85), (0.85, 1.01)]
    print(f"\n  {'Confidence':>15} | {'Samples':>8} | {'Accuracy':>10} | {'Calibrated?'}")
    print("  " + "-"*55)
    for lo, hi in bins:
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        avg_conf = confs[mask].mean()
        acc = corrects[mask].mean()
        gap = abs(avg_conf - acc)
        calib = "✅" if gap < 0.10 else ("⚠️" if gap < 0.20 else "❌ Overconfident")
        print(f"  {lo:.0%}–{hi:.0%}  (avg={avg_conf:.2f}) | {mask.sum():>8} | {acc*100:>9.1f}% | {calib}")

    avg_conf_all = confs.mean()
    avg_acc_all  = corrects.mean()
    print(f"\n  Avg confidence: {avg_conf_all*100:.1f}% | Avg accuracy: {avg_acc_all*100:.1f}%")
    if avg_conf_all - avg_acc_all > 0.20:
        print("  ❌ Model OVERCONFIDENT — cần temperature scaling hoặc thêm dropout")
    else:
        print("  ✅ Calibration chấp nhận được")


def print_verdict():
    print("\n" + "="*55)
    print("KẾT LUẬN VỀ 100% ACCURACY")
    print("="*55)
    ood = RESULTS.get("ood_acc", 0)
    perm = RESULTS.get("permutation_acc", 0)

    print(f"""
  Vấn đề với 100% accuracy trên Synthetic Data:
  ─────────────────────────────────────────────
  Synthetic data được tạo với công thức RẤT TÁCH BIỆT:
    • ALERT:      EAR ≈ 0.32, PERCLOS ≈ 0.02
    • DROWSY:     EAR ≈ 0.22, PERCLOS ≈ 0.20-0.35
    • YAWNING:    MAR tăng đột biến ở giữa sequence
    • DISTRACTED: Yaw lệch hẳn sang 1 bên

  Các class này KHÔNG OVERLAP → linear classifier cũng 100%.
  LSTM chỉ cần học: "nếu PERCLOS > 0.15 thì DROWSY"

  Đây là TRIVIAL LEARNING, không phải học thật.

  OOD Test: {ood*100:.1f}% → {'Model generalize được' if ood > 0.7 else 'Model có học vẹt phần nào'}
  Permutation: {perm*100:.1f}% → {'OK (≈25%)' if perm < 0.35 else 'FAIL - data leakage?'}

  ĐỂ KIỂM TRA THẬT SỰ:
  ─────────────────────
  1. Train trên NTHU-DDD thật → test trên subject CHƯA TRAIN
  2. Expected accuracy: 80-92% (không phải 100%)
  3. Nếu vẫn 100% → kiểm tra data leakage trong split
  4. Subject-disjoint split là tiêu chuẩn bắt buộc
""")


if __name__ == "__main__":
    model = load_model()
    model.eval()

    test1_ood(model)
    test2_permutation(model)
    test3_adversarial(model)
    test4_feature_ablation(model)
    test5_confidence_calibration(model)
    print_verdict()
