# 🚗 Driver Monitoring System — THS2026-77

> **Hệ thống AI nhận diện buồn ngủ và mất tập trung tài xế**  
> Đại học Cần Thơ · Trường CNTT-TT · 2026

---

## ⚡ Chạy nhanh (Rule-Based, không cần train)

```bash
cd driver_monitoring
pip install -r requirements.txt
python demo/live_demo.py --no-lstm
```

---

## 📁 Cấu Trúc Dự Án

```
driver_monitoring/
├── core/
│   ├── face_analyzer.py       ← MediaPipe + EAR/MAR/Head Pose/Gaze
│   └── attention_scorer.py    ← Tính điểm tập trung 0-100%
├── models/
│   ├── lstm_classifier.py     ← Kiến trúc Bi-LSTM + Attention
│   └── saved/                 ← Weights đã train (.pt)
├── training/
│   ├── dataset_builder.py     ← Xử lý NTHU-DDD + anti-bias
│   ├── train_lstm.py          ← Script huấn luyện
│   ├── evaluate.py            ← Đánh giá chi tiết
│   └── download_pretrained.py ← Tải model HuggingFace
├── demo/
│   └── live_demo.py           ← Real-time webcam demo
├── collect_data.py            ← Thu thập data tùy chỉnh
└── requirements.txt
```

---

## 🔬 Pipeline Hoạt Động

```
[Webcam Frame]
      ↓
[MediaPipe Face Mesh] → 478 3D landmarks (pre-trained by Google)
      ↓
[Feature Extraction] → EAR, MAR, Yaw/Pitch/Roll, Gaze (X,Y), PERCLOS
      ↓
[Rule-Based] ─────────────────────────────────┐
[Bi-LSTM Classifier] (nếu đã train) ──────────┤→ Ensemble → [Attention Score 0-100%]
[HuggingFace YOLO/MobileViT] (optional) ──────┘
      ↓
[Label + Alert]
```

---

## 🛡️ Chiến Lược Chống Học Vẹt

### Vấn đề với dataset hình ảnh

| Dataset | Nguy cơ Bias | Giải pháp |
|---------|-------------|-----------|
| **NTHU-DDD** | Camera cố định → học vị trí đầu mặc định | Chỉ extract **geometric features** (EAR/MAR/pose), không dùng pixel |
| **NTHU-DDD** | Alert >> Drowsy (imbalanced) | Weighted Loss + WeightedRandomSampler |
| **MRL Eye** | Background đen hồng ngoại nhất quán | Crop + threshold + histogram eq trước khi dùng |
| **StateFarm** | Cùng driver ở train và test | **Driver-disjoint split bắt buộc** |
| **Tất cả** | Static pose learning | Dùng **sequence 30 frames** → LSTM học temporal pattern |

### Tại sao Geometric Features thay vì Pixel?

```
❌ Raw pixel: Model học "ảnh mắt nhắm trông như thế nào"
              → Nhạy với ánh sáng, màu da, góc camera
              → Học vẹt vị trí trong frame

✅ EAR/MAR/Pose: Model học "tỷ lệ hình học của mắt/miệng thay đổi thế nào"
                 → Bất biến với ánh sáng, màu sắc, khoảng cách camera
                 → Hoạt động tốt với người Việt Nam
```

---

## 🏋️ Huấn Luyện

### 1. Download dataset (Kaggle)
```bash
# NTHU-DDD
kaggle datasets download -d nthu-ddd
# MRL Eye
kaggle datasets download -d imadeddinedjerbi/mrl-eye-dataset
```

### 2. Build dataset từ NTHU-DDD
```bash
python training/train_lstm.py --nthu path/to/nthu-ddd
```

### 3. Train với synthetic data (test pipeline, không cần dataset)
```bash
python training/train_lstm.py
```

### 4. Fine-tune với data tự thu thập
```bash
python collect_data.py --subject s001
python training/train_lstm.py --data data/processed
```

### 5. Download pre-trained models từ HuggingFace
```bash
python training/download_pretrained.py --all
```

### 6. Đánh giá
```bash
python training/evaluate.py --model driver_lstm_v1
```

---

## 🎮 Demo Real-Time

```bash
# Chạy full (rule-based + LSTM)
python demo/live_demo.py

# Rule-based only (không cần model)
python demo/live_demo.py --no-lstm

# Debug mode off
python demo/live_demo.py --no-debug

# Chọn camera khác
python demo/live_demo.py --camera 1
```

### Phím tắt trong demo:
| Phím | Chức năng |
|------|-----------|
| `Q`  | Thoát |
| `R`  | Reset trạng thái |
| `D`  | Toggle debug panel |

---

## 📊 Output của Hệ Thống

### Nhãn hành vi (Gaze Zones):
| Nhãn | Mô tả | Điểm tập trung |
|------|-------|---------------|
| `FORWARD` | Nhìn thẳng | 100% |
| `RIGHT_MIRROR` | Nhìn gương phải | 75% |
| `LEFT_MIRROR` | Nhìn gương trái | 75% |
| `LOOKING_DOWN` | Nhìn xuống (điện thoại?) | 45% |
| `EYES_CLOSED` | Mắt nhắm | 0% |
| `YAWNING` | Đang ngáp | 50% |
| `FAR_LEFT/RIGHT` | Mất tập trung hoàn toàn | 20% |

### Trạng thái tổng hợp:
| Trạng thái | Màu | Điểm |
|-----------|-----|------|
| TẬP TRUNG | 🟢 Xanh lá | ≥ 75% |
| PHÂN TÂM NHẸ | 🟡 Vàng | 55-74% |
| MẤT TẬP TRUNG | 🟠 Cam | 35-54% |
| BUỒN NGỦ | 🔴 Đỏ | 15-34% |
| NGUY HIỂM! | 🚨 Đỏ flash | < 15% |

---

## 🤖 Pre-trained Models Tích Hợp

| Model | Source | Task | Weight trong Ensemble |
|-------|--------|------|----------------------|
| MediaPipe Face Mesh | Google | Landmark detection | N/A (feature extractor) |
| YOLO drowsiness cls | HuggingFace | Drowsy/Non-drowsy | 20% |
| MobileViT-v2 | HuggingFace | Drowsy/Non-drowsy | - |
| **Custom Bi-LSTM** | Tự train | 4 classes | **35%** |
| **Rule-Based** | Logic | 6 zones | **45%** |

---

## 📈 Targets Hiệu Năng

| Metric | Target | Note |
|--------|--------|------|
| Overall Accuracy | ≥ 85% | Trên test set subject-disjoint |
| DROWSY Recall | ≥ 90% | Không bỏ sót buồn ngủ |
| Latency | ≤ 33ms | ≥ 30 FPS |
| False Alarm Rate | ≤ 15% | Không báo oan quá nhiều |

---

## 📚 Datasets

| Dataset | Samples | Link |
|---------|---------|------|
| NTHU-DDD | 36 subjects, 5 conditions | Kaggle: `nthu-ddd` |
| MRL Eye | 84,898 images | Kaggle: `mrl-eye-dataset` |
| StateFarm | 22,424 images, 10 classes | Kaggle: `state-farm-distracted-driver-detection` |
| **Custom** | Tự thu thập | `python collect_data.py` |

---

*THS2026-77 · Đại học Cần Thơ · Trường CNTT-TT*
