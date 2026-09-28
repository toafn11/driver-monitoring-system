"""
training/dataset_info.py
=========================
Phân tích chi tiết từng dataset: cấu trúc thư mục, số lượng, bias risks.
In hướng dẫn download và xử lý.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATASET_GUIDE = """
╔══════════════════════════════════════════════════════════════════════╗
║           HƯỚNG DẪN DATASET - DRIVER MONITORING SYSTEM              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. NTHU-DDD (Dataset chính cho drowsiness)                         ║
║     Link: https://www.kaggle.com/datasets/nthu-ddd                  ║
║     Hoặc: https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/  ║
║                                                                      ║
║     Cấu trúc sau khi giải nén:                                       ║
║     nthu-ddd/                                                        ║
║       training_data/                                                  ║
║         drowsy/       ← video .avi                                   ║
║         nondrowsy/    ← video .avi                                   ║
║       testing_data/                                                   ║
║         drowsy/                                                       ║
║         nondrowsy/                                                    ║
║                                                                       ║
║     ⚠️  BIAS RISKS:                                                  ║
║     - 36 subjects nhưng KHÔNG subject-disjoint mặc định              ║
║     - Alert segments nhiều hơn Drowsy → class imbalance              ║
║     - Điều kiện ánh sáng khác nhau có thể tạo bias                   ║
║                                                                       ║
║     ✅  CÁCH XỬ LÝ:                                                  ║
║     - dataset_builder.py extract FEATURES (không phải pixel)         ║
║     - Dùng WeightedRandomSampler để cân bằng class                   ║
║     - Skip 2 giây đầu mỗi video (setup time)                         ║
║                                                                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  2. MRL Eye Dataset (calibrate EAR threshold)                        ║
║     Link: https://www.kaggle.com/datasets/imadeddinedjerbi/mrl-eye-dataset ║
║                                                                       ║
║     Cấu trúc:                                                         ║
║     mrl-eye-dataset/                                                  ║
║       open/           ← ảnh hồng ngoại mắt mở                       ║
║       closed/         ← ảnh hồng ngoại mắt nhắm                     ║
║                                                                       ║
║     ⚠️  BIAS RISKS:                                                   ║
║     - Background đen (infrared) nhất quán → model học background     ║
║     - Nhiều ảnh mờ (Laplacian variance < 50)                         ║
║     - Metadata không đồng nhất (kính, không kính)                    ║
║                                                                       ║
║     ✅  CÁCH XỬ LÝ:                                                  ║
║     - Crop vùng mắt, loại bỏ viền đen (threshold=10)                 ║
║     - Lọc ảnh mờ (Laplacian var < 50)                                ║
║     - Chỉ dùng để VALIDATE ngưỡng EAR, KHÔNG train LSTM chính        ║
║                                                                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  3. StateFarm Distracted Driver (hành vi mất tập trung)              ║
║     Link: https://www.kaggle.com/c/state-farm-distracted-driver-detection ║
║                                                                       ║
║     10 classes: c0=safe, c1=text-right, c2=phone-right,              ║
║                 c3=text-left, c4=phone-left, c5=radio,               ║
║                 c6=drinking, c7=reaching-back, c8=makeup, c9=talking ║
║                                                                       ║
║     ⚠️  BIAS RISKS (NGHIÊM TRỌNG):                                   ║
║     - Cùng drivers ở train và test → model học NHẬN DẠNG NGƯỜI       ║
║       thay vì học hành vi → học vẹt 100%                             ║
║     - Ảnh tĩnh → không học temporal pattern                          ║
║     - Xe và góc camera cố định → bias background                     ║
║                                                                       ║
║     ✅  CÁCH XỬ LÝ:                                                  ║
║     - BẮT BUỘC driver-disjoint split                                 ║
║     - Crop chỉ vùng tay và điện thoại (không dùng toàn frame)        ║
║     - Chỉ dùng để detect "có điện thoại hay không" (binary)          ║
║                                                                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  4. Data tự thu thập (collect_data.py)                               ║
║     python collect_data.py --subject s001                            ║
║                                                                       ║
║     Hướng dẫn thu thập đúng cách:                                    ║
║     - Ngồi ở khoảng cách camera 40-80cm (giống ngồi lái xe)          ║
║     - Thu thập mỗi nhãn ít nhất 200 sequences (200 giây)             ║
║     - Đổi điều kiện: ánh sáng ban ngày, đèn, tối                    ║
║     - Đeo kính và không đeo kính                                      ║
║     - Ngồi nhiều tư thế khác nhau (không chỉ tư thế hoàn hảo)       ║
║     - Thu thập từ ít nhất 5 người khác nhau                          ║
║                                                                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  5. Pre-trained models từ HuggingFace                                ║
║     python training/download_pretrained.py --all                     ║
║                                                                       ║
║     Models:                                                           ║
║     - mosesb/drowsiness-detection-yolo-cls  (YOLO11x fine-tuned)    ║
║     - mosesb/drowsiness-detection-mobileViT-v2  (nhẹ hơn)           ║
║     - chbh7051/driver-drowsiness-detection  (ViT)                   ║
║                                                                       ║
║     Tích hợp: Ensemble với rule-based (45%) + LSTM (35%) + HF (20%) ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

QUICK_COMMANDS = """
┌─────────────────────────────────────────────────────┐
│                  LỆNH NHANH                         │
├─────────────────────────────────────────────────────┤
│ # Demo ngay (không cần train)                       │
│ python demo/live_demo.py --no-lstm                  │
│                                                     │
│ # Train với synthetic data (test pipeline)          │
│ python training/train_lstm.py --epochs 40           │
│                                                     │
│ # Train với NTHU-DDD                               │
│ python training/train_lstm.py --nthu path/to/nthu   │
│                                                     │
│ # Thu thập data tùy chỉnh                          │
│ python collect_data.py --subject ten_ban            │
│                                                     │
│ # Tải pre-trained models                            │
│ python training/download_pretrained.py --all        │
│                                                     │
│ # Đánh giá model                                    │
│ python training/evaluate.py                         │
│                                                     │
│ # Demo full (rule + LSTM + HF ensemble)             │
│ python demo/live_demo.py                            │
└─────────────────────────────────────────────────────┘
"""

if __name__ == "__main__":
    print(DATASET_GUIDE)
    print(QUICK_COMMANDS)
