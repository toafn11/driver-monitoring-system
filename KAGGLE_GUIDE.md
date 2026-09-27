# Hướng Dẫn Upload Lên Kaggle

## Bước 1: Nén toàn bộ project thành dataset

```bash
cd d:\hard
# Nén thư mục driver_monitoring (không kèm __pycache__)
# Tạo file zip để upload lên Kaggle
python -c "
import zipfile, os
from pathlib import Path

src = Path('driver_monitoring')
out = Path('driver_monitoring_code.zip')

with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
    for file in src.rglob('*'):
        if file.is_file() and '__pycache__' not in str(file) and '.pyc' not in str(file):
            zf.write(file, file.relative_to(src.parent))

print(f'Created: {out} ({out.stat().st_size / 1e6:.1f} MB)')
"
```

## Bước 2: Upload Code lên Kaggle Datasets

1. Vào https://www.kaggle.com/datasets/new
2. Đặt tên: `driver-monitoring-system`
3. Upload file `driver_monitoring_code.zip`
4. Nhấn **Create**

## Bước 3: Attach Datasets vào Notebook

Trong Kaggle Notebook → Settings → Add Data:
- `nthu-ddd` (search hoặc link trực tiếp)
- `mrl-eye-dataset`
- `driver-monitoring-system` (code vừa upload)

## Bước 4: Upload notebook

1. Vào https://www.kaggle.com/notebooks/new
2. Upload file `kaggle_train.ipynb`
3. Settings:
   - **Accelerator**: GPU T4 x2 (hoặc P100)
   - **Persistence**: Files only
   - **Internet**: ON (để download face_landmarker.task)

## Bước 5: Chạy

Nhấn **Run All** — pipeline tự động:
1. Phân tích bias dataset
2. Extract geometric features từ videos
3. Train Bi-LSTM 60 epochs
4. Đánh giá trên test set (subject-disjoint)
5. Sanity check (permutation test + feature ablation)
6. Export model weights

## Output sẽ có trong /kaggle/working/:
- `models/saved/driver_lstm_nthu.pt` — model weights
- `training_curves.png` — loss/accuracy curves
- `confusion_matrix.png` — confusion matrix
- `feature_importance.png` — feature ablation
- `nthu_analysis.png` — dataset analysis
- `model_metadata.json` — metadata

## Lưu ý quan trọng

- **Expected accuracy trên NTHU-DDD thật: 80-92%** (KHÔNG phải 100%)
- Nếu acc > 97% → nghi ngờ data leakage, kiểm tra lại split
- Drowsy Recall > 90% quan trọng hơn Overall Accuracy (không bỏ sót buồn ngủ)
- Sau khi train xong, download `.pt` file về để dùng trong `live_demo.py`
