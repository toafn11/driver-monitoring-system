# Huấn luyện phiên bản 20 đặc trưng — kế hoạch trước mắt

Ngày đối chiếu dataset: 27/09/2026. Các checkpoint 12 đặc trưng cũ được giữ nguyên.
Pipeline mới dùng schema `geometric20-v2`; không trộn với `enrich_sequence()` cũ.
Notebook mới: `kaggle_train20.ipynb`. Chưa có kết quả huấn luyện thật của phiên bản này.

## 1. Chọn dữ liệu

### Ưu tiên: UTA-RLDD video gốc

- Nguồn tác giả: https://sites.google.com/view/utarldd/home
- Bản Kaggle đã kiểm tra danh sách file:
  https://www.kaggle.com/datasets/thuannn18022005/uta-rldd-full
- API kiểm tra: https://www.kaggle.com/api/v1/datasets/list/thuannn18022005/uta-rldd-full?pageSize=1000
- Bản mirror có 182 file video, khoảng 120.45 GB theo tổng kích thước file trong API,
  5 fold; một số video của người 32 và 49 chia thành `10_1`, `10_2`.
  Đây là kiểm tra danh sách file, chưa kiểm tra checksum/tính toàn vẹn tất cả video.
- Dataset tác giả mô tả 60 người, khoảng 30 giờ, nhãn `0=alert`,
  `5=low vigilance`, `10=drowsy`. Đây là nhãn trạng thái chủ đạo của video,
  không phải nhãn chính xác từng lần nhắm mắt hay ngáp.
- Giai đoạn đầu: chỉ dùng `0` và `10`, bỏ `5` một cách tường minh.
  Sau đó có thể chạy thí nghiệm 3 lớp với `--include-low`.
  **Không đổi tên `LOW_VIGILANCE` thành `YAWNING` hay `DISTRACTED`.**
- Fold 1–3 train, fold 4 validation, fold 5 test. Đây là một holdout để phát triển,
  chưa phải kết quả 5-fold cross-validation theo giao thức đầy đủ của tác giả.
- Attach dataset vào Kaggle thay vì tải 120 GB về máy. Bước trích xuất là công việc
  CPU nặng; GPU hữu ích chủ yếu cho bước train. Giữ cache giữa các lần chạy.

### Các bản dễ chọn nhầm

| Dataset | Điều đã kiểm tra | Cách dùng |
|---|---|---|
| `minhngt02/uta-rldd` | File mẫu là `test/active/image_0162.jpg`… | Không mặc định coi là chuỗi liên tục; chưa thấy ánh xạ người/video/thời gian đủ dùng |
| `ikhlaselhamly/nthu-ddd` | File mẫu còn mã người và frame, nhưng số frame có khoảng khuyết như 1003,1005,1012… | Có thể dùng cho bài toán ảnh; không nối frame tùy ý để tính vận tốc/nháy mắt |
| `rishab260/uta-reallife-drowsiness-dataset` | API trả 145 video thuộc fold 1–4, khoảng 96.63 GB | Thiếu fold 5 nếu dùng một mình; đừng gọi là bản đầy đủ |
| `mathiasviborg/uta-rldd-videos-cropped-by-faces` | Video crop mặt khoảng 6.7 GB; mẫu chia clip 10 giây | Phương án thí nghiệm nhẹ sau khi kiểm tra FPS, nguồn clip và ảnh hưởng crop lên pose; chưa có adapter tự động ở pipeline này |

Các kết luận về mirror dựa vào API file công khai, không chỉ tên dataset.

### Bổ sung sau: ngáp và mất tập trung

- YawDD video: https://www.kaggle.com/datasets/enider/yawdd-dataset
  API đã thấy video `.avi` trong `Dash/Dash/Female/...`; khoảng 5.46 GB theo listing dataset.
  Cần gán nhãn theo đoạn thời gian, phân biệt nói chuyện/hát/ngáp; không gán toàn bộ
  một video có ngáp thành `YAWNING`.
- Mất tập trung: thu clip trong môi trường mô phỏng với hướng nhìn lệch kéo dài,
  nhìn gương ngắn, nhìn xuống và các tình huống bình thường. Giữ mã người/phiên/thời gian.
  Hướng đầu không đủ để kết luận người dùng đang cầm điện thoại.
- Với nhãn hành vi đồng thời (ngáp + buồn ngủ, ngáp + nhìn lệch), hướng dài hạn là
  tách đầu ra trạng thái buồn ngủ và hành vi, thay vì ép mọi thứ vào một nhãn loại trừ nhau.
- MRL/ảnh mắt rời không bổ sung đủ 20 đặc trưng khuôn mặt theo thời gian.
- Kiểm tra điều kiện sử dụng từ nguồn gốc trước khi tái phân phối video; mirror Kaggle
  không tự thay thế các điều kiện của tác giả.

## 2. Mô hình và đặc trưng đã chuẩn bị

- GRU một lớp, hidden=64, attention pooling, dropout=0.2, đầu ra theo tên nhãn lưu
  trong metadata. Với hai lớp: GRU có 18.723 tham số; LSTM một lớp có 24.227 tham số.
  Kiến trúc Bi-LSTM 20 đặc trưng/bốn lớp hiện có là 665.133 tham số.
- Chuẩn hóa từng đặc trưng bằng thống kê **chỉ từ train**, lưu trong checkpoint.
  Dữ liệu nằm trong RAM CPU, chuyển từng batch lên GPU.
- Chỉ dùng weighted sampling để cân bằng lớp; không nhân thêm inverse-frequency loss.
- Chọn checkpoint bằng validation macro-F1, early stopping, giảm learning rate.
- Không mở test để chọn model. Lệnh đánh giá test là bước riêng sau khi chốt cấu hình.
- Lưu seed, phiên bản Python/PyTorch/NumPy, danh sách người, cấu hình tiền xử lý,
  hash manifest, history, trọng số, confusion matrix và dự đoán từng mẫu.
- Chạy cùng cấu hình 12 và 20 đầu vào để đo giá trị của 8 đặc trưng mới.
  Bản 12 vẫn sử dụng mask để bỏ frame không có mặt, nhưng mask không là input của RNN.

**Không cam kết GRU chính xác hơn Bi-LSTM.** Mục đích hiện tại là giảm số tham số,
giảm chi phí thử nghiệm và tạo phép so sánh có kiểm soát.

### Định nghĩa mới tránh lệch giữa train và suy luận

- Mặc định lấy 15 mẫu/giây, cửa sổ 60 mẫu (~4 giây), bước trượt 15 mẫu (~1 giây).
  15 Hz là cấu hình tiết kiệm chi phí; chớp mắt rất ngắn có thể bị bỏ lỡ.
  Thí nghiệm 30 Hz phải trích xuất lại và dùng 120 mẫu nếu muốn giữ cửa sổ 4 giây.
- Tính vận tốc theo giây từ timestamp; không coi frame nào cũng cách nhau 1/30 giây.
- PERCLOS tính trên quan sát hợp lệ trong cửa sổ. Thời lượng đóng mắt/mở miệng
  bắt đầu ở biên cửa sổ là cận dưới, không phải thời lượng đầy đủ của sự kiện.
- `EAR_window_ratio` là tỷ lệ so với phân vị 80 trong cửa sổ, chưa phải hiệu chỉnh cá nhân.
- `Face_valid` là mask 0/1 thật, thay cho `Face_conf=1` cố định.
- Không nối thời gian qua frame mất mặt: giữ mask, reset thời lượng, bỏ cửa sổ có
  dưới 80% frame hợp lệ hoặc mất mặt liên tục trên 0.2 giây.
- Dùng quy ước góc đầu `camera-v2`: sửa ánh xạ trục và loại phép quay trung tính 180°.
  Đã kiểm tra bằng phép chiếu 3D tổng hợp; vẫn cần kiểm tra camera/người thật.
- Bộ trích xuất hiện dựa trên `frame_index / source_fps` cho video tốc độ khung hình
  cố định. Video VFR cần chuẩn hóa thời gian bằng công cụ giữ timestamp trước khi dùng.
- `core.features20.FeatureWindow` dùng cùng hàm tiền xử lý cho một luồng suy luận.
  Khi triển khai phải giữ cách lấy mẫu, resize, IMAGE mode và quy ước góc đầu theo checkpoint.

## 3. Quy trình Kaggle

1. Upload `driver_monitoring_kaggle20.zip` thành dataset code riêng; attach code và
   `thuannn18022005/uta-rldd-full` vào notebook mới.
2. Import `kaggle_train20.ipynb`, chọn GPU cho giai đoạn train; bật Internet để cài
   thư viện và tải `face_landmarker.task` nếu chưa attach model.
3. Cell đầu tìm thư mục code hoặc giải nén đúng file code đã đóng gói.
   Kiểm tra `DATA_ROOT` mà notebook tìm được trước bước xử lý dài.
4. Tạo manifest và kiểm tra số người/số video mỗi split.
5. Build dữ liệu; cache thô được lưu theo video. Nếu session dừng giữa chừng,
   bảo toàn cache và chạy lại với thư mục output dataset mới nếu thư mục cũ đã có file.
   Có thể dùng lệnh `cache` chia công việc trích xuất thành nhiều shard, sau đó gom cache.
6. Train GRU-12 và GRU-20; xem validation macro-F1 và recall DROWSY.
   Thử LSTM-20 nếu cần. Dùng nhiều seed trước khi kết luận một mô hình tốt hơn.
7. Chốt model dựa trên validation, bật cell đánh giá test và lưu toàn bộ output.

Các lệnh tương đương (chạy từ thư mục dự án):

```bash
python -m training.prepare20 manifest-rldd --root /kaggle/input/uta-rldd-full --output /kaggle/working/manifest.csv
python -m training.prepare20 build --manifest /kaggle/working/manifest.csv --model /kaggle/working/face_landmarker.task --output /kaggle/working/processed20 --cache-dir /kaggle/working/cache20
python -m training.train20 train --data /kaggle/working/processed20 --output /kaggle/working/gru12 --feature-dim 12
python -m training.train20 train --data /kaggle/working/processed20 --output /kaggle/working/gru20 --feature-dim 20
python -m training.train20 train --data /kaggle/working/processed20 --output /kaggle/working/lstm20 --cell lstm
# Sau khi chốt cấu hình bằng validation:
python -m training.train20 evaluate --data /kaggle/working/processed20 --checkpoint /kaggle/working/gru20/best.pt --output /kaggle/working/test_final
```

Cache riêng, ví dụ shard 0/4 (lặp lại shard 1,2,3 rồi gom cache):

```bash
python -m training.prepare20 cache --manifest /kaggle/working/manifest.csv --model /kaggle/working/face_landmarker.task --cache-dir /kaggle/working/cache20 --num-shards 4 --shard-index 0
```

Cache fingerprint có đường dẫn/size/mtime nguồn và hash model/mã trích xuất. Khi chuyển
session, nếu Kaggle thay đường dẫn hoặc mtime thì cache được tạo lại để tránh dùng sai nguồn.

Với dữ liệu khác, tạo CSV: `path,subject,split,label,start_s,end_s`.
`end_s` trống là hết video. Mỗi đoạn phải có nhãn đã xác minh; không chồng đoạn.
`subject` phải là cùng một định danh cho cùng người qua mọi phiên/camera/dataset trùng nguồn.
Train/val/test phải có đủ cùng bộ nhãn. Tuyệt đối không tạo nhãn bằng ngưỡng EAR rồi
báo cáo model học được buồn ngủ từ ground truth độc lập.

## 4. Giới hạn và việc tiếp theo

- Phiên bản mới chưa train trên dữ liệu thật; thử nghiệm tổng hợp chỉ chứng minh luồng
  phần mềm hoạt động, không đo độ chính xác sản phẩm.
- `demo/live_demo.py` vẫn dùng pipeline/checkpoint cũ. Không đưa `best.pt` mới vào demo cũ;
  phải dùng `models.temporal20.load_checkpoint()` và `FeatureWindow` đúng schema.
- Không dùng số FPS suy luận model để công bố FPS camera-to-alert. Báo cáo hiện ghi rõ
  độ trễ model-only. Chưa đo cảnh báo giả mỗi giờ hoặc thời gian phản ứng sự kiện.
- Chưa thực hiện pruning/quantization: cần model FP32 đã có chất lượng đối chứng trước.
- Dữ liệu Việt Nam, kiểm thử webcam, calibration và quyết định cảnh báo vẫn là bước kế tiếp.
- Thí nghiệm đầu tiên cần trả lời: (1) có đủ mặt hợp lệ ở mọi lớp/người không,
  (2) 20 đặc trưng có hơn 12 không, (3) lỗi tập trung vào người/điều kiện nào,
  (4) model nhẹ có đủ tốt so với LSTM không.

## 5. Kiểm tra đã thực hiện tại máy phát triển

- 14 kiểm tra của pipeline mới đã qua: đặc trưng/thời gian/mask, phân chia người,
  góc đầu bằng chiếu 3D, giải mã video/lấy mẫu/cache, train ngắn và lưu/nạp/evaluate.
- Kiểm tra riêng với MediaPipe thật và video trống: giải mã/lấy mẫu chạy được,
  không tạo giả phát hiện khuôn mặt.
- Toàn bộ 26 file Python của dự án và các code cell notebook mới hợp lệ cú pháp.
- Chưa kiểm tra toàn bộ dataset, GPU Kaggle, video có mặt người thật hoặc độ chính xác
  model mới. Bộ test tích hợp dùng dữ liệu nhỏ tổng hợp chỉ để kiểm tra phần mềm.
- Môi trường kiểm tra: Python 3.12, PyTorch 2.14.0 CPU, NumPy 2.5.3,
  MediaPipe 1.0.1, OpenCV 5.0.0. Phiên bản thực tế trên Kaggle được ghi trong output.
