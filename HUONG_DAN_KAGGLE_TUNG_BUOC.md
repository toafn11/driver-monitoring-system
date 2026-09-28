# Chạy lại RLDD sau lỗi FPS — hướng dẫn thao tác

## Đã sửa gì?

Log cũ dừng sau 35 video vì video người 18 có FPS khoảng 12,002, thấp hơn cấu hình 15 FPS. Chưa có epoch train nào hoàn thành trong log đó.

Pipeline mới kiểm tra toàn bộ video trước MediaPipe, lưu cache nguyên tử theo video, ghi tiến độ từng nhóm và giữ các video đã thành công khi video khác lỗi. Không tự bỏ video lỗi để train một tập thiếu dữ liệu.

Ba notebook thay thế bản một lượt `kaggle_train20.ipynb`:

1. `kaggle_01_extract20.ipynb`: kiểm tra, trích xuất theo 4 nhóm.
2. `kaggle_02_build20.ipynb`: ghép cache thành các tập train/validation/test.
3. `kaggle_03_train20.ipynb`: huấn luyện và so sánh 12/20 đặc trưng.

Mã được ghim ở commit `8aeacc1`; không tự lấy thay đổi mới từ main. Notebook lấy mã từ GitHub, nên cần Internet On. Kiểm tra tên notebook và các input trước khi chạy.

## Bước 0 — Giữ lại kết quả cũ

- Mở phiên lỗi cũ, xem Output có `cache20` hay không. Nếu có, tải về hoặc lưu thành dataset riêng tư. Không xóa trước khi kiểm tra.
- Các cache cũ 15 FPS **không được dùng trực tiếp** cho cấu hình mới 10 FPS. Phiên bản này chưa chuyển đổi cache 15 → 10 FPS; lần chạy 10 FPS sẽ trích xuất lại. Giữ cache cũ để có thể phân tích hoặc bổ sung công cụ chuyển đổi sau.
- `Files only` của phiên nháp không có nghĩa cache tự xuất hiện ở mọi phiên Save & Run All. Muốn chạy tiếp phiên mới, phải attach output/cache cũ thành Input.
- Nếu còn phiên chạy trùng, kiểm tra và dừng phiên không cần thiết trước khi bắt đầu lại.

## Bước 1 — Import notebook kiểm tra

Trong Kaggle tạo notebook, chọn **File → Import Notebook → Link**, nhập:

`https://raw.githubusercontent.com/toafn11/driver-monitoring-system/main/kaggle_01_extract20.ipynb`

Hoặc dùng File để upload notebook cùng tên từ máy.

Ở **Add Input**, gắn:

- `thuannn18022005/uta-rldd-full` — video gốc đầy đủ.
- `toznf11/driver-monitoring-face-landmarker-v1` — input riêng tư đã tạo, chứa `face_landmarker.task`.

Chỉ gắn một bản model. Không cần dataset NTHU ảnh cho lần chạy này.

Trong Session options: **Accelerator = None**, **Internet = On**. Giữ `RUN_EXTRACTION = False`, chạy các cell theo thứ tự hoặc Run All.

Kết quả cần xem:

- `inventory20.json` ghi FPS, số frame, thời lượng và thử giải mã đầu/giữa/cuối của từng video.
- Với cấu hình nhị phân hiện tại, dự kiến 122 video; số thực tế phải được đối chiếu với manifest.
- `failed = 0` và `suggested_fps >= 10` thì có thể dùng `SAMPLE_FPS = 10`.
- Nếu gợi ý nhỏ hơn 10 hoặc có lỗi, dừng để đọc các mục `issues`. Không bỏ video hoặc ép 10 FPS bằng cách nhân đôi frame. Nếu quyết định dùng FPS thấp hơn, cập nhật **tất cả** nhóm trích xuất và notebook ghép dữ liệu cùng giá trị nguyên đó.

Giới hạn: đây là kiểm tra nhanh, không giải mã hết video và không chứng nhận video là CFR. Các timestamp thử được ghi để chẩn đoán; extractor vẫn dùng frame_index/FPS. Nếu nghi video có tốc độ frame biến đổi, cần kiểm tra timestamp đầy đủ và nâng cấp bộ đọc trước khi công bố kết quả về thời gian chớp mắt. Một video vượt kiểm tra nhanh vẫn có thể lỗi ở đoạn khác khi trích xuất thật.

## Bước 2 — Trích xuất thành 4 nhóm

Sau khi kiểm tra đạt, tạo 4 notebook từ bản 01, đặt tên dễ phân biệt như `RLDD extract 0`, `RLDD extract 1`, `RLDD extract 2`, `RLDD extract 3`.

Trong mỗi bản:

```python
RUN_EXTRACTION = True
SHARD_INDEX, NUM_SHARDS = 0, 4
```

Thay `SHARD_INDEX` lần lượt bằng **0, 1, 2, 3**. Giữ `NUM_SHARDS = 4` và cùng FPS. Chạy tuần tự nếu hạn mức tài khoản không cho nhiều phiên cùng lúc.

Chọn **Save Version → Save & Run All**. Mỗi nhóm khoảng 30–31 video. Theo tốc độ log cũ, toàn bộ 122 video ở 15 FPS có thể mất khoảng 11–12 giờ chỉ để trích xuất; thời gian mới ở 10 FPS cần đo lại, không coi là cam kết nhanh hơn theo tỷ lệ cố định.

Một nhóm thành công phải có:

- `cache20_v3/`: các file `.npz` đã trích xuất.
- `cache20_v3/progress_N.json`: trạng thái từng video, mọi mục là `ok`.
- `shard_N_complete.json`: dấu xác nhận nhóm hoàn thành.

Không có dấu complete thì chưa xem là nhóm hoàn tất. Nếu có video lỗi, pipeline xử lý tiếp các video khác rồi báo thất bại; cache thành công vẫn được giữ tại thư mục output. Kiểm tra Kaggle có lưu được output của phiên thất bại hay không, rồi tải/lưu các file đó trước khi tạo phiên mới.

### Khi cần chạy tiếp

1. Gắn output của phiên trước qua **Add Input → Notebook output**; nếu output phiên lỗi không dùng được trực tiếp, tạo dataset riêng tư từ cache tải về.
2. Giữ nguyên thư mục `cache20_v3` khi đóng gói. Notebook tự tìm thư mục này trong Input.
3. Giữ nguyên input video, model, commit mã, FPS và chỉ số nhóm; chạy lại cùng nhóm.
4. File cache khớp sẽ được đọc từ input, video thiếu sẽ trích xuất vào working. Cache không khớp thì không tái sử dụng.
5. Khi ghép, attach cả output cũ lẫn output chạy tiếp: output chạy tiếp không sao chép các cache đã đọc từ input.

Khóa cache kiểm tra đường dẫn video, kích thước, thời điểm sửa file, FPS, hash model và hash bộ trích xuất khuôn mặt. Nếu Kaggle thay đường dẫn/metadata nguồn giữa các phiên, cache có thể không khớp dù tên video giống nhau. Khi đó không đổi tên cache để ép dùng; kiểm tra input/version, hoặc trích xuất lại. Không chạy hai tiến trình ghi cùng một thư mục cache đồng thời.

## Bước 3 — Ghép dataset

Import:

`https://raw.githubusercontent.com/toafn11/driver-monitoring-system/main/kaggle_02_build20.ipynb`

Chọn CPU, Internet On. Attach:

- RLDD Full và model như bước 1.
- Output của đủ 4 nhóm, cộng các output chứa cache gốc nếu từng chạy tiếp.

Đặt `SAMPLE_FPS` giống bước 2. Chọn Save Version → Save & Run All.

Notebook yêu cầu đủ dấu complete của 4 nhóm đúng phiên bản. Sau đó chỉ đọc cache; nếu thiếu hoặc không khớp sẽ báo lỗi, không bất ngờ chạy MediaPipe nhiều giờ.

Với 10 FPS: mỗi chuỗi có 40 frame, cửa sổ danh nghĩa 4 giây, bước trượt 10 frame. Dữ liệu giữ tách người: fold 1–3 train, fold 4 validation, fold 5 test.

Output hoàn chỉnh phải có `processed20_v3/metadata.json`, `manifest.csv`, `train.npz`, `val.npz`, `test.npz`. Chỉ khi có metadata hoàn tất mới chuyển sang train.

Nếu ghép lỗi giữa chừng trong cùng phiên, chọn tên DATA mới, ví dụ `processed20_v3_retry`; khi dùng tên khác phải sửa bộ tìm thư mục trong notebook 03 cho khớp. Không xóa cache để sửa lỗi thư mục dataset không rỗng.

## Bước 4 — Train model

Import:

`https://raw.githubusercontent.com/toafn11/driver-monitoring-system/main/kaggle_03_train20.ipynb`

Attach **chỉ một** output dataset hoàn chỉnh từ bước 3. Không cần gắn video gốc hoặc Face Landmarker ở notebook train.

Chọn GPU nếu còn hạn mức và Internet On. Chạy Save Version → Save & Run All. Đầu log phải xác nhận `Device: cuda` nếu dùng GPU; nếu hiển thị CPU, kiểm tra cấu hình phiên và bản PyTorch trước khi chạy dài.

Notebook train GRU12 rồi GRU20, tối đa 40 epoch, dừng sớm theo validation. Hai model dùng cùng chuỗi, người tham gia và nhãn. Mỗi lần chạy tạo thư mục thí nghiệm mới để không nhầm checkpoint của một lần chạy dở là hoàn tất. Chưa hỗ trợ tiếp tục optimizer từ epoch dở.

Giữ `RUN_FINAL_TEST = False` trong giai đoạn chọn model. So sánh macro-F1 và recall DROWSY, không chỉ accuracy. Chưa thể kết luận 20 đặc trưng tốt hơn chỉ dựa vào số lượng đặc trưng; cần lặp lại với các seed khác sau khi pipeline ổn định.

## Bước 5 — Lưu model và đánh giá cuối

Tải/lưu toàn bộ thư mục `experiment_...`, đặc biệt:

- `gru12_seed42/best.pt`, `gru20_seed42/best.pt`.
- `history.json` và các cấu hình/báo cáo đi kèm trong từng thư mục.
- `complete.json` xác nhận cả hai lượt train đã chạy xong.
- `processed20_v3/metadata.json` để giữ đúng FPS, số frame và chuẩn hóa.

Sau khi chốt model bằng validation, bật cell test trong phiên còn biến DATA/CHOSEN, hoặc tạo notebook đánh giá riêng: attach dataset bước 3 và output model bước 4, đặt DATA và CHOSEN đến đúng đường dẫn rồi gọi `evaluate`. Không chạy lại cell train chỉ để lấy biến. Giữ test ngoài vòng chỉnh model.

Model này mới phân biệt ALERT/DROWSY theo nhãn video; không chứng minh nhận dạng ngáp hoặc mất tập trung. `live_demo.py` cũ chưa đọc checkpoint/schema mới. Chỉ tích hợp camera sau khi có kết quả và kiểm tra preprocessing khớp.

## Trạng thái bàn giao

Đã kiểm thử bằng video tổng hợp và model mô phỏng cho các trường hợp FPS thấp, tiếp tục sau lỗi, đọc cache từ input, tách người và train/evaluate nhỏ. Chưa chạy lại trên toàn bộ RLDD sau bản sửa này. Không có số accuracy mới được khẳng định.
