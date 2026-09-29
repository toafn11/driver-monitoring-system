# Chạy lại RLDD sau lỗi FPS — hướng dẫn thao tác

## Khi notebook mới vẫn báo nguyên lỗi cũ

Notebook hiện đã bổ sung kiểm tra hash mã và đường dẫn module, đồng thời chạy audit/trích xuất bằng tiến trình Python mới. Chỉ đổi `REV` hoặc import notebook trong cùng kernel trước đây chưa đủ: Python có thể giữ `training.preflight20` cũ trong `sys.modules`.

Import lại notebook 01 đã cập nhật, chạy từ cell đầu với `RUN_EXTRACTION=False`. Cell đầu phải in `Code revision VERIFIED: a637d62`, ba dòng `Verified module` có đường dẫn thuộc `/kaggle/working/code_a637d62/`, và `Verified timestamp version: ffprobe-best-effort-v1`. Nếu không có các dòng này thì chưa chạy bootstrap mới. Các notebook 02/03 cũng đã cập nhật bootstrap; dùng bản mới khi đến bước đó.

Nếu audit vẫn thất bại, gửi log mới cùng mục video trong inventory. Không bỏ dòng chặn lỗi: với đúng mã mới, trường hợp thiếu frame sẽ có `ffprobe_audit` hoặc thông báo lỗi FFprobe/không khớp số frame, thay vì chỉ lỗi thiếu frame của bản 5a28ee8. Bản sửa xử lý việc nạp mã cũ, không khẳng định mọi video Kaggle đã qua kiểm tra.

## Đã sửa gì?

Log cũ dừng sau 35 video vì video người 18 có FPS khoảng 12,002, thấp hơn cấu hình 15 FPS. Chưa có epoch train nào hoàn thành trong log đó.

Pipeline mới kiểm tra toàn bộ video trước MediaPipe, lưu cache nguyên tử theo video, ghi tiến độ từng nhóm và giữ các video đã thành công khi video khác lỗi. Không tự bỏ video lỗi để train một tập thiếu dữ liệu.

Ba notebook thay thế bản một lượt `kaggle_train20.ipynb`:

1. `kaggle_01_extract20.ipynb`: kiểm tra, trích xuất theo 4 nhóm.
2. `kaggle_02_build20.ipynb`: ghép cache thành các tập train/validation/test.
3. `kaggle_03_train20.ipynb`: huấn luyện và so sánh 12/20 đặc trưng.

Mã hiện tại được ghim ở commit `a637d62`; không tự lấy thay đổi mới từ main. Notebook lấy mã từ GitHub, nên cần Internet On. Kiểm tra tên notebook và các input trước khi chạy.

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

Kiểm tra nhanh vẫn không chứng nhận mọi frame đều nguyên vẹn. Nếu số frame OpenCV thấp hơn metadata, bản mới gọi FFprobe: chỉ chấp nhận chênh lệch khi hai bộ đọc cùng đếm được số frame thực tế, FFprobe không báo lỗi và timestamp tăng liên tục. Chênh lệch được giữ trong warnings, không xóa khỏi báo cáo.

Khi trích xuất, mọi video được FFprobe đọc timestamp từng frame; OpenCV đọc tuần tự và phải khớp số frame. Lấy mẫu dựa trên timestamp thật, không lấy chỉ số frame chia FPS trung bình. Timestamp thiếu/trùng/lùi hoặc hai bộ đọc không khớp sẽ bị chặn. Không nhân đôi frame; cửa sổ có khoảng trống thời gian lớn bị bộ lọc chất lượng loại. Hai bộ đọc không đảm bảo phát hiện mọi lỗi hình ảnh bị decoder che giấu.

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

- `cache20_v4/`: các file `.npz` đã trích xuất.
- `cache20_v4/progress_N.json`: trạng thái từng video, mọi mục là `ok`.
- `shard_N_complete.json`: dấu xác nhận nhóm hoàn thành.

Không có dấu complete thì chưa xem là nhóm hoàn tất. Nếu có video lỗi, pipeline xử lý tiếp các video khác rồi báo thất bại; cache thành công vẫn được giữ tại thư mục output. Kiểm tra Kaggle có lưu được output của phiên thất bại hay không, rồi tải/lưu các file đó trước khi tạo phiên mới.

### Khi cần chạy tiếp

1. Gắn output của phiên trước qua **Add Input → Notebook output**; nếu output phiên lỗi không dùng được trực tiếp, tạo dataset riêng tư từ cache tải về.
2. Giữ nguyên thư mục `cache20_v4` khi đóng gói. Notebook tự tìm thư mục này trong Input.
3. Giữ nguyên input video, model, commit mã, FPS và chỉ số nhóm; chạy lại cùng nhóm.
4. File cache khớp sẽ được đọc từ input, video thiếu sẽ trích xuất vào working. Cache không khớp thì không tái sử dụng.
5. Khi ghép, attach cả output cũ lẫn output chạy tiếp: output chạy tiếp không sao chép các cache đã đọc từ input.

Khóa cache có phiên bản timestamp mới và kiểm tra đường dẫn video, kích thước, thời điểm sửa file, FPS, hash model và hash bộ trích xuất khuôn mặt. Nếu Kaggle thay đường dẫn/metadata nguồn giữa các phiên, cache có thể không khớp dù tên video giống nhau. Khi đó không đổi tên cache để ép dùng; kiểm tra input/version, hoặc trích xuất lại. Không chạy hai tiến trình ghi cùng một thư mục cache đồng thời.

## Bước 3 — Ghép dataset

Import:

`https://raw.githubusercontent.com/toafn11/driver-monitoring-system/main/kaggle_02_build20.ipynb`

Chọn CPU, Internet On. Attach:

- RLDD Full và model như bước 1.
- Output của đủ 4 nhóm, cộng các output chứa cache gốc nếu từng chạy tiếp.

Đặt `SAMPLE_FPS` giống bước 2. Chọn Save Version → Save & Run All.

Notebook yêu cầu đủ dấu complete của 4 nhóm đúng phiên bản. Sau đó chỉ đọc cache; nếu thiếu hoặc không khớp sẽ báo lỗi, không bất ngờ chạy MediaPipe nhiều giờ.

Với 10 FPS: mỗi chuỗi có 40 frame, cửa sổ danh nghĩa 4 giây, bước trượt 10 frame. Dữ liệu giữ tách người: fold 1–3 train, fold 4 validation, fold 5 test.

Output hoàn chỉnh phải có `processed20_v4/metadata.json`, `manifest.csv`, `train.npz`, `val.npz`, `test.npz`. Chỉ khi có metadata hoàn tất mới chuyển sang train.

Nếu ghép lỗi giữa chừng trong cùng phiên, chọn tên DATA mới, ví dụ `processed20_v4_retry`; khi dùng tên khác phải sửa bộ tìm thư mục trong notebook 03 cho khớp. Không xóa cache để sửa lỗi thư mục dataset không rỗng.

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
- `processed20_v4/metadata.json` để giữ đúng FPS, số frame và chuẩn hóa.

Sau khi chốt model bằng validation, bật cell test trong phiên còn biến DATA/CHOSEN, hoặc tạo notebook đánh giá riêng: attach dataset bước 3 và output model bước 4, đặt DATA và CHOSEN đến đúng đường dẫn rồi gọi `evaluate`. Không chạy lại cell train chỉ để lấy biến. Giữ test ngoài vòng chỉnh model.

Model này mới phân biệt ALERT/DROWSY theo nhãn video; không chứng minh nhận dạng ngáp hoặc mất tập trung. `live_demo.py` cũ chưa đọc checkpoint/schema mới. Chỉ tích hợp camera sau khi có kết quả và kiểm tra preprocessing khớp.

## Trạng thái bàn giao

Đã kiểm thử bằng video tổng hợp và model mô phỏng cho các trường hợp FPS thấp, tiếp tục sau lỗi, đọc cache từ input, tách người và train/evaluate nhỏ. Chưa chạy lại trên toàn bộ RLDD sau bản sửa này. Không có số accuracy mới được khẳng định.


## Cập nhật 29/09 — Decode probe failed

Bản `5a28ee8` đọc tuần tự lại video khi kiểm tra nhảy frame thất bại. Chỉ khi số frame giải mã đạt độ dài khai báo (cho phép sai số làm tròn tối đa 1 frame) mới chuyển lỗi nhảy frame thành warning. Nếu kết thúc sớm, vẫn chặn và ghi `sequential_audit` trong inventory; không tự bỏ video. Cách này không phát hiện mọi trường hợp bộ giải mã che giấu frame hỏng và chưa chứng nhận VFR.

Nếu notebook hiện in `Code revision: 8aeacc1`:

1. Trong cell đầu, đổi `REV = '8aeacc1'` thành `REV = '5a28ee8'` ở cả ba notebook. Hoặc import lại các notebook mới từ GitHub theo đường dẫn bên trên.
2. Restart session/kernel trước khi chạy lại để Python không dùng module cũ trong bộ nhớ. Không chọn Factory reset để tránh xóa output chưa lưu.
3. Giữ `RUN_EXTRACTION = False`, `SAMPLE_FPS = 10` và chạy lại notebook 01.
4. Chờ đọc tuần tự các video gặp lỗi; sẽ có log `Seek probe failed ... checking sequential decode`. Bước này không gọi MediaPipe, nhưng vẫn phải giải mã video nên có thể mất vài phút.
5. Nếu `failed = 0`, tiếp tục bước 2 của hướng dẫn. Nếu vẫn lỗi, xem/gửi mục `issues` và `sequential_audit` của video đó trong `inventory20.json`; chưa bật trích xuất.

Cảnh báo xung đột dopamine-rl/gym trong log không phải lỗi dừng preflight này. Không cần hạ gym để xử lý lỗi video. Kiểm tra đã dùng đúng revision mới trước khi thử tiếp.


## Bản hiện tại a637d62 — tiếp tục từ lỗi video người 50

Phần cập nhật 5a28ee8 bên trên là lịch sử; dùng a637d62 cho lượt chạy mới.

1. Import lại **cả ba notebook** từ các URL main bên trên. Nên giữ bản cũ để tham khảo nhưng không chạy trộn các phiên bản.
2. Restart kernel/session, không Factory reset. Cell đầu phải in `Code revision: a637d62`.
3. Notebook 01: giữ `RUN_EXTRACTION = False`, `SAMPLE_FPS = 10`. Run All để kiểm tra.
4. Với video người 50, nếu kiểm tra trên Kaggle khớp log đã cung cấp (18.174 frame và timestamp hợp lệ), lỗi metadata sẽ trở thành WARNING. Chỉ tiếp tục khi tổng `failed = 0`.
5. Đặt `RUN_EXTRACTION = True`, chạy nhóm 0 trước. Khi nhóm 0 thành công, chạy tiếp nhóm 1, 2, 3. Giữ `NUM_SHARDS = 4`.
6. Dùng output các nhóm để chạy notebook 02, rồi output dataset để chạy notebook 03 theo các bước phía trên.

Cache/dataset mới mang tên `cache20_v4` và `processed20_v4`. Không tái sử dụng cache từ bản tính thời gian bằng FPS trung bình, kể cả cache 10 FPS; khóa cache mới chủ động không khớp bản cũ. Không cần xóa cache cũ.

FFprobe sẽ giải mã thêm một lượt để lấy timestamp trước MediaPipe, nên có chi phí thời gian bổ sung. Không hứa bản này nhanh hơn bản trước; mục tiêu là tính đúng đặc trưng thời gian và xử lý chênh lệch metadata có bằng chứng. Các trường hợp lỗi được giữ báo cáo thay vì tự bỏ video.

Kiểm thử cục bộ dùng video tổng hợp, detector mô phỏng và phản hồi FFprobe mô phỏng (máy kiểm thử không có ffprobe). Chưa chạy lại bản này trên toàn bộ RLDD/Kaggle. Sau bước audit và nhóm 0, cần kiểm tra output thực tế trước khi kết luận mọi video đều xử lý được.
