# Phục hồi nhóm 0: giữ 30 cache, trích xuất bổ sung video 11/0.mp4

Chẩn đoán đã nhận: 18.818 frame, không thiếu timestamp, không có timestamp đi lùi, đúng một cặp trùng tại frame 15026–15027, PTS 501.9471 giây, FFprobe không báo lỗi. Bản mã hiện tại `04aa1f5` chỉ chấp nhận ngoại lệ nếu toàn bộ các điều kiện này khớp đúng video. Nó bỏ frame thứ hai của cặp trùng khi lấy mẫu; không sửa thời gian hoặc nhân đôi frame. Điều này không khẳng định hai frame có nội dung ảnh giống nhau. Các bất thường khác vẫn bị chặn.

## Bạn làm theo thứ tự

1. Tạo dataset **Private** từ `D:\hard\driver_monitoring_recovery\rldd_shard0_cache_recovery.zip`. Giữ thư mục `cache20_v4` trong dataset, bên trong có 30 file `.npz`.
2. Tạo notebook mới, File → Import Notebook → Link:

   `https://raw.githubusercontent.com/toafn11/driver-monitoring-system/main/kaggle_recover_shard0.ipynb`

3. Attach đúng ba nguồn: `thuannn18022005/uta-rldd-full`, model `driver-monitoring-face-landmarker-v1`, và dataset cache phục hồi vừa tạo.
4. Chọn CPU (Accelerator None), Internet On. Notebook phục hồi đã đặt `RUN_EXTRACTION=True`, nhóm 0/4, FPS=10; không đổi các giá trị này.
5. Chọn Save Version → Save & Run All. Đầu log phải có `Code revision VERIFIED: 04aa1f5`.
6. Trước khi trích xuất, phải có `RECOVERY VERIFIED: all 30 previous video caches are readable; no re-extraction required`. Nếu không đọc đủ cache, notebook dừng ngay; không đổi guard hoặc xóa cache để chạy tiếp. Kiểm tra đã attach đúng dataset, đúng thư mục và phiên bản video/model.
7. Log vẫn liệt kê 31 video, nhưng 30 video đọc cache và chỉ video thiếu chạy FFprobe/MediaPipe. Audit đầu phiên vẫn cần thời gian kiểm tra video. Không cam kết thời gian hoàn thành khi chưa đo trên Kaggle.
8. Thành công phải có `shard_0_complete.json`. Output mới chủ yếu chứa cache của video bổ sung; **30 cache cũ vẫn nằm ở input**, không được tự sao chép sang output mới.

## Khi ghép dữ liệu ở notebook 02

Dùng notebook 02 mới từ main. Attach **dataset 30 cache cũ + output phục hồi nhóm 0 + output nhóm 1,2,3**, cùng RLDD Full và Face Landmarker. Không chỉ attach output phục hồi vì như vậy sẽ thiếu 30 cache cũ.

Notebook 02 mới chấp nhận dấu hoàn thành của các bản timestamp `a637d62`, `a64400b` và `04aa1f5`; không nhận cache thời gian cũ trước schema timestamp. Các nhóm 1–3 đã hoàn thành trên a637d62 không phải trích xuất lại. Nếu chưa chạy chúng, dùng notebook 01 mới trên main.

Khóa cache của 30 video không thay đổi. Riêng video có ngoại lệ timestamp có khóa riêng. Nếu đường dẫn, mtime, file model hoặc nguồn video thay đổi trên Kaggle thì cache cũ có thể không khớp; guard sẽ chặn để tránh vô tình chạy lại toàn bộ.

Đã kiểm thử cục bộ 24 bài; chưa chạy notebook phục hồi thực tế trên tài khoản Kaggle. Không tạo dấu complete bằng tay và không bỏ video khỏi tập test để vượt kiểm tra.


## Cập nhật phục hồi khi Missing matching cache

Bản 04aa1f5 sửa thêm trường hợp khóa cache không khớp sau khi Kaggle thay metadata nguồn. Cache gốc không lưu đủ thông tin để xác định trực tiếp khóa cũ thuộc video nào; không được đổi tên file theo phỏng đoán.

Notebook phục hồi mới:

- Đối chiếu `progress_0.json`, số frame và tỷ lệ mặt hợp lệ để lọc ứng viên.
- Đọc toàn bộ timestamp video bằng FFprobe; yêu cầu mọi timestamp đã lấy mẫu khớp với cache và chỉ còn một ứng viên duy nhất.
- Đối chiếu lại tối đa 32 frame có mặt trên toàn video với MediaPipe (sai số tuyệt đối 0,0002). Đây là kiểm tra lấy mẫu, không phải chứng minh mọi đặc trưng trong cache đều đúng.
- Chỉ sau khi các bước đạt mới ghi cache theo khóa hiện tại, kèm báo cáo provenance và SHA-256 toàn bộ video. Không thay đổi cache gốc.
- Các cache mới có chữ ký và checksum nguồn; phiên sau chỉ được bỏ qua khác biệt mtime nếu checksum nội dung video khớp. Nếu kiểm tra không đạt thì dừng, không tự nới sai số hoặc chạy trích xuất lại cả nhóm.

Bước xác minh này vẫn đọc/giải mã video, nên có thể mất hàng chục phút hoặc hơn tùy Kaggle; không chỉ chạy video thiếu ngay lập tức. Tuy vậy, chỉ chạy MediaPipe trên tối đa 32 frame/video để nhận diện cache, thay vì toàn bộ hàng nghìn frame. Nhật ký `Legacy recovery N/30` và `Restored N/30` cho biết tiến độ. Mỗi cache phục hồi được lưu ngay; giữ output nếu phiên bị gián đoạn.

Dùng notebook recovery mới và notebook 02 mới trên main. Chỉ đổi REV trong notebook cũ sẽ làm hash xác minh không khớp. Sau khi di chuyển thành công cả 30 cache, guard cũ tiếp tục xác nhận rồi mới trích xuất video 11/0.mp4. Những cache được di chuyển sẽ có mặt trong output mới. Vẫn giữ dataset cache cũ và attach cùng các output khi ghép để tránh thiếu file nếu một số cache đã được dùng trực tiếp từ input.
