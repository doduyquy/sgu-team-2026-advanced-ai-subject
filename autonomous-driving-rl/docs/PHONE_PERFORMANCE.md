# Tối ưu host điện thoại — kiểm tra ngày 20/11/2026

Đã triển khai chế độ người chơi với AI chạy ở tiến trình riêng, vật lý bước cố
định 0,02 giây, bù tối đa ba bước khi render chậm, chụp hình có giới hạn nhịp
và H.264 phần cứng. Cơ chế so sánh AI lockstep giữ riêng.

## RTX 3050 Ti — phiên bản hiện tại

Đã chuyển mặc định sang **h264_nvenc**. Driver NVIDIA 566.07 được giữ nguyên;
FFmpeg 7.1 trong imageio-ffmpeg 0.6.0 mở NVENC thành công khi bộ codec của
PyAV 17 không tương thích API. Render và mã hóa đều dùng NVIDIA. Không tự
chuyển sang Intel hay CPU ở chế độ mặc định.

Đo 10 giây sau 2 giây khởi động, đường Easy/SCCS, một người chơi cùng năm AI
còn hoạt động trong toàn bộ phép đo:

| Chỉ số | RTX NVENC |
|---|---:|
| Camera 960×540 | 29.2 FPS |
| Bước vật lý / giây | 50.4 |
| Giây mô phỏng / giây thực | 1.007 |
| Áp dụng lệnh tại host, trung bình / P95 | 15.34 / 30.6 ms |
| Mã hóa, trung bình / P95 | 7.78 / 15.1 ms |
| Lỗi kỹ thuật / chuyển sang Intel hoặc CPU | 0 / 0 |

Renderer báo **NVIDIA GeForce RTX 3050 Ti Laptop GPU/PCIe/SSE2**; backend
video báo **h264_nvenc / ffmpeg_nvenc**. Dữ liệu:
`artifacts/phone_optimization/rtx-verified/performance.json`.
Đã giải mã 80/80 khung NVENC ở 360p, 540p và 720p, đổi độ phân giải trong
cùng một phiên; bài kiểm tra hai peer WebRTC thật cũng qua với NVENC.

Các số này đo tại laptop, chưa phải phép đo độ trễ đầu-cuối trên iPhone thật.
Hai lượt QSV và NVENC chạy ở thời điểm khác nhau nên không dùng để kết luận
mức tăng tốc chỉ do thay encoder.

## Phép đo trên máy phát triển

Cùng một cửa sổ native, cùng đường TRAIN Easy/SCCS, một xe người chơi và
cả năm AI còn hoạt động trong toàn bộ khoảng lấy mẫu. Mỗi pha đo 8 giây sau
2 giây khởi động. Cả hai pha cùng dùng encoder QSV, nên bảng này chủ yếu
so sánh kiến trúc vòng lặp và việc tách AI, không phải phép đo QSV so với x264.

| Chỉ số | Lockstep cũ | Chế độ tối ưu |
|---|---:|---:|
| FPS camera mới | 7.3 | 30.0 |
| Thời gian áp dụng lệnh tại host, trung bình | 115.85 ms | 18.23 ms |
| Thời gian áp dụng lệnh tại host, P95 | 150.9 ms | 32.5 ms |
| Bước vật lý / giây thực | 7.6 | 50.2 |
| Giây mô phỏng / giây thực | 0.757 | 1.004 |

Lần đo ban đầu: render **NVIDIA GeForce RTX 3050 Ti Laptop GPU/PCIe/SSE2**,
encoder **h264_qsv**.
Trong pha tối ưu, mã hóa trung bình 8.47 ms, P95
16.8 ms. NVENC không mở được với driver 566.07/API
12.2 và bộ mã hóa PyAV yêu cầu API 13.0; bản cũ đã chọn Quick Sync.
Đây là dữ liệu lịch sử trước khi chuyển mặc định sang NVENC.

Đây là kiểm tra ngắn trên một đường, không phải benchmark chất lượng lái AI
hoặc chứng nhận FPS ở mọi cấu hình. Thời gian áp dụng lệnh tính từ lúc host
nhận mẫu đầu vào đến khi đưa vào bước vật lý, không gồm mạng, giải mã và màn
hình iPhone. FPS camera không đồng nghĩa FPS giải mã ở điện thoại. Khi thử
trình duyệt trên cùng laptop, trình duyệt còn chia sẻ tài nguyên với mô phỏng.

Dữ liệu thô: `artifacts/phone_optimization/verified/performance.json`.
Lặp lại: `python scripts/benchmark_phone_runtime.py --output artifacts/phone-perf-new --seconds 12`.

## Chi tiết thực thi

- Mỗi AI có một tiến trình giữ trạng thái, một yêu cầu đang tính và một snapshot
  chờ mới nhất. Thread đọc kết quả tránh dò named pipe trên luồng render.
- LiDAR/quan sát đắt được lấy mới đúng lúc AI cần quyết định; kiểm tra va chạm,
  trạng thái xe và vật lý vẫn cập nhật trong vòng mô phỏng.
- Hành động AI quá 350 ms được thay bằng phanh. Xe người chơi có deadman
  450 ms. Các quyết định và hành động áp dụng được ghi riêng.
- GPU render camera theo nhịp mục tiêu 30 FPS; đọc RGB về RAM theo yêu cầu,
  frame mailbox chỉ giữ ảnh mới nhất. Điều kiện đánh thức encoder thay thế
  việc polling bằng asyncio; đồng hồ dùng perf_counter độ phân giải cao.
- Video dùng H.264 Baseline, không B-frame, keyframe ngắn và đàm phán codec
  trước khi áp dụng SDP offer. Packet đã mã hóa đi qua API media của aiortc.
- Mặc định NVENC nghiêm ngặt; nếu PyAV không khớp SDK/driver, dùng NVENC qua
  FFmpeg 7.1 được ghim ở imageio-ffmpeg 0.6.0, không chuyển sang Intel/CPU.
  Auto chỉ thử NVIDIA rồi CPU. QSV chỉ dùng khi người chạy chọn rõ qua CLI.
  Mã hóa chạy ở thread riêng; đổi độ phân giải khởi tạo lại codec với header mới.
  Có kiểm tra NVENC trước khi mở QR và đóng tiến trình khi ngắt kết nối.
- Chất lượng Auto / Mượt / Cân bằng / Rõ nét, có điều chỉnh tự động trong
  menu **···**. Menu hiện FPS giải mã, RTT, GPU/CPU và độ phân giải.
- Mọi phiên người chơi async đều `canonical_run=false`. Chạy đơn/so sánh AI
  giữ cơ chế cũ; `--legacy-phone` dùng để chẩn đoán đối chiếu.

## Kiểm tra

23 kiểm thử race/MPPI, 10 kiểm thử điện thoại (có hai peer WebRTC thật),
8 kiểm thử worker/encoder/realtime và 6 kiểm thử JavaScript xoay màn hình đều qua.
Đã kiểm tra giải mã H.264, đổi độ phân giải, điều khiển độc quyền, phanh khi
lệnh cũ, tiến trình AI không chặn, ảnh khác nhau cho ba mẫu xe và các góc nhìn.
Safari trên iPhone thật cần kiểm tra lại FPS và cảm giác lái sau khi kết nối.

## Chỉnh màn hình điện thoại ngày 21/11/2026

Giảm kích thước nút theo chiều cao nhìn thấy thực tế và chuyển tên xe,
luật, thông tin luồng hình vào menu. Tốc độ, thứ hạng và tiến độ ở trên;
camera bám sau nhìn vào độ cao 0,9 m để hiện trọn thân xe. Các ô chọn dùng
chữ 16 px để tránh Safari tự phóng khi chọn. Luật/vật lý không thay đổi.

Auto bắt đầu 960×540 / 3,5 Mbps, được nâng tới 1280×720 / 6 Mbps khi FPS,
RTT và thời gian mã hóa ổn định; có hạ từng nấc khi kém liên tiếp. Chọn HD
thủ công giữ nguyên lựa chọn. Không thay encoder hay cài driver mới.

25 kiểm tra Python và 19 JavaScript đạt. Chromium kiểm tra sáu viewport
320–1024 px, bao gồm ngang cao 250/254/290 px: nút chạm ít nhất 44 px,
không chồng nhau, không che vùng xe ở giữa, giữ trái cùng ga và mở menu
thì phanh đúng. Nguồn: `artifacts/phone_clarity_20261121/layout/`.

Kiểm tra MetaDrive thật với Cybertruck và năm AI, hai peer WebRTC trên
cùng laptop: Auto tự lên 720p, giải mã khoảng 29 FPS, NVENC qua FFmpeg,
mã hóa trung bình/P95 13,67/21,37 ms ở mẫu cuối. Khung mới nhất 18,1 ms
tại host; con số này không phải độ trễ đầu-cuối. Đổi HD thủ công và xoay
dọc vẫn nhận hình. Bằng chứng: `artifacts/phone_clarity_20261121/native-final/`.
Chưa kiểm tra bản sửa trên Safari/điện thoại vật lý qua Wi-Fi.
