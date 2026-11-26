# Đồ họa nhẹ cho máy local

Race Mode mặc định dùng `--graphics lite`, phù hợp để bắt đầu trên laptop
RTX 3050 Ti 4 GB. Hai file `CHAY_DUA_XE.cmd` và `CHAY_DIEN_THOAI.cmd` đều
nhận cấu hình này qua launcher Python, không cần sửa file CMD.

## Sửa giật map Việt Nam ngày 21/11/2026

Launcher mặc định trở lại **Lite** và chỉ cho mở một phiên Race Mode trong
cùng checkout, tránh hai cảnh 3D cùng chiếm VRAM. Lite giữ chữ ở độ phân giải
cửa sổ, khử răng cưa 2×, cảnh 3D 75% và tối đa 829.440 pixel khi phóng to.
Balanced vẫn cho phép vẽ cảnh ở độ phân giải cửa sổ.

LOD cây và mặt tiền nhà hiện đo khoảng cách trước khi phóng to model.
Trước đây cây lớn 2× cũng làm tăng khoảng cách vẽ chi tiết, khiến rất nhiều
cây xa dùng mesh gần. Bản sửa giữ lá và vật liệu ở gần, giảm mesh/lá và bóng
cây ở xa; đồng thời bỏ phần debug chữ MetaDrive vốn dựng lại mỗi bước dù
HUD riêng đã che nó.

Phép đo ngắn trên RTX 3050 Ti, map Việt Nam, đủ năm AI:

| Lượt đo | Cửa sổ | FPS | Khoảng khung hình P95 |
|---|---:|---:|---:|
| Trước sửa, Balanced, bật CPU profiler | 1280 × 800 | 28,98 | 42,91 ms |
| Sửa LOD, Balanced, bật CPU profiler | 1280 × 800 | 38,99 | 33,36 ms |
| Bản cuối, Lite, tắt CPU profiler | 1920 × 1000 | 49,93 | 24,79 ms |

Hai dòng đầu dùng cùng điều kiện để so sánh LOD. Dòng cuối đo trải nghiệm
thực trong 10 giây sau khởi động, đạt vật lý 50 Hz và tỷ lệ thời gian mô
phỏng/thực 1,003; cả năm worker hoạt động, không lỗi kỹ thuật. Ảnh đã được
kiểm tra có đầy đủ cảnh 3D; render target là 1260 × 651. Profiler có
chi phí riêng, nên không dùng dòng cuối để tính tỷ lệ tăng tốc với baseline.
Đây là số đo trên máy hiện tại, chưa phải bảo đảm FPS toàn bộ tuyến hoặc
điện thoại thật. Báo cáo: `artifacts/vn-performance/{before,lod-fixed,scene-verified}/`.
Hai mẫu `fullscreen-final` và `fullscreen-verified` bị loại do cảnh 3D đen
trong quá trình sửa resize; không dùng số FPS của chúng.

Lặp lại từ thư mục `autonomous-driving-rl` khi đã đóng game:

```powershell
..\.venv\metadrive\python.exe -B scripts/benchmark_vietnam_runtime.py --graphics lite --size 1920x1000 --seconds 10 --output artifacts/vn-performance-new
```

81 kiểm tra về đồ họa, race, điện thoại, đường Việt Nam và luật đua đã qua;
kiểm tra launcher xác nhận lần mở thứ hai thoát mà không tạo thêm cảnh 3D.
Các số đo cũ bên dưới được giữ để tham khảo theo ngày.

Từ thư mục gốc repository:

```powershell
.\CHAY_DUA_XE.cmd
# Muốn cảnh 3D rõ hơn ở độ phân giải cửa sổ:
.\CHAY_DUA_XE.cmd --graphics balanced
```

| Thiết lập | Lite mặc định | Balanced |
|---|---:|---:|
| Cửa sổ và chữ giao diện | 1280 × 800 | 1280 × 800 |
| Cảnh 3D tại kích thước cửa sổ trên | 960 × 600 | 1280 × 800 |
| Garage render riêng | 768 × 300 | 1024 × 400 |
| MSAA | 2× | 2× |
| Cạnh dài nhất của texture xe | 512 px | 1024 px |
| Trần render garage | 30 FPS | 30 FPS |
| Trần render chạy đơn/so sánh | 60 FPS | 60 FPS |

Vòng realtime điện thoại giữ trần render 30 FPS hiện có. Các mức FPS là
giới hạn trên, không phải cam kết hiệu năng. Cảnh 3D tự giữ tỷ lệ khi resize
cửa sổ; chữ và các nút vẫn vẽ ở độ phân giải cửa sổ.

## Những gì được tối ưu

- Gom các mảnh thân xe cố định thành ba Geom/material; giữ vertex color
  riêng cho từng thuật toán. Geometry mẫu dùng chung qua cache theo mẫu/màu.
- Từ khoảng cách camera 55 m, đổi sang thân đơn giản 40–44 tam giác, hai Geom.
  Camera nhìn từ trên cao mặc định khoảng 40 m vẫn thấy thân chi tiết.
- Chỉ sao chép, giảm kích thước và yêu cầu GPU nén texture trên phần bánh xe
  đang dùng ảnh. Ba thân xe tự dựng dùng vertex color, không có ảnh để nén.
- Giới hạn đầu ra render bằng task trước `igLoop`; mọi task, xử lý input,
  vật lý, kiểm tra va chạm và tính toán AI vẫn chạy theo vòng hiện có.
- Không áp dụng nén/giảm kích thước toàn cục cho texture. MetaDrive đọc một
  số texture thành mảng số cho terrain; thay đổi chúng gây lỗi và có thể
  thay đổi dữ liệu mô phỏng. Chỉ bản sao texture mỹ thuật được tối ưu.

Khung Bullet, khối lượng, vị trí/tâm bánh và các thông số vận hành giữ nguyên.
Các mẫu hiện tại vẫn là **ba ngoại hình tự dựng** trong
`race_vehicle_models.py`, được ghi `vehicle_asset_source=procedural_original`
trong `session.json`. Chưa nhập model xe thật từ bên ngoài;
[Lamborghini của Sidra](https://sketchfab.com/3d-models/lamborghini-huracan-low-poly-3d-2524841912a74e6ab13d99b4da98ce79)
cần mua, còn [Download API của Sketchfab](https://sketchfab.com/developers/download-api/downloading-models)
cần xác thực tài khoản.

## Phép đo ngày 20/11/2026

GPU render thực tế: **NVIDIA GeForce RTX 3050 Ti Laptop GPU**.

So sánh cùng đường Easy/SCCS, camera mặc định, năm xe đứng tại lưới xuất phát.
Chỉ chạy render, không gọi policy hay tiến vật lý. Tạm bỏ trần render cho phép
đo này. Baseline trước sửa lấy mẫu 6 giây; bản cuối Lite lấy mẫu 8 giây sau
30 khung khởi động. Thực tế có **0 xe giao thông** trong snapshot này.

| Chỉ số | Trước sửa | Lite |
|---|---:|---:|
| Trung bình mỗi vòng render | 24.391 ms | 9.606 ms |
| P95 mỗi vòng render | 39.359 ms | 12.998 ms |
| Vòng render/giây khi bỏ trần | 40.99 | 104.06 |
| Geom thân Lamborghini gần camera | 136 | 3 |
| Geom thân Tesla gần camera | 112 | 3 |
| Geom thân VF3 gần camera | 156 | 3 |

Thời gian trung bình vòng render giảm khoảng **60.6%** trong phép đo ngắn này.
Tải nền chưa được cô lập: một phiên host điện thoại khác bắt đầu chạy sau
baseline và cùng dùng GPU trong khoảng đo Lite. Vì vậy số chênh lệch này chỉ
là quan sát tham khảo, không phải so sánh có kiểm soát hoàn toàn.
Đây là thời gian vòng native render đo trên CPU, gồm chờ pipeline; chưa phải
GPU timestamp riêng hoặc FPS toàn cuộc đua. Geom ở bảng chỉ tính thân xe,
không tính bánh, nhãn hay đường. Cả hai LOD được giữ trong RAM nhưng chỉ một
LOD được vẽ cho mỗi camera. Chưa đo được mức giảm VRAM riêng của ứng dụng.

Kiểm tra realtime riêng trên cùng máy: một người chơi + năm AI, 6 giây đo
sau 2 giây khởi động, đầu vào điều khiển tổng hợp, encoder H.264 QSV:

- Vật lý 49.8 bước/giây thực; tỷ lệ thời gian mô phỏng/thực 0.997.
- Camera host 29.9 FPS, 960 × 540.
- Cả năm AI hoạt động trong khoảng lấy mẫu; không có lỗi kỹ thuật.
- Có 322 lượt phanh bảo vệ do AI trả chậm/chưa có kết quả. Tối ưu GPU không
  loại bỏ giới hạn CPU/policy. Đây không phải đánh giá chất lượng lái.

Dữ liệu và ảnh nằm trong `artifacts/gpu_lite_20261120/`:
`baseline/graphics.json`, `final/graphics.json`, `realtime/performance.json`.
Ảnh garage ba mẫu và ảnh cảnh năm xe do renderer của ứng dụng xuất ra đã
được xem. Công cụ điều khiển cửa sổ Windows không khởi tạo được trong phiên
này; chưa xác minh thao tác bằng chuột hay trải nghiệm trên điện thoại thật.

Lặp lại từ `autonomous-driving-rl` (chọn thư mục output mới):

```powershell
& '..\.venv\metadrive\python.exe' -B scripts/benchmark_race_graphics.py --graphics lite --seconds 8 --output artifacts/gpu-render-new
& '..\.venv\metadrive\python.exe' -B scripts/race_mode.py --graphics lite --smoke-test --steps 4 --output artifacts/gpu-smoke-new
```

25 kiểm thử `test_race*.py` và 8 kiểm thử `test_realtime_phone.py` đã qua.
Native smoke hoàn tất 5/2/1/3 xe trên ba đường, kiểm tra pause/focus/camera/reset.
Smoke ngắn không chứng minh AI về đích hoặc độ ổn định dài hạn.
