# Race Mode: nhiều thuật toán lái xe đồng thời

Race Mode có **Đua đồng thời** và **Chạy đơn** trong cùng cửa sổ MetaDrive.
Mặc định chọn cả năm thuật toán: A*, BFS, Dijkstra, Hybrid A* và MPPI thử nghiệm.
Mỗi thuật toán điều khiển một xe riêng. Các xe cùng xuất phát, cùng chạy trong
một thế giới vật lý, nhìn thấy nhau và có thể va chạm.

## Mở nhanh

Bấm đúp **`CHAY_DUA_XE.cmd`** ở thư mục gốc dự án, hoặc chạy từ PowerShell:

```powershell
.\CHAY_DUA_XE.cmd
```

Chờ nạp và kiểm tra ba đường. Mặc định chọn map gốc SCTCS/seed 0 và đánh dấu
cả năm thuật toán; `--map-set train` mở lại bộ đường TRAIN trước đây.
Chọn đường, giao thông và tốc độ phát, rồi bấm **BẮT ĐẦU ĐUA / Enter**.
Sau đếm ngược 3–2–1, tất cả xe bắt đầu chạy. Không cần điều khiển ga hoặc lái.

## Chọn xe và quan sát

Bấm **02 Thiết lập đường đua ▾** để mở bảng chọn trực tiếp từ tập TEST:

- Bốn tab Dễ / Trung bình / Khó / Cực khó, mỗi tab ba map với ảnh có sẵn
  trong `datasets/mapsuite_v1/previews/`.
- Chọn map để xem đường nhìn từ trên xuống, chiều dài, khúc cua, nút giao
  và seed hình học. Chọn kịch bản 1–5 tương ứng seed môi trường 9101–9105.
- **Dùng bản đồ này** áp dụng lựa chọn vào Garage; **Đóng / Esc** bỏ thay đổi
  đang chọn. **Về map gốc** quay lại ba lựa chọn nhanh trước đây.
- Sau khi áp dụng, bấm **Xuất phát** để nạp đường. Duyệt bảng, đổi tab và
  xem ảnh không tạo engine mới hoặc tiến bước vật lý.

Danh sách lấy từ `datasets/mapsuite_v1/splits/test_geometries.csv` và
`datasets/mapsuite_v1/evaluation_cases/test_cases.csv`, đối chiếu 60 case đã khóa
trong `results/audits/evaluation_protocol/test_case_manifest.csv`.
Hình học lưu sẵn được kiểm tra SHA-256 theo case. Khi chọn TEST, seed, mật độ
giao thông và horizon giữ nguyên dữ liệu; nút giao thông nhanh được khóa theo
kịch bản. Chạy thử có `canonical_run=false`, lưu case TEST thật vào session/log;
không sửa dữ liệu hay ghi đè kết quả benchmark chính thức.

- **ĐUA ĐỒNG THỜI:** đánh dấu từ 2 đến 5 thuật toán. Bấm **Chọn cả 5** hoặc
  **Bỏ chọn hết** để chọn nhanh. Mỗi thuật toán tương ứng một xe.
- **CHẠY ĐƠN:** chọn một thuật toán để luyện riêng trên đường đã chọn.
- Xe có màu và tên thuật toán ở phía trên. Bảng bên phải hiện tiến độ, tốc độ
  hoặc kết quả dừng của từng xe.
- Bấm tên ở bảng, phím **1–5**, hoặc **V / Đổi xe xem** để chuyển xe theo dõi.
  Đồng hồ tốc độ, thanh tiến độ, minimap và Chi tiết AI thuộc xe đang theo dõi.
- Đua nhiều xe mặc định nhìn từ trên cao và camera theo xe đang chọn.
  **C / Camera** đổi sang góc bám xe; cuộn chuột thay đổi độ cao / khoảng cách.
- **I / Chi tiết AI** hiện đầu vào, quyết định và diagnostics thật của xe đang
  theo dõi. Bảng này tạm che bảng trạng thái; **Tab** đổi trang chi tiết.

## Phím và nút

| Thao tác | Nút / phím |
|---|---|
| Bắt đầu | **BẮT ĐẦU ĐUA / Enter** |
| Chọn / bỏ chọn thuật toán trong Garage | Bấm tên hoặc **1–5** |
| Chuyển xe đang theo dõi trong lúc chạy | Bấm tên xe, **1–5**, **V / Đổi xe xem** |
| Tạm dừng / tiếp tục toàn bộ xe | **P / Esc / Tạm dừng** |
| Hủy đếm ngược, về Garage | **Esc / G / Garage** |
| Chạy lại toàn bộ lượt đua | **R / Chạy lại** |
| Dừng lượt, về Garage | **G / Garage** |
| Chi tiết AI của xe đang xem | **I / Chi tiết AI**, **Tab** đổi trang |
| Camera trên cao / bám xe | **C / Camera** |
| Thoát | **Thoát** hoặc đóng cửa sổ |

Tạm dừng dừng vật lý của tất cả xe; cảnh và menu vẫn phản hồi. Chạy lại reset
mọi xe và policy với agent seed 42. Không có khoảng chờ hoặc tự chuyển sang
một thuật toán khác sau khi một xe kết thúc.

Xe về đích, gặp sự kiện an toàn hoặc lỗi policy sẽ được ghi kết quả và đỗ tại
chỗ. Các xe khác tiếp tục chạy; xe đã đỗ vẫn là vật cản. Bảng kết quả chung mở
khi mọi xe đã dừng. Về Garage hoặc thoát giữa chừng ghi những xe còn chạy là
`ABORTED`; không thay kết quả những xe đã dừng trước đó.

## Cách mô phỏng và giới hạn

Người lái tay và AI cùng dùng luật `race_strict_v1`: đè vạch liền, ra khỏi
đường hoặc va chạm thì dừng xe đó; vạch đứt cho phép đổi làn. Chỉ về đích
không vi phạm mới được tính hạng, theo thời gian mô phỏng; xe bị loại không
được dẫn đầu vì có tiến độ cao. Mọi xe trong lượt tính hạng cùng giới hạn
lái `[-0.6,0.6]`, ga/phanh `[-0.8,0.6]`. Nút đưa xe về đường chỉ dùng trong
tập lái không xếp hạng. Xem [luật và nút điện thoại](PHONE_RACING_GUIDE.md).

Trần tốc độ hiện là **20 km/h**, áp dụng giống nhau cho người và AI để đua
sát với nhịp của các planner hiện tại. Xe giảm ga từ 19 km/h và không tăng
ga khi chạm trần; phanh vẫn dùng bình thường. Tập lái cũng có cùng mức trần.
Tốc độ phát 1×/2×/5× là nhịp thời gian mô phỏng, độc lập với giới hạn km/h.

Riêng trong Race Mode, A*, BFS, Dijkstra và Hybrid A* dùng tốc độ mục tiêu
đường thẳng **15 km/h**, vẫn giảm theo góc cua như controller hiện có. Giá
trị chuẩn hóa được tính theo mẫu số tốc độ của từng xe; không đổi dữ liệu
đầu vào hoặc file JSON gốc. Cả chạy đồng bộ và worker chạy mượt nhận cùng
thiết lập. `runtime_planner_options` trong lineup/kết quả ghi cấu hình thực
sự áp dụng. MPPI giữ mục tiêu cấu hình riêng và cùng chịu trần 20 km/h.

Chạy đơn, đua AI và lái bằng điện thoại đều mặc định dùng chế độ tương tác
mượt: vật lý bước cố định 0,02 giây, AI mục tiêu 10 quyết định/giây mô phỏng
trong các tiến trình riêng. CPU tính MPPI không chặn luồng vẽ. Laptop không
stream mục tiêu 50 FPS chuyển động; khi stream thêm camera điện thoại, ngân
sách vẽ là 30 FPS. Giữ cảnh quan/map gốc và render trên GPU NVIDIA.

Mỗi bước vật lý áp dụng lệnh mới nhất cho toàn bộ xe trong cùng thế giới,
không đợi tất cả AI trả lời. Lệnh AI cũ quá 350 ms chuyển sang phanh; tuổi
lệnh và lệnh thực sự áp dụng được ghi log. Đếm ngược chờ tiến trình AI sẵn
sàng để tránh xe đứng chờ sau xuất phát. Tạm dừng/tiếp tục đặt lại đồng hồ
để không chạy bù toàn bộ thời gian đã dừng.

`--lockstep` giữ đường chẩn đoán đồng bộ cũ: các policy đọc cùng trạng thái,
tính tuần tự rồi tiến 0,1 giây một lần. `--smoke-test` chủ động dùng đường này
để giữ các kiểm tra quyết định cũ. Launcher benchmark chính thức không đổi.

Xe xuất phát theo hai hàng, cách nhau 8 m theo chiều dọc và phân bố trên các
làn hiện có. Vị trí xuất phát khác nhau. Tuyến tham chiếu công khai của mỗi xe
theo tâm làn hiện tại, thay vì bắt mọi xe nhập về làn 0. Xe đua khác được đưa vào
traffic context và LiDAR của simulator. Chính thuật toán quyết định tránh hoặc
phanh. Race Mode giới hạn lệnh áp dụng theo khung chung nêu trên và ghi cả
lệnh yêu cầu/lệnh áp dụng; không thay bằng điều khiển kịch bản.

Đây là **demo tương tác giữa các xe**, không phải benchmark độc lập
có cùng vị trí xuất phát. Bản đồ gốc giữ phân vùng thật trong log, không gán
nhầm thành TRAIN. A*, BFS, Dijkstra và Hybrid A* dùng policy hiện có;
registry và cách chấm benchmark chính thức không đổi. Xe có thể phanh chờ,
không tìm thấy đường, va chạm hoặc ra khỏi đường.

MPPI dùng **cùng agent `stage3_mppi` với CLI / Workbench**, triển khai trong
`src/agents/MPPI.py`. Cấu hình hiện tại đọc từ
`configs/experiments/mppi_v2_h30.json`: 512 mẫu, chân trời 30 quyết định
(3 giây), tốc độ mục tiêu 16 m/s (57,6 km/h). Đây là tốc độ mong muốn của
thuật toán, không bảo đảm xe đạt được. Sửa khối `parameters` rồi mở lại ứng dụng
để dùng cấu hình mới; file `mppi_v1.json` chưa được chọn làm cấu hình chạy.

Agent dùng mô hình xe đạp, dự đoán vận tốc không đổi của xe khác, hình học thân
xe và LiDAR để tính chi phí. Chưa chứng nhận chất lượng lái xe. MPPI chỉ nhận
`AgentInputV1`, không gọi simulator thử tương lai. Biến thể hiện tại chưa có
correction term của importance sampling đầy đủ. Nền tảng thuật toán:
[Williams và cộng sự, Information Theoretic Model Predictive Control](https://arxiv.org/abs/1707.02342).

Tốc độ phát 1× / 2× / 5× là tốc độ xem mục tiêu, phụ thuộc máy; không tăng công
suất xe hoặc thay đổi bước vật lý. Năm policy, đặc biệt Hybrid A*, có thể làm
tốc độ thực tế thấp hơn mức đã chọn.

## Log

Mỗi phiên tạo thư mục `autonomous-driving-rl/artifacts/race_mode/<thời-điểm>/`:

- `session.json`: cấu hình, lineup, nguồn mã, trạng thái phiên; đường dẫn,
  tham số và SHA-256 của cấu hình MPPI thực sự được nạp lúc mở ứng dụng.
- `decisions.jsonl`: một dòng cho mỗi xe có quyết định, chứa `episode`,
  `vehicle_id`, `agent_id`, `race_tick`, `input_step`, lệnh và diagnostics.
  Với chế độ mượt, đây là kết quả worker trả về, có tuổi quyết định; khác
  với hành động thực sự đã áp dụng ở mỗi bước vật lý.
- `applied_actions.jsonl`: lệnh thực sự cho toàn bộ xe, tuổi lệnh AI, số bước
  vật lý và thời gian xử lý của chế độ tương tác mượt.
- `episodes.jsonl`: kết quả riêng từng xe, lỗi kỹ thuật tách khỏi kết quả lái.
- `comparison-001.json`, …: lineup, kết quả đến theo thứ tự xe dừng, trạng thái
  `RUNNING` / `COMPLETE` / `STOPPED` và danh sách chưa có kết quả.

`COMPLETE` chỉ có nghĩa mọi xe đã có kết quả, không có nghĩa mọi xe về đích.

## Kiểm tra cho người phát triển

Mặt đường dùng lớp vật liệu chi tiết theo phạm vi từng map; vạch trắng/vàng
được dựng thành dải hình học liên tục theo đường cong, không phụ thuộc ảnh
địa hình có độ phân giải thấp. Vạch đứt theo nhịp sơn 2 m / khoảng trống 2 m
của phiên bản MetaDrive đang dùng.
Lớp hiển thị này tự tạo lại khi đổi map hoặc chạy lại, dùng chung cho camera
laptop và điện thoại. Hình học map đã khóa, địa hình vật lý và cảm biến giữ
nguyên. Sau khi cập nhật mã, đóng cửa sổ Race Mode cũ rồi mở lại launcher.

```powershell
& '.\.venv\metadrive\python.exe' -B -u autonomous-driving-rl/scripts/race_mode.py --smoke-test --steps 100
```

Kiểm tra lần lượt các lineup 5 / 2 / 1 / 3 xe, ba đường, bật giao thông,
tạm dừng tất cả xe, chuyển xe theo dõi, camera và bảng chi tiết. Những **lượt
kiểm tra** nối tiếp nhau để xác minh reset; trong mỗi lượt, các xe chạy đồng thời.
Chương trình chụp ảnh rồi tự thoát. `smoke_passed`, `simultaneous_verified`,
`pause_verified`, `persistent_window_verified` chỉ được ghi khi hoàn tất.
Các xe chạy tới giới hạn kiểm tra ghi `PREVIEW_LIMIT`, không coi là về đích.

Nếu bị giật, dùng 1×, tắt Chi tiết AI hoặc chọn ít xe hơn. Nếu có lỗi kỹ thuật,
xem `episodes.jsonl` và cửa sổ dòng lệnh. Không mở nhiều phiên ứng dụng cùng lúc.

### Kiểm tra chuyển động ngày 20/11/2026

Cùng map gốc SCTCS/seed 0, Tesla, MPPI 512 mẫu/chân trời 30, đồ họa balanced,
RTX 3050 Ti. Sau khi worker sẵn sàng và khởi động cảnh 2 giây, lấy mẫu 8 giây:

| Chỉ số | Đồng bộ cũ | Chế độ mượt |
|---|---:|---:|
| Khung vẽ/giây | 10.7 | 48.2 |
| Khung thực sự đổi vị trí xe/giây | 8.4 | 48.1 |
| Khoảng cách giữa lần đổi vị trí, trung bình | 119.16 ms | 20.76 ms |
| Giây mô phỏng/giây thực | 0.839 | 1.001 |

Dữ liệu `artifacts/motion_optimization/verified3/motion-performance.json`.
Lượt đủ năm AI trên cùng map/đồ họa đạt 39.5 FPS, 39.4 khung đổi vị trí/giây,
vật lý 49.9 Hz, tỉ lệ thời gian mô phỏng/thực 0.995, không có lỗi kỹ thuật:
`artifacts/motion_optimization/five-ai/motion-performance.json`.
Đây là phép đo ngắn trên máy đang dùng; không bảo đảm 50 FPS mọi thời điểm.
Hai chế độ có cách lập lịch AI khác nhau nên không dùng bảng này so sánh
chất lượng thuật toán hoặc kết quả benchmark. Lặp lại từ thư mục con:

```powershell
& '..\.venv\metadrive\python.exe' -B scripts/benchmark_race_motion.py --output artifacts/motion-new --seconds 8
```

## Đường đua Việt Nam riêng

Chọn ô **Việt Nam** ngay bên phải **Khó**, rồi bấm **XUẤT PHÁT**.
Tuyến Võ Văn Kiệt dài khoảng **10 km**, ba làn, đua đồng thời tối đa năm AI
và một người chơi. Tốc độ mục tiêu của AI là **60 km/h**, giới hạn chung
**80 km/h**; xe giảm tốc trước khúc cua. Ba map cũ giữ thiết lập 15/20 km/h.

```powershell
.\CHAY_DUA_XE.cmd --road vietnam --graphics balanced
```

Bản đồ dùng hình tuyến OpenStreetMap, có minimap và quãng đường còn lại.
Đây là đường đua tùy biến: ba làn được chuẩn hóa, cao độ phẳng, lược bỏ
giao lộ; nhà, cây và kênh là cảnh dựng, không phải bản sao khảo sát Sài Gòn.
Tài nguyên cây, mặt tiền nhà và vật liệu có nguồn CC0 tại
`assets/scenery/sources.json`; dữ liệu đường và ghi công ODbL nằm trong
`assets/maps/vietnam/vo_van_kiet_route.json`.

Cây có lá cắt theo alpha thật, thân cây và mặt tiền nhà có chi tiết hình học,
mặt đường dùng normal map, vỉa hè có bó vỉa nổi và mặt nước có phản chiếu
thay đổi theo góc nhìn. Cảnh dùng chung cho camera laptop và camera điện thoại;
thiết bị iPhone thực tế cần được kiểm tra riêng.

Lượt kiểm tra ngày 21/11/2026: A*, BFS và Dijkstra đều chạy hết tuyến
9.99 km không có NPC, không va chạm, không ra ngoài đường hoặc đè vạch liền;
tốc độ cao nhất khoảng 60.6 km/h. Báo cáo nằm ở `artifacts/vn-map/*-main-final.json`.
Kiểm tra ngắn năm AI đồng thời, reset Việt Nam → map cũ → Việt Nam,
tạm dừng và camera nằm ở `artifacts/vn-map/native-integration/`.
Các lượt này là kiểm tra tương tác, không phải kết quả benchmark TEST.

## Đồ họa nhẹ cho laptop

Chọn `--graphics lite` để dùng cảnh 3D 75% độ phân giải, chữ giữ rõ,
thân xe gom batch và có LOD, texture bánh được tối ưu riêng. Muốn cảnh rõ hơn,
chạy `CHAY_DUA_XE.cmd --graphics balanced`. Xem cấu hình, số đo và giới hạn
kiểm chứng tại [LOCAL_GPU_LITE.md](LOCAL_GPU_LITE.md).
