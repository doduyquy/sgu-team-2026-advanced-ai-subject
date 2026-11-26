# Điện thoại cầm lái — đua cùng AI trên laptop

## Mở và kết nối

1. Bấm đúp `CHAY_DIEN_THOAI.cmd` ở thư mục gốc. Lần đầu launcher cài thêm
   các thư viện trong `autonomous-driving-rl/requirements-mobile.txt` nếu thiếu.
2. Chọn **Bạn đua với AI** hoặc **Lái tự do**, mẫu xe và đường trên laptop.
   Chế độ đua cho chọn 1–5 đối thủ AI; Lái tự do chỉ có xe người chơi.
3. Mở **Điện thoại / QR**, cho điện thoại và laptop vào cùng Wi-Fi, quét QR.
4. Trên trình duyệt điện thoại bấm **KẾT NỐI TAY LÁI**, chọn xe rồi bấm
   **SẴN SÀNG · XUẤT PHÁT**. Chờ đếm ngược 3 giây. Có thể bấm Xuất phát
   trên laptop sau khi điện thoại đã kết nối.
5. Xoay ngang điện thoại. Giữ **◀ TRÁI / ▶ PHẢI** để đánh lái, thả để trả
   thẳng; giữ **GA** để tăng tốc. **PHANH** ưu tiên hơn ga. Hai nút lái cùng
   được giữ thì trả thẳng; có thể giữ nút lái và ga/phanh bằng nhiều ngón.

Không cần ứng dụng điện thoại, tài khoản hoặc dịch vụ truyền hình bên ngoài.
QR có mã ghép nối riêng từng phiên. Chỉ một điện thoại giữ quyền điều khiển.
Nút Thoát trên điện thoại ngắt tay lái và làm xe tự phanh; ứng dụng laptop vẫn chạy.

Trạng thái **Đã kết nối tay lái** chỉ hiện khi cả kênh lệnh và kênh lái đã
mở. Trang hiện từng bước ghép nối; quá 15 giây chưa sẵn sàng sẽ cho thử lại.
Trong màn hình chờ có nút **Ngắt kết nối tay lái**. Nếu laptop đang đua AI,
web hiện **Về Garage để cầm lái**, không tự kết thúc lượt AI chỉ vì kết nối.
Đã kết nối thì có hình xem trước ngay trong Garage, trước khi xuất phát.

### Safari trên iPhone khi xoay máy

Trang tay lái đo khung nhìn thực tế, kể cả khi thanh địa chỉ Safari ẩn/hiện,
và chừa vùng tai thỏ/thanh Home. Màn hình chờ chia hai cột khi nằm ngang;
hai nút lái và bàn đạp co theo chiều cao còn lại. Xuất phát không tự ép video
vào chế độ toàn màn hình của Safari. Khi xoay, ga được ngắt và hướng lái trả
thẳng; chạm lại để tiếp tục lái. Nếu Safari tạm dừng phát hình, bấm
**Chạm để hiện lại hình** để tiếp tục trên cùng kết nối.

Sau khi cập nhật giao diện, tải lại trang Safari để nhận bản mới; không cần
khởi động lại mô phỏng nếu chỉ đổi HTML. Khi thay camera/cấu hình luồng trên
laptop, mở lại Race Mode rồi quét QR của phiên mới.

Giao diện lái thu gọn tốc độ, thứ hạng và tiến độ ở trên; trái/phải, ga/phanh
ở hai góc dưới. Nút **···** mở chất lượng, FPS/mạng, luật và thoát tay lái.
Mở menu sẽ nhả ga và tự phanh; đóng menu rồi chạm lại để tiếp tục lái.
Camera bám sau nhìn thấp hơn để thấy trọn xe, tránh xe bị cắt dưới màn hình.

Trong cửa sổ đua thông thường `CHAY_DUA_XE.cmd`, các nút chọn xe, hai chế độ
người chơi và **Điện thoại / QR** cũng có sẵn. Dùng launcher điện thoại để
cài các phụ thuộc tùy chọn trước lần kết nối đầu tiên.

## Xe và góc nhìn

Ba mẫu **Lamborghini Aventador, Tesla Cybertruck, VinFast VF3** có
ngoại hình khác nhau; hai mẫu đầu dùng artwork glTF, VF3 là mẫu dựng minh họa.
Nguồn và giới hạn ở [VEHICLE_SHOWROOM.md](VEHICLE_SHOWROOM.md). Các mẫu dùng
chung thân va chạm và hệ truyền động MetaDrive; không mô phỏng thông số hãng.

Mẫu xe độc lập với thuật toán. Ở chế độ AI, chọn thẻ thuật toán rồi chọn mẫu
xe bên dưới showroom. Ở hai chế độ người chơi, showroom đổi mẫu xe của bạn.

Điện thoại có ba góc: **Bám sau**, **Nắp capo**, **Ghế lái**. Ghế lái đặt
camera trong vị trí người lái, ẩn thân xe riêng trên luồng điện thoại và có
ốp bảng điều khiển trên màn hình; chưa có nội thất 3D chi tiết.
Laptop tiếp tục dùng **C / Camera** để đổi trên cao/bám xe, **1–6 / V** để
chọn xe theo dõi. Đổi camera laptop không thay đổi camera điện thoại.
Các đối thủ vẫn hiện trên cả hai màn hình vì cùng một thế giới vật lý.

## Luật đua chung cho người và AI

Chế độ **Bạn đua với AI**, **Chạy đơn AI** và **Đua đồng thời** cùng dùng
`race_strict_v1`:

- Đè vạch liền vàng/trắng, ra khỏi mặt đường hoặc va chạm: dừng lượt xe đó.
  Vạch đứt được đổi làn. Kiểm tra dùng tiếp xúc thân va chạm MetaDrive, không
  suy đoán từ ảnh xe trang trí hay từng bánh xe của model.
- Va chạm/ra khỏi đường/vạch liền được ưu tiên trước về đích trong cùng bước.
  Xe bị loại không có hạng thắng, dù tiến độ đường cao hơn xe còn chạy.
- Chỉ về đích không vi phạm được tính hạng; thời gian mô phỏng ít hơn xếp
  trước. Thời gian bằng nhau đồng hạng. Hết giờ chưa về đích không tính thắng.
- Cùng giới hạn điều khiển chuẩn hóa cho tất cả xe: lái `[-0.6, 0.6]`,
  ga/phanh `[-0.8, 0.6]`. Lệnh yêu cầu và lệnh sau giới hạn được ghi riêng.
  Đây là giới hạn trong Race Mode, không sửa mã policy hay benchmark chính thức.
- Trần tốc độ **20 km/h** áp dụng cho người chơi và mọi AI, cả ở chế độ tập.
  Từ 19 km/h xe giảm dần ga; tới trần không tăng ga. Phanh của người lái vẫn
  có tác dụng. Màn hình điện thoại và thanh trạng thái laptop hiện mức trần.
- Nút **Về đường / H** bị chặn trong cuộc đua, cả trên giao diện lẫn xử lý
  lệnh. Chạy lại bắt đầu lượt mới cho toàn bộ xe.

**Tập lái / Lái tự do** không xếp hạng. Có thể tiếp tục sau vi phạm và dùng
**Về đường / H** để đưa xe về chỗ trống trên làn; nếu mọi vị trí đều bị chắn,
xe giữ nguyên. Kết quả tập không được ghi là chiến thắng hợp lệ.

Các xe vẫn xuất phát ở các ô khác nhau và người/AI có cách ra quyết định khác
nhau. Đây là cuộc đua tương tác cùng luật, không phải phép đo benchmark có
cùng vị trí xuất phát và cùng tần số ra quyết định.

## Mạng và hiệu năng

Laptop giữ toàn bộ trạng thái mô phỏng, AI và vật lý. Một camera RGB riêng
truyền H.264 qua WebRTC; hai data channel tách lệnh lái mới nhất và nút thao tác.
Không có STUN/TURN hoặc relay công cộng được cấu hình.
Máy chủ chỉ phục vụ trang tay lái, trạng thái phiên và thương lượng WebRTC;
không chia sẻ thư mục dự án.

Lệnh lái được gửi khoảng 50 lần/giây. Chế độ người chơi dùng bước vật lý
0,02 giây, tối đa ba bước bù khi một lần render chậm. Tần số đọc lệnh thực tế
phụ thuộc tốc độ render. AI chạy trong các tiến trình riêng với snapshot
đầu vào được lấy tại lúc ra quyết định, mục tiêu 10 Hz; AI không giữ luồng
vật lý chờ tính toán. Lệnh AI quá 350 ms thì xe AI phanh chờ lệnh mới.
Không nhận lệnh mới trong 450 ms thì xe tự phanh tại bước mô phỏng kế tiếp.
Thao tác đổi ứng dụng, hủy chạm, mất tiêu điểm và ngắt kết nối xóa ga/hướng lái.
Luồng hình chỉ giữ frame mới nhất, không tích hàng đợi video.

Chế độ người chơi cố định tốc độ phát mục tiêu 1×. Chạy đơn AI và so sánh AI
cũng dùng vật lý 0,02 giây và worker riêng để tránh nhảy vị trí theo nhịp 10 Hz.
`--lockstep` mở lại cơ chế đồng bộ cũ để chẩn đoán. Các lượt chơi async có nhãn
`async_ai_shared_world`, không dùng thay kết quả benchmark chính thức.

Mặc định dùng **NVIDIA NVENC** để mã hóa, cùng RTX 3050 Ti đang dựng cảnh.
Chế độ `nvenc` yêu cầu NVIDIA và kiểm tra mã hóa thật trước khi hiện QR.
Nếu NVENC đi kèm PyAV đòi driver mới hơn, chương trình dùng FFmpeg 7.1 từ
`imageio-ffmpeg==0.6.0` để vẫn mã hóa bằng RTX với driver 566.07 hiện có.
Không thay driver hay sửa aiortc. FFmpeg chạy ẩn, giữ một tiến trình cho mỗi
luồng video; chỉ dùng RTP nội bộ 127.0.0.1 để nhận ranh giới khung hình H.264.
Tiến trình đóng khi ngắt điện thoại hoặc đổi độ phân giải.

Thanh FPS phải hiện **GPU NVIDIA**. Chế độ mặc định không tự chuyển sang
Intel hoặc CPU. Có thể chủ động chọn `--encoder qsv` (Intel), `--encoder cpu`,
hoặc `--encoder auto` (thử NVIDIA, cho phép CPU khi lỗi).

Chọn **Tự động**, **Mượt 360p**, **Cân bằng 540p**, hoặc **Rõ nét 720p** trước
khi xuất phát, hoặc vào **··· → Chất lượng hình** khi đang lái. Tự động
bắt đầu ở 540p / 3,5 Mbps, nâng lên 720p / 6 Mbps sau sáu mẫu tốt liên tiếp
(ít nhất 26 FPS, RTT dưới 80 ms, P95 mã hóa dưới 22 ms). Nếu ba mẫu liên
tiếp kém, hạ lần lượt 720p → 540p → 360p / 1,5 Mbps. Giữa các lần đổi có
khoảng nghỉ 8 giây; chọn chất lượng thủ công sẽ giữ lựa chọn đó.
Thanh trạng thái hiện FPS giải mã tại trình duyệt, RTT của data channel,
encoder thực sự được dùng và độ phân giải. RTT không phải độ trễ từ ngón tay
tới hình ảnh. Mục tiêu hình là 30 FPS; số thực tế phụ thuộc máy và mạng.

Ảnh được đọc về RAM theo nhịp camera, chỉ giữ frame mới nhất. H.264 dùng
không B-frame, keyframe ngắn; đổi độ phân giải tạo lại encoder với header mới.
Chưa chứng nhận FPS, độ trễ hoặc tương thích trên mọi điện thoại/router.

Nếu điện thoại không mở được trang, kiểm tra cùng mạng, không dùng Wi-Fi
khách chặn các thiết bị liên lạc với nhau, và quyền kết nối mạng của Python.
Nếu QR chọn nhầm card mạng (VPN, hotspot), chỉ định IP Wi-Fi của laptop:

```powershell
.\CHAY_DIEN_THOAI.cmd --advertise-host 192.168.1.10 --port 8765
```

Thay IP ví dụ bằng IP thật. Với Windows Mobile Hotspot, dùng IP adapter
hotspot. Cổng đã bận thì đổi `--port`. Chỉ host trong LAN, chưa triển khai
truy cập qua Internet. Không cần camera/microphone hay cảm biến nghiêng của điện thoại.

## Nhật ký và kiểm tra

Mỗi phiên lưu trong `artifacts/race_mode/<timestamp>/`. `phone-qr.png` và
`phone-connection.json` chứa liên kết ghép nối; giữ riêng trong LAN.
`decisions.jsonl` ghi lệnh người chơi và AI trong cùng `race_tick`.
Ở chế độ async, `decisions.jsonl` ghi quyết định AI trả về; `applied_actions.jsonl`
ghi lệnh thực sự áp dụng cho tất cả xe, tuổi lệnh AI, số bước vật lý và thời gian xử lý.
`episodes.jsonl` ghi `controller`, `vehicle_model`, `recoveries`.
Log còn ghi `race_rules`, `ranked`, cờ vạch liền/đứt và `clean_success` cho cả
người chơi lẫn AI. `comparison-*.json` chỉ cấp `finish_ranks` cho xe về đích
hợp lệ. Màn hình điện thoại hiện lý do bị loại và khóa nút lái khi xe đã dừng.
Người chơi có ID `human_phone` / `racer_5`; các AI giữ ID cũ. Các lượt chơi
đều là demo tương tác, không phải kết quả benchmark chính thức; bản đồ giữ
phân vùng gốc trong metadata.

Chạy từ thư mục con `autonomous-driving-rl`:

```powershell
& '..\.venv\metadrive\python.exe' -B -m unittest discover -s tests -p test_race_mode.py -v
& '..\.venv\metadrive\python.exe' -B -m unittest discover -s tests -p test_phone_racing.py -v
& '..\.venv\metadrive\python.exe' -B scripts/smoke_phone_racing.py --output artifacts/phone-smoke-new
node --test tests/test_phone_viewport.mjs
& '..\.venv\metadrive\python.exe' -B -m unittest discover -s tests -p test_realtime_phone.py -v
& '..\.venv\metadrive\python.exe' -B scripts/benchmark_phone_runtime.py --output artifacts/phone-perf-new --seconds 12
```

Kiểm tra WebRTC qua NVENC thật trên máy có RTX: đặt
`$env:PHONE_TEST_ENCODER='nvenc'` trước khi chạy `test_phone_racing.py`, sau đó
`Remove-Item Env:PHONE_TEST_ENCODER`. Mặc định bài kiểm tra dùng CPU để chạy
được trên máy không có NVIDIA.

Thư mục output kiểm tra phải chưa tồn tại. Native smoke dùng lệnh điện thoại
giả lập để kiểm tra sáu xe, cả ba mẫu, ba camera, reset và đưa xe về đường.

Kiểm tra bản nút lái/luật ngày 20/11/2026: 45 kiểm tra Python và 9 kiểm tra
JavaScript đạt. `artifacts/phone_buttons_rules_20261120/native-v4/rules-audit.json`
ghi tiếp xúc vật lý thật cho vạch vàng/trắng, vạch đứt, ra đường và va chạm:
người/AI nhận cùng kết quả. Ở mép trắng sát lề, cả hai đồng thời chạm vạch và
lề nên kết quả chính là `CRASH_SIDEWALK`; cả hai cờ vẫn được giữ trong log.
`browser/browser-audit.json` trong cùng thư mục kiểm tra bốn viewport 320–844 px
và thao tác đa chạm bằng Chromium mô phỏng. Chưa kiểm tra điện thoại vật lý.
Kiểm thử WebRTC dùng hai peer thật trên máy, xác minh giải mã hình, lệnh lái,
mã ghép nối và quyền điều khiển độc quyền. Các kiểm tra đó không thay thế
việc lái thử bằng điện thoại thật qua Wi-Fi.

Giới hạn tốc độ ngày 21/11/2026: thêm bộ điều tốc ở lượt đua, giữ nguyên
chuẩn hóa tốc độ trong đầu vào AI. Vận tốc ngang được chặn sau mỗi tick vật
lý; hướng chuyển động, độ cao và vị trí không bị đặt lại. Kiểm tra MetaDrive
với hai xe cùng giữ ga và tốc độ ban đầu 50 km/h đạt trần 15 km/h ở cả nhịp
0,02 s và 0,1 s. Log: `artifacts/race_speed_limit_20261121/native/speed-audit.json`.
51 kiểm tra Python và 15 JavaScript đạt; đây là kiểm tra cơ chế giới hạn,
không khẳng định các thuật toán có cùng tốc độ thực tế trên mọi góc cua.

Sau yêu cầu tăng nhẹ toàn bộ nhóm, trần được nâng từ 15 lên **20 km/h**.
Nhóm A*/BFS/Dijkstra/Hybrid A* đồng thời tăng mục tiêu đường thẳng lên
**15 km/h** trong riêng lượt đua. Cơ chế giảm tốc vào cua giữ nguyên; MPPI
cùng chịu trần mới. Cấu hình benchmark và file JSON gốc không bị chỉnh.
53 kiểm tra Python và 15 JavaScript đạt, gồm quyết định BFS thật trong worker
và việc tạo policy mới vẫn giữ cấu hình gốc. Bằng chứng tích hợp:
`artifacts/race_pace_20_20261121/native/pace-audit.json`.

## Sửa kết nối treo ngày 21/11/2026

Đã tái hiện lỗi khi xuất phát: thứ hạng có kiểu `numpy.int64` khiến bộ gửi
telemetry lỗi JSON và đóng kết nối dù mô phỏng tiếp tục chạy. Thứ hạng giờ
trả số Python, toàn bộ trạng thái được chuẩn hóa trước khi gửi HTTP/WebRTC.
`phone-events.jsonl` lưu giai đoạn kết nối và lý do đóng, không ghi mã ghép nối
hay SDP. Callback từ lần kết nối cũ không ghi đè trạng thái lần kết nối mới.

49 kiểm tra Python và 14 JavaScript đạt. Kiểm tra Chromium kết nối thực với
MetaDrive, NVENC, xuất phát, GA, pause, ngắt và kết nối lại đã đạt tại
`artifacts/phone_connection_fix_20261120/native-20261121-v5/`.
Lần đầu có tay lái và hình khoảng 3,36 giây, kết nối lại khoảng 0,63 giây
trong phép thử cùng máy. Đây không phải số đo trên điện thoại thật hoặc
độ trễ từ ngón tay tới hình ảnh qua Wi-Fi.
