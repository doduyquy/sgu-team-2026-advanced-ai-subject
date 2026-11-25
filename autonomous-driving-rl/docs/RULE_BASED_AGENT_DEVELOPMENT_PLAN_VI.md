# Kế hoạch phát triển Rule-Based Agent cho MetaDrive

Ngày: 2026-10-30. Agent: `stage1_rule_based`.

Bảo trì source 2026-10-31: phiên bản chạy hiện tại 0.6.1, cùng luật/tham số P4
với 0.6.0. [Hồ sơ cleanup](SOURCE_CLEANUP_V0_6_1.md): 547 tests, 1992 bước replay
khớp và một CLI smoke TRAIN. Các kết quả driving 16 case bên dưới thuộc bản
0.6.0 đã đóng băng; P5 chưa triển khai thêm trong đợt bảo trì.

## 1. Mục tiêu và phạm vi

Xây dựng agent lái xe theo luật có thể giải thích, đi đúng tuyến, giữ thân xe
trong hành lang, điều chỉnh tốc độ theo tình huống, bám xe, nhường đường và
xử lý đường tách/nhập, ngã ba, ngã tư, vòng xuyến trong bộ MapSuite hiện tại.
Ưu tiên clean success và tránh va chạm, sau đó giảm thời gian hoàn thành.

Đây là mục tiêu phát triển trong simulator và miền dữ liệu công khai của dự án.
Không coi một tập luật hữu hạn là bảo đảm xử lý mọi tình huống giao thông thực.
Giữ Stage 1 là luật, điều khiển phản hồi và dự báo hình học ngắn hạn; chưa đưa
tree search, MCTS hoặc học policy vào baseline này.

### Trạng thái cập nhật — P4 ở 0.6.0

Đã triển khai metadata làn kề/vạch thực, gap trước/sau, TTC/quãng phanh,
swept-body/LiDAR, chuyển bắt buộc theo tuyến, target/progress/commit, hủy trước
commit, phanh và tiếp tục sau commit, vượt có điều kiện và trở lại làn khi đã
quan sát đủ clearance. Thiết kế: [P4](RULE_BASED_P4_DESIGN.md).
26 kiểm thử P4 và bộ tích hợp 279 tests đã qua; sáu controlled physical probes
đạt, 929 bước replay khớp cả archive thực. [Work log 0.6](RULE_BASED_V0_6_0_WORKLOG.md)
ghi đầy đủ lỗi và run dừng. **P4 đã hoàn tất triển khai và nghiệm thu có phạm vi**,
đóng băng development release 0.6.0: cùng 16 TRAIN case, success 13/16 giữ nguyên
so với 0.5, không mất success, không crash/road/solid-boundary; ba Extreme vẫn
timeout. Có sáu thao tác đổi làn với lý do vượt hoàn tất trong năm case tự nhiên. Xem
[báo cáo 0.6](RULE_BASED_V0_6_0_EXPERIMENT_REPORT.md).

P4 có phạm vi làn kề cùng chiều trên đoạn gần thẳng đủ dài, qua vạch đứt trắng.
Không đổi trong nút/vòng xuyến hoặc ép chen/vượt. P5/full 108 TRAIN, audit schema
1.3.0 và canonical VALIDATION/TEST vẫn chưa hoàn thành. Benchmark vẫn tắt.

### Mốc lịch sử — triển khai P2/P3 ở 0.5.0

Thiết kế: [P2/P3](RULE_BASED_P2_P3_DESIGN.md). Toàn bộ lần thử, lỗi, regression
và kết quả đo: [work log 0.5](RULE_BASED_V0_5_0_WORKLOG.md).

| Giai đoạn | Đã triển khai ở 0.5.0 | Phạm vi còn lại |
|---|---|---|
| P0 | Signed ego-motion/heading, local passage/yield boundary, actor road hypotheses qua adapter chung | Audit contract 1.2.0; dữ liệu vạch, làn đích, đèn/biển nếu có nguồn |
| P1 | Giữ cruise 45/junction 24/recovery 8 km/h, braking/headway và diagnostics | Jerk/steering-rate, tối ưu controller và tốc độ sau nghiệm thu an toàn |
| P2 | Association một-một, TTL 0.6 s, ego compensation, CV/CT/possible road paths, footprint/gap/TTC, LiDAR, overflow, timed cut-in, rear-side guard | Dự báo có giới hạn, không biết ý định thật; xem coverage và ca chưa đạt trong báo cáo |
| P3 | FSM approach/yield/commit/traverse/exit/blocked, branch memory, space-time admission, exit queue guard, roundabout merge sections | Chỉ geometry-based yielding khi không có nguồn quyền ưu tiên/đèn; nghiệm thu theo TRAIN đã công bố |
| P4 | Chưa triển khai | Metadata biên/làn đích, chuyển làn bắt buộc, hủy/vượt an toàn |
| P5 | Snapshot/hash/trace, paired experiments, replay frozen policy và tests | Full 108 TRAIN, audit, canonical VALIDATION/TEST và mục tiêu >=95% |

Không diễn giải đã có đủ thành phần P2/P3 thành bảo đảm mọi bản đồ đều thành
công. Không tăng ga do timeout chờ; traffic bị chặn có thể cần P4 để đi tiếp.
Không đọc navigation/policy hay hành động tương lai của actor để làm oracle.

Kết quả đóng băng 0.5: cùng 16 case TRAIN, success 10 → 13, crash 4 → 0,
không mất case thành công; ba Extreme vẫn timeout. 252 tests và replay đã đạt.
Đây là nghiệm thu có phạm vi, chưa đạt full 108 TRAIN hoặc mục tiêu >=95%.
Chi tiết: [báo cáo 0.5](RULE_BASED_V0_5_0_EXPERIMENT_REPORT.md).

### Mốc lịch sử — 0.4.0, ngày 2026-10-30

Phiên bản 0.4.0 đã đóng băng source/config và lưu experiment đầy đủ. Xem
[changelog](RULE_BASED_CHANGELOG.md) và [báo cáo](RULE_BASED_V0_4_0_EXPERIMENT_REPORT.md).

| Giai đoạn | Đã triển khai | Phần chưa nghiệm thu/triển khai |
|---|---|---|
| P0 | Waypoint đúng làn, nối theo endpoint/heading; fixture checkpoint drift/gap; kiểm chứng hướng/range LiDAR | Heading đầy đủ, dữ liệu vạch/làn bên cạnh và audit contract |
| P1 | Giữ tốc độ thích ứng 45/24/8 km/h và quãng phanh | Jerk/steering-rate, kiểm chứng rộng |
| P2 | Corridor cong, footprint actor, gap/TTC, prediction crossing/cut-in, LiDAR obstacle, overflow guard | Tracking ID/TTL, ý định actor; hai case Hard vẫn crash nên chưa đạt nghiệm thu |
| P3 | Nhường trước lối vào junction, behavior diagnostics, yield-release timer 0.5 s/reset | Quyền ưu tiên đầy đủ, khóa nhánh/commit và FSM traverse/roundabout |
| P4 | Chưa triển khai | Dữ liệu làn đích/vạch, chuyển làn bắt buộc, hủy và vượt an toàn |
| P5 | Runner có snapshot/hash/trace; so sánh paired 9 case, CLI smoke và snapshot replay | Full 108 TRAIN episode, canonical VALIDATION/TEST và audit benchmark |

Kết quả cùng 9 case TRAIN/env0: success 5/9 → 7/9, crash 4/9 → 2/9;
không mất case thành công. Hai case còn crash có completion giảm, được giữ
trong báo cáo. Không coi kết quả này là P2/P3 hoàn tất hay đạt mục tiêu 95%.
Draft đầu đã thử 12 case, nhưng cấu hình cuối chưa nghiệm thu 3 case Extreme.

### Mốc lịch sử trước 0.4.0

| Thành phần | Bản 0.2.0 | Baseline 0.3.0 | Kế hoạch đặt ra ở mốc 0.3.0 |
|---|---|---|---|
| Giữ làn, sửa hướng, preview cua | Có | Giữ lại | Tâm làn liên tục, hiệu chỉnh động học |
| Tốc độ | Trần cố định 22 km/h | Trần đường thoáng 45 km/h; nhiều giới hạn động | Profile có cấu hình và điều khiển êm |
| Phanh trước cua | Preview gần | Xét heading các waypoint hợp lệ trong lookahead | Kiểm chứng đường cong/làn tiếp nối |
| Tiếp cận giao lộ | Chưa có luật riêng | Giới hạn tiếp cận 24 km/h | Nhường đường, xác định xung đột, chọn thời điểm qua |
| Xe phía trước | Chưa dùng traffic context | Bám xe theo tiếp tuyến làn, bumper gap, TTC khẩn cấp | Chiếu theo hành lang cong, tracking, cut-in |
| Vật cản từ LiDAR | Chưa có | Chưa có | Kiểm chứng sensor rồi tích hợp |
| Chuyển làn/vượt | Chưa có | Chưa có | Chỉ thêm sau khi giữ làn và tương tác giao thông đạt |
| Vạch vàng/trắng, liền/đứt | Chưa có nhãn công khai | Vẫn chưa có | Extension hình học/vạch đường dùng chung |
| Trạng thái qua nhiều bước | Stateless | Stateless | State machine, hysteresis, timer; đổi metadata khi triển khai |

Không diễn giải cột kế hoạch thành chức năng đã có.

## 2. Những điểm nền tảng cần kiểm chứng

1. **Đã sửa ở 0.4.0:** `extract_task_context()` trước đây lấy `lanes[0]`, gây
   waypoint lệch làn. Nay bắt đầu bằng `navigation.current_lane`, chọn nối theo
   endpoint/heading và ngừng ở gap không hợp lệ. Đây chưa phải dữ liệu làn đích,
   quyền đổi làn hoặc bảo đảm quỹ đạo qua mọi giao lộ.
2. `asin(2*obs[2]-1)` chỉ giải mã heading đúng nhánh khi xe lệch dưới 90°.
   Không dùng nó để kết luận hướng xe khi quay ngược/mất định hướng.
3. Hệ ego có +y sang trái; lateral của lane MetaDrive dương sang phải.
   Runtime đã đo steering dương quay trái. Các tầng phải chuyển dấu nhất quán.
4. 80 km/h là hệ số giải mã tốc độ của cấu hình xe, không phải tốc độ được
   phép chạy trên mọi đường. Không sửa hệ số này để tăng tốc xe.
5. `decision_point.available_turns` chỉ cho biết đường nào tồn tại. Nó không
   cung cấp quyền ưu tiên, đèn tín hiệu hay thời điểm an toàn để đi qua.
6. `TrafficContextV1` chứa tối đa 8 actor trong bán kính 50 m. Không có actor
   không chứng minh đường trống; phải xét mask, overflow và giới hạn cảm biến.

## 3. Kiến trúc và thứ tự phân xử

```text
AgentInput công khai
  -> giải mã ego + hành lang tuyến + actor + mức đầy đủ của quan sát
  -> tính sai số bám làn và các nguy cơ
  -> chọn trạng thái hành vi bằng luật có ưu tiên
  -> tạo mục tiêu lái và từng giới hạn tốc độ
  -> chọn tốc độ nhỏ nhất và điều khiển ga/phanh
  -> lớp khẩn cấp kiểm tra lần cuối
  -> action + diagnostics
```

Mỗi luật phải có điều kiện vào, điều kiện thoát, thời gian xác nhận, hành động
và lý do ghi log. Giới hạn dọc được kết hợp bằng `min`; không để luật cruise
ghi đè luật nhường đường. Lệnh phanh khẩn cấp vẫn phối hợp với lệnh lái khả thi,
không tự trả lái về 0 nếu làm xe lao khỏi đường.

| Ưu tiên | Nhóm luật | Hành động chính |
|---:|---|---|
| 0 | Nguy cơ va chạm tức thời/không còn quãng phanh | Phanh khẩn cấp, giữ hướng tránh làm xấu tình huống |
| 1 | Dữ liệu điều khiển thiếu hoặc mâu thuẫn | Hạ tốc/controlled stop; không tăng ga từ dữ liệu không rõ |
| 2 | Thân xe sắp cắt biên cấm | Giảm tốc và phục hồi vào hành lang khả thi |
| 3 | Xung đột giao lộ, vòng xuyến, nhập làn | Nhường hoặc dừng trước vùng xung đột |
| 4 | Theo xe, cut-in, chướng ngại trên hành lang | Giữ khoảng cách, giảm tốc/dừng |
| 5 | Chuẩn bị thao tác bắt buộc theo tuyến | Chọn làn/nhánh hợp lệ; không bỏ qua luật 0–4 |
| 6 | Giới hạn cua, độ nhìn thấy, độ ổn định | Hạ tốc đủ sớm, giới hạn độ lớn/tốc độ đổi lái |
| 7 | Đường trống, đủ dữ liệu | Tăng tốc dần tới trần cấu hình |

## 4. Chính sách tốc độ thích ứng

### Cấu hình tốc độ từ 0.3.0, giữ lại ở 0.4.0

| Tình huống | Mục tiêu/giới hạn hiện tại |
|---|---|
| Thẳng, đủ tầm nhìn, không có ràng buộc khác | Tối đa 45 km/h |
| Cua | Tính từ độ cong và gia tốc ngang 1.5 m/s² |
| Tiếp cận điểm phân nhánh | Phanh theo khoảng cách để về tối đa 24 km/h |
| Có xe trước | Giới hạn theo bumper gap, tốc độ tương đối và headway 1.8 s |
| Gần biên làn/đường | Tối đa 8 km/h và tiếp tục sửa lái |
| Đoạn tuyến nhìn thấy ngắn | Giới hạn để còn khả năng phanh trong đoạn hợp lệ |
| Bumper gap <= 3 m hoặc TTC đóng khoảng cách <= 1 s | Phanh -1, không chỉ hạ target |

Ga dương tối đa tăng từ 0.4 lên 0.55. Giữ giới hạn gia tốc ngang 1.5 m/s²,
không đồng thời nới tốc độ vào cua khi nâng trần đường thẳng.
`Stage1RuleBasedAgent(cruise_speed_kmh=35.0)` cho phép chọn trần thấp hơn.
CLI hiện chưa có cờ riêng để ghi đè tham số này.

Các công thức, với tốc độ nội bộ theo m/s:

```text
v_curve = sqrt(a_lateral / max(abs(curvature), epsilon))
d_usable = max(0, d_to_constraint - response_time * v_current - buffer)
v_approach = sqrt(v_at_constraint² + 2 * comfortable_deceleration * d_usable)
v_target = min(v_cruise, v_curve_approach, v_visibility,
               v_front_vehicle, v_junction_approach, v_recovery)
```

Tham số hiện tại: response 0.4 s, buffer 3 m, giảm tốc mô hình 2.5 m/s².
Đây là mô hình thiết kế, phải đo phản ứng phanh thực tế của MetaDrive.

### Quy tắc cho bản đầy đủ

- Nếu có speed limit được môi trường cung cấp xác thực, thêm nó vào `min`.
  Không suy speed limit từ tên map, tier Easy/Hard hoặc kết quả evaluator.
- Khi xe đang bám làn tốt, hết cua và nguy cơ đã hết, cho tăng tốc trở lại;
  không giữ trần phục hồi sau khi nguyên nhân đã biến mất.
- Phanh sớm cho cua phía trước bằng bao quãng phanh; không ép về tốc độ cua
  ngay khi cua còn rất xa, cũng không đợi xe đã lệch làn mới phanh.
- Đặt giới hạn gia tốc/jerk, chống tích phân bão hòa nếu dùng PI/PID.
  Hysteresis chỉ làm mượt việc bỏ giới hạn, không trì hoãn phanh khẩn cấp.
- Khi overflow actor hoặc quan sát bị che, giảm trần theo khả năng dừng;
  không coi thiếu actor là đường trống.
- Không dùng tốc độ tối thiểu bắt buộc để xe lao qua một nguy cơ đang còn.

Không đặt ngay trần 60–80 km/h chỉ vì xe hỗ trợ tốc độ đó. Với giảm tốc mô hình
2.5 m/s², response 0.4 s và buffer 3 m, 45 km/h cần khoảng 39.3 m để dừng,
60 km/h cần khoảng 65.2 m. Vì lookahead hiện tại chỉ 50 m, mức 60 km/h phải đợi
tăng tầm dữ liệu hợp lệ và kiểm chứng lại bộ điều khiển, không bật mặc định.

## 5. Các luật cần triển khai để hoàn thiện hành vi

Các ngưỡng trong phần này là điểm bắt đầu thử nghiệm TRAIN, chưa phải tham số
đã chứng minh đạt cho mọi map.

### A. Giữ làn và cua

| ID | Điều kiện | Luật/hành động |
|---|---|---|
| L01 | Tuyến/làn liên tục, sai số nhỏ | Bám tâm làn hiện tại; sửa heading và lateral bằng điều khiển phản hồi |
| L02 | Tốc độ tăng | Tăng preview trong phạm vi hợp lệ; giảm gain gây dao động, giới hạn steering rate |
| L03 | Cua sắp tới | Ước lượng độ cong trên nhiều đoạn; phanh trước điểm bắt đầu cua |
| L04 | Đầu/đuôi gần biên cấm | Xét footprint thân xe, không chỉ điểm tâm; giảm tốc và sửa vào trong |
| L05 | Sai số giảm nhưng chưa ổn định | Giữ chế độ phục hồi đến khi clearance và heading vượt ngưỡng thoát |
| L06 | Đổi đoạn đường/làn tham chiếu | Chọn tiếp nối theo kết nối, vị trí và hướng; không nhảy sang lanes[0] |
| L07 | Hướng xe mơ hồ hoặc quay ngược | Không giải thích arcsin như heading đầy đủ; dừng có kiểm soát và báo degraded |

Tích hợp với `LANE_MARKING_AGENT_DESIGN.md`. Không thêm tìm kiếm quỹ đạo lớn
vào Stage 1; luật biên và mô hình động học ngắn hạn phải minh bạch, giới hạn.

### B. Xe trước, vật cản và xe cắt vào

| ID | Điều kiện | Luật/hành động |
|---|---|---|
| T01 | Actor nằm trên hành lang cong được chọn | Tính khoảng dọc theo tuyến và khoảng cách giữa hai bumper |
| T02 | Xe trước chậm | Bám xe bằng headway và mô hình dọc kiểu IDM hoặc luật khoảng cách đã hiệu chỉnh |
| T03 | Khoảng cách đóng nhanh | Dùng TTC cùng stopping distance; không chỉ một ngưỡng mét cố định |
| T04 | Xe trước đứng yên | Dừng cách bumper một khoảng dự phòng; không tiếp tục bò do target có floor |
| T05 | Xe lấn/cắt vào hành lang | Dự báo footprint và tốc độ ngang; hạ ga trước khi tâm actor vào làn |
| T06 | Xe ở làn bên cạnh hoặc phía sau | Không phanh chỉ vì gần về Euclidean distance; xét đường dự định đi |
| T07 | LiDAR thấy vật cản trong hành lang | Giảm tốc/dừng theo khoảng dọc; kiểm chứng hướng tia, range, mask và phản hồi |
| T08 | Actor biến mất/đổi thứ tự | Association bằng vị trí/vận tốc, TTL ngắn; không dùng index slot làm ID |
| T09 | Context overflow/độ tin cậy giảm | Tăng dự phòng, giảm tốc; giữ giới hạn cảm biến trong log |

Không gộp tia LiDAR vào actor có phân loại giả. Tia khoảng cách chỉ cho biết
một bề mặt phản xạ; khả năng phát hiện static object phải kiểm tra với collision
mask và physics world đang dùng. Đối chiếu cảm biến bằng các vật cản thử biết trước.

### C. Ngã ba, ngã tư và rẽ theo tuyến

| ID | Điều kiện | Luật/hành động |
|---|---|---|
| J01 | Sắp đến điểm phân nhánh | Đọc hướng tuyến, khóa nhánh được chọn qua giao lộ; rule-based không random |
| J02 | Đã biết stop/yield line hoặc vùng xung đột | Hạ tốc/dừng trước vạch/vùng, có xét chiều dài đầu xe |
| J03 | Xe cắt ngang có khoảng chiếm vùng xung đột chồng nhau | Nhường; tính ETA và khoảng thời gian footprint đi qua |
| J04 | Actor đứng yên nhưng có thể đi vào vùng xung đột | Dùng dự phòng trạng thái; không suy quyền đi chỉ từ tốc độ 0 |
| J05 | Khoảng trống đủ cho toàn bộ thân xe đi qua | Chuyển từ yield sang traverse; tham số gap ban đầu 3–4 s để thử |
| J06 | Phía sau giao lộ bị chặn | Chờ trước giao lộ, không tiến vào rồi chắn ngang |
| J07 | Đã vào giao lộ | Giữ nhánh và hành lang; không đổi quyết định mỗi frame khi checkpoint thay đổi |
| J08 | Tầm nhìn chưa đủ | Bò thăm dò 3–5 km/h chỉ khi có hành lang và quãng dừng hợp lệ; nếu không thì chờ |
| J09 | Đã ra khỏi vùng giao lộ và ổn định | Nhả giới hạn giao lộ dần, quay về theo xe/cruise |

Ngưỡng TTC theo trục dọc hiện tại không thay được dự báo va chạm cắt ngang.
Không khẳng định agent hiểu luật ưu tiên pháp lý khi input không có biển/đèn.
Nếu simulator không mô hình hóa biển/đèn, dùng luật nhường bảo thủ dựa trên
xung đột hình học và ghi rõ giả định đó trong thí nghiệm.

### D. Vòng xuyến, nhập/tách làn và chuyển làn

| ID | Điều kiện | Luật/hành động |
|---|---|---|
| M01 | Tiếp cận vòng xuyến | Hạ tốc theo curvature, chọn tuyến ra và đánh giá xe đang lưu thông |
| M02 | Đang chạy trong vòng xuyến | Bám làn/tuyến liên tục, giữ headway; không cắt tâm vòng để rút ngắn đường |
| M03 | Đến lối ra đúng tuyến | Ra theo connector đã khóa; không đổi lane đột ngột để đuổi checkpoint |
| M04 | Nhập làn | Kiểm tra cả xe trước và sau ở làn đích, TTC và quãng phanh; không chỉ xem phía trước |
| M05 | Bắt buộc chuyển làn để theo tuyến | Chỉ chuẩn bị khi biên cho phép, connector hợp lệ và gap đủ |
| M06 | Chuyển làn đang thực hiện | Duy trì mục tiêu và tiến độ; xác định rõ trước/sau điểm còn có thể hủy |
| M07 | Gap xấu đi trước điểm cam kết | Hủy êm về làn cũ nếu còn an toàn; nếu không, chọn hành vi ít nguy cơ nhất |
| M08 | Bị xe chậm cản | Ưu tiên bám xe; chỉ bật vượt sau khi luật M04–M07 được xác minh |

Ban đầu ưu tiên chuyển làn bắt buộc theo tuyến. Vượt để tối ưu tốc độ là hạng
mục sau, vì nó làm tăng đáng kể số điều kiện xung đột và kiểm thử.

### E. Dừng, lỗi dữ liệu và kẹt xe

| ID | Điều kiện | Luật/hành động |
|---|---|---|
| F01 | Thiếu waypoint ngắn hạn | Hạ tốc; chỉ giữ hành lang cũ trong TTL đã công bố, không dùng cache vô hạn |
| F02 | Mất dữ liệu kéo dài/hành lang không khả thi | Dừng có kiểm soát, diagnostics chỉ rõ dữ liệu thiếu |
| F03 | Tốc độ thấp kéo dài | Phân biệt dừng do xe trước/yield với lỗi bám tuyến bằng lịch sử dữ liệu công khai |
| F04 | Chờ lâu nhưng vẫn còn nguy cơ | Tiếp tục chờ, không có luật timeout ép tăng ga |
| F05 | Cuối route nằm trong lookahead | Giảm tốc theo đoạn đường còn nhìn thấy; không coi flag này là đã đến đích |
| F06 | Không thể phục hồi mà vẫn trong hành lang | Báo blocked/degraded; không tự lùi hoặc quay đầu qua vạch cấm |

Không dùng reward, route_completion, crash flag, terminal reason hoặc split/tier
làm input ra quyết định. Chúng chỉ xuất hiện trong harness đánh giá bên ngoài.

## 6. State machine và bộ nhớ

Các trạng thái đề xuất: `CRUISE`, `FOLLOW`, `APPROACH_JUNCTION`, `YIELD`,
`TRAVERSE_JUNCTION`, `MERGE`, `LANE_CHANGE`, `RECOVER_LANE`, `DEGRADED_STOP`,
`BLOCKED`. Emergency là lớp ghi đè có ưu tiên cao hơn mọi trạng thái.

- Điều kiện chuyển dựa trên hình học/tốc độ/khoảng cách công khai.
- Dùng ngưỡng vào/ra khác nhau; xác nhận giải phóng nguy cơ khoảng 0.3–0.5 s
  để thử, khẩn cấp có hiệu lực ngay ở bước hiện tại.
- Lưu branch/lane target khi cam kết, PID memory, actor track TTL, timer phục hồi.
- Khi triển khai bộ nhớ, đổi `stateful_within_episode=True` ở descriptor và registry.
  `reset()` phải xóa toàn bộ trạng thái episode; deterministic replay phải giữ nguyên.
- Không dùng số bước tuyệt đối để đoán vị trí/ngã rẽ theo map seed.

## 7. Dữ liệu cần bổ sung qua platform

Giữ vector 259D nguyên nghĩa. Các dữ liệu mới phải qua snapshot bất biến có
phiên bản/profile, dùng cùng quyền thông tin cho các phương pháp so sánh.
Policy không được tự đọc engine, vehicle, live lane hoặc road network.

| Dữ liệu | Hiện có | Phần cần thêm/kiểm chứng |
|---|---|---|
| Ego speed, lateral, heading | Observation chuẩn hóa | Giá trị có đơn vị, heading đầy đủ, yaw rate, kích thước/cấu hình lái xác thực |
| Tuyến phía trước | 20 waypoint, spacing 2.5 m | Centerline đúng làn, connector liên tục, branch theo route và validity rõ |
| Các hướng ngã rẽ | `RoadDecisionPointV1` | Maneuver theo tuyến, entry/exit, conflict zone/stop line nếu có nguồn |
| Xe quanh ego | 8 actor, 50 m | Association/ID công khai nếu platform hỗ trợ; quan sát liên quan theo hành lang |
| Vạch/biên | Khoảng cách biên ước lượng | Metadata màu/loại và polyline biên, nhãn UNKNOWN khác NONE |
| Làn bên cạnh | Chưa có cấu trúc | Centerline, connectivity, hướng di chuyển, quyền đổi làn |
| Tốc độ cho phép/đèn/biển | Chưa có trong input | Optional field chỉ khi simulator có nguồn xác thực, thiếu thì None/UNKNOWN |

Extension hiện ở schema 1.2.0 pending audit: decision point, waypoint semantics,
ego-motion, route passage và actor road hypotheses. Vector 259D, actor array
8×7 và range 50 m giữ nguyên. Dữ liệu làn bên cạnh/biên có nhãn chưa được bổ sung.
Không coi hash Gate 6 lịch sử là chứng nhận cho các quyền thông tin mới.
Trước benchmark phải audit/version lại chuỗi contract phụ thuộc; không tự bật
`benchmark_eligible` chỉ vì unit tests qua.

## 8. Lộ trình triển khai theo điều kiện nghiệm thu

| Giai đoạn | Công việc | Điều kiện chuyển tiếp |
|---|---|---|
| P0 — Chốt phép đo | Kiểm chứng dấu, đơn vị, waypoint đúng làn, sensor range/mask; xác lập baseline | Có fixture và probe cho thẳng, cua hai phía, lane nối |
| P1 — Tốc độ thích ứng | Phanh trước cua/giao lộ, visibility, bám xe cơ bản, AEB, diagnostics | Đã có 0.3.0; mở rộng kiểm thử trước khi coi là hoàn tất |
| P2 — Hành lang và tương tác xe | Centerline liên tục, footprint, traffic trên cua, LiDAR obstacle, cut-in | Không tăng crash/boundary contact trên suite đã chốt; dừng và khởi hành ổn định |
| P3 — Giao lộ/vòng xuyến | State machine, khóa nhánh, conflict ETA, yield, tránh kẹt trong giao lộ | Qua T/X/O trên nhiều seed; chờ đúng và thoát khi gap đủ |
| P4 — Nhập/chuyển làn | Metadata biên, gap trước/sau, thao tác bắt buộc, hủy | Không cắt biên cấm; không gây va chạm sau khi chuyển |
| P5 — Củng cố và đóng băng | Tinh chỉnh TRAIN, chạy VALIDATION đúng protocol, audit contract | Khóa cấu hình, nguồn và version rồi mới TEST đúng suite |

Không gán lịch ngày cứng trước khi đo độ khó P0/P2; mỗi giai đoạn có artifact,
test và tiêu chí đạt độc lập. Nếu một chức năng chưa đạt, giữ trạng thái planned.

Tổ chức code khi lớn hơn bản hiện tại:

```text
src/agents/rule_based_agent.py          # Lifecycle và phối hợp các luật
src/agents/rule_based/config.py        # Tham số có đơn vị, profile đã đóng băng
src/agents/rule_based/observation.py   # Giải mã public input, không đọc live simulator
src/agents/rule_based/lane_control.py  # Sai số làn, heading, steering
src/agents/rule_based/speed_control.py # Speed envelope và ga/phanh
src/agents/rule_based/traffic_rules.py # Actor corridor, headway, TTC, cut-in
src/agents/rule_based/junction_rules.py
src/agents/rule_based/behavior.py      # State machine và phân xử
```

Chỉ tách module khi triển khai phần tương ứng; không tạo nhiều file rỗng trước.

## 9. Ma trận kiểm thử và tiêu chí đánh giá

### Unit và scenario có điều kiện biết trước

- Đường thẳng, cua trái/phải, cua S, chuyển straight/curve, làn hẹp.
- Xe đứng yên/chạy chậm/chạy cùng tốc độ; xe bên cạnh/phía sau không gây phanh giả.
- Cut-in ở tốc độ ngang khác nhau, actor bị mất/đổi slot, context overflow.
- Vật cản trong/ngoài hành lang, khoảng cách vừa đủ/không đủ để dừng.
- Giao lộ trống/có xe cắt ngang/xe đứng chờ/exit bị chặn; route rẽ và đi thẳng.
- Vòng xuyến: xe đang lưu thông, lối vào/ra, giữ làn cong.
- Merge: xe sau tiến nhanh, khe hẹp, biên cấm, bỏ thao tác và ổn định lại.
- Mất waypoint/biên, heading mơ hồ, reset và replay.

Không yêu cầu tránh được mọi va chạm từ một trạng thái ban đầu vật lý không
thể cứu. Với tình huống cố ý không đủ stopping distance, nghiệm thu là can
thiệp khẩn cấp đúng bước, không xuất action nguy hiểm hơn và ghi nguyên nhân.

### Đánh giá simulator

1. Giữ các case so sánh trước/sau giống nhau về geometry và environment seed.
2. TRAIN mở rộng: 12 sequence family × 3 geometry seed TRAIN × 3 environment
   seed = 108 episode; chốt danh sách và quy tắc chọn trước khi chạy.
3. Chỉ hiệu chỉnh trên TRAIN/VALIDATION theo protocol; không dùng TEST để chỉnh.
4. VALIDATION phải là full suite 96 case; TEST full suite 60 case theo launcher.
   Agent deterministic không cần nhân bản bằng agent seed như random baseline.
5. Báo cáo theo family và tổng thể: clean success, completion, crash, out-of-road,
   timeout, thời gian mô phỏng, tốc độ trung bình trên episode thành công,
   sai số làn, minimum gap/TTC quan sát được, jerk và độ trễ act p95.
6. Không kết luận bản nhanh hơn từ episode crash sớm hoặc tốc độ tối đa đơn lẻ.
   Ghi cả trường hợp thụt lùi và mức biến thiên; tập 7 case chỉ là smoke test.

Mục tiêu phát triển ban đầu cho suite 108 case: clean success >= 95%, không có
va chạm/vượt biên trong các fixture có điều kiện tránh được, và giảm thời gian
trên các case cùng thành công mà không tăng tỷ lệ lỗi an toàn. Đây là mục tiêu,
không phải kết quả hiện đã đạt hoặc bảo đảm an toàn thống kê.
Ngân sách act của platform là 100 ms ở 10 Hz; đặt mục tiêu p95 < 10 ms cho luật.

Diagnostics cần đủ để truy nguyên: state, active rules, winning speed constraint,
giới hạn tốc độ theo từng nguyên nhân, lane target, branch target, gap/TTC,
clearance, validity/overflow và raw/final control. Không nhét đối tượng simulator.

## 10. Kết quả lịch sử bước P1 — 0.2.0 so với 0.3.0

So sánh headless ngày 2026-10-30, cùng environment seed 0, toàn bộ case là TRAIN.
`steps × 0.1 s` là thời gian mô phỏng; wall-clock phụ thuộc máy nên không dùng
để suy ra xe chạy nhanh hơn. Dữ liệu: `results/audits/rule_based/speed_policy_comparison.csv`.

| TRAIN case | 0.2.0: bước / tốc độ TB | 0.3.0: bước / tốc độ TB | Kết quả |
|---|---|---|---|
| Easy/SCSS/0 | 745 / 21.15 km/h | 450 / 35.00 km/h | Cả hai đến đích, không crash/chạm biên |
| Easy/SCS/0 | 680 / 21.07 | 418 / 34.21 | Cả hai đến đích, không crash/chạm biên |
| Easy/SCCS/0 | 971 / 21.35 | 597 / 34.66 | Cả hai đến đích, không crash/chạm biên |
| Easy/SCCS/7 | 838 / 21.24 | 539 / 33.00 | Cả hai đến đích, không crash/chạm biên |
| Medium/SCTCS/2 | 933 / 20.70 | 674 / 28.63 | Cả hai đến đích, không crash/chạm biên |
| Medium/SCXCS/14 | 641 / 19.97 | 538 / 23.78 | Cả hai đến đích, không crash/chạm biên |
| Hard/SCXOCS/0 | 641 / 21.01 | 642 / 24.56 | Cả hai crash; completion 0.6124 → 0.7161 |

Sáu case cùng thành công giảm thời gian mô phỏng khoảng 16.1–39.6%.
Tốc độ tối đa bản mới đạt khoảng 44.1–45.0 km/h, so với 22.0 km/h bản cũ.
Sai số lateral lớn nhất bản mới trong các case thành công khoảng 0.462 m;
một số case Easy tăng sai số dù vẫn không chạm vạch: phải theo dõi khi mở rộng.
Hard vẫn crash; chưa giải quyết triệt để interaction và roundabout bằng P1.
Không công bố 0.3.0 là agent hoàn thiện hoặc benchmark-ready từ kết quả này.

## 11. Tài liệu tham khảo và nguồn tham số

- [Stanley — bài gốc của nhóm Stanford](https://robots.stanford.edu/papers/thrun.stanley05.html):
  tham khảo phân tầng và điều khiển heading/lateral. Bản hiện tại là heuristic
  lấy cảm hứng, không tuyên bố tái hiện đầy đủ controller Stanley.
- [Treiber, Hennecke, Helbing — IDM](https://arxiv.org/abs/cond-mat/0002177):
  cơ sở lựa chọn mô hình bám xe cho P2. Luật khoảng cách 0.3.0 chưa phải IDM.
- [CARLA Agents 0.9.15](https://carla.readthedocs.io/en/0.9.15/adv_agents/):
  tham khảo tách điều khiển, hành vi, target speed và safety time. Không sao chép
  các ngưỡng CARLA sang MetaDrive như thể đã được kiểm chứng.
- Nguồn local: `AGENT_DEVELOPMENT_GUIDE.md`, `STAGE_ROADMAP.md`,
  `LANE_MARKING_AGENT_DESIGN.md`, source pinned MetaDrive 0.4.3 và các audit.

Các giá trị 45/24 km/h, headway 1.8 s, TTC 1 s và các ngưỡng nghiệm thu là
lựa chọn thiết kế cho dự án này. Cần tiếp tục hiệu chỉnh và kiểm chứng TRAIN;
chúng không phải giới hạn giao thông pháp lý hoặc giá trị an toàn được chứng nhận.
