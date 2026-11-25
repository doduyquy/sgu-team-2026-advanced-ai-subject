# Rule-based 0.5 — thiết kế P2/P3

Phạm vi: simulator MetaDrive 0.4.3 đã pin, public profile STATE_DECISION_V1,
SANDBOX/AUDIT. Đây không phải chứng nhận an toàn giao thông thực hoặc benchmark.
Tiến trình thử và lỗi đã gặp: [work log](RULE_BASED_V0_5_0_WORKLOG.md).

## Dữ liệu và quyền thông tin

- Giữ observation 259D, traffic context 8 actor trong 50 m, route 20 waypoint
  trong 50 m. Không đưa tier, seed, reward, crash hay route completion vào policy.
- Schema 1.2.0 thêm optional `EgoMotionV1`: displacement và signed heading delta
  từ control step trước, velocity trong ego frame và signed lane heading error.
  Policy tự tích lũy frame tương đối từ reset; không nhận tọa độ/global heading.
- `TaskContextV1.route_passage` chứa ID connector/branch cục bộ, loại section,
  maneuver, khoảng cách entry/exit. Exit không nằm trong lookahead là None.
- Với vòng xuyến, adapter xét overlap hình học các làn ở 20 m cuối approach
  để tìm `yield_distance_m`; entry node không được giả làm stop line pháp lý.
  `conflict_exit_distance_m` là hết inlet merge; branch vẫn được khóa đến fork.
- `TrafficContextV1.motion_paths` map-match pose/heading quan sát vào hình học
  làn và liệt kê tối đa 8 đường tiếp nối có thể có cho mỗi actor. Các nhánh
  đều là giả thuyết, không đọc lane/navigation/policy của xe khác. Mỗi path
  chỉ nằm trong bán kính ego 50 m và dài tối đa 50 m từ actor; có overflow flag.
- CORE_ONLY_V1 che các extension này. Tất cả agent dùng cùng adapter/profile;
  policy không đọc engine, lane, vehicle hay network trực tiếp. Contract còn
  pending audit; không thay hash chứng nhận lịch sử hoặc bật benchmark.

## P2 — theo dõi và dự báo actor

`rule_based_memory.py` duy trì frame odometry và actor tracks theo episode.
Association dùng dự đoán vị trí, vận tốc, heading, kích thước; ghép một-một với
tie-break xác định. Slot không phải ID. ID nội bộ chỉ dùng diagnostics/bộ nhớ,
không phải danh tính ground truth từ simulator.

- Gate vị trí 2.5 m; TTL 0.6 s; track bị mất vẫn được dự báo với margin tăng.
- Bù quay/tịnh tiến ego trước khi đo yaw của actor. Yaw rate giới hạn ±1.2 rad/s,
  lọc và chỉ dùng khi có đủ lịch sử, ego-motion xác thực, association không mơ hồ.
- Dự báo vận tốc không đổi và quay đều khi có bằng chứng actor đang rẽ.
  Đây là các giả thuyết ngắn hạn, không biết trước ý định thật của actor.
- Khi có road hypotheses, xét mọi nhánh đã xuất; CV/CT free-motion chỉ dùng
  2 s gần trong junction. Ngoài passage, turn model của actor song song gần
  được giữ tới 4 s để xét cut-in, với lateral uncertainty 0.15 m/s và speed cap
  theo thời gian chiếm vùng. P2 horizon tại passage là 3 s, đường thường 4 s;
  P3 dùng road hypotheses 12 s, không kéo đường thẳng xa xuyên qua vòng xuyến.
  Road prediction kết thúc khi hết hình học biết được, không giữ actor đứng
  ở cuối polyline hoặc bịa connector ngoài tầm nhìn.
- Footprint trên corridor cong, bumper gap/headway/TTC và AEB có ưu tiên cao.
- Rear-side cut-in được chiếu lên phần kéo dài có dấu phía sau ego. Chỉ thay
  giới hạn bám xe mềm bằng tăng tốc có giới hạn khi tránh được overlap của mọi
  actor dự báo, vẫn giữ hard stopping/headway phía trước và mọi giới hạn khác.
  Không tăng tốc qua một lệnh dừng/AEB hoặc khi có xe đứng chắn phía trước.
  Luật escape này chỉ điều khiển ở đường thường; trong passage dùng phân xử P3.
- LiDAR hit không thuộc actor đã biết được xét như bề mặt chưa phân loại;
  không tự gán hit thành xe hay tín hiệu giao thông. Actor overflow hạ trần;
  scan không dùng được dẫn tới dừng có kiểm soát.

Probe vật lý riêng spawn TrafficBarrier trong/ngoài đường và gỡ vật sau khi
ego dừng ổn định. Traffic bị tắt có chủ ý; không tính probe vào success driving.

## P3 — vòng đời qua nút

Các phase: CRUISE → APPROACH_JUNCTION → YIELD → COMMIT_JUNCTION →
TRAVERSE_JUNCTION → EXIT_JUNCTION → CRUISE. BLOCKED/DEGRADED_STOP nêu rõ lý do;
phanh khẩn cấp vẫn ghi đè điều khiển ở mọi phase.

1. Đọc tuyến chỉ định, không random nhánh. Dùng surface type unstructured cho
   X/T và connectivity/circular geometry cho merge/fork của vòng xuyến.
2. Tính thời gian ego tới các điểm dọc route với khả năng tăng tốc từ đứng yên;
   không dùng `speed * t = 0` để kết luận ego dừng thì không có xung đột.
3. So khoảng chiếm hành lang của footprint actor trên horizon 12 s, sample 0.2 s,
   gap dự phòng 3 s. Actor đứng yên có thêm giả thuyết khởi hành có giới hạn.
4. Kiểm tra chỗ trống sau vùng xung đột; chờ bên ngoài nếu exit có queue hoặc
   chưa đủ dữ liệu. Vòng xuyến xét từng inlet merge/section, không đòi giữ trống
   toàn bộ vòng cùng lúc.
   Với intersection box bị chặn, thêm 8 m khoảng chờ upstream và xác nhận hết
   queue trong 2 s; buffer này không áp vào inlet merge của vòng xuyến.
5. Chỉ cho commit sau clearance 0.5 s. Khóa branch bằng polyline trong local
   frame; kiểm tra lại permission khi đầu xe còn chưa vào connector. Gap
   xấu đi thì nhường tiếp, vẫn giữ mục tiêu branch.
6. Khi đã vào, giữ branch tới khi đuôi qua exit + buffer. Fresh route chỉ được
   so trên phần giao với cache; waypoint mới ngoài cache không phải mismatch.
   Có thể nối thêm phần route mới liên tục; không sửa prefix đã cam kết.
7. Dữ liệu route lệch/mất kéo dài hơn TTL 1 s thì degraded stop. Chờ quá 10 s
   được ghi BLOCKED, không tự bật ga để thoát nguy cơ hoặc đổi nhánh.

Không có biển/đèn/quyền ưu tiên pháp lý trong input, nên luật yield là dựa
trên hình học và khoảng thời gian. Nhập/chuyển làn/vượt và metadata vạch thuộc
P4. Giới hạn sensor, association và dự báo phải được giữ trong kết luận nghiệm
thu, kể cả khi các fixture và case TRAIN đều đạt.

## Hồ sơ kiểm chứng

Mỗi attempt có source/config hash, source snapshot, resolved case/seed, runtime,
trace public input/action/diagnostics và outcome. Tests/probe/family regressions
được ghi tách biệt; không gộp số episode lặp thành số tình huống độc lập.
Kết luận nghiệm thu chỉ được ghi sau khi các run tương ứng hoàn tất.
