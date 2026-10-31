# P4 — chuyển làn theo tuyến và vượt xe

Triển khai 0.6.0, phát triển trên TRAIN. Đây là quy tắc có giới hạn quan sát và
dự báo, không phải bảo đảm an toàn cho mọi hành vi của xe khác. Đọc kết quả và
giới hạn trong báo cáo 0.6.0; P5/audit/benchmark vẫn là công việc riêng.

## Dữ liệu chung

Schema 1.3.0 bổ sung `TaskContextV1.lane_context`, giữ nguyên vector 259D,
traffic 8×7 và lookahead tuyến 20×2.5 m. Mọi agent dùng profile chính đều nhận
cùng dữ liệu; CORE_ONLY che toàn bộ lane context. Dữ liệu là các dataclass frozen,
tuple số hữu hạn, không có simulator handle, vị trí toàn cục hay thông tin đánh giá.

Tối đa ba làn cùng chiều trên edge hiện tại: current, left, right. Mỗi làn có
ID cục bộ, bề rộng, đoạn còn lại tối đa 50 m, kiểu hai biên thực sự được dựng,
khả năng nối tuyến trong cửa sổ quan sát, và tối đa 28 mẫu centerline với khoảng
cách có dấu [-15,50] m. Trong PGBlock, biên chung bên phải lấy từ LEFT của làn
bên phải; đọc nominal RIGHT của làn hiện tại là sai khi metadata khác nhau.

Chỉ `BROKEN_WHITE` khớp ở cả hai phía được phép cắt. `SOLID_WHITE`,
`SOLID_YELLOW`, `EDGE`, `UNKNOWN`, thiếu metadata hoặc bất đồng đều bị từ chối.
Neighbor chỉ lấy từ cùng directed edge, không lấy làn ngược chiều ở gần.

`route_compatible` là thông tin **trong cửa sổ cục bộ**: khi endpoint chưa nằm
trong 50 m, làn vẫn còn chạy được trong cửa sổ; khi nhìn thấy endpoint, yêu cầu
gap <=0.3 m và lệch heading <0.35 rad tới next checkpoint edge. Chưa xác định
được tuyến thì dùng None. Current không nối được còn neighbor nối được sẽ tạo
`required_side`. Không nhảy ngang waypoint qua gap một bề rộng làn như trước;
extractor dừng trước gap >0.3 m. Không có làn phù hợp thì giảm tốc/dừng trước end.

## Thứ tự quyết định

1. Chuyển làn bắt buộc theo tuyến.
2. Trở lại làn trước sau khi đã quan sát thấy xe bị vượt nằm sau đủ xa, cooldown
   đã hết, và mọi điều kiện an toàn được kiểm tra lại.
3. Vượt khi giới hạn bám xe <65% cruise liên tục >=1 s và có làn hợp lệ.
4. Tiếp tục bám làn/bám xe khi các điều kiện không đủ. Không ép vượt do timeout.

Chỉ bắt đầu khi heading lệch <=0.2 rad, offset <=0.4 m và tốc độ <=30 km/h;
giai đoạn chuẩn bị hạ trần về 24 km/h. Chiều dài chuyển làn max(16 m, v×3.5 s),
giới hạn 30 m, cần thêm 10 m trước cuối cả hai làn. Từ chối đoạn cong/phân kỳ
có thay đổi hướng >0.35 rad trong dữ liệu. Không khởi tạo trong passage/branch
đã khóa hoặc cách passage/fork dưới 35 m.
Riêng vượt/trở lại làn tùy chọn yêu cầu không có passage/fork trong toàn bộ
50 m đang quan sát (`lane_change_overtake_junction_buffer_m`); không vượt vào
approach chỉ để rồi phải dừng ở làn mới sát nút khi quyền ưu tiên chưa được biết.

Ngưỡng approach 35 m là điều kiện **khởi tạo**. Một maneuver đã bắt đầu hợp lệ
trên cùng edge phải được hoàn tất trước cuối làn; thấy junction vừa vào ngưỡng
approach không tự làm xe dừng giữa hai làn. Junction đã khóa thực sự, sensor
overflow, vạch/geometry thay đổi và xung đột vẫn chặn tiếp tục. Có kiểm thử hồi
quy cho sự khác nhau giữa bắt đầu và hoàn tất thao tác.

## Gap và thân xe

Làn đích kiểm tra cả trước lẫn sau, có kích thước và hướng actor. Gap yêu cầu:
4 m + thời gian phản ứng × vận tốc xe theo sau + max(headway 1.8 s,
chênh lệch quãng phanh). TTC đóng khoảng phải >=4 s. Actor bên hông, track ghost
và uncertainty đều tham gia; overflow actor/path hoặc LiDAR không khả dụng chặn
khởi tạo. Không dùng slot như actor ID và không đọc navigation của xe khác.

Quỹ đạo không gian quintic nối từ tâm ego thực tới centerline làn đích, đạo hàm
đầu/cuối bằng không trên đường thẳng. Đường được giữ trong frame odometry của
episode; chuyển nearest lane của navigation không đổi mục tiêu. Dùng cùng P1
controller, giới hạn cua/visibility, P2 traffic/LiDAR/AEB và outer-road guard.
Vạch đứt nằm trong hành lang chuyển làn không kích hoạt lane-recovery 8 km/h.
Ước lượng thân xe tới biên đường vẫn dùng heading của đường đo được, không dùng
heading sai số tới quỹ đạo chuyển làn.

Guard dự báo footprint theo ba profile tiến dọc: giữ tốc, phanh êm, tăng tốc
giới hạn 1.2 m/s², với CV/CT/road hypotheses của actor trong 4 s. Có margin do
track, uncertainty theo thời gian, chuyển động giữa các sample 0.2 s, offset và
heading của thân xe thực so với đường mẫu. LiDAR không thuộc actor đã biết
được kiểm tra trên hành lang quét, không tự vượt một vật không phân loại.

## Lifecycle

`KEEP_LANE → PREPARE_CHANGE → CHANGE_LANE → KEEP_LANE`, thêm
`ABORT_CHANGE` và `BLOCKED_CHANGE`. Mục tiêu/đường không được chọn lại mỗi tick.
Progress đo theo khoảng cách tới hai tâm làn; commit từ 22% bề rộng hoặc sớm hơn
khi envelope thân xe theo hướng thực đã tới vạch chung (có buffer 0.05 m).
Xe đang xoay không được xem là còn có thể hủy chỉ vì tâm chưa qua 22%.
Gap xấu trước commit chỉ được
hủy khi đường quay về và gap làn cũ cũng an toàn. Không an toàn để quay về thì
giữ mục tiêu và phanh. Sau commit không tự đảo mục tiêu sang làn cũ; phanh theo
guard/P2/AEB khi cần, tiếp tục đường đã chọn khi guard trở lại an toàn.

Hoàn tất sau offset đích <0.25 m, heading <0.12 rad trong 0.5 s; cooldown 6 s.
Thiếu geometry/odometry giữa thao tác: giữ đường đã quan sát và dừng, không
cấp phép vượt tiếp bằng dữ liệu thiếu. Reset xóa toàn bộ maneuver/return/track.

## Chứng cứ và giới hạn

Diagnostics `lane_change` nằm trong budget 50 key chung, ghi phase/reason,
source/target ID, progress/commit, gap trước/sau, TTC, veto và actor xung đột.
Đã bỏ key curvature dư ở top level để giữ budget, không tăng giới hạn contract.
Trace/replay phục hồi cả schema cũ lẫn mới bằng input module đã snapshot.
Khi đang maneuver, `lateral_offset_right_m` và `heading_error_rad` là lỗi tới
đường đang bám. Offset/hướng của làn vật lý vẫn có trong core[8]/ego-motion;
`lane_clearance_estimate_m` và road envelope dùng phép đo vật lý này. CSV
`max_abs_lane_offset_m` lấy từ simulator bên ngoài policy, theo nearest lane.

Unit tests bao phủ metadata renderer, mandatory route, trái/phải, target không
đổi khi nearest lane đổi, trước/sau, xe bên hông, LiDAR, overflow, vạch cấm,
đoạn ngắn, tốc độ chuẩn bị, hủy/commit/reset và hoàn tất/cooldown. Physical probes
ghi riêng can thiệp traffic/spawn/route-availability, geometry/seeds, từng input,
action canonical, vị trí/thân xe và cờ an toàn; không cộng vào driving success.

Chưa hỗ trợ đổi nhiều làn một lần, đổi trong giao lộ/vòng xuyến, vượt qua vạch
cấm, ép chen qua gap hay đổi trên cua gắt. Quyền ưu tiên pháp lý/đèn/biển chưa có
nguồn công khai được audit. Mandatory yêu cầu một làn kề đã nối được tuyến trong
cửa sổ cục bộ; việc lập chuỗi đổi qua nhiều làn để tới một connector xa cần
planner hướng tuyến riêng. Xe ngoài 50 m hoặc tương lai bất ngờ vẫn là giới hạn
cảm biến/dự báo. Postcommit stop là chính sách bảo thủ, không phải chứng minh
tối ưu rủi ro giữa mọi hành động khả dĩ. P5 cần chạy rộng và audit trước benchmark.
