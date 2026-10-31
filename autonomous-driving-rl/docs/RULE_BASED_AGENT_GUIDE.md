# Hướng dẫn Rule-Based Agent, phiên bản 0.6.1

Bản 0.6.1 dọn source và cache; các luật/tham số P4 giữ như 0.6.0. Xem
[hồ sơ bảo trì](SOURCE_CLEANUP_V0_6_1.md). CLI vẫn dùng `--agent stage1_rule_based`.

Kế hoạch phát triển đầy đủ, luật ưu tiên và các phần chưa triển khai:
[RULE_BASED_AGENT_DEVELOPMENT_PLAN_VI.md](RULE_BASED_AGENT_DEVELOPMENT_PLAN_VI.md).
Thay đổi từng phiên bản: [changelog](RULE_BASED_CHANGELOG.md).
Thiết kế P2/P3: [tracking, prediction và junction lifecycle](RULE_BASED_P2_P3_DESIGN.md).
Các lần thử 0.5: [work log](RULE_BASED_V0_5_0_WORKLOG.md).
Kết quả bản đóng băng: [báo cáo experiment 0.5](RULE_BASED_V0_5_0_EXPERIMENT_REPORT.md).

P4 ở 0.6: [thiết kế chuyển làn](RULE_BASED_P4_DESIGN.md),
[work log đầy đủ](RULE_BASED_V0_6_0_WORKLOG.md). Agent ưu tiên chuyển làn theo
tuyến, kiểm tra gap trước/sau và vạch thực tế, giữ target trong thao tác, hủy
trước commit khi an toàn, phanh/giữ target sau commit nếu có nguy cơ. Vượt xe
chậm và trở lại làn cũ chỉ khi dữ liệu quan sát xác nhận đủ khoảng trống.
Kết quả đóng băng: [báo cáo 0.6](RULE_BASED_V0_6_0_EXPERIMENT_REPORT.md),
13/16 TRAIN success như baseline 0.5, không crash/road/solid-boundary và không
mất case thành công; ba Extreme vẫn timeout. P4 đã nghiệm thu trong phạm vi
công bố, P5/audit/benchmark chưa hoàn tất.

## Chạy thử

Tại thư mục `autonomous-driving-rl`, trong environment `metadrive`:

```powershell
python -X utf8 -m src.launcher.cli run --mode SANDBOX --agent stage1_rule_based --tier Easy --render NATIVE
```

`-X utf8` giúp in đường dẫn tiếng Việt khi terminal dùng bảng mã khác UTF-8.
Đổi thành `--render OFF` nếu muốn chạy không mở cửa sổ.

## Các bước quyết định của agent

1. Giải mã tốc độ, hướng làn và độ lệch tâm làn từ `core_observation`.
2. Sửa hướng xe và kéo xe về tâm làn hiện tại.
3. Dùng `relative_heading` của waypoint phía trước để chuẩn bị cua.
4. Tính mức độ đổi hướng phía trước, giảm tốc độ mục tiêu khi cua cong hơn.
5. Tính quãng phanh tới cua/giao lộ và phần tuyến còn nhìn thấy; giới hạn tốc
   độ theo xe trước trên hành lang cong, phanh khẩn cấp khi gap/TTC quá nhỏ.
6. Ước lượng thân xe còn bao nhiêu khoảng trống tới biên làn/đường; nếu gần
   biên, giảm tốc độ mục tiêu nhưng vẫn giữ điều chỉnh lái.
7. Bù ego-motion, ghép actor qua nhiều frame và giữ track bị mất tối đa 0.6 s.
   Dự báo CV/CT cùng các nhánh đường có thể đi; slot không được coi là ID.
8. Đường thường xét cut-in 4 s và thời gian chiếm vùng. Tại passage, P3 xét
   road hypotheses 12 s, giữ free-motion gần và AEB; xác nhận gap 0.5 s,
   khóa branch đến khi đuôi xe ra khỏi section. Chờ xa hơn nếu exit X/T có queue.
9. Xét hit LiDAR chưa thuộc actor đã biết trong hành lang. Hạ tốc khi actor
   overflow; dừng có kiểm soát nếu scan không dùng được hoặc hướng tuyến mơ hồ.
10. Lấy tốc độ nhỏ nhất trong các luật đang có, rồi điều khiển ga/phanh.
    Với rear-side cut-in ở đường thường, chỉ cho tăng tốc thoát nếu dự báo
    còn clear và vẫn thỏa hard front headway/stopping cùng mọi giới hạn đường.

Lệnh lái trước giới hạn:

```python
heading_error = asin(2 * observation[2] - 1)
lateral_right_m = (observation[8] - 0.5) * 4.5
lateral_correction = atan2(k_lateral * lateral_right_m, speed_mps + speed_softening_mps)
steering = k_angle * heading_error + lateral_correction + k_preview * preview_delta
```

Trong runtime đã thử, steering dương làm heading tăng, quay về +y của hệ ego
(trái). Lệch phải cho lateral dương, cần sửa sang trái. Mô tả dấu lái âm=trái
ở một số tài liệu cũ không khớp phép đo này; không đổi dấu tùy ý khi tinh chỉnh.

Từ schema 1.1.1, extractor bắt đầu từ làn xe hiện tại và chọn làn tiếp nối theo
endpoint/heading; không mặc định `lanes[0]`. Nếu nối bị đứt quá bề rộng làn,
không xuất waypoint qua khoảng đứt và không coi đó là hết route thật.
Agent dùng vị trí waypoint để dựng hành lang traffic; điều khiển lái vẫn dựa
trên sai số làn hiện tại và heading preview. Schema còn pending audit.
Schema hiện tại 1.2.0 thêm ego-motion cục bộ, route passage và các giả thuyết
road path của actor. Không nhận tọa độ toàn cục hay tuyến riêng của xe khác;
CORE_ONLY che các extension. Vector 259D, actor array 8×7 và range 50 m giữ nguyên.

## Tham số và cách tinh chỉnh

| Tham số | Mặc định | Chức năng / ảnh hưởng |
|---|---:|---|
| `lookahead_m` | 5 m | Khoảng nhìn trước tối thiểu; lớn hơn chuẩn bị cua sớm hơn |
| `lookahead_time_s` | 0.5 s | Tăng khoảng nhìn trước theo tốc độ |
| `k_angle` | 1.2 | Độ mạnh sửa hướng; quá lớn có thể lắc lái |
| `k_lateral` | 1.5 | Độ mạnh kéo về tâm làn; quá lớn có thể vượt qua tâm |
| `k_preview` | 0.5 | Độ mạnh chuẩn bị cua; quá lớn có thể cắt cua sớm |
| `speed_softening_mps` | 2 m/s | Làm hiệu chỉnh lệch tâm dịu hơn ở tốc độ thấp |
| `max_steering` | 0.5 | Giới hạn độ lớn action lái chuẩn hóa |
| `cruise_speed_kmh` | 45 km/h | Trần thử nghiệm đường thoáng; vẫn chịu các giới hạn khác |
| `comfortable_deceleration_mps2` | 2.5 m/s² | Giảm tốc mô hình để phanh trước cua/giao lộ |
| `response_time_s` | 0.4 s | Dự phòng thời gian phản ứng khi tính quãng phanh |
| `braking_buffer_m` | 3 m | Khoảng dự phòng phía trước ràng buộc |
| `junction_speed_kmh` | 24 km/h | Trần tại điểm phân nhánh; luật xung đột có thể yêu cầu dừng |
| `following_time_s` | 1.8 s | Khoảng thời gian bám xe cơ bản |
| `minimum_following_gap_m` | 3 m | Khoảng cách bumper tối thiểu; nhỏ hơn kích hoạt phanh khẩn cấp |
| `emergency_ttc_s` | 1 s | Ngưỡng TTC đóng khoảng cách kích hoạt phanh khẩn cấp |
| `recovery_speed_kmh` | 8 km/h | Tốc độ mục tiêu khi thân xe gần biên |
| `max_lateral_accel_mps2` | 1.5 m/s² | Giới hạn dùng để chọn tốc độ vào cua |
| `k_speed` | 0.12 | Độ mạnh điều khiển ga/phanh theo sai số km/h |
| `max_throttle` | 0.55 | Giới hạn ga tăng tốc; phanh vẫn có thể tới -1 |
| `safety_margin_m` | 0.3 m | Khoảng dự phòng từ thân xe tới biên theo ước lượng |
| `prediction_horizon_s` / `prediction_dt_s` | 4 s / 0.2 s | Cut-in đường thường; P2 trong passage dùng tối đa 3 s |
| `track_ttl_s` | 0.6 s | Giữ dự báo actor mất quan sát trong khoảng ngắn |
| `junction_prediction_s` / `junction_gap_s` | 12 s / 3 s | Kiểm tra thời gian chiếm section theo các road hypotheses |
| `junction_entry_buffer_m` | 4 m | Dự phòng ngoài nửa chiều dài ego trước yield boundary |
| `junction_blocked_extra_buffer_m` / `queue_release_s` | 8 m / 2 s | Chờ upstream và chống nhích vào queue tại X/T |
| `junction_scan_m` | 30 m | Tầm xét nhường trước lối vào junction |
| `junction_conflict_radius_m` | 12 m | Vùng lối vào dùng cho luật xung đột bảo thủ |
| `yield_release_s` | 0.5 s | Thời gian chờ sau khi xung đột không còn |
| `overflow_speed_kmh` | 27 km/h | Trần khi traffic context overflow |

Thử cấu hình mặc định trước. Sau đó chỉnh một tham số mỗi lần, so sánh cùng
map/seed và lưu cấu hình đã chạy. Không chỉ nhìn xe chạy đẹp ở một lần thử.

| Hiện tượng | Thử nghiệm đầu tiên |
|---|---|
| Cua quá nhanh rồi chạm biên | Giảm `max_lateral_accel_mps2` từ 1.5 xuống 1.2 |
| Lắc trái/phải trên đường thẳng | Giảm `k_angle` hoặc `k_lateral`, từng tham số riêng |
| Cắt vào trong cua từ quá sớm | Giảm `k_preview` từ 0.5 xuống 0.35 |
| Chuẩn bị cua quá muộn | Thử tăng `lookahead_m` từ 5 lên 6 |
| Sai lệch tâm còn lớn | Thử tăng `k_lateral` từ 1.5 lên 1.7 ở tốc độ thấp |
| Thường xuyên bật `recover_lane` | Xem độ lệch, hướng và tốc độ trong diagnostics trước |

Các điều chỉnh là giả thuyết để thử, không phải bảo đảm cải thiện trong mọi map.

Trần 45 km/h khác với hệ số giải mã observation 80 km/h. Không thay hệ số
giải mã để tăng tốc. Các trần này là cấu hình mô phỏng, không phải giới hạn
pháp lý của đường. Code hỗ trợ `Stage1RuleBasedAgent(cruise_speed_kmh=35.0)`;
CLI launcher chưa có cờ đổi cruise. Cấu hình thay đổi cần được ghi nhận bằng
runner experiment có source/config hash và trace đầy đủ.

## Quan sát và kiểm tra

Agent xuất diagnostics gồm tốc độ, độ lệch, sai hướng, curvature, gap/TTC,
giới hạn tốc độ theo nguyên nhân, lệnh lái/ga và behavior. P4 có diagnostics
`lane_change` gồm phase/reason/source/target/progress/commit/gap/TTC/veto.
Các behavior gồm
`CRUISE`, `FOLLOW`, `APPROACH_JUNCTION`, `YIELD`, `STOP_FOR_OBSTACLE`,
`COMMIT_JUNCTION`, `TRAVERSE_JUNCTION`, `EXIT_JUNCTION`, `BLOCKED`,
`REAR_SIDE_ESCAPE`, `RECOVER_LANE`, `DEGRADED_STOP`, `EMERGENCY_BRAKE`.
P4 thêm `PREPARE_CHANGE`, `CHANGE_LANE`, `ABORT_CHANGE`, `BLOCKED_CHANGE`.
Agent lưu tracks, timer và branch cache; `reset()` xóa toàn bộ trạng thái episode.

CLI tự ghi `agent_provenance.json`, source/config hash và toàn bộ public
input/action/diagnostics trong `agent_decisions.jsonl.gz` tại thư mục run.
Experiment paired dùng runner mới, giữ telemetry đánh giá ngoài policy:

```powershell
python -X utf8 scripts/run_rule_based_experiment.py --baseline-version 0.5.0 --env-seeds 0 --run-id rb_new_paired_attempt
python -X utf8 scripts/probe_rule_based_lane_change.py --run-id rb_new_p4_probe
python -X utf8 -m unittest tests.test_rule_based_agent tests.test_rule_based_corridor tests.test_rule_based_p2_p3 tests.test_rule_based_p4 tests.test_rule_based_replay tests.test_agent_experiment_logging tests.test_agent_contract -q
```

Mỗi lần thử dùng run ID mới; giữ cả run thất bại và regression. Xem
[workflow experiment](../experiments/rule_based/README.md) để tổng hợp, replay
snapshot và chạy full manifest. Tool partial-trace cũ đã được đưa vào
`experiments/legacy_source/pre_cleanup_20261031/` để giữ lịch sử nguồn;
lệnh chạy hiện tại dùng CLI và các runner/probe trong bảng `scripts/README.md`.

Kết quả lịch sử trên môi trường local ngày 2026-10-10:

| Trường hợp | Kết quả |
|---|---|
| Bản 0.1.0, Easy/SCCS/geometry 0/environment 0 | Chạm vàng ở bước 210, tốc độ tối đa khoảng 41.35 km/h |
| Bản 0.2.0, cùng trường hợp | Đến đích bước 971; không chạm vạch liền; lệch tâm lớn nhất khoảng 0.143 m |
| Bản 0.2.0, Easy/SCCS/geometry 7/environment 0 | Đến đích bước 838; không chạm vạch liền; lệch tâm lớn nhất khoảng 0.307 m |
| Bản 0.2.0, Medium/SCTCS/geometry 2/environment 0 | Đến đích bước 933; không chạm vạch liền; không crash; lệch tâm lớn nhất khoảng 0.436 m |
| Bản 0.2.0 qua CLI, Easy/SCCS/geometry 6/environment 0 | COMPLETE, clean success 1/1, route completion khoảng 0.993 |
| Unit tests agent và contract | 47 tests đạt |

Đây là kiểm tra phát triển SANDBOX/TRAIN, không phải kết quả benchmark và không
chứng minh tất cả bản đồ đều thành công. Các dòng kết quả 0.2.0 là lịch sử;
bản 0.3.0 đã bổ sung bám xe phía trước cơ bản và tốc độ thích ứng. So sánh
7 case TRAIN mới nằm trong tài liệu kế hoạch và `speed_policy_comparison.csv`:
6 case Easy/Medium cùng đến đích nhanh hơn, case Hard vẫn crash ở cả hai bản.

Bản 0.4.0 ngày 2026-10-30: trên cùng 9 case TRAIN/env0, success 5/9 → 7/9,
crash 4/9 → 2/9. Hai case Hard còn crash có completion giảm; chưa nghiệm thu
Extreme với cấu hình cuối. Chi tiết từng case và hash nằm trong báo cáo 0.4.0.

Bản 0.5.0 ghép cặp trên 16 case TRAIN: success 10/16 → 13/16, crash 4 → 0;
ba Extreme vẫn timeout. Đây là phạm vi khác tập 9 case ở 0.4; không so trực
tiếp hai tỷ lệ khác coverage. Báo cáo 0.5 dùng baseline 0.4 chạy cùng 16 case.

## Giới hạn cần hiểu

- Đọc metadata vạch vàng/trắng/liền/đứt từ adapter trạng thái của PG map;
  chưa nhận diện vạch bằng camera. Chỉ đổi qua vạch đứt trắng được xác nhận.
- Giải mã được kiểm tra với MetaDrive 0.4.3: side/lane-line lasers tắt,
  không random model, xe default dài 4.515 m/rộng 1.852 m, tốc độ chuẩn hóa
  theo 80 km/h, MAX_LANE_WIDTH=4.5 m, road scale=18 m. Nếu đổi cấu hình,
  phải kiểm chứng lại; không chỉnh các hằng số như thể chúng là gain điều khiển.
- `asin` giải mã hướng khi xe gần cùng hướng làn (<90 độ); chưa xử lý quay đầu.
- Envelope thân xe lấy theo tiếp tuyến làn, chưa dự đoán footprint trên biên cong.
- Tracking dùng ID nội bộ từ association, không phải danh tính thật của actor.
  Road hypotheses là các hướng có thể đi, không biết ý định tương lai thật.
  Luật nhường bảo thủ vẫn có thể chờ lâu/kẹt; không timeout ép tăng ga.
- Chuyển làn kề nhau trên đoạn gần thẳng đủ dài; không đổi nhiều làn một lần,
  không đổi trong passage/vòng xuyến hoặc sát ngã rẽ, không ép vượt khi thiếu gap.
  Đèn/biển, quyền ưu tiên pháp lý và jerk/steering-rate đầy đủ vẫn cần dữ liệu
  và kiểm chứng riêng. Dự báo có giới hạn 4 s, cảm biến 50 m.
- Vector 259D và điều kiện kết thúc khi chạm vạch được giữ nguyên. Semantics
  waypoint đã sửa từ schema 1.1.1; extension 1.3.0 vẫn pending audit, benchmark tắt.
  Thiết kế P4 hiện thực nằm trong `RULE_BASED_P4_DESIGN.md`; tài liệu
  `LANE_MARKING_AGENT_DESIGN.md` là đề xuất lịch sử, không phải contract hiện tại.
