# Rule-based 0.5.0 — P2/P3 work log

Ngày bắt đầu: 2026-10-30. Trạng thái: **đã đóng băng development release 0.5.0**,
chỉ SANDBOX/AUDIT/TRAIN. [Báo cáo cuối](RULE_BASED_V0_5_0_EXPERIMENT_REPORT.md).
Không sửa archive 0.4.0. Mỗi lần thử có source/config hash và run ID riêng.

## Giả thuyết từ trace baseline

- Hard/XTOCS/0/env0: nhường sau khi đã vào connector rẽ; ego đứng trong đường
  xe cắt ngang. Giảm tốc tại điểm va chạm chưa đủ để tránh bị xe khác đâm vào.
- Hard/SCXOCS/0/env0: mất decision_point sau entry; xe tiến vào khi actor còn
  đang rẽ vào hành lang. Cần nhớ toàn bộ lượt qua nút, không chỉ một frame.
- Context actor không có ID; thứ tự slot thay đổi theo khoảng cách. Cần
  association hình học, ego-motion compensation và TTL, không dùng slot làm ID.

## Phạm vi triển khai

1. Public snapshot có ego-motion cục bộ và đoạn qua nút trên tuyến; giữ 259D,
   không đưa tọa độ toàn cục/evaluator progress vào policy. Contract pending audit.
2. Actor tracking bằng vị trí/vận tốc/heading, TTL và dự báo vận tốc/quay đều;
   vẫn xét actor đứng yên, context overflow và LiDAR chưa phân loại.
3. FSM approach/yield/commit/traverse/exit/blocked, khóa nhánh, chờ clearance,
   kiểm tra exit bị chặn, không timeout ép tăng ga khi còn nguy cơ.
4. Kiểm thử scenario có điều kiện và paired TRAIN; ghi mọi regression.

## Chẩn đoán trước khi chỉnh

- Probe topology đầu tiên thất bại do đọc `lane.type`; thuộc tính đúng của
  MetaDrive là `metadrive_type`. Đây là lỗi script chẩn đoán, không phải driving.
- Probe sửa lại đã đọc được connector X/T là LANE_SURFACE_UNSTRUCTURE;
  vòng xuyến dùng CircularLane thường, có node merge và fork. Không dùng tên
  tier/sequence/seed để ra quyết định; adapter sẽ xuất hình học/topology cục bộ.

Kết quả test/experiment và trạng thái nghiệm thu được bổ sung sau khi chạy,
không suy thành công từ kế hoạch trên.

## Trial01 — `rule_based_v050_trial01_20261030`

- Paired 4 case TRAIN từ `train_targeted_v040.json`, env0, baseline archive 0.4.0.
- Baseline: 2 success / 2 crash. Draft đầu: 0 success / 0 crash / 4 timeout.
  Không có lỗi kỹ thuật/out-of-road/chạm vạch. Đây là regression về completion.
- Trace Medium/SCXCCS cho thấy entry dừng ở 7.2587 m, exit 44.7587 m; yêu cầu
  clearance 6 m vượt lookahead 50 m. Generic braking buffer 3 m bị cộng lần
  hai ngoài entry buffer, làm xe không tiến tới vị trí có đủ tầm nhìn.
- Trial02 hủy phần buffer trùng khi tính stop envelope cho junction; vẫn giữ
  entry buffer 2 m + nửa chiều dài ego và reaction distance. Không tăng ga do
  chờ lâu, không bỏ kiểm tra exit/crossing. Có regression test tái hiện.
- Kiểm tra sau sửa: 93 tests agent/corridor/P2P3/contract đạt (21 test mới).
- Source/config/trace draft01 giữ nguyên, không ghi đè bằng code trial02.

## Trial02 / Trial03 — xác minh và sửa regression

- `rule_based_v050_trial02_20261030`: 4 case, 0 success / 1 crash / 3 timeout.
  Buffer trùng đã sửa nhưng kiểm tra nhánh so cả waypoint mới nằm ngoài cache
  50 m, gây false mismatch và dừng gần exit. Trial03 chỉ so phần route giao nhau.
- `rule_based_v050_trial03_20261030`: 4 case, 1 success / 2 crash / 1 timeout.
  Medium/SCXCCS đến đích; Hard/SCTXrCS kẹt ở straight ramp do adapter gán nhầm
  cả road dài là một passage cần nhìn thấy hết. Giới hạn generic fork cho
  curved circulation, không coi nhánh thẳng ramp là giao lộ P3; tương tác actor
  vẫn do P2 xử lý, maneuver nhập/chuyển làn thuộc P4.
- Hai va chạm còn lại xảy ra trước inlet vòng xuyến. Entry là boundary của
  centerline, không phải stop line vật lý; actor có footprint nhô qua boundary.
  Trial04 tăng entry buffer 2 → 4 m ngoài nửa chiều dài ego, không đổi cruise.
- 97 tests agent/P2P3/contract đạt sau các sửa trên (25 test mới).

## Kiểm tra tích hợp và cảm biến

- `rule_based_v050_obstacle_probe01_20261030`: hai scenario có can thiệp,
  traffic density=0, barrier 20 m phía trước / lệch ngang 6 m; không tính vào
  điểm driving. Cả hai đạt: ego dừng ổn định trước barrier, chạy lại sau khi
  bỏ barrier; vật bên cạnh không tạo obstacle limit. Không crash/out-of-road.
- Probe on-path: dừng ổn định bước 74; tốc độ sau gỡ vật đạt 44.81 km/h;
  min center distance 6.259 m (không gọi đây là mesh clearance).
- Lần chạy integrated tests bằng Python mặc định thiếu Panda3D/MetaDrive;
  3 import errors và 1 test còn hard-code version 0.4. Test provenance đã sửa
  để kiểm đúng descriptor thực tế; chạy lại trong conda `metadrive` và lưu
  log riêng, không ghi đè log lỗi môi trường đầu tiên.

## Trial04 / Trial05 / Trial06 / Trial07

| Run ID | Coverage | Kết quả thực tế |
|---|---|---|
| `rule_based_v050_trial04_family12_20261030` | Paired 12 family/env0 | Baseline 7 success / 3 crash / 2 timeout; draft 7 / 4 / 1. Regression, không phát hành |
| `rule_based_v050_trial05_geometry_20261030` | 4 target/env0 | 3 success / 1 crash. SCXOCS mới đến đích; XTOCS còn crash ở vòng xuyến |
| `rule_based_v050_trial06_road_paths_20261030` | 4 target/env0 | 4 success / 0 crash, không lỗi kỹ thuật/biên |
| `rule_based_v050_trial07_family12_20261030` | 12 family/env0 | 9 success / 1 crash / 2 timeout, không lỗi kỹ thuật/biên |
| `rule_based_v050_env1_targeted_20261030` | Paired 4 target/env1 | Baseline 3 success / 1 crash; draft 4 success / 0 crash |

- Trial05 bổ sung signed geometric yield boundary: node entry không phải
  stop line. Vòng xuyến chia admission merge và branch exit, không giữ trống
  toàn vòng. Branch cache được nối tiếp phần geometry mới khi exit dần lộ ra.
- Trial06 bổ sung ActorPathV1: map-match pose/heading công khai vào các làn và
  liệt kê đường tiếp nối có thể có, không đọc lane/navigation/policy của actor.
  Horizon gap 8 → 12 s; vẫn giữ range public 50 m, actor features 7D và core 259D.
- Trial07 không bỏ root lane thứ ba khi nhiều connector cùng entry tangent;
  deduplicate geometry, giữ mọi root hợp lệ trong cap 8 và báo overflow khi thiếu.
- Trace gzip dùng compression level 3, vẫn giữ đủ từng dòng public input,
  action và diagnostics. Không đổi semantics hoặc sửa raw trace cũ.
- Integrated tests: trial04 conda 232 đạt; trial07 240 đạt. Frozen policy/input
  replay toàn bộ 1065 bước Hard/XTOCS của trial07 khớp action và diagnostics.
  Báo cáo máy đọc dưới `results/audits/rule_based/v0.5.0_checks/`.
- Probe02 lỗi setup vì MetaDrive không có `__version__`; lưu FAILED bổ sung
  ngay sau lỗi, đánh dấu retrospective error record, giữ runner gốc. Probe03
  chuyển sang package metadata, state ghi trước setup; 2/2 scenario đạt với
  bằng chứng raw LiDAR hit cả trong/ngoài làn. Không gộp probe vào driving score.

## Rear-side cut-in và phân xử P2/P3

- Trial07 Extreme/CrXROSTR crash ở bước 721: truck đổi làn vào rear quarter.
  Projection cũ clamp arc phía sau về 0, bỏ mất overlap thân xe. Thêm signed
  approach projection và đánh giá tăng tốc có giới hạn, chỉ khi mọi actor dự
  báo còn clear và các hard front stopping/headway, curve/visibility cap cho phép.
- `rule_based_v050_trial08_rear_side_20261030` dừng sạch INTERRUPTED, 0 episode
  hoàn tất: unit tests phát hiện thêm diagnostics làm vượt 50 keys ở một số
  phase. Chuyển các thông tin mới vào nested `rear_side`; không nới contract.
- `rule_based_v050_trial09_rear_side_20261030`: 1 crash, completion 0.3124.
  Phát hiện cảnh báo quá muộn; không dùng một frame acceleration để tuyên bố
  đã giải quyết cut-in. Có unit fixture từ trạng thái quan sát bước 712.
- Trial10 (`rule_based_v050_trial10_timed_cut_in_20261030`) giữ turn model
  actor song song gần trong 4 s, lateral uncertainty 0.15 m/s; thêm speed cap
  theo thời gian chiếm vùng, vì braking-distance cap đơn thuần vẫn có thể
  cho tăng ga về một điểm xung đột tương lai. Qua được rear-side event cũ.
- Trial10 sau đó crash bước 1335, completion 0.3657: ego đứng yên khoảng
  27.6 s trước giao lộ, bị actor trong pile-up dịch lùi vào. Thêm 8 m khoảng
  chờ upstream khi exit intersection bị chặn và 2 s queue-clear hysteresis;
  không force launch sau timeout, không bật lùi/chuyển làn.
- Run `rule_based_v050_regression12_20261030` được dừng sạch INTERRUPTED với
  9/12 episode hoàn tất để chuyển sang queue guard; không xuất full-suite score.
  Run seed1 cùng source `rule_based_v050_regression_env1_20261030` đã hoàn tất
  4/4 success, giữ nguyên dù source bị thay thế.
- Lần nghiệm thu nháp `rule_based_v050_final_family12_20261030` bị supersede
  sau khi phát hiện regression; run seed1 `rule_based_v050_final_env1_20261030`
  hoàn tất nhưng có Hard/SCXOCS crash. Tên run có chữ final không là chứng nhận.
- Nguyên nhân ghép luật: free-turn forecast 4 s + timed yield của đường thường
  làm xe dừng khi đang qua nút vì một nhánh xa chỉ là giả thuyết. Cấu hình
  acceptance dùng P2 cut-in dài ở đường thường; tại passage dùng road-path
  admission P3, free motion gần 2 s, P2 horizon 3 s và AEB vật lý. Extra queue
  buffer/hysteresis chỉ áp dụng intersection box, không áp vào inlet vòng xuyến.
- Thêm test cho timed cut-in không được tăng ga, phân xử theo context và
  immediate collision vẫn phải AEB trong junction. Run acceptance mới có ID
  `rule_based_v050_acceptance_family12_20261030` và
  `rule_based_v050_acceptance_env1_20261030`; kết luận chỉ sau khi hoàn tất.

## Tiếp tục công việc và xác minh hồ sơ

- Hai lệnh replay/tổng hợp cuối lượt trước không được thực thi vì automatic
  approval review báo hết hạn mức. Đây là lỗi dịch vụ duyệt, không phải kết
  luận lệnh không an toàn; không có kết quả experiment nào được giả lập từ đó.
  Sau yêu cầu tiếp tục của người dùng, chạy lại cùng cơ chế duyệt thành công.
- Bộ tích hợp cuối đạt **252 tests**, log và hash 8 file policy/platform/logger
  lưu ở `results/audits/rule_based/v0.5.0_checks/tests_acceptance_metadrive_20261030.*`.
  Khi tiếp tục, kiểm lại cả 8 hash đều khớp source hiện tại; không chạy lại bộ
  test chỉ để tăng số lần pass khi code chưa đổi.
- Replay source/helper/input types đóng băng: Hard/SCXOCS/0/env1 khớp toàn bộ
  **1482/1482** public input round-trip, decision/diagnostics và canonical action.
  Latency không so bằng; shared platform API/adapter hash ghi trong report.
- `rule_based_v050_cli_smoke_20261030`: CLI chính thức Hard/XTOCS/0/env0,
  COMPLETE, clean success 1/1, 1065 bước. Kiểm tra đủ 1065 dòng trace, 259D,
  ego-motion đủ mọi dòng; 747 dòng có road hypotheses. W&B DISABLED, render OFF.
- `rule_based_v050_obstacle_probe04_acceptance_20261030`: source nghiệm thu
  cuối, 2/2 controlled scenario đạt. Dừng ổn định bước 74, chạy lại sau gỡ vật
  đạt 44.81 km/h; vật bên cạnh vẫn có raw LiDAR hit nhưng không tạo corridor stop.
  Không crash/out-of-road. Probe không được cộng vào số map thành công.
- Seed1 acceptance hoàn tất 4/4 success, 0 crash. Family12 acceptance tiếp tục
  tới case cuối theo horizon đã khai báo; không rút ngắn timeout hoặc sửa trạng
  thái để xuất báo cáo. Các case còn kẹt phải giữ nguyên TIMEOUT, không gọi success.
- Runner mới mặc định baseline 0.4.0; `runner.py` và request của các run cũ giữ
  nguyên default tại thời điểm chạy. Khi ghép report, baseline được chỉ định
  rõ bằng run chứa snapshot 0.4.0, không suy baseline từ default không dùng
  trong các run chỉ có variant current.

## Chốt kết quả và snapshot

- Acceptance family12 hoàn tất **12/12 episode**, 9 success / 0 crash / 3 timeout;
  baseline cùng 12 case 7 success / 3 crash / 2 timeout. Seed1 acceptance 4/4
  success, baseline 3/4 success / 1 crash. Tổng đúng 16 case: success 10 → 13,
  crash 4 → 0; không mất case thành công. Không có lỗi kỹ thuật/biên ở candidate.
- Ba timeout Extreme giữ đủ horizon 3188/3429/2914 bước. Phân tích 100 bước
  cuối mỗi case: `front_vehicle` thắng cả 100 bước, ego dưới 0.016 km/h, actor
  ghép làm lead có median speed 0.003–0.005 m/s. Ghi là quan sát từ snapshot,
  không suy vượt/chuyển làn luôn khả thi, không đổi TIMEOUT thành success.
- Đóng băng `versions/0.5.0/` sau khi report kiểm nguồn/config/trace hoàn tất.
  Lưu 11 draft exact byte từ raw source dưới `versions/dev050_*`; đánh dấu đây
  là archive bổ sung sau experiment, không backdate release hay bịa API cũ.
- `rule_based_v050_snapshot_smoke_20261030` load archive 0.5 như baseline độc
  lập: Medium/SCXCCS/env0 success 895 bước. Public input, decision/diagnostics
  và canonical action khớp toàn bộ 895 bước live-policy; latency không so bằng.
- `experiment_index.json` tổng hợp 23 run, gồm failed/interrupted/probe/smoke;
  `development_snapshot_index.json` chỉ rõ draft ↔ nguồn. Đây không phải 23
  case độc lập. Các run partial không bị dùng làm full-suite score.
- Báo cáo kết quả, regression, giới hạn và lệnh tái chạy ở
  `RULE_BASED_V0_5_0_EXPERIMENT_REPORT.md`. P4/P5 và mục tiêu >=95% chưa hoàn tất.
- Kiểm tra đóng sổ: cả 12 archive (official + 11 draft) khớp SHA và import ra
  đúng cấu hình gốc; baseline 0.4 không đổi. Replay Hard/XTOCS/env0 source cuối
  khớp thêm 1065/1065 bước. Mọi run 0.5 có trạng thái cuối, không còn RUNNING;
  `release_verification.json` ghi source hash, test/replay/probe/CLI và các phần
  còn lại. Link cục bộ trong 8 tài liệu đã được kiểm tra.
