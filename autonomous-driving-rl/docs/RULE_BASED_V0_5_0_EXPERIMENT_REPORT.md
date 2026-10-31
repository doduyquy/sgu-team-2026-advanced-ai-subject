# Rule-based 0.5.0 — P2/P3 và kết quả experiment

Ngày: 2026-10-30. Agent: `stage1_rule_based`, development release SANDBOX/AUDIT.
Source/config đã đóng băng. P2/P3 đã có các thành phần triển khai và kiểm thử
trong phạm vi dưới đây; chưa hoàn tất toàn bộ kế hoạch agent hoặc benchmark.

## 1. Kết quả chính

So sánh cùng geometry/environment seed, toàn bộ là TRAIN:

| Tập ghép cặp | 0.4.0 success | 0.5.0 success | Crash trước → sau | Timeout trước → sau |
|---|---:|---:|---:|---:|
| 12 family, environment seed 0 | 7/12 | **9/12** | **3 → 0** | 2 → 3 |
| 4 case giao lộ/vòng xuyến, environment seed 1 | 3/4 | **4/4** | **1 → 0** | 0 → 0 |
| Tổng 16 case riêng biệt | 10/16 | **13/16** | **4 → 0** | 2 → 3 |

Không mất case thành công nào. Không có out-of-road/chạm vạch trong 16 case
của candidate. Không gọi ba timeout là thành công hoặc tuyên bố đạt >=95%.

Ba case mới đến đích: Hard/SCXOCS/0/env0, Hard/XTOCS/0/env0 và
Hard/SCXOCS/0/env1. Chạy lại nhiều draft trên cùng case không làm tăng coverage.

Nguồn máy đọc được:

- [So sánh 12 family](../results/audits/rule_based/v0.5.0/comparison.json),
  [CSV từng case](../results/audits/rule_based/v0.5.0/paired_cases.csv).
- [So sánh seed 1](../results/audits/rule_based/v0.5.0_env1/comparison.json).
- [Tổng hợp 16 case](../results/audits/rule_based/v0.5.0/combined_summary.json).

## 2. Đã thay đổi gì

Chi tiết thuật toán: [thiết kế P2/P3](RULE_BASED_P2_P3_DESIGN.md).
Nhật ký từng lần thử và nguyên nhân sửa: [work log](RULE_BASED_V0_5_0_WORKLOG.md).

### P2 — actor, vật cản và xung đột

- Bù chuyển động ego trong frame cục bộ, ghép actor bằng vị trí/vận tốc/heading
  và kích thước, một-một, tie-break xác định. Slot không phải ID qua nhiều frame.
- Giữ track mất quan sát tối đa 0.6 s với uncertainty tăng; reset xóa toàn bộ.
- Dự báo vận tốc không đổi, quay đều có bằng chứng từ lịch sử, và các đường
  kết nối có thể đi. Không đọc tuyến riêng, policy hoặc hành động tương lai actor.
- Bám xe trên corridor cong, xét footprint, bumper gap, headway/TTC và AEB.
- Ngoài passage, giữ dự báo cut-in tới 4 s và ràng buộc tốc độ theo thời gian
  chiếm vùng. Braking-distance cap đơn thuần từng cho phép tăng ga về nguy cơ.
- Xét rear-side body overlap trên projection có dấu. Chỉ cho tăng tốc thoát ở
  đường thường nếu mọi actor dự báo còn clear, front stopping/headway và các
  giới hạn đường vẫn cho phép; không bỏ qua lệnh dừng hoặc AEB.
- Giữ xử lý hit LiDAR chưa phân loại trong corridor, loại hit thuộc actor đã
  biết, hạ tốc khi overflow và controlled stop khi dữ liệu quan sát không đủ.

### P3 — giao lộ và vòng xuyến

- FSM approach/yield/commit/traverse/exit, blocked/degraded cùng diagnostics
  nguyên nhân. Khóa branch bằng route polyline tới khi đuôi xe qua exit.
- Tìm local passage từ surface type/connectivity. Tại vòng xuyến, yield boundary
  hình học có thể nằm trước entry node; node không bị giả làm stop line pháp lý.
- Kiểm tra space-time gap bằng footprint và thời gian ego tăng tốc từ đứng yên;
  P3 dùng road hypotheses 12 s, sample 0.2 s, gap 3 s, xác nhận clear 0.5 s.
- Xét chỗ trống sau vùng xung đột. Khi exit X/T bị chặn, chờ thêm 8 m upstream,
  xác nhận hết queue 2 s. Vòng xuyến xét inlet merge/section, không giữ trống cả vòng.
- Kiểm tra lại permission trước khi đầu xe vào connector. Branch vẫn giữ khi
  phải nhường lại; không đổi nhánh mỗi frame hoặc ép tăng ga sau timeout chờ.
- Tại passage, giữ free-motion gần 2 s và P2 horizon tối đa 3 s bên cạnh road
  hypotheses. Không áp timed cut-in xa của đường thường vào phase qua nút:
  thử nghiệm đã cho thấy ghép sai có thể làm xe dừng ngay trong vòng xuyến.
- Route cache chỉ so trên phần geometry giao nhau với input mới, nối thêm
  phần liên tục đã quan sát; mất/lệch input quá TTL 1 s dẫn tới degraded stop.

Giữ trần cruise **45 km/h**, junction **24 km/h**, recovery **8 km/h**, ga 0.55.
Không tăng giới hạn bằng cách sửa hệ số giải mã observation 80 km/h.

## 3. Dữ liệu và phạm vi so sánh

Schema public tăng 1.1.1 → **1.2.0**, status pending audit:

- `EgoMotionV1`: displacement/heading delta cục bộ, velocity và full lane heading error.
- `RoutePassageV1`: branch, entry/exit, geometric yield và local conflict section.
- `ActorPathV1`: tối đa 8 giả thuyết đường cho mỗi actor, có overflow. Map-match
  từ pose/heading quan sát; không dùng lane/navigation riêng của actor.

Core vẫn 259D, actor array 8×7, radius/lookahead 50 m. CORE_ONLY che các
extension. Policy không nhận global pose, tier, seed, reward, terminal reason
hay route completion. Các trường evaluator chỉ ở harness/log bên ngoài policy.

**So sánh này thay đổi cả policy lẫn input extraction/extensions.** Không phải
policy-only ablation; không quy toàn bộ cải thiện cho riêng một luật. Mọi phương
pháp so sánh sau này cần cùng information profile đã được audit.

MetaDrive 0.4.3, source commit
`85e5dadc6c7436d324348f6e3d8f8e680c06b4db`, control dt 0.1 s.
Case được resolver/geometry-hash kiểm tra là TRAIN. Không thay traffic density,
termination, physics timestep hoặc rút horizon của các driving experiment.

## 4. Thời gian chạy xe và giới hạn còn thấy

| Case cùng thành công | 0.4.0 | 0.5.0 | Nhận xét |
|---|---:|---:|---|
| Easy/SCCS/0/env0 | 59.7 s | 59.7 s | Giữ lại |
| Medium/SCTCS/2/env0 | 67.4 s | 70.0 s | Chậm hơn 2.6 s |
| Medium/SCXCCS/0/env0 | 86.5 s | 89.5 s | Chậm hơn 3.0 s |
| Medium/SCXCS/14/env0 | 54.1 s | 56.6 s | Chậm hơn 2.5 s |
| Hard/SCTXrCS/2/env0 | 160.7 s | 143.2 s | Nhanh hơn 17.5 s |
| Hard/SCTXrCS/2/env1 | 157.2 s | 164.7 s | Chậm hơn 7.5 s |
| Hard/XTOCS/0/env1 | 81.9 s | 96.6 s | Chậm hơn 14.7 s |

Không tuyên bố tăng tốc toàn diện. Case baseline crash không được dùng thời
gian crash sớm để so xe chạy nhanh hơn. CSV vẫn ghi cả completion thấp hơn;
các case cùng success có completion cuối hơi khác do vị trí chạm điều kiện đích.

### Ba case chưa đến đích

| Extreme/env0 | 0.4.0 | 0.5.0 | Completion trước → sau |
|---|---|---|---|
| CrXROSTR/0 | Timeout | Timeout, 3188 bước | 0.4033 → 0.4031 |
| SCTXORyCCS/0 | Timeout | Timeout, 3429 bước | 0.4745 → 0.4727 |
| SCXOCrTYCS/0 | Crash | Timeout, 2914 bước | 0.3366 → 0.4515 |

[Phân tích 100 bước cuối](../results/audits/rule_based/v0.5.0/timeout_tail_analysis.json)
cho thấy winning constraint đều là `front_vehicle` ở cả 100 bước mỗi case;
ego dưới 0.016 km/h và actor được ghép làm xe trước có tốc độ median khoảng
0.003–0.005 m/s. Đây là bằng chứng quan sát về nguyên nhân dừng, không chứng
minh mọi cách vượt đều bất khả thi hoặc P4 chắc chắn sẽ giải quyết được.

Không thêm luật timeout ép chạy qua actor, không tự lùi/chuyển làn. P4 chưa có.
Các case này vẫn là thất bại về hoàn thành tuyến và được giữ nguyên trong score.

Act latency p95 trong các case cuối khoảng **0.63–28.91 ms**. Một số run chạy
đồng thời; không suy tốc độ tính toán tương đối từ tải máy khác nhau. Mục tiêu
p95 <10 ms chưa đạt ở case đông xe. Act timing loại trừ input assembly,
physics và ghi trace theo contract, nên không đại diện tổng FPS.

## 5. Kiểm chứng và log

- **252 tests** agent/input/contract/logger/launcher/visual/replay đạt. Test log
  và source hash nằm ở `results/audits/rule_based/v0.5.0_checks/`.
- Frozen policy/helpers/input replay Hard/SCXOCS/env1 khớp **1482/1482** bước,
  gồm action/diagnostics và canonical action; không so latency.
- Chạy simulator độc lập từ archive 0.5.0, Medium/SCXCCS/env0, success 895 bước;
  cả 895 public input/action/diagnostics khớp lượt live-policy tương ứng.
- CLI chính thức Hard/XTOCS/env0: COMPLETE, clean success 1/1, 1065 dòng trace;
  có full ego-motion và public road hypotheses, W&B DISABLED.
- Probe vật lý source cuối: barrier 20 m trong đường / lệch ngang 6 m,
  **2/2 đạt**. Ego dừng ổn định bước 74, chạy lại tới 44.81 km/h sau gỡ vật;
  raw LiDAR vẫn thấy vật bên cạnh nhưng không gây corridor stop. Traffic tắt
  có chủ ý, probe không tính vào driving score. Gap ghi là ước lượng/center
  distance, không phải mesh clearance.
- Script tổng hợp kiểm hash source/config, case, step liên tục và số dòng trace
  khớp outcome trước khi công bố. Run interrupted không được xuất full score.

## 6. Artifact và phiên bản

- [Snapshot 0.5.0](../experiments/rule_based/versions/0.5.0/snapshot.json): cấu
  hình đầy đủ, source SHA-256, dependency/API, contract pending audit và giới hạn.
- [Index experiment](../results/audits/rule_based/v0.5.0/experiment_index.json):
  23 run 0.5, gồm draft thất bại/hủy, CLI, probe và archive smoke; không xem
  23 run là 23 case độc lập hoặc gộp probe/replay vào success.
- [Index 11 draft](../results/audits/rule_based/v0.5.0/development_snapshot_index.json):
  `versions/dev050_*` giữ đúng byte source từ raw snapshot, kể cả bản lỗi.
  Đây là archive bổ sung sau experiment, không giả mạo timestamp release cũ.
- Raw `runs/<id>/` có source/config/hash, runtime, resolved case/seed, trace,
  outcome và trạng thái; bị gitignore theo quy tắc dự án. Các bản source và
  kết quả quan trọng được giữ riêng dưới `experiments/`, `results/`, `docs/`.
- Các run dùng tên `final` nhưng bị supersede vẫn giữ status thật. Không sửa
  raw dữ liệu để che regression; xem đầy đủ diễn biến trong work log.

## 7. Tái chạy và phần tiếp theo

Trong environment `metadrive`, tại thư mục `autonomous-driving-rl`:

```powershell
# Xem case Hard đã chuyển từ crash sang success; CLI tự ghi log
python -X utf8 -m src.launcher.cli run --mode SANDBOX --agent stage1_rule_based --tier Hard --sequence XTOCS --geom-seed 0 --env-seed 0 --render NATIVE --wandb DISABLED --finish-visual hologram

# Experiment mới: so snapshot 0.4 với bản hiện tại; run ID phải chưa tồn tại
python -X utf8 scripts/run_rule_based_experiment.py --baseline-version 0.4.0 --env-seeds 0 --run-id rb_new_paired_attempt

# Chạy độc lập source archive 0.5; không sửa baseline cũ
python -X utf8 scripts/run_rule_based_experiment.py --baseline-version 0.5.0 --variants baseline --manifest experiments/rule_based/train_targeted_v040.json --env-seeds 0 --limit-cases 1 --run-id rb_new_snapshot_smoke
```

P4 nhập/chuyển làn/vượt, dữ liệu vạch/làn đích, jerk/steering-rate, full 108
TRAIN và canonical VALIDATION/TEST vẫn còn. Cần audit schema/chuỗi contract
trước benchmark. Không tuyên bố hiểu quyền ưu tiên pháp lý khi input không có
biển/đèn, không coi association/hypotheses hữu hạn là biết ý định thật của actor.
