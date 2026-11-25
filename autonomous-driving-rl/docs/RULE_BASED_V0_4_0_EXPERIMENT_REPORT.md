# Rule-Based 0.4.0 — báo cáo triển khai và thí nghiệm

Ngày: 2026-10-30. Trạng thái: development release, SANDBOX/AUDIT.
Source đã đóng băng, chưa chứng nhận benchmark hay hoàn tất toàn bộ kế hoạch.

## 1. Nội dung thay đổi

Mục tiêu lần này là P0 (sửa dữ liệu làn) và phần nền P2 (tương tác xe/vật cản),
thêm một luật P3 để tránh dừng quá muộn trong vùng giao cắt.

1. **Tuyến đúng làn:** đầu vào dùng làn thực tế đang chạy và chọn connector
   theo tính liên tục endpoint/heading, thay cho lấy `lanes[0]` ở mọi đoạn.
   Nếu không có nối hợp lệ thì không giả lập waypoint nhảy qua chỗ đứt.
2. **Theo xe trên cua:** chiếu vị trí/footprint actor lên polyline route cong;
   chọn xe trước theo hành lang đã định thay vì một dải thẳng kéo dài từ ego.
3. **Crossing/cut-in:** dự báo vận tốc không đổi trong 3 s, mẫu 0.2 s, có padding
   giữa mẫu. Actor đang ngoài làn vẫn có thể tạo ràng buộc phanh.
4. **Nhường trước junction:** xét xe đi qua/đối hướng có thể chiếm vùng nút
   và dừng trước lối vào. Đây là heuristics bảo thủ, không phải luật quyền ưu tiên.
5. **LiDAR:** nhận hit chưa phân loại trong hành lang; loại hit của actor đã
   nhận diện để xe đang chạy không bị coi như chướng ngại đứng yên.
6. **Trạng thái:** thêm behavior diagnostics và timer 0.5 s trước khi nhả chế độ
   nhường. `reset()` xóa timer; descriptor và registry đều khai báo stateful.
7. **Dữ liệu không đủ:** hạ tốc khi actor overflow; controlled stop khi scan
   LiDAR không dùng được hoặc hướng tuyến mơ hồ.

Giữ trần cruise 45 km/h, junction 24 km/h, recovery 8 km/h và ga tối đa 0.55.
Không thay action adapter, termination, physics timestep, traffic density hoặc
quyền đọc evaluator-private telemetry của agent.

## 2. Source và cấu hình đóng băng

`experiments/rule_based/versions/0.4.0/snapshot.json` chứa đầy đủ cấu hình và hash.

| File trong snapshot | SHA-256 |
|---|---|
| `policy.py` | `03f51ec69127dedffec62bb04032f2d27bcdb15d10121dbca7a8b10b23e029fc` |
| `rule_based_traffic.py` | `958bcf07e6f9afd4ccab2d6adaba8b7f36fcb742861b3482032b69a5f2283760` |
| `agent_input.py` | `5f7fc106087ed1145c649cdbcb28498d403c74b15a747314c447707d6e96ea83` |

Snapshot 0.3.0 được lưu trước khi sửa 0.4.0, gồm cả policy lẫn extractor cũ.
MetaDrive source commit kiểm tra khi đóng snapshot:
`85e5dadc6c7436d324348f6e3d8f8e680c06b4db` (0.4.3).
Các runtime file cũ giữ nguyên trường đã ghi; trường commit đầy đủ được bổ sung
trong runner mới và xuất hiện ở snapshot-replay run, không backfill giả vào run cũ.

Schema input tăng 1.1.0 → 1.1.1 để công bố semantics waypoint đã sửa, không thêm
chiều observation hay nhãn mới. Chuỗi hash audit lịch sử chưa được recertify;
benchmark eligibility giữ false.

## 3. Thiết kế experiment

Chỉ dùng TRAIN đã được resolver/geometry-hash checks của platform chấp nhận.
Cùng geometry/environment seed, physics 0.02 s × repeat 5 = control dt 0.1 s.
Không dùng agent seed để tạo nhiều mẫu giả cho policy deterministic.

| Run | Coverage | Vai trò |
|---|---:|---|
| `rule_based_v030_baseline12_20261030` | 12 family × env0 | Baseline đã đóng băng, 12 episode hoàn tất |
| `rule_based_v040_trial01_20261030` | Cùng 12 case | Draft đầu, source/config riêng, 12 episode hoàn tất |
| `rule_based_v040_trial02_20261030` | 4 case TRAIN mục tiêu | Điều tra/kiểm tra luật nhường trước junction |
| `rule_based_v040_regression9_20261030` | 9 Easy/Medium/Hard × env0 | Bản cuối, ghép cặp đúng 9 case tương ứng của baseline |
| `rule_based_v040_cli_smoke_20261030` | Easy/SCS/0/env0 | Đường chạy CLI chính thức và logging |
| `rule_based_v040_snapshot_replay_20261030` | Easy/SCCS/0/env0 | Replay policy/input từ archive 0.4.0 |
| `rule_based_v040_interrupt_probe_20261030` | Dừng chủ động trước episode | Kiểm tra status/log, không tính driving score |

Baseline 12 case có 5 success, 6 crash, 1 timeout. Draft01 có 6 success,
4 crash, 2 timeout. Không so trực tiếp các tỷ lệ trên với 7/9 của bản cuối vì
coverage khác nhau. Cấu hình cuối chưa được nghiệm thu trên 3 Extreme case.

So sánh chính là **policy và input extractor cùng thay đổi**. Đây không phải
policy-only ablation, full VALIDATION/TEST hoặc bằng chứng an toàn thống kê.
Tất cả run driving nói trên hoàn tất, không có lỗi kỹ thuật; probe dừng có
status INTERRUPTED và 0 episode hoàn tất đúng kỳ vọng.

## 4. Kết quả ghép cặp của bản cuối

Nguồn máy đọc được:
[paired_cases.csv](../results/audits/rule_based/v0.4.0/paired_cases.csv),
[comparison.json](../results/audits/rule_based/v0.4.0/comparison.json).

| Case (environment seed 0) | 0.3.0 | 0.4.0 cuối |
|---|---|---|
| Easy/SCCS/0 | Success, 59.7 s | Success, 59.7 s |
| Easy/SCS/0 | Success, 41.8 s | Success, 41.8 s |
| Easy/SCSS/0 | Success, 45.0 s | Success, 45.0 s |
| Medium/SCTCS/2 | Success, 67.4 s | Success, 67.4 s |
| Medium/SCXCCS/0 | Crash, completion 0.4641 | **Success, completion 0.9936** |
| Medium/SCXCS/14 | Success, 53.8 s | Success, 54.1 s |
| Hard/SCTXrCS/2 | Crash, completion 0.2677 | **Success, completion 0.9933**, 160.7 s |
| Hard/SCXOCS/0 | Crash, completion 0.7161 | Crash, completion **0.5689** |
| Hard/XTOCS/0 | Crash, completion 0.1459 | Crash, completion **0.1429** |

- Clean success: **5/9 → 7/9**; crash: **4/9 → 2/9**.
- Không mất case thành công nào trong 5 case bản cũ đã làm được.
- Hai case còn crash có completion thấp hơn: regression cần tiếp tục xử lý.
- Không có out-of-road/chạm vạch trong 9 case của mỗi bản.
- Không tuyên bố tăng tốc toàn diện: case Medium/SCXCS chậm hơn 0.3 s; case
  Hard mới thành công chạy thận trọng và mất 160.7 s.
- Act latency p95 của bản cuối trên tập này khoảng 1.05–9.16 ms. Một số run
  draft chạy đồng thời có latency lớn hơn; không dùng latency khác tải máy để
  khẳng định một phiên bản nhanh hơn về tính toán.
- `min_observed_gap_m` là gap ước lượng theo corridor và envelope, không phải
  khoảng cách vật lý chính xác giữa mesh; giá trị 0 không tự đồng nghĩa crash.

## 5. Phát hiện từ các lần thử thất bại

Trial01 vẫn crash Hard/SCTXrCS: các dòng trace cuối cho thấy ego đã dừng trong
vùng giao cắt và một actor đang rẽ đi vào vị trí đó. Dự báo thẳng vận tốc không
đổi chưa đủ để biết actor sẽ bẻ cua. Trial02 thêm luật nhường từ trước cửa
junction; cùng case chuyển sang success và được xác nhận lại ở regression run.

Hard/SCXOCS và Hard/XTOCS vẫn crash. Chưa có mô hình ý định actor, conflict zone
chính xác, khóa nhánh/commit state đầy đủ hay quyền ưu tiên. Không dùng hai
case này để tuyên bố P3 hoàn tất.

Draft01 ở Extreme/CrXROSTR và Extreme/SCTXORyCCS timeout; Extreme/SCXOCrTYCS
crash. Đây là giới hạn của draft, không phải kết quả của cấu hình cuối. Các
log này giữ nguyên trong `runs/` và báo cáo `v0.4.0_draft01/` để không che lỗi.

## 6. Kiểm tra tính tái tạo và độ đầy đủ log

- 205 tests input/agent/logger/launcher/visual đạt ở bước xác minh tích hợp.
- Thêm 2 tests bảo đảm report từ chối run chưa hoàn tất và snapshot bị sửa.
- Script tổng hợp kiểm tra source/config hash, đúng case, số dòng trace liên
  tục và khớp số bước của từng episode, rồi mới xuất báo cáo.
- Snapshot replay tái tạo **597/597 public input, action và diagnostics giống
  hệt** run gốc. Latency không tham gia phép so bằng vì phụ thuộc tải máy.
- CLI chính thức ghi `agent_provenance.json`, `sources/` và 418 dòng
  `agent_decisions.jsonl.gz` cho Easy/SCS/0/env0.
- Probe STOP_REQUESTED ghi INTERRUPTED, 0/1 episode hoàn tất; không bị tính
  thành COMPLETE hoặc success. Raw run ID luôn mới, file cũ không bị ghi đè.

Raw runs bị gitignore theo cấu trúc dự án. Các báo cáo CSV/JSON, source archive,
config và digest dưới `experiments/`/`results/` là hồ sơ được giữ cùng repository.

## 7. Tái chạy

```powershell
# Case Hard chuyển từ crash sang success ở phiên bản này
python -X utf8 -m src.launcher.cli run --mode SANDBOX --agent stage1_rule_based --tier Hard --sequence SCTXrCS --geom-seed 2 --env-seed 0 --render NATIVE --wandb DISABLED --finish-visual hologram

# Experiment phát triển paired; dùng run ID chưa tồn tại
python -X utf8 scripts/run_rule_based_experiment.py --env-seeds 0 --limit-cases 9 --run-id rb_new_paired_attempt

# Kiểm chứng snapshot 0.4 làm baseline độc lập
python -X utf8 scripts/run_rule_based_experiment.py --baseline-version 0.4.0 --variants baseline --env-seeds 0 --limit-cases 1 --run-id rb_new_snapshot_replay
```

Chưa hoàn tất: lane change/vượt, vạch/biên có nhãn, actor tracking, mọi trạng
thái junction/roundabout, jerk/steering-rate control và suite 108 TRAIN +
canonical VALIDATION/TEST. Đây là phạm vi các phiên bản tiếp theo.
