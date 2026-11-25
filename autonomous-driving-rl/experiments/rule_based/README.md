# Rule-based experiment records

P4/0.6: [thiết kế](../../docs/RULE_BASED_P4_DESIGN.md) và
[work log từng attempt](../../docs/RULE_BASED_V0_6_0_WORKLOG.md).
Để so riêng P4 với bản 0.5 đã đóng băng, chỉ định baseline rõ ràng:

```powershell
python -X utf8 scripts/run_rule_based_experiment.py --baseline-version 0.5.0 --env-seeds 0 --run-id rb_p4_new_paired_attempt
python -X utf8 scripts/probe_rule_based_lane_change.py --run-id rb_p4_new_physical_attempt
```

Probe có sáu can thiệp được ghi rõ, không tính vào driving score. Run ID phải
mới. Báo cáo có thể tái dùng một variant của run đã hoàn tất làm baseline với
`--baseline-variant current`; mọi source/config hash, trace, định nghĩa case và
cấu hình simulator phải khớp. Không đổi tên/copy raw folder để giả thành variant
khác. Thứ tự case trong shard được phép khác; geometry/seed/traffic/horizon không.

Replay physical probe dùng frozen policy/helpers/input của scenario:

```powershell
python -X utf8 scripts/replay_rule_based_trace.py --variant-dir runs/rb_p4_new_physical_attempt/safe_overtake --controlled-scenario --output results/audits/rule_based/new_physical_replay.json
```

`collect_v060_evidence.py` là định nghĩa thu thập evidence cho đúng các run 0.6
đã ghi trong source của nó; kiểm tra tất cả hoàn tất, không mất success, không
crash/road/solid-boundary, 279 tests và sáu replay trước khi cho đóng băng. Nó từ
chối ghi đè report/version cũ; không phải công cụ chạy benchmark.

Theo dõi phiên bản trong [changelog](../../docs/RULE_BASED_CHANGELOG.md).
Đây là workflow phát triển TRAIN không canonical; VALIDATION/TEST vẫn phải
chạy đúng launcher và chỉ khi đủ điều kiện benchmark.

## Chạy paired experiment

Trong environment `metadrive`, tại `autonomous-driving-rl`:

```powershell
python -X utf8 scripts/run_rule_based_experiment.py --run-id rb_paired_new_attempt
```

Mặc định so sánh snapshot 0.4.0 với policy/input hiện tại trên 12 geometry TRAIN
và environment seed 0, 1: tổng 48 episode. Đây là so sánh cả policy lẫn extractor;
không được quy toàn bộ chênh lệch cho riêng thuật toán traffic.

Chạy riêng các lỗi giao lộ đang điều tra:

```powershell
python -X utf8 scripts/run_rule_based_experiment.py --manifest experiments/rule_based/train_targeted_v040.json --variants current --run-id rb_targeted_new_attempt
```

`--env-seeds 0` hoặc `--limit-cases 3` tạo subset smoke được ghi rõ trong
resolved manifest. Không gọi kết quả subset là full suite. Cùng một run ID
đã tồn tại sẽ bị từ chối, kể cả run trước bị lỗi.

Muốn yêu cầu dừng sạch một run của script mới, tạo file `STOP_REQUESTED` trong
thư mục run đó. Script kiểm tra mỗi 10 bước, đóng gzip/env và ghi INTERRUPTED.

## File mỗi run

```text
runs/<run_id>/
  resolved_cases.json        # Case, geometry/environment seed và manifest hash
  runtime.json               # Python, packages, git và control dt
  runner.py                  # Chính script đã chạy
  run_state.json             # Trạng thái, số episode dự kiến/đã xong, lỗi
  episodes.csv               # Kết quả theo case và variant, flush từng episode
  summary.json               # Chỉ xuất sau khi hoàn tất vòng case
  baseline/ hoặc current/
    agent_provenance.json    # Descriptor, mọi public config và SHA-256
    sources/                # Policy, extractor và helper thực tế của lần chạy
    episode_000/
      environment.json      # Case và cấu hình môi trường
      observation_assumptions.json # Kiểm chứng giải mã/sensor (runner mới)
      decisions.jsonl.gz    # Public input + action + diagnostics + act latency
      result.json           # Outcome chuẩn, completion, speed, gap, latency
      error.txt             # Nếu có lỗi kỹ thuật
      interrupted.json      # Nếu có yêu cầu dừng ở giữa episode
```

Chỉ tổng hợp các episode đã hoàn tất. Đừng tính phần trace dang dở như một
episode thành công. Driving crash là outcome cần báo cáo; lỗi Python là
technical error riêng. Summary không được che một trong hai loại.

Các run cũ giữ nguyên định dạng ở thời điểm chạy. Runner mới bổ sung commit
MetaDrive và kiểm tra observation assumptions; không backfill dữ liệu giả vào
raw run cũ. Commit lúc đóng băng còn được ghi trong snapshot release.

## Tổng hợp và replay

Chỉ tổng hợp run TRAIN đã COMPLETED. Script xác minh source/config hash,
case/step và số dòng trace trước khi xuất report vào thư mục mới:

```powershell
python -X utf8 scripts/summarize_rule_based_experiment.py --baseline-run runs/rule_based_v030_baseline12_20261030 --candidate-run runs/rule_based_v040_regression9_20261030 --output-dir results/audits/rule_based/new_comparison
python -X utf8 scripts/run_rule_based_experiment.py --baseline-version 0.4.0 --variants baseline --env-seeds 0 --limit-cases 1 --run-id rb_new_snapshot_replay
```

Report chỉ so các case chung và liệt kê case không ghép được; không so tỷ lệ
khác coverage. Thư mục report đã tồn tại bị từ chối như run ID đã tồn tại.
Replay dùng policy/helper và extractor đóng băng của phiên bản được chọn.
Bản 0.5 có thể kiểm tra từng public input với policy/helper thực tế đã lưu:

```powershell
python -X utf8 scripts/replay_rule_based_trace.py --variant-dir runs/rule_based_v050_acceptance_family12_20261030/current --episode-index 8 --output results/audits/rule_based/new_replay_verification.json
python -X utf8 scripts/probe_rule_based_obstacle.py --run-id rb_new_obstacle_probe
```

Replay kiểm tra input round-trip, toàn bộ action/diagnostics và canonical action;
không so latency, không tạo driving score mới. Probe barrier tắt traffic và có
can thiệp spawn/remove, được ghi riêng. Không gộp probe/replay vào số map thành công.

Báo cáo release đã xác minh nằm tại `results/audits/rule_based/v0.4.0/`;
draft đầu giữ riêng tại `v0.4.0_draft01/`.

Bản 0.5: report 12 family tại `results/audits/rule_based/v0.5.0/`, 4 case seed1
tại `v0.5.0_env1/`, combined summary và index từng attempt trong thư mục chính.
Source chính thức ở `versions/0.5.0/`. `versions/dev050_*` là draft lịch sử
(kể cả bản lỗi), không phải release; xem `development_snapshot_index.json`.
Từ đợt thu gọn ngày 2026-10-31, 20 snapshot `dev050_01`–`dev050_11` và
`dev060_01`–`dev060_09` được lưu trong ZIP tại [history/](history/README.md).
Các đường dẫn draft trong index/report cũ là đường dẫn gốc cần khôi phục;
metadata lịch sử được giữ nguyên. Phải giải nén snapshot cần dùng theo hướng dẫn
trong history trước khi chọn `--baseline-version dev050_01` cho TRAIN diagnosis.
Giữ nguyên version string và source/config hash gốc, luôn dùng run ID mới.
`verified_runner.py` trong snapshot là bản tham chiếu lịch sử; lệnh chạy chính
là `scripts/run_rule_based_experiment.py --baseline-version <label>`.

## Quy tắc phiên bản

- Snapshot một lần dưới `versions/<version>/`; bảo toàn byte và SHA-256.
- Không chỉnh snapshot cũ để làm cho baseline chạy tốt hơn.
- Draft đã đóng gói phải giữ nguyên byte/đường dẫn gốc và có manifest SHA-256;
  xem [quy cách lưu và khôi phục](history/README.md). Release vẫn ở dạng file rời.
- Ghi tham số trước/sau, giả thuyết, test, run ID và mọi regression vào changelog.
- Đóng băng source/cấu hình cuối, rồi mới ghi kết luận release.
- Raw runs bị gitignore theo quy tắc dự án; kết quả quan trọng cần được trích
  sang báo cáo/CSV dưới `results/` kèm run ID và hash để còn truy nguyên.
