# Rule-Based Agent — lịch sử phiên bản và experiment

Mỗi phiên bản phải ghi thay đổi hành vi, cấu hình, source/config hash, kiểm thử,
experiment ID, kết quả và giới hạn còn lại. Raw run không được ghi đè.
Các trạng thái đang chạy/chưa đủ chứng cứ không được mô tả như đã đạt nghiệm thu.

## 2026-10-31 — thu gọn bản lưu phát triển, giữ agent 0.6.1

Đổi cách lưu lịch sử theo phương án người dùng đã duyệt; không tạo phiên bản agent mới.

- Đóng gói 180 file của 20 snapshot `dev050_01`–`dev050_11`,
  `dev060_01`–`dev060_09` vào một ZIP, kèm manifest đường dẫn và SHA-256.
  Chỉ bỏ bản rời sau khi đọc lại từng entry ZIP và đối chiếu hash với nguồn.
- Giữ nguyên snapshot release 0.3.0–0.6.1, gồm `maintenance_sources` của 0.6.1;
  giữ toàn bộ raw runs, CSV, report draft/trial và bằng chứng replay.
- Bổ sung hướng dẫn giải nén draft trước khi dùng đường dẫn lịch sử hoặc
  `--baseline-version dev...`; không sửa metadata/index thí nghiệm đã đóng băng.
- Đối chiếu danh sách file và SHA-256 trước/sau cho core, tools, tests, configs,
  datasets, release, results và raw runs. Không chạy lại test/simulator/experiment
  theo yêu cầu người dùng; không có kết quả lái xe mới trong đợt thu gọn này.
- Manifest, kết quả đối chiếu và hướng dẫn khôi phục tại
  [history/README.md](../experiments/rule_based/history/README.md).
- Kết quả: 180/180 file ZIP khớp nguồn; 4.151/4.151 file được bảo vệ giữ nguyên
  danh sách và SHA-256, gồm 3.595 file raw runs. Git giảm 518 xuống 340 mục thay đổi
  (180 file rời được thay bằng ZIP + README); không stage/commit.

## 0.6.1 — dọn source, giữ hành vi P4

Maintenance release, 2026-10-31. [Chi tiết và các lượt kiểm tra](SOURCE_CLEANUP_V0_6_1.md).

- Bỏ tính curvature preview không dùng và counter elapsed không được đọc.
  Dọn 79 import không dùng trong 31 file runtime; giữ public consumers/exports
  và giữ nguyên từng byte adapter để replay nguồn cũ vẫn tương thích.
- Retire 5 root inspect scripts, manual `test_course_env.py`, và audit partial-trace
  cũ. Giữ nguyên bản nguồn/hash trong `experiments/legacy_source/` cho lịch sử.
- Texture FINISH sinh tự động chuyển sang `runs/.cache/finish_visual/`; ba PNG
  khớp hash bản cũ. Xóa 10 cache bytecode và 3 PNG sinh tự động khỏi source.
- Legacy evaluator có main guard và finally/close, không chạy simulator khi import.
  README/tool guides chỉ rõ hai agent ID và các file helper. Thêm một test runner
  thống nhất, tách backend/Qt thành hai process để tránh native import-order crash.
- Dọn assertions GUI cũ: registry đã có research agents, hash bốn fixture gốc
  vẫn khóa nguyên; so sánh path theo resolve cho alias Windows 8.3. Không đổi
  benchmark lock/hash hoặc scientific contract để làm tests qua.

Kết quả cuối: **547 tests** (423 backend +124 GUI), **1992 bước replay** khớp
decision/diagnostics/action với nguồn 0.6.0, toàn bộ tham số policy giữ nguyên.
CLI TRAIN Easy/SCS/geom0/env0/agent42 COMPLETE, success **418 bước**. Hai lần
discovery chung bị native Qt crash và lượt GUI có 4 assertion lỗi được giữ log.

Snapshot maintenance ở `experiments/rule_based/versions/0.6.1/`; inventory,
hash trước/sau, deleted/retired paths, test/replay/CLI evidence ở
`results/audits/source_cleanup/v0.6.1/`. Không xóa hoặc ghi đè release cũ/log
experiment. Đây không phải chạy lại 16-case P4 benchmark cho 0.6.1; evidence
driving cũ vẫn thuộc 0.6.0. P5/audit/benchmark eligibility không thay đổi.

## 0.6.0 — P4, chuyển làn theo tuyến và vượt có điều kiện

Đóng băng development release SANDBOX/AUDIT ngày 2026-10-31 (Asia/Saigon).
**P4 hoàn tất triển khai và nghiệm thu trong phạm vi công bố**; toàn bộ kế hoạch
vẫn còn P5, full 108 TRAIN và audit. Benchmark eligibility giữ false.
[Báo cáo](RULE_BASED_V0_6_0_EXPERIMENT_REPORT.md),
[thiết kế](RULE_BASED_P4_DESIGN.md), [work log](RULE_BASED_V0_6_0_WORKLOG.md).

- Schema 1.3.0 có lane geometry/vạch thực/cục bộ nối tuyến, masked trong
  CORE_ONLY, giữ 259D và các cap traffic/route. PG shared right divider đọc từ
  left boundary của neighbor; UNKNOWN/solid/edge không cấp phép đổi làn.
- Mandatory trước return/pass; gap trước/sau, TTC, quãng phanh, quét thân xe
  theo CV/CT/road paths và ba profile tiến dọc, LiDAR và overflow guards.
- Quintic path/target ổn định qua nearest-lane switch; progress và commit xét
  thân xe xoay. Hủy trước commit khi đường cũ an toàn; giữ target, phanh và tiếp
  tục khi clear sau commit. Cooldown 6 s, settle .5 s; reset xóa hết episode state.
- Giữ 45/24/8 km/h cơ bản. P4 prepare/change 24, admission <=30 km/h, TTC >=4 s,
  min gap 4 m + response/headway/phanh, path 16–30 m và end buffer 10 m.
  Mandatory upstream >35 m; optional pass/return không bắt đầu khi thấy nút
  trong 50 m. Không dùng init gate để dừng một change hợp lệ đang thực hiện.
- Waypoint không nhảy qua connector gap >.3 m; endpoint rounding dedup có
  regression test. Diagnostics giữ budget 50 key, mở rộng replay typed lane data.

Nghiệm thu **cùng 16 TRAIN case** với 0.5: success **13 → 13**, crash **0 → 0**,
không mất success, không road/solid-boundary. Ba Extreme vẫn TIMEOUT. Sáu đổi
làn với lý do vượt hoàn tất trong năm case tự nhiên; không gán thêm success
cho controlled probes. Hai Hard SCXOCS có completion cuối nhỏ hơn nhưng vẫn
SUCCESS, giữ số đo trong CSV/report. Trần latency 10 ms chưa đạt ở traffic đông.

**279 tests**, trong đó **26 P4 tests**; **6 physical probes** đạt, **929 bước**
replay khớp cả khi import flat archive thật. CLI Hard/XTOCS/env0/agent42 COMPLETE,
success 1063 bước. Snapshot chính thức ở `experiments/rule_based/versions/0.6.0/`;
chín draft source groups được archive từ raw snapshots, có nhãn retrospective
và DRAFT_NOT_RELEASE. Hash/config tại snapshot và `results/audits/rule_based/v0.6.0/`.

Index giữ **37 run attempts**: 16 COMPLETED, 2 CLI COMPLETE, 16 INTERRUPTED,
1 setup FAILED, 2 COMPLETED_WITH_FAILED_EXPECTATIONS. Một candidate accept02
Extreme có CRASH_VEHICLE nên không release; source cuối tăng guard optional
approach 50 m rồi chạy lại toàn bộ coverage. Mọi lỗi, regression và run dừng
được giữ nguyên, cùng source/config/trace/outcome tương ứng. Không ghi đè 0.5.

## 0.5.0 — triển khai P2/P3, tracking và lifecycle qua nút

Trạng thái: **đã đóng băng development release SANDBOX/AUDIT**, đã nghiệm thu
trong phạm vi TRAIN công bố; chưa hoàn tất toàn bộ kế hoạch/benchmark.
[Báo cáo 0.5.0](RULE_BASED_V0_5_0_EXPERIMENT_REPORT.md),
[work log từng lần thử](RULE_BASED_V0_5_0_WORKLOG.md),
[thiết kế P2/P3](RULE_BASED_P2_P3_DESIGN.md).

### Đã thay đổi

- Ego-motion compensation, association actor một-một không dùng slot làm ID,
  TTL 0.6 s, CV/CT và giả thuyết đường kết nối từ pose/heading quan sát.
- Actor hypotheses có cap/overflow; không đọc navigation/policy/tương lai actor.
  Public schema 1.2.0 còn pending audit, giữ 259D, 8×7 và range 50 m.
- FSM approach/yield/commit/traverse/exit/blocked, khóa branch qua nút, yield
  boundary hình học và phân biệt inlet merge/branch exit của vòng xuyến.
- Space-time admission 12 s, xác nhận gap 0.5 s; kiểm tra exit queue. X/T bị
  chặn chờ thêm 8 m upstream và queue-clear hysteresis 2 s, không timeout ép ga.
- P2 timed cut-in 4 s ngoài passage, xét thân xe phía sau và tăng tốc thoát có
  giới hạn khi an toàn theo dự báo. Trong passage giữ phân xử road-path P3,
  free-motion gần, P2 horizon 3 s và AEB để tránh dừng do giả thuyết xa.
- Giữ trần 45/24/8 km/h và ga 0.55. Trace gzip level 3 vẫn lưu đủ từng dòng.
- Script replay kiểm public input round-trip cùng action/diagnostics; probe
  barrier vật lý lưu riêng, có runtime và trạng thái kể cả lỗi setup.

### Nhận được từ experiment

| Coverage ghép cặp | Success 0.4 → 0.5 | Crash 0.4 → 0.5 |
|---|---:|---:|
| 12 family/env0 | 7 → 9 | 3 → 0 |
| 4 case mục tiêu/env1 | 3 → 4 | 1 → 0 |
| Tổng 16 case riêng biệt | **10 → 13** | **4 → 0** |

Không mất case thành công; không out-of-road/chạm vạch trong candidate. Ba
Extreme vẫn timeout, không gọi thành success. Có case chậm hơn và p95 >10 ms
ở traffic đông; tất cả giữ trong CSV/report. Đây là policy + input change,
không phải policy-only ablation, full 108 TRAIN hay canonical VALIDATION/TEST.

252 tests đạt. Replay Hard/SCXOCS/env1 khớp 1482 bước; simulator từ archive
0.5 tái tạo đúng 895 bước Medium/SCXCCS. CLI Hard/XTOCS COMPLETE success 1/1
với 1065 dòng trace; physical obstacle probe source cuối đạt 2/2 scenario.

Source/config chính thức tại `experiments/rule_based/versions/0.5.0/`.
11 source draft trước đó được giữ riêng dưới `versions/dev050_*`, gồm bản lỗi;
không sửa snapshot 0.3/0.4 hoặc ghi đè raw run. Index 23 run và source/config
hash nằm ở `results/audits/rule_based/v0.5.0/experiment_index.json`.

### Còn lại

P4 chuyển làn/vượt và metadata biên/làn đích, jerk/steering-rate, audit contract
1.2.0, suite TRAIN mở rộng và canonical VALIDATION/TEST. Prediction/association
hữu hạn không là bằng chứng biết ý định thật hoặc bảo đảm mọi map đều đến đích.

## 0.4.0 — phát triển P0/P2 và một phần P3

Trạng thái: **đã đóng băng bản phát triển SANDBOX/AUDIT**. Chưa benchmark-ready.
Chi tiết: [báo cáo 0.4.0](RULE_BASED_V0_4_0_EXPERIMENT_REPORT.md).

### Thay đổi

- Shared input: waypoint bắt đầu từ làn hiện tại; chọn làn nối theo endpoint
  và heading, không mặc định `lanes[0]`. Không xuất qua khoảng đứt lớn hơn bề
  rộng làn. Phân biệt route hết thật với dữ liệu nối bị thiếu.
- Contract schema 1.1.1 ghi rõ semantics trên; vector 259D và quyền thông tin
  không thêm trường mới. Vẫn pending audit, chưa mở benchmark eligibility.
- Xe trước được chiếu lên polyline tuyến cong, xét kích thước thân xe và
  khoảng cách bumper thay cho dải thẳng vô hạn trước ego.
- Dự báo chuyển động vận tốc không đổi trong 3 s, lấy mẫu 0.2 s và nới envelope
  giữa các mẫu để phát hiện xe cắt ngang/cắt vào sớm hơn.
- LiDAR: phanh với hit chưa phân loại nằm trong hành lang; loại hit thuộc thân
  actor đã biết để không coi xe đang chạy như một bức tường đứng yên.
- Giảm trần khi actor context overflow; dừng có kiểm soát nếu phần lớn scan
  LiDAR không dùng được hoặc hướng tuyến mơ hồ.
- Nhường tại lối vào junction khi xe khác có thể chiếm vùng nút; đây là luật
  bảo thủ về xung đột hình học, chưa phải quyền ưu tiên giao thông đầy đủ.
- Ghi trạng thái hành vi và giữ khoảng chờ giải phóng nguy cơ 0.5 s.
  Descriptor/registry đổi sang `stateful_within_episode=True`; reset xóa timer.
- Giữ trần đường thoáng 45 km/h, junction 24 km/h, phục hồi 8 km/h, ga 0.55.

### Các lần thử

| Experiment ID | Mục đích | Nguồn/kết quả |
|---|---|---|
| `rule_based_v030_baseline12_20261030` | Chạy lại snapshot 0.3.0 trên 12 family TRAIN, env seed 0 | Raw trong `runs/<id>`; dùng đúng extractor lịch sử |
| `rule_based_v040_trial01_20261030` | Thử corridor, prediction, LiDAR và hysteresis đầu tiên | Giữ nguyên source/config snapshot của lần thử này |
| `rule_based_v040_trial02_20261030` | Kiểm tra luật nhường trước lối vào trên 4 case TRAIN đã xác định | Manifest `train_targeted_v040.json` |
| `rule_based_v040_cli_smoke_20261030` | Kiểm tra CLI chính thức và logging mới | Easy/SCS/0/env0, không phải benchmark |
| `rule_based_v040_regression9_20261030` | Xác minh cấu hình cuối trên 9 Easy/Medium/Hard case | 7/9 clean success; baseline cùng 9 case 5/9 |
| `rule_based_v040_snapshot_replay_20261030` | Replay từ archive 0.4.0 | 597 public input/action/diagnostics khớp hoàn toàn run gốc, không so latency |
| `rule_based_v040_interrupt_probe_20261030` | Thử dừng sạch bằng STOP_REQUESTED | INTERRUPTED, 0/1 episode hoàn tất; không tính vào điểm driving |

Phát hiện từ trial01: Medium/SCXCCS/0/env0 chuyển từ crash ở 0.3.0 sang đến
đích ở draft 0.4.0. Hard/SCTXrCS/2/env0 vẫn crash: trace cho thấy ego đã dừng
trong vùng xung đột và bị xe đang rẽ tiến vào. Vì vậy trial02 thêm luật nhường
trước lối vào; không xóa các thất bại của trial01.

Trial01 đã hoàn tất đủ 12 case (6 success, 4 crash, 2 timeout), giữ báo cáo
riêng `results/audits/rule_based/v0.4.0_draft01/`. Đây là source/config draft,
khác bản đóng băng cuối dù cùng chuỗi version trong giai đoạn phát triển.

Kết quả bản cuối trên 9 case ghép cặp: clean success **5 → 7**, crash **4 → 2**,
không mất case thành công nào. Hai cải thiện là Medium/SCXCCS/0/env0 và
Hard/SCTXrCS/2/env0. Hard/SCXOCS/0/env0 và Hard/XTOCS/0/env0 vẫn crash, completion
còn giảm so với 0.3; đã ghi rõ trong CSV và báo cáo, không coi là cải thiện.
Không dùng 7/9 để tuyên bố đạt mục tiêu >=95% hoặc thành công trên Extreme.

Kiểm tra: 205 tests agent/input/logger/launcher/visual qua; thêm 2 kiểm tra
integrity report qua. Hash source/config và toàn bộ trace count của các run
ghép cặp đã được kiểm chứng bằng script tổng hợp. Snapshot cuối nằm trong
`experiments/rule_based/versions/0.4.0/`, không ghi đè bản 0.3.0.

### Logging bắt buộc từ phiên bản này

- Thư mục run mới: metadata/runtime, manifest case đã resolve, config/hash,
  source snapshots, trace public input/action/diagnostics từng bước, kết quả
  từng episode, CSV tổng hợp, summary và trạng thái COMPLETED/FAILED/INTERRUPTED.
- CLI/Workbench cho agent có `export_config()` ghi `agent_provenance.json`,
  `sources/` và `agent_decisions.jsonl.gz` bên cạnh các log platform hiện có.
- Mỗi lần thử source khác nhau có run ID và source hash riêng, kể cả cùng
  version đang phát triển. Snapshot phiên bản chính thức chỉ tạo sau xác minh.
- Quy tắc này được lưu trong `autonomous-driving-rl/AGENTS.md` cho công việc sau.

### Chưa hoàn thành

Chưa chuyển làn/vượt, chưa phân biệt vạch, chưa tracking actor có ID/TTL,
chưa xử lý đầy đủ quyền ưu tiên/đèn/biển và khóa nhánh qua junction. Prediction
không biết ý định tương lai thật của actor. Chưa có điều khiển jerk/steering-rate
đầy đủ, chưa nghiệm thu 108 TRAIN episode và canonical VALIDATION/TEST.

## 0.3.0 — tốc độ thích ứng và theo xe cơ bản

**Hồ sơ lịch sử được bổ sung hồi tố**, không phải log release đã tồn tại từ đầu.

- Nâng trần đường thoáng 22 → 45 km/h, ga 0.4 → 0.55.
- Thêm quãng phanh trước cua/junction, trần visibility và headway/TTC xe trước.
- Junction 24 km/h, phục hồi 8 km/h; policy deterministic, stateless.
- So sánh lịch sử trên 7 TRAIN case: 6 Easy/Medium cùng đến đích, thời gian giảm
  16.1–39.6%; Hard/SCXOCS/0/env0 crash ở cả 0.2 và 0.3.
- Dữ liệu: `results/audits/rule_based/speed_policy_comparison.csv`.
- Source policy và extractor được lưu nguyên trạng trước chỉnh 0.4 tại
  `experiments/rule_based/versions/0.3.0/`. Thời điểm lưu snapshot là lần phát
  triển 0.4 này; không giả định đó là timestamp release 0.3.
- Các trace mới của baseline là thí nghiệm chạy lại, không phải trace gốc của
  lần so sánh 7 case lịch sử.

## 0.2.0 — giữ làn, preview cua, tốc độ thấp

**Ghi nhận hồi tố từ tài liệu và CSV hiện có.** Không còn snapshot source riêng
đã đóng băng để bảo đảm tái tạo nguyên trạng toàn bộ phiên bản này.

- Giải mã heading/lateral/speed; giữ làn bằng phản hồi và heading preview.
- Trần cruise 22 km/h, phục hồi 8 km/h; giảm tốc theo curvature.
- Evidence: `after_easy.csv`, `after_easy_seed7.csv`, `after_medium.csv` và phần
  kết quả lịch sử trong `RULE_BASED_AGENT_GUIDE.md`.
- Không có luật theo xe, nhường đường, vật cản hay chuyển làn đầy đủ.

## 0.1.0 — bản đầu được ghi trong tài liệu cũ

Hồ sơ không đầy đủ, ghi hồi tố: waypoint following và ga cố định; có kết quả
chạm vạch được mô tả trong guide và `results/audits/rule_based/before.csv`.
Không suy diễn source hash/config/seed thiếu từ tên file hoặc kết quả quan sát.
