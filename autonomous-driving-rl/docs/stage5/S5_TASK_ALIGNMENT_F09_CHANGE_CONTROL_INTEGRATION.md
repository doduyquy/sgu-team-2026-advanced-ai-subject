# Stage 5 P1 / F09 — Tích hợp version producer

**Quyết định `F09_FIXED_INTEGRATED` trong phạm vi Stage 5 TRAIN/development.** Base là PR #41, SHA `92f3c840e44411edd0008f9959a9e3198577cea8`. Runtime được kiểm ở `976404853f2196aa2fe3e7b1cc1ed00f623aab62`. Benchmark eligibility của Stage 5 vẫn false.

## Semantics và compatibility

| Producer ID | Fingerprint SHA-256 | Lựa chọn và thực thi |
|---|---|---|
| `legacy_v1` | `c67e6bf80ec1156dcc63cb313938e2755d6570c58670aa242bc0d63638c56a26` | Mặc định. Checkpoint không có record producer chỉ hợp lệ ở đây. Các Stage khác và các runner lịch sử giữ nhánh này. Fingerprint gắn với source của extractor legacy và helper dùng chung. |
| `reference_route_station_proposal_v1` | `a7df3942bb874b40ea16474be43c7b7a48796646c9d59cf61da5d71ba5291725` | Stage 5 opt-in. 20 waypoint, 2,5 m, tim lane 0, cửa sổ checkpoint ±1, seam tolerance 1 mm. Source correction giữ SHA `948d6f7ae2e33b4052544220e85ea4ff90fc9d054f0567df96da6345946c972f` chuẩn hóa LF. |

- `build_agent_input` là điểm chọn producer dùng chung. Stage5TrainingEnv và Launcher executor truyền cùng ID ở đây. Chỉ waypoint/mask/route-end thuộc TaskContext đổi theo producer. Actor vẫn nhận 409D, Task 84D, Continuous Box2. Geometry/pose chỉ dùng tại producer và audit. RewardSpecV1, clock 5 × 0,02s và các fingerprint contract đã khóa giữ nguyên.
- Baseline CLI nhận `--task-context-producer` và ghi `training_run.json` trước bước simulator, kèm effective config, producer ID/fingerprint và trạng thái kết thúc. Trainer ghi identity trong log, summary và checkpoint. Khi save, env thực và metadata phải cùng ID. Loader kiểm exact producer record sau checksum và trước SB3 load, kể cả env được gắn. Policy input fingerprint mới cũng bao hàm producer. Checkpoint cũ giữ nguyên file và khả năng nạp.
- Launcher SANDBOX CLI và evaluator TRAIN tổng quát nhận cùng flag. Launcher factory ràng buộc SHA model và producer, executor đối chiếu với agent và ghi identity trong `run_manifest.json`. Agent corrected từ chối `act()` qua mặc định legacy. Evaluator ghi identity theo episode và summary. Các script lịch sử đang gọi checkpoint verifier mặc định chỉ chấp nhận legacy.
- Workbench Watch hiện ID theo checkpoint được chọn, chuyển nó thành tham số rõ ràng cho subprocess. Viewer xác thực bundle, producer và observation transform. Không mở thử cửa sổ MetaDrive native trong đợt này. Các test vòng đời Qt đã chạy. Comparison chỉ xem artifact và không nạp model.
- Stage 2 planners, Stage 3 MCTS, Stage 6 Learned A* và training data generator dùng producer mặc định. Contract cấu trúc của Launcher, benchmark preflight và từng agent vẫn giữ nguyên. Runtime mới chỉ được chọn qua Stage 5 SANDBOX/evaluator TRAIN hoặc trainer cấu hình opt-in.

## TRAIN parity và tần suất mask

Cùng trạng thái simulator, so từng bit của 409D từ đường train và Launcher. So thêm 325D trước Task với producer legacy. Tất cả sample bên dưới dùng seed môi trường 0.

| Geometry / trace | Observation khớp | Checkpoint đổi | Mẫu trên giao lộ T | Waypoint masked | Observation mask toàn bộ |
|---|---:|---:|---:|---:|---:|
| Easy SCS0, tham chiếu lane 0 đến gần đích | 419 | 5 | 0 | 8,449% | 0 |
| Easy SCS0, cua ở làn kề | 900 | 5 | 0 | 0,700% | 0 |
| Easy SCCS0, nhiều cua | 901 | 5 | 0 | 0% | 0 |
| Medium SCTCS11, giao lộ T | 1.201 | 9 | 272 | 0,121% | 0 |
| Medium SCTCS8, probe dừng sớm | 298 | 4 | 0 | 0% | 0 |

Tại witness 614, train và Launcher đều trả waypoint đầu `(3.0086, 4.9221, -0.0703)` trong sai số float32 như P1. Padding được xác nhận zero. Cột masked tính trên toàn bộ slot, gồm near-end. Trace P1 giữ phép đo station/world độc lập trước tích hợp. Tolerance seam và đi qua checkpoint được kiểm theo phạm vi fixture. Bản đồ tự giao, localization sai chưa bị phát hiện trong cửa sổ ±1 và junction ngoài nhánh đã đo vẫn chưa có coverage thực nghiệm.

**Quy tắc dừng:** Khi producer corrected không xác lập được station, cả hai wrapper phát `TaskContextLocalizationError` ngay ở decision đó. Trainer không đưa mẫu đó vào replay và Launcher đi qua đường đánh dấu run FAILED. Phép tiêm lỗi unset current lane tại reset trên Medium SCTCS11 kiểm cả hai wrapper. Với route định vị được, padding sau cuối tuyến là hợp lệ.

## Kiểm tra checkpoint và regression

- Kiểm checkpoint A S5.5 ở 30k còn thiếu trường producer. Nạp bằng legacy, chạy một TRAIN diagnostic 12 bước và một episode SANDBOX trên SCS0. Đối chứng corrected là bundle 0-step/0-update tạo từ trainer, chạy cùng hai runtime. Cả hai diagnostic xác nhận trạng thái model giữ nguyên và cả hai Launcher ghi COMPLETE, mỗi run một episode. Manifest, factory và evaluator đồng nhất identity. Đây là kiểm tích hợp.
- Launcher smoke thay kết quả preflight **chỉ trong harness** vì preflight nguyên bản tính hash của cả manifest VALIDATION và TEST ngay cả trong SANDBOX. Resolver dùng geometry TRAIN xác minh trước khi chạy. Giữ nguyên production preflight. Không đánh giá VALIDATION/TEST.
- Regression **164 PASS, 0 FAIL, 0 SKIP**, exit 0. Bao gồm Stage 5/Workbench, Platform AgentContract, RewardMetrics, MCTS và A* Learning. Full Launcher suite có test đọc holdout manifest nên dùng smoke nói trên. Kiểm producer sai phiên bản, sai fingerprint/config, sai env, legacy thiếu field và corrected actor thiếu lựa chọn.
- Trong triển khai, lần audit đầu lỗi serialize bộ đếm NumPy. Một fixture evaluator chưa nhận keyword mới, và smoke ban đầu đọc sai mức lồng manifest. Thử fault injection sau đó phát hiện đường lỗi khi native lane width được đọc trước localization. Các lỗi đó đã được sửa và chạy lại. Artifact compact ghi hash của các lần chạy cuối.

```powershell
conda run -n metadrive_sb3 python scripts/audit_stage5_f09_integration.py runs/stage5_f09_integration/geometry_release.json
conda run -n metadrive_sb3 python scripts/smoke_stage5_f09_producers.py runs/stage5_meter_reward/s5_5_meter_reward_seed101_20261120/arms/A/checkpoints/step_030000 runs/stage5_f09_integration/runtime_release
conda run -n metadrive_sb3 python -m unittest tests.test_stage5_task_producer tests.test_stage5_reference_station_proposal tests.test_stage5_task_alignment_p0 tests.test_stage5_training_foundation tests.test_stage5_sac tests.test_stage5_pilot tests.test_stage5_pilot_monitor tests.test_stage5_learnability tests.test_stage5_tau_ab tests.test_stage5_meter_reward tests.test_stage5_s55_evaluation tests.test_stage5_s55_analysis tests.test_reward_metrics tests.test_agent_contract tests.test_mcts_agent tests.test_astar_learning -q
```

Chọn output path mới khi tái lập. Geometry audit cần trace lịch sử TRAIN SCS0 dùng ở P0. Checkpoint model và raw logs nằm trong `runs/`. Xem [evidence JSON](../../results/stage5/task_alignment_p1_integration/evidence.json).

**Bước tiếp theo đề xuất:** kiểm chứng Task & Reward Alignment bằng trace TRAIN. Các số trên xác nhận producer, chưa đo năng lực lái hoặc thay đổi hành vi stationary của SAC.
