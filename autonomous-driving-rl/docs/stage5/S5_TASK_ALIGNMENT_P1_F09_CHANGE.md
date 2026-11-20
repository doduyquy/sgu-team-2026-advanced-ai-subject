# Stage 5 P1 / F09 — Station trên tuyến tham chiếu

**Quyết định `F09_BLOCKED_CHANGE_CONTROL`.** Bản correction đã qua kiểm chứng TRAIN và được tách riêng để review. Production producer cần được phê duyệt trước khi đổi semantics waypoint của policy.

## Nguyên nhân và cách sửa

`src/platform/agent_input.py::extract_task_context` lấy danh sách cặp node từ `navigation.checkpoints` bắt đầu tại `_target_checkpoints_index[0]`. Mỗi cặp chọn `road_network.graph[u][v][0]` làm tim làn tham chiếu, nhưng station bắt đầu lại lấy từ `navigation.current_lane.local_coordinates(ego_pos)`. Tại bước 614, current lane 1 và reference lane 0 thuộc cùng edge `1S0_0_ → 2C0_0_` với checkpoint [3,4]. Chênh lệch độ dài cung giải thích lỗi tại mẫu này ngay cả khi edge được chọn đúng. Sai station còn khiến waypoint đầu được lấy ở segment cong cũ trong khi station đúng đã vượt qua seam.

Ở stress case riêng, đặt xe qua một segment làm lane localization báo edge kế tiếp trong ba bước còn checkpoint giữ [0,1]. Đây là cơ chế thứ hai. Chỉ chiếu vào lane 0 của checkpoint được báo có thể vẫn chọn sai edge.

Bản đề xuất `reference_route_station_proposal_v1` dùng danh tính lane từ graph để xác nhận đúng một edge ở checkpoint hiện tại hoặc hai edge giáp nó. Lựa chọn này không phụ thuộc khoảng cách Euclid tới các nhánh quanh ego. Chiếu pose lên lane 0 của edge đã xác nhận, cộng chiều dài các edge đi trước để có station toàn tuyến. Lấy điểm tại station này cộng `k × 2,5 m` và chuyển phần dư qua các edge kế tiếp. Khi thiếu route, không xác định được edge duy nhất, gặp segment dài 0 hoặc seam hở hơn 1 mm, trả waypoint zero và mask false. Trạng thái không định vị được cũng không đặt cờ route end. Biến đổi heading về ego dùng cùng wrap và làm tròn 4 chữ số như canonical.

## Regression TRAIN

Geometry `geom_easy_SCS_seed0`, environment seed 0, MetaDrive 0.4.3 tại `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`. Fixture tái dùng hành động tham chiếu lịch sử cho chuyến lane 0 và bộ điều khiển tốc độ thấp dùng riêng để đo tuyến trên adjacent lane. Legacy và proposal có cùng pose, action, reward, progress, terminal và checkpoint ở từng bước. Không có dữ liệu fixture đi vào SAC replay.

| Mẫu | Số observation | Waypoint lỗi legacy → proposal | Kết quả |
|---|---:|---:|---|
| Straight và curve lane 0 | 419 | 0 → 0 | Task 84D bằng từng bit trên toàn trace |
| Adjacent straight | 45 | 0 → 0 | Giữ tim lane 0 và Task 84D bằng từng bit |
| Adjacent curve cùng 5 lần đổi checkpoint | 900 | 6.300 → 0 | Correction ở 315 observation |
| Checkpoint chậm trong stress case | 3 | 60 → 0 | Anchor đúng edge giáp checkpoint |
| Near end | có trong trace lane 0 | 0 → 0 | Mẫu cuối 1 waypoint valid; padding còn lại zero |
| Missing, degenerate, route hở, lane ngoài tuyến | unit fixtures | — | Mask false, padding zero |
| Hướng ego ngược tuyến | unit fixture | — | Waypoint hợp lệ có x âm, không clip |

Ở bước **614**, station current **130,074596 m**, station chiếu lên tim reference **137,759163 m**, delta **−7,684567 m**. Station mục tiêu toàn tuyến cho waypoint đầu là **261,585068 m**. Ego-relative `(x m, y m, heading rad)` đổi từ `(-4.6615, 5.3766, -0.0181)` thành `(3.0086, 4.9221, -0.0703)`. Sai số world từ **7,683527 m** xuống **0,000058 m**, sai số station đo so với segment mục tiêu còn **0,000027 m**, sai số heading còn **0,0000253 rad**. Qua checkpoint 614→615, station tăng **0,242303 m** và world waypoint đầu dịch **0,242233 m**.

Dùng chuẩn station, world position và heading từ tim tuyến và chiều dài từng edge, với dung sai **0,00015 m / 0,00015 m / 0,00008 rad** cho sai số do làm tròn 4 chữ số và float32. Sai số proposal lớn nhất trên bốn trace là **0,00007053 m world**, **0,00006835 m station**, **0,0000500 rad heading**. Evidence có tọa độ thế giới, pose, lane, checkpoint, projected station, từng điểm, mask và các lần đổi checkpoint. World pose chỉ lưu trong audit.

## Change control và tương thích

- Producer chung được gọi từ Launcher executor, Stage 5 training env và pipeline tạo dữ liệu A* heuristic. Consumer trực tiếp gồm `planning_graph`, Hybrid A*, Stage 3 MCTS và Stage 5 encoder/SAC. Stage 5 inference nhận AgentInputV1 đã được tạo sẵn và không có graph hoặc pose để khôi phục station đúng từ 409D. Chỉ sửa training env sẽ tạo lệch nghĩa giữa train và inference.
- PR chỉ thêm proposal, audit, tests và bằng chứng. Canonical producer, Launcher, Stage 5 env, agent và checkpoint loader vẫn dùng behavior hiện hành. Các checkpoint S5.3/S5.4/S5.5 vẫn đọc legacy semantics. Tại audit seam, 325 chiều Core/Navigation/LiDAR/Traffic, lane width và goal direction so bằng bit giữa hai chế độ. 409D, Box2, 5 substeps × 0,02s, RewardSpecV1 và quyền thông tin giữ nguyên.
- Đề xuất phê duyệt một producer version có fingerprint, chọn rõ tại cả training và Launcher evaluation, gắn vào run/checkpoint provenance và giữ Legacy V1 mặc định cho checkpoint cũ. Sau phê duyệt cần xác nhận train/inference cùng producer ID và kiểm các consumer trước khi kích hoạt. SHA-256 nguồn proposal chuẩn hóa newline LF là `948d6f7ae2e33b4052544220e85ea4ff90fc9d054f0567df96da6345946c972f`.
- Phạm vi chưa xác minh gồm junction tự giao thực tế, map có seam gián đoạn chủ đích và localization báo sai edge ngoài cửa sổ ±1. Unit fixtures chứng minh không snap vào lane lân cận ngoài tuyến và reject edge nằm xa checkpoint. Hiện chưa đo ảnh hưởng tới nguyên nhân stationary hoặc năng lực lái của SAC.

## Tái lập

Từ `autonomous-driving-rl/`, cần có trace Stage 1 TRAIN tại đường dẫn artifact S5.5 mà P0 đã dùng. Chọn tên output mới nếu chạy lại.

```powershell
conda run -n metadrive_sb3 python scripts/audit_stage5_f09.py --mode legacy --output runs/stage5_task_alignment_p1/legacy_regression.json
conda run -n metadrive_sb3 python scripts/audit_stage5_f09.py --mode proposal --output runs/stage5_task_alignment_p1/proposal_regression.json
conda run -n metadrive_sb3 python -m unittest tests.test_stage5_reference_station_proposal tests.test_stage5_task_alignment_p0 tests.test_stage5_training_foundation tests.test_stage5_sac tests.test_stage5_pilot tests.test_stage5_pilot_monitor tests.test_stage5_learnability tests.test_stage5_tau_ab tests.test_stage5_meter_reward tests.test_stage5_s55_evaluation tests.test_stage5_s55_analysis tests.test_reward_metrics tests.test_agent_contract tests.test_mcts_agent -q
```

Legacy regression exit **1** đúng kỳ vọng tại `c09a07b7242fbce8c97ea3bdbee60bd234f59835`. Proposal exit **0** tại `00c367406f7e7dba6c6a9d8e9c3c166cb3e68894`. Lần audit legacy đầu đã ghi trace nhưng gặp lỗi serialize bộ đếm NumPy khi in stdout. Lần legacy cuối ghi số vi phạm và exit 1 đúng theo invariant. Regression **155 PASS, 0 FAIL, 0 SKIP**, exit **0**. Launcher suite đầy đủ có test đọc holdout manifest nên nằm ngoài đợt chạy này. Compact [evidence.json](../../results/stage5/task_alignment_p1_f09/evidence.json) chứa SHA-256 raw before/after. Raw trong `runs/` bị gitignore.
