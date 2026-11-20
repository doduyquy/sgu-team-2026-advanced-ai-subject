# S5.5 / P2-F — Protocol đăng ký trước

## Câu hỏi

Kiểm tra xem reset TRAIN tại một điểm hợp lệ gần đích, theo lịch cố định, có giúp SAC từ đầu nhận được kinh nghiệm hoàn thành sạch và có chuyển giao sang bài toán full-route từ điểm xuất phát chính thức hay không.

## Phạm vi và hợp đồng giữ nguyên

Chỉ dùng tám geometry Easy `PILOT_TRAIN` và sáu geometry Easy `PILOT_DIAGNOSTIC` đã khóa trong protocol P2-D. Không đọc VALIDATION/TEST. AgentInputV1 vẫn 409D; producer `reference_route_station_proposal_v1`; transform `pilot_groupwise_physical_v1`; action `continuous_box2_v1`; RewardSpecV1 và horizon gốc không đổi. SAC khởi tạo mới; actor LeakyReLU(0.01), critic ReLU; sampling stochastic SAC chuẩn. Không dùng teacher, Stage 3, imitation hoặc demonstration replay.

## Hai nhánh

- A: mỗi episode bắt đầu tại reset chính thức.
- B: episode 0–2 còn 60m đến đích, episode 3–5 còn 120m, episode 6 trở đi bắt đầu tại điểm gốc. Lịch chỉ phụ thuộc chỉ số episode, không đọc kết quả. Destination gốc được giữ nguyên và xe bắt đầu đứng yên.

Chỉ khác biệt đăng ký là phân phối start-state trong Stage 5 TRAIN. Mỗi checkpoint: 0/2k/5k/10k, tối đa 10,000 bước thực mỗi nhánh. Không resume hay nối run nếu bị gián đoạn. Không tự gia hạn ngân sách.

## Feasibility

`check_stage5_p2f_feasibility.py` đã kiểm tra 8 geometry × 2 mốc (16 reset) trong MetaDrive 0.4.3, TRAIN seed 0. Các kiểm tra gồm geometry/destination/checkpoint suffix, reference lane, sai số vị trí và station, remaining distance, tốc độ khởi tạo, cờ terminal, horizon và waypoint F09 hợp lệ. Evidence và SHA-256 nằm trong `results/stage5/p2f_curriculum_preflight_20261121/`.

## Reset và reward accounting

Route spawn được giải từ reset route thật của cùng geometry/environment seed; destination được pin theo route đó. Không cắt đích, không gắn nhãn terminal và không đổi horizon. Reward progress được tính từ `route_completion` sau reset, không nhận reward giả tại bước 0. MetaDrive tái chuẩn hóa route fraction trên edge suffix sau spawn giữa tuyến; vì vậy reward theo progress fraction của B có thể có scale vật lý khác A. Đây là giới hạn diễn giải của treatment, được ghi log chứ không che giấu. RewardSpecV1 implementation vẫn nguyên vẹn.

## Evaluation đã khóa

Mỗi checkpoint: deterministic 300-step probes trên sáu full-route diagnostic TRAIN case; censored vẫn là `PROBE_CENSORED`. Tại 10k: full episodes trên sáu full-route diagnostic case cho A/B, cộng đánh giá competence riêng trên sáu diagnostic geometry ở mốc 60m và 120m. Các kết quả curriculum không được gọi là full MapSuite competence hay benchmark. Không có benchmark eligibility.

## Điều kiện diễn giải

Bằng chứng curriculum success chỉ là hoàn thành hành trình bắt đầu giữa route. Chuyển giao chỉ được xem xét từ full-route diagnostics. Tốc độ hoặc progress không thay thế clean success và safety. Một training seed là pilot, không đủ kết luận tổng quát hoặc ưu thế thống kê. Nếu không có clean success hoặc tín hiệu phân biệt đáng tin cậy, kết quả là âm/inconclusive; không huấn luyện tiếp trong P2-F.

## Reproduction

Từ `autonomous-driving-rl/`, trong môi trường `metadrive_sb3`:

```powershell
conda run -n metadrive_sb3 python scripts/check_stage5_p2f_feasibility.py
conda run -n metadrive_sb3 python scripts/register_stage5_p2f_protocol.py
conda run -n metadrive_sb3 python scripts/smoke_stage5_p2f_curriculum.py
conda run -n metadrive_sb3 python scripts/run_stage5_p2f_curriculum.py
conda run -n metadrive_sb3 python scripts/run_stage5_p2f_evaluation.py --run-id s5_p2f_start_distance_seed101_20261121_retry3
conda run -n metadrive_sb3 python scripts/summarize_stage5_p2f.py
```

P2-F protocol revision 4 SHA-256: `212ea389ebe94c294850443d4ad43688638cd05ca6d4b845c90844c99c6578a8`.

Revision 2 added explicit metadata for standard SB3 stochastic sampling and inert OU compatibility defaults after the first training launch stopped before model construction. Revision 3 exposed `rho` and `stationary_std` at the keys expected by the shared P2 runner after a second launch stopped before model construction. Revision 4 recorded the shared helper's exact `train_freq` key after kwargs construction stopped before model creation or environment steps. No optimizer update or training transition occurred in those failed launches. A separate model-construction preflight then passed. The failed launch directories are preserved under ignored `runs/stage5_p2f/` and excluded from scientific results.

One evaluation attempt initially stopped before any episode due the P2-D evaluator expecting `training.root_seed`; a later attempt recorded one probe before finding a wrong helper import. Those partial outputs are preserved separately. The final evaluation uses the logged adapter in `runs/.../evaluation/EVALUATION_ADAPTER.json`; it maps `training.root_seed` to the identical registered `sac.root_seed` and imports the existing P2-D actor-analysis helper. It did not modify training protocol, checkpoints, or results.

Protocol JSON, protocol SHA-256, source hashes, geometry manifest, producer fingerprint, initial weight identity, checkpoint/model hashes and output run manifest are the sources of verification. Raw checkpoints and replay are stored under `runs/stage5_p2f/` and are not committed to Git.
