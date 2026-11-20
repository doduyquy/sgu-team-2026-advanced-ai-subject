# S5.5 / P2-F — Protocol đăng ký trước

## Câu hỏi

Kiểm tra xem reset TRAIN tại một điểm hợp lệ gần đích, theo lịch cố định, có giúp SAC từ đầu nhận được kinh nghiệm hoàn thành sạch và có chuyển giao sang bài toán full-route từ điểm xuất phát chính thức hay không.

## Phạm vi và hợp đồng giữ nguyên

Chỉ dùng tám geometry Easy `PILOT_TRAIN` và sáu geometry Easy `PILOT_DIAGNOSTIC` đã khóa trong protocol P2-D. Không đọc VALIDATION/TEST. AgentInputV1 vẫn 409D; producer `reference_route_station_proposal_v1`; transform `pilot_groupwise_physical_v1`; action `continuous_box2_v1`; RewardSpecV1 và horizon gốc không đổi. SAC khởi tạo mới; actor LeakyReLU(0.01), critic ReLU; sampling stochastic SAC chuẩn. Không dùng teacher, Stage 3, imitation hoặc demonstration replay.

## Hai nhánh

- A: mỗi episode bắt đầu tại reset chính thức.
- B: episode 0–2 còn 60m đến đích, episode 3–5 còn 120m, episode 6 trở đi bắt đầu ở điểm gốc. Lịch chỉ phụ thuộc chỉ số episode, không đọc kết quả. Destination gốc được giữ nguyên và xe bắt đầu đứng yên.

Chỉ khác biệt đăng ký là phân phối start-state trong Stage 5 TRAIN. Mỗi checkpoint: 0/2k/5k/10k, tối đa 10,000 bước thực mỗi nhánh. Không resume hay nối run nếu bị gián đoạn. Không tự gia hạn ngân sách.

## Feasibility

`check_stage5_p2f_feasibility.py` đã kiểm tra 8 geometry × 2 mốc (16 reset) trong MetaDrive 0.4.3, TRAIN seed 0. Các kiểm tra gồm geometry/destination/checkpoint suffix, reference lane, sai số vị trí và station, remaining distance, tốc độ khởi tạo, cờ terminal, horizon và waypoint F09 hợp lệ. Evidence và SHA-256 nằm trong `results/stage5/p2f_curriculum_preflight_20261121/`.

## Reset và reward accounting

Route spawn được giải từ reset route thật của cùng geometry/environment seed; destination được pin theo route đó. Không cắt đích, không gắn nhãn terminal và không đổi horizon. Reward progress được tính từ `route_completion` sau reset, không nhận reward giả tại bước 0. MetaDrive tái chuẩn hóa route completion trên edge suffix sau spawn giữa tuyến; vì vậy reward theo progress fraction của B có thể có scale khác A. Đây là giới hạn diễn giải của treatment, được ghi log chứ không che giấu. RewardSpecV1 implementation vẫn nguyên vẹn.

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
conda run -n metadrive_sb3 python scripts/evaluate_stage5_p2f_curriculum.py
```

Protocol JSON, protocol SHA-256, source hashes, geometry manifest, producer fingerprint, initial weight identity, checkpoint/model hashes và output run manifest là nguồn xác minh. Raw checkpoints và replay được lưu dưới `runs/stage5_p2f/`, không commit vào Git.

## Protocol revision 2

A first technical launch halted before model construction because the inherited helper expected inert OU compatibility parameters in a shared protocol shape. No optimizer step or environment transition occurred. Revision 2 adds explicit metadata for standard SB3 stochastic sampling and those inert compatibility defaults; OU noise remains disabled. Revision 2 supersedes protocol SHA-256 8f0257bc144ee02009753483072eeae8b617f9e7c03faedea12d23f0658042ae. The failed pre-model launch is preserved separately and is not counted as a scientific run.

Revision 3 follows a second failed pre-training launch: the shared P2 helper requires ho and stationary_std as direct exploration keys. The process stopped before constructing a model or taking a transition. Revision 3 exposes these explicitly as inert compatibility fields; standard SAC sampling remains in force and OU noise is disabled. It supersedes protocol SHA-256 3b3beae47f519967bd500cd3550c31b9711f0d4d5307c211f97179c0aff5c96e. Both failed launch directories are preserved and excluded from scientific results.
