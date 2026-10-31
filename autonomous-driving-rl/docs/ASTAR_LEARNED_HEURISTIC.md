# Stage 2 A* và Stage 6 A* học heuristic

Mục tiêu của cặp này là đo ảnh hưởng của heuristic học được khi giữ chung thông tin
đầu vào, đồ thị, chi phí cạnh, ngân sách tìm kiếm, bộ điều khiển và action adapter.
Stage 2 không cập nhật trọng số. Stage 6 có một bước huấn luyện riêng trước khi đánh giá.

## Chạy trong Workbench

Từ thư mục `autonomous-driving-rl/`:

```powershell
& '..\.venv\metadrive\python.exe' -B -m src.workbench.app
```

1. Trong **1. Experiment Setup**, bấm **Train heuristic cho A*** hoặc mở tab con
   **Train · A* + heuristic học được**.
2. Chọn **Train thử** rồi bấm **Huấn luyện heuristic**. Worker chạy MetaDrive headless
   để thu thập dữ liệu TRAIN và cập nhật trọng số trên CPU. Biểu đồ hiển thị loss thật
   theo epoch; epoch 0 là trước khi tối ưu.
3. Khi COMPLETE, Workbench chọn sẵn Stage 2 và Stage 6 cùng bộ demo TRAIN.
4. Bấm **Kiểm tra toàn bộ kế hoạch**, rồi **Chạy bộ đánh giá**. Có thể bỏ chọn
   **Hiển thị xe** để chạy headless.
5. Xem metric riêng từng run. Bấm đúp dòng Stage 6 để mở Results; mục Provenance
   chứa training seed và SHA-256 checkpoint đã dùng.

Không cần nhập tên experiment, map hoặc hyperparameter cho từng lượt. Cấu hình chung
được lưu trong `configs/experiments/astar_learning_v1.json`; định nghĩa bộ đánh giá
nằm trong `configs/experiments/workbench_profiles_v1.json`.

`FORCE TERMINATE` dừng worker ngay. Phiên huấn luyện bị kill có thể còn manifest RUNNING;
nó không được công bố như checkpoint COMPLETE. Checkpoint đã chọn trước đó vẫn được giữ.

## Hai thuật toán thực sự làm gì?

`src/agents/planning_graph.py` xây đồ thị cục bộ từ `AgentInputV1` công khai:
10 waypoint đầu của route lookahead và ba vị trí ngang trong làn ở mỗi độ sâu.
Cạnh tiến một độ sâu, kiểm tra các actor lân cận và cộng chi phí độ dài, đổi ngang,
khoảng cách vật cản. Đây là xấp xỉ hình học cục bộ; chưa phải Hybrid A* hay tìm đường
toàn cục có ràng buộc động học và dự đoán chuyển động xe khác.

`src/agents/astar_agent.py` chứa cả hai policy:

| Thành phần | Stage 2 `stage2_astar` | Stage 6 `stage6_astar_learned` |
|---|---|---|
| Input | STATE_DECISION_V1 | STATE_DECISION_V1 |
| Đồ thị và chi phí | Chung cấu hình | Chung cấu hình |
| Heuristic | Khoảng cách Euclidean đến đích cục bộ | Euclidean + residual MLP không âm |
| Tìm kiếm | A*, tối đa 256 node mở rộng | A*, tối đa 256 node mở rộng |
| Điều khiển | Chung path follower | Chung path follower |
| Action | discrete9_lowbranch_v1 | discrete9_lowbranch_v1 |
| Trọng số khi inference | Không có | Cố định từ checkpoint |

Đích cục bộ là waypoint trung tâm cuối đồ thị, không phải một tọa độ đích toàn cục
được lấy từ evaluator. Agent không nhận live simulator, reward, split, case ID,
route completion hoặc cờ kết thúc. Không có reverse Dijkstra trong vòng inference.

## Vòng lặp train

`src/training/astar_heuristic.py` thực hiện:

1. Đọc manifest geometry đã khóa và chỉ chọn **TRAIN**; interleave các tier/family.
2. Chạy Stage 2 trong MetaDrive đúng pin; environment seed dẫn xuất từ training seed
   và geometry. Mỗi 10 bước lấy một đồ thị công khai.
3. Dùng reverse Dijkstra trên đồ thị đó để có nhãn chi phí còn lại chính xác
   **trên mô hình đồ thị cục bộ**. Bỏ node không thể tới đích và node đích.
4. Học residual `(cost_to_goal - Euclidean) / 50` từ 12 feature hình học/traffic
   trong đồ thị. MLP có một hidden layer 32 unit tanh, đầu ra tuyến tính, MSE,
   backpropagation và Adam thực hiện bằng NumPy.
5. Chia fit/holdout theo geometry, không chia ngẫu nhiên từng node. Khoảng 25%
   geometry TRAIN dùng chọn checkpoint. Mean/std được fit trên nhóm fit.
6. Chạy các epoch, ghi loss JSONL, giữ trọng số có MSE holdout nội bộ thấp nhất.
7. Lưu model, hash, thông tin dữ liệu/cấu hình, môi trường, Git và chẩn đoán tìm kiếm.
   Chỉ khi toàn bộ training seed hoàn tất mới cập nhật `latest.json` bằng atomic replace.

Đây là **supervised learning cho heuristic trong search**, không phải PPO/DQN hay
reinforcement learning. Holdout nội bộ vẫn thuộc TRAIN; nó không thay thế bộ
VALIDATION/TEST chính thức và không đo thành công lái xe.

| Preset | Geometry TRAIN | Bước tối đa / geometry / seed | Epoch / model | Training seed |
|---|---:|---:|---:|---|
| smoke | 8, trải bốn tier | 80 | 40 | 101 |
| standard | toàn bộ 180 | 300 | 80 | 101, 202, 303 |

Episode có thể kết thúc trước ngân sách khi xảy ra outcome. Số mẫu thực tế được ghi
trong manifest. Standard huấn luyện ba model độc lập; ba training seed này không phải
ba inference seed của một model. TRAIN collection chỉ dùng behavior policy Stage 2,
nên có thể thiếu các trạng thái mà policy learned sẽ gặp.

## Chạy bằng CLI

```powershell
# Xem geometry và ngân sách, chưa chạy simulator
& '..\.venv\metadrive\python.exe' -B experiments/train_astar_heuristic.py --preset smoke --plan

# Thu thập và train thật
& '..\.venv\metadrive\python.exe' -B experiments/train_astar_heuristic.py --preset smoke

# Ba model độc lập trên toàn bộ TRAIN
& '..\.venv\metadrive\python.exe' -B experiments/train_astar_heuristic.py --preset standard

# Thay <training_manifest> bằng đường dẫn manifest được in khi train xong
& '..\.venv\metadrive\python.exe' -B -m src.launcher.cli run --mode SANDBOX --agent stage6_astar_learned --tier Easy --sequence SCS --geom-seed 0 --env-seed 0 --training-manifest '<training_manifest>' --training-seed 101 --wandb DISABLED
```

## Artifact và cách kiểm chứng

```text
artifacts/astar_learning/train_astar_<id>/
  training_manifest.json
  seed_101/
    dataset.npz
    training_metrics.jsonl
    heuristic.npz
  seed_202/ ...                # standard
  seed_303/ ...                # standard
artifacts/astar_learning/latest.json
runs/<run_id>/                # các lượt inference, qua Gate-7 như cũ
```

NPZ model chỉ chứa tensor số, normalization và metadata JSON, được nạp với
`allow_pickle=False`. Adapter kiểm tra hash, feature schema, cấu hình planner,
training seed, optimizer steps, hai nhóm geometry rời nhau và đều thuộc TRAIN đã khóa.
Checkpoint và data SHA được gắn vào resolved plan và experiment config; file bị sửa
hoặc đổi cấu hình sẽ làm preflight thất bại trước khi logger/simulator khởi tạo.

`CheckpointLaunchRequestV1` và `CheckpointResolvedExperimentPlanV1` là extension
riêng. Field schema của request/plan/snapshot cũ và event enum cũ được giữ nguyên;
training telemetry dùng protocol `@@TRAINING@@` riêng.

## Giới hạn khi diễn giải kết quả

Heuristic Euclidean là cận dưới vì chi phí cạnh không nhỏ hơn độ dài hình học.
MLP chưa có bảo đảm không overestimate; do đó không tuyên bố learned A* giữ tính tối ưu.
Manifest ghi tỷ lệ overestimate, số node mở rộng và chi phí tìm được so với chi phí
chính xác trên một số đồ thị holdout TRAIN. Loss phải được đọc cùng
`euclidean_holdout_residual_mse`: giảm loss so với mạng khởi tạo không có nghĩa hơn
heuristic Euclidean, ít node hơn hoặc lái tốt hơn.

Demo TRAIN có thể dùng geometry đã xuất hiện trong dữ liệu huấn luyện. Hai agent
cùng chạy một case TRAIN chỉ kiểm chứng tích hợp; không dùng chênh lệch route completion
của demo để kết luận tổng quát hóa hoặc thuật toán thắng.

Stage 2 và Stage 6 hiện là **CANDIDATE**, `benchmark_eligible=False`.
VALIDATION/TEST vẫn bị chặn cho tới khi phương pháp/adapter được review và chứng nhận;
không có bước tự bật quyền benchmark. Git sạch, MetaDrive pin, metric và các manifest
đánh giá vẫn giữ quy tắc cũ.

Kiểm tra tích hợp thực sự chạy simulator và worker Qt offscreen:

```powershell
& '..\.venv\metadrive\python.exe' -B -c "from PySide6.QtWidgets import QApplication; import unittest; suite=unittest.defaultTestLoader.discover('tests', pattern='test_astar*.py'); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())"
```

Kiểm tra tạo artifact riêng và không thay checkpoint đang được người dùng chọn.
Lệnh preload Qt trước các module simulator để tránh xung đột DLL trên Windows.
