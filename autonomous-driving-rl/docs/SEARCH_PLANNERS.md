# BFS, Dijkstra, A* và Hybrid A* trong Stage 2

Stage 2 hiện có bốn phương pháp chạy qua cùng launcher, metric và Workbench.
**Graph search** là nhóm kỹ thuật tìm kiếm trên đồ thị, không phải thuật toán thứ năm.
Các phương pháp Stage 2 này không cần train; Stage 6 A* học heuristic có trainer riêng.

## Chạy cùng điều kiện

Từ `autonomous-driving-rl/`, mở lại Workbench để nạp các agent mới:

```powershell
& '..\.venv\metadrive\python.exe' -B -m src.workbench.app
```

Trong **1. Experiment Setup → Bộ đánh giá cố định**:

1. Bấm **Chọn nhóm tìm kiếm trên TRAIN · BFS / Dijkstra / A* / Hybrid A***.
   GUI chọn sẵn cả bốn phương pháp và case TRAIN cố định.
2. Chọn có hiển thị xe hay chạy headless.
3. Bấm **Kiểm tra toàn bộ kế hoạch**, rồi **Chạy bộ đánh giá** khi READY.
4. Xem bảng metric từng run. Bấm đúp một dòng để xem Results; Provenance chứa
   cấu hình planner được lưu cùng run và `config_sha256`.

CLI có một file chạy cả nhóm, không cần nhập experiment ID/map/seed từng lượt:

```powershell
# Chỉ xem lịch; chưa chạy simulator
& '..\.venv\metadrive\python.exe' -B experiments/compare_search_planners.py

# Chạy thật cả bốn phương pháp trên TRAIN/Easy/SCS, geometry 0, environment 0
& '..\.venv\metadrive\python.exe' -B experiments/compare_search_planners.py --run
```

Metric chính thức và provenance vẫn ở `runs/<run_id>/`. Bản tóm tắt CLI ở
`artifacts/search_comparison/<id>/KET_QUA.md`; `comparison.json` giữ các request,
preflight, report và episode. Cả nhóm phải COMPLETE và VERIFIED trước khi công bố bảng.
COMPLETE mô tả trạng thái ghi kết quả, không có nghĩa xe đã đến đích.

## Mỗi phương pháp tối ưu điều gì?

| Phương pháp | Agent ID | Không gian tìm kiếm | Thứ tự mở rộng / mục tiêu |
|---|---|---|---|
| BFS | `stage2_bfs` | Đồ thị waypoint/lateral chung | FIFO; ít cạnh nhất, bỏ qua trọng số khi chọn đường |
| Dijkstra | `stage2_dijkstra` | Đồ thị chung | Chi phí tích lũy `g` nhỏ nhất |
| A* | `stage2_astar` | Đồ thị chung | `g + h`, với `h` Euclidean |
| Hybrid A* | `stage2_hybrid_astar` | Pose liên tục `(x, y, heading)`, bin để gộp trạng thái | `g + h` trên cung chuyển động bicycle |

BFS vẫn báo tổng trọng số của đường đã chọn nhưng không tối ưu tổng đó. Trên đồ thị
lattice hiện tại, các đường đi đến goal đều qua cùng số tầng; vì vậy BFS có nhiều
đường đồng hạng và kết quả phụ thuộc thứ tự láng giềng cố định. Dijkstra/A* dùng cùng
chi phí độ dài, đổi ngang và clearance. Euclidean là cận dưới trên đồ thị này vì
chi phí cạnh không nhỏ hơn khoảng cách hình học.

## Hybrid A* thực sự tìm trên chuyển động xe

Planner giữ pose liên tục và sinh các cung từ mô hình bicycle. Key `(bin_x, bin_y,
bin_heading)` chỉ dùng phát hiện trạng thái trùng; pose không bị snap về tâm ô.
Khi một pose rẻ hơn thay thế một bin, chuỗi parent liên tục cũ vẫn được giữ để truy vết.

Các thiết lập riêng nằm trong `configs/experiments/search_planners_v1.json`:

- cung tiến 1,5 m; kiểm tra va chạm mỗi tối đa 0,25 m;
- độ phân giải XY 0,5 m, 72 heading bin;
- năm mức steering nội bộ `0, ±0,3, ±0,6`, tích phân cung giải tích;
- vehicle footprint danh định 4,6 × 1,9 m, wheelbase mô hình 2,5 m;
- goal region 1,2 m và sai lệch heading tối đa 20°;
- kiểm tra bốn góc thân xe trong corridor route và SAT với actor rectangle có hướng;
- chi phí chiều dài cung cộng phạt steering không âm.

Heuristic là khoảng cách đến goal region. Giới hạn mở rộng 256 node lấy từ cấu hình
planner chung. Nếu hết ngân sách hoặc không có đường thì agent phanh; không báo một
đường chưa đến goal là đã tìm thấy đích.

Đây là biến thể **forward-only** cho nền tảng hiện tắt reverse. Không bật số lùi hay
thay action adapter. Đường nội bộ có mức steering trung gian; đầu ra thực tế vẫn qua
bộ bám đường discrete9 chung. Mô hình chưa mô tả trượt lốp, giới hạn tốc độ đổi góc
lái hoặc chuyển động tương lai của actor. Kiểm tra va chạm được lấy mẫu, không phải
chứng minh va chạm liên tục. Chưa có analytic Reeds–Shepp/Dubins expansion hay bước
làm mượt tối ưu như một số triển khai Hybrid A* đầy đủ.

Ý tưởng pose liên tục trong node rời rạc dựa trên
[Dolgov và cộng sự, STAIR 2008](https://ai.stanford.edu/~ddolgov/dolgov08gppSTAIR.html).
Bản trong repo là triển khai cục bộ có giới hạn nêu trên, chưa tái tạo toàn bộ hệ thống của bài báo.

## Phạm vi so sánh và provenance

Cả bốn agent nhận **STATE_DECISION_V1**, xuất **discrete9_lowbranch_v1**, dùng chung
path follower và đặt lại state trước episode. Không nhận live MetaDrive, reward,
case/split ID, route completion hay cờ kết thúc. Không dùng checkpoint cho bốn agent này.

`configs/experiments/astar_learning_v1.json` tiếp tục giữ đồ thị chung, ngân sách và
điều khiển. File đó không bị thay bởi việc thêm Hybrid A*, nên checkpoint Stage 6
đã train vẫn kiểm tra được. Bộ điều khiển được tách sang `src/agents/path_following.py`;
vòng train mới cũng ghi hash file này trong source provenance.

Factory cung cấp một hàm metadata thuần `experiment_parameters()`. Sau preflight,
executor lưu snapshot vào `algorithm_hyperparameters.planner`, gồm phương pháp,
mục tiêu, state space, controller, các tham số và hash. Hook không tạo agent hoặc
simulator. Field schema của các contract Platform V1 được giữ nguyên.

So sánh BFS/Dijkstra/A* là thay cách tìm kiếm trên cùng đồ thị. Hybrid A* còn thay
không gian trạng thái, mục tiêu cục bộ và mô hình chi phí, nên đây là so sánh các
planner hoàn chỉnh. Không so trực tiếp con số `local_path_cost` của Hybrid A* với
chi phí lattice, hoặc gán mọi khác biệt lái xe chỉ cho heuristic.

Các phương pháp hiện là CANDIDATE, SANDBOX/AUDIT eligible và
`benchmark_eligible=False`. Bộ VALIDATION/TEST vẫn giữ điều kiện chứng nhận, Git sạch,
MetaDrive pin và protocol đã khóa. Một case TRAIN chỉ là kiểm chứng tích hợp.

## Kết quả chạy đầy đủ

[Báo cáo 624 episode](../results/search_planners_20261101/README.md) ghi kết quả từ
bản đóng băng riêng đã qua kiểm tra readiness. Cả tám run COMPLETE / VERIFIED,
nhưng cả bốn phương pháp đều chưa đến đích an toàn trên TEST. Source chạy và bản
triển khai hiện tại được phân biệt trong báo cáo; không dùng kết quả TEST để sửa
tham số rồi coi lượt đánh giá tiếp theo là holdout chưa từng xem.
