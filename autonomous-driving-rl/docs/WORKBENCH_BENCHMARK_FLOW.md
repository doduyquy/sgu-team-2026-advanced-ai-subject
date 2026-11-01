# Luồng đánh giá cố định trong Workbench

Bản thử trên nhánh `feat/workbench-fixed-benchmark-flow` đưa việc chọn bộ đánh giá
và thuật toán lên đầu quy trình. Các định nghĩa benchmark Platform V1 không thay đổi.

## Cách dùng

Mở Workbench bằng Python của dự án, từ `autonomous-driving-rl/`:

```powershell
& '..\.venv\metadrive\python.exe' -B -m src.workbench.app
```

Trong tab **1. Experiment Setup**, màn hình mặc định là **Bộ đánh giá cố định**:

1. Chọn VALIDATION, TEST hoặc demo TRAIN.
2. Chọn các thuật toán đã triển khai. Phiên bản, input và action adapter được hiển thị.
3. Xem lịch tự tạo: từng agent, seed, số case và tổng số episode.
4. Bấm **Kiểm tra toàn bộ kế hoạch**. Mỗi lượt đi qua resolver và preflight hiện có.
5. Nếu mọi lượt đạt, bấm **Chạy bộ đánh giá**. Worker chạy tuần tự, không chồng tác vụ.
6. Sau khi toàn bộ lịch hoàn tất, bảng hiển thị metric đã lưu của từng run/replicate.
   Bấm đúp một dòng để xem bằng chứng ở tab Results.

Form cũ nằm trong **Nâng cao · chạy thủ công**, phục vụ phát triển và debug.

Nút **Chọn nhóm tìm kiếm trên TRAIN** chọn sẵn BFS, Dijkstra, A*, Hybrid A* và
case demo cố định. Cả bốn không cần train. Xem [khác biệt giữa các planner](SEARCH_PLANNERS.md)
trước khi diễn giải chi phí đường đi hoặc kết quả lái xe.

## Các bộ chạy

Định nghĩa tập trung tại `configs/experiments/workbench_profiles_v1.json`:

| Bộ chạy | Điều kiện |
|---|---|
| VALIDATION | Toàn bộ 96 case; không chọn map/seed môi trường riêng; render OFF |
| TEST | Toàn bộ 60 case; dùng sau khi đã chốt phương pháp; render OFF |
| Demo TRAIN | Easy/SCS, geometry 0, environment 0; một lượt; có thể hiển thị xe |

Stochastic inference chạy các agent seed `101, 202, 303` trên benchmark; deterministic
inference chạy một lượt. Demo dùng seed `101` khi agent stochastic. W&B tắt trong luồng này.
Mỗi case giữ horizon và điều kiện trong manifest; không áp dụng một horizon tùy ý cho cả bộ.

Stage 6 A* học heuristic có luồng **Train** riêng, chọn checkpoint theo nhóm giữ lại
trong TRAIN. Benchmark yêu cầu đủ ba model train độc lập với training seed 101, 202,
303; mỗi model chạy một lượt inference deterministic. Không dùng ba inference seed
để giả lập ba model. Xem [hướng dẫn A* và learned heuristic](ASTAR_LEARNED_HEURISTIC.md).
Hyperparameter riêng của thuật toán nằm trong cấu hình/phiên bản đã
đăng ký, không có ô sửa trực tiếp trên màn hình benchmark. Nhóm chạy cùng nhau cần cùng
input profile và action adapter; chưa hỗ trợ so sánh có điều kiện giữa các adapter khác nhau.

## Trạng thái triển khai hiện tại

`stage0_random_naive`, `stage2_astar` và `stage6_astar_learned` có thể chạy demo TRAIN
(Stage 6 cần checkpoint đã train). Mặc định chọn Stage 2. Sau Train thử, GUI chọn cặp
Stage 2–6 và demo TRAIN để kiểm tra cùng điều kiện. Các agent hiện đăng ký đều chưa được
bật quyền benchmark, nên VALIDATION/TEST sẽ BLOCKED. Workbench hiển thị nguyên nhân;
không tự thay cờ eligibility, bỏ qua Git sạch hoặc bỏ qua xác minh MetaDrive.

Fixture hạ tầng chỉ hiện trong phần nâng cao, không được trình bày như thuật toán nghiên cứu.

## Kết quả và kiểm tra

Chỉ hiển thị bảng metric sau khi đủ toàn bộ lịch và các run đều COMPLETE / VERIFIED,
đúng agent/version/seed và case. Với benchmark còn kiểm tra canonical provenance,
MetaDrive pin, Git sạch, các contract, thứ tự/các trường manifest và không có lỗi kỹ thuật.
Các giá trị được đọc từ `summary.json`, không tính lại metric chính thức trong GUI.

Bảng giữ riêng từng replicate, không gộp episode thành mẫu huấn luyện độc lập, không
tạo điểm tổng hợp hay tự kết luận thuật toán thắng. Demo luôn mang nhãn không phải benchmark.

Các file khoa học vẫn nằm tại `runs/<run_id>/`. Chỉ mục của lịch chạy được ghi riêng
trong `artifacts/benchmark_campaigns/`; nó liên kết các run và không sửa nội dung của chúng.
PLAN không ghi chỉ mục, không chạy mô phỏng. Lịch dừng khi một RUN thất bại hoặc người
dùng dừng cưỡng bức; những tác vụ còn lại mang trạng thái NOT_RUN.

## Phạm vi bản thử

Đã có lựa chọn nhiều thuật toán, lịch inference cố định, kiểm tra từng lượt, chạy tuần tự,
supervised trainer cho heuristic A*, checkpoint có provenance và xem metric theo run.
Chưa có trainer RL tổng quát, thống kê suy luận hoặc
chứng nhận agent cho benchmark. Tài liệu so sánh khoa học trong `docs/comparison/` vẫn
giữ trạng thái dự thảo; bản thử này không thay cho quyết định khoa học của nhóm.
