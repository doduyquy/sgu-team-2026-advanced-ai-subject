# Kết quả và Log Đánh giá 5 Thuật toán (21/11/2026)

Thư mục này đóng gói log, episode CSV và bảng kết quả TEST của 5 thuật toán chính (kèm baseline Random). Nguồn kết quả gồm hai đợt chạy khác nhau:

- **Dijkstra, A*, BFS, Hybrid A*:** bản lưu của các run `fullsearch_test_*_20261101`, agent phiên bản **0.1.0**, mỗi thuật toán 60 episode. Đây là kết quả lịch sử, không phải kết quả chạy lại controller Stage 2 phiên bản 0.2.1 trong mã nguồn hiện tại.
- **MPPI và Random:** lượt chạy ngày **21/11/2026**, mỗi agent 180 episode trên 60 case TEST với ba seed 101/202/303. `candidate_test_session.json` ghi `kind=UNCERTIFIED_CANDIDATE_TEST` và `canonical_run=false`; đây là đánh giá candidate, chưa phải run benchmark được chứng nhận.

Bảng bên dưới dùng để tra cứu các kết quả đã lưu. Khác biệt phiên bản và quy trình chạy không cho phép coi bảng này là phép so sánh đồng điều kiện của các agent hiện tại. Giữ nguyên manifest, log và số liệu gốc để kiểm tra xuất xứ.

## 1. Cấu trúc thư mục

* **`01_Dijkstra/`**:
  - `episodes.csv`: Kết quả chi tiết 60 episode TEST của Dijkstra.
  - `summary.json`, `timing.csv`, `run_manifest.json`.
* **`02_A_Star/`**:
  - `episodes.csv`: Kết quả chi tiết 60 episode TEST của A*.
  - `summary.json`, `timing.csv`, `run_manifest.json`.
* **`03_BFS/`**:
  - `episodes.csv`: Kết quả chi tiết 60 episode TEST của BFS.
  - `summary.json`, `timing.csv`, `run_manifest.json`.
* **`04_Hybrid_A_Star/`**:
  - `episodes.csv`: Kết quả chi tiết 60 episode TEST của Hybrid A*.
  - `summary.json`, `timing.csv`, `run_manifest.json`.
* **`05_MPPI/`**:
  - `episodes_seed101_60.csv`: 60 episode TEST chuẩn của MPPI (Seed 101).
  - `episodes_all_180.csv`: Toàn bộ 180 episode MPPI (3 seed: 101, 202, 303).
  - `runtime_seed101.log`, `runtime_seed202.log`, `runtime_seed303.log`: Nhật ký chạy chi tiết.
  - `episodes_seed*.jsonl`.
* **`06_Random_Naive/`** (Baseline đối chứng):
  - `episodes_seed101_60.csv` và `episodes_all_180.csv`.
  - Toàn bộ log và jsonl của 3 seed.
* **`ALL_5_ALGORITHMS_COMPARISON.csv`**: Bảng tổng hợp so sánh các chỉ số để mở trực tiếp trên Excel.

---

## 2. Bảng tổng hợp kết quả tập TEST

| Thuật toán | Số episode TEST | Clean Success Rate | Safety Failure Rate | Mean Route Completion | Median Route Completion |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dijkstra** | 60 | **0.00%** (0/60) | **100.00%** (60/60) | 22.87% | 17.09% |
| **A\*** | 60 | **0.00%** (0/60) | **100.00%** (60/60) | 22.87% | 17.09% |
| **BFS** | 60 | **0.00%** (0/60) | **6.67%** (4/60) | 78.67% | 94.31% |
| **Hybrid A\*** | 60 | **0.00%** (0/60) | **25.00%** (15/60) | 32.30% | 19.32% |
| **MPPI (Seed 101)** | 60 | **33.33%** (20/60) | **63.33%** (38/60) | 44.70% | 32.01% |
| **MPPI (3 Seed gộp)** | 180 | **48.33%** (87/180) | **48.33%** (87/180) | 62.96% | 73.88% |
| **Random (Seed 101)** | 60 | **0.00%** (0/60) | **100.00%** (60/60) | 15.30% | 14.27% |
| **Random (3 Seed gộp)** | 180 | **0.00%** (0/180) | **100.00%** (180/180) | 17.45% | 16.24% |
