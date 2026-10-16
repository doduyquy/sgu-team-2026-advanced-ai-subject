# Lab03 – Tìm kiếm cục bộ (Local Search)

**Môn học:** Trí tuệ nhân tạo nâng cao — SGU Team 2026
**Chủ đề:** Local Search & Optimization Algorithms

## 1. Phân công thực hiện

| Thành viên | Phân công | Nội dung chi tiết |
|---|---|---|
| **Nguyễn Hữu Tri** | Bài 1, 2, 3, 4 | Mê cung (Hill Climbing, Manhattan, BFS), Mô hình hóa n-Queens, 3 biến thể Hill Climbing (Steepest, Stochastic, First-choice), Random Restart & Simulated Annealing độc lập. |
| **Nguyễn Đăng Khoa** | Bài 5, 6, 7, 8 | 4 lịch nhiệt độ SA (Mũ, Tuyến tính, Fast, Logarithmic), Tối ưu hàm mục tiêu với Numba JIT $O(n)$, Traveling Salesman Problem (TSP) toán tử hoán vị swap & SA, Task 6 phân tích hành vi & toán tử lân cận. |

---

## 2. Cấu trúc thư mục Lab03

```text
weekly assignments/lab03 - local search/
├── Maze_Example_LocalSearch.ipynb              # [Bài 1] Tìm kiếm cục bộ trên mê cung (Tri)
├── n_queens.ipynb                              # [Bài 2, 3, 4, 8] n-Queens tích hợp toàn diện:
│                                               #   - Bài 2: Mô hình hóa, đếm xung đột, láng giềng
│                                               #   - Bài 3 (Task 1-3): Steepest, Stochastic, First-choice
│                                               #   - Bài 4 (Task 4-5): Random Restart, SA thuần NumPy
│                                               #   - Bài 8 (Task 6): So sánh 100 lượt, hội tụ, quy mô, toán tử
├── Simulated_Annealing_Schedules.ipynb         # [Bài 5] Khảo sát 4 lịch nhiệt độ SA (Khoa)
├── n_queens_fast_conflict_calculation_with_numba.ipynb # [Bài 6] Tăng tốc đếm xung đột với Numba JIT (Khoa)
├── traveling_salesman_problem.ipynb            # [Bài 7] Bài toán người du lịch TSP (Khoa)
├── L_maze.txt                                  # Dữ liệu mê cung mẫu
├── maze_helper.py                              # Module tiện ích trực quan hóa mê cung
├── n_queens_sa.py                              # Module SA phục vụ khảo sát lịch nhiệt Bài 5
├── requirements.txt                            # Danh sách thư viện phụ thuộc của Lab03
├── test_all_units.py                           # Suite kiểm thử tích hợp tự động Bài 1–8
├── ket_qua_bai3/                               # Dữ liệu CSV thực nghiệm Bài 3
├── ket_qua_bai4/                               # Dữ liệu CSV & hình ảnh thực nghiệm Bài 4
└── notebooks_theo_bai/                         # Các notebook độc lập Bài 2, 3, 4 (bản gốc của Tri)
```

---

## 3. Hướng dẫn cài đặt và tái lập thực nghiệm

### 3.1. Yêu cầu môi trường
- Python 3.10+ (khuyến nghị Python 3.11 – 3.13).
- Cài đặt các gói phụ thuộc:
```bash
pip install -r requirements.txt
```

### 3.2. Kiểm thử tự động nhanh (Smoke & Unit Test)
Chạy bộ kiểm thử tích hợp toàn bộ các hàm từ Bài 1 đến Bài 8:
```bash
python test_all_units.py
```

### 3.3. Thực thi lại toàn bộ Notebook
Mở từng notebook trong Jupyter Lab / VS Code và chọn **Restart Kernel and Run All Cells**, hoặc chạy dòng lệnh:
```bash
python run_notebook.py n_queens.ipynb
```

---

## 4. Các điểm quan trọng về tích hợp & chuẩn hóa

1. **Tính độc lập thuật toán:**
   - Trong `n_queens.ipynb`, toàn bộ các Task 1–5 sử dụng 100% thuật toán do Tri tự cài đặt bằng NumPy thuần túy, không phụ thuộc vào `n_queens_sa.py`.
   - Task 6 (Bài 8) kết nối trực tiếp với các thuật toán của Tri (thông qua adapter chuẩn hóa các trường đầu ra: `best_board`, `best_f`, `evaluations`, `current_history`, `best_history`), bảo đảm tính trung thực khoa học của số liệu so sánh.
2. **Khảo sát toán tử lân cận (Advanced Task):**
   - Hỗ trợ cả toán tử chuyển hàng bất kỳ (`any`: $n(n-1)$ láng giềng) và toán tử dời hàng kề (`adjacent`: $n$ đến $2n$ láng giềng) trực tiếp trên hàm `hill_climb_first_choice` của Tri.
3. **Mô-đun Bài 5 (`n_queens_sa.py`):**
   - Được bảo lưu đầy đủ trong thư mục để phục vụ các cell khảo sát lịch nhiệt trong `Simulated_Annealing_Schedules.ipynb`.
4. **Hạn chế thực nghiệm đã ghi nhận:**
   - Trong Bài 7 (TSP), thực nghiệm Random Restart ở cấu hình $n=40$ có thể tiệm cận giới hạn số bước hoặc cần nhiều thời gian đánh giá hơn so với các kích thước nhỏ; đây là đặc trưng độ phức tạp của bài toán hoán vị kích thước lớn và đã được ghi nhận rõ trong phần nhận xét.
