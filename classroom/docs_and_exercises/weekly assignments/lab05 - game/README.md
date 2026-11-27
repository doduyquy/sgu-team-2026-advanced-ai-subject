# TTNTNC - LAB 05: TÌM KIẾM ĐỐI KHÁNG VÀ TRÒ CHƠI (ADVERSARIAL SEARCH AND GAMES)

Dự án thực hành môn Trí tuệ Nhân tạo Nâng cao. Nhóm thực hiện đầy đủ 10 bài tập (7 bài Tic-Tac-Toe cơ bản và 3 bài tập mở rộng Connect 4, Mean Connect 4, Dots and Boxes).

---

## 👥 BẢNG PHÂN CHIA NHIỆM VỤ NHÓM

| Thành viên | Bài tập phụ trách | Nội dung thực hiện | Câu hỏi thảo luận |
| :--- | :--- | :--- | :---: |
| **Tùng** | **Bài 1, Bài 2, Bài 9** | - **Bài 1:** Định nghĩa trò chơi, hàm luật, thực nghiệm 1.000 ván Random vs Random, phân tích lợi thế đi trước.<br>- **Bài 2:** AND-OR Tree Search, giải thích conditional plan qua 4 test cases, thực nghiệm 100 ván vs Random.<br>- **Bài 9:** "Mean" Connect 4 (xây dựng môi trường, Minimax/Alpha-Beta, Move Ordering, Heuristic cutoff, PMCS Best First Move). | **Câu 1** |
| **Thái** | **Bài 3, Bài 4, Bài 8** | - **Bài 3:** Minimax đệ quy, Alpha-Beta Pruning, Move Ordering, thực nghiệm đối đầu Random.<br>- **Bài 4:** Heuristic Alpha-Beta với giới hạn độ sâu (cutoff = 2, cutoff = 4).<br>- **Bài 8:** Connect 4 chuẩn (ước tính không gian trạng thái, Random Agent, Minimax/Alpha-Beta bàn nhỏ, Move Ordering, Heuristic cutoff bàn chuẩn 6x7 và Pure Monte Carlo Search). | **Câu 2, 3** |
| **Thi** | **Bài 5, Bài 6, Bài 10** | - **Bài 5:** Pure Monte Carlo Search (PMCS với N = 10, 100, 1000 playouts).<br>- **Bài 6:** MCTS với chính sách UCB1 (`UCT_depth1()`), phân tích cân bằng Exploration vs Exploitation.<br>- **Bài 10:** Dots and Boxes (biểu diễn đồ thị lưới, cài đặt luật chơi, Alpha-Beta, xác định kích thước bàn cờ tối đa). | **Câu 4, 5** |
| **Thắng** | **Bài 7, Báo cáo Word** | - **Bài 7:** Chơi Tic-Tac-Toe tương tác (`tictactoe_interactive.ipynb`), ghi lại biên bản ván đấu mẫu.<br>- **Báo cáo Word:** Biên soạn và hoàn thiện file Báo cáo Word (.docx) tổng hợp toàn bộ kết quả của nhóm theo chuẩn format nộp bài. | **Câu 6** |

---

## 📂 CẤU TRÚC THƯ MỤC

```text
TTNTNC_Lab05_Game/
│
├── README.md                                      # Giới thiệu đồ án, phân công nhiệm vụ, hướng dẫn cách chạy
├── REPORT.md                                      # Báo cáo tổng hợp chi tiết kết quả thực nghiệm
├── CauHoiThaoLuan.md                              # Báo cáo trả lời toàn bộ 6 câu hỏi thảo luận
├── tictactoe.py                                   # Module nền tảng (trạng thái, actions, result, utility, board)
│
├── tictactoe_definitions.ipynb                     # Bài 1: Định nghĩa trò chơi & Thực nghiệm Random 1000 ván (Tùng)
├── tictactoe_and_or_tree_search.ipynb             # Bài 2: AND-OR Tree Search & Kế hoạch điều kiện (Tùng)
├── tictactoe_alpha_beta_tree_search.ipynb         # Bài 3: Minimax & Alpha-Beta Pruning, Move Ordering (Thái)
├── tictactoe_heuristic_alpha_beta_tree_search.ipynb # Bài 4: Heuristic Alpha-Beta với cutoff (Thái)
├── tictactoe_pure_monte_carlo_search.ipynb        # Bài 5: Pure Monte Carlo Search (PMCS) (Thi)
├── tictactoe_monte_carlo_tree_search_restricted.ipynb # Bài 6: MCTS với chính sách UCB1 (Thi)
├── tictactoe_interactive.ipynb                    # Bài 7: Chơi Tic-Tac-Toe tương tác (Thắng)
│
├── assignment_connect4.ipynb                      # Bài 8 (Mở rộng): Connect 4 chuẩn (Thái)
├── assignment_mean_connect4.ipynb                 # Bài 9 (Mở rộng): "Mean" Connect 4 (Tùng)
├── assignment_dots_and_boxes.ipynb                # Bài 10 (Mở rộng): Dots and Boxes (Thi)
├── profiling_code.ipynb                           # Hỗ trợ đo lường hiệu năng
└── store_agent_state_information.ipynb            # Hỗ trợ lưu trữ trạng thái tác tử
```

---

## 🚀 HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY (STEP-BY-STEP)

### Bước 1: Cài đặt thư viện
Mở terminal tại thư mục dự án và cài đặt các thư viện cần thiết:
```bash
pip install numpy matplotlib notebook
```

### Bước 2: Khởi động Jupyter Notebook
Chạy lệnh sau để mở giao diện làm việc:
```bash
jupyter notebook
```
*(Hoặc mở trực tiếp thư mục dự án bằng **VS Code** và sử dụng extension **Jupyter**).*

### Bước 3: Chạy các bài tập
Mở file `.ipynb` tương ứng với phần việc của từng thành viên và chọn **Run All** (hoặc **Kernel** $\rightarrow$ **Restart & Run All**) để thực thi và xem kết quả.

### Bước 4: Xem báo cáo và câu hỏi thảo luận
- Bảng tổng hợp so sánh các thuật toán: xem tại [`REPORT.md`](REPORT.md).
- Đáp án chi tiết 6 câu hỏi thảo luận: xem tại [`CauHoiThaoLuan.md`](CauHoiThaoLuan.md).
