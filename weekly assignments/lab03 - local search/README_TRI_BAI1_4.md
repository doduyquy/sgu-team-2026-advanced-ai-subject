# Lab03 – Phần thực hành Bài 1–4 (Nguyễn Hữu Tri)

**Sinh viên thực hiện:** Nguyễn Hữu Tri — **MSSV:** 3123410387
**Môn học:** Trí tuệ nhân tạo nâng cao (SGU).
**Phạm vi gói:** Bài 1–4. **Chưa phải bộ nộp toàn bộ Lab03**: phần Bài 5–8, Task 6 (n-Queens) và báo cáo tổng sẽ do nhóm ghép sau.

## Hai notebook chính dùng để bàn giao/ghép bài

| Notebook | Phạm vi đã hoàn thành | File phụ trợ |
|---|---|---|
| `Maze_Example_LocalSearch.ipynb` | **Bài 1:** Hill Climbing trên mê cung, animation, thay thứ tự hướng, 100 seed, lý giải local optimum và đối chiếu BFS | `L_maze.txt`, `maze_helper.py` cùng thư mục |
| `n_queens.ipynb` | **Bài 2:** biểu diễn n-Queens, đếm xung đột, hàng xóm; **Bài 3:** Task 1–3; **Bài 4:** Task 4–5 (Random Restart, SA do Tri tự viết) | Không import module của thành viên khác |

### Bố cục trong `n_queens.ipynb`

- **Bài 2:** khảo sát `random_board`, `show_board`, `conflicts`, láng giềng trên n=4 và n=8; kiểm thử vét cạn n=4, 1000 board ngẫu nhiên n=8.
- **Task 1:** Steepest-ascent Hill Climbing; **Task 2:** Inefficient Stochastic; **Task 3:** Efficient Stochastic (First-choice); chạy 100 lượt/thuật toán/kích thước.
- **Task 4:** Random Restart cho cả ba thuật toán, không quá 100 khởi tạo mỗi lượt và dừng theo ngân sách.
- **Task 5:** Simulated Annealing độc lập, ghi lịch sử nghiệm hiện tại/tốt nhất và khảo sát lịch nhiệt.
- **Task 6, Advanced Task:** còn nguyên placeholder trong đề gốc; **chưa hoàn thành** vì thuộc phần tiếp theo của nhóm.

## Hướng dẫn chạy lại từ đầu

1. Giải nén trọn vẹn gói; vào `weekly assignments/lab03 - local search/` làm **working directory**.
2. Dùng Python 3.10+ (khuyến nghị 3.11–3.13). Cài thư viện: `python -m pip install -r requirements_bai1_4.txt`.
3. Mở VS Code/Jupyter, chọn kernel Python 3, chạy `Kernel → Restart & Run All` cho hai notebook chính. Không cần Google Drive, đường dẫn riêng của máy hoặc `n_queens_sa.py`.
4. Hai notebook chính có output thực nghiệm đã lưu trong file; khi chạy lại, thời gian có thể thay đổi tùy máy. Các CSV/PNG dùng để đối chiếu được lưu trong `ket_qua_bai3/` và `ket_qua_bai4/`.
5. Nếu chạy toàn bộ các notebook nằm trong `notebooks_theo_bai/`, đặt working directory vẫn ở thư mục `lab03 - local search/` để CSV được xuất đúng vị trí. Các notebook này **chỉ là bản theo từng bài**, không cần nộp trùng nếu đã sử dụng notebook `n_queens.ipynb`.

## Thiết kế thực nghiệm & giới hạn kết luận

- Bài 1: `NESW`, ba thứ tự cố định khác, 100 lần đổi hướng có seed và BFS chỉ làm phép đối chứng để kiểm tra đường đi tồn tại.
- Bài 3: 100 board ban đầu cho mỗi n=4,8 và cho mỗi thuật toán; đo f cuối, tỷ lệ thành công, bước, lần gọi `conflicts()` và thời gian. Dừng First-choice là tiêu chí lấy mẫu **xác suất**, không chứng minh mọi hàng xóm đều không tốt hơn.
- Bài 4: Random Restart tối đa 100 khởi tạo; so sánh SA và ba Restart trên n=4,8 với 40 lượt/cấu hình, cùng ngân sách tối đa 3000 lần đánh giá cho mỗi lượt; khảo sát ảnh hưởng lịch nhiệt qua 30 lượt/cấu hình.
- Kết quả thực nghiệm có thể biến động do seed và phần cứng; không suy luận tính tối ưu tổng quát hoặc thống kê đủ mạnh từ các phép đo hữu hạn.

## Minh chứng và các file tùy chọn

- `ket_qua_bai3/`: 2 CSV — 600 lượt so sánh chính và 300 lượt khảo sát dừng First-choice.
- `ket_qua_bai4/`: 4 CSV — bảng dữ liệu thô và tổng hợp; hình PNG xuất từ notebook.
- `notebooks_theo_bai/`: ba notebook riêng Bài 2, Bài 3, Bài 4 để kiểm tra hoặc giải thích; mã thuật toán đã được tích hợp vào `n_queens.ipynb`.

## Lưu ý ghép vào repository chung

Repository hiện đã có file `n_queens.ipynb` có thể chứa sửa đổi từ thành viên khác. **Không chép đè trực tiếp lên file chung mà không kiểm tra**; hãy lấy Task 1–5 và phần Bài 2 của Tri trong notebook này để tích hợp, giữ các phần khác của nhóm. Tên chuẩn `n_queens.ipynb` ở đây phục vụ cấu trúc đầu ra theo yêu cầu PDF khi nộp chính thức.

**Báo cáo tổng:** chưa làm theo phân công (sẽ ghép khi Bài 5–8 hoàn tất). **Minh bạch công cụ hỗ trợ:** ChatGPT được sử dụng để lập trình/kiểm thử/diễn giải; người nộp cần kiểm tra và hiểu mã nguồn.
