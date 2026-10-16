# Kết quả kiểm tra trước khi bàn giao phần Bài 1–4

**Mục đích:** xác minh file đã đóng gói, không phải báo cáo tổng môn học.

- Giải nén bộ ZIP vào một thư mục sạch, **không có code Bài 5–8 hoặc code của Khoa**.
- Notebook `Maze_Example_LocalSearch.ipynb`: **12/12 cell code chạy được**, không có exception; 3 hình PNG được sinh.
- Notebook `n_queens.ipynb`: **39/39 cell code chạy được**, không có exception; 14 hình PNG được sinh.
- `nbformat.validate` thông qua với hai notebook chính; `zipfile.testzip` xác nhận CRC lưu trữ.
- Sáu CSV trong `ket_qua_bai3/` và `ket_qua_bai4/` tái tạo **đúng tất cả cột/số dòng và giá trị ngoài thời gian**. Các giá trị runtime khác giữa hai lượt chạy là dự kiến.
- Các Task 1–5 đã có code độc lập trong `n_queens.ipynb`; Task 6 và bonus **vẫn là placeholder của đề** vì không thuộc phần giao của Tri.
- Bài 5–8, notebook TSP và báo cáo tổng của nhóm **không nằm trong gói này**.

### Số dòng dữ liệu thí nghiệm đối chiếu

| File CSV | Số dòng | Giá trị ngoài thời gian |
|---|---:|---|
| `bai3_first_choice_nguong_dung.csv` | 300 | Trùng khớp |
| `bai3_thuc_nghiem_100_luot.csv` | 600 | Trùng khớp |
| `bai4_khao_sat_lich_nhiet.csv` | 360 | Trùng khớp |
| `bai4_so_sanh_restart_sa.csv` | 320 | Trùng khớp |
| `bai4_tong_hop.csv` | 8 | Trùng khớp |
| `bai4_tong_hop_lich_nhiet.csv` | 12 | Trùng khớp |

**Phạm vi kiểm tra:** chạy toàn bộ hai notebook chính bằng Python/Jupyter trong môi trường hiện tại; không phải chứng nhận mọi máy hoặc mọi phiên bản thư viện.
