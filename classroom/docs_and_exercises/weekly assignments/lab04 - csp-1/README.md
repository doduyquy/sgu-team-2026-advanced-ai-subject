# Lab 04 — Constraint Satisfaction Problem (CSP)

Bài giải đủ **8 bài trong PDF**: 4 bài tô màu trong ảnh và 4 bài n-Queens bổ sung; hoàn thành Assignment tô màu, Task 1–6 và Advanced task của n-Queens.

## Bài làm

- [Tô màu: notebook đầy đủ](CSP_graph_coloring_example.ipynb) và [HTML đã chạy](CSP_graph_coloring_example.html).
- [n-Queens: notebook đầy đủ](n_queens.ipynb) và [HTML đã chạy](n_queens.html).
- [Báo cáo tổng hợp](BAO_CAO_LAB04.md), [bảng chung hai bài toán](results/lab04_combined_summary.csv).
- Báo cáo chi tiết hoàn chỉnh: [Word](BAO_CAO_LAB04_HOAN_CHINH.docx) và [PDF](BAO_CAO_LAB04_HOAN_CHINH.pdf), gồm 34 trang, 28 bảng và 17 hình; Times New Roman, nội dung 12 pt, tiêu đề 14–16 pt, chữ đen.
- [Số liệu thực nghiệm và cấu hình chạy](results/).
- [Đối chiếu từng yêu cầu PDF và kết quả kiểm tra lại](DOI_CHIEU_YEU_CAU_LAB04.md).

| Yêu cầu | Nội dung đã thực hiện |
| --- | --- |
| Bài 1 | Chạy đúng mẫu 10 đỉnh, seed 1111; tam giác hóa Delaunay; in neighbors; xác định X/D/C; giải thích i < j; khảo sát 3/4 màu. |
| Bài 2 | Phân biệt đầy đủ, nhất quán và nghiệm; kiểm thử phép gán rỗng, từng phần, đầy đủ hợp lệ, đầy đủ xung đột, ngoài miền và ngoài tập biến. |
| Bài 3 | Backtracking gốc; MRV theo miền còn hợp lệ; LCV; đo COUNT, lần thử gán, thời gian; vẽ màu đúng theo khóa biến. |
| Bài 4 | Forward Checking với trail hoàn tác; Min-Conflicts có bước ngang, bước cải thiện và restart; so sánh nhiều đồ thị với 3/4 màu. |
| Assignment bổ sung | Số cạnh trung bình theo n; 5 đồ thị mỗi kích thước 10/20/40/80/160; thử mở rộng 320/640 đỉnh; báo bước cải thiện, restart và chi phí đạt nghiệm. |
| Bài 5 | Mô hình n-Queens, hàm h đếm cặp xung đột, bàn cờ 4/8 hậu và nghiệm [1,3,0,2]; đối chiếu 256 bàn cờ với cách đếm từng cặp. |
| Bài 6 / Task 1–3 | Steepest, Inefficient Stochastic, Efficient Stochastic HC; tiêu chí dừng theo xác suất lấy mẫu; số bước, đánh giá h và tình huống mắc kẹt. |
| Bài 7 / Task 4–5 | Random Restart cho cả 3 HC, tối đa 100 restart; SA với hai lịch geometric/inverse; phân biệt h tốt nhất và trạng thái cuối. |
| Bài 8 / Task 6 | 100 bàn cờ phân biệt mỗi cấu hình tại n=4/8; 8 cấu hình, 1.600 lượt; hội tụ; log-log n=4,8,12,16,20 với 2 thuật toán, 1.000 lượt. |
| Advanced task | 5 toán tử: single, single-step, swap, dual, adaptive; 100 lượt/toán tử tại n=8/12, tổng 1.000 lượt; hội tụ trung bình, phân bố h cuối, thời gian đạt nghiệm. |

Các kiểm chứng thuật toán nằm ngay trong notebook: 640 đối chiếu Backtracking với vét cạn trên mọi đồ thị 4 đỉnh ở 2/3 màu, kiểm tra hoàn tác miền, K4 vô nghiệm với 3 màu và kiểm tra Min-Conflicts trên nhiều seed.

## Chạy lại

Trong thư mục `weekly assignments/lab04 - csp`:

```powershell
python -m pip install -r requirements.txt
```

Mở `CSP_graph_coloring_example.ipynb` bằng Jupyter/VS Code, chọn **Restart Kernel → Run All** rồi lưu notebook. Làm tương tự với `n_queens.ipynb`. Thứ tự tô màu trước rồi n-Queens sẽ cập nhật các bảng CSV và báo cáo tổng hợp. Hai notebook chứa đầy đủ mã, không cần `run_lab.py`.

Để cập nhật hai bản HTML từ notebook đã chạy và lưu:

```powershell
python -m nbconvert --to html CSP_graph_coloring_example.ipynb n_queens.ipynb
```

Ngân sách thí nghiệm chính: 0,4 giây/lần chạy; Backtracking tối đa 30.000 lần thử gán; Min-Conflicts tối đa 3.000 bước mỗi khởi tạo và 4 restart. Phần mở rộng dùng 1 giây/lần chạy. Chi tiết và phiên bản thư viện được lưu trong [results/settings.json](results/settings.json). Thời gian đo và kết quả khi hết ngân sách có thể khác giữa các máy.

`solved` nghĩa là đã kiểm tra được nghiệm; `unsat` chỉ xuất hiện khi Backtracking duyệt hết và chứng minh vô nghiệm; `limit` nghĩa là chưa giải quyết được trong ngân sách. Min-Conflicts không dùng `unsat`. Báo cáo tách chi phí trung bình trên mọi lần chạy và chi phí chỉ trên các lần tìm được nghiệm.

Với n-Queens, các tham số và phiên bản được lưu trong [results/queens_settings.json](results/queens_settings.json). HC chỉ nhận giảm h; first-choice có giới hạn đề xuất và điều kiện dừng theo xác suất; RR tối đa 100 restart; SA tối đa 5.000 đề xuất; phần Advanced tối đa 5.000 đề xuất/lượt. Các thuật toán được tự cài đặt.

**Team: Quý - Phúc - Tri - Khoa**. Công cụ hỗ trợ đã được ghi là OpenAI Codex.

## Đề bài

[TTNTNC — Lab04 — Bài toán thỏa mãn ràng buộc](TTNTNC%20-%20Lab04%20-%20Bai%20toan%20thoa%20man%20rang%20buoc.pdf).
