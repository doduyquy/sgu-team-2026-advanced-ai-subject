# Kết quả đánh giá nhóm tìm kiếm Stage 2

Đã hoàn tất 624 episode: bốn phương pháp × (96 VALIDATION + 60 TEST).
Cả tám run COMPLETE / VERIFIED, không có lỗi kỹ thuật. TEST chỉ chạy sau khi toàn bộ VALIDATION hoàn tất; không chỉnh phương pháp giữa hai bộ.

| Bộ | Phương pháp | Episode | Đến đích an toàn | Lỗi an toàn | Mean route | Median route |
|---|---|---:|---:|---:|---:|---:|
| VALIDATION | A* | 96 | 0.00% | 100.00% | 4.85% | 4.36% |
| VALIDATION | BFS | 96 | 8.33% | 11.46% | 76.32% | 94.26% |
| VALIDATION | Dijkstra | 96 | 0.00% | 100.00% | 4.85% | 4.36% |
| VALIDATION | Hybrid A* | 96 | 0.00% | 0.00% | 0.93% | 0.84% |
| TEST | A* | 60 | 0.00% | 100.00% | 22.87% | 17.09% |
| TEST | BFS | 60 | 0.00% | 6.67% | 78.67% | 94.31% |
| TEST | Dijkstra | 60 | 0.00% | 100.00% | 22.87% | 17.09% |
| TEST | Hybrid A* | 60 | 0.00% | 25.00% | 32.30% | 19.32% |

Cả bốn phương pháp đều chưa có lần đến đích an toàn trên TEST trong lượt đánh giá này.

Các chênh lệch trong bảng mô tả bốn bản triển khai này, không chứng minh một thuật toán luôn tốt hơn thuật toán khác.

## Phạm vi của kết quả

Đây là kết quả của các bản triển khai cục bộ và bộ điều khiển hiện tại. A*/Dijkstra cho cùng metric; cần điều tra đồ thị, chọn đường và bộ bám đường trong nghiên cứu tiếp theo.
Hybrid A* dùng không gian trạng thái và chi phí riêng. Mỗi phương pháp deterministic chỉ chạy một lượt trên mỗi bộ; không suy diễn ý nghĩa thống kê từ các lượt này.

Nguồn thực thi đóng băng: commit `63b36c851f7a4bb2c246705573ba5268c61fa8b1`, tree `1d5393bc90c3726276854d7f0ee23758d0b3de86`.
Commit chạy này nằm trên nhánh local `run/search-full-20261101`; không phải ref đã được công bố trên remote.
Nhánh triển khai hiện tại tích hợp các cập nhật HUD mới hơn và vẫn giữ trạng thái candidate. Kết quả này không được trình bày như một lần chạy lại trên source mới.

## Bằng chứng và kiểm tra lại

[metrics.json](metrics.json) giữ metric lấy trực tiếp từ summary Gate-7, Run ID, cấu hình từng planner, contract hash, source hash và SHA-256 của các artifact đã xác minh.
Raw runs/log và đường dẫn máy cá nhân không nằm trong bản công bố. Các bản gốc vẫn giữ tại `runs/<run_id>/` của workspace dùng chạy thí nghiệm.
Từ `autonomous-driving-rl/`, dùng script xuất lại từ bản gốc vào thư mục trống:

```powershell
& '..\.venv\metadrive\python.exe' -B scripts/export_search_results.py --campaign artifacts/search_full_20261101/progress.json --runs-root runs --output artifacts/search_report_check
```

Script kiểm tra đủ cả tám run, integrity, source/provenance, thứ tự/số lượng case và việc VALIDATION hoàn tất trước TEST; không chạy simulator hoặc tính lại metric chính thức.
