# Xem BFS / Dijkstra / A* / Hybrid A* trên MetaDrive

Từ thư mục gốc repository:

```powershell
& '.\.venv\metadrive\python.exe' -B -u autonomous-driving-rl/scripts/view_search_planners.py
```

Mặc định mở cửa sổ MetaDrive và chạy lần lượt BFS, Dijkstra, A*, Hybrid A* trên
96 case VALIDATION, sau đó 60 case TEST: tổng lịch 624 episode. Case lấy từ manifest
đã khóa; geometry được kiểm tra hash; seed, traffic, horizon và nhịp điều khiển
10 Hz giữ theo case gốc. Viewer mặc định ưu tiên **mượt**, mục tiêu **60 FPS**:
vẽ cả năm bước vật lý trong mỗi quyết định, thay vì chỉ vẽ ở cuối quyết định.
Tốc độ phát yêu cầu mặc định **5×**; tốc độ thực tế phụ thuộc tải máy và số khung
hình cần vẽ. Nhấn **M** để đổi sang chế độ ưu tiên chạy nhanh, cho phép bỏ qua
khung hình trung gian. Cả hai chế độ vẫn chạy đủ từng quyết định và mọi bước vật lý.

Cấu hình hiển thị tiết kiệm tải: cửa sổ 1280×800, MSAA 2× thay cho mặc định 16×
trên Windows, tắt bóng PSSM và giảm độ phân giải texture terrain. GPU vendor,
renderer và driver được đọc trực tiếp từ OpenGL, ghi vào log và `session.json`.

Đây là **lượt chạy trực quan riêng, không phải benchmark chính thức**. Không đổi
`benchmark_eligible`, không đổi thuật toán và không ghi đè kết quả 624 episode cũ.
`session.json` ghi hash source/config hiện tại và đối chiếu từng file với bản
đóng băng `63b36c8`. Việc giống source planner không tự chứng minh toàn bộ runtime
hoặc mọi số đo giống lượt trước. Không dùng thời gian chạy có render làm số đo
hiệu năng benchmark, hoặc dùng lượt TEST đã xem để coi lần đánh giá sau là holdout mới.

## Trên màn hình

- Góc trên bên phải: thuật toán, split, case, chiến lược tìm kiếm, số node đã duyệt,
  chi phí đường nếu có, điểm bám và lệnh lái/ga/phanh thực tế.
- Phía dưới: năm dòng lịch sử, cập nhật mỗi giây mô phỏng hoặc khi trạng thái
  planner đổi. File log vẫn lưu **đủ từng quyết định** ở 10 Hz.
- Giải thích là bản tóm tắt diagnostics thật; không bịa lý do thất bại hoặc khẳng
  định an toàn khi chỉ có `PATH_FOUND`.
- Điểm bám theo hệ tọa độ xe: `+x` phía trước, `+y` bên trái. Hiển thị steering
  bằng giá trị số thực sự đưa vào MetaDrive.
- Diagnostics/input thuộc thời điểm trước action; tốc độ và tiến độ thuộc sau bước vật lý.
- Dòng cuối bảng hiện chế độ, **FPS thực đo từ số khung hình đã vẽ**, tốc độ phát
  yêu cầu và tốc độ mô phỏng thực tế. FPS và tốc độ mô phỏng là hai số khác nhau.

## Phím điều khiển

| Phím | Chức năng |
|---|---|
| P | Dừng / tiếp tục mô phỏng |
| M | Đổi giữa ưu tiên mượt và ưu tiên chạy nhanh |
| 1 / 2 / 5 | Chọn tốc độ phát 1× / 2× / 5× ngay khi đang chạy |
| 0 | Chạy tối đa theo khả năng máy, không thêm thời gian chờ |
| N | Bỏ qua phần còn lại của case, mở case kế tiếp |
| A | Bỏ qua các case còn lại của thuật toán trong split hiện tại |
| Q | Kết thúc toàn bộ lượt xem và đóng log |
| I | Hiện / ẩn bảng input/output chi tiết |
| Tab | Đổi trang bảng input/output |

Case bị bỏ qua được ghi `SKIPPED`, không được tính là hoàn tất đánh giá.

## Log và kiểm tra nhanh

Mỗi lần mở tạo thư mục riêng `artifacts/search_visual/<timestamp>/`:

- `session.json`: lịch, source/config hashes, tiến độ và trạng thái lượt xem.
- `decisions.log`: giải thích tiếng Việt từng bước và mốc bắt đầu/kết thúc episode.
- `decisions.jsonl`: case/split, input step/time, diagnostics, action, giải thích,
  latency riêng của `act()`, tốc độ và tiến độ sau bước vật lý, tốc độ phát yêu
  cầu/thực tế, FPS, chế độ hiển thị và cờ có vẽ khung hình.
- `episodes.jsonl`: trạng thái và kết quả của từng episode đã chạy.
- `episode_XXX.png`: ảnh native ở bước 30 của mỗi episode.

Chỉ xem BFS trên TEST:

```powershell
& '.\.venv\metadrive\python.exe' -B -u autonomous-driving-rl/scripts/view_search_planners.py --agents bfs --splits TEST
```

Kiểm tra giao diện tám lượt ngắn (bốn thuật toán × hai split):

```powershell
& '.\.venv\metadrive\python.exe' -B -u autonomous-driving-rl/scripts/view_search_planners.py --limit-cases 1 --preview-steps 12 --speed 5 --capture-step 10
```

`--preview-steps` đánh dấu `PREVIEW_LIMIT`; `--smoke-steps N` dừng cả lượt xem sau
N quyết định. Các lượt rút ngắn không được ghi là hoàn thành benchmark.
`--plan-only` chỉ kiểm tra lịch và source, không mở simulator hoặc tạo log.
Để dừng từ script khác, tạo file tên `STOP` trong đúng thư mục log của lượt xem.

## Tiếp tục lượt xem đã dừng

```powershell
& '.\.venv\metadrive\python.exe' -B -u autonomous-driving-rl/scripts/view_search_planners.py --speed 5 --resume-from autonomous-driving-rl/artifacts/search_visual/live-20261119/run
```

Viewer tạo thư mục log mới, ghi nguồn tiến độ trước đó và giữ nguyên số thứ tự
trong lịch 624 episode. Case đã `COMPLETE` hoặc chủ động `SKIPPED` không chạy lại.
Case `INTERRUPTED` cần chạy lại từ đầu case vì chưa lưu snapshot đầy đủ của vật lý;
`PREVIEW_LIMIT` không được coi là hoàn tất. Nếu từng chọn agents/splits/limit riêng,
cần dùng lại đúng các lựa chọn đó. Từ chối resume khi lượt cũ còn RUNNING hoặc
source/config planner và lịch đã thay đổi.

`--view-mode smooth --render-fps 60` ưu tiên hình mượt. Giữ cửa sổ MetaDrive ở
phía trước khi kiểm tra FPS; số đo trong nền có thể thấp hơn tùy hệ thống/driver.
`--view-mode fast --speed 0` ưu tiên xử lý nhanh nhất; chế độ này giảm số khung
hình vẽ, do đó không đảm bảo chuyển động mượt như chế độ smooth.
