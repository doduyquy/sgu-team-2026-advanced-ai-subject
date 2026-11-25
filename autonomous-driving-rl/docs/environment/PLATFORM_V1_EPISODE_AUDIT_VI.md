# Kiểm định vòng đời episode, điều kiện kết thúc và giới hạn thời gian của Platform V1

> **Trạng thái:** KIỂM ĐỊNH VÀ KHUYẾN NGHỊ THIẾT KẾ (GATE 3)  
> **Trạng thái đóng băng:** CHƯA ĐÓNG BĂNG — CHỈ LÀ ĐẶC TẢ NGHIÊN CỨU  
> **Phạm vi:** Hợp đồng vòng đời episode dùng chung cho các Stage 0 đến Stage 7  
> **Ngày:** Tháng 9 năm 2026 (đã cập nhật các điều chỉnh về tính toàn vẹn kiểm định Gate 3)

---

## Cách đọc tài liệu

Tài liệu phân biệt rõ ba loại kết luận:

1. **Hành vi đã kiểm chứng từ mã nguồn:** Cơ chế được xác nhận trực tiếp bằng mã nguồn MetaDrive 0.4.3 đang được ghim phiên bản.
2. **Hành vi quan sát bằng thực nghiệm:** Kết quả đo được trong những lần chạy mô phỏng cụ thể.
3. **Chính sách thiết kế của nền tảng:** Quy tắc do Platform V1 đề xuất. Những quy tắc này chưa được đóng băng nên vẫn có thể thay đổi sau nghiên cứu tiếp theo.

`episode` trong tài liệu có nghĩa là một lượt xe chạy từ lúc môi trường được `reset()` cho tới khi lượt chạy kết thúc hoặc bị cắt vì hết ngân sách bước.

---

## 1. Tóm tắt và mục đích

Gate 3 của Research Platform V1 xác lập các quy tắc khoa học dùng chung về **vòng đời episode, khả năng tái lập khi reset, điều kiện kết thúc và giới hạn số bước** cho toàn bộ các hướng nghiên cứu:

- **Stage 0:** Mốc so sánh dùng hành động ngẫu nhiên.
- **Stage 1:** Luật và heuristic.
- **Stage 2:** Lập kế hoạch và tìm kiếm.
- **Stage 3:** Mô phỏng nhìn trước, ví dụ MCTS.
- **Stage 4:** Học bắt chước.
- **Stage 5:** Học tăng cường không mô hình, ví dụ PPO/SAC.
- **Stage 6:** Kết hợp tìm kiếm và học máy.
- **Stage 7:** Học tăng cường dựa trên mô hình, ví dụ world model.

Các yêu cầu chính của Gate 3:

- Chuẩn hóa giao diện episode theo Gymnasium: tách **kết thúc do trạng thái nhiệm vụ** (`terminated=True`) khỏi **bị cắt vì hết ngân sách bước** (`truncated=True`).
- Xây dựng hệ thống nguyên nhân kết thúc có tính tất định và ưu tiên an toàn: va chạm hay ra khỏi đường phải được xếp trước việc tới đích.
- Tính giới hạn bước theo chiều dài tuyến đường để tránh kết luận thất bại sai đối với tuyến dài.
- Bảo đảm `reset()` có thể tái lập và không để trạng thái của episode trước rò sang episode sau.
- Cung cấp các hàm thuần trong `src/platform/`, có thể unit test mà không cần khởi động mô phỏng 3D.
- Chưa thiết kế reward ở Gate 3. Việc đó được chuyển sang Gate 4. `CourseEnvV1` chỉ là baseline thử nghiệm cũ.

---

## 2. Nguồn dữ liệu chuẩn được dùng để kiểm chứng

Nguồn có thẩm quyền là bản MetaDrive cục bộ đã được ghim:

- **Đường dẫn lúc thực hiện kiểm định:** `D:\SGU\CNTT\TTNTNC\metadrive-src`
- **Git commit:** `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- **Phiên bản package:** `0.4.3`
- **Trạng thái Git khi kiểm định:** working tree sạch.
- **Script kiểm định tự động:** `scripts/audit_episode_contract.py`.

Đường dẫn trên là đường dẫn của máy đã tạo báo cáo, không phải yêu cầu mọi máy phải dùng đúng đường dẫn đó. Commit và phiên bản mới là thông tin quan trọng để tái lập.

---

## 3. Ý nghĩa của `reset()` và việc làm sạch trạng thái

### Hành vi đã kiểm chứng từ mã nguồn

Trong `metadrive/envs/base_env.py`, mỗi lần reset thực hiện các bước chính:

1. `_reset_global_seed(seed)` gán seed hiện tại cho môi trường.
2. `engine.reset()` tạo hoặc đổi mạng đường, lấy mẫu tham số giao thông và khởi tạo các agent.
3. `reset_sensors()` xóa dữ liệu còn lại trong bộ đệm depth, LiDAR và camera.
4. Bộ đếm bước, tổng reward và cờ kết thúc được khởi tạo lại:

```python
self.dones = {agent_id: False for agent_id in self.agents.keys()}
self.episode_rewards = defaultdict(float)
self.episode_lengths = defaultdict(int)
```

### Kết quả thực nghiệm

Nhóm ghi lại observation ngay sau `env.reset(seed=42)`, trước khi thực hiện bất kỳ hành động nào. Ở lần chạy đầu, xe được cố ý điều khiển ra khỏi đường tại bước 27. Sau đó, cùng đối tượng môi trường được reset nhiều lần với cùng seed 42.

Kết quả giữa các lần reset:

- Sai khác tuyệt đối lớn nhất của observation: `0.00 × 10^0`.
- Sai khác vị trí: `0.0000000000 m`.
- Sai khác hướng xe: `0.0000000000 rad`.
- Sai khác vận tốc: `0.0000000000 km/h`.
- Các checkpoint của tuyến đường và chuỗi block giống nhau hoàn toàn.
- Sau reset, các cờ đều sạch: `arrive_dest=False`, `out_of_road=False`, `crash=False`.

Kết luận: trong cấu hình đã kiểm tra, trạng thái lỗi của episode trước không rò sang episode sau.

---

## 4. Điều kiện kết thúc trong `done_function()`

Ở mỗi bước mô phỏng, MetaDrive tạo tập thông tin sau:

```python
done_info = {
    TerminationState.CRASH_VEHICLE: vehicle.crash_vehicle,
    TerminationState.CRASH_OBJECT: vehicle.crash_object,
    TerminationState.CRASH_BUILDING: vehicle.crash_building,
    TerminationState.CRASH_HUMAN: vehicle.crash_human,
    TerminationState.CRASH_SIDEWALK: vehicle.crash_sidewalk,
    TerminationState.OUT_OF_ROAD: self._is_out_of_road(vehicle),
    TerminationState.SUCCESS: self._is_arrive_destination(vehicle),
    TerminationState.MAX_STEP: max_step,
    TerminationState.ENV_SEED: self.current_seed,
}
```

Các nguyên nhân trực tiếp:

- `SUCCESS`/`arrive_dest`: xe được xem là tới đích và `done=True`.
- `OUT_OF_ROAD`: kết thúc nếu `out_of_road_done=True`; mặc định là bật.
- `CRASH_VEHICLE`: kết thúc nếu `crash_vehicle_done=True`; mặc định là bật.
- `CRASH_OBJECT`: kết thúc nếu `crash_object_done=True`; mặc định là bật.
- `CRASH_BUILDING`: luôn kết thúc, không phụ thuộc tùy chọn.
- `CRASH_HUMAN`: kết thúc nếu `crash_human_done=True`; mặc định là bật.
- `MAX_STEP`: chỉ đặt `done=True` khi `truncate_as_terminate=True`.

Điểm cần hiểu: `done` nội bộ của simulator và hai giá trị `terminated`/`truncated` trả ra theo Gymnasium có liên quan nhưng không hoàn toàn đồng nghĩa.

---

## 5. Phân biệt `truncated` và `terminated`

MetaDrive xây dựng hai kết quả như sau:

```python
truncateds = {k: step_infos[k].get(TerminationState.MAX_STEP, False) for k in self.agents.keys()}
terminateds = {k: self.dones[k] for k in self.agents.keys()}
```

- `terminated=True`: episode kết thúc do trạng thái bên trong nhiệm vụ, chẳng hạn tới đích, va chạm hoặc ra khỏi đường.
- `truncated=True`: episode bị cắt bởi giới hạn bên ngoài, ở đây là hết số bước cho phép.

Hai thí nghiệm với giới hạn 15 bước:

1. Khi `truncate_as_terminate=False`:

   ```text
   max_step=True, terminated=False, truncated=True
   ```

   Đây là cách biểu diễn đúng theo quy ước Gymnasium.

2. Khi `truncate_as_terminate=True`:

   ```text
   max_step=True, terminated=True, truncated=True
   ```

   MetaDrive cảnh báo rằng thông thường chỉ `truncated` nên là `True` khi hết bước.

Platform V1 chọn **`truncate_as_terminate=False`**. Vì vậy timeout vẫn làm vòng lặp dừng, nhưng được ghi nhận là bị cắt do ngân sách bước, không bị coi là một trạng thái kết thúc tự nhiên của nhiệm vụ.

---

## 6. Xe được xem là tới đích khi nào?

MetaDrive dùng phép chiếu vị trí xe lên làn đích cuối cùng:

```python
long, lat = vehicle.navigation.final_lane.local_coordinates(vehicle.position)
flag = (vehicle.navigation.final_lane.length - 5 < long < vehicle.navigation.final_lane.length + 5) and (
    vehicle.navigation.get_current_lane_width() / 2 >= lat >=
    (0.5 - vehicle.navigation.get_current_lane_num()) * vehicle.navigation.get_current_lane_width()
)
```

Điều kiện không kiểm tra trực tiếp `vehicle.lane == final_lane`. Xe được xem là đến đích nếu:

- Tọa độ dọc `long` nằm trong cửa sổ từ 5 m trước đến 5 m sau cuối làn đích:

  ```text
  final_lane.length - 5 m < long < final_lane.length + 5 m
  ```

- Tọa độ ngang `lat` nằm trong hành lang ngang được phép của đoạn đường cuối.

Việc tới đích không phụ thuộc vào tổng reward hay một ngưỡng reward nào. Reward cao không khiến xe được tính là thành công nếu `arrive_dest` chưa bật.

---

## 7. Ra khỏi đường, chạm vạch liền và va vào vỉa hè

Mã nguồn MetaDrive:

```python
def _is_out_of_road(self, vehicle):
    ret = not vehicle.on_lane
    if self.config["out_of_route_done"]:
        ret = ret or vehicle.out_of_route
    elif self.config["on_continuous_line_done"]:
        ret = ret or vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line or vehicle.crash_sidewalk
    if self.config["on_broken_line_done"]:
        ret = ret or vehicle.on_broken_line
    return ret
```

Ý nghĩa chi tiết:

- Nếu xe không còn nằm trên làn (`not vehicle.on_lane`), xe bị xem là ra khỏi đường.
- Nếu bật `out_of_route_done`, xe đi lệch tuyến định trước cũng có thể bị xem là ra khỏi đường.
- Do cấu trúc `if ... elif`, nhánh kiểm tra vạch liền chỉ chạy khi `out_of_route_done` không bật.
- Nếu bật `on_continuous_line_done`, chạm vạch vàng liền, vạch trắng liền hoặc vỉa hè đều làm `out_of_road=True`.
- Nếu bật `on_broken_line_done`, chạm vạch đứt cũng bị xem là ra khỏi đường.

Các phát hiện được tài liệu nhấn mạnh:

1. `crash_sidewalk` có trong `done_info`, nhưng báo cáo cho rằng phiên bản MetaDrive được kiểm tra không dùng riêng cờ này làm điều kiện trực tiếp trong phần quyết định `done`.
2. Va vỉa hè vẫn có thể kết thúc episode gián tiếp thông qua `_is_out_of_road()` khi `on_continuous_line_done=True`.
3. Chỉ cần xe cắt qua vạch trắng liền bên ngoài, chuỗi sự kiện có thể là:

   ```text
   on_white_continuous_line=True
       → out_of_road=True
       → terminated=True
   ```

Vì vậy hình ảnh có thể cho cảm giác xe vẫn đang chạy trên mặt đường, nhưng simulator đã xem nó vi phạm biên đường và kết thúc episode.

---

## 8. Ý nghĩa các loại va chạm

- `crash_vehicle`: Bullet physics phát hiện tiếp xúc với đối tượng thuộc nhóm xe; episode kết thúc.
- `crash_object`: va vào vật thể giao thông tĩnh như cọc tiêu hoặc rào chắn; episode kết thúc.
- `crash_building`: va công trình; luôn kết thúc.
- `crash_human`: va người đi bộ; episode kết thúc.

Các cờ này mô tả loại vật thể bị va chạm, không mô tả mức độ hư hỏng hay vận tốc va chạm.

---

## 9. Nhiều sự kiện xảy ra cùng lúc và thứ tự ưu tiên an toàn

Ở bước cuối, nhiều cờ có thể cùng bật. Ví dụ xe vừa đi qua vùng đích vừa va vào xe khác. Platform V1 cần chọn một `primary_reason` duy nhất để thống kê.

Thứ tự ưu tiên được đề xuất:

```text
1. CRASH_HUMAN         Va chạm người
2. CRASH_VEHICLE       Va chạm xe khác
3. CRASH_OBJECT        Va chạm vật thể giao thông
4. CRASH_BUILDING      Va chạm công trình
5. CRASH_SIDEWALK      Va vào vỉa hè
6. OUT_OF_ROAD         Rời khỏi vùng được phép chạy
7. SUCCESS             Tới đích an toàn
8. TIMEOUT             Hết ngân sách bước
9. UNKNOWN_TERMINATION Simulator kết thúc nhưng không nhận diện được nguyên nhân
10. UNDETERMINED       Episode vẫn đang chạy
```

Mục đích là ngăn “success hacking”: một agent không được tính thành công chỉ vì chạm vùng đích đồng thời với một tai nạn nghiêm trọng.

Trong `src/platform/episode.py`, `classify_episode_outcome()` dùng `SAFETY_FIRST_PRECEDENCE` và `REASON_TO_RAW_FLAG` làm nguồn quy tắc duy nhất để giảm nguy cơ tài liệu và code lệch nhau.

Thành công sạch được định nghĩa:

```text
clean_success = arrive_dest
                AND không crash_human
                AND không crash_vehicle
                AND không crash_object
                AND không crash_building
                AND không crash_sidewalk
                AND không out_of_road
```

Nếu xe tới đích đồng thời va xe hoặc vỉa hè, nguyên nhân chính sẽ là lỗi va chạm và `clean_success=False`. Dù vậy, tất cả cờ gốc vẫn được giữ trong `EpisodeOutcome.raw_flags` để phân tích sau này.

---

## 10. Chuẩn hóa kết quả episode và phần code triển khai

`src/platform/episode.py` cung cấp:

- `TerminalReason`: enum chứa các nguyên nhân kết quả đã chuẩn hóa.
- `EpisodeOutcome`: dataclass bất biến chứa `terminated`, `truncated`, `primary_reason`, `clean_success` và `raw_flags`.
- `compute_route_aware_horizon()`: hàm tính số bước tối đa theo chiều dài tuyến, có kiểm tra dữ liệu đầu vào. Hàm báo `ValueError` nếu chiều dài âm, tốc độ không dương hoặc giới hạn không hợp lệ.

Có 10 unit test thuần trong `tests/test_episode_lifecycle.py`. Chúng chạy dưới 0,01 giây và không cần Panda3D.

---

## 11. Hiệu chỉnh giới hạn bước trên 12 kịch bản MapSuite

Nhóm chạy 12 bản đồ chính và thay thế với mật độ giao thông benchmark thật, dùng `TrafficMode.Trigger` và dựng lại đúng hình học từ `MapGenerateMethod.PG_MAP_FILE`.

### Kết quả có giao thông benchmark

| Mức | Vai trò | Chuỗi | Seed | Dài tuyến | Mật độ | Xe dự kiến | Xe hoạt động TB | Đỉnh | Kết quả IDM tham chiếu | Horizon đề xuất | Thời gian |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Easy | Chính | `SCS` | 11 | 349,6 m | 0,00 | 0 | 0,00 | 0 | Thành công ở bước 407 | 1049 | 104,9 s |
| Easy | Thay thế | `SCSS` | 9 | 403,1 m | 0,00 | 0 | 0,00 | 0 | Thành công ở bước 478 | 1210 | 121,0 s |
| Easy | Thay thế | `SCCS` | 9 | 444,1 m | 0,00 | 0 | 0,00 | 0 | Thành công ở bước 533 | 1333 | 133,3 s |
| Medium | Chính | `SCXCS` | 11 | 523,8 m | 0,08 | 9 | 4,51 | 8 | Ra khỏi đường ở bước 383 | 1572 | 157,2 s |
| Medium | Thay thế | `SCTCS` | 0 | 521,1 m | 0,08 | 7 | 3,57 | 5 | Thành công ở bước 629 | 1564 | 156,4 s |
| Medium | Thay thế | `SCXCCS` | 13 | 688,4 m | 0,08 | 12 | 5,62 | 9 | Thành công ở bước 844 | 2066 | 206,6 s |
| Hard | Chính | `SCXOCS` | 2 | 643,0 m | 0,15 | 27 | 5,56 | 9 | Va xe ở bước 249 | 1930 | 193,0 s |
| Hard | Thay thế | `SCTXrCS` | 1 | 748,8 m | 0,15 | 28 | 12,76 | 23 | Va xe ở bước 780 | 2247 | 224,7 s |
| Hard | Thay thế | `XTOCS` | 19 | 496,9 m | 0,15 | 17 | 7,35 | 12 | Thành công ở bước 678 | 1491 | 149,1 s |
| Extreme | Chính | `CrXROSTR` | 6 | 938,6 m | 0,25 | 70 | 16,42 | 26 | Va xe ở bước 563 | 2816 | 281,6 s |
| Extreme | Thay thế | `SCXOCrTYCS` | 16 | 1051,8 m | 0,25 | 79 | 20,20 | 35 | Va xe ở bước 618 | 3156 | 315,6 s |
| Extreme | Thay thế | `SCTXORyCCS` | 4 | 1003,2 m | 0,25 | 58 | 15,09 | 19 | Va xe ở bước 405 | 3010 | 301,0 s |

`Planned Traffic` là tổng xe được lên kế hoạch cho cả bản đồ. `Active Mean` và `Active Max` là số xe cùng hoạt động trung bình và lớn nhất tại một thời điểm. Vì chế độ Trigger kích hoạt xe theo khu vực, không phải toàn bộ xe dự kiến đều xuất hiện đồng thời.

### Kết quả tham chiếu khi không có giao thông

| Mức | Bản đồ | Dài tuyến | Kết quả | Số bước | Thời gian | Tốc độ TB IDM | Nếu chỉ có 1000 bước |
|---|---|---:|---|---:|---:|---:|---|
| Easy | `SCS` | 349,6 m | Thành công | 407 | 40,7 s | 29,0 km/h | Đủ thời gian |
| Easy | `SCSS` | 403,1 m | Thành công | 478 | 47,8 s | 29,1 km/h | Đủ thời gian |
| Easy | `SCCS` | 444,1 m | Thành công | 533 | 53,3 s | 29,2 km/h | Đủ thời gian |
| Medium | `SCXCS` | 523,8 m | Ra khỏi đường | 383 | 38,3 s | 28,8 km/h | Lỗi ở bước 383 |
| Medium | `SCTCS` | 521,1 m | Thành công | 629 | 62,9 s | 29,3 km/h | Đủ thời gian |
| Medium | `SCXCCS` | 688,4 m | Thành công | 830 | 83,0 s | 29,5 km/h | Đủ thời gian |
| Hard | `SCXOCS` | 643,0 m | Thành công | 779 | 77,9 s | 29,4 km/h | Đủ thời gian |
| Hard | `SCTXrCS` | 748,8 m | Thành công | 897 | 89,7 s | 29,5 km/h | Đủ thời gian |
| Hard | `XTOCS` | 496,9 m | Thành công | 599 | 59,9 s | 29,2 km/h | Đủ thời gian |
| Extreme | `CrXROSTR` | 938,6 m | Thành công | 1123 | 112,3 s | 29,4 km/h | Bị timeout trước đích |
| Extreme | `SCXOCrTYCS` | 1051,8 m | Thành công | 1255 | 125,5 s | 29,6 km/h | Bị timeout trước đích |
| Extreme | `SCTXORyCCS` | 1003,2 m | Thành công | 1218 | 121,8 s | 29,6 km/h | Bị timeout trước đích |

Phân tích:

- Trong `TrafficMode.Trigger`, các xe giao thông được kích hoạt theo từng đợt khi ego tiến gần khu vực kích hoạt.
- Ở Hard và Extreme, trung bình có khoảng 5,6–20,2 xe hoạt động đồng thời; đỉnh lên tới 35 xe.
- `IDMPolicy` gặp khó khăn khi bám làn qua góc cua gắt và chỗ nhập làn đông xe. Nó có thể quệt lề hoặc va giao thông, vì vậy không thể xem IDM là chuyên gia mạnh ở các bản đồ phức tạp.
- Khi bỏ giao thông, IDM tới đích ở cả ba tuyến Extreme sau 1123, 1255 và 1218 bước.

Kết luận quan trọng: giới hạn cố định 1000 bước, tương đương 100 giây, sẽ cắt ba tuyến Extreme trước khi tới đích ngay cả khi xe chạy sạch với tốc độ khoảng 29,5 km/h. Vì vậy giới hạn phải phụ thuộc chiều dài tuyến.

---

## 12. Chính sách horizon theo chiều dài tuyến

`horizon` là số bước tối đa của một episode. Công thức được chọn:

```text
horizon_steps = max(
    min_steps,
    min(
        max_steps,
        ceil(route_length_m / floor_speed_mps × safety_margin × control_frequency)
    )
)
```

Các tham số:

- Tốc độ sàn tham chiếu: `18 km/h = 5 m/s`, cho phép xe chạy thận trọng qua vòng xoay và giao lộ.
- Hệ số an toàn: `1,5`.
- Tần số điều khiển: `10 Hz`.
- Giới hạn dưới: `1000 bước = 100 giây`.
- Giới hạn trên: `4000 bước = 400 giây`.

Với các giá trị này, công thức rút gọn thành:

```text
horizon_steps = max(1000, ceil(route_length_m × 3.0))
```

Code thực tế còn chặn kết quả ở tối đa 4000 bước, nên biểu diễn đầy đủ phải hiểu là:

```text
horizon_steps = max(1000, min(4000, ceil(route_length_m × 3.0)))
```

Ví dụ:

- Tuyến 200 m: tính được 600 bước nhưng được nâng lên tối thiểu 1000 bước.
- Tuyến 500 m: 1500 bước, tương đương 150 giây mô phỏng.
- Tuyến 1000 m: 3000 bước, tương đương 300 giây.
- Tuyến quá dài tạo kết quả trên 4000 sẽ bị giới hạn ở 4000 bước.

Lý do lựa chọn:

1. Horizon là ngưỡng an toàn khẩn cấp để tránh episode chạy vô hạn, không phải yêu cầu xe phải duy trì tốc độ cao.
2. Với các lượt tham chiếu thành công, công thức cho thời gian dư khoảng 2,20–2,58 lần thời gian IDM cần để hoàn thành.
3. Ở bản đồ đông xe nơi IDM đã thất bại, không thể tính biên thời gian hoàn thành đáng tin cậy nên báo cáo ghi `N/A`.
4. Chính sách chưa đóng băng; cần được xác minh lại bằng agent học máy và agent lập kế hoạch trong tương lai.

---

## 13. Vòng đời giao thông: `Trigger` và `Respawn`

### Từ mã nguồn

- `TrafficMode.Trigger`: xe giao thông được cấp phát trước trên các block đường và chỉ được kích hoạt một lần khi ego đi vào vùng kích hoạt.
- `TrafficMode.Respawn`: xe đã tới đích được thay liên tục bằng xe mới.

### Từ thực nghiệm

Hai lần chạy độc lập, mỗi lần 60 bước với seed 42 cho kết quả:

- Đều lập kế hoạch 9 xe.
- Chuỗi số lượng xe hoạt động theo từng bước giống nhau.
- Vị trí xe giao thông đầu tiên giống nhau chính xác.
- Với `Respawn`, số xe hoạt động được duy trì liên tục gần mức cực đại.

Kết luận: `Trigger` tạo một quần thể giao thông hữu hạn, được lập kế hoạch trước và có khả năng tái lập với cùng seed và cùng policy. Dấu vết máy đọc được lưu tại `results/audits/episode/traffic_lifecycle.csv`.

---

## 14. Khả năng tái lập và không rò trạng thái

Thử nghiệm reset liên tiếp với seed 42 sau khi episode đầu ra khỏi đường ở bước 27 cho thấy:

- Vị trí ban đầu khớp chính xác.
- Hướng ban đầu khớp chính xác.
- Vận tốc ban đầu khớp chính xác.
- Observation ban đầu không có sai khác.
- Các checkpoint tuyến đường khớp.
- Chuỗi block đều là `ISCXCS`.
- Cả hai lần đều lập kế hoạch 9 xe giao thông.
- Các cờ `arrive_dest`, `out_of_road` và `crash` trở về `False`.

Kết luận của phạm vi thử nghiệm: không phát hiện rò rỉ trạng thái giữa các episode sau một lỗi kết thúc.

---

## 15. Ghi chú lộ trình cho Research Launcher

Một giao diện Research Launcher trong tương lai có thể cung cấp:

- Chọn độ khó Easy, Medium, Hard hoặc Extreme.
- Chọn bản đồ chính hoặc bản đồ thay thế.
- Chọn agent từ Stage 0 đến Stage 7.
- Hiển thị động horizon theo chiều dài tuyến.
- Chọn chạy không giao diện hoặc render 3D tương tác.

Đây chỉ là ghi chú lộ trình. Gate 3 không triển khai phụ thuộc GUI như PySide6.

---

## 16. Rủi ro và vấn đề chưa giải quyết

1. **IDM gặp nút thắt trong giao thông dày:** Ở mật độ 0,15–0,25, xe IDM có thể lấn mép làn khi nhiều xe tập trung ở lối vào giao lộ. Điều này cho thấy cần thuật toán lập kế hoạch từ Stage 2 trở lên để thương lượng và chuyển làn tốt hơn.
2. **Va chạm người đi bộ:** `crash_human` đã có trong taxonomy và classifier, nhưng MapSuite chuẩn đặt `accident_prob=0.0`. Trường hợp này cần được stress test trong benchmark độ bền sau này.
3. **Reward chưa được căn chỉnh:** Gate 3 không giải quyết reward shaping; toàn bộ nội dung này được chuyển sang Gate 4.

---

## 17. Đề xuất `EpisodeSpecV1` chưa đóng băng

```json
{
  "spec_version": "1.0.0",
  "control_frequency_hz": 10,
  "truncate_as_terminate": false,
  "traffic_mode": "trigger",
  "horizon_policy_type": "route_aware",
  "reference_floor_speed_kmh": 18.0,
  "safety_margin": 1.5,
  "min_horizon_steps": 1000,
  "max_horizon_steps": 4000
}
```

Diễn giải: simulator nhận quyết định 10 lần mỗi giây; timeout được ghi là truncation; giao thông dùng chế độ Trigger; giới hạn bước được tính theo chiều dài tuyến, tốc độ sàn 18 km/h và hệ số an toàn 1,5, nhưng không thấp hơn 1000 và không cao hơn 4000 bước.

---

## 18. Reward thuộc Gate 4

Thiết kế lại reward, hình phạt, thưởng tiến độ và reward giữ xe theo phương ngang không thuộc phạm vi Gate 3. Các nội dung này được dành riêng cho Gate 4 và `CourseEnvV1` không bị thay đổi bởi báo cáo này.

---

## Tóm tắt hành vi thực tế quan trọng nhất

Trong luồng chạy Platform V1 hiện tại, một episode tiếp tục khi cả `terminated=False` và `truncated=False`. Episode dừng trong các tình huống sau:

- Xe tới vùng đích hợp lệ: `SUCCESS`.
- Xe va chạm, ra khỏi đường, chạm biên đường bị cấm hoặc gặp một lỗi an toàn khác: nguyên nhân `CRASH_*` hoặc `OUT_OF_ROAD`.
- Xe chưa tới đích nhưng dùng hết horizon: `TIMEOUT`, biểu diễn bằng `truncated=True`.
- Simulator tự kết thúc nhưng không có cờ đã biết: `UNKNOWN_TERMINATION`.

Khi nhiều sự kiện xảy ra cùng lúc, lỗi an toàn được chọn trước thành công. Vì vậy tới đích đồng thời với va chạm không được tính là `clean_success`.
