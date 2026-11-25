# Thiết kế thông tin vạch đường và agent giữ làn

## 1. Mục tiêu và trạng thái

Mục tiêu là cho agent biết loại vạch, màu vạch và hình dạng biên làn phía trước,
để giữ toàn bộ thân xe trong hành lang di chuyển khi đi thẳng và vào cua.
Giai đoạn đầu chỉ xét chạy trong làn, không vượt xe hoặc chủ động chuyển làn.

Đây là thiết kế đề xuất cho extension V2, chưa tích hợp vào launcher.
Agent V1 phiên bản 0.2.0 đã được cải thiện và kiểm tra riêng về giữ làn/qua cua;
điều đó không có nghĩa là các chức năng phân biệt vạch V2 đã được triển khai.
Các giá trị thử nghiệm bên dưới không phải tham số đã tối ưu.

Chọn phương án **thông tin hình học cục bộ từ simulator** cho bản đầu tiên.
Đó là đầu vào trạng thái có nhãn từ môi trường, không phải nhận diện màu từ ảnh
camera và không cần training. Không mô tả kết quả của phương án này như một
hệ thống thị giác máy tính. Nếu nghiên cứu nhận diện từ ảnh sau này, cần một
profile camera riêng và đánh giá lỗi nhận diện của profile đó.

## 2. Những gì đã kiểm tra trong code hiện tại

- `Stage1RuleBasedAgent` bản 0.1.0 chỉ đọc waypoint và dùng ga cố định.
  Bản 0.2.0 đã dùng độ lệch tâm làn, hướng làn, tốc độ và khoảng cách biên
  từ observation V1; vẫn chưa có thông tin màu/loại vạch.
- MetaDrive mặc định có `on_continuous_line_done=True`,
  `out_of_route_done=False`, `out_of_road_done=True`. Vì vậy chạm vạch liền vàng
  hoặc trắng có thể kết thúc episode dưới nhãn `OUT_OF_ROAD`.
- Làn PG có `line_types` và `line_colors`. Trong `pg_block.py`, phần dựng vạch
  ghép chúng với hai vị trí ngang `-width/2` và `+width/2`.
- Trong `StraightLane`, tọa độ lateral dương hướng sang phải của làn, còn
  tọa độ ego mà `agent_input.py` xuất có +y hướng trái. Phải chuyển đổi dấu
  rõ ràng, không dùng chung hai quy ước như thể chúng giống nhau.
- `PGLineType.CONTINUOUS` có tên giá trị gắn với trắng nhưng màu được lưu
  riêng ở `line_colors`. Không suy ra màu vàng/trắng chỉ từ `line_types`.
- `LaneLineDetector` hiện xuất các khoảng cách; vector input hiện tại không
  chứa nhãn màu vạch. Chỉ bật detector không tự tạo khả năng phân biệt màu.
- `extract_task_context()` hiện chọn `lanes[0]` cho các đoạn tuyến, trong khi
  tính tọa độ dọc từ `navigation.current_lane`. Khi các làn khác nhau, waypoint
  có thể không thuộc tâm làn xe cần giữ; cần kiểm tra tính nhất quán này.
- Dữ liệu audit `control_response.csv` ghi action +0.25 làm heading tăng
  khoảng +4.58 độ; -0.25 làm heading giảm khoảng -4.58 độ. Tài liệu contract
  lại gắn nhãn âm là trái, dương là phải. Phép thử runtime với +0.25 trong
  15 bước đã xác nhận heading tăng +0.07997 rad. Với hệ ego +y trái, bản 0.2.0
  dùng steering dương để quay trái và đã kiểm tra sai số bám làn giảm.
  Đây là sai khác trong phần mô tả cũ, không phải thay đổi action adapter.

Nguồn trong repo:

- `src/agents/rule_based_agent.py`
- `src/platform/agent_input.py`
- `src/launcher/executor.py:build_metadrive_case_config`
- `results/audits/observation_action/control_response.csv`
- `docs/AGENT_DEVELOPMENT_GUIDE.md`, phần mở rộng thông tin công khai.

Nguồn MetaDrive local được kiểm tra:

- `metadrive/constants.py`: `PGLineType`, `PGLineColor`.
- `metadrive/component/lane/straight_lane.py`, `pg_lane.py`.
- `metadrive/component/pgblock/pg_block.py`: dựng vạch hai bên làn.
- `metadrive/component/sensors/distance_detector.py`.
- `metadrive/obs/state_obs.py`, `metadrive/envs/metadrive_env.py`.

## 3. Đầu vào công khai đề xuất

Không nhét dữ liệu mới vào `AgentInputV1` hoặc đổi ý nghĩa vector 259D.
Đề xuất profile mới `STATE_LANE_V2` và wrapper `AgentInputV2`:

```text
AgentInputV2
  base_input: AgentInputV1
  lane_context: LaneContextV2
  ego_motion: EgoMotionV2
```

Các tên V2 ở đây là đề xuất, chưa có trong registry hoặc contract hiện tại.

| Trường | Kiểu / đơn vị | Mục đích |
|---|---|---|
| `lane_context.valid` | bool | Có đủ thông tin để giữ làn hay không |
| `centerline_ego` | tuple các điểm (x, y), mét | Tâm làn đang được chọn, có đoạn nhìn trước |
| `boundaries` | tuple các biên có giới hạn số lượng | Biên trái/phải của làn và đoạn tiếp nối |
| `lane_width_m` | mét | Bề rộng làn tại vị trí xe |
| `center_offset_left_m` | mét, dương = lệch trái | Độ lệch tâm xe khỏi tâm làn |
| `lane_heading_error_rad` | radian | Hướng làn trừ hướng xe, wrap về [-pi, pi] |
| `ego_motion.speed_mps` | m/s | Điều khiển tốc độ, không cần đoán hệ số chuẩn hóa |
| `ego_motion.yaw_rate_rad_s` | rad/s có dấu | Ước lượng chuyển động quay của xe |
| `ego_motion.length_m`, `width_m` | mét | Kiểm tra thân xe, không chỉ điểm tâm |

Mỗi phần tử `LaneBoundaryV2`:

| Trường | Giá trị / đơn vị | Ý nghĩa |
|---|---|---|
| `valid` | bool | Nhãn/hình học có dữ liệu hợp lệ |
| `side` | LEFT / RIGHT | Bên của hành lang làn được chọn |
| `color` | YELLOW / WHITE / OTHER / UNKNOWN | Màu metadata được chuẩn hóa |
| `style` | SOLID / BROKEN / ROAD_EDGE / NONE / UNKNOWN | Kiểu biên |
| `polyline_ego` | tuple điểm (x, y), mét | Hình dạng biên cục bộ |
| `line_width_m` | mét hoặc thiếu dữ liệu | Bề rộng vạch nếu có nguồn xác thực |
| `source` | SIMULATOR_LOCAL_GEOMETRY | Khai báo nguồn thông tin |

Chia polyline khi màu/kiểu biên đổi giữa các đoạn đường. Không gán nhãn của
đoạn thẳng hiện tại cho toàn bộ đoạn cua tiếp theo. `UNKNOWN` khác `NONE`:
không biết có vạch không khác với xác nhận không có vạch. Không đưa một trường
`confidence` giả định như thể đã có thuật toán nhận diện ảnh.

Giới hạn cục bộ đề xuất: nhìn trước 30 m, lấy thêm đoạn phía sau đủ bao phủ
thân xe, lấy mẫu hình học khoảng 0.5 m. Khoảng nhìn và mật độ lấy mẫu phải được
kiểm thử với tốc độ và độ cong; có thể cần lấy mẫu dày hơn ở cua.

## 4. Tầng trích xuất dữ liệu

Tầng platform được phép đọc simulator để xây snapshot công khai; agent chỉ
được đọc snapshot. Không truyền live lane, vehicle, engine hoặc environment
vào policy. Chỉ xuất giá trị hữu hạn, enum, tuple/array bất biến và validity mask.

Quy trình:

1. Xác định làn trên tuyến đang chứa xe tại spawn. Giữ hành lang làn này thay
   vì tự kéo xe về `lanes[0]`.
2. Tại nối đoạn, chọn làn tiếp nối theo kết nối tuyến, vị trí và hướng liên tục.
   Không mặc định giữ cùng số index khi đường tách/nhập làn. Nếu mơ hồ, đánh
   dấu thiếu dữ liệu; không tự đoán rằng biên bị mất là được phép vượt.
3. Lấy tâm làn và hai biên bằng API hình học của lane. Kiểm tra quy ước trái/
   phải với đường thẳng và cả hai chiều cua.
4. Đọc màu và kiểu biên riêng; chuẩn hóa màu GREY của PG về WHITE theo bảng
   màu đã kiểm tra. Màu không khớp đi vào OTHER/UNKNOWN, không ép thành trắng.
5. Chuyển tất cả điểm sang hệ ego: +x phía trước, +y phía trái, mét.
6. Khi có biên dùng chung, xử lý trùng và nhãn mâu thuẫn rõ ràng; không tùy tiện
   chọn nhãn từ một trong hai làn kề nhau. So sánh với vạch được dựng thật.
7. Xuất snapshot; không xuất cờ `on_yellow_continuous_line`, terminal reason,
   reward, route completion, split/case ID hoặc dữ liệu tương lai từ evaluator.

Khoảng nhìn trước là bản đồ tham chiếu cục bộ, không phải ground-truth quỹ đạo
tương lai của xe. Đây là quyền thông tin mới, phải được công bố và cung cấp
theo cùng quy tắc cho các phương pháp được đem so sánh.

## 5. Luật sử dụng vạch

Màu và kiểu đều cần thiết. Quy tắc chạy thẳng/giữ làn đầu tiên:

| Loại | Hành vi |
|---|---|
| Vàng liền | Không chạm/cắt qua; ưu tiên giữ khoảng cách |
| Trắng liền | Không chạm/cắt qua theo cấu hình termination hiện tại |
| Trắng đứt | Vẫn giữ làn; chỉ cho phép chuyển làn khi có module riêng |
| Vàng đứt | Ghi nhận riêng; giai đoạn này vẫn giữ làn, không tự cho phép vượt |
| Biên đường | Không vượt ra khỏi mặt đường |
| UNKNOWN | Giảm tốc; không diễn giải thành quyền vượt biên |
| NONE | Không có vạch được xác nhận; vẫn bám hành lang tuyến hợp lệ |

Màu vạch không phải toàn bộ luật giao thông. Ở phiên bản này mục tiêu là tuân
thủ cấu hình môi trường và giữ làn, chưa xử lý quyền ưu tiên hoặc vượt xe.

## 6. Điều khiển qua cua và tránh chạm vạch

Luồng mỗi bước điều khiển:

```text
LaneContextV2 + EgoMotionV2
  -> chọn hành lang tâm làn liên tục
  -> tính lệnh bám tâm + hướng làn
  -> chọn tốc độ theo độ cong phía trước
  -> thử các lệnh lái/ga lân cận trong mô hình ngắn hạn
  -> loại lệnh dự đoán thân xe chạm biên cấm
  -> chọn lệnh còn lại gần lệnh bám tâm nhất
  -> nếu không có lệnh hợp lệ: giảm tốc/phanh và đánh lái phục hồi khả thi
```

Khoảng kiểm tra dự đoán ban đầu: 0.8–1.5 giây, cập nhật mỗi 0.1 giây như
launcher. Đây là mô hình động học nội bộ, không clone hay gọi evaluator làm
oracle. Cần hiệu chỉnh mô hình lái và kiểm tra sai số dự đoán.

Để tránh ngộ nhận tâm xe còn trong làn là đủ an toàn, dựng hình chữ nhật thân
xe từ chiều dài, chiều rộng và hướng dự đoán. Kiểm tra khoảng cách/giao cắt
của toàn bộ hình chữ nhật với biên và hành lang. Phải xét khoảng giữa các bước
dự đoán hoặc dùng bước đủ nhỏ để tránh bỏ sót lần cắt biên.

Ước lượng thô chỉ để giải thích: phần thân xe chiếm theo phương ngang là
`width/2 * abs(cos(error)) + length/2 * abs(sin(error))`. Khi xe quay lệch tâm
làn, phần đầu/đuôi có thể chạm vạch trước khi tâm xe tới gần vạch. Với cua thật,
dùng biên cong và footprint thay vì chỉ công thức đường thẳng này.

Chừa khoảng cách từ thân xe đến vạch khoảng 0.3–0.5 m cho lần thử đầu, có thêm
biên sai số mô hình. Kiểm tra hành lang có đủ rộng cho xe; không áp dụng ngưỡng
khiến mọi action đều bị loại ở làn hẹp. Nếu có bề rộng vạch, tính khoảng cách
tới mép vạch hoặc cộng nửa bề rộng vào vùng cấm. Phanh không đảm bảo cứu được
xe nếu đã sát vạch và đang quay nhanh: phải can thiệp từ trước cua.

Tốc độ mục tiêu giảm khi độ cong tăng. Luật ga dùng sai số tốc độ thay cho ga
cố định; có thể dùng giới hạn `v <= sqrt(a_lateral_limit / abs(curvature))`
với kiểm soát trường hợp đường thẳng và giới hạn tốc độ thử nghiệm. Đây là
giới hạn động học sơ khởi, không phải bảo đảm an toàn của động lực học MetaDrive.

Mất dữ liệu tạm thời: nếu giữ hành lang bước trước thì phải khai báo stateful,
giới hạn thời gian sử dụng và xóa trong reset(). Không tăng ga chỉ vì dữ liệu
vạch biến mất. Không phanh cứng vô hạn mỗi khi biên không tồn tại ở giao lộ.

## 7. Thứ tự triển khai và tiêu chí nghiệm thu

### A. Kiểm chứng bám đường trước

- Đo chiều heading/độ lệch với steering dương và âm trong runtime headless.
- Đưa mục tiêu trái/phải có tọa độ ego biết trước; action phải làm giảm sai số.
- Xác nhận waypoint hiện tại có cùng làn và cùng tọa độ dọc với làn xe giữ.
- Kiểm tra tốc độ, thời điểm bắt đầu cua và footprint sát biên. Không tiếp tục
  tăng gain khi chưa biết sai số do dấu lái, chọn làn hay tốc độ.

### B. Trích xuất loại vạch

- Unit test vàng/trắng, liền/đứt, biên đường, NONE và UNKNOWN.
- Kiểm tra mẫu trên đường thẳng, cua trái/phải, đoạn chuyển tiếp và tách/nhập làn.
- Đối chiếu trái/phải và nhãn snapshot với vạch render/geometry thực tế.
- Đảm bảo snapshot không có simulator handle hoặc evaluator-private telemetry.

### C. Điều khiển

- Test ở đoạn thẳng và cua không traffic trước; dùng cả spawn ở các làn khác nhau.
- So sánh agent cũ và mới trên cùng map/seed với cấu hình tham số lưu đầy đủ.
- Ghi khoảng cách nhỏ nhất từ thân xe tới biên cấm, độ lệch tâm, tốc độ vào cua,
  độ cong, action trước/sau bộ lọc và lý do can thiệp.
- Đo tiến độ tuyến, clean success, OUT_OF_ROAD, timeout. Xe không chạm vạch
  nhưng dừng mãi cũng không đạt yêu cầu.
- Chỉ dùng TRAIN/VALIDATION theo quy trình phát triển; không chỉnh từ TEST.
- Chạy nhiều hình học/cua, không tuyên bố bảo đảm mọi bản đồ từ một lần chạy.

### D. Tích hợp phiên bản mới

Đưa schema, extraction và profile mới vào extension riêng; cập nhật kiểm tra
immutability, masking, registration, resolver/executor, input serialization và
contract/version của extension. Làm rõ CORE_ONLY che dữ liệu vạch thế nào.
Ban đầu chỉ cho chạy SANDBOX/AUDIT. Không đổi hash, schema hoặc ý nghĩa của V1
một cách âm thầm; đánh giá V1 và profile mở rộng phải ghi rõ quyền thông tin.

## 8. Bố trí file đề xuất

```text
src/platform/lane_context_v2.py       # snapshot/enums/validation
src/platform/lane_extraction_v2.py    # simulator -> snapshot cục bộ
src/platform/agent_input_v2.py        # wrapper/profile mở rộng
src/agents/lane_marking_agent.py      # luật giữ làn, tốc độ, lọc action
tests/test_lane_context_v2.py         # schema/labels/frame/immutability
tests/test_lane_marking_agent.py      # phản ứng trái-phải/biên/mất dữ liệu
scripts/audit_lane_marking_agent.py   # kiểm tra tích hợp headless có báo cáo
```

Agent mới dùng chung action adapter continuous đã có. Những file này là bố trí
đề xuất, chưa được tạo bởi tài liệu này.
