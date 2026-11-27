# Observation sensitivity của SAC V2–5.000

**Nguồn ảnh hưởng lớn nhất ở ca Extreme là TrafficContext.** Trên cùng observation lúc bắt đầu pha dừng cuối, bỏ riêng NPC thứ tư ở sát 50 m đổi đầu ra ga/phanh từ **−0,3344 sang +0,9121**. NPC đó không có tia LiDAR first-hit. Thử độc lập với NPC **31,4 m** tại observation cuối cho kết quả cùng chiều: bỏ đúng một slot đổi **−0,2234 sang +0,9807**. Hai kết quả xác định độ nhạy đầu vào của Actor. Khả năng đi qua đoàn xe sau biến đổi chưa được kiểm chứng vòng kín.

## Phạm vi và đối chứng

V2–5.000 chạy nguyên bản trên sáu geometry, gồm đúng ca Extreme `geom_extreme_SCTXORyCCS_seed2`, environment seed **33702**, và năm ca TRAIN đại diện. Chạy headless với bốn process, giữ timestep 0,02 giây và năm physics steps mỗi quyết định. Lưu 6.423 observation/action, gồm observation terminal dùng riêng cho inference, và chọn 1.998 frame cho các phép thử offline. Tập thử gồm đường thẳng, cua, đoạn vào vòng xuyến, xe gần biên, xe đổi lane index trong cùng đoạn road và giao thông nhiều tier. Easy là đối chứng không giao thông. Tập này không thay thế panel 48 ca.

Mỗi frame lưu vector 409D từ encoder, vector sau Terminal Preview và transform, action deterministic, toàn bộ danh sách producer trước giới hạn tám slot, các hit LiDAR được xác minh và trạng thái simulator dùng cho đối chiếu. Replay nguyên bản tái hiện timeout ở **39,7294%**. Actor SB3 đã nạp có FlattenExtractor, hai lớp 256 ReLU và đầu ra mean Box2 qua tanh. Sai số tối đa khi tái chạy inference theo batch trên vector nguyên bản là **1,34×10⁻⁵**.

| Slice Python | Ý nghĩa và thang đo thực tế |
|---|---|
| `0:9` | Khoảng cách biên trái, biên phải chia 18 m và clip `[0,1]`; độ thẳng hướng với lane qua lateral alignment `(clip(cos)+1)/2`; tốc độ `(km/h+1)/81`; steering vật lý theo MAX_STEERING 60; hai action trước theo `(a+1)/2`; yaw-rate proxy không dấu và clip; offset tâm lane `offset/4,5 m + 0,5` và clip. |
| `9:19` | Hai bộ checkpoint 5D. Trong mỗi bộ: tọa độ trước/phải của checkpoint giới hạn độ dài theo NAVI_POINT_DIST và ánh xạ về `[0,1]`; bán kính cua theo giới hạn cấu hình; chiều cua; góc cung theo giới hạn cấu hình. Quy ước bên phải của navigation khác trường y hướng trái trong TrafficContext. |
| `19:259` | LiDAR động 240 tia, tầm 50 m, tia 0 hướng đầu xe, góc tăng 1,5° theo heading, độ cao 1,2 m. Giá trị là hit fraction; 1,0 biểu thị không có return. Runtime không có dropout, noise hoặc slot xe bổ sung trong core LiDAR. |
| `259:315` | Tám actor × `x,y,vx,vy,heading,length,width`. Vị trí và vận tốc tương đối chiếu sang hệ ego, +x trước và +y trái; heading tương đối quấn quanh π. Vector Actor chia lần lượt cho 50 m, 50 m, 30 m/s, 30 m/s, π, 5 m, 5 m. |
| `315:323`, `323:325` | Valid prefix 1, padding actor và mask bằng 0. Hai metric là số slot hợp lệ / 8 và `min(overflow,10)/10`. Producer chọn xe active trong bán kính Euclidean 50 m rồi sắp theo khoảng cách, hòa theo x và y. Tái dựng từ dữ liệu producer đầy đủ nên slot overflow có thể được đôn lên khi xóa actor. |
| `325:385`, `385:405` | 20 waypoint theo station, cách nhau 2,5 m nhìn đến 50 m; `x,y,heading` chia cho 50 m, 50 m, π sau Terminal Preview. Mask và padding giữ theo encoder/adapter. |
| `405:409` | Bề rộng lane / 5 m, hai thành phần hướng mục tiêu chuẩn hóa theo vector hướng, cờ route-end. |

Source kiểm chứng gồm `src/platform/agent_input.py`, `src/training/stage5/observation_encoder.py`, `pilot_observation_transform.py`, `src/agents/stage5_terminal_preview_v1.py` và MetaDrive pin `85e5dadc6c7436d324348f6e3d8f8e680c06b4db` tại `obs/state_obs.py`, `component/sensors/{lidar,distance_detector}.py`, `component/navigation_module/node_network_navigation.py` và `constants.py`. Cảm biến vạch và side detector đều tắt. Mask LiDAR gồm Vehicle, InvisibleWall, TrafficObject và TrafficParticipants; không bao gồm lane-line collision groups. Actor không nhận màu vạch vàng. Khoảng cách biên, heading, lane offset và waypoint vẫn có thể làm Actor thay đổi steering.

## Kết quả ở ca Extreme

| Mốc quyết định | Tốc độ | Ga/phanh gốc | Bằng chứng |
|---|---:|---:|---|
| 10 | 10,74 km/h | +0,982 | Hai tia pending first-hit; TrafficContext trống. |
| 20 | 20,02 km/h | +0,635 | Ba actor active vào context; che actor xa trong context cho +0,997. |
| 42 | 27,45 km/h | −0,169 | Lần đầu qua −0,05. Che riêng các actor >30 m trong context cho +0,912; thay riêng hai tia liên quan cho +0,018. Actor gần khoảng 26,4 m vẫn nằm trong context. |
| 47 | 23,09 km/h | −0,322 | Đợt giảm tốc đầu tiên đo rõ. |
| 2.455 | 0,013 km/h | +0,932 | Ba slot active, trước pha phanh cuối. |
| 2.456 | 1,010 km/h | −0,334 | Thêm đúng slot thứ tư tại 49,994 m, ngoài các first-hit LiDAR. Che riêng slot đó cho +0,912. |
| 2.457 | 0,464 km/h | −0,235 | Bắt đầu 51,7 giây cuối dưới 1 km/h; bỏ slot thứ tư cho +0,931. |
| 2.972, action thực thi cuối | 0,010 km/h | −0,221 | Bỏ slot thứ tư cho +0,933. |
| 2.973, input terminal chỉ để đo | 0,016 km/h | −0,223 | Bỏ slot thứ tư cho +0,933; bỏ xe 31,4 m cho +0,981. |

Quét riêng bán kính NPC thứ tư trên frame 2.456 cho ga/phanh **−0,3345 ở 50,00 m** và **+0,9121 ở 50,01 m**. Frame 2.973 cho **−0,2234** và **+0,9326**. Đây là bước nhảy tại biên đưa xe vào TrafficContext, khi cả slot và active count thay đổi đúng schema. Dữ liệu chưa định danh riêng mức đóng góp của count, mask và đặc trưng slot; giữ một count giả với mask mâu thuẫn sẽ tạo input sai hợp đồng. [Biểu đồ ngưỡng 50 m](../../../runs/kaggle/observation-sensitivity/traffic_range_edge.png)

### TrafficContext và LiDAR ở cùng observation terminal

| Nhóm | A/E gốc | B chỉ che context | C chỉ thay LiDAR | D thay cả hai |
|---|---:|---:|---:|---:|
| Xe 31,4 m | −0,2234 | +0,9807 | −0,3623 | +0,9775 |
| Xe slot 4 sát 50 m | −0,2234 | +0,9326 | −0,2234 | +0,9326 |
| Toàn bộ xe active >30 m | −0,2234 | +0,9998 | −0,5317 | +0,9998 |
| NPC pending | −0,2234 | −0,2234 | −0,2234 | −0,2234 |

Che context theo Euclidean được đối chiếu với ngưỡng 45 đến 5 m. Phép tương ứng theo station trên dải waypoint, chỉ chọn xe cách dải tối đa 3 m, không loại xe nào ở mẫu cuối. Điều đó mô tả bất đồng giữa bán kính producer và proxy dải tham chiếu hiện tại, chưa chứng minh quỹ đạo thay thế thông thoáng. Tại frame terminal, che riêng xe 31,4 m trong context đổi steering **−0,0240 → −0,0160**; che toàn bộ xe active xa đổi **−0,0240 → +0,8737**. Che xa đồng loạt có thể tác động rất khác khi xe vẫn đang di chuyển.

Phép che LiDAR recast bằng Bullet với cùng origin, mask, hướng tia và lọc ego. Chỉ thay tia khi first-hit duy nhất khớp hit fraction gốc đến sai số 2×10⁻⁵. Nếu tia có hit phía sau đã xác minh, dùng hit kế tiếp; nếu không, trả về 1,0. Không xóa vùng góc chung. Đã quy nguồn tất cả hit được ghi nhận theo phép đối chiếu này. Trong ca gốc, pending là first-hit ở 20 frame đầu có ghi nhận, không có first-hit pending ở hai mốc phanh nêu trên. Trên 70 observation đã lấy mẫu có pending first-hit ở bốn geometry, che pending không đảo dấu ga/phanh với deadband ±0,02; |Δ| lớn nhất **0,0109**. Kết quả bác bỏ pending first-hit là lời giải cho pha phanh cuối này. Trigger và chuỗi xe đứng yên vẫn là bối cảnh giao thông có thật.

### Khoảng cách, vận tốc và làn

- Giữ nguyên heading, hướng nhìn và các slot còn lại, dời xe mục tiêu 31,4 m theo cùng bearing làm ga/phanh ở 5/30/35/40/45 m lần lượt **−0,856 / −0,284 / −0,087 / +0,108 / +0,504**. Không suy ra ngưỡng an toàn từ sweep này.
- Giữ nguyên khoảng cách và heading thật của xe 31,4 m, đặt vận tốc đích theo heading của chính xe ở 0/2/5/10 m/s cho ga/phanh **−0,224 / +0,225 / +0,716 / +0,966**. Vận tốc tương đối đã trừ đúng vector ego. Khi dùng giả lập cắt ngang với heading và vận tốc ngang phù hợp, ga/phanh là **−0,630**. Actor có sử dụng vận tốc, dù chưa thể xác nhận nó đã học đúng rủi ro cho mọi tình huống.
- Ở cùng bán kính và tốc độ NPC 5 m/s, di chuyển mục tiêu giữa trước, lệch trái, lệch phải, sau và ngược chiều cho ga/phanh **+0,820 / +0,855 / +0,777 / −0,909 / +0,230**. Steering tương ứng **−0,123 / −0,177 / −0,084 / −0,882 / −0,709**. Actor phân biệt vị trí. Ga/phanh âm mạnh khi đặt xe cùng hướng phía sau là dấu hiệu cần kiểm tra thêm bằng cảnh tự nhiên. Các sweep chỉ đổi TrafficContext và giữ LiDAR gốc, nên có thể là input ngoài phân phối hoặc mâu thuẫn liên cảm biến.

### Mức độ lặp lại qua geometry

Bảng chỉ tính mẫu đã thực sự bỏ ít nhất một actor >30 m trong TrafficContext (LiDAR nguyên bản). Mẫu liên tiếp tương quan cao; tỷ lệ mô tả tập probe, không phải tần suất ngoài thực tế. Đổi dấu dùng deadband ±0,02, mỗi tỷ lệ có mẫu số phù hợp với dấu gốc.

| Ca | Mẫu | Phanh → ga | Ga → phanh | Đảo steering | Trung vị Δ ga/phanh |
|---|---:|---:|---:|---:|---:|
| Extreme gốc | 1.052 | 782/794 | 10/239 | 492/752 | +1,209 |
| Extreme còn lại | 194 | 39/52 | 20/138 | 69/171 | +0,237 |
| Hard | 155 | 28/37 | 4/116 | 52/148 | +0,036 |
| Medium | 119 | 16/18 | 4/99 | 43/113 | +0,099 |
| Medium cua | 45 | 0/7 | 18/38 | 18/42 | −0,381 |
| Easy không traffic | 0 | — | — | — | Không có actor để bỏ |

Độ nhạy lặp ở nhiều geometry. Dấu của tác động thay đổi theo cảnh, đặc biệt ở Medium cua. Che mọi xe xa hoặc giảm tầm nhìn đồng loạt sẽ can thiệp cả những cảnh có phản ứng theo chiều ngược lại.

## Biên đường và waypoint

Ở ca gốc cuối episode, khoảng cách biên trái/phải là **1,955/8,545 m** trong cua; xe lệch tâm lane khoảng **0,205 m** và lệch hướng khoảng **0,173 rad**. Phân phối lại hai khoảng cách biên để cả hai bằng 5,25 m, giữ tổng chiều rộng và các feature khác, đổi ga/phanh từ −0,223 sang **+0,176**, steering từ −0,024 sang **+0,300**. Chỉ giảm khoảng biên trái từ mức gốc xuống 1,50 m cho ga/phanh **−0,273**. Chỉ đổi offset tâm lane từ −1,5 m sang +1,5 m đưa ga/phanh từ −0,062 xuống −0,334. Sweep riêng signed lateral alignment cho biến thiên nhỏ hơn, từ **−0,233 đến −0,167**.

Ở 143 mẫu gần biên nhưng hướng gần song song, đặt biên trái tại 1,5 m có trung vị |Δ steering| **0,527** trong probe gộp, với trung vị |Δ ga/phanh| **0,008**. Tác động tùy bên. Các ca đang hướng ra ngoài, giữa lane, trong cua và đổi lane index được giữ riêng trong CSV. Các probe thay cặp biên giữ tổng bề rộng, còn offset, heading và waypoint được thử độc lập. Không suy ra va chạm hoặc ý định chuyển làn từ thao tác feature giả định.

Dịch waypoint ngang −1,5/+1,5 m ở cùng mẫu cuối cho ga/phanh **−0,654/+0,502**. Làm thẳng suffix waypoint ở cùng anchor cho **−0,910**, trong khi đặt riêng nhóm curvature của hai checkpoint về encoding đường thẳng cho **+0,015**. Waypoint và navigation là nguồn ảnh hưởng phụ thuộc cảnh. Dữ liệu chưa quy được nguyên nhân duy nhất cho độ cong và cũng không có tín hiệu màu vạch để quy lỗi trực tiếp cho vạch vàng.

## Kết luận và phương án trình duyệt

1. **Đã xác nhận** NPC active ở xa có thể duy trì lệnh phanh của Actor trên đúng observation. Bước nhảy slot thứ tư ngay biên 50 m có bằng chứng mạnh nhất cho pha dừng cuối.
2. **Đã xác nhận** Actor dùng khoảng cách và vận tốc. **Chưa đủ bằng chứng** về độ hợp lý của các ngưỡng qua mọi tình huống.
3. **Có bằng chứng hỗ trợ** Actor phân biệt vị trí trước, cạnh và sau. Độ tin cậy cho nhận diện làn cùng tuyến vẫn chưa được xác nhận.
4. **Đã xác nhận** TrafficContext chi phối tại điểm dừng. LiDAR của xe 31,4 m ở đó có hiệu ứng ngược lại khi bị xóa.
5. **Bác bỏ trong ca gốc** pending first-hit gây pha dừng cuối. Pending được thấy bởi LiDAR ở đầu lượt, trước các mốc phanh.
6. **Bác bỏ tín hiệu màu vạch trực tiếp**, **xác nhận độ nhạy với biên**. Chưa đủ bằng chứng để kết luận né vạch vàng hoặc mọi phản ứng biên đều quá mức.
7. **Xác nhận độ nhạy với waypoint/navigation** trong cùng mẫu. Hiệu ứng phụ thuộc probe; chưa có bằng chứng rằng riêng curvature tạo pha dừng.
8. **Có bằng chứng hỗ trợ** Actor tận dụng một tương quan chưa đủ bền vững với sự hiện diện/packing của actor xa. Chưa truy được tương quan đó về mẫu replay hay gradient nào trong huấn luyện.
9. **Xác nhận độ nhạy trên nhiều geometry**, với cả hai chiều đổi dấu. Chuỗi dừng cụ thể tại 39,73% đã tái hiện trên đúng một geometry/seed.
10. **Đủ để đề xuất thiết kế**, chưa đủ để thay checkpoint hay hợp đồng observation đã phát hành.

Thiết kế ưu tiên xin duyệt là giữ tầm 50 m và tạo biểu diễn **relevance liên tục** cho branch TrafficContext. Tách mức độ hiện diện khỏi nguy cơ theo station, khoảng lệch hành lang, closing speed và độ tin cậy của track. Dùng pooling có mask và giảm ảnh hưởng discontinuity khi actor cắt qua biên bán kính, thay vì đưa count/slot biên làm dấu hiệu nguy hiểm. Giữ LiDAR để đối chiếu vật thể chưa có track. Đối với biên đường, tách khoảng cách hiện tại khỏi khả năng footprint hướng ra biên, rồi đánh giá policy retention riêng cho steering. Tiêu chí chấp nhận của một vòng sau là giảm bước nhảy trên các cặp qua 50 m và giữ phản ứng trước xe cắt ngang/cùng làn, sau đó mới xem kết quả vòng kín trên panel ghép cặp. Đợt hiện tại chỉ bàn giao chẩn đoán.

## Tái lập và hiện vật

Chạy lần lượt `python -m cloud.kaggle.observation_capture`, `python -m cloud.kaggle.observation_sensitivity` và `python -m cloud.kaggle.observation_sensitivity_figures` trong môi trường MetaDrive của dự án. Dùng thư mục capture mới khi cần tái hiện; CLI capture nhận `--out`, analysis nhận `--capture` và `--out`. Tại `runs/kaggle/observation-sensitivity/` có sáu CSV theo yêu cầu, bốn biểu đồ PNG, manifest SHA-256, sáu file NPZ và JSON metadata. CSV giữ geometry, seed, step, dải tốc độ, action gốc và action phản thực tế; dữ liệu private của simulator chỉ nằm trong telemetry chẩn đoán. Model và mọi file full state đã qua xác minh SHA-256 sau phân tích. Gradient updates, replay rows và số bước simulator dùng observation phản thực tế đều bằng 0.
