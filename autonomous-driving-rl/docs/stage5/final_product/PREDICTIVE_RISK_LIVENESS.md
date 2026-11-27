# Predictive Risk & Liveness với SAC Stage 5

Thực nghiệm mô phỏng này dùng actor V2–5.000 đã khóa. Quan sát 409D, action Box2, native reward, Reference Route Station, Terminal Preview, traffic, horizon và evaluator giữ nguyên. Mọi lượt chạy đều headless, với tối đa bốn process. Timestep vật lý 0,02 giây, decision repeat 5 và tần số điều khiển 10 Hz.

## Ca Extreme đã mắc kẹt

Chẩn đoán nội bộ chạy đúng geometry `geom_extreme_SCTXORyCCS_seed2` với environment seed 33702.

| Controller | Tiến tuyến cuối | Kết quả | Can thiệp |
|---|---:|---|---:|
| V2–5.000 | 39,73% | Timeout | 0 |
| IDM từ lúc reset | 19,38% | Va chạm xe | 243 bước IDM |
| V2 tới bước 1.360, sau đó IDM | 44,36% | Timeout | 1.613 bước IDM |
| V2 + predictive | 34,24% | Ra khỏi đường | 82 bước nhả phanh |
| V2 + guarded | 39,93% | Timeout | 18 bước nhả phanh |

Đối chứng chuyển sang IDM tái tạo cùng trạng thái V2 tại bước 1.360 theo lane, tiến tuyến 39,70%, vận tốc 0,024 km/h và action. Sau đổi controller, ego đi thêm 4,63 điểm phần trăm so với kết quả V2, tới ngay đoạn trước trigger của vòng xuyến. Nó vẫn dừng và chưa kích hoạt lô tiếp theo. Kết quả chỉ chứng minh còn chỗ tiến thêm tại điểm dừng của V2 trong ca này. IDM dùng thông tin simulator đặc quyền và được giữ riêng với controller chỉ đọc 409D.

Trace IDM của NPC xác định ba đầu hàng xe đều dùng một xe **chưa được kích hoạt** làm front vehicle. Cuối lượt V2, 17 xe active gần như đứng yên; chuỗi dài nhất có bảy xe active, cuối chuỗi tới một xe pending. Ba xe đầu hàng có khoảng hở IDM xấp xỉ 9,9–10,0 m, target speed 30 km/h và gia tốc IDM gần 0. Các xe active trong snapshot vẫn báo kết nối routing; trace không thấy vòng phụ thuộc giữa những xe đang dừng hoặc dấu hiệu nhánh fallback trong các mẫu đó. Kết quả gắn hiện tượng xếp hàng với điều kiện trigger và hành vi bám xe, chưa đủ để kết luận Traffic Manager có lỗi tổng quát. Bằng chứng cũ cho thấy LiDAR có thể thấy NPC pending trong khi TrafficContext lấy từ danh sách active. [Chẩn đoán Risk-aware trước](RISK_AWARE_SAC.md)

Ở cuối lượt V2, ego vẫn phát lệnh phanh −0,221 ở 0,016 km/h. Bản predictive giảm thời gian đứng yên vì kết thúc sớm do ra khỏi đường. Bản guarded giữ tỷ lệ ổn định trong làn xấp xỉ 77,6% và RMS thay đổi steering khoảng 0,300 như V2 trong ca này; proxy thời gian đứng yên khi có vẻ thông thoáng chỉ giảm từ 190,4 xuống 189,1 giây cộng dồn toàn episode. Hai biến thể chưa giải được điểm nghẽn.

## Phương pháp

- Dự đoán quỹ đạo ego trong 3 giây với bước 0,25 giây bằng mô hình bicycle xấp xỉ. Ngoại suy chuyển động NPC từ vị trí, vận tốc và heading, rồi nới footprint theo thời gian để biểu diễn bất định. Kiểm tra giao nhau của footprint theo từng mốc, khoảng hở có dấu theo các trục phân tách, lead gap theo tuyến, closing speed, TTC khi đang thu hẹp khoảng cách, time headway và khoảng dừng tham chiếu.
- Phân loại nhánh xung đột, có khoảng trống theo mô hình hoặc bất định. LiDAR bổ sung return chưa khớp với NPC đã được theo dõi. Thiếu waypoint, overflow, cấu hình khoảng cách biên không hợp lệ hoặc Terminal Preview chặn nhả phanh. Cảnh báo biên dựa trên footprint sát ego và một dải waypoint phía trước. Kiểm thử có cả trường hợp vạch ngang bên thân xe nhưng quỹ đạo thẳng vẫn lọt, lẫn trường hợp footprint vượt biên.
- Controller đầu tiên giữ action của actor làm mặc định. Khi có giao cắt footprint trong khoảng 1 giây, controller thử phanh mạnh hơn và chỉ nhận nếu trì hoãn hoặc loại giao cắt dự đoán. Khi tốc độ thấp, controller thử nhả phanh với steering gốc hoặc steering bám tiếp tuyến và yêu cầu năm frame đủ khoảng trống. Bản **guarded** giới hạn nhả phanh ở một lần dừng liên tục ít nhất 4 giây và giữ steering raw. Cả hai đều giữ điều khiển phanh theo cùng quy tắc.
- Mô hình dự đoán chưa sử dụng mô hình tương tác của NPC. Khoảng hở có dấu là proxy từ footprint dự đoán, không phải khoảng cách tiếp xúc đo bằng simulator. Không dùng khoảng hở này làm cam kết an toàn. Bộ điều khiển hiện chỉ xét một dải tham chiếu địa phương và không lập quỹ đạo chuyển làn.

### Nguồn thiết kế

Nghiên cứu Safety Potential xây dựng chi phí từ diện tích chồng lấn các ô dự đoán qua thời gian rồi đưa vào reward. Thực nghiệm này chỉ dùng ý tưởng kiểm tra vùng chiếm dụng trong tương lai ở lớp controller. [Safety Potential](https://www.mdpi.com/2079-9292/14/22/4446)

Brito và cộng sự dùng RL tạo tham chiếu vận tốc và khoảng bám cho MPC tương tác có ràng buộc. Trong thử nghiệm này, số phương án action hữu hạn được kiểm tra trực tiếp bằng footprint. Kamran và cộng sự tách chi phí vi phạm khỏi mục tiêu lái trong constrained RL. Thiết kế ở đây giữ kiểm tra xung đột riêng với tiêu chí tiến tuyến. [Interaction-Aware Guidance](https://pure.tudelft.nl/ws/portalfiles/portal/134864576/Learning_Interaction_Aware_Guidance_for_Trajectory_Optimization_in_Dense_Traffic_Scenarios.pdf) · [Constrained RL](https://tdsimao.github.io/assets/pdf/Kamran2022modern.pdf)

HACO phân biệt phần can thiệp của người với hành vi agent để học từ điều khiển từng đoạn. Nghiên cứu CILRLv3 dùng side-tuning cùng nhánh pretrained đóng băng và huấn luyện kết hợp dữ liệu expert/online. Hai nguồn này định hướng việc ghi nhận can thiệp và giữ V2 làm tham chiếu cố định. [HACO](https://decisionforce.github.io/HACO/) · [CILRLv3](https://www.mdpi.com/2076-3417/15/15/8411)

## Hành lang đa làn

Quan sát hiện hành có dải waypoint tham chiếu, khoảng cách biên gần ego và tập phương tiện giới hạn. Chưa có chuỗi tâm, biên và liên kết theo từng làn tới đoạn route sau, quyền chuyển làn theo vạch, hay vùng quan sát phía sau có dấu hiệu occlusion và confidence. Do đó thử nghiệm chưa cấp hành động vượt xe hoặc chuyển làn.

Thay đổi phù hợp là một schema có version riêng, ví dụ `LaneContextV2`. Mỗi hành lang cùng chiều có mẫu tâm/biên trong hệ ego, tiếp tuyến và curvature theo station, hướng, continuity tới nhánh tuyến kế tiếp, quyền đổi làn và mặt nạ hợp lệ. Bổ sung vùng quan sát trước/sau, tuổi track, độ tin cậy và vùng chưa quan sát. Chỉ dùng thông tin tuyến và cảm nhận hợp lệ ở inference. Scene IDs, trạng thái IDM, trigger road nội bộ và các track ngoài cảm biến chỉ dành cho chẩn đoán. Schema mới cần dữ liệu tương thích trước khi huấn luyện.

## Cổng teacher và quyết định bàn giao

Cổng yêu cầu bảy ca TRAIN ghép cặp hoàn chỉnh, tối thiểu hai ca cải thiện an toàn và không phát sinh lỗi an toàn ở ca vốn an toàn hoặc mất clean success. Ca Extreme gốc phải tiến thêm ít nhất 1 điểm phần trăm tuyến mà không phát sinh lỗi an toàn. Một ca TRAIN được tính cải thiện nếu chuyển sang clean success hoặc tiến thêm ít nhất 5 điểm phần trăm và không có lỗi an toàn.

Bản predictive biến một ca Medium từ va chạm ở 80,02% thành clean success ở 99,43% với 14 bước can thiệp phanh và không nhả phanh. Ở một ca Extreme TRAIN, nó chuyển từ ra khỏi đường ở 46,14% sang timeout ở 62,63%. Tuy vậy, nó ra khỏi đường trên đúng ca mắc kẹt nơi V2 chỉ timeout. Bản guarded tránh được lỗi mới ở ca gốc, nhưng tiến thêm khoảng 0,20 điểm phần trăm và vẫn timeout. Cả hai trượt cổng. [CSV theo ca](predictive_comparison.csv)

Trên bảy ca TRAIN, V2, predictive và guarded lần lượt đạt 2/7, 3/7 và 2/7 clean. Guarded vẫn cải thiện ca Medium kể trên, nhưng phát sinh lỗi ra khỏi đường ở ca Hard mà V2 hoàn thành sạch. Tổng cộng 28 lượt chạy chẩn đoán thực hiện 30.270 quyết định. Hai vòng TRAIN đo 33,56 và 35,49 quyết định/giây tính cả setup, tương đương tổng ×3,36 và ×3,55 ở 10 Hz. Đây là đo đầu-cuối của workload có dự đoán và telemetry, không đạt ×50.

Không tạo teacher replay, không có gradient update, không có checkpoint fine-tune hay số đo drift do learning. Giữ nguyên V2–5.000 làm model bàn giao. SHA-256 của model và các file full state đã được đối chiếu với delivery lock. Kết quả V2 trên panel 48 ca trước đó là Easy 12/12, Medium 12/12, Hard 3/12 và Extreme 0/12 clean. Các controller thử nghiệm dừng tại vòng chẩn đoán và sàng lọc.

## Tái lập

Chạy `python -m cloud.kaggle.predictive_diagnostic <phase> --out <thư-mục-mới>` lần lượt với `idm`, `exact`, `train-gate`, `revision-exact` và `revision-train`. Mỗi lượt xác minh SHA-256 của model từ delivery lock trước khi nạp. JSON chứa kết quả từng episode, số can thiệp và snapshots chẩn đoán. Bản ghi mới có raw, guided, executed và lý do cho từng lần can thiệp. Panel ghép cặp dùng cùng geometry và seed, chỉ phục vụ chọn phương pháp, không phải benchmark chính thức.

Các proxy phanh và đứng yên lấy cùng định nghĩa ở mọi controller. Steering reversal dùng deadband 0,05; chỉ số giữ làn yêu cầu liên tục ít nhất 10 bước và tâm ego nằm trong 1/4 bề rộng làn tính từ tim làn. Số lần đổi lane index cùng road và proxy thân xe chạm vạch đánh giá vị trí hình học, không suy ra ý định vượt xe. Yaw, gia tốc ngang và jerk được tính từ cấu hình 10 Hz. Các trường không có mẫu để trống trong CSV.
