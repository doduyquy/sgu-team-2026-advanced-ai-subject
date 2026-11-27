# Short-range và Dynamic TrafficContext trên SAC V2–5.000

Giữ V2–5.000 với TrafficContext mặc định 50 m. Trên ca Extreme gốc, cả năm cấu hình thu hẹp dừng hoặc gặp lỗi trước vị trí 39,73%. Các lượt ghép cặp cho thấy thay đổi bán kính có thể giúp một ca Medium, đồng thời làm mất một ca Hard mà baseline đã hoàn thành sạch.

## Ca Extreme gốc

Geometry `geom_extreme_SCTXORyCCS_seed2`, environment seed `33702`. Mỗi profile chạy một episode deterministic. Thời gian đứng yên được cộng trên cả episode, với ngưỡng tốc độ dưới 1 km/h.

| Profile | TrafficContext | Tiến tuyến | Clean | Lỗi an toàn | Timeout | Đứng yên, s | NPC kích hoạt |
|---|---|---:|---:|---|---:|---:|---:|
| A | 50 m | 39,73% | 0 | 0 | 1 | 204,0 | 29 |
| B | 20 m | 9,72% | 0 | 0 | 1 | 280,4 | 9 |
| C | 15 m | 11,97% | 0 | Ra khỏi đường | 0 | 0,0 | 9 |
| D | 10 m | 7,88% | 0 | 0 | 1 | 284,4 | 9 |
| E | 5 m | 8,87% | 0 | 0 | 1 | 283,5 | 9 |
| F | Động 5–50 m | 6,86% | 0 | 0 | 1 | 286,9 | 9 |

A tái hiện 51,7 giây đứng yên liên tục cuối lượt và còn 20 NPC pending. B–F đều còn 40 NPC pending. Không lượt nào kích hoạt nhóm tiếp theo sau điểm dừng gốc, nên phép thử chưa cho thấy dòng xe phía trước được giải phóng. Phạm vi cơ sở F thực đo 6,0–24,9 m, với 27 lượt giữ actor ngoài bán kính cơ sở do dự báo xung đột.

B, D, E và F vẫn phanh khi kết thúc, lúc TrafficContext có 0 slot hợp lệ. Lệnh ga/phanh cuối lần lượt là −0,321, khoảng −0,993 ở D và E, và −0,904 ở F. LiDAR vẫn có tầm 50 m. Các lượt đã đi qua những trạng thái khác nhau, nên kết quả này chưa xác định feature nào duy trì phanh ở các điểm dừng mới.

Trong 5 giây trước khi C ra khỏi đường, tốc độ tăng từ 20,7 lên 39,0 km/h và ga cuối là +0,998. V2 gốc lần đầu đi qua cùng mức tiến tuyến ở khoảng 23,0 km/h. So sánh này mô tả diễn biến trên hai quỹ đạo khác nhau. Trace 50 bước cuối có trong `exact.json`.

## Kiểm chứng ghép cặp

Chạy A, B và F trên cùng sáu geometry/seed, gồm hai Medium, hai Hard và hai Extreme. B tiến xa nhất trong nhóm bán kính cố định thu hẹp mà không gặp lỗi an toàn ở ca gốc. F được giữ để đánh giá riêng phương án thích ứng.

| Profile | Clean / 6 | Lỗi an toàn | Timeout |
|---|---:|---:|---:|
| A | 1 | 5 | 0 |
| B | 1 | 5 | 0 |
| F | 1 | 3 | 2 |

- B và F hoàn thành sạch Medium `SCTCS`, nơi A va chạm. Trên Medium `SCXCCS`, A va chạm tại 80,02%, B ra khỏi đường tại 29,62% và F timeout tại 24,36%.
- A hoàn thành sạch Hard `XTOCS` ở 99,31%. B và F va chạm ở 11,81% và 10,49%. Không profile nào đạt clean ở hai ca Extreme ghép cặp.
- Đối chiếu phanh khi cùng một xe phía trước giảm tốc ≥1,8 m/s² và ego đang rút khoảng cách >0,5 m/s. Số bước phanh trên số bước đủ điều kiện là 8/51 ở A, 10/31 ở B và 2/16 ở F. Các quỹ đạo gặp số tình huống khác nhau. Những tỷ lệ này dùng để định vị hành vi trong trace, chưa đủ để xếp hạng khả năng phanh.

## Bước nhảy tại biên mới

Mỗi profile có 540 cặp suy luận offline, đặt một actor tổng hợp ở hai phía biên với sai khác ±0,01 m, giữ nguyên LiDAR và đóng gói lại slot, mask, count, overflow theo schema. Đây là phép đo độ nhạy bằng observation phản thực tế. Với F, biên bao gồm cả ngoại lệ dự báo xung đột và dao động khoảng 6–50 m trên các probe.

| Profile | P90 \|Δ ga/phanh\| | Max \|Δ ga/phanh\| | Max \|Δ steering\| |
|---|---:|---:|---:|
| A | 1,271 | 1,909 | 1,621 |
| B | 0,851 | 1,839 | 1,999 |
| C | 0,838 | 1,765 | 1,999 |
| D | 0,695 | 1,777 | 1,999 |
| E | 0,683 | 1,837 | 1,999 |
| F | 0,773 | 1,928 | 1,999 |

P90 bước nhảy ga/phanh giảm, nhưng từng probe ở biên mới vẫn có thể đảo gần toàn dải điều khiển. Thu hẹp phạm vi đã chuyển thay đổi rời rạc vào gần ego hơn. Kết quả vòng kín chưa đủ để chọn profile thử nghiệm hoặc đề xuất fine-tune ngay.

## Tái lập và artifact

Pipeline headless dùng pool tối đa tám worker. Sáu lượt gốc chạy song song, sau đó các công việc ghép cặp dùng lại pool. Thông lượng đo được 253 decisions/s ở ca gốc và 252 decisions/s ở nhóm ghép cặp, tương đương khoảng 25,3× và 25,2× thời gian mô phỏng. Physics 0,02 s, decision repeat 5, tần số điều khiển 10 Hz và LiDAR 50 m/240 tia được xác nhận lúc chạy. Các profile thử nghiệm thay đổi tập NPC tại producer, giữ đơn vị, normalization, 409D, Box2 và native reward.

```powershell
python -m cloud.kaggle.traffic_context_experiment
python -m cloud.kaggle.traffic_context_boundary
```

Dữ liệu thực nghiệm trong `runs/kaggle/short-dynamic-traffic/`. `exact.csv` và `paired.csv` chứa outcome từng lượt, hai tệp JSON tương ứng chứa trace. `boundary-probes.csv` chứa action của từng cặp. `manifest.json` lưu SHA-256 của kết quả. Kết quả dùng 0 SAC updates. V4 được dừng ở `stage-00128`, đã nạp lại full state và kiểm tra cả minibatch SAC kế tiếp lẫn auxiliary sampler với 0 updates bổ sung. Bằng chứng trong `runs/kaggle/risk-consistent-v4/pause-proof.json`.
