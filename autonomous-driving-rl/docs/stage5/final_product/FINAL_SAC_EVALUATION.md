# SAC V2–5.000 · Đánh giá cuối VALIDATION và TEST

Đã xác minh 296/312 episode hợp lệ. Bundle được chọn từ `stage-05000`. Checksum model khớp delivery lock trước và sau khi đọc kết quả. Đợt này có 0 SAC updates và 0 hàng nạp replay.

## Phạm vi so sánh

- Runner candidate sử dụng manifest Gate 5 đã khóa, gồm 48 geometry VALIDATION với seed 5101–5102 và 12 geometry TEST với seed 9101–9105. Mỗi điều kiện có cùng case ID, block sequence đã đối chiếu SHA-256, route, horizon và thứ tự protocol.
- Traffic ON dùng `max(0.08, manifest_density)` và OFF dùng `0.0`, theo quy tắc Search locked được nêu cho đợt đối chiếu này. Mật độ ON lần lượt là 0,08 / 0,08 / 0,15 / 0,25 cho Easy / Medium / Hard / Extreme. Cả hai dùng TrafficMode.Trigger, timestep 0,02 s, decision repeat 5 và điều khiển 10 Hz.
- Hai script đối chiếu Search/candidate và receipt gốc của các agent trong bảng nhóm chưa có trong checkout và các ref đã kiểm tra. Vì vậy reviewer có thể dùng cùng mẫu số 60 và điều kiện traffic nêu trên, với giới hạn là chưa xác minh độc lập rằng từng hàng của nhóm được tạo bởi cùng manifest, seed, traffic và outcome semantics. Giữ nguyên các hàng do nhóm cung cấp.
- Đây là `SAC_V2_PAIRED_TRAFFIC_EVALUATION`, `canonical_run=false`, `training_replicates=1`. SAC dùng observation 409D, F09, Terminal Preview V1 và Box2 theo bundle đã bàn giao. Một checkpoint deterministic được chạy trên nhiều environment seed. Điều đó không cung cấp phương sai giữa các lượt huấn luyện độc lập hoặc chứng nhận Gate 5.
- Loader inference hiện hành không mở full state hay replay nguồn. SB3 đọc các tensor optimizer có sẵn trong `model.zip` khi nạp bundle. Không có optimizer step.
- Báo cáo tính route completion tại bước kết thúc, không dùng tiến độ cực đại. Lỗi an toàn được ưu tiên hơn đến đích và timeout. Thời gian thành công chỉ lấy từ episode clean. Inference latency là số đo local của `agent.act`, không dùng để xếp hạng với agent khác.

## Kết quả tổng hợp

| Split | Traffic | Episodes | Clean Success | Safety Failure | Timeout | Mean Final Route | Median Final Route | Mean Time to Clean Success (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VALIDATION | ON | 96/96 | 46/96 (47.92%) | 32.29% | 19.79% | 66.97% | 70.09% | 60.41 |
| VALIDATION | OFF | 86/96 | Chưa đủ ca | N/A | N/A | N/A | N/A | N/A |
| TEST | ON | 60/60 | 19/60 (31.67%) | 58.33% | 10.00% | 55.33% | 43.61% | 51.75 |
| TEST | OFF | 54/60 | Chưa đủ ca | N/A | N/A | N/A | N/A | N/A |

## VALIDATION theo độ khó

| Tier | ON Clean | OFF Clean | ON Safety | OFF Safety | ON Mean route | OFF Mean route |
| --- | --- | --- | --- | --- | --- | --- |
| Easy | 22/24 (91.67%) | 24/24 (100.00%) | 8.33% | 0.00% | 96.13% | 98.78% |
| Medium | 19/24 (79.17%) | Chưa đủ ca | 20.83% | N/A | 90.17% | N/A |
| Hard | 5/24 (20.83%) | Chưa đủ ca | 58.33% | N/A | 51.67% | N/A |
| Extreme | 0/24 (0.00%) | Chưa đủ ca | 41.67% | N/A | 29.91% | N/A |

## TEST theo độ khó

| Tier | ON Clean | OFF Clean | ON Safety | OFF Safety | ON Mean route | OFF Mean route |
| --- | --- | --- | --- | --- | --- | --- |
| Easy | 12/15 (80.00%) | 15/15 (100.00%) | 20.00% | 0.00% | 86.16% | 99.00% |
| Medium | 7/15 (46.67%) | Chưa đủ ca | 53.33% | N/A | 70.80% | N/A |
| Hard | 0/15 (0.00%) | 5/15 (33.33%) | 93.33% | 13.33% | 32.22% | 61.98% |
| Extreme | 0/15 (0.00%) | Chưa đủ ca | 66.67% | N/A | 32.15% | N/A |

## Ảnh hưởng của giao thông theo cặp

| Split | Cặp đã đo | Cả hai clean | Chỉ OFF clean | Chỉ ON clean | Cả hai thất bại | OFF − ON trên cặp có outcome (điểm %) | OFF − ON toàn tập (điểm %) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VALIDATION | 86/96 | 43 | 13 | 2 | 28 | 12.79 | N/A |
| TEST | 54/60 | 18 | 17 | 0 | 19 | 31.48 | N/A |

## Số liệu slide và poster

| Phương pháp | TEST Traffic ON | TEST Traffic OFF |
| --- | --- | --- |
| SAC V2–5.000 | 19/60 (31.67%) | Chưa đủ ca |

| Phương pháp | Traffic | Clean Success Rate | Safety Failure Rate | Mean Route Completion | Median Route Completion |
| --- | --- | --- | --- | --- | --- |
| SAC V2–5.000 | ON | 31.67% | 58.33% | 55.33% | 43.61% |
| SAC V2–5.000 | OFF | N/A | N/A | N/A | N/A |

## Mật độ MapSuite mặc định

Tổng hợp Easy OFF với Medium, Hard và Extreme ON theo mật độ manifest. Các hàng này dùng lại episode đã đo.

| Split | Episode | Clean | Safety failure | Mean final route |
| --- | --- | --- | --- | --- |
| VALIDATION | 96/96 | 48/96 (50.00%) | 30.21% | 67.63% |
| TEST | 60/60 | 22/60 (36.67%) | 53.33% | 58.54% |

## Tái lập và kiểm tra

Chạy từ thư mục `autonomous-driving-rl` trong môi trường MetaDrive/SB3 của Stage 5.

```powershell
python -m cloud.kaggle.final_sac_evaluation
python -m cloud.kaggle.final_sac_report
```

Artifact nằm trong `runs/kaggle/final-sac-evaluation/`. `parity-proof.json` đối chiếu demo Medium, mật độ thực và trace action. `integrity-check.json` ghi độ phủ và SHA-256. `seed-summary.csv` phân nhóm theo environment seed. Các nhóm chưa đủ ca để trống tỷ lệ.

Còn 16 ca thiếu. Lỗi đã ghi nhận ở các ca còn thiếu gồm TaskContextLocalizationError (16 ca). F09 dừng lượt trước khi Actor nhận observation thiếu định vị. Chi tiết từng lần chạy nằm trong `integrity-check.json` và `infrastructure.jsonl`.

| Nhóm thiếu ca | Outcome hợp lệ | Clean quan sát | Lỗi an toàn quan sát | Timeout quan sát |
| --- | --- | --- | --- | --- |
| VALIDATION OFF | 86/96 | 56 | 20 | 10 |
| TEST OFF | 54/60 | 35 | 11 | 8 |

## Recovery F09

Đã chạy lại 16 ca thiếu theo thứ tự TEST rồi VALIDATION. Log ban đầu chỉ có exception và stack trace, nên phục hồi trực tiếp từ log được 0 ca. Recovery có thêm 0 outcome hợp lệ. Tổng cuối **296/312**.

Runner gốc phân loại outcome ngay sau `env.step` và dừng vòng lặp trước lần tạo observation kế tiếp nếu đã terminal. Instrumentation chỉ đọc flags và trạng thái khi F09 báo lỗi. Outcome vẫn do evaluator chung quyết định.

| Split | Điều kiện F09 thất bại | Outcome tại bước lỗi | Số ca |
| --- | --- | --- | --- |
| TEST | CURRENT_LANE_NOT_ON_ROUTE | UNDETERMINED | 6 |
| VALIDATION | CURRENT_LANE_NOT_ON_ROUTE | UNDETERMINED | 10 |

Ở bước 195–296, `navigation.current_lane` không trùng object lane nào trong chuỗi edge từ checkpoint của route. F09 yêu cầu đúng một lane làm điểm neo quanh checkpoint đích. Cả 16 ca có zero điểm neo, route station chưa xác định và không tạo được waypoint hợp lệ. Lane được MetaDrive chọn có prefix `-3X` ở 14 ca, `-3T` ở 2 ca. Ví dụ, 5 seed của TEST Medium `SCXCS` được nhận diện trên edge `-3X1_0_ → 3X0_0_` trong khi reference vẫn là `2C0_1_ → 3X0_0_`. Tất cả native termination, truncation và safety flags đều false. Đây là lỗi localization trong episode, chưa đủ bằng chứng để phân loại `OUT_OF_ROAD` hay outcome khác. Sửa quy tắc chọn lane làm điểm neo sẽ đổi điều kiện sinh observation, nên bộ kết quả giữ riêng 16 ca này.

Đối chứng TRAIN Medium ON/OFF cho cùng SHA-256 chuỗi action Box2, primary reason, final route completion và simulation time. Đã đối chiếu SHA-256 của manifest gốc, 296 receipt gốc và model. Không đổi producer hay evaluator. 0 SAC updates và 0 hàng nạp replay.

Chạy lại chẩn đoán bằng `python -m cloud.kaggle.final_sac_recovery`. [Manifest đánh giá gốc](../../../runs/kaggle/final-sac-evaluation/manifest.json) · [Log lỗi gốc](../../../runs/kaggle/final-sac-evaluation/infrastructure.jsonl) · [Trace localization](../../../runs/kaggle/final-sac-evaluation-recovery/localization-diagnostics.csv) · [Receipt recovery](../../../runs/kaggle/final-sac-evaluation-recovery/recovery-receipts.jsonl) · [Bảng poster sau recovery](../../../runs/kaggle/final-sac-evaluation-recovery/poster-table.csv)
