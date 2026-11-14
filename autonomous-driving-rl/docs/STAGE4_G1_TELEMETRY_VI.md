# G1 bước 1 — Kiểm chứng vận tốc và dấu lái

Ngày thực hiện: **13/11/2026 (Asia/Saigon)**. Phạm vi: kiểm chứng trên TRAIN,
chưa thu bộ dữ liệu BC, chưa chốt FIT/DEV hoặc quy tắc chất lượng, chưa mở G2.

## Kết luận và bản sửa

**Đạt kiểm chứng bước 1 trên ba hình học TRAIN.** MetaDrive 0.4.3 cung cấp
`vehicle.speed_km_h`, đơn vị km/h. Launcher trước sửa đọc `speed_kmh`, không
thấy trường thì ghi 0. Lỗi ảnh hưởng vận tốc trung bình/cực đại trong CSV,
tổng hợp và vận tốc trong sự kiện tiến độ. Tên khóa đầu ra `speed_kmh`,
`mean_speed_kmh`, `max_speed_kmh` được giữ nguyên.

Launcher hiện đọc `speed_km_h` qua `src/launcher/telemetry.py`; thiếu trường,
giá trị âm hoặc không hữu hạn gây lỗi kỹ thuật thay vì được ghi thành số 0.
Vận tốc được lấy **sau** mỗi `env.step`; trung bình/cực đại tính trên các mẫu
sau bước, kể cả bước kết thúc. Sự kiện tiến độ dùng cùng mẫu, mỗi 10 bước.

Với xe **đang đi tới**, hành động lái chuẩn **dương quay trái, âm quay phải**.
Giá trị này là lệnh lái chuẩn hóa trong [-1, 1], không phải góc tính bằng radian.
Đã sửa chú thích và `action_space_description` trong `action_adapter.py`.
Phép ánh xạ số của cả ba adapter, thuật toán/cấu hình/phiên bản Stage 1 0.6.1,
đăng ký launcher và các hợp đồng JSON đóng băng giữ nguyên.

## Bằng chứng trước/sau

Ba lượt tự nhiên ở mỗi phía dùng Stage 1, agent seed 42, environment seed 0,
render OFF, W&B DISABLED, tần số quyết định 10 Hz. Hình học được nạp qua cơ chế
băm/bản đồ đóng băng của launcher, không sinh hình học thay thế theo seed môi trường.

| TRAIN: họ / chuỗi / seed hình học | Bước | Kết thúc | TB trước sửa | TB sau sửa | Cực đại sau sửa |
|---|---:|---|---:|---:|---:|
| Easy / SCS / 0 | 418 | SUCCESS | 0 | 34.227616 | 44.999758 |
| Medium / SCXCS / 14 | 566 | SUCCESS | 0 | 22.614839 | 44.090953 |
| Hard / SCTXrCS / 2 | 1,432 | SUCCESS | 0 | 18.228190 | 44.982783 |

Đơn vị vận tốc trong bảng: km/h. Cả hai phía có **2,416 bước tự nhiên**;
2,296 bước có vận tốc trên 1 km/h nhưng cơ chế cũ vẫn trả 0.
Tham chiếu là `3.6 * hypot(vx, vy)` từ vận tốc thân cứng Bullet trong mặt phẳng.
Sai lệch lớn nhất giữa `speed_km_h` và tham chiếu là khoảng **7.11e-15 km/h**,
nhỏ hơn dung sai 1e-5 km/h. Cả 240 sự kiện tiến độ sau sửa khớp mẫu vật lý
tại đúng bước; trung bình/cực đại CSV của cả ba lượt đều khớp.

Vết quyết định được đối chiếu với bộ ghi thụ động bao quanh `env.step` của
**ExperimentExecutor thực tế**: quan sát t, hành động t và trạng thái trước/sau
bước được nối bằng case ID, chỉ số bước, hành động số. Vận tốc `ego_motion`
trong đầu vào công khai trước hành động khớp tham chiếu cùng thời điểm,
sai lệch tối đa khoảng 7.11e-15 km/h. SUCCESS ở đây chỉ là kết quả lượt kiểm
chứng, chưa chứng nhận quỹ đạo đạt chất lượng để học.

So sánh hai vết đầy đủ cho thấy **2,416/2,416 quan sát, quyết định, chẩn đoán
và hành động bằng nhau chính xác**, chỉ bỏ `act_latency_ms`. Xem
[đối chiếu từng tình huống](../results/audits/stage4/g1/telemetry_v1/paired_trace_comparison.json).

## Phép thử dấu lái

Mỗi hình học chạy ba phép thử riêng: 20 bước đi thẳng, ga 0.4; sau đó 6 bước
lái -0.25 / 0 / +0.25, ga 0.2. Tắt giao thông là can thiệp được ghi rõ;
các phép thử không được đưa vào BC. Chỉ đo khi xe tiến tới, trước chạm biên.
Heading delta được quấn về [-pi, pi], dịch chuyển ngang chiếu lên trục trái
của xe tại đầu xung, không suy hướng từ tên làn hoặc dấu của tọa độ màn hình.

| Lệnh lái | Đổi heading (rad) | Dịch chuyển theo trục trái ban đầu (m) | Hướng |
|---:|---:|---:|---|
| -0.25 | -0.081396 | -0.163266 | Phải |
| 0 | xấp xỉ 0 | xấp xỉ 0 | Thẳng |
| +0.25 | +0.081391 | +0.163262 | Trái |

**9/9 phép thử đạt ở mỗi phía**, tổng 234 bước/phía. Các hình học cùng xuất
phát trên đoạn thẳng nên kết quả xung giống nhau; đây không phải chín dạng
đường độc lập. Kết luận dấu lái áp dụng chuyển động tới đã kiểm chứng,
không suy rộng sang xe lùi. Các lượt tự nhiên còn cung cấp số đo dọc tuyến.

## Hồ sơ và khả năng tái lập

- Định nghĩa: `experiments/stage4/g1_telemetry_v1.json`.
- Công cụ: `scripts/audit_stage4_telemetry.py`, schema 1.0.0.
- Trước sửa: `runs/s4_g1_telemetry_before_03/`, COMPLETE, tái hiện lỗi ghi 0.
- Sau sửa: `runs/s4_g1_telemetry_after_01/`, COMPLETE, mọi phép kiểm đạt.
- Bằng chứng tổng hợp: [chỉ mục](../results/audits/stage4/g1/telemetry_v1/evidence_index.json).
- Kiểm thử: `runs/s4_g1_telemetry_checks_01/`; 5 kiểm thử mới bao phủ tên
  trường, lỗi cảm biến, quy đổi đơn vị, hệ trục/quấn góc và ánh xạ số adapter.

Kiểm thử cuối đạt **576/576**: 452 backend và 124 GUI, không lỗi hoặc bỏ qua.
Xem [hồ sơ kiểm thử](../results/audits/stage4/g1/telemetry_v1/test_verification.json).
Khi xuất báo cáo đã kiểm lại toàn bộ danh mục toàn vẹn hai lượt hoàn tất và
hai lượt lỗi, băm nhật ký kiểm thử và băm nguồn `src/`/`tests/` đã chạy kiểm thử.

Mỗi lần thử lưu nguồn và SHA-256, cấu hình, phiên bản runtime, kế hoạch với
case/seed, bản đồ, số đo từng bước, vết quan sát–hành động, kết quả, trạng thái
và danh mục toàn vẹn. MetaDrive được xác minh 0.4.3, commit
`85e5dadc6c7436d324348f6e3d8f8e680c06b4db`, cây nguồn sạch; Python 3.10.22.
Nguồn Stage 1, bản phát hành cũ, MapSuite, hợp đồng nền tảng và bằng chứng G0
được đối chiếu băm trước/sau trong từng lần và giữ nguyên.

Giữ hai lần thử lỗi: `before_01` lỗi tuần tự hóa `Config` khi lưu bản đồ;
`before_02` kết thúc sớm với xung lái 12 bước. Sau đó sửa tuần tự hóa và dùng
xung 6 bước đã ghi trong định nghĩa. Lỗi import ban đầu và lần sandbox từ chối
tạo thư mục xảy ra trước khi có hồ sơ, không được dựng thành lần mô phỏng.

Chạy lại bằng môi trường Python MetaDrive đã cài, từ `autonomous-driving-rl/`:

```powershell
python -B -X utf8 scripts/audit_stage4_telemetry.py --run-id s4_g1_telemetry_new --expected-logging corrected
python -B -X utf8 scripts/run_tests.py --output-dir runs/s4_g1_checks_new
```

Dùng ID mới, không ghi đè bằng chứng. `legacy_zero` dành cho nguồn trước sửa
đã lưu, không phải kỳ vọng cho mã hiện tại. Chỉ Git không chứa toàn bộ vết thô
trong `runs/`; cần giữ các thư mục gốc để kiểm lại mã băm và phép đo từng bước.

## Ảnh hưởng tới dữ liệu lịch sử và bước tiếp theo

Không ghi đè cột vận tốc của lượt cũ. Khi xem lịch sử, phân biệt đường ghi:
launcher trước sửa có lỗi; `run_rule_based_experiment.py` đã đọc `speed_km_h`
đúng và không tự động bị kết luận sai. Muốn dùng vận tốc lịch sử phải kiểm
nguồn cụ thể và có phép tái dựng kèm mã băm riêng nếu cần.

Bước 1 đã có bằng chứng; **G1 còn đang thực hiện**. Sau mốc báo cáo này đã
[thu thử](STAGE4_G1_PILOT_VI.md), [đo tài nguyên](STAGE4_G1_RESOURCES_VI.md) và
[chốt FIT/DEV](STAGE4_G1_SPLIT_KAGGLE_VI.md). Việc còn lại là duyệt chất lượng
và xây BC-A/BC-B. Chưa chọn ngưỡng chất lượng từ những lượt SUCCESS này.
