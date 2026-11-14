# Kết quả kiểm định G0 — Bản trình bày tiếng Việt

Đây là bản diễn giải bằng chứng G0 đã có. Lần Việt hóa không chạy lại mô phỏng,
không tạo số đo mới và giữ nguyên mọi tệp JSON/nhật ký đã đóng băng.

| Kết quả lịch sử được xác minh | Giá trị |
|---|---:|
| Tình huống TRAIN lái tự nhiên | 16 |
| Hình học bản đồ riêng biệt | 12 |
| Quyết định đã kiểm tra | 22,228 |
| Đến đích an toàn | 13 |
| Hết thời gian | 3 |
| Bước phát lại khớp | 1,992 |
| Khoảng TIMEOUT chờ xem xét | 118 |
| Khoảng đã chấp nhận vào BC | 0 |
| Tệp được bảo vệ giữ nguyên | 308 |
| Kiểm thử phần lõi trong phụ lục xác minh G0 | 447 |

G0 đã nghiệm thu ngày **13/11/2026**, với giới hạn theo
[biên bản nghiệm thu](STAGE4_G0_REVIEW_VI.md). Sau đó G1 đã bắt đầu với
[kiểm chứng vận tốc và dấu lái](STAGE4_G1_TELEMETRY_VI.md); chưa có mô hình
Stage 4 đã huấn luyện. Các JSON audit giữ trạng thái chờ
xem xét tại thời điểm tạo; quyết định mới nằm ở biên bản và kế hoạch.
Số liệu lái thuộc tác nhân Stage 1 lịch sử; các kiểm thử mới riêng
cho việc Việt hóa được ghi tại `runs/s4_vi_01/`, không thay kết quả lịch sử trên.

## Mã trạng thái giữ nguyên để phần mềm đọc

| Mã | Nghĩa tiếng Việt |
|---|---|
| INITIALIZING | Đang khởi tạo |
| RUNNING | Đang chạy |
| COMPLETE / COMPLETED | Đã thực thi xong; chưa đồng nghĩa cổng được nghiệm thu |
| FAILED | Thất bại |
| INTERRUPTED | Bị ngắt |
| PASS / PASSED | Phép kiểm tra đạt |
| AWAITING_REVIEW | Chờ xem xét |
| APPROVED | Đã nghiệm thu cổng trong phạm vi quyết định |
| READY_TO_START | Đủ điều kiện chuyển sang cổng tiếp theo; chưa triển khai |
| NOT_OPEN | Chưa mở cổng tiếp theo |
| NOT_RUN | Chưa chạy phép kiểm tra |
| UNREVIEWED | Chưa xem xét chất lượng |
| PROPOSED_NOT_APPROVED | Đề xuất chưa được phê duyệt |
| EXACT | Khớp mã băm byte gốc |
| LF_CONVERSION_MATCH_ONLY | Chỉ khớp khi đổi xuống dòng CRLF sang LF; kiểm tra byte gốc chưa đạt |
| MISMATCH / MISSING | Không khớp / bị thiếu |
| SUCCESS / TIMEOUT | Đến đích an toàn / hết thời gian |
| TRAIN / FIT / DEV | Tập phát triển; phần dùng học; phần giữ lại nội bộ |
| VALIDATION / TEST | Tập chọn mô hình / tập đánh giá cuối giữ kín |
| Easy / Medium / Hard / Extreme | Dễ / trung bình / khó / cực khó |

## Các vấn đề còn mở

- Đã chấp nhận cách ghi nhận 12 file chỉ khớp sau chuẩn hóa xuống dòng; vẫn chưa khớp byte gốc và giữ nguyên lịch sử, theo biên bản nghiệm thu G0.
- Ngưỡng chất lượng và toàn bộ nhãn đoạn TIMEOUT vẫn chưa được phê duyệt.
- Phạm vi bằng chứng của tác nhân hướng dẫn còn hạn chế, gồm ba tình huống Extreme hết thời gian.
- Tên trường đo vận tốc không khớp; cần sửa và kiểm chứng riêng ở bước sau.
- Mô tả dấu góc lái mâu thuẫn; giữ nguyên hành động số.
- Chưa triển khai ràng buộc bản lưu mô hình với lần chạy.
- Kiểm định AgentContract 1.3 và điều kiện tham gia đánh giá chuẩn còn chờ.
- Cần chạy thử quy mô nhỏ để xác định môi trường PyTorch và ngân sách tính toán trước huấn luyện.

## Phân phối hành vi đã ghi

| Mã | Diễn giải | Số bước |
|---|---|---:|
| CRUISE | Chạy ổn định | 5711 |
| FOLLOW | Bám theo xe trước | 3724 |
| YIELD | Nhường đường | 3121 |
| APPROACH_JUNCTION | Tiếp cận giao lộ | 436 |
| COMMIT_JUNCTION | Cam kết đi qua giao lộ | 879 |
| TRAVERSE_JUNCTION | Đang qua giao lộ | 4674 |
| RECOVER_LANE | Phục hồi vị trí trong làn | 510 |
| EXIT_JUNCTION | Ra khỏi giao lộ | 1235 |
| EMERGENCY_BRAKE | Phanh khẩn cấp | 526 |
| STOP_FOR_OBSTACLE | Dừng vì vật cản | 17 |
| PREPARE_CHANGE | Chuẩn bị đổi làn | 79 |
| CHANGE_LANE | Đang đổi làn | 213 |
| BLOCKED_CHANGE | Đổi làn bị chặn | 14 |
| BLOCKED | Bị chặn | 1089 |

## Đọc các tệp bằng chứng gốc

- [Tổng hợp số đo và giới hạn](../results/audits/stage4/g0/attempt_01/audit_summary.json) — `audit_summary.json`.
- [Chỉ mục nguồn gốc và mã băm](../results/audits/stage4/g0/attempt_01/evidence_index.json) — `evidence_index.json`.
- [Kiểm tra bản chụp lịch sử](../results/audits/stage4/g0/attempt_01/historical_snapshot_check.json) — `historical_snapshot_check.json`.
- [Nguồn gốc từng lượt và tình huống TRAIN](../results/audits/stage4/g0/attempt_01/natural_train_evidence.json) — `natural_train_evidence.json`.
- [Hàng đợi xem xét chất lượng TIMEOUT](../results/audits/stage4/g0/attempt_01/quality_review_queue.json) — `quality_review_queue.json`.
- [Đề xuất chia 144 FIT / 36 DEV](../results/audits/stage4/g0/attempt_01/proposed_fit_dev.json) — `proposed_fit_dev.json`.
- [Phát lại chính sách hiện tại](../results/audits/stage4/g0/attempt_01/live_replay.json) — `live_replay.json`.
- [Toàn vẹn lượt kiểm tra nhanh Gate 7](../results/audits/stage4/g0/attempt_01/gate7_smoke_integrity.json) — `gate7_smoke_integrity.json`.
- [Danh mục khả năng tích hợp và vấn đề đã phát hiện](../results/audits/stage4/g0/attempt_01/integration_inventory.json) — `integration_inventory.json`.
- [Mã băm mã nguồn tại lần kiểm định](../results/audits/stage4/g0/attempt_01/source_inventory.json) — `source_inventory.json`.

Mã SHA-256 của bản tổng hợp gốc: `3813e187534eccf5bfa150d5597a52c3103d6e8a7b9cf07fa6ba95f3437ad951`.

Tài liệu này chỉ đổi cách trình bày. Khóa dữ liệu, tên tệp/API, lệnh và mã trạng
thái giữ nguyên để không làm hỏng khả năng chạy lại hoặc kiểm chứng bằng chứng.
