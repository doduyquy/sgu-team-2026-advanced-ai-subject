# Biên bản nghiệm thu Stage 4 — S4-G0

- Mã quyết định: `S4-G0-REVIEW-20261113`.
- Ngày nghiệm thu: **13/11/2026**, múi giờ `Asia/Saigon`.
- Người nghiệm thu: **Đăng Khoa**, theo yêu cầu trong phiên làm việc:
  “cập nhập nghiệm thu đi, cho tôi biết cập nhập ở đâu”.
- Quyết định: **APPROVED — nghiệm thu G0 với các giới hạn được ghi nhận**.
- Cổng tiếp theo: **S4-G1 / READY_TO_START**, chưa triển khai hoặc chạy.

## 1. Phạm vi quyết định

Chấp nhận công cụ kiểm định ngoại tuyến, bằng chứng, thiết kế BC và phương án
bản chụp tác nhân hướng dẫn của G0. Quyết định dựa trên hồ sơ lịch sử và các
phép đối chiếu chỉ đọc trong phiên xem xét. Lần cập nhật nghiệm thu không chạy
lại mô phỏng, phát lại chính sách, kiểm thử hồi quy hoặc huấn luyện.

Quyết định này cho phép chuyển kế hoạch sang G1. Chưa nghiệm thu bộ dữ liệu
BC-A/BC-B, bản phát hành tác nhân hướng dẫn để thu quy mô lớn, mô hình học được
hoặc điều kiện tham gia VALIDATION/TEST. G2–G5 vẫn chưa mở.

## 2. Kết quả đã xem xét

| Nội dung | Kết quả và kết luận |
|---|---|
| Bản chụp lịch sử Stage 1 v0.6.1 | Đối chiếu lại 52 file: 40 `EXACT`, 12 `LF_CONVERSION_MATCH_ONLY`; không có `MISSING`/`MISMATCH`. |
| Xử lý 12 file lệch xuống dòng | Chấp nhận ghi nhận giới hạn; chỉ khớp sau CRLF → LF, chưa khớp byte gốc. Giữ nguyên bản chụp và mã băm lịch sử; không đổi nhãn thành `EXACT`. |
| Định danh ứng viên | `s4_g0_audit_02:stage1_rule_based@0.6.1`, bản chụp tại `runs/s4_g0_audit_02/candidate_expert/`. |
| Toàn vẹn ứng viên | 11/11 file nguồn khớp `agent_provenance.json`; 13/13 file ứng viên khớp hồ sơ toàn vẹn của lần audit. Mã băm provenance và liên kết với chỉ mục bằng chứng đã công bố khớp. |
| Cấu hình và nguồn hiện tại | Descriptor, nội dung cấu hình và mã băm cấu hình khớp bản bảo trì 0.6.1; 11/11 mục nguồn tương ứng trong `src/` khớp ứng viên tại thời điểm xem xét. |
| Phát lại lịch sử | Hồ sơ `live_replay.json` ghi `PASS`, 1.992 bước, không lệch quyết định/hành động; không phải điểm lái mới hoặc chứng minh toàn TRAIN. |
| Chạy thử lịch sử | Hồ sơ Gate 7 ghi TRAIN Easy/SCS/geom0/env0/agent42, SUCCESS sau 418 bước; lỗi ghi vận tốc 0 vẫn là vấn đề mở. |
| Kiểm thử lịch sử | Phụ lục xác minh G0 ghi 447/447 kiểm thử phần lõi đạt, gồm 24 kiểm thử Stage 4; lần Việt hóa ghi 24 kiểm thử đạt. Không chạy lại khi nghiệm thu. |
| Phạm vi dữ liệu | 16 lượt, 12 hình học TRAIN, 22.228 quyết định, 13 SUCCESS và 3 TIMEOUT. Đây là bằng chứng phát triển hạn chế, chưa đại diện 180 hình học TRAIN. |
| Chất lượng dữ liệu | 118 khoảng TIMEOUT giữ `UNREVIEWED`, chưa có đoạn được chấp nhận vào BC. SUCCESS cũng cần kiểm định chất lượng trước khi dùng học. |

Ứng viên được chấp nhận làm cơ sở chuẩn bị G1 với phạm vi bằng chứng trên.
Việc chấp nhận ở G0 chưa thay thế kiểm chứng vận hành, quy tắc chất lượng và
đóng băng phiên bản trước thu dữ liệu quy mô lớn ở G1.

## 3. Bằng chứng và mã băm

Các đường dẫn dưới đây tính từ `autonomous-driving-rl/`. Mã SHA-256 được đối
chiếu khi xem xét; các tệp gốc được giữ nguyên.

| Đường dẫn | SHA-256 |
|---|---|
| `results/audits/stage4/g0/attempt_01/audit_summary.json` | `3813e187534eccf5bfa150d5597a52c3103d6e8a7b9cf07fa6ba95f3437ad951` |
| `results/audits/stage4/g0/attempt_01/evidence_index.json` | `78bce824fa5cf8bbaf67ce55223bb1471f566dc9c82813493c1035fcfa30727f` |
| `results/audits/stage4/g0/attempt_01/historical_snapshot_check.json` | `6f215240064a431b00be582502aaf18601dca2429e49f0b4687d2b77fa0d9f61` |
| `results/audits/stage4/g0/final_verification.json` | `6aaf7f53c53481f7a5b841a98244897787a5827ae80207f55ee038dd8fadc74f` |
| `runs/s4_g0_audit_02/candidate_expert/agent_provenance.json` | `5c2bfe52a4de42221a0f8318ed98bbe7a3312cdc0ea19680a9100960a6d501b4` |

Mã băm cấu hình ứng viên:
`61a28354a10054a5351de7a7d45b6a5efcb6af21ef581a5ed33d14f36bb1489f`.
Các dữ liệu gốc trong `runs/` được Git bỏ qua và phải được bảo quản riêng để
tái lập; biên bản không thay thế hoặc phân phối các dữ liệu đó.

## 4. Việc chuyển tiếp

| Việc cần thực hiện | Cổng và điều kiện |
|---|---|
| Kiểm chứng trường vận tốc `speed_kmh`/`speed_km_h` và mô tả dấu lái trên TRAIN; giữ nguyên nhãn hành động số lịch sử | G1, trước dùng các trường này để đánh giá chất lượng hoặc thu quy mô lớn; thay đổi runtime phải có phiên bản và bằng chứng mới. |
| Đo tài nguyên thử, xác định môi trường phụ thuộc; chốt ứng viên hướng dẫn và cách thu có nguồn gốc | G1, chạy thử nhỏ trước mở rộng. |
| Xem xét và đóng băng phân chia 144 FIT / 36 DEV, tiêu chí chất lượng, nhãn TIMEOUT, ngưỡng/trọng số/tỷ lệ; xây BC-A/BC-B | G1; các mục này chưa được phê duyệt bởi nghiệm thu G0. |
| Mở rộng dữ liệu TRAIN, hướng tới đủ 180 hình học và ghi nhận mọi kết quả | Các bước tiếp theo sau thử nhỏ; số lượt chạy không tự bảo đảm chất lượng dữ liệu học. |
| Chuẩn hóa, huấn luyện thử và liên kết bản lưu với dữ liệu/mã nguồn; tích hợp chính sách độc lập | G2–G3 sau nghiệm thu cổng tương ứng. |
| P5 của tác nhân hướng dẫn, AgentContract 1.3 và điều kiện đánh giá chuẩn | Vẫn PENDING; phải hoàn tất các kiểm định cần thiết trước mở đánh giá chuẩn ở G5. |

## 5. Nơi phản ánh trạng thái hiện tại

[Kế hoạch nghiên cứu](../experiments/stage4/plan_v1.json) ghi G0 `APPROVED`,
G1 `READY_TO_START` và trỏ về biên bản này. [Changelog](STAGE4_CHANGELOG.md)
ghi lại việc nghiệm thu. Các báo cáo JSON và trạng thái trong những lần chạy
lịch sử vẫn giữ `AWAITING_REVIEW`/`gate_approved=false` đúng thời điểm tạo;
quyết định mới không sửa ngược các bằng chứng đó.
