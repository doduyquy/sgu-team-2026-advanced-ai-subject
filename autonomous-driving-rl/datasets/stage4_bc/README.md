# Bộ dữ liệu BC Stage 4 — Hồ sơ thiết kế

**Trạng thái: đã có dữ liệu thô pilot; BC-A/BC-B chưa xây dựng hoặc nghiệm thu.**

G1 bước 2 đã có [sổ đăng ký dữ liệu thô thử nhỏ](pilot_raw_v1.json): 5 lượt
TRAIN, 4 hình học, 5,744 cặp đã kiểm tra nguồn gốc/ghép bước và phát lại.
Mọi mẫu `UNREVIEWED`, chưa nhận vào BC-A/BC-B. Đã chốt
[144 FIT / 36 DEV theo hình học](fit_dev_v1/manifest.json); pilot được
phân phía qua [ánh xạ có băm](fit_dev_v1/pilot_assignments.json), giữ nguyên sổ cũ.
Xem [báo cáo thu thử](../../docs/STAGE4_G1_PILOT_VI.md).

MapSuite V1 giữ vai trò bộ tình huống đánh giá đã đóng băng. Thư mục này mô tả
một bộ dữ liệu hành vi mẫu riêng, sẽ được lấy hoàn toàn từ TRAIN.

- **BC-A:** quỹ đạo SUCCESS đã kiểm định chất lượng.
- **BC-B:** đúng phần BC-A đó, bổ sung từng đoạn TIMEOUT được xem xét riêng.
- Giữ mọi kết quả gốc, lần thử thất bại và khoảng bị loại hoặc chưa xác định.
- SUCCESS chưa tự chứng nhận chất lượng; TIMEOUT chưa đủ kết luận hành động kém.
  Các phép thử vật lý có can thiệp được quản lý riêng.

Xem [cấu trúc và quy trình kiểm định chất lượng](SCHEMA.md),
[báo cáo G0](../../docs/STAGE4_G0_TECHNICAL_AUDIT_VI.md) và
[kế hoạch nghiên cứu](../../experiments/stage4/plan_v1.json).

`quality_review_queue.json` của G0 là hàng đợi UNREVIEWED, chưa phải dữ liệu
huấn luyện BC. Ranh giới hành vi chỉ giúp xem lại vết, chưa cắt hoặc chấp nhận
đoạn học. Ngưỡng, trọng số và tỷ lệ lấy mẫu vẫn để null.

Dữ liệu dung lượng lớn sau này đặt trong `artifacts/stage4/` hoặc `runs/` đã được
Git bỏ qua; tệp kê khai có phiên bản và mã băm nằm ở đây. Các tài liệu trong Git
không đồng nghĩa các quỹ đạo gốc cục bộ đã được phân phối.
