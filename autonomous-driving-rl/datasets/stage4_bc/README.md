# Bộ dữ liệu BC Stage 4 — Hồ sơ thiết kế

**Trạng thái: BC-A/BC-B thử v1 đã xây dựng và kiểm định; chưa huấn luyện.**

Xem [manifest bộ thử](bc_pilot_v1/manifest.json),
[hồ sơ hoàn tất G1 ở phạm vi pilot](g1_completion_v1.json) và
[cách đọc dữ liệu/đơn vị đo](../../docs/STAGE4_G1_DATASETS_VI.md).
BC-A có 566 mẫu FIT / 832 mẫu DEV; BC-B có 666 mẫu FIT / 1.332 mẫu DEV.
Đây là số mẫu quan sát–hành động. Phân chia bản đồ vẫn là 144 FIT / 36 DEV
trên 180 TRAIN; nguồn thử mới có 6 bản đồ, chưa thu đủ 180.

G1 bước 2 đã có [sổ đăng ký dữ liệu thô thử nhỏ](pilot_raw_v1.json): 5 lượt
TRAIN, 4 hình học, 5,744 cặp đã kiểm tra nguồn gốc/ghép bước và phát lại.
Sổ thu gốc giữ trạng thái `UNREVIEWED`; tại thời điểm thu chưa nhận vào BC-A/BC-B. Đã chốt
[144 FIT / 36 DEV theo hình học](fit_dev_v1/manifest.json); pilot được
phân phía qua [ánh xạ có băm](fit_dev_v1/pilot_assignments.json), giữ nguyên sổ cũ.
Xem [báo cáo thu thử](../../docs/STAGE4_G1_PILOT_VI.md).

Bước 5 bổ sung hai lượt Extreme FIT (3,603 bước, một TIMEOUT và một va chạm),
giữ mọi kết quả. [Quy tắc chất lượng v1 và quyết định giữ/loại](../../docs/STAGE4_G1_QUALITY_VI.md)
là lớp chú thích mới; không ghi đè sổ dữ liệu thô. Chỉ các mẫu đủ điều kiện
theo sổ này mới được bộ xây dữ liệu ở bước 6 sử dụng.
[Sổ chất lượng v1 có mã băm](quality_v1.json) trỏ tới quy tắc đóng băng,
quyết định và báo cáo kiểm chứng; các ngưỡng này chỉ được chốt cho pilot.

MapSuite V1 giữ vai trò bộ tình huống đánh giá đã đóng băng. Thư mục này mô tả
một bộ dữ liệu hành vi mẫu riêng, lấy hoàn toàn từ TRAIN.

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
đoạn học. Ngưỡng pilot được ghi riêng trong quy tắc chất lượng v1; trọng số
và tỷ lệ lấy mẫu vẫn để null.

Dữ liệu dung lượng lớn sau này đặt trong `artifacts/stage4/` hoặc `runs/` đã được
Git bỏ qua; tệp kê khai có phiên bản và mã băm nằm ở đây. Các tài liệu trong Git
không đồng nghĩa các quỹ đạo gốc cục bộ đã được phân phối.
