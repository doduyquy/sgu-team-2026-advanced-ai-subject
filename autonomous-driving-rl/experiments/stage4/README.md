# Stage 4 — Nghiên cứu học bắt chước hành vi

**G0 đã nghiệm thu ngày 13/11/2026; G1 đã hoàn tất chuẩn bị bộ thử, G2 chưa mở.**
Xem [báo cáo BC-A/BC-B](../../docs/STAGE4_G1_DATASETS_VI.md),
[định nghĩa dựng dữ liệu](g1_dataset_pilot_v1.json) và
[hồ sơ hoàn tất G1](../../datasets/stage4_bc/g1_completion_v1.json).
**Bước 3 đã đo tài nguyên; môi trường huấn luyện NOT_READY**, xem
[báo cáo tài nguyên](../../docs/STAGE4_G1_RESOURCES_VI.md) và
[định nghĩa đo](g1_resources_v1.json).
**Bước 4 đã chốt 144 FIT / 36 DEV và đích huấn luyện Kaggle** theo
[quyết định người dùng](g1_decision_fit_dev_kaggle_v1.json), xem
[manifest](../../datasets/stage4_bc/fit_dev_v1/manifest.json) và
[báo cáo/việc tiếp theo](../../docs/STAGE4_G1_SPLIT_KAGGLE_VI.md).
**Bước 5 có quy tắc chất lượng v1 cho pilot**: xem
[định nghĩa nhóm FIT khó](g1_quality_calibration_v1.json),
[tham số](g1_quality_rules_v1.json) và [báo cáo](../../docs/STAGE4_G1_QUALITY_VI.md).
Quy tắc đóng băng bằng FIT rồi áp dụng lên DEV; bước 6 đã xây BC-A/BC-B.
Xem [vận tốc và dấu lái trên TRAIN](../../docs/STAGE4_G1_TELEMETRY_VI.md),
định nghĩa [kiểm chứng bước 1](g1_telemetry_v1.json).
Đã [thu và kiểm tra dữ liệu thô bước 2](../../docs/STAGE4_G1_PILOT_VI.md) theo
[định nghĩa pilot](g1_pilot_v1.json): 5 lượt, 4 hình học, 5,744 cặp;
sổ thu gốc giữ UNREVIEWED, lớp kiểm định mới được ghi riêng ở bước 5.
Xem [biên bản nghiệm thu và giới hạn](../../docs/STAGE4_G0_REVIEW_VI.md).
Đã có bộ dữ liệu thử được kiểm định; chưa có chính sách học được,
vòng huấn luyện hoặc đăng ký đánh giá chuẩn cho Stage 4.

- [Kế hoạch nghiên cứu](plan_v1.json): thiết kế có phiên bản và điều kiện nghiệm thu từng cổng.
- [Định nghĩa kiểm định](g0_audit_v1.json): các bằng chứng TRAIN lịch sử cần đối chiếu.
- [Báo cáo kỹ thuật và thiết kế](../../docs/STAGE4_G0_TECHNICAL_AUDIT_VI.md).
- [Nhật ký phát triển](../../docs/STAGE4_CHANGELOG.md).
- [Thiết kế bộ dữ liệu](../../datasets/stage4_bc/SCHEMA.md).
- [Kết quả và giải thích mã trạng thái bằng tiếng Việt](../../docs/STAGE4_KET_QUA_VI.md).

Dùng môi trường Python MetaDrive đã cố định phiên bản cho Platform V1. Kiểm định
bằng chứng G0 cần NumPy và các phụ thuộc hiện có, chưa cần PyTorch. Phần kiểm
định dữ liệu không khởi tạo mô phỏng; tiến trình kiểm thử hồi quy phần lõi có thể
khởi tạo môi trường mẫu để kiểm tra hợp đồng, không thu dữ liệu nghiên cứu hoặc
đánh giá mô hình Stage 4. Chạy từ `autonomous-driving-rl/`:

```powershell
python -B -X utf8 scripts/stage4.py audit --run-id s4_g0_lan_thu_moi
python -B -X utf8 scripts/stage4.py publish-audit --run-id s4_g0_lan_thu_moi --output results/audits/stage4/g0/lan_thu_moi
```

Mã thoát 0 nghĩa là đã thực thi các phép kiểm tra. **COMPLETE chưa tự nghiệm thu
cổng:** `gate_approved` trong mỗi hồ sơ audit vẫn là false; quyết định nghiệm thu
G0 được ghi riêng trong biên bản và kế hoạch, không sửa báo cáo lịch sử. Thiếu/hỏng bằng chứng hoặc kiểm
thử không đạt tạo FAILED, mã thoát 1; ngắt bằng bàn phím tạo INTERRUPTED, mã 130.
Mỗi lần thử dùng thư mục mới; giữ cả lần thất bại. Không chấp nhận ID hoặc đường
dẫn xuất đã tồn tại. `--skip-tests` ghi NOT_RUN và chặn xuất báo cáo.

Lệnh mặc định dùng chương trình kiểm thử hồi quy phần lõi hiện có trong tiến
trình riêng. Thay đổi G0 chưa cần kiểm thử giao diện Qt; thay đổi hệ thực thi hoặc
giao diện ở các cổng sau phải chạy nhóm kiểm thử riêng thích hợp.

## Dữ liệu gốc và khả năng tái lập

MapSuite trong Git chứa định nghĩa tình huống, chưa chứa dữ liệu lái mẫu. G0 đọc
`runs/rule_based_v060_accept04_*`, các vết phát lại vật lý và
`runs/rule_based_v061_cleanup_smoke_20261031`. Các thư mục này được Git bỏ qua.
Chỉ sao chép kho Git chưa đủ để tái lập toàn bộ kiểm định. Cần giữ hoặc khôi phục
bản gốc có mã băm đã xác minh; không tạo dữ liệu thay cho vết bị thiếu. Lệnh sẽ
ghi nhận thất bại nếu không có đầu vào bắt buộc.

Mỗi lần kiểm định lưu bản chụp mã nguồn/cấu hình, bản chụp tác nhân hướng dẫn
ứng viên riêng, mã băm đầu vào, tình huống, kết quả phát lại, nhật ký kiểm thử,
danh sách tệp được bảo vệ trước/sau và hồ sơ toàn vẹn SHA-256. Bản chụp ứng viên
ghi lại byte hiện tại, chưa thay thế bản phát hành cũ hoặc cho phép thu dữ liệu.

`publish-audit` kiểm tra toàn vẹn lần chạy trước khi sao chép báo cáo JSON tổng
hợp vào nơi mới. Bản xuất không kèm đường dẫn máy cá nhân và toàn bộ nhật ký gốc.
Bản ghi gốc vẫn có thẩm quyền và cần được bảo quản riêng. Không ghi đè bằng chứng.

## Chuyển cổng nghiệm thu

G0 đã được chấp nhận theo biên bản ngày 13/11/2026. G1 đã hoàn tất ở phạm vi pilot;
bước 1 đã kiểm chứng vận tốc và dấu lái; bước 2 đã thu thử nhỏ và kiểm tra ghép
bước/nguồn gốc. Bước 3 đã đo tài nguyên và ghi vấn đề môi trường. Bước 6
đã xây và kiểm tra BC-A/BC-B từ các chú thích chất lượng;
FIT/DEV đã đóng băng. Đích huấn luyện là Kaggle sau chuẩn bị module, runtime
Kaggle và runtime suy luận local cần được kiểm tra trước các cổng sử dụng.
Huấn luyện, tích hợp bản lưu mô hình, mô hình có lịch sử, DAgger và đánh giá chuẩn
thuộc các cổng sau. G0 giữ nguyên đăng ký, hợp đồng nền tảng và mã tác nhân cũ.
