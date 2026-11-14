# G1 bước 4 — Chốt FIT/DEV và đích huấn luyện Kaggle

Ngày ghi quyết định: **14/11/2026 (Asia/Saigon)**. Người dùng xác nhận sau
khi chuẩn bị dữ liệu sẽ đóng gói module để huấn luyện trên Kaggle và nhận
checkpoint; 180 hình học TRAIN chia **144 FIT / 36 DEV**.
[Bản ghi quyết định](../experiments/stage4/g1_decision_fit_dev_kaggle_v1.json)
giữ nguyên lời xác nhận và tách phạm vi chốt khỏi nghiệm thu G1 hoặc huấn luyện.

## Cách chia đã cố định

Mỗi họ có 15 hình học TRAIN. Xếp theo
`SHA256(UTF8('stage4-fit-dev-v1|' + geometry_id))`, phân xử khi trùng bằng
`geometry_id`: ba hình học đầu DEV, mười hai còn lại FIT. Tổng 12 họ:
144 FIT / 36 DEV, mỗi tier 36 FIT / 9 DEV.

Giữ **chính xác danh sách đề xuất G0**, không đổi sau khi thấy SUCCESS/TIMEOUT.
Kiểm chứng đầy đủ 180 hình học TRAIN, ID/hash duy nhất, 0 hình học giao nhau
giữa FIT và DEV, không chứa hình học VALIDATION/TEST. Tất cả seed môi trường,
lượt, đoạn và lịch sử của cùng hình học phải đi cùng một phía.

- [Manifest đóng băng](../datasets/stage4_bc/fit_dev_v1/manifest.json): trạng thái
  `APPROVED_FROZEN`, nguồn phân chia chính thức, thuật toán và SHA-256 các tệp.
- [Danh sách 180 hình học](../datasets/stage4_bc/fit_dev_v1/geometry_assignments.csv).
- [Ánh xạ các lượt pilot](../datasets/stage4_bc/fit_dev_v1/pilot_assignments.json).
- [Bằng chứng và kiểm chứng](../results/audits/stage4/g1/partition_v1/evidence_index.json).
- [Kiểm tra cuối và 24 kiểm thử liên quan đạt](../results/audits/stage4/g1/partition_v1_checks.json).

Chuẩn hóa chỉ fit bằng phần tử FIT hợp lệ. Định danh hình học, partition,
kết quả và nhãn chất lượng là metadata; không đưa vào đặc trưng chính sách.
Tỷ lệ 80/20 theo hình học không có nghĩa số bước quan sát cũng chia 80/20.

Đây là danh sách cho 180 hình học có sẵn trong MapSuite, **chưa thu dữ liệu
đủ 180 hình học**. Pilot hiện vẫn chỉ có 4 hình học/5 lượt.

## Pilot hiện thuộc những phía nào?

| Pilot | Phía | Số bước | Kết thúc |
|---|---|---:|---|
| Easy SCS seed 0 / env 0 | DEV | 418 | SUCCESS |
| Easy SCS seed 0 / env 1 | DEV | 414 | SUCCESS |
| Medium SCXCS seed 14 / env 0 | FIT | 566 | SUCCESS |
| Hard SCTXrCS seed 2 / env 0 | FIT | 1,432 | SUCCESS |
| Extreme SCXOCrTYCS seed 0 / env 0 | DEV | 2,914 | TIMEOUT |

FIT: 2 hình học/2 lượt/1,998 bước. DEV: 2 hình học/3 lượt/3,746 bước.
Hai seed của Easy cùng DEV. Pilot FIT chưa có TIMEOUT; do đó cần bổ sung
nhóm FIT khó trước khi hiệu chỉnh quy tắc chất lượng đoạn TIMEOUT.
Không chuyển Extreme sang FIT để thuận tiện và không lấy DEV để đặt ngưỡng.

Các lượt DEV đã xuất hiện trong kiểm chứng kỹ thuật; danh sách vẫn giống G0
đã đề xuất trước pilot. Chưa fit trọng số, chuẩn hóa hoặc ngưỡng chất lượng.
Mọi mẫu vẫn UNREVIEWED, 0 mẫu nhận vào BC.

Giữ nguyên byte sổ đăng ký/log pilot cũ, kể cả trường UNASSIGNED đã ghi tại
thời điểm thu. `pilot_assignments.json` là lớp ánh xạ mới có băm, gắn với
đúng mã băm sổ đăng ký cũ. Bộ xây dữ liệu sau này cần dùng manifest đóng
băng này để phân phía, thay vì sửa ngầm bản ghi lịch sử.

## Đích thực thi đã chốt

```text
Local: thu Stage 1 → kiểm định → xây bộ BC và gói dữ liệu
Kaggle: module huấn luyện độc lập + đánh giá DEV → checkpoint đầy đủ
Local: Agent 4 nạp checkpoint → chạy độc lập → đánh giá theo giao thức
```

Trainer cần dùng chung encoder/model với Agent 4, hoạt động độc lập với
MetaDrive/Panda3D/GUI và có lưu/tiếp tục checkpoint. Checkpoint kèm cấu hình,
chuẩn hóa FIT, schema, phiên bản nguồn/môi trường, seed và băm dữ liệu/split.
Notebook là lớp gọi module; chưa đóng gói hoặc tải dữ liệu lên Kaggle.

Môi trường Kaggle phải được kiểm tra thực tế khi sang G2; chưa có session
hoặc GPU được xác minh. Môi trường MetaDrive local hiện thiếu PyTorch theo
bước 3; runtime nạp model local vẫn phải được chuẩn bị trước khi Agent 4 chạy.
Không cần mặc định huấn luyện GPU ngay trên máy local.

## Bước tiếp theo

**Bước 5: chốt quy tắc chất lượng.** Trước hết bổ sung một nhóm FIT nhỏ có
tình huống khó, định nghĩa case trước chạy và giữ mọi kết quả. Dùng FIT để
khảo sát/hiệu chỉnh tiêu chí; xem SUCCESS về chất lượng điều khiển và tách
TIMEOUT thành đoạn tiến triển hợp lệ, nhường/chờ có lý do, bế tắc/lặp hoặc
chưa rõ. Ghi bằng chứng, lý do và phiên bản quy tắc; không mặc định SUCCESS tốt.

Sau khi đóng băng quy tắc, áp dụng nhất quán lên DEV. Tiếp theo bước 6 tạo
BC-A/BC-B có chỉ mục và băm. Khi G1 hoàn tất mới triển khai/kiểm tra trainer
Kaggle ở G2. Thu quy mô lớn hướng 180 hình học sau khi thử huấn luyện nhỏ và
chạy độc lập đạt yêu cầu như kế hoạch ban đầu.

Hồ sơ chốt metadata: `runs/s4_g1_partition_freeze_01/`, có nguồn đầu vào,
mã băm trước/sau, quyết định, danh sách/overlay và kiểm chứng. Không chạy
mô phỏng, cập nhật trọng số hoặc thay chính sách Stage 1 trong bước này.
