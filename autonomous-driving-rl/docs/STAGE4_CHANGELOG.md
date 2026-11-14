# Nhật ký nghiên cứu Stage 4

## 2026-11-14 — G1 bước 4: chốt FIT/DEV và huấn luyện Kaggle

Theo xác nhận trực tiếp của người dùng, ghi quyết định
`S4-G1-FITDEV-KAGGLE-20261114`: sau chuẩn bị dữ liệu sẽ đóng gói trainer lên
Kaggle, nhận checkpoint; **180 TRAIN chia 144 FIT / 36 DEV** theo hình học.
G1 vẫn IN_PROGRESS, G2–G5 NOT_OPEN; chưa đóng gói/upload/huấn luyện.

- Đóng băng `datasets/stage4_bc/fit_dev_v1/manifest.json` và danh sách hình
  học, giữ nguyên SHA256 ranking/danh sách đề xuất G0. Mỗi họ 12 FIT/3 DEV,
  tổng 180 hình học duy nhất, 0 giao nhau, không chứa VALIDATION/TEST.
- Mọi environment seed/lượt/đoạn/lịch sử cùng hình học ở cùng phía;
  chuẩn hóa chỉ fit FIT. Bổ sung quy tắc Git giữ byte cho các tệp có mã băm.
- Pilot hiện có FIT: 2 lượt/2 hình học/1,998 bước, đều SUCCESS;
  DEV: 3 lượt/2 hình học/3,746 bước, 2 SUCCESS/1 TIMEOUT. Hai seed Easy cùng DEV.
  Giữ nguyên sổ/log pilot cũ; ánh xạ có băm mới trỏ đúng nguồn, 0 mẫu được duyệt BC.
- Cần bổ sung FIT khó để hiệu chỉnh quy tắc chất lượng TIMEOUT ở bước 5;
  không đổi split theo kết quả hoặc dùng DEV để đặt ngưỡng. Bước 6 xây BC-A/BC-B.
- Ghi đích thực thi: local thu/kiểm định và chạy kín; Kaggle trainer/DEV
  ngoại tuyến; checkpoint mang encoder/chuẩn hóa/cấu hình/nguồn gốc dùng chung.
  Kaggle runtime chưa kiểm tra; runtime nạp model local vẫn cần chuẩn bị.
- `s4_g1_partition_freeze_01` COMPLETE: chốt metadata, không có lượt lái hoặc
  cập nhật trọng số. Đối chiếu nguồn/bằng chứng được bảo vệ trước/sau không đổi.
- `s4_g1_partition_checks_01`: kiểm tra băm/CSV/kế hoạch và **24/24 kiểm thử
  liên quan đạt**, log/nguồn lệnh/hồ sơ toàn vẹn lưu riêng; không đổi chính sách,
  phiên bản Stage 1 0.6.1 hoặc hợp đồng nền tảng.

Xem [báo cáo và việc tiếp theo](STAGE4_G1_SPLIT_KAGGLE_VI.md),
[chỉ mục bằng chứng](../results/audits/stage4/g1/partition_v1/evidence_index.json).
Việc chốt danh sách không đồng nghĩa đã thu đủ 180 hình học; pilot vẫn chỉ
4 hình học, chất lượng UNREVIEWED và G1 chưa nghiệm thu.

## 2026-11-14 — G1 bước 3: đo tài nguyên và kiểm tra môi trường

Hoàn tất [phép đo và báo cáo](STAGE4_G1_RESOURCES_VI.md), công cụ/schema
resources 1.0.0. G1 vẫn IN_PROGRESS, bước 3 ghi
`MEASURED_WITH_TRAINING_ENV_FINDINGS`; G2–G5 NOT_OPEN. Không thay Stage 1 0.6.1,
cấu hình/đăng ký, giao diện hoặc hợp đồng nền tảng; không huấn luyện/cài phụ thuộc.

- Thêm `experiments/stage4/g1_resources_v1.json`,
  `scripts/profile_stage4_resources.py profile/publish`,
  `src/imitation/resources.py`, 17 kiểm thử mới và báo cáo tiếng Việt.
- `s4_g1_resource_checks_01`: **486/486 backend đạt**; chạy xong trước workload
  để không tranh CPU trong lúc đo. GUI không thay đổi, không chạy lại nhóm GUI.
- `s4_g1_resources_01` COMPLETE, workload `s4_g1_resources_01_collect`
  COMPLETE: lặp nguyên 5 case TRAIN, 5,744 bước, 4 SUCCESS/1 TIMEOUT; khớp
  case/số bước/kết thúc với pilot gốc, tiếp tục kiểm định ghép bước/phát lại.
- Thực đo **863.88 giây**, gồm khởi động/thu/ghi log/kiểm định; **28.04 MiB**
  hồ sơ collector; 6,393 mẫu RAM, đỉnh tổng RSS/working set **315.08 MiB**,
  bộ đo riêng có đỉnh 119.95 MiB. Khoảng mẫu trung bình 0.135s, lớn nhất
  0.428s; 0 lỗi quyền đọc, 2 quan sát tiến trình biến mất khi thoát.
- RAM toàn hệ thống khả dụng có lúc chỉ **147.74 MiB**. Trước workload,
  CPU toàn máy 84.2%, RAM khả dụng 1.97 GiB; thời gian là đo trong điều kiện
  có tải/áp lực bộ nhớ, chưa tách riêng overhead giám sát. RSS không phải
  cam kết dung lượng RAM đủ cho pipeline hoặc huấn luyện.
- Ngoại suy theo 45 geometry TRAIN/tier, chạy tuần tự: 180 lượt/1 env seed
  khoảng **9.75h / 0.93 GiB** nếu độ dài giống nhóm đo; kịch bản hết horizon
  **11.51h / 1.19 GiB**. Với 2 seed/geometry, 360 lượt khoảng
  **19.48h / 1.85 GiB** hoặc **23.01h / 2.37 GiB** tương ứng.
  Chỉ đo 4/12 họ; các giá trị là kịch bản, không phải cận trên/khoảng tin cậy.
- CPU i5-12450H 8 nhân/12 luồng, RAM OS nhận 15.71 GiB, RTX 3050 Laptop
  VRAM 4 GiB. Môi trường Python 3.10.22/MetaDrive 0.4.3 **chưa cài PyTorch**;
  `pip check` mã 1 với 11 dòng thiếu/xung đột phụ thuộc. `NOT_READY` cho môi
  trường tensor; không thực hiện tensor smoke, không cập nhật trọng số.
- Giữ nguyên byte pilot bước 2 và sổ đăng ký; hồ sơ đo có snapshot nguồn,
  RAM từng mẫu, sự kiện/console, phần cứng, pip check/freeze, dung lượng theo
  loại và SHA-256. Hồ sơ collector mới giữ đủ nguồn/map/seed/quan sát/hành động.
- [Chỉ mục bằng chứng](../results/audits/stage4/g1/resources_v1/evidence_index.json),
  [xác minh cuối và chi tiết phụ thuộc](../results/audits/stage4/g1/resources_v1_verification.json).
  Đã tính lại thống kê RAM/phase/disk/ngoại suy từ tệp gốc, kiểm manifest TRAIN
  đóng băng, source/tests và băm log kiểm thử, liên kết các hồ sơ toàn vẹn.

Giới hạn: một lần đo trên nhóm nhỏ và máy có tải, chưa đo training time/VRAM,
khả năng dùng CUDA của PyTorch hoặc độ ổn định bộ nhớ cho 180 geometry liên tục.
Cần chuẩn bị môi trường PyTorch/phụ thuộc trước G2. G1 còn chốt FIT/DEV,
quy tắc chất lượng và BC-A/BC-B; 0 mẫu BC được nhận, chưa mở thu lớn.

## 2026-11-14 — G1 bước 2: thu thử dữ liệu thô

Hoàn tất [thu thử và kiểm tra nguồn gốc/ghép bước](STAGE4_G1_PILOT_VI.md).
Kế hoạch ghi `G1_IN_PROGRESS_STEP2_RAW_PILOT_VERIFIED`; G1 vẫn IN_PROGRESS,
G2–G5 NOT_OPEN. Bộ thu/schema pilot 1.0.0; Stage 1 giữ 0.6.1, cấu hình mặc
định, adapter, đăng ký và nền tảng không đổi. Chưa phát hành chính sách BC.

- Thêm định nghĩa `experiments/stage4/g1_pilot_v1.json`, lệnh
  `scripts/collect_stage4_pilot.py collect/publish`, bộ kiểm
  `src/imitation/pilot.py`, 17 kiểm thử mới và sổ đăng ký `pilot_raw_v1.json`.
- Thu 5 lượt/4 hình học TRAIN: Easy SCS seed 0 env 0/1, Medium SCXCS seed 14
  env 0, Hard SCTXrCS seed 2 env 0, Extreme SCXOCrTYCS seed 0 env 0. Agent seed
  42, render OFF, W&B DISABLED, giữ nguyên traffic/horizon, không can thiệp.
- `s4_g1_pilot_01` FAILED: lớp Python trong cấu hình runtime chưa có mã hóa
  JSON; 5 tiến trình thất bại trước bước vật lý đầu tiên. Giữ 0 bước dữ liệu
  hợp lệ, nguồn/log/trạng thái/mã băm. Sửa bộ ghi để lưu tên import lớp/hàm và
  từ chối đối tượng không biết; không sửa byte của lần thử cũ.
- `s4_g1_pilot_02` COMPLETE: **5,744 bước**, **4 SUCCESS / 1 TIMEOUT**, không
  lỗi kỹ thuật. Các lượt lần lượt 418/414/566/1,432/2,914 bước. TIMEOUT giữ
  nguyên toàn bộ horizon, tiến độ cuối khoảng 45.15%, chưa xét chất lượng.
- Mỗi lượt lưu source/config/runtime, case/seed, cấu hình thực tế/bản đồ,
  quan sát–quyết định–hành động, trạng thái trước/sau và toàn bộ kết quả bước,
  sự kiện/console/outcome, chỉ mục mẫu và danh mục toàn vẹn. Metadata riêng
  phục vụ kiểm tra, không thêm đặc trưng vào đầu vào chính sách.
- 5,744 bước ghép đúng với hành động gửi vào simulator và phát lại khớp Stage 1;
  tính lại cờ kết thúc/chỉ số/reward nền tảng; 572 sự kiện tiến độ đủ và đúng.
  Ba trường hợp đối chứng khớp chính xác 2,416 bước với vết bước 1 khi bỏ độ trễ.
- Kiểm trước/sau: 130 tệp nguồn/định nghĩa và 182 tệp thuộc nhóm được bảo vệ
  giữ nguyên. `publish` xác minh 1,018 tệp thô theo mã băm trước khi xuất.
- `s4_g1_pilot_checks_01`: 467 backend đạt trước bản sửa cấu hình cuối;
  `s4_g1_pilot_checks_02`: **469/469 backend đạt**, gồm 17 kiểm thử mới. GUI
  không đổi, không chạy lại nhóm GUI cho thay đổi bộ thu/kiểm định này.
- [Bằng chứng](../results/audits/stage4/g1/pilot_v1/evidence_index.json),
  [xác minh cuối và hồ sơ lần lỗi](../results/audits/stage4/g1/pilot_v1_verification.json),
  [sổ đăng ký thô](../datasets/stage4_bc/pilot_raw_v1.json); vết đầy đủ giữ trong `runs/`.

Giới hạn: chỉ 4/180 hình học, một seed ở ba họ và hai seed ở Easy. Mọi mẫu
UNREVIEWED, 0 mẫu nhận vào BC, chưa gán FIT/DEV, chưa lọc SUCCESS/TIMEOUT.
Bước tiếp theo là đo tài nguyên/kiểm tra môi trường huấn luyện (bước 3), rồi
chốt chia hình học, quy tắc chất lượng và BC-A/BC-B; chưa chạy G2.

## 2026-11-13 — G1 bước 1: kiểm chứng vận tốc và dấu lái

Hoàn tất [kiểm chứng trên ba hình học TRAIN](STAGE4_G1_TELEMETRY_VI.md).
G1 chuyển `IN_PROGRESS`, bước 1 `VERIFIED_ON_THREE_TRAIN_GEOMETRIES`;
G1 chưa nghiệm thu, G2–G5 chưa mở. Công cụ telemetry schema 1.0.0, không phát
hành chính sách mới; Stage 1 giữ 0.6.1 và cùng cấu hình/đăng ký.

- Lưu nguồn trước sửa tại `runs/s4_g1_telemetry_before_03/source_snapshot/`
  cùng SHA-256. Launcher tái hiện vận tốc 0 sai trên ba lượt SUCCESS trong
  khi thân cứng đang chuyển động. Nguồn trước sửa và lịch sử giữ nguyên byte.
- Sửa launcher đọc `speed_km_h`, giữ tên trường log; thiếu/không hữu hạn/âm
  thành lỗi kỹ thuật. Sửa mô tả dấu lái: đi tới thì dương trái, âm phải;
  giữ nguyên ánh xạ số và hợp đồng JSON. Không đổi thuật toán tác nhân.
- Định nghĩa `experiments/stage4/g1_telemetry_v1.json`, công cụ
  `scripts/audit_stage4_telemetry.py`, 9 phép thử lái và 3 lượt Stage 1 mỗi phía.
  Mỗi lượt có nguồn/cấu hình/runtime, case/seed, bản đồ, số đo trước/sau,
  vết quan sát–hành động, kết quả/trạng thái và danh mục toàn vẹn.
- `s4_g1_telemetry_after_01` COMPLETE: 2,416 bước tự nhiên, 234 bước thử lái;
  9/9 phép thử dấu đạt, ba CSV và 240 mẫu tiến độ khớp vận tốc vật lý với
  dung sai 1e-5 km/h. Quan sát/quyết định/hành động trước/sau giống hệt trên
  2,416 bước khi bỏ độ trễ. Toàn bộ nguồn/bản phát hành Stage 1, MapSuite,
  hợp đồng JSON nền tảng và bằng chứng G0 giữ nguyên theo kiểm tra băm.
- `s4_g1_telemetry_checks_01`: 576/576 kiểm thử đạt (452 backend, 124 GUI),
  gồm 5 kiểm thử mới. Không ghi nhận hồi quy hành động ở ba tình huống kiểm.
- Giữ `before_01` FAILED do tuần tự hóa Config và `before_02` FAILED do
  xung lái 12 bước kết thúc sớm; bản cuối dùng 6 bước. Import lỗi và sandbox
  từ chối trước khi tạo hồ sơ được phân biệt với các lần mô phỏng đã ghi.
- [Chỉ mục bằng chứng](../results/audits/stage4/g1/telemetry_v1/evidence_index.json)
  giữ băm nguồn, tổng hợp trước/sau, đối chiếu vết, kiểm thử và lần lỗi.
  Vết đầy đủ nằm trong `runs/`, không kèm toàn bộ trong Git.

Giới hạn: ba hình học Easy/Medium/Hard, một seed môi trường; phép thử dấu tại
đoạn thẳng xuất phát, chuyển động tới. Chưa kiểm chứng toàn bộ TRAIN hoặc xe
lùi. Dữ liệu chẩn đoán chưa đủ điều kiện BC dù kết thúc SUCCESS. Các bước
thu thử nhỏ, đo tài nguyên, FIT/DEV và quy tắc chất lượng vẫn chưa thực hiện;
không huấn luyện. Cột vận tốc launcher cũ cần xem nguồn riêng, không sửa ngầm
và không kết luận đường ghi của `run_rule_based_experiment.py` cũng sai.

## 2026-11-13 — Nghiệm thu S4-G0

Theo yêu cầu của Đăng Khoa, ghi [biên bản nghiệm thu](STAGE4_G0_REVIEW_VI.md),
quyết định `S4-G0-REVIEW-20261113`: **G0 APPROVED; G1 READY_TO_START**, chưa
triển khai hoặc chạy G1. `gate_approved=true` trong kế hoạch chỉ áp dụng G0;
G2–G5 vẫn NOT_OPEN. Thuật toán, cấu hình, phiên bản tác nhân và đăng ký launcher
không thay đổi; công cụ kiểm định vẫn 0.0.2, chưa phát hành chính sách BC.

Đối chiếu chỉ đọc xác nhận 40/52 file lịch sử EXACT, 12 chỉ khớp sau đổi LF;
11/11 nguồn ứng viên và 13/13 file ứng viên khớp hồ sơ toàn vẹn, cấu hình khớp
0.6.1 và 11/11 nguồn hiện tại khớp ứng viên. Giữ giới hạn 16 lượt/12 hình học,
13 SUCCESS/3 TIMEOUT; 118 khoảng TIMEOUT chưa duyệt vào BC. Các vấn đề vận tốc,
dấu lái, chất lượng/độ phủ, môi trường huấn luyện và điều kiện benchmark chuyển
sang các cổng tiếp theo như biên bản. Không chạy lại mô phỏng, replay, huấn luyện
hoặc kiểm thử; các kết quả 447 kiểm thử G0 và 24 kiểm thử Việt hóa là lịch sử.

Cập nhật kế hoạch, báo cáo và các trang chỉ dẫn để phản ánh quyết định mới.
Giữ nguyên JSON audit, bản chụp và log lịch sử, kể cả trạng thái chờ xem xét
được ghi tại thời điểm chạy. Các mục bên dưới mô tả thời điểm trước nghiệm thu.

## Công cụ kiểm định 0.0.2 — Việt hóa

Việt hóa tài liệu, chú thích, mô tả cấu hình, trợ giúp và thông báo lỗi của phần
Stage 4. Giữ tên API, khóa JSON, mã trạng thái, lệnh, đường dẫn và định danh thí
nghiệm. Bằng chứng đã đóng băng vẫn nguyên byte; bản trình bày tiếng Việt nằm ở
[kết quả G0](STAGE4_KET_QUA_VI.md). Bản chụp trước sửa, mã băm, kiểm thử và nhật
ký lần Việt hóa nằm tại `runs/s4_vi_01/`. Chưa mở G1 hoặc thay đổi thuật toán.

## Công cụ kiểm định 0.0.1 — S4-G0

Trạng thái: **bằng chứng và thiết kế G0 đang chờ xem xét; G1 chưa mở**. Chưa có
bản phát hành chính sách học được. Nhánh/commit nền: `khoa` / `9a36c313`.

### Giả thuyết và phần đã triển khai

Vết quyết định công khai của tác nhân theo luật có thể phục vụ nghiên cứu BC
tái lập được nếu kiểm định nguồn gốc và ghép quan sát/hành động, đồng thời tách
việc chấp nhận chất lượng khỏi kết quả cuối lượt. Vì vậy G0 triển khai kiểm tra
ngoại tuyến và hồ sơ bền vững; chưa tạo mạng, bộ thu dữ liệu hoặc bộ lọc TIMEOUT
tự động.

- Thêm `src/imitation/audit.py` và lệnh `scripts/stage4.py audit/publish-audit`.
- Thêm tệp đầu vào lịch sử chính xác, kế hoạch theo cổng, thiết kế chất lượng và
  báo cáo kỹ thuật. Thiết kế các bước sau vẫn cần nghiệm thu cổng trước.
- Tái sử dụng `restore_input`, bộ chuyển hành động, bản chụp nguồn gốc, hàm băm
  theo nội dung và chương trình kiểm thử phần lõi chạy trong tiến trình riêng.
- Giữ tác nhân, hợp đồng nền tảng/bộ đánh giá, bản chụp và dữ liệu lịch sử. G0
  so mã băm các đầu vào được bảo vệ trước/sau.
- Bản chụp ứng viên mới ghi byte hiện tại và tách biệt 12 lệch newline lịch sử;
  chưa là bản phát hành tác nhân hướng dẫn được nghiệm thu.

### Kiểm tra và hồ sơ thí nghiệm

| Lần thử | Mục đích | Kết quả |
|---|---|---|
| `s4_g0_checks_01` | Kiểm thử phần lõi sau triển khai ban đầu | Đạt 445 kiểm thử, gồm 22 kiểm thử G0 mới; yêu cầu chạy, nhật ký và tổng hợp trong `runs/`. |
| `s4_g0_audit_01` | Kiểm định tích hợp đầu tiên | FAILED: bộ kiểm nhầm rằng bản sao phụ Stage 0 và tệp khởi tạo gói phải có trong danh mục P4. Giữ trạng thái, bản chụp ứng viên và hồ sơ lỗi. |
| `s4_g0_audit_02` | Kiểm định tích hợp sau sửa phân loại nguồn | COMPLETE; đạt 446 kiểm thử phần lõi; 308 tệp được bảo vệ giữ nguyên. Xem [chỉ mục bằng chứng](../results/audits/stage4/g0/attempt_01/evidence_index.json) và [tổng hợp](../results/audits/stage4/g0/attempt_01/audit_summary.json). |
| `s4_g0_checks_02` | Kiểm tra cuối sau xử lý lỗi ghi toàn vẹn | Đạt 447 kiểm thử, gồm 24 kiểm thử Stage 4. Xem [phụ lục xác minh](../results/audits/stage4/g0/final_verification.json); mã băm và nhật ký gốc trong `runs/`. |

Lần đầu khởi động `s4_g0_checks_01` bị môi trường thực thi từ chối tạo thư mục
trước khi chương trình ghi hồ sơ. Đã chạy lại với quyền thực thi cần thiết vào
thư mục chưa tồn tại. Dữ liệu lịch sử không đổi. Lịch sử lệnh còn giữ lỗi thiết
lập này; không tạo hồ sơ giả cho giai đoạn chưa khởi tạo được.

Lần kiểm định tích hợp 01 phát hiện lỗi bộ kiểm. Bộ ghi hiện có sao chép mọi mô
đun tác nhân, kể cả Stage 0 và khởi tạo gói. Bản sửa ánh xạ
`agents/rule_based_agent.py` về `policy.py`, các helper về tên tham chiếu P4;
yêu cầu đủ tám phụ thuộc chính; phân loại hai bản sao phụ riêng và vẫn kiểm
đúng byte đã ghi. Kiểm thử bao phủ bí danh, thiếu phụ thuộc, đổi tham chiếu P4 và
sửa byte bản sao phụ. Không sửa nguồn hoặc mã băm trong bằng chứng lịch sử.

Sau lần tích hợp 02, đã gia cố bước kết thúc: phát hiện lỗi ghi toàn vẹn phải
chuyển COMPLETE sang FAILED và dừng. Kiểm thử gây lỗi xác minh chuyển trạng thái;
ổ đĩa không ghi được vẫn chặn xuất báo cáo. Lần kiểm thử cuối xác minh mã sửa
riêng; lần 02 giữ đúng bản chụp nguồn và kết quả 446 kiểm thử ban đầu. Không gán
lại phép kiểm quỹ đạo hoặc điểm lái cho mã nguồn khác.

### Diễn giải và phần còn lại

Dữ liệu đã đối chiếu: 16 tình huống TRAIN tự nhiên, 12 hình học, 22.228 quyết
định; 13 SUCCESS và 3 TIMEOUT. Phát lại chính sách hiện tại bao phủ 1.992 bước,
gồm phép thử có kiểm soát; đây là kiểm định log lịch sử. Mã băm nguồn ở năm lượt
nghiệm thu gốc khớp. Bản chụp 0.6.1 có 12 lệch mã băm byte, chỉ tương đương sau
đổi LF. Chưa đặt ngưỡng/trọng số/tỷ lệ và mọi đoạn TIMEOUT vẫn chưa phê duyệt.

G0 ghi nhận lỗi tên trường vận tốc, mâu thuẫn mô tả dấu lái, thiếu ràng buộc bản
lưu mô hình và thiếu kiểm định đầu vào/điều kiện đánh giá chuẩn. Các sửa chữa sau
cần phiên bản và bằng chứng mới. Thu đủ 180 hình học, huấn luyện BC, lái vòng kín,
thí nghiệm đối chứng, ba seed và đánh giá chuẩn thuộc các cổng sau.
