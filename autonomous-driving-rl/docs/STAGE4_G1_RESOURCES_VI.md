# G1 bước 3 — Tài nguyên thu dữ liệu và môi trường huấn luyện

Ngày thực hiện: **14/11/2026 (Asia/Saigon)**. Phạm vi: đo thời gian, dung
lượng tệp, RAM thu dữ liệu thô và kiểm tra môi trường tensor. Không huấn luyện,
không cập nhật trọng số hoặc cài phụ thuộc; G2 chưa mở.

## Trạng thái

**Đã hoàn tất đo và kiểm tra bước 3**, lần `s4_g1_resources_01` COMPLETE.
Nhóm đo có 5 lượt, 4 hình học, **5,744 bước, 4 SUCCESS / 1 TIMEOUT**, khớp
case/số bước/kết thúc của pilot gốc. Môi trường tensor **NOT_READY**: chưa cài
PyTorch và `pip check` báo phụ thuộc thiếu/xung đột. Việc hoàn tất phép đo
không mở G2 hoặc chứng nhận môi trường huấn luyện đã sẵn sàng.

| Chỉ số thực đo | Kết quả |
|---|---:|
| Thời gian toàn workload | **863.88 giây — 14 phút 23.88 giây** |
| Số mẫu RAM | 6,393 |
| Đỉnh tổng working set cây collector | **315.08 MiB** |
| Đỉnh working set riêng của bộ đo | 119.95 MiB, không cộng vào dòng trên |
| RAM khả dụng toàn máy thấp nhất | **147.74 MiB** |
| Tổng dung lượng hồ sơ collector | **28.04 MiB** — 29,406,223 byte |
| Số tệp collector | 1,021, gồm tệp danh mục toàn vẹn |
| Khoảng mẫu trung bình / lớn nhất | 0.135 / 0.428 giây |
| Lỗi không đọc được tiến trình do quyền | 0 |

RAM khả dụng rất thấp cho thấy máy có áp lực bộ nhớ trong lần đo. RSS là
working set lấy mẫu dưới điều kiện đó, không phải cam kết pipeline chỉ cần
315 MiB RAM hoặc máy còn đủ RAM để huấn luyện. Các trang có thể được chia sẻ
hoặc bị OS chuyển ra khỏi working set. Chưa đo memory commit/VRAM huấn luyện.

Thời gian case, gồm thu và kiểm định/phát lại:

| Case | Giây | Bước | Kết thúc |
|---|---:|---:|---|
| Easy / SCS / seed 0 / env 0 | 29.92 | 418 | SUCCESS |
| Easy / SCS / seed 0 / env 1 | 23.91 | 414 | SUCCESS |
| Medium / SCXCS / seed 14 / env 0 | 18.87 | 566 | SUCCESS |
| Hard / SCTXrCS / seed 2 / env 0 | 107.18 | 1,432 | SUCCESS |
| Extreme / SCXOCrTYCS / seed 0 / env 0 | 625.64 | 2,914 | TIMEOUT |

Dựng/kết thúc batch ngoài các case: 58.37 giây. Pilot bước 2 trước đây ghi
447.95 giây cho collector; lần đo này collector tự ghi 858.09 giây, bộ đo
bao quanh tiến trình ghi 863.88 giây. Không quy chênh lệch này riêng cho giám
sát vì tải máy/cache/áp lực bộ nhớ giữa hai lần khác nhau.

- [Chỉ mục và mã băm hồ sơ đo](../results/audits/stage4/g1/resources_v1/evidence_index.json).
- [Xác minh cuối, kiểm thử và chi tiết phụ thuộc](../results/audits/stage4/g1/resources_v1_verification.json).
- [Số đo, phân loại byte và toàn bộ phép ngoại suy](../results/audits/stage4/g1/resources_v1/resource_summary.json).

Dung lượng từng nhóm: vết quyết định 14.88 MiB; vết sau bước 4.42 MiB;
bản đồ 1.20 MiB; nguồn/expert chung 5.91 MiB; nguồn sao chép từng lượt
0.92 MiB; phần còn lại là chỉ mục, console và metadata. Đây là byte tệp
logic, chưa gồm kích thước môi trường Python, assets ngoài run, dữ liệu mã
hóa BC sau này hoặc checkpoint.

Ngoại suy theo **45 hình học TRAIN mỗi tier**, chạy tuần tự trên cùng máy:

| Kịch bản | Seed môi trường / hình học | Lượt | Thời gian ước lượng | Hồ sơ ước lượng |
|---|---:|---:|---:|---:|
| Độ dài giống nhóm đo | 1 | 180 | 9.75 giờ | 0.93 GiB |
| Mọi lượt hết horizon | 1 | 180 | 11.51 giờ | 1.19 GiB |
| Độ dài giống nhóm đo | 2 | 360 | 19.48 giờ | 1.85 GiB |
| Mọi lượt hết horizon | 2 | 360 | 23.01 giờ | 2.37 GiB |

Đây là hai kịch bản xấp xỉ, **không phải khoảng tin cậy, cận trên hoặc chi phí
huấn luyện**. Chúng dựa trên một lần đo 4/12 họ ở điều kiện máy đang có tải.
Không suy ra được chi phí/thời gian/VRAM học model từ các con số trên.

## Cấu hình và phạm vi đo

Định nghĩa: [`g1_resources_v1.json`](../experiments/stage4/g1_resources_v1.json).
Công cụ: `scripts/profile_stage4_resources.py`; phép tính trong
`src/imitation/resources.py`.

Chạy lại đúng nhóm 5 lượt/4 hình học TRAIN ở bước 2, trong thư mục mới
`runs/s4_g1_resources_01_collect/`. Giữ Stage 1 0.6.1, agent seed 42,
geometry/environment seed, traffic, horizon, render OFF và W&B DISABLED.
Mỗi lượt vẫn qua `ExperimentExecutor` và được kiểm tra ghép bước/phát lại.
Pilot chính `runs/s4_g1_pilot_02/` cùng sổ đăng ký giữ nguyên, được kiểm băm
trước/sau lần đo. Dữ liệu của lần đo chưa được thêm vào BC.

Kiểm thử hồi quy được chạy **trước** phép đo để không tranh CPU với workload.
Các tiến trình khác của máy vẫn có thể hoạt động; không đóng ứng dụng người dùng.

- **Thời gian toàn workload:** đồng hồ monotonic bao quanh tiến trình collector,
  gồm khởi động, nạp nguồn/bản đồ, mô phỏng, ghi log, kiểm định/phát lại và kết thúc.
- **Thời gian từng case:** từ sự kiện stdout `collecting` đến `verified`, gồm
  cả khởi động worker và kiểm định ngoại tuyến. Có sai số quan sát theo nhịp lấy mẫu.
  Điều khiển 10 Hz là tần số theo thời gian mô phỏng; thời gian thực đo riêng.
- **RAM:** nghỉ 0.1 giây giữa các vòng đọc, cộng RSS/working set của collector và toàn bộ
  tiến trình con đang sống. Loại trùng theo PID và thời điểm tạo; RAM của bộ
  đo được ghi riêng. Khoảng giữa hai mẫu còn gồm chi phí đọc/lập lịch OS;
  ghi cả RAM khả dụng toàn hệ thống và khoảng lấy mẫu thực tế.
- **Dung lượng:** số byte nội dung tệp thực trên đĩa, tách vết quyết định,
  vết sau bước, bản đồ, chỉ mục, console và mã nguồn/siêu dữ liệu. Không nhầm
  tổng hồ sơ với dung lượng riêng các cặp quan sát–hành động.

RSS tổng có thể cộng lặp các trang bộ nhớ được chia sẻ giữa tiến trình.
Đỉnh lấy mẫu có thể bỏ lỡ đột biến ngắn giữa hai lần đọc. Đây là số đo working
set của pipeline, không phải RAM/VRAM huấn luyện hoặc số đo private bytes.
RAM khả dụng toàn hệ thống còn chịu ảnh hưởng ứng dụng khác.

## Phần cứng và điều kiện máy

| Thành phần | Ghi nhận |
|---|---|
| CPU | Intel Core i5-12450H, 8 nhân vật lý / 12 luồng |
| RAM hệ thống | Tổng OS nhận 16,869,056,512 byte, khoảng 15.71 GiB |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4,096 MiB VRAM |
| NVIDIA driver | 610.78 |
| Python | 3.10.22, môi trường MetaDrive hiện tại |
| MetaDrive | 0.4.3, commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db` |
| NumPy / Panda3D / psutil | 2.2.6 / 1.10.16 / 7.1.3 |

Khi kiểm tra trước workload, CPU toàn máy ở **84.2%**, RAM khả dụng khoảng
**1.97 GiB**, dung lượng trống ổ dự án khoảng **19.76 GiB**. Đây là điều kiện
máy đang có tải; số đo thời gian cần được hiểu cùng điều kiện này.
Chênh lệch với thời gian pilot cũ chịu ảnh hưởng tải máy, cache và giám sát;
chưa có đối chứng đủ để tách riêng overhead của bộ đo.
Snapshot GPU trước chạy có 0 MiB VRAM dùng theo `nvidia-smi`; không suy ra GPU
không dùng bộ nhớ trong toàn workload hoặc PyTorch CUDA đã hoạt động.

## Kiểm tra môi trường huấn luyện

- **PyTorch chưa được cài** trong môi trường Python được chọn. Phép kiểm tensor
  CPU/CUDA không chạy; chưa đo tốc độ học, VRAM học hoặc tạo model Stage 4.
- `pip check` mã thoát **1**: thiếu các phụ thuộc như `cffi`, `uritemplate`,
  `pyasn1-modules`, `anyio`, `httpx`, `sniffio`, `pywin32`, `jinja2` ở những gói
  đang được cài; đồng thời `protobuf 7.36.2` không khớp yêu cầu dưới 6 của
  `google-ai-generativelanguage` và `grpcio-status` được báo tại máy.
- Thu dữ liệu MetaDrive được kiểm tra độc lập bằng workload thực và kiểm thử
  nền tảng. Các lỗi môi trường trên chưa được tự sửa hoặc quy thành lỗi lái.

Trước G2 cần chuẩn bị môi trường PyTorch có phiên bản/phụ thuộc được chốt,
kiểm tra tensor CPU, CUDA nếu sử dụng, và môi trường không còn lỗi phụ thuộc
liên quan. Thấy GPU qua driver chưa chứng minh model có thể dùng GPU.

## Cách ngoại suy chi phí

Lấy **số hình học TRAIN thực tế của từng mức khó** từ manifest đóng băng,
không lấy trung bình chung của 5 lượt vì Easy được lặp hai seed.

Với mỗi tier, tính trung bình thời gian case, số bước và byte hồ sơ case.
Chi phí dựng hồ sơ nguồn chung được tách riêng và tính một lần mỗi batch.
Sổ chỉ mục mẫu được tính theo số bước. Giữ cách lưu bản đồ và bản sao nguồn
riêng từng lượt như collector hiện tại; chưa giả định cơ chế khử trùng lặp.

Hai nhóm kịch bản cho 1 và 2 seed môi trường/hình học:

1. **Độ dài quan sát tương tự nhóm đo:** dùng số bước trung bình theo tier.
2. **Tất cả lượt chạy hết horizon:** tính horizon từ chiều dài từng hình học
   TRAIN bằng hàm nền tảng, rồi nhân chi phí/bước đo được theo tier.

Đây là **kịch bản lập ngân sách**, không phải khoảng tin cậy hoặc cận trên
bảo đảm. Chỉ 4 trong 12 họ đã được đo; các tương tác giao thông, kết quả an
toàn và tải máy khác có thể đổi chi phí. Kịch bản hết horizon vẫn dùng chi
phí/bước đã chứa khởi động/kiểm định của nhóm nhỏ nên chỉ là xấp xỉ.

Không nhân RAM đỉnh với số lượng hình học khi chạy tuần tự. Nhóm 5 lượt cũng
chưa chứng minh việc giữ bộ nhớ ổn định trong một lần chạy dài 180 hình học.
Không dùng số đo thu dữ liệu để kết luận RAM/VRAM hoặc thời gian huấn luyện.

## Kiểm thử và tái lập

`runs/s4_g1_resource_checks_01/`: **486/486 kiểm thử backend đạt**, gồm 17 kiểm
thử mới về đỉnh RAM đồng thời, PID trùng/tái sử dụng, lỗi đọc tài nguyên,
phân loại dung lượng, trọng số tier, seed, horizon, mốc thời gian, trạng thái
lỗi và môi trường thiếu PyTorch. GUI và chính sách không thay đổi.

Chạy từ `autonomous-driving-rl/` bằng môi trường MetaDrive hiện có:

```powershell
python -B -X utf8 scripts/profile_stage4_resources.py profile --run-id s4_g1_resources_new
python -B -X utf8 scripts/profile_stage4_resources.py publish --run-id s4_g1_resources_new --output results/audits/stage4/g1/resources_new
```

Mỗi lần dùng ID mới, sinh một hồ sơ đo và một hồ sơ collector riêng có hậu tố
`_collect`. Bản đo lưu mẫu RAM gốc nén, sự kiện/console, kiểm tra phần cứng,
`pip check`/`pip freeze`, nguồn/cấu hình có băm và danh mục toàn vẹn. Collector
giữ đủ quan sát/hành động, bản đồ, seed và mọi kết quả như bước 2.

G1 vẫn IN_PROGRESS. Sau bước đo đã [chốt chia hình học và đích Kaggle](STAGE4_G1_SPLIT_KAGGLE_VI.md).
Việc còn lại là chốt quy tắc chất lượng và xây BC-A/BC-B. Mọi dữ liệu vẫn
UNREVIEWED, số mẫu nhận vào BC là 0.
