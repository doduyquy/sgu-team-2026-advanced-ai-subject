# G1 bước 6 — Bộ dữ liệu thử BC-A và BC-B

Hai bộ dữ liệu được dựng từ đúng sổ duyệt chất lượng v1.0.0 ở bước 5.
Không thu thêm bản đồ, đổi quy tắc, fit chuẩn hóa hoặc huấn luyện trong bước này.

## Các con số có đơn vị

**Cách chia toàn bộ kho bản đồ giữ nguyên: 180 bản đồ TRAIN = 144 FIT + 36 DEV.**
Đây là phân chia hình học, chưa phải số bản đồ đã thu đủ dữ liệu. Nguồn thử
đang có 7 lượt chạy trên 6 bản đồ, tổng 9.347 bước; chỉ phần được duyệt mới
được đưa vào hai bộ dưới đây.

| Bộ dữ liệu thử | Phía | Số mẫu quan sát–hành động | Số bản đồ có mẫu | Số lượt có mẫu |
|---|---|---:|---:|---:|
| BC-A | FIT | 566 | 1 | 1 |
| BC-A | DEV | 832 | 1 | 2 |
| BC-B | FIT | 666 | 2 | 2 |
| BC-B | DEV | 1.332 | 2 | 3 |

Mỗi mẫu là đúng một cặp `agent_input(t) → canonical_action(t)`.
BC-A có tổng 1.398 mẫu, BC-B có tổng 1.998 mẫu; **không cộng hai tổng này
thành số mẫu duy nhất**, vì BC-B chứa nguyên vẹn BC-A. Phần BC-B thêm là
600 mẫu TIMEOUT: 100 FIT và 500 DEV. DEV phục vụ đánh giá, không dùng cập
nhật trọng số hoặc fit chuẩn hóa.

## Đã lưu những gì, ở đâu?

- [Manifest và thống kê chung](../datasets/stage4_bc/bc_pilot_v1/manifest.json).
- [Manifest BC-A](../datasets/stage4_bc/bc_pilot_v1/BC-A/manifest.json) và
  [báo cáo chất lượng BC-A](../datasets/stage4_bc/bc_pilot_v1/BC-A/quality_report.json).
- [Manifest BC-B](../datasets/stage4_bc/bc_pilot_v1/BC-B/manifest.json) và
  [báo cáo chất lượng BC-B](../datasets/stage4_bc/bc_pilot_v1/BC-B/quality_report.json).
- [Định nghĩa dựng bộ thử](../experiments/stage4/g1_dataset_pilot_v1.json),
  [sổ quyết định nguồn](../results/audits/stage4/g1/quality_v1/review.json) và
  [quy tắc đã đóng băng](../results/audits/stage4/g1/quality_rules_v1/freeze.json).

**Dữ liệu đầy đủ để đọc/huấn luyện:** `artifacts/stage4/bc_pilot_v1/BC-A/`
và `artifacts/stage4/bc_pilot_v1/BC-B/`. Thư mục artifacts được Git bỏ qua;
chỉ clone kho Git chưa có các quan sát đầy đủ. Phải giữ bản dữ liệu đầy đủ
và kiểm tra mã băm trước khi phân phối hoặc đưa vào gói huấn luyện.

Mỗi thư mục đầy đủ chứa:

| File | Vai trò |
|---|---|
| `frames_FIT.jsonl.gz`, `frames_DEV.jsonl.gz` | Quan sát công khai trước hành động, được khử trùng lặp theo bước trong lượt. Không chứa nhãn hành động, kết quả hoặc chẩn đoán teacher. |
| `samples_FIT.jsonl.gz`, `samples_DEV.jsonl.gz` | ID lưu trữ, nhãn hành động float32 thực tế đã gửi vào simulator và tham chiếu lịch sử 10 khung. |
| `sample_index.jsonl.gz` | Danh sách từng mẫu: lượt nguồn, bước, bản đồ/phía, mã băm vết/chuyển bước và mã băm nội dung quan sát/nhãn. Đây là metadata, không phải đặc trưng mô hình. |
| `provenance.json` | Khoảng được chọn, tình huống/seed/horizon/giao thông, nguồn/cấu hình teacher và môi trường runtime gốc. |
| `quality_report.json`, `input_contract.json` | Kết quả giữ/loại, độ phủ, giới hạn và hợp đồng đọc. |
| `manifest.json`, `CHECKSUMS.sha256` | Phiên bản, số lượng và mã băm mọi tệp dữ liệu. |

Trong `datasets/stage4_bc/bc_pilot_v1/` chỉ xuất bản metadata, danh sách mẫu
và checksum bản đầy đủ. Các file checksum ở đó còn tham chiếu các shard quan
sát/hành động trong artifacts; không dùng thư mục metadata làm bộ dữ liệu
đầy đủ cho `verify` hoặc trainer.

## Lịch sử quan sát và ngăn rò rỉ

Định dạng hỗ trợ một khung hoặc mười khung; chưa quyết định kiến trúc học.
Lịch sử lấy `[t-9, ..., t]`, không có tương lai, đặt lại ở mỗi lượt. Các khung
trước bước 0 là `None` và có mặt nạ. Không ghép lịch sử từ cuối đoạn này sang
đầu đoạn khác khi giữa chúng có khoảng bị loại.

BC-B FIT có 675 khung quan sát nhưng chỉ 666 mẫu; BC-B DEV có 1.341 khung
nhưng chỉ 1.332 mẫu. **Chín khung thêm ở mỗi phía chỉ làm ngữ cảnh**, không
có thêm nhãn được nhận. Chúng đến từ bước 1891–1899 của TIMEOUT FIT và
791–799 của TIMEOUT DEV. BC-A không có khung chỉ làm ngữ cảnh.

Bộ đọc chỉ trả quan sát công khai, mặt nạ lịch sử và nhãn hành động. Các ID
bản đồ/lượt, kết quả, quy tắc chất lượng, reward và chẩn đoán teacher nằm
ngoài đầu vào học. ID làn/nhánh còn trong bản sao quan sát để khớp nguồn;
encoder ở bước huấn luyện phải bỏ chúng khỏi đặc trưng số.

## Cách đọc và tái dựng

Đọc bằng Python, chưa cần MetaDrive/NumPy/PyTorch:

```python
from src.datasets.stage4_bc import iter_examples, verify_dataset

directory = "artifacts/stage4/bc_pilot_v1/BC-A"
verify_dataset(directory)
example = next(iter_examples(directory, "FIT", history_frames=1))
# example: observations, history_mask, canonical_action
```

Tái dựng từ vết nguồn đầy đủ, chạy trong môi trường MetaDrive đã cố định:

```powershell
python -B -X utf8 scripts/build_stage4_dataset.py build --run-id <new-build-id> --output artifacts/stage4/<new-dataset-dir>
python -B -X utf8 scripts/build_stage4_dataset.py publish --run-id <new-build-id> --output datasets/stage4_bc/<new-registry-dir>
python -B -X utf8 scripts/build_stage4_dataset.py verify --dataset artifacts/stage4/<new-dataset-dir>/BC-A
```

Mỗi đầu ra phải mới. Dữ liệu gốc vẫn bất biến; lỗi dựng giữ nguyên hồ sơ
FAILED. Gzip dùng thời gian cố định và JSON chuẩn hóa để các shard dữ liệu
có thể tái tạo cùng byte; timestamp của manifest ghi lần dựng thực tế.

## Kiểm chứng hoàn tất ngày 14/11/2026

- `s4_g1_dataset_build_01`: COMPLETE, dựng đủ hai bộ, tái kiểm tra ghép
  bước trên các lượt nguồn được dùng và xác minh mọi checksum.
- `s4_g1_dataset_checks_01`: **536/536 backend đạt**, gồm 24 kiểm thử mới
  về chọn đoạn, nhãn float32, hash, rò rỉ, lịch sử và bảo toàn lần lỗi.
  Không sửa GUI, không chạy lại nhóm Qt trong bước này.
- `s4_g1_dataset_portability_01`: bản sao ở thư mục độc lập đọc đủ mẫu cho
  lịch sử 1/10 bằng `python -I -S`, không nạp MetaDrive, NumPy, PyTorch hoặc
  GUI. Kiểm tra chặn mọi truy cập dữ liệu ra ngoài thư mục bản sao.
- `s4_g1_dataset_finalcheck_01`: mọi quan sát và nhãn trong cả hai bộ khớp
  vết nguồn; BC-B chứa đúng BC-A và chỉ thêm TIMEOUT đã duyệt. Hai bộ đầy
  đủ chiếm tổng dung lượng tệp khoảng **3,05 MiB**, chưa gồm module huấn luyện.
- Dữ liệu thô, quy tắc chất lượng, split và nguồn teacher giữ nguyên byte.
  Các trạng thái `bc_datasets_built=false` ở báo cáo bước 5 mô tả mốc cũ;
  trạng thái mới nằm ở [hồ sơ hoàn tất G1](../datasets/stage4_bc/g1_completion_v1.json).

Manifest được tạo trước kiểm chứng cuối còn ghi `g1_complete=false` tại thời
điểm dựng. Giữ nguyên bản đó để bảo toàn mã băm; trạng thái hoàn tất G1 hiện
tại lấy từ `g1_completion_v1.json` và kế hoạch, không lấy từ snapshot lúc dựng.

[Báo cáo kiểm chứng máy đọc](../results/audits/stage4/g1/dataset_pilot_v1_checks.json)
ghi mã băm, nguồn kiểm thử, kiểm tra tính độc lập và thống kê mỗi phía.
**Bước 6 và yêu cầu chuẩn bị bộ thử của G1 đã hoàn tất.** G2 chưa triển khai;
không ghi giả rằng chủ nhiệm đã ký duyệt hay rằng đã thu đủ 180 bản đồ.

## Giới hạn và việc tiếp theo

BC-A FIT chỉ phủ một bản đồ Medium. BC-B FIT thêm 100 bước từ một bản đồ
Extreme. Đây là bộ thử để kiểm tra luồng dữ liệu và huấn luyện thử, chưa đại
diện 144 FIT hay đủ để tuyên bố Agent 4 tổng quát tốt. V1 chưa có đoạn chờ
đứng yên riêng được duyệt; sàng lọc có thể loại chuyển làn hợp lệ. Không sửa
ngưỡng bằng DEV để tăng số mẫu.

Tiếp theo là thiết kế/triển khai module huấn luyện dùng chung encoder/model
với Agent 4, kiểm tra module rồi đóng gói lên Kaggle. Bước này chưa tạo model,
checkpoint, fit chuẩn hóa, upload Kaggle hoặc thu quy mô lớn trên 180 bản đồ.
