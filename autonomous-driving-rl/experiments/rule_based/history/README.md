# Lịch sử snapshot phát triển rule-based

Ngày thu gọn: **2026-10-31**, theo phương án người dùng đã duyệt.
Agent vẫn là **0.6.1**; đây là thay đổi cách lưu lịch sử, không phải một phiên bản
thuật toán hay một thí nghiệm lái xe mới.

## Nội dung được đóng gói

[development_history_20261031.zip](development_history_20261031.zip) chứa:

| Nhóm | Snapshot | File gốc |
|---|---:|---:|
| `dev050_01` đến `dev050_11` | 11 | 99 |
| `dev060_01` đến `dev060_09` | 9 | 81 |
| Tổng | **20** | **180** |

- Mỗi snapshot có 9 file: policy, input extractor, các helper/công cụ tương ứng
  và `snapshot.json`. Tổng dung lượng file gốc: **3.007.811 byte**.
- ZIP có **181 entry**: 180 file nguyên bản, giữ đường dẫn tương đối tính từ
  `autonomous-driving-rl/`, và `_archive/manifest.json` được bổ sung cho đợt thu gọn.
- Manifest ghi nhãn, đường dẫn, kích thước và SHA-256 từng file gốc; đồng thời
  giữ danh sách/hash các file được bảo vệ trước khi thu gọn và trạng thái Git lúc đó.
  Đây là metadata của lần đóng gói, không thay thế metadata của thí nghiệm gốc.
- Đã đọc giải nén từng entry và đối chiếu **180/180 file** khớp độ dài/SHA-256 với
  nguồn trước khi bỏ bản rời và 20 thư mục rỗng. Không sửa nội dung snapshot.
- ZIP: **1.008.736 byte**; SHA-256:

```text
3f54990fb2c6f37631e5d8aa00491b52881d513c108b73872768048b95543b1f
```

## Phần được giữ nguyên

- Core, policy hiện hành, tools, tests, configs và datasets.
- Các release `versions/0.3.0/`–`versions/0.6.1/`, gồm toàn bộ
  `0.6.1/maintenance_sources/`; source cũ trong `experiments/legacy_source/`.
- Toàn bộ `results/`, kể cả CSV/report draft, trial, các index và replay trung gian.
- Toàn bộ `runs/`: CSV, trace từng bước, source của từng run, cấu hình/seed,
  trạng thái hoàn thành/lỗi/bị ngắt và console log. `runs/` vẫn bị Git ignore.

Chỉ hai tài liệu có sẵn được cập nhật để giải thích cách lưu và ghi log đợt này:
`experiments/rule_based/README.md` và `docs/RULE_BASED_CHANGELOG.md`.
Không chạy lại test, simulator hoặc experiment theo yêu cầu người dùng.

## Khôi phục một bản dev

Các đường dẫn `versions/dev050_*` và `versions/dev060_*` trong index/report cũ
vẫn là đường dẫn gốc; nội dung lịch sử không bị viết lại. Trước khi dùng những
đường dẫn này hoặc `--baseline-version dev...`, cần khôi phục snapshot tương ứng.
Launcher dùng agent hiện tại và các baseline release không cần bước này.

Ví dụ PowerShell, **đứng tại `autonomous-driving-rl/`**; các lệnh dưới đây là
hướng dẫn cho lần cần phục hồi, không được thực thi trong đợt thu gọn:

```powershell
$zipPath = 'experiments/rule_based/history/development_history_20261031.zip'
$expectedHash = '3f54990fb2c6f37631e5d8aa00491b52881d513c108b73872768048b95543b1f'
if ((Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {
    throw 'ZIP hash mismatch'
}
$restoreRoot = Join-Path ([IO.Path]::GetTempPath()) ('rule_based_history_' + [guid]::NewGuid().ToString('N'))
Expand-Archive -LiteralPath $zipPath -DestinationPath $restoreRoot
$manifestPath = Join-Path $restoreRoot '_archive/manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($record in $manifest.files) {
    $file = Join-Path $restoreRoot $record.path
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $record.sha256) {
        throw ('Restored file hash mismatch: ' + $record.path)
    }
}
$draftLabel = 'dev050_01'
$relativeDraft = 'experiments/rule_based/versions/' + $draftLabel
$draftSource = Join-Path $restoreRoot $relativeDraft
if (Test-Path -LiteralPath $relativeDraft) { throw 'Draft already exists; refusing overwrite' }
Copy-Item -LiteralPath $draftSource -Destination $relativeDraft -Recurse
```

Đổi `$draftLabel` thành nhãn cần phục hồi; không ghi đè bản đang tồn tại.
Khi chạy thử sau này, luôn dùng run ID mới và giữ nguyên các log cũ.

## Kết quả đối chiếu

Hoàn tất đối chiếu SHA-256 lúc **2026-10-31 05:23:19 UTC** (12:23:19 giờ Việt Nam).
Danh sách và nội dung **4.151/4.151 file được bảo vệ** khớp trước/sau;
không thiếu file và không có file mới ngoài phạm vi tài liệu/history đã nêu.

| Phạm vi được bảo vệ | File khớp SHA-256 |
|---|---:|
| `src/` | 58 |
| `scripts/` | 22 |
| `tests/` | 21 |
| `configs/` | 18 |
| `datasets/` | 26 |
| `results/` | 243 |
| `docs/`, trừ changelog được cập nhật | 46 |
| `experiments/`, trừ draft đã đóng gói, history và README được cập nhật | 120 |
| `runs/` | 3.595 |
| `README.md`, `AGENTS.md` tại gốc project | 2 |
| **Tổng** | **4.151** |

Đây là số file thực tế, gồm cả file đã được Git theo dõi và file bị ignore;
khác với số file mới trong bảng phân loại trước khi thu gọn.

Đối chiếu trạng thái Git toàn repository xác nhận chỉ có đúng thay đổi dự kiến:
180 file dev không còn ở dạng rời, thêm ZIP và README. Hai tài liệu được cập nhật
vốn đã nằm trong danh sách file mới nên không làm tăng số lượng thay đổi.

| Trạng thái Git thực tế | Trước | Sau |
|---|---:|---:|
| File mới, chưa được theo dõi | 463 | 285 |
| File đã sửa | 49 | 49 |
| File đã có trạng thái xóa từ trước | 6 | 6 |
| **Tổng** | **518** | **340** |

Giảm **178** mục trong danh sách thay đổi. Không stage/commit và không chạy lại
test, simulator hoặc experiment; không có số đo hiệu năng lái xe mới.
