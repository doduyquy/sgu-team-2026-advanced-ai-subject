# G1 bước 5 — Quy tắc chất lượng dữ liệu thử

Cập nhật sau mốc này: [bước 6 đã xây BC-A/BC-B và hoàn tất G1 ở phạm vi
bộ thử](STAGE4_G1_DATASETS_VI.md). Các trạng thái chưa xây/chưa hoàn tất bên
dưới mô tả thời điểm kết thúc bước 5; giữ nguyên hồ sơ JSON lịch sử.

Phạm vi: kiểm định dữ liệu Stage 1 0.6.1 trên TRAIN, đóng băng quy tắc cho
pilot và ghi quyết định giữ/loại có bằng chứng. Đây là kiểm định dữ liệu;
chưa xây BC-A/BC-B, chưa huấn luyện và chưa nghiệm thu toàn bộ G1.

## Cách xây quy tắc

Giữ nguyên [144 FIT / 36 DEV](../datasets/stage4_bc/fit_dev_v1/manifest.json).
Hai lượt FIT pilot cũ là Medium SCXCS seed 14 và Hard SCTXrCS seed 2, đều
SUCCESS. TIMEOUT cũ thuộc DEV nên không dùng để chọn ngưỡng chất lượng.

Thu thêm đúng hai hình học Extreme FIT, đã định nghĩa trước khi chạy tại
[g1_quality_calibration_v1.json](../experiments/stage4/g1_quality_calibration_v1.json):
SCXOCrTYCS seed 14 và CrXROSTR seed 15, cùng environment seed 0, agent seed 42.
Chọn hình học FIT đứng đầu theo ranking đóng băng trong mỗi họ đó, giữ mọi
kết quả. Không đổi chính sách, giao thông, horizon, không can thiệp hay đổi
case sau khi biết kết quả. Bộ thu kiểm tra phía FIT trước khi chạy.

Các ngưỡng là sàng lọc kỹ thuật có giải thích, được xem xét trên FIT; không
phải ngưỡng tối ưu thống kê hoặc chứng nhận an toàn. Chỉ sau khi đóng băng
quy tắc, mã chương trình và nguồn phụ thuộc mới áp dụng lên DEV. Nếu DEV bộc
lộ hạn chế, báo hạn chế; không sửa v1 để làm kết quả DEV đẹp hơn.

## Quy tắc v1.0.0

[Định nghĩa máy đọc](../experiments/stage4/g1_quality_rules_v1.json) là nguồn
tham số. Tham số chỉ có hiệu lực đóng băng cùng hồ sơ `freeze.json`.

| Lớp kiểm tra | Tiêu chí và ý nghĩa |
|---|---|
| Tính hợp lệ | Lần thu COMPLETE; mã băm nguồn/vết/bản đồ khớp; TRAIN đúng hình học/seed; tái kiểm tra ghép từng bước và phát lại Stage 1. Thiếu hoặc sửa bằng chứng làm lần kiểm định FAILED. |
| Khoảng xem xét | Cửa sổ 50 bước = 5 giây; khoảng `[start, end)`. Kiểm tra thời gian dừng liên tục xuyên biên cửa sổ, không đặt lại bộ đếm ở biên. |
| An toàn/điều khiển | Không có cờ va chạm/ra đường; góc lệch làn tối đa 0,35 rad; khoảng hở thân xe ước lượng tới biên làn ít nhất 0,20 m; biến thiên lái tối đa 0,25 mỗi 0,1 giây; tối đa 8 lần đảo dấu đáng kể/5 giây. Vi phạm giữ ngoài dữ liệu. |
| Vận tốc | Dùng vận tốc vật lý đã xác minh, tối đa 55 km/h cho pilot; thay đổi trên 10 km/h mỗi 0,1 giây cần loại khỏi mẫu. Đây là sàng lọc bất thường, chưa đo độ êm theo jerk. |
| Tiếp cận xe trước | Nếu TTC đóng khoảng cách dưới 1 giây ở vận tốc trên 5 km/h mà hành động chưa phanh ít nhất 0,1 thì loại cửa sổ. TTC là dự đoán vận tốc tương đối tuyến tính, chưa phải chứng minh tránh va chạm. |
| Ngữ cảnh thiếu | Tràn số xe/đường khả dĩ, LiDAR không khả dụng hoặc làn hiện tại chưa được xác nhận tương thích tuyến: UNCERTAIN, không nhận. |
| Tiến triển | Tiến độ tuyến tăng ít nhất 0,001 trong cửa sổ, không có giảm tiến độ quá 0,002 ở một bước và vượt các kiểm tra khác. Dùng `route_completion` để kiểm định, không thay bằng quãng đường xe di chuyển. |
| Chờ có lý do | Xe dưới 1 km/h, đang phanh/giữ ga tối đa 0,05; có xe phía trước với khoảng hở ≤8 m hoặc dự đoán chiếm dụng vùng trước xe trong 3 giây từ quan sát công khai. Giới hạn chuỗi chờ được nhận là 20 giây. Không suy lý do chờ chỉ từ nhãn YIELD. |
| Chờ chưa rõ | Cho phép tối đa 1 giây dừng thiếu hỗ trợ trong cửa sổ; miễn kiểm tra dừng trong 1 giây khởi động. Dừng lâu hơn thiếu bằng chứng thì UNCERTAIN. Không tự biến thiếu bằng chứng thành hành vi xấu. |
| Nghi bế tắc/lặp | Thiếu tiến triển, đứng yên không có lý do với ngữ cảnh cục bộ rõ, hoặc di chuyển ≥2 m mà không tiến triển tuyến: giữ ngoài BC. Đây là dấu hiệu cần xem, chưa khẳng định nguyên nhân vật lý. |

Khoảng hở được tính lại từ quan sát công khai, góc xe và kích thước xe
1,852 × 4,515 m đã dùng trong cấu hình nguồn. Đây là ước lượng hình chữ nhật
theo tiếp tuyến làn, có giới hạn trên cua. Nhãn/chẩn đoán của teacher chỉ được
thống kê để đối chiếu; không làm bằng chứng duy nhất cho quyết định nhận.
Vật cản chưa đủ ngữ cảnh hoặc chờ vì khuất tầm nhìn vẫn có thể bị giữ ngoài;
đó là lựa chọn thận trọng của pilot, không khẳng định chờ là sai.

`UNSAFE_OR_UNSTABLE` là nhãn không qua sàng lọc, chưa phải kết luận vi phạm
đã được chứng minh độc lập. Ví dụ, v1 chưa xác nhận hành lang chuyển làn:
khoảng hở tới biên làn hiện tại có thể âm khi chuyển làn hợp lệ. Những đoạn
này vẫn bị giữ ngoài; không dùng chúng để báo tỷ lệ lái nguy hiểm. Mẫu được
giữ cũng chưa được chứng nhận tối ưu hoặc an toàn trong mọi tình huống.

FIT thử nhỏ chưa có cửa sổ `JUSTIFIED_WAIT` được nhận riêng: những cửa sổ
được giữ đều có tiến triển. Nhánh nhận đoạn chờ được kiểm thử logic nhưng
chưa có ví dụ thực tế dương tính đã được duyệt để khẳng định độ đúng của
quy tắc chờ. Không tạo nhãn chờ tốt để bù độ phủ còn thiếu.

## Nhận vào BC-A và BC-B ở bước 6

- **BC-A:** chỉ nhận toàn bộ lượt SUCCESS khi mọi cửa sổ đã qua kiểm định.
  Một lượt tới đích nhưng có đoạn chưa rõ vẫn chưa được nhận; v1 không cắt
  lấy riêng phần đẹp của lượt SUCCESS đó.
- **BC-B:** giữ đúng phần BC-A và bổ sung các khoảng TIMEOUT đạt tiêu chí.
  Loại vùng đệm 50 bước trước/sau mọi cửa sổ chưa đạt/chưa rõ, loại 100 bước
  cuối TIMEOUT; chỉ giữ khoảng liên tục còn ít nhất 50 bước.
- Các lượt thất bại an toàn, khoảng bị loại và khoảng chưa rõ vẫn giữ nguyên
  dữ liệu gốc. Sổ quyết định mới là lớp chú thích, không sửa sổ pilot lịch sử.
- Mọi khoảng kế thừa FIT/DEV của hình học. DEV chỉ phục vụ đánh giá, không
  trở thành mẫu cập nhật trọng số hoặc fit chuẩn hóa.
- Lịch sử tối đa 10 khung chỉ lấy hiện tại/quá khứ cùng lượt. Ngữ cảnh trước
  một đoạn được nhận không tự trở thành nhãn huấn luyện; không đọc tương lai.

Quyết định này chưa đặt tỷ lệ lấy mẫu hoặc trọng số. Bước 6 phải báo riêng
số mẫu, số đoạn, số hình học và hành vi; không mặc định nhiều bước đứng yên
là nhiều kinh nghiệm hữu ích. Khi thay quy tắc phải tạo phiên bản/hồ sơ mới,
giữ v1 và các khoảng đã loại để truy lại được.

## Tái kiểm định

Chạy từ `autonomous-driving-rl/` bằng môi trường MetaDrive đã cố định:

```powershell
python -B -X utf8 scripts/collect_stage4_pilot.py collect --run-id <new-fit-run> --definition experiments/stage4/g1_quality_calibration_v1.json
python -B -X utf8 scripts/review_stage4_quality.py profile --source-run s4_g1_pilot_02 --source-run <new-fit-run> --rules experiments/stage4/g1_quality_rules_v1.json --output runs/<new-fit-profile>
python -B -X utf8 scripts/review_stage4_quality.py freeze --profile runs/<new-fit-profile> --output results/audits/stage4/g1/<new-rule-freeze>
python -B -X utf8 scripts/review_stage4_quality.py apply --source-run s4_g1_pilot_02 --source-run <new-fit-run> --rules results/audits/stage4/g1/<new-rule-freeze>/rules.json --frozen results/audits/stage4/g1/<new-rule-freeze> --output runs/<new-review>
python -B -X utf8 scripts/review_stage4_quality.py publish --review runs/<new-review> --frozen results/audits/stage4/g1/<new-rule-freeze> --output results/audits/stage4/g1/<new-evidence>
```

Mỗi đầu ra phải mới. Công cụ dừng nếu nguồn thiếu/hỏng, hồ sơ chưa hoàn tất,
hoặc quy tắc/mã kiểm định/phụ thuộc thay sau đóng băng. Kết quả chỉ có chú
thích đủ điều kiện, không tự tạo bộ dữ liệu BC hoặc cập nhật mô hình.

## Kết quả ngày 14/11/2026

Đã kiểm định **7 lượt / 6 hình học / 9.347 bước**, chia thành **191 cửa sổ**.
Nguồn thô là `s4_g1_pilot_02` và `s4_g1_quality_fit_01`, giữ nguyên mọi byte.
Nhóm FIT bổ sung hoàn tất 2/2 lượt: TIMEOUT 2.655 bước và CRASH_VEHICLE 948
bước. Va chạm là kết quả lái được giữ lại, không phải lỗi kỹ thuật thu dữ liệu.

| Phía | Bước thô đã xem | SUCCESS đủ điều kiện BC-A | TIMEOUT bổ sung cho BC-B | Tổng đủ điều kiện BC-B |
|---|---:|---:|---:|---:|
| FIT | 5.601 | 566 | 100 | 666 |
| DEV | 3.746 | 832 | 500 | 1.332 |

DEV giữ đúng phía để đánh giá, không trở thành dữ liệu cập nhật trọng số.
Đây là **số bước đủ điều kiện theo chú thích**, không phải bộ BC đã được xây.

- Medium FIT: giữ toàn bộ `[0,566)`.
- Hard FIT: chưa nhận toàn bộ 1.432 bước dù SUCCESS. Bốn cửa sổ, tổng 200
  bước, còn dừng thiếu bằng chứng công khai theo sàng lọc v1. Các cửa sổ còn
  lại đạt nhưng quy tắc BC-A yêu cầu toàn bộ lượt; không kết luận teacher
  dừng sai chỉ từ việc chưa đủ bằng chứng.
- TIMEOUT FIT SCXOCrTYCS seed 14: sau vùng đệm chỉ giữ `[1900,2000)`.
  Tiến độ tuyến tăng khoảng 0,55586 → 0,58998; vận tốc không quá 16,04 km/h,
  khoảng hở làn tối thiểu khoảng 0,8237 m, không tràn ngữ cảnh trong đoạn.
  Hai cửa sổ sát trước/sau vẫn bị bỏ làm vùng đệm theo quy tắc.
- Va chạm FIT CrXROSTR seed 15: loại toàn bộ khỏi BC-A/BC-B, giữ bằng chứng.
- Hai SUCCESS Easy DEV: giữ lần lượt 418 và 414 bước, cùng một hình học.
- TIMEOUT Extreme DEV: giữ `[0,450)` và `[800,850)`; phần còn lại bị giữ
  ngoài theo quy tắc đã đóng băng. Không dùng kết quả này để chỉnh v1.

BC-A FIT chỉ có **một hình học Medium**; BC-B FIT thêm 100 bước từ một hình
học Extreme. Đây là bộ thử rất hẹp để kiểm tra quy trình, chưa đại diện 144
FIT hoặc chứng minh khả năng khái quát. Tỷ lệ giữ thấp và việc thiếu mẫu chờ
được nhận phải được báo khi sang G2; không nới ngưỡng bằng DEV để tăng số mẫu.

Hồ sơ chính:

- [Nguồn và kết quả nhóm FIT bổ sung](../results/audits/stage4/g1/quality_fit_v1/evidence_index.json).
- [Quy tắc đóng băng trước DEV](../results/audits/stage4/g1/quality_rules_v1/freeze.json),
  kèm thống kê FIT, mã kiểm định và mã băm phụ thuộc.
- [Sổ đầy đủ cửa sổ/đoạn, lý do và bằng chứng](../results/audits/stage4/g1/quality_v1/review.json).
- [Tổng hợp FIT/DEV](../results/audits/stage4/g1/quality_v1/summary.json),
  [kiểm chứng cuối](../results/audits/stage4/g1/quality_v1_checks.json) và
  [sổ chất lượng cho bộ xây dữ liệu](../datasets/stage4_bc/quality_v1.json).

`s4_g1_quality_profile_01` là khảo sát FIT ban đầu, chưa phải quy tắc đóng
băng. `s4_g1_quality_profile_02` là hồ sơ FIT dùng để đóng băng;
`s4_g1_quality_review_01` áp dụng nguyên v1 lên FIT/DEV. Người xem ghi trong
hồ sơ là Codex; không ghi giả rằng người dùng đã ký duyệt từng đoạn.

`s4_g1_quality_checks_01` phát hiện lỗi so sánh đường dẫn ngắn/dài Windows
trong kiểm thử thư mục tạm. Đã sửa chuẩn hóa đường dẫn, giữ nguyên lần lỗi.
`s4_g1_quality_checks_02`: **512/512 backend đạt**, gồm 24 kiểm thử chất lượng
mới và 2 kiểm thử bộ thu chặn DEV/split bị sửa trước mô phỏng. Không sửa GUI,
không chạy lại Qt trong bước này. Kiểm tra byte của nguồn đã kiểm thử khớp
nguồn hiện tại; các hồ sơ lịch sử, split và chính sách Stage 1 giữ nguyên.

**Bước 5 hoàn tất trong phạm vi pilot.** Tiếp theo là bước 6: xây BC-A/BC-B
có chỉ mục mẫu, nguồn gốc, mã băm và báo cáo phân bố. G1 vẫn chưa hoàn tất,
G2 chưa mở, chưa huấn luyện hoặc thu quy mô lớn.
