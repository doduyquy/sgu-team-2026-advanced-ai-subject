# Stage 4 — S4-G0: Kiểm định kỹ thuật và khoa học

**G0 đã nghiệm thu ngày 13/11/2026; G1 sẵn sàng bắt đầu, chưa triển khai.**
Đây là trạng thái tại mốc G0 của báo cáo này. Tiến độ hiện tại đã đến G1
bước 4; xem [chốt FIT/DEV và Kaggle](STAGE4_G1_SPLIT_KAGGLE_VI.md) và
[kế hoạch hiện hành](../experiments/stage4/plan_v1.json).
Quyết định và giới hạn được ghi tại [biên bản nghiệm thu](STAGE4_G0_REVIEW_VI.md).

Người phụ trách: Đăng Khoa; nhánh `khoa`. Commit nền trước khi triển khai là
`9a36c313eb3ccdbc671842a81ddb1769bd6ae747`. Các tệp mới có bản chụp và mã băm
riêng trong hồ sơ từng lần thử, chưa mặc định thuộc commit nền.

Stage 4 chọn học bắt chước hành vi (Behavior Cloning, BC), dùng Stage 1 làm tác
nhân hướng dẫn. Mục tiêu là học chính sách điều khiển chạy độc lập và đánh giá
khả năng lái thực tế. G0 đã tạo công cụ kiểm định, bằng chứng và thiết kế; chưa
thu dữ liệu mẫu mới, xây mạng nơ-ron hoặc huấn luyện.

## 1. Bằng chứng và tái lập

- [Kết quả diễn giải bằng tiếng Việt](STAGE4_KET_QUA_VI.md), gồm ý nghĩa mã trạng thái.
- [Định nghĩa kiểm định](../experiments/stage4/g0_audit_v1.json): năm lượt TRAIN
  nghiệm thu `accept04`, lượt kiểm tra nhanh 0.6.1 và bảy vết phát lại.
- [Chỉ mục bằng chứng](../results/audits/stage4/g0/attempt_01/evidence_index.json)
  và [tổng hợp số đo](../results/audits/stage4/g0/attempt_01/audit_summary.json).
- [Phụ lục xác minh G0](../results/audits/stage4/g0/final_verification.json):
  **447/447 kiểm thử phần lõi đạt**, gồm **24 kiểm thử mới**; **308 tệp được bảo
  vệ giữ nguyên**. Lần tích hợp trước bước gia cố kết thúc có 446 kiểm thử;
  hai mốc mã nguồn và kết quả được ghi riêng.
- [Kế hoạch nghiệm thu](../experiments/stage4/plan_v1.json),
  [hướng dẫn dòng lệnh](../experiments/stage4/README.md),
  [nhật ký phát triển](STAGE4_CHANGELOG.md).

Trong môi trường Python MetaDrive đã cố định phiên bản, chạy từ
`autonomous-driving-rl/`:

```powershell
python -B -X utf8 scripts/stage4.py audit --run-id s4_g0_lan_thu_moi
python -B -X utf8 scripts/stage4.py publish-audit --run-id s4_g0_lan_thu_moi --output results/audits/stage4/g0/lan_thu_moi
```

Mỗi lần thử dùng ID và thư mục mới, lưu thời gian UTC thực tế, trạng thái, bản
chụp mã nguồn/cấu hình, tình huống, nhật ký kiểm thử và hồ sơ toàn vẹn. COMPLETE
chỉ nghĩa là thực thi xong; `gate_approved=false` vẫn giữ nguyên. Lượt lỗi hoặc
bị ngắt được giữ và chưa đủ điều kiện xuất. `--skip-tests` chặn xuất bằng chứng.

Các lượt gốc trong `runs/` không nằm trong Git. Sao chép kho Git chưa đủ tái lập
toàn bộ phép kiểm: cần giữ hoặc khôi phục đúng dữ liệu gốc và xác minh mã băm.
Không tạo quỹ đạo thay dữ liệu thiếu. Kiểm định dữ liệu không khởi tạo mô phỏng;
nhóm kiểm thử phần lõi có thể tạo môi trường mẫu để kiểm tra hợp đồng.

## 2. Mã nguồn và phạm vi tác nhân hướng dẫn

| Thành phần | Kết quả đối chiếu | Ý nghĩa |
|---|---|---|
| Stage 0 | v0.2.0, ngẫu nhiên, có bộ nhớ theo lượt, discrete9 | Mốc so sánh riêng; giữ lưới hành động. |
| Stage 1 | v0.6.1, xác định; nhớ xe, tuyến, giao lộ và đổi làn | Ứng viên hướng dẫn TRAIN; một quan sát tức thời có thể thiếu trạng thái. |
| Đầu vào | Lõi 259D; xe/đường khả dĩ; tuyến/giao cắt/làn; chuyển động xe | Chỉ dùng lõi là thí nghiệm giảm thông tin so với tác nhân hướng dẫn. |
| Vòng đời | `descriptor/reset/act/close`, `AgentInputV1`, `AgentDecision` | Giữ giao thức; không cấp đối tượng mô phỏng hoặc dữ liệu riêng của bộ đánh giá. |
| Bộ khởi chạy | Stage 0/1 chỉ SANDBOX/AUDIT; chưa đăng ký Stage 4 | Chưa tự bật quyền đánh giá chuẩn. |
| Bộ tình huống | 180 TRAIN, 48 VALIDATION, 12 TEST; bộ đánh giá 96/60 tình huống | Chỉ tách FIT/DEV từ TRAIN. |

Năm lượt nghiệm thu P4 gồm **16 tình huống TRAIN, 12 hình học, 22.228 quyết
định**, có **13 SUCCESS, 3 TIMEOUT**. Cả ba tình huống Extreme đã khảo sát đều
hết thời gian. Bằng chứng này chưa đại diện toàn bộ TRAIN hoặc VALIDATION/TEST.

Bộ kiểm đối chiếu mã nguồn/cấu hình với P4, chương trình chạy, trạng thái, số
lượt con, hình học/seed/giới hạn bước và mọi cặp quan sát/hành động. Khôi phục
qua `restore_input()` rồi tuần tự hóa lại phải giữ đúng đầu vào công khai.
Hành động phải khớp kết quả bộ chuyển đổi gửi sang float32. Lệch làm tròn lớn
nhất khoảng `2.98e-8`; so trực tiếp float Python sẽ báo sai cả 22.228 dòng.

Chính sách 0.6.1 hiện tại phát lại **1.992 bước**, khớp quyết định, chẩn đoán và
hành động. Sáu vết là phép thử vật lý có can thiệp, một vết là Hard/XTOCS. Đây
là kiểm tra trên đầu vào đã ghi; chưa tạo điểm lái mới hoặc chứng minh mọi trạng thái.

Bản chụp 0.6.1 có **40/52 tệp khớp byte**, **12/52 chỉ khớp sau đổi CRLF sang
LF**. Kiểm tra mã băm byte gốc vẫn chưa đạt. Không suy diễn thời điểm/nguyên
nhân, sửa lịch sử hoặc xem tương đương LF là chứng nhận. G0 lưu ứng viên mới từ
byte hiện tại bằng `save_agent_provenance` tại
`runs/<audit_run_id>/candidate_expert/`, kèm mã băm và đối chiếu cấu hình.
Ứng viên đã được kiểm chứng cho G0 theo biên bản nghiệm thu; chưa là bản phát
hành được duyệt để thu dữ liệu quy mô lớn.

**Kết luận:** 0.6.1 đủ làm ứng viên phát triển có điều kiện. Chất lượng từng quỹ
đạo vẫn cần xem xét; P5 và kiểm định hợp đồng/đánh giá chuẩn còn thiếu. Điểm P4
của 0.6.0 giữ nguyên danh tính lịch sử.

## 3. Tái sử dụng và vấn đề còn mở

| Đã có | Cần phát triển |
|---|---|
| `DecisionTrace`: bước, đầu vào, quyết định/chẩn đoán, hành động, độ trễ | Kê khai dữ liệu, chỉ mục nguồn mẫu, FIT/DEV và sổ chất lượng. |
| `save_agent_provenance`: nguồn, mô tả tác nhân, cấu hình và mã băm | Định danh bất biến dữ liệu/bộ mã hóa/chuẩn hóa/mô hình, liên kết với đánh giá. |
| `restore_input`: phục hồi đủ ngữ cảnh P4 | Bộ mã hóa chung khi xây dữ liệu, học và suy luận. |
| Bộ ghi cục bộ: kết quả, chỉ số, thời gian, trạng thái và toàn vẹn | Bộ huấn luyện, bản lưu, đánh giá vòng hở và chính sách BC độc lập. |
| Tình huống cố định, phân vai seed, bộ khởi chạy/kiểm tra trước chạy | Kiểm định đầu vào chung và điều kiện từng tác nhân. |

Vết ghi trước `env.step()`, chưa có xác nhận riêng bước đã áp dụng, quan sát sau
bước, tiến độ tuyến hoặc cờ an toàn từng bước. BC không cần phần thưởng/chuyển
trạng thái, nhưng phải loại lượt thiếu/lỗi và so số dòng với kết quả. An toàn
cuối lượt chưa tự chứng minh mọi đoạn bên trong tốt.

1. `AgentContractV1` 1.3.0 đang `SANDBOX-EXTENSION-PENDING-AUDIT`; chứng nhận cũ
   chưa tự chứng nhận phần mở rộng đầu vào.
2. `requires_checkpoint` mới là siêu dữ liệu; chưa liên kết đầy đủ trọng số,
   bộ mã hóa và chuẩn hóa với kế hoạch. Cần bổ sung khi tích hợp.
3. Ghi vết phụ thuộc `export_config()`: Stage 1 có, Stage 0 chưa có.
4. Bộ thực thi đọc `speed_kmh`, MetaDrive định nghĩa `speed_km_h`, khiến lượt
   kiểm tra nhanh ghi vận tốc 0 dù đến đích. Chưa dùng cột này phân loại TIMEOUT;
   sửa sau G0 cần kiểm chứng TRAIN và hồ sơ phiên bản riêng.
5. Mô tả bộ chuyển ghi dấu lái dương là phải, nguồn/kiểm thử Stage 0/1 ghi trái.
   Giữ nhãn số và xác minh trên TRAIN trước thu, không tự đảo dấu.
6. Một số README cũ chưa cập nhật; lấy nguồn và bằng chứng đã đối chiếu làm căn cứ.

## 4. Bộ dữ liệu kết hợp có kiểm định chất lượng

[Cấu trúc đề xuất](../datasets/stage4_bc/SCHEMA.md) quy định ghép mẫu, truy xuất,
xem xét và giới hạn đặc trưng. **G0 chưa chấp nhận dữ liệu học.** BC-A dùng
SUCCESS đã kiểm định; BC-B dùng đúng BC-A đó cộng từng đoạn TIMEOUT được duyệt.
Giữ mọi kết quả gốc và đoạn loại/chưa rõ; tách phép thử có can thiệp.

Phân loại TIMEOUT: tiến triển hợp lệ; dừng/nhường có lý do; bế tắc/lặp vô ích;
chưa đủ bằng chứng. Đối chiếu làn/tuyến, xe và đường khả dĩ, tràn dung lượng,
điều khiển và diễn biến xung quanh; chẩn đoán hướng dẫn chưa tự chứng minh chất lượng.

G0 tạo **118 khoảng chờ xem xét** theo nhãn hành vi liên tiếp, có nguồn, mã băm
vết, `[start,end)`, ga/phanh, vận tốc/dịch chuyển từ quan sát công khai. Tất cả
`UNREVIEWED`, `approved_for_bc=false`. Ranh giới hành vi chưa phải ranh giới chất
lượng; dịch chuyển chưa phải tiến độ tuyến.

G1 khảo sát FIT, xem ví dụ đại diện/giáp ranh, hiệu chỉnh tiêu chí giải thích
được, ghi bất đồng rồi đóng băng phiên bản trước xây BC-B. Ngưỡng, trọng số và
tỷ lệ vẫn null; quyết định phải có người xem, thời điểm, lý do và ngữ cảnh.
Dữ liệu có **1.593 bước phanh dưới -0.1**, **213 bước CHANGE_LANE**; đây chỉ là
thống kê, chưa quy định chất lượng hoặc tỷ lệ lấy mẫu.

Đề xuất chia theo hình học: mỗi họ xếp 15 hình học TRAIN bằng SHA-256 ổn định;
3 DEV, 12 FIT, tổng **36/144**. Mọi lượt/đoạn/lịch sử cùng hình học ở cùng phía.
Chỉ FIT xác định thống kê chuẩn hóa; DEV là phần giữ lại nội bộ, khác VALIDATION.
Đề xuất vẫn chưa phê duyệt.

## 5. Thiết kế mô hình và quy trình

```text
TRAIN → tác nhân hướng dẫn → vết gốc → kiểm định chất lượng → BC-A/BC-B
      → bộ mã hóa chung → học có giám sát → gói mô hình đóng băng
      → chính sách độc lập → lái vòng kín qua launcher → báo cáo xác minh
```

Bộ mã hóa giữ lõi 259D; 8 xe với mặt nạ/tràn dung lượng; tối đa 8 đường công
khai/xe, 22 điểm/đường; 20 điểm tuyến; điểm quyết định/giao cắt; 3 làn tối đa
28 điểm/làn với biên/khả năng nối tuyến; chuyển động xe. Góc dùng sin/cos, loại
rời rạc dùng one-hot, giá trị thiếu có mặt nạ. Chuẩn hóa bằng phần tử FIT hợp
lệ, đệm giữ 0. Không nhúng định danh bản đồ/làn/nhánh; không xem vị trí mảng xe
là định danh bền vững. Giữ chung bộ mã hóa, thứ tự đặc trưng, phiên bản và mã băm.

Kết quả, phần thưởng, chẩn đoán/bộ nhớ hướng dẫn, nhãn chất lượng và dữ liệu
riêng của bộ đánh giá không vào chính sách. Không bổ sung nhãn hành động từ vết
vào lịch sử; giữ lõi công khai theo hợp đồng.

Mô hình cơ sở 0.1.0: `D → 256 → 256 → 2`, ReLU, đầu ra tanh, mất mát MSE.
D từ cấu trúc mã hóa đóng băng. Không âm thầm cắt hành động. Tốc độ học, cỡ lô,
số vòng và thiết bị chốt bằng thử nghiệm nhỏ, ghi trước huấn luyện. Môi trường
MetaDrive hiện thiếu PyTorch; cần xác minh tương thích/ngân sách, chưa mặc định GPU.

Phiên bản 0.2.0 so **10 khung tại 10 Hz** với một khung, chỉ hiện tại/quá khứ,
có mặt nạ và đặt lại theo lượt. Stage 1 nhớ xe khuất, nhánh tuyến, thời gian chờ
và đường đổi làn; một khung có thể thiếu trạng thái, mười khung cũng chưa bảo
đảm đủ. Chỉ đề xuất mô hình hồi tiếp khi có bằng chứng lỗi do thiếu lịch sử.

Bộ huấn luyện tách suy luận. Gói chứa `state_dict`, cấu hình mạng, mã hóa/chuẩn
hóa, mã băm dữ liệu/nhãn/nguồn/cấu hình, seed, hồ sơ chọn bản lưu, phiên bản môi
trường và mã băm tệp. Chính sách dùng `eval()`/`torch.inference_mode()`, cố định
trọng số/chuẩn hóa, không gọi hướng dẫn sửa hành động.
[Hướng dẫn PyTorch](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

Giữ `AgentPolicy` và bộ chuyển hành động. Khi tích hợp, hàm tạo trỏ tệp mô hình
bất biến; kiểm trước chạy xác minh tệp/mã băm/cấu trúc, nạp đúng byte đã kiểm,
ghi liên kết với kế hoạch/Git commit. Đổi bản lưu tạo định danh gói mới, không
dùng tham chiếu mới nhất có thể đổi. Không sửa hợp đồng khoa học để vượt kiểm tra.

## 6. Công việc và nghiệm thu từng cổng

| Cổng | Sản phẩm | Điều kiện chuyển |
|---|---|---|
| G0 | Công cụ, kiểm thử, bản chụp, bằng chứng, báo cáo, thiết kế | Tái lập được; giải trình giới hạn/lệch xuống dòng; người phụ trách chấp nhận. |
| G1 | Đo tài nguyên thử, dữ liệu/nguồn gốc, FIT/DEV, quy tắc, BC-A/B | Cách ly TRAIN, ghép mẫu/toàn vẹn đúng; TIMEOUT có bằng chứng; đóng băng ứng viên/quy tắc. |
| G2 | Quy trình PyTorch, chuẩn hóa, bản lưu, đánh giá vòng hở | Học khớp FIT nhỏ; mất mát/đạo hàm hữu hạn; so dự đoán hằng trên DEV; nạp lại/seed tái lập. |
| G3 | Stage 4 độc lập trong SANDBOX/AUDIT trên TRAIN | Hành động/đặt lại/nhật ký đúng; trọng số cố định; đến đích ở tình huống hướng dẫn từng thành công; báo mọi lượt. |
| G4 | Đối chứng, phân loại lỗi, cải thiện | Kiểm soát kiến trúc/tình huống/ngân sách; tách ảnh hưởng chất lượng, số mẫu, độ phủ, lịch sử. |
| G5 | Kiểm định điều kiện, VALIDATION đầy đủ, TEST đóng băng, so Stage 0/1/4 | Đạt hợp đồng/điều kiện; cây làm việc sạch, phiên bản cố định; đủ bộ và toàn vẹn. |

Sau nghiệm thu G0, bản thử nhỏ dùng một seed đi G1–G3. Khi quy trình đúng, mở
rộng TRAIN và ba mô hình độc lập với seed **101/202/303** trên cùng dữ liệu đóng
băng. Seed huấn luyện tách seed tác nhân; suy luận xác định khi đánh giá chuẩn
dùng `agent_seed=None`. Không chọn riêng seed đẹp nhất.

Vòng hở báo MSE/MAE từng hành động/hành vi. Vòng kín báo tỷ lệ đến đích an toàn,
mất an toàn, tiến độ tuyến, hết thời gian, thời gian đến đích và độ trễ theo mức
khó/họ. Không dùng phần thưởng/mất mát thay khả năng lái. Báo số hình học/lượt
và biến thiên seed; các khung không phải phép thử khoa học độc lập.

DEV chọn bản lưu ban đầu; **96 tình huống VALIDATION** chọn cấu hình sau kiểm
định điều kiện. Ưu tiên đến đích an toàn cao, rồi mất an toàn thấp, rồi tiến độ
cao. TEST **60 tình huống/bản lưu** chỉ cho phương pháp cuối đã đóng băng với
mọi seed định trước; không dùng điều chỉnh. Thiếu ngân sách/kiểm định thì ghi
thăm dò. Giữ bộ chuyển Stage 0 và nêu giới hạn lưới rời rạc.

Nếu vòng kín kém dù mất mát tốt, phân tích lệch phân phối trước DAgger. Hướng dẫn
có bộ nhớ phải được cập nhật đúng trên quỹ đạo tác nhân học; không đặt lại rồi
hỏi riêng từng khung. Chưa triển khai hướng này.
[Nghiên cứu DAgger, Ross và cộng sự, 2011](https://proceedings.mlr.press/v15/ross11a.html)

## 7. Tổ chức mã và nhật ký

G0 thêm `src/imitation/audit.py`, `scripts/stage4.py`, kiểm thử hỏng dữ liệu/rò
rỉ/nguồn gốc/vòng đời, kế hoạch và thiết kế dữ liệu. Chưa thêm chính sách giả
hoặc bật đăng ký Stage 4.

Các thành phần tương lai dưới `src/imitation/`: dữ liệu, chất lượng, đặc trưng,
mô hình, bộ huấn luyện và gói mô hình. Chính sách ở
`src/agents/behavior_cloning_agent.py`. Chỉ bổ sung thu/xây/học/đánh giá ở cổng
tương ứng; mở rộng đăng ký/kiểm trước chạy/thực thi khi tích hợp bản lưu.

Mỗi lần thử giữ nguồn/cấu hình/môi trường, đúng tình huống/seed, vết đầu vào/hành
động nếu lái, nguồn dữ liệu/đoạn, mất mát/cập nhật/chọn bản lưu nếu học, trạng
thái lỗi/dừng và toàn vẹn. Hồ sơ cục bộ có thẩm quyền; W&B là bản sao theo dõi,
phát triển mặc định DISABLED. Không ghi thông tin xác thực. Trước sửa bản phát
hành phải giữ bản chụp bất biến, cập nhật giả thuyết, thay đổi, kiểm thử, ID thí
nghiệm, kết quả, hồi quy và giới hạn.

**Kết luận G0:** đã nghiệm thu bằng chứng và thiết kế theo
[quyết định ngày 13/11/2026](STAGE4_G0_REVIEW_VI.md); ứng viên đã được kiểm chứng
với giới hạn được ghi nhận. G1 sẵn sàng bắt đầu, chưa triển khai. Tiêu chí chất
lượng, ngân sách, kiểm chứng vận hành trước thu quy mô lớn và điều kiện đánh
giá chuẩn vẫn cần hoàn tất ở cổng tương ứng; chưa chứng nhận dữ liệu hoặc mô hình.
