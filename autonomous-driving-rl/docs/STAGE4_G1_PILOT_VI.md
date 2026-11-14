# G1 bước 2 — Thu thử dữ liệu thô Stage 1

Ngày thực hiện: **14/11/2026 (Asia/Saigon)**. Phạm vi: thu thử và kiểm tra
nguồn gốc/tính đầy đủ/ghép bước trên TRAIN. Chưa duyệt chất lượng, chưa chốt
FIT/DEV, chưa xây BC-A/BC-B và chưa huấn luyện.

## Trạng thái

**Bước 2 đã hoàn tất:** `s4_g1_pilot_02` COMPLETE, 5/5 lượt, 4 hình học,
**5,744 cặp quan sát–hành động**, 4 SUCCESS và 1 TIMEOUT. Toàn bộ bước đã được
đối chiếu với simulator và phát lại Stage 1 khớp quyết định/chẩn đoán/hành động.
Các lượt vẫn mang nhãn `UNREVIEWED`; số mẫu được nhận vào BC là **0**.

| Lượt | Bước đã thu và kiểm tra | Kết thúc |
|---|---:|---|
| episode_000 | 418 | SUCCESS |
| episode_001 | 414 | SUCCESS |
| episode_002 | 566 | SUCCESS |
| episode_003 | 1,432 | SUCCESS |
| episode_004 | 2,914 | TIMEOUT |

Extreme kết thúc theo horizon gốc, tiến độ cuối khoảng **45.15%**; không bị
cắt sớm hoặc loại vì TIMEOUT. Chưa suy ra nguyên nhân/chất lượng từng đoạn.

- [Chỉ mục bằng chứng và mã băm](../results/audits/stage4/g1/pilot_v1/evidence_index.json).
- [Xác minh cuối: chỉ mục mẫu, cấu hình, vết tham chiếu, kiểm thử và lần lỗi](../results/audits/stage4/g1/pilot_v1_verification.json).
- [Sổ đăng ký dữ liệu thô](../datasets/stage4_bc/pilot_raw_v1.json); đây chưa phải manifest của BC-A/BC-B.

Đã kiểm lại 1,018 tệp trong hồ sơ toàn vẹn trước khi xuất. Chỉ mục 5,744 mẫu
không trùng, liên tục trong mỗi lượt và gắn đúng mã băm vết; cấu hình runtime
thực tế khớp case/seed/giao thông/horizon/điều khiển. Đủ **572** sự kiện tiến độ,
đúng mỗi 10 bước. Nguồn/cấu hình được chụp và giữ nguyên: 130 tệp nguồn/định
nghĩa; 182 tệp trong các nhóm được bảo vệ không đổi.

Ba tình huống trùng với kiểm chứng bước 1 (Easy env 0, Medium và Hard) có
**2,416/2,416 quan sát/quyết định/chẩn đoán/hành động giống hệt vết tham chiếu**
khi bỏ độ trễ `act_latency_ms`. Bộ ghi mới không làm đổi hành vi ở các đối chứng này.

## Lựa chọn tình huống và phương pháp

Định nghĩa có phiên bản nằm tại
[`g1_pilot_v1.json`](../experiments/stage4/g1_pilot_v1.json): 5 lượt trên
4 hình học, phủ bốn mức khó. Easy chạy với hai seed môi trường để kiểm tra
việc tách lượt và truy xuất nguồn gốc khi bản đồ giống nhau. Có Extreme trong
nhóm nhỏ để kiểm tra đường lưu kết quả khó/TIMEOUT, không chỉ đường SUCCESS.

| Lượt | Mức khó / chuỗi | Seed hình học | Seed môi trường | Giới hạn bước |
|---|---|---:|---:|---:|
| episode_000 | Easy / SCS | 0 | 0 | 1,224 |
| episode_001 | Easy / SCS | 0 | 1 | 1,224 |
| episode_002 | Medium / SCXCS | 14 | 0 | 1,098 |
| episode_003 | Hard / SCTXrCS | 2 | 0 | 2,208 |
| episode_004 | Extreme / SCXOCrTYCS | 0 | 0 | 2,914 |

Mọi lượt đều dùng Stage 1 **0.6.1**, cấu hình mặc định đã lưu mã băm, agent
seed **42**, `SANDBOX`, render OFF, W&B DISABLED, điều khiển 10 Hz. Giữ nguyên
giao thông và giới hạn bước do nền tảng tính theo tuyến; không can thiệp
hành động, đặt lại xe giữa lượt, rút ngắn horizon hoặc loại kết quả xấu.
Hình học được xác minh bằng tệp phân chia và cơ chế nạp/băm bản đồ đóng băng.

Mỗi lượt chạy trong một tiến trình riêng. Bộ thực thi vẫn là
`ExperimentExecutor`; bộ ghi bao quanh `reset/step` chỉ đọc trạng thái và lưu
kết quả. Đầu vào công khai và hành động của Stage 1 giữ nguyên. Lỗi kỹ thuật
được giữ và làm lần thu không đạt; bộ thu vẫn ghi kết quả các lượt còn lại.
TIMEOUT và thất bại an toàn là kết quả lái hợp lệ để lưu, không phải lỗi kỹ thuật.

## Dữ liệu được lưu

Thư mục gốc: `runs/s4_g1_pilot_02/`.

| Tệp/nhóm | Nội dung |
|---|---|
| `request.json`, `resolved_plans.json` | Định nghĩa, tình huống chính xác, seed, nguồn và mã băm bằng chứng bước 1 |
| `source_snapshot/`, `simulator_source/`, `runtime.json` | Byte nguồn dự án, nguồn Python simulator, phiên bản phụ thuộc, commit và trạng thái Git |
| `expert/` | Nguồn/cấu hình và descriptor của Stage 1 cùng SHA-256 |
| `episodes/episode_*/request.json`, `environment.json` | Case, cấu hình yêu cầu, tần số điều khiển, trạng thái can thiệp |
| `reset.json` | Quan sát đầu, thông tin reset và cấu hình runtime thực tế |
| `map.json.gz` | Siêu dữ liệu bản đồ thực tế, chuỗi block và hình học làn |
| `launcher/stage1/agent_decisions.jsonl.gz` | Quan sát công khai t, quyết định/chẩn đoán và hành động float32 t trước `env.step` |
| `transitions.jsonl.gz` | Cùng chỉ số t: băm quan sát trước bước, hành động gửi đi, trạng thái xe trước/sau, quan sát sau bước, reward simulator, cờ và toàn bộ `step_info` |
| `launcher/stage1/` | CSV kết quả, chỉ số, độ trễ, nguồn/cấu hình tác nhân và hồ sơ toàn vẹn chuẩn của launcher |
| `events.json`, `console.log`, `episode_state.json` | Sự kiện vòng đời, đầu ra tiến trình và trạng thái hoàn tất/lỗi |
| `verification.json`, `pilot_summary.json` | Kết quả đối chiếu từng lượt và tổng hợp mọi kết quả |
| `raw_sample_index.jsonl.gz` | Chỉ mục `(run_id, episode_id, step_index)` và băm vết; mọi dòng chưa được nhận vào BC |
| `run_state.json`, `run_integrity.json` | Trạng thái lần thu và SHA-256 theo byte của mọi tệp, gồm cả snapshot và log |

Tham chiếu lớp/hàm trong cấu hình runtime được mã hóa rõ dạng
`{"python_type": "module:Class"}` hoặc `python_callable`, định dạng
`json-with-python-references/1.0.0`; không gọi chúng, bỏ trường hoặc đổi đối
tượng không biết thành chuỗi `repr`. Nguồn module và phiên bản môi trường
được giữ để truy lại các tham chiếu này.

Trạng thái toàn cục, kết quả, reward, vị trí vật lý và chẩn đoán chỉ phục vụ
kiểm tra/phân tích, nằm tách khỏi nhãn quan sát–hành động. Không bổ sung chúng
làm đặc trưng của mô hình BC. Quan sát sau bước cũng không thay thế quan sát
trước bước trong cặp học.

## Kiểm tra tính đầy đủ và đúng bước

Với từng lượt, bộ kiểm đọc tuần tự và yêu cầu:

1. Thuộc TRAIN, đúng hình học/seed/giới hạn bước, không dùng seed holdout.
2. Bản đồ thực tế có chuỗi block giống cấu hình đã khóa; nguồn/cấu hình tác
   nhân khớp bản chụp đầu lần thu, các tệp nguồn khớp mã băm.
3. Case ID đúng; chỉ số liên tục từ 0; không thiếu/thừa dòng giữa vết quyết
   định và vết chuyển bước; không có bước sau kết thúc.
4. Quan sát công khai t khớp quan sát đầu hoặc quan sát trả về từ bước t−1;
   băm trước bước khớp; hành động nhãn đúng float32 và đúng lệnh gửi simulator.
5. Dấu lái thực thi, mốc thời gian 0.1 giây và vận tốc trước/sau khớp số đo.
6. Phát lại Stage 1 từ đầu vào công khai đã khôi phục cho cùng quyết định,
   chẩn đoán và hành động, đặt lại bộ nhớ riêng ở mỗi lượt.
7. Tính lại kết thúc theo cờ từng bước, gồm ưu tiên an toàn trước SUCCESS và
   TIMEOUT theo horizon. Kiểm lại vận tốc, tiến độ và reward nền tảng trong CSV.
8. Hồ sơ launcher hoàn tất, không có lỗi kỹ thuật; sự kiện vòng đời và vận tốc
   tiến độ khớp. Nguồn/cấu hình/bằng chứng được bảo vệ giữ nguyên trước/sau.

Kiểm tra trên chỉ chứng minh tính hợp lệ và nguồn gốc của dữ liệu thô.
Nhãn SUCCESS/TIMEOUT, hành vi YIELD hoặc kết quả phát lại khớp chưa chứng minh
chất lượng hành động để học.

## Kiểm thử và lần thử lỗi

- `s4_g1_pilot_01`: FAILED. Cả 5 tiến trình bị chặn trước bước vật lý đầu tiên
  vì cấu hình runtime chứa lớp Python mà bộ ghi JSON chưa hỗ trợ. Giữ nguồn,
  log, trạng thái và hồ sơ toàn vẹn; không tái sử dụng ID hoặc sửa bằng chứng.
- Sau sửa mã hóa tham chiếu lớp/hàm, `s4_g1_pilot_02` thu lại toàn bộ định nghĩa.
- `s4_g1_pilot_checks_01`: 467 kiểm thử backend đạt ở phiên bản đầu; không thay
  thế kiểm thử cuối vì sau đó đã bổ sung bản sửa mã hóa cấu hình.
- `s4_g1_pilot_checks_02`: **469/469 kiểm thử backend đạt**, gồm 17 kiểm thử mới
  về lệch bước/quan sát/hành động, đơn vị vận tốc, cấu hình runtime, mã băm,
  bảo toàn lần lỗi và cấm ghi đè. Lần này không sửa nền tảng/giao diện GUI.

## Chạy lại

Từ `autonomous-driving-rl/`, dùng môi trường Python MetaDrive đã cố định:

```powershell
python -B -X utf8 scripts/collect_stage4_pilot.py collect --run-id s4_g1_pilot_new
python -B -X utf8 scripts/collect_stage4_pilot.py publish --run-id s4_g1_pilot_new --output results/audits/stage4/g1/pilot_new
```

Dùng thư mục mới cho từng lần. `publish` chỉ xuất khi toàn bộ lượt dự kiến đã
hoàn tất/đối chiếu và mọi mã băm còn khớp. Các bản tổng hợp trong Git cần đi
kèm dữ liệu thô gốc trong `runs/` nếu muốn tái kiểm định từng bước.

## Giới hạn và bước sau

Nhóm nhỏ chỉ có 4/180 hình học TRAIN, không đại diện toàn bộ phân phối và
không phải đánh giá chuẩn. Hai lượt cùng hình học Easy phải ở cùng một phía
khi chốt FIT/DEV sau này; hiện giữ `UNASSIGNED_PENDING_STEP4`.

Tại thời điểm bước 2, mốc thời gian trong log là siêu dữ liệu nguồn gốc,
chưa phải kết luận ngân sách. Sau đó đã có [đo riêng tài nguyên và môi trường
ở bước 3](STAGE4_G1_RESOURCES_VI.md), trên một lần thu mới; môi trường huấn luyện
hiện NOT_READY. Quy tắc chất lượng và BC-A/BC-B vẫn chờ các bước 4–6;
G1 chưa hoàn tất, các cổng huấn luyện chưa mở.
