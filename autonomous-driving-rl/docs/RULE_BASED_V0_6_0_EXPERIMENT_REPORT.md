# Rule-Based 0.6.0 — P4

**P4 đã hoàn tất triển khai và nghiệm thu có phạm vi**, đóng băng development
release 0.6.0 ngày 2026-10-31 (Asia/Saigon; raw timestamps theo UTC).
Nghiệm thu TRAIN, SANDBOX/AUDIT; số liệu dưới đây lấy từ artifact đã xác minh.
Đây không phải full 108 TRAIN, canonical VALIDATION/TEST hoặc chứng nhận benchmark.

| Cùng 16 TRAIN case | 0.5.0 | 0.6.0 |
|---|---:|---:|
| Clean success | 13 | 13 |
| Crash | 0 | 0 |
| Out-of-road / solid-boundary | 0 / 0 | 0 / 0 |
| Timeout | 3 | 3 |

Không mất hoặc thêm case success. Source cuối có **sáu thao tác đổi làn với lý do vượt hoàn tất
trong năm case tự nhiên**: Hard SCXOCS/env0 (1), XTOCS/env0 (1), Medium SCXCCS/env1
(1), Hard SCTXrCS/env1 (1), SCXOCS/env1 (2). P4 có tác dụng trong driving thật;
chưa cải thiện tỷ lệ success trên coverage này. Đây là số kết thúc FSM đổi làn;
việc đi qua được thân xe chắn được đo riêng ở controlled pass probes.
Return-after-pass được kiểm tra
ở unit/lifecycle tests; sáu controlled probes không được diễn giải thành kiểm
chứng vật lý đầy đủ cho mọi trường hợp trở lại làn.

Hai case Hard SCXOCS/env0 và env1 có completion cuối nhỏ hơn baseline nhưng vẫn
SUCCESS; giữ nguyên giá trị và thời gian trong paired CSV. Không xóa chênh lệch
hoặc quy nó thành cải thiện. Ba timeout vẫn là CrXROSTR, SCTXORyCCS và SCXOCrTYCS,
geom0/env0; lần lượt completion 0.403125, 0.472694, 0.451484. Agent không ép ga
hoặc ép đổi làn khi bị chặn trong passage/overflow hoặc không đủ điều kiện.

Artifact xác minh: [tổng hợp](../results/audits/rule_based/v0.6.0/combined_summary.json),
[từng episode](../results/audits/rule_based/v0.6.0/episodes.csv),
[maneuver tự nhiên](../results/audits/rule_based/v0.6.0/lane_change_events.json),
[index 37 run](../results/audits/rule_based/v0.6.0/experiment_index.json).
Archive chính thức: `experiments/rule_based/versions/0.6.0/`; chín source draft
được chép nguyên từ snapshot raw sang `versions/dev060_*`, có nhãn retrospective
archive và DRAFT_NOT_RELEASE. Không sửa snapshot 0.5.0.
[Proof archive](../results/audits/rule_based/v0.6.0/release_verification.json) kiểm
SHA-256 cả 0.5/0.6 và import flat archive 0.6 thật để replay đủ 929 bước.

## Thay đổi so với 0.5.0

- Public schema 1.3.0 có current/neighbor lanes, centerline, vạch thực sự dựng
  và khả năng nối tuyến cục bộ. Vẫn 259D, 8×7 traffic và route 20×2.5 m;
  CORE_ONLY không nhận lane context. Không có live simulator handle trong policy.
- Mandatory route change trước optional return/pass; kiểm tra front/rear gap,
  closing TTC và quãng phanh, swept footprint với CV/CT/road hypotheses, ghost,
  uncertainty, LiDAR chưa phân loại và overflow.
- Đường quintic được giữ trong frame odometry của episode; target không đổi khi
  navigation chuyển nearest lane. Hủy trước commit khi làn cũ cũng an toàn;
  sau commit phanh/giữ target và tiếp tục khi clear. Commit xét cả thân xe xoay.
- Vượt chỉ qua broken white trên đoạn gần thẳng đủ dài; không vượt tùy chọn
  trong toàn bộ approach 50 m đang thấy. Mandatory cần upstream >35 m. Không
  lấy điều kiện khởi tạo approach để dừng một thao tác đã hợp lệ giữa hai làn.
- Route waypoint dừng trước connector gap >0.3 m; không nhảy ngang một làn.
  Endpoint gần grid được deduplicate sau làm tròn để không tạo khoảng cách trùng.
- Giữ cruise 45/junction 24/recovery 8 km/h. P4 prepare/change cap 24 km/h,
  admission <=30 km/h, min transition 16 m, end buffer 10 m, TTC >=4 s,
  min gap 4 m + headway/phanh, cooldown 6 s và settle 0.5 s.

Chi tiết luật, giới hạn và nguồn vạch:
[thiết kế P4](RULE_BASED_P4_DESIGN.md). Lịch sử từng lỗi, điều chỉnh và lần chạy:
[work log](RULE_BASED_V0_6_0_WORKLOG.md).

## Cách so sánh

So với original `current` variant của hai run nghiệm thu 0.5 đã COMPLETED:
`rule_based_v050_acceptance_family12_20261030` và
`rule_based_v050_acceptance_env1_20261030`. Baseline nguồn/params đã đóng băng;
không đổi tên/copy raw folder hoặc chạy lại một baseline khác rồi gán nhãn cũ.

Candidate cuối: prefix `rule_based_v060_accept04_`, chia thành roads9/env0,
ba shard Extreme riêng biệt/env0, targeted4/env1. Tổng **16 case riêng biệt**,
cùng geometry hash, geometry/environment seeds, traffic density, horizon và
simulator config với baseline. Chỉ thứ tự case trong shard khác.

Report generator kiểm source/config hash, trạng thái COMPLETED, TRAIN split,
mọi trace step/case/259D/row count và định nghĩa/cấu hình vật lý trước khi ghép
cặp. Đây là **policy + public extractor change**, không phải policy-only ablation.
Mỗi run ghi runtime, exact cases/parameters, source snapshots và per-step input,
decision, action canonical, diagnostics, act latency cùng outcome riêng biệt.

## Kiểm thử và controlled probes

Source cuối đạt **279/279 tests**, trong đó **26 kiểm thử P4**. Console gốc:
`runs/v060_integrated_checks_05.log` (42.946 s).

`rule_based_v060_probe10_20261030`: **6/6** scenario đạt, không crash, ra đường
hoặc chạm vạch liền. Cả **929 bước** khớp input round-trip, decision/diagnostics
và canonical action khi replay frozen source, loại riêng latency khỏi so sánh.

| Probe | Bước | Điều kiện xác minh |
|---|---:|---|
| safe_overtake | 150 | Vượt xe đứng chắn và hoàn tất đổi làn |
| fast_rear | 170 | Veto lúc xe sau áp sát, đổi sau khi gap an toàn |
| mandatory_route | 171 | Đổi phải theo availability của tuyến |
| mandatory_left | 171 | Đổi trái theo availability của tuyến |
| precommit_abort | 95 | Hủy về làn cũ trước commit khi có rear intrusion |
| postcommit_front_block | 172 | Phanh, giữ target, tiếp tục sau khi blocker được bỏ |

Đây là can thiệp có kiểm soát, không cộng vào tỷ lệ driving success: traffic nền
tắt, actor được spawn/script theo log. Hai mandatory probes hạn chế availability
của next route edge, giữ nguyên physics/rendered lanes và dừng trước edge đó.
Postcommit probe ghi thời điểm spawn/clear blocker; không dùng dữ liệu đánh giá
để quyết định action của agent. Tất cả can thiệp nằm ngoài policy.

CLI chính thức trên Hard/XTOCS/geom0/env0/agent42 cũng COMPLETE, clean success
1/1, 1063 bước (`rule_based_v060_cli_smoke02_20261030`). Đây là case trùng trong
tập 16; không nâng coverage thành 17. Source/config và trace launcher được giữ.

## Những lần chưa đạt

Giữ toàn bộ run lỗi/dừng riêng. Có lỗi setup đọc phiên bản MetaDrive, probe không
đạt điều kiện spawn, diagnostics vượt budget 50 key, endpoint rounding, guard
quá chặt gây dừng giữa thao tác, init-approach gate áp dụng nhầm khi đang đổi,
và một CRASH_VEHICLE ở Extreme sau pass vào approach. Các lỗi này có source,
outcome hoặc ghi chú đúng phạm vi dữ liệu còn lại; không xóa hoặc ghi đè.

Ca crash ở accept02/Extreme SCXOCrTYCS là regression thực, làm candidate đó không
đạt nghiệm thu dù các controlled probes đã qua. Source cuối cấm optional pass
khi đã thấy junction trong 50 m. Các run `accept04_*` và probe10 dùng source đã
sửa; các run trước chỉ là draft, không phải bản release 0.6.0.

## Giới hạn

Chỉ đổi làn kề cùng chiều, không nhảy nhiều làn hoặc đổi trong nút/vòng xuyến;
không vượt qua vạch cấm hoặc dữ liệu UNKNOWN. Chưa có nguồn legal signal/sign
priority được audit. Dự báo 4 s và cảm biến 50 m không biết ý định tương lai hay
actor chưa quan sát. Postcommit stop là lựa chọn bảo thủ, không chứng minh tối
ưu rủi ro toàn cục. Những timeout còn lại phải giữ đúng nhãn và không ép ga.

Act p95 cao nhất là **72.951 ms**, vượt mục tiêu 10 ms; đo trong lúc chạy song song
phát triển, chỉ bao gồm `act`, không phải end-to-end realtime benchmark. P5 cần
kiểm chứng rộng, controller/latency, audit contract 1.3.0 và protocol canonical.
Benchmark eligibility vẫn false, không tuyên bố đạt mục tiêu >=95%.
