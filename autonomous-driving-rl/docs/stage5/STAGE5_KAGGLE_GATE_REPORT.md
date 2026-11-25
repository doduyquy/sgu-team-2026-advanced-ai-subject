# Stage 5 Kaggle handoff · 2026-11-25

## Easy learning pilot J08

Hai checkpoint version 1 được đánh giá trên đúng 12 case Easy của baseline đóng băng. Mốc 4.096 giữ 8/12 clean, giảm 4 ca văng đường thành 4 timeout và nhanh hơn 31,2 giây trên 8 ca thành công chung. Mốc 8.192 đạt 0/12 clean. Tám ca từng clean chuyển thành văng đường khi đã đi qua 92,8–96,1% route. J08 chỉ ghé 7/39 hình học Easy TRAIN qua ba family. Báo cáo ghép case nằm trong `runs/kaggle/easypilot01-paired-report.json`.

Lượt học tiếp theo dùng mốc 4.096 làm warm-start weights-only cho T4 và hai môi trường, với 25.000 transitions và checkpoint ở mỗi 12.500. Learner dựng replay, optimizer và sampler mới. Đánh giá tín hiệu học giữa kỳ và cuối trên cùng panel trước khi cân nhắc mốc 50.000.

## Cập nhật cloud activation04

**G2 PASS.** Output version 1 của J01 `s5-910b612f2a22-j01-a1` đã được tải bằng API theo đúng version. Artifact khớp checksum nội bộ, parity trên cả hai hình học TRAIN và counter của checkpoint SAC. Job ghi 1.280 transitions, 256 SAC updates, PyTorch 2.14.1+cpu và CPython 3.13.18. SHA-256 của `COMPLETE.json` là `18fa92be8690cb5683ed24d524d12aabf8e81fc5f87d6a46056dff8740623fcc`. Kế hoạch capacity và resume dùng bundle đã vượt G2. Registry lưu tiến độ mới nhất của G3 và G4.

B1 local đã nạp 256 transitions thật vào learner hai môi trường, chạy 8 SAC updates và đo thay đổi ở actor, critic, target. Chunks phải chia hết cho số môi trường. Kiểm chứng transfer qua Kaggle vẫn thuộc G5.

**G3 PASS.** J07 version 1 giữ nguyên 1.280 dòng đầu của replay, tiếp tục đến 2.560 transitions, tăng actor, critic và entropy optimizer từ 256 lên 1.536 bước, và tăng sampler cursor từ 2 lên 5. Resume reset ở biên episode.

**G4 PASS.** Ba mẫu có cùng runtime, checkpoint nguồn và ngân sách 1.280 transitions, 256 updates. Dùng J01 làm CPU1 thay cho J02. CPU1, CPU2 và Tesla T4 với hai môi trường lần lượt đạt 8,38 / 7,98 / 5,47 transitions/s tính từ khởi động đến log cloud cuối và 19,34 / 20,43 / 57,86 transitions/s sau warm-up. CPU1 được chọn cho Easy pilot ngắn đầu tiên.

## Cập nhật cloud activation02

Kaggle tiếp nhận J01 `s5-94d391e6b785-j01-a1` version 1 lúc 18:56:53 ICT sau một lần thử cấp phiên CPU. Job chuyển sang `RUNNING`, sau đó dừng ở kiểm tra Python ABI trước khi cài dependency hoặc chạy parity. Cả 8 account đã xác minh danh tính. G2–G4 vẫn `BLOCKED`. Phiên bản kế tiếp hỗ trợ CPython 3.13 trên Linux x86-64 và cần bundle cùng fingerprint mới. Một lần probe khi capacity chưa xác định đã được sử dụng trong campaign `stage5`.

## Cập nhật cloud activation03

J01 `s5-d77e2268cbc3-j01-a1` version 1 đã vượt kiểm tra Python 3.13 và tải được MetaDrive assets. Tiến trình parity dừng với `SIGSEGV` sau khi pip cài PyTorch kèm CUDA vào môi trường gốc của image CPU và báo xung đột với các gói có sẵn. Log của phiên này chưa chỉ ra native frame gây lỗi. Runtime kế tiếp dùng virtualenv riêng, wheel PyTorch CPU từ nguồn chính thức cho job CPU, `pip check` và Python faulthandler. G2 vẫn chờ artifact đã kiểm chứng.

Các mục dưới đây lưu kết quả của đợt local trước khi kích hoạt cloud.

**Đợt 1 PARTIAL. Cloud READY_FOR_CREDENTIALS.** Kaggle CLI 2.2.4 có sẵn. `account_01`–`account_08` đều MISSING. Owner, quota, ACL, kernel/version và thời gian chạy thực tế chưa được xác minh. Registry có 4 job `PLANNED`, 0 job đã gửi. J05/J06 chờ xác minh B1 cloud, J07 chờ state cha đã harvest.

## A. Repository và bảo toàn

Base local `10785d3aedb4bbd4600965a98b7c06c93011c0cf`. Nhánh làm việc `feat/stage5-kaggle-parallel-training`. Remote F09 base `2c572ea620e4254667665814f74485ddf57815e6` đã đối chiếu. Dùng adapter V1 có sẵn trong checkout cùng test rồi đưa vào nhánh review với provenance riêng. Bản sao patch, untracked source ban đầu và danh mục artifacts ở `runs/kaggle/inventory-20261125/`. Phần sửa evaluator có sẵn và các thử nghiệm anti-stall tiếp tục ở working tree.

**Sự cố local.** Trong smoke đầu, sáu mảng `.npy` trong `runs/stage5_native_reward/stage5_native_reward_easy_scs0_seed101_20261121_retry1/replay_memmap/` bị factory replay disk-backed của checkpoint mở lại với mode `w+`. Đã xác nhận 100k row đầu của cả sáu mảng bằng 0. 44 artifact có checksum gốc khớp, gồm checkpoint 100k. Chưa tìm thấy bản sao phù hợp dưới TTNTNC. Kiểm tra VSS ổ D: cần Administrator. Factory lịch sử đã bị loại khỏi đường nạp trước deserialize, regression xác nhận sentinel file giữ nguyên. Xem kiến trúc để biết phạm vi ảnh hưởng.

## B. Local evidence

Tất cả lệnh chạy từ `autonomous-driving-rl`.

| Kiểm thử | Kết quả | Bằng chứng local dưới `runs/kaggle/` |
|---|---|---|
| `python -B -m pytest -q tests/test_stage5_kaggle.py tests/test_stage5_sac.py tests/test_stage5_pilot.py tests/test_stage5_terminal_preview_v1.py` trong env `metadrive_sb3` | 40 PASS, exit 0 | Gồm DPAPI round-trip, mock submit timeout, cố định terminal/truncation và vector 1/2/4 synthetic, checkpoint optimizer/alpha continuation và loader chống ghi |
| `runtime parity --source <100k> --out runs/kaggle/parity.json` | Exit 0, SCS0/SCS1 | `parity.json`; exact geometry, native reward, 409D/Box2, cadence, parity action trước trigger |
| `runtime train --source <100k> --workers 1/2 --transitions 48 --learning-starts 16 --capacity 128` | Exit 0 cả hai, 48 transition/32 update mỗi topology | `local-cpu1-v3/`, `local-cpu2-v3/`; khoảng 11.25 và 14.77 decision/s, peak RSS lấy mẫu 381 MB và 937 MB. Mẫu ngắn và startup-sensitive |
| `runtime train --workers 2 --resume runs/kaggle/local-cpu2-v3 --parent-sha <SHA> --transitions 24 --learning-starts 16 --capacity 128` | Exit 0, 24 transition/24 update thêm, 72/56 tích lũy | `local-cpu2-resume-v3/`, UTD cửa sổ 1.0 |
| `replay source-snapshot`, `replay collect --rows 256`, `replay import` và import lại với `--parent` | Exit 0, 256 rows thực và 8 SAC updates. Duplicate bỏ qua | `local-b1-snapshot-v2/`, `local-b1-chunk-v2/`, `local-b1-update-v2/`; actor delta tối đa 0.000807, critic 0.000793, target 0.000657 |
| `runtime evaluate --model <100k> --model-sha <SHA> --tier Easy` | Exit 0, 8/12 clean, 4/12 OUT_OF_ROAD, 0 censored | `local-source-easy-panel.json`; baseline cố định, không dùng VALIDATION hoặc TEST |
| `control doctor`, `control package`, `control plan --campaign stage5 --round preflight02 --account account_01`, `control status` | Exit 0 | `job_registry.json`, `bundle.json`, `plan-stage5-preflight02.json`, `campaign_state.json` |
| Giải nén bundle vào thư mục riêng, chạy `runtime parity --reference parity.json` | Exit 0 trên Windows | `isolated-runtime-v2/verified-parity.json`, dùng source và manifest trong bundle |
| `control launch --campaign stage5 --round preflight02 --job j01 --fingerprint <plan SHA>` | Exit 1 dự kiến, identity slot chưa xác minh | J01 `s5-f640c4e3621b-j01-a1` vẫn PLANNED |

Các lệnh `runtime`, `replay`, `control` là module `python -B -m cloud.kaggle.<module>`. Full suite, GPU local và local MetaDrive 4 worker nằm ngoài tập lệnh trên.

Bundle `runs/kaggle/package-33594d5061eee243faa9/runtime.zip`, 230306 bytes, SHA-256 `9f0a6226b814e0f1422afe5ed1ab0ef11b39fdaf09a7fbfec395803711f4f847`. Checkpoint input riêng 7615798 bytes, SHA-256 `96b6b78cc1e89ddf967df6bf98ead74cb7415553c7fb7c4f07a8265de6750b8f`. `source_inventory.json` ở cùng package lưu fingerprint từng file, Git base và pin. Bundle cũ preflight01 được giữ làm evidence và đánh dấu superseded vì thiếu dependency F09 import muộn.

## C. Gate matrix

| Gate | Trạng thái | Evidence hoặc điều kiện |
|---|---|---|
| G0 Source integrity | PASS | Checkpoint theo SHA yêu cầu, MetaDrive pin, MapSuite manifest khóa, parity 2 geometry. Sự cố replay nguồn được kê riêng ở mục A |
| G1 Local correctness | PASS | 40 targeted tests, CPU1/CPU2 thật, resume process mới |
| G2 Kaggle runtime | BLOCKED | Credential và private input version 1, sau đó J01 |
| G3 Cloud resume | BLOCKED | Cần một state cha harvest đúng version để chạy J07 |
| G4 Cloud capacity | BLOCKED | Chờ J02–J04 sau kiểm chứng runtime và quota |
| G5 External replay | PENDING_CLOUD | B1 local đã chứng minh import/update/dedup. Chờ snapshot và chunk transfer qua private ACL. B2 chưa triển khai |
| G6 Learning pilot | NOT_RUN | Đã đo baseline trên fixed Easy panel. Chờ pilot đối chứng theo cùng budget |
| G7 Scientific benchmark | NOT_RUN | Chờ eligibility và governance, ngoài phạm vi preflight |

## D–F. Vận hành tiếp

Dùng [Quickstart](STAGE5_KAGGLE_QUICKSTART.md) để nhập token trực tiếp, kiểm tra tài khoản, submit J01, rồi theo dõi và harvest qua registry. Chưa có notebook URL, accepted version, runtime charge, cloud result hoặc phiên huấn luyện dài. G2–G4 là điều kiện trước khi cân nhắc chạy online dài.

