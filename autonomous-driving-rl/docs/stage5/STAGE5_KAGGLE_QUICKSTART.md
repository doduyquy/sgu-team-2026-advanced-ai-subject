# Stage 5 Kaggle preflight

Chạy từ thư mục `autonomous-driving-rl` trong PowerShell. CLI điều phối dùng Python 3.10+ và Kaggle CLI 2.2.4. Các lệnh mô phỏng local dùng môi trường `metadrive_sb3`.

## Nhập token trực tiếp trên Windows

```powershell
$owner = Read-Host 'Kaggle username của chủ tài khoản'
$minutes = [int](Read-Host 'Số phút GPU còn khả dụng theo Kaggle UI, nhập 0 nếu chưa rõ')
$slots = [int](Read-Host 'Số phiên song song còn khả dụng theo Kaggle UI, nhập 0 nếu chưa rõ')
python -B -m cloud.kaggle.control setup --account account_01 --owner $owner --gpu-minutes $minutes --concurrency $slots
python -B -m cloud.kaggle.control doctor
```

Mỗi thành viên chạy launcher với tài khoản của mình trong Windows user riêng. Chọn alias thông qua `STAGE5_KAGGLE_OPERATOR_SLOT` trước khi setup và vận hành. Launcher mặc định dùng `account_01`. Windows DPAPI bảo vệ token của operator. Chia sẻ bundle hoặc artifact qua quyền private đã được cấp cho thành viên. [Kaggle Terms](https://www.kaggle.com/terms) · [Kaggle API](https://www.kaggle.com/docs/api)

`control doctor` lấy quota GPU/TPU của operator từ API. Sau khi kiểm tra số phiên còn trống trong Kaggle Session Management, cập nhật số quan sát mà không nhập lại token. Quan sát có hiệu lực 24 giờ.

```powershell
python -B -m cloud.kaggle.control observe --account account_01 --available-sessions 1
```

## Đóng gói và xem kế hoạch

Bundle hiện có được chỉ ra trong `runs/kaggle/bundle.json`. Khi đóng lại từ source mới, dùng tên round mới để giữ kế hoạch cũ.

```powershell
$py = 'C:/Users/ADMIN/anaconda3/envs/metadrive_sb3/python.exe'
$source = 'runs/stage5_native_reward/stage5_native_reward_easy_scs0_seed101_20261121_retry1/checkpoints/step_100000/model.zip'
& $py -B -m cloud.kaggle.runtime parity --source $source --out runs/kaggle/parity.json
python -B -m cloud.kaggle.control package
python -B -m cloud.kaggle.control plan --campaign stage5 --round activation02 --account account_01
```

Kế hoạch `activation02` có J01 parity CPU, J02 CPU1, J03 CPU2, J04 GPU2, tối đa 20 phút/job và chừa 3 phút cho lưu/export. Bắt đầu bằng J01 để thu bằng chứng runtime trước khi dùng thêm tài nguyên. Ví dụ đọc fingerprint của kế hoạch có sẵn:

```powershell
$plan = Get-Content runs/kaggle/plan-stage5-activation02.json -Raw | ConvertFrom-Json
python -B -m cloud.kaggle.control prepare-inputs --campaign stage5 --round activation02 --job j01 --fingerprint $plan.fingerprint
python -B -m cloud.kaggle.control launch --campaign stage5 --round activation02 --job j01 --fingerprint $plan.fingerprint
$job = $plan.jobs.j01.job
python -B -m cloud.kaggle.control status --job $job
python -B -m cloud.kaggle.control harvest --job $job
python -B -m cloud.kaggle.control analyze --job $job
```

Nếu chưa rõ còn bao nhiêu phiên CPU, thêm `--admission-probe` vào lệnh `launch` của J01. Launcher tra danh sách notebook và kiểm tra các lượt chạy gần đây qua API. Kaggle quyết định tiếp nhận vì API chỉ công bố quota GPU/TPU. Launcher gửi tối đa một lần cho mỗi attempt. Quan sát còn 0 phiên vẫn chặn submit. [Kaggle Kernels CLI](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/kernels.md)

Để thử lại sau lỗi runtime, tạo round mới với bundle đã sửa và chạy `launch --admission-probe --retry-of <failed-job-id>`. Launcher kiểm tra attempt trước là probe gần nhất, cùng tài khoản, đã được API xác nhận đúng version private ở trạng thái `FAILED`, và registry đã ghi chẩn đoán qua lệnh `logs`. Bundle phải có fingerprint mới và có thay đổi tệp liên quan đến nguyên nhân đã chẩn đoán. `CAPACITY_REJECTED`, `SUBMIT_UNKNOWN` và các job active đều chặn lượt thử lại theo cơ chế này.

Nếu job lỗi, `control logs --job <job>` lấy log của version đã xác nhận và lưu nhóm lỗi kỹ thuật. J02–J04 và J07 yêu cầu artifact J01 xác minh G2 cùng campaign và runtime. Kế hoạch mới hỗ trợ `--assign j03=account_02 --assign j04=account_03` cho các slot có phiên còn trống đã xác minh. Dataset được chuẩn bị private theo account được giao; không dùng public sharing để chuyển artifact.

Sau G2, tạo plan chỉ gồm J02–J04 bằng `plan --capacity-after <verified-j01-id>`. Launcher dùng runtime và budget đã ghi trong receipt. Khi API chưa cho biết số phiên còn lại, `launch --server-admission` cho phép một preflight phụ thuộc được Kaggle xét tiếp nhận trên tài khoản sau khi kiểm tra các lượt chạy gần đây. Launcher chặn gửi khi còn job active, receipt chưa rõ hoặc có lỗi capacity. Dùng tài khoản của chính người vận hành theo [điều khoản Kaggle](https://www.kaggle.com/terms).

Trong lần đầu, Kaggle có thể trả trạng thái dataset đang được tạo. Chạy lại lệnh launch khi dataset version 1 đã sẵn sàng. Khi registry ghi `SUBMIT_UNKNOWN`, đối chiếu notebook slug/version trên Kaggle trước khi làm gì thêm. Khi Kaggle chấp nhận kernel, máy local có thể tắt. Registry ghi `ref`, version, thời điểm UTC/ICT và trạng thái. Output chỉ được đánh dấu `HARVESTED_VERIFIED` khi đúng attempt, đúng checksum và checkpoint xác nhận số SAC updates. [Kaggle kernel commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)

## Easy learning pilot

`cloud.kaggle.capacity --round capacity04 --cpu1-reference <J01>` tạo bảng số đo cùng ngân sách. `control pilot-plan --campaign stage5 --round easypilot01 --capacity-round capacity04` yêu cầu G2–G4 đã xác minh. Pilot lấy dữ liệu từ Easy trong `pilot_train`, dùng 8.192 transitions với ngân sách 20 phút và xuất checkpoint giữa kỳ ở bước 4.096. Job chạy parity trước khi train. Xuất và kiểm tra cả hai mốc bằng `control harvest` và `control harvest-periodic --job <J08> --step 4096`. Đánh giá hai model bằng `runtime evaluate --tier Easy` rồi dùng `cloud.kaggle.pilot` ghép từng case với baseline 8/12. Panel Easy lấy từ `pilot_diagnostic`.

`runtime train` hỗ trợ phase, ngân sách thời gian và checkpoint qua `--checkpoint-every` cùng `--checkpoint-dir`. `cloud.kaggle.checkpoint export` chia state thành các phần tối đa 64 MiB, với checksum riêng cho từng phần. `runtime train --resume-package <thư-mục> --resume-index-sha <sha>` kiểm tra rồi dựng lại toàn bộ state trước khi nạp. Pilot ngắn dùng ZIP cho checkpoint cuối. Phiên dài cần xác minh đường vận chuyển và resume cloud với artifact lớn.

## Snapshot B1

`control publish-b1-snapshot --snapshot <thư-mục> --snapshot-sha <sha>` đối chiếu policy nguồn, lineage, version và recipe với bundle đã đóng gói. Lệnh tạo private dataset version 1 và tải lại `snapshot.bin` để kiểm SHA-256. Learner kiểm snapshot được cấp qua `replay import --snapshot <thư-mục> --snapshot-sha <sha>` và tính tuổi policy từ checkpoint learner đã xác minh.

## Resume trong Kaggle session mới

Sau khi harvest được online state, giữ nguyên bundle và tạo round tiếp theo cho J07:

```powershell
$parent = (Get-Content runs/kaggle/plan-stage5-activation02.json -Raw | ConvertFrom-Json).jobs.j01.job
python -B -m cloud.kaggle.control plan --campaign stage5 --round resume01 --account account_01 --parent $parent
$resume = Get-Content runs/kaggle/plan-stage5-resume01.json -Raw | ConvertFrom-Json
python -B -m cloud.kaggle.control launch --campaign stage5 --round resume01 --job j07 --fingerprint $resume.fingerprint
python -B -m cloud.kaggle.control harvest --job $resume.jobs.j07.job
```

## Hợp đồng input/output Kaggle

| Private input | Mount mặc định | Artifact |
|---|---|---|
| `<owner>/s5-runtime-<20 ký tự SHA>-bin` version 1 | `/kaggle/input/s5-runtime-<sha20>-bin` | `runtime.bin`, chứa nguyên byte `runtime.zip`, gồm source, MapSuite manifests, dependency pins và parity anchor |
| `<owner>/s5-source-96b6b78cc1e89ddf967d-bin` version 1 | `/kaggle/input/s5-source-96b6b78cc1e89ddf967d-bin` | `model.bin`, chứa nguyên byte checkpoint 100k |
| `<owner>/s5-parent-<sha20>-bin` version 1, J07 | `/kaggle/input/s5-parent-<sha20>-bin` | `parent.bin`, chứa nguyên byte `parent.zip`, state từ lần harvest đã xác minh |

Kernel cũng kiểm tra mount theo dạng `/kaggle/input/datasets/<owner>/<slug>/` nếu image dùng namespace này. Kaggle tự giải nén file upload mang đuôi .zip. Launcher tải lại private version 1 để xác minh tên tệp .bin và SHA-256. Kernel kiểm lại SHA-256 từ mount trước khi đọc. Internet cần cho pip, tải source MetaDrive ở commit đã pin và release assets 0.4.3. Source dự án và checkpoint lấy từ private inputs. Output là `/kaggle/working/<job>.zip` với `model.zip`, `replay.pkl`, `rng.pt`, metrics, runtime versions, parity, execution identity và marker checkpoint. Trạng thái trên Kaggle và việc tải output theo phiên bản cần được chứng minh ở vòng cloud.

## Tiếp tục trong phiên Codex sau

> Tiếp tục Stage 5 Kaggle từ `autonomous-driving-rl/runs/kaggle/job_registry.json` và `plan-stage5-activation02.json`. Đọc gate report, chạy doctor, kiểm tra account và private input version. Submit J01 nếu đủ điều kiện hoặc kiểm tra version hiện có và harvest nếu đã hoàn tất. Chỉ submit J02–J04 và J07 khi dependency và quota cho phép. Báo receipt, lỗi hoặc blocker rồi kết thúc phiên. Giữ G2–G4 cho phép trước mọi phiên dài và giữ B1 cloud ở trạng thái chờ chứng minh transfer. Credential nhập ngoài chat.

