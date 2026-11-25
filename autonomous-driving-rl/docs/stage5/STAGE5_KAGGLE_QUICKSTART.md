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

Lệnh hỏi xác nhận chủ tài khoản rồi nhận API token bằng input ẩn. Làm tương tự cho `account_02` đến `account_08` khi cần. Token bảo vệ bằng Windows DPAPI theo Windows user trong `%LOCALAPPDATA%\XeTuHanh\Kaggle\`. Số phút và concurrency được chủ tài khoản quan sát trực tiếp, có hiệu lực lập lịch trong 24 giờ. Thành viên có thể chạy launcher với tài khoản của mình. [Kaggle Terms](https://www.kaggle.com/terms) · [Kaggle API](https://www.kaggle.com/docs/api)

Với tệp local do chủ tài khoản chỉ định và có đúng 8 access token, `control import-slots --file <đường-dẫn>` xác minh từng danh tính và lưu vào DPAPI theo thứ tự xuất hiện. Slot có sẵn được giữ nguyên. `control doctor` lấy quota GPU/TPU từ API. Sau khi kiểm tra số phiên còn trống trong Kaggle Session Management, cập nhật số quan sát mà không nhập lại token:

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
python -B -m cloud.kaggle.control plan --campaign stage5 --round preflight02 --account account_01
```

Kế hoạch `preflight02` có J01 parity CPU, J02 CPU1, J03 CPU2, J04 GPU2, tối đa 20 phút/job và chừa 3 phút cho lưu/export. Bắt đầu bằng J01 để thu bằng chứng runtime trước khi dùng thêm tài nguyên. Ví dụ đọc fingerprint của kế hoạch có sẵn:

```powershell
$plan = Get-Content runs/kaggle/plan-stage5-preflight02.json -Raw | ConvertFrom-Json
python -B -m cloud.kaggle.control prepare-inputs --campaign stage5 --round preflight02 --job j01 --fingerprint $plan.fingerprint
python -B -m cloud.kaggle.control launch --campaign stage5 --round preflight02 --job j01 --fingerprint $plan.fingerprint
$job = $plan.jobs.j01.job
python -B -m cloud.kaggle.control status --job $job
python -B -m cloud.kaggle.control harvest --job $job
python -B -m cloud.kaggle.control analyze --job $job
```

Nếu job lỗi, `control logs --job <job>` lấy log của version đã xác nhận và lưu nhóm lỗi kỹ thuật. J02–J04 và J07 yêu cầu artifact J01 xác minh G2 cùng campaign và runtime. Kế hoạch mới hỗ trợ `--assign j03=account_02 --assign j04=account_03` cho các slot có phiên còn trống đã xác minh. Dataset được chuẩn bị private theo account được giao; không dùng public sharing để chuyển artifact.

Trong lần đầu, Kaggle có thể trả trạng thái dataset đang được tạo. Chạy lại lệnh launch khi dataset version 1 đã sẵn sàng. Khi registry ghi `SUBMIT_UNKNOWN`, đối chiếu notebook slug/version trên Kaggle trước khi làm gì thêm. Khi Kaggle chấp nhận kernel, máy local có thể tắt. Registry ghi `ref`, version, thời điểm UTC/ICT và trạng thái. Output chỉ được đánh dấu `HARVESTED_VERIFIED` khi đúng attempt, đúng checksum và job có cập nhật SAC. [Kaggle kernel commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)

## Resume trong Kaggle session mới

Sau khi harvest được online state, giữ nguyên bundle và tạo round tiếp theo cho J07:

```powershell
$parent = (Get-Content runs/kaggle/plan-stage5-preflight02.json -Raw | ConvertFrom-Json).jobs.j01.job
python -B -m cloud.kaggle.control plan --campaign stage5 --round resume01 --account account_01 --parent $parent
$resume = Get-Content runs/kaggle/plan-stage5-resume01.json -Raw | ConvertFrom-Json
python -B -m cloud.kaggle.control launch --campaign stage5 --round resume01 --job j07 --fingerprint $resume.fingerprint
python -B -m cloud.kaggle.control harvest --job $resume.jobs.j07.job
```

## Hợp đồng input/output Kaggle

| Private input | Mount mặc định | Artifact |
|---|---|---|
| `<owner>/s5-runtime-<20 ký tự SHA>` version 1 | `/kaggle/input/s5-runtime-<sha20>` | `runtime.zip`, gồm source, MapSuite manifests, dependency pins và parity anchor |
| `<owner>/s5-source-96b6b78cc1e89ddf967d` version 1 | `/kaggle/input/s5-source-96b6b78cc1e89ddf967d` | `model.zip`, nguồn 100k |
| `<owner>/s5-parent-<sha20>` version 1, J07 | `/kaggle/input/s5-parent-<sha20>` | `parent.zip`, state từ lần harvest đã xác minh |

Kernel cũng kiểm tra mount theo dạng `/kaggle/input/datasets/<owner>/<slug>/` nếu image dùng namespace này. Mỗi file được kiểm SHA-256 trước khi đọc. Internet cần cho pip, tải source MetaDrive ở commit đã pin và release assets 0.4.3. Source dự án và checkpoint lấy từ private inputs. Output là `/kaggle/working/<job>.zip` với `model.zip`, `replay.pkl`, `rng.pt`, metrics, runtime versions, parity, execution identity và marker checkpoint. Trạng thái trên Kaggle và việc tải output theo phiên bản cần được chứng minh ở vòng cloud.

## Tiếp tục trong phiên Codex sau

> Tiếp tục Stage 5 Kaggle từ `autonomous-driving-rl/runs/kaggle/job_registry.json` và `plan-stage5-preflight02.json`. Đọc gate report, chạy doctor, kiểm tra account và private input version. Submit J01 nếu đủ điều kiện hoặc kiểm tra version hiện có và harvest nếu đã hoàn tất. Chỉ submit J02–J04 và J07 khi dependency và quota cho phép. Báo receipt, lỗi hoặc blocker rồi kết thúc phiên. Giữ G2–G4 cho phép trước mọi phiên dài và giữ B1 cloud ở trạng thái chờ chứng minh transfer. Credential nhập ngoài chat.

