# Stage 5 Kaggle Parallel SAC

## Source và recipe

Base local `10785d3aedb4bbd4600965a98b7c06c93011c0cf`, từ `feat/stage5-native-sac-integration`. Bundle chứa file SHA theo từng entry. Checkpoint nguồn có SHA-256 `96b6b78cc1e89ddf967df6bf98ead74cb7415553c7fb7c4f07a8265de6750b8f`. MetaDrive được pin ở `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`. Linux bootstrap dùng pins trong `cloud/kaggle/linux-lock.txt` và cần xác minh resolver, tải asset và runtime trên Kaggle.

Online dùng `Stage5TrainingEnv`, F09, Box2, native reward và encoder 409D. Terminal Preview V1 biến đổi bản sao 409D public trước `pilot_groupwise_physical_v1`. Warm start giữ actor, hai critic và target critics. Replay, optimizer và auto-alpha khởi tạo mới vì distribution quan sát thay đổi. Thử nghiệm ngắn dùng warm-up random chuẩn SB3 và learning rate cố định theo checkpoint. Learner tính transition mới từ 0, metadata ghi source 100k thông qua SHA. Entrypoint inference nạp model nguồn bằng cách thay lớp/kwargs replay nhúng thành replay trong RAM một hàng trước deserialize.

## Online, checkpoint và curriculum

Baseline dùng DummyVecEnv. MetaDrive nhiều worker dùng SubprocVecEnv spawn, mỗi worker một simulator và được khởi tạo bên trong process. Worker chọn case từ split 156/24 qua manifest TRAIN có hash khóa. Smoke dùng SCS0 và SCS1. Lịch phase cố định theo `recipe.json`, family được shuffle dựa trên root và cursor. Panel đóng băng lấy 24 geometry diagnostic, mỗi tier 6 geometry × 2 seed. Metric tính theo toàn episode và lưu từng case. Mốc tham chiếu cho pilot sau là ≥80% clean, ≤20% safety và lặp ở hai mốc cùng coverage. Chuyển phase cần quyết định sau đánh giá.

UTD sau warm-up dùng `gradient_steps=-1`. State lưu weights, optimizer và alpha qua SB3, replay gồm next_obs/timeouts, RNG Python/NumPy/Torch, cursor, counters và ledger. Save tại điểm giữa các rollout/update blocks. Resume giữ worker count, phase, capacity, warm-up, hashes và LR schedule cố định. Worker bắt đầu episode tiếp theo từ cursor đã lưu. Episode dở được ghi trong boundary, các transition thực đã thu vẫn hợp lệ. GitHub và Kaggle không nhận tệp replay nguồn lịch sử. Đích Kaggle giữ replay mới chỉ trong private output.

## External replay

B1 local dùng snapshots được hash-verify và sample stochastic trên CPU. Transport NPZ có header/shape/dtype/bounds kiểm tra trước khi nạp, manifest chứa geometry, seed, snapshot, phiên bản, episode và step. Successor cuối episode được chụp trực tiếp từ env. Terminal ưu tiên hơn truncation. Namespace collector tách worker online, mỗi sequence chiếm một khối cursor cố định và mọi decision ID phải liên tục. Import hiện giới hạn một env để dùng chính xác API `ReplayBuffer.add` sẵn có. Import ghi ledger và replay chung một state, có short upper bound cập nhật và chứng minh actor/critic/target đổi khi buffer chỉ chứa external rows. B1 cloud và B2 còn phụ thuộc private sharing, throughput và learning pilot. A online là đường chạy hiện tại.

## Bảo toàn và sự cố replay nguồn

Bản sao trước chỉnh sửa nằm ở `runs/kaggle/inventory-20261125/`, có patch staged/unstaged và bảy tệp untracked. Working patch SHA-256 `4e42ec86a7d8ba69d52710686a7e19a27f65adb1b4262ae62c5268c604d7d9b6`.

Trong smoke đầu, `SAC.load` dựng lại replay factory disk-backed đã serialize và mở sáu tệp `.npy` dưới `runs/stage5_native_reward/stage5_native_reward_easy_scs0_seed101_20261121_retry1/replay_memmap/` với mode `w+`. Kiểm tra đọc-only cho thấy sáu mảng đều bằng 0 trong vùng 100k row đã dùng. Tất cả 44 file trong manifest artifact nguồn vẫn khớp, gồm checkpoint. Kiểm tra các thư mục dưới TTNTNC chưa thấy replay backup phù hợp. VSS cần quyền Administrator để xác định snapshot D:. Bản sửa đã thay replay factory trước deserialize và có regression test kiểm tra đường nạp không ghi vào file cũ. Không dùng state từ smoke ban đầu làm parent của preflight.

## Cổng và giới hạn

`control` dùng một lock local theo invocation, chuẩn bị private inputs theo hash, lưu trạng thái trước submit và nhận version từ CLI. Resume Kaggle định tuyến qua file harvest chạy độc lập. Bundle local và input checksums xác định software snapshot. Các bằng chứng về Linux/CPU/GPU, remote version semantics, ACL, persistence, throughput và học tổng quát nằm ở các gate cloud/pilot.

