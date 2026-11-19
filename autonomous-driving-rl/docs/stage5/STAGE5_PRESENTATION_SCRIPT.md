# Kịch bản trình bày Stage 5

Thông điệp xuyên suốt: **nhóm đã xây dựng được pipeline SAC có thể huấn luyện, lưu checkpoint, suy luận và kiểm tra; bằng chứng hiện tại chưa cho thấy policy lái xe thành công.** Không gọi B là winner và không gọi các probe bị censor là timeout hay an toàn.

## Bản khoảng 5 phút

### 1. Câu hỏi nghiên cứu (0:00–0:35)

**Nói:** “Stage 5 nghiên cứu reinforcement learning không mô hình cho xe tự hành. Câu hỏi là liệu SAC có thể học điều khiển steering và tốc độ liên tục bằng cách tương tác với MetaDrive và nhận reward hay không. Training chạy xong chưa đồng nghĩa xe đã lái tốt; ta phải đo hành vi.”

**Chiếu/mở:** sơ đồ tám Stage của dự án và Workbench.  
**Chuyển ý:** “Trước tiên, agent được quan sát và điều khiển những gì?”

### 2. Observation và action (0:35–1:20)

**Nói:** “Policy nhận observation có cấu trúc, mã hóa thành 409 số thực: 259 core, 66 traffic và 84 task/waypoint. Action có hai số: steering và throttle/brake, mỗi số từ −1 đến +1. Agent ra quyết định 10 lần mỗi giây; mỗi quyết định được MetaDrive giữ qua năm physics substeps, mỗi bước 0,02 giây.”

**Mở code:** [encoder 409D](../../src/training/stage5/observation_encoder.py), [action adapter](../../src/platform/action_adapter.py).  
**Chuyển ý:** “Action làm thay đổi simulator; reward biến kết quả thành tín hiệu học.”

### 3. Reward và SAC (1:20–2:15)

**Nói:** “RewardSpecV1 cộng route progress, trừ time cost và cộng thưởng/phạt terminal. SAC có actor đề xuất action và hai critic ước lượng giá trị dài hạn. SB3 lưu trải nghiệm vào replay buffer rồi lấy minibatch để cập nhật. SAC còn có entropy để duy trì tính ngẫu nhiên trong học.”

“Reward không phải điểm benchmark. Ta xem riêng clean success, safety, route completion và tốc độ. Actor loss hay critic loss giảm không chứng minh xe đã lái tốt.”

**Mở code:** [RewardSpecV1](../../src/platform/reward.py), sau đó xem sơ đồ trong [tài liệu giải thích](STAGE5_TEACHER_EXPLANATION.md).  
**Chuyển ý:** “Vậy đối chứng cuối S5.3M thực sự cho thấy điều gì?”

### 4. Kết quả S5.3M (2:15–3:15)

**Nói:** “Hai nhánh SAC mới được huấn luyện 10.000 environment steps mỗi nhánh, cùng seed 101 và protocol; chỉ khác tau: 0,005 so với 0,001. Các checkpoint được so trên cùng tám geometry TRAIN nội bộ.”

“Ở mốc 10k, route-progress delta ghép cặp của B cao hơn A trung bình 0,002675. Nhưng cả hai stationary 100% theo ngưỡng dưới 1 km/h, không có clean success, và cả tám probe mỗi nhánh đều bị censor ở giới hạn 300 bước. Đây là tín hiệu nhỏ, chưa phải chiến thắng.”

**Chiếu:** [báo cáo S5.3M](../../results/stage5/s5_3m_tau_ab_seed101_20261119_retry1/FINAL_REPORT.md) và biểu đồ paired trong Results → Comparison.  
**Chuyển ý:** “Ta hãy xem các record và provenance trực tiếp trong Workbench.”

### 5. Demo Workbench (3:15–4:35)

1. Mở **7. SAC Training**; chọn `s5_3d_tierrr_gamma099_seed101_20261115`, checkpoint `10000`.
2. Chỉ các trường steps, episode đã hoàn tất, số geometry, phân bố action, route progress và loss. Nói rõ: “Đây là telemetry huấn luyện; loss không phải điểm lái xe.”
3. Mở **6. Results → Comparison**; chọn SAC A và SAC B ở 10k.
4. Chỉ các case ghép cặp, checkpoint SHA và nhãn `PROBE_CENSORED`. Đọc banner **TRAIN ONLY — NOT BENCHMARK**.
5. Nếu còn thời gian, trở lại SAC Training và bấm **Mở simulator**. Giải thích đây là policy deterministic đã đóng băng trên geometry TRAIN nội bộ. Bản xem trước này không có outcome nếu đóng trước khi episode kết thúc.

### 6. Kết luận (4:35–5:00)

**Nói:** “Nhóm đã hoàn thành pipeline SAC và công cụ theo dõi, chẩn đoán, so sánh. Tuy nhiên policy đo được chưa chứng minh khả năng lái thành công. Stage 5 vẫn là candidate, chưa đủ điều kiện benchmark. Kết quả hiện tại cần được trình bày trung thực như một kết quả âm về driving performance, cùng một chênh lệch progress nhỏ chưa được xác nhận độc lập.”

## Bản khoảng 12–15 phút

### 1. Động cơ và vị trí Stage 5 (0:00–1:30)

**Nói:** “Dự án so sánh các phương pháp AI trong cùng simulator và cùng scientific contracts. Rule-Based thi hành luật do người viết; Planning tìm kiếm đường/hành động; Imitation Learning học từ ví dụ; Model-Free RL học từ tương tác và reward. Stage 5 hỏi liệu tương tác có tạo ra policy điều khiển liên tục hay không.”

“Chọn SAC vì action của xe liên tục, còn SAC hỗ trợ off-policy replay và entropy regularization. Đây là baseline phù hợp để kiểm tra, không phải lời hứa rằng SAC chắc chắn sẽ thắng.”

**Chiếu/mở:** sơ đồ Stage và phần mở đầu [tài liệu giảng giải](STAGE5_TEACHER_EXPLANATION.md).

### 2. MDP và ranh giới observation/action (1:30–3:30)

**Nói:** “Trạng thái nội bộ simulator phong phú hơn dữ liệu policy được nhận. Policy nhận public `AgentInputV1`, gồm thông tin xe, traffic và task, rồi encoder biến thành vector 409 chiều. Các trường evaluator-private như clean success, reward, case ID, TEST metadata hay trạng thái tương lai không đưa vào policy.”

“Output là `[steering, throttle_brake]`. Steering âm là trái, dương là phải; throttle/brake âm là phanh, dương là ga. Adapter giữ nguyên thứ tự và dấu theo canonical contract.”

**Mở code:** `src/platform/agent_input.py`, `src/training/stage5/observation_encoder.py`, `src/platform/action_adapter.py`. Chỉ validity masks trong schema.

### 3. Reward và episode (3:30–5:00)

**Nói:** “Mỗi quyết định nhận `delta_route_completion - 0.25 / horizon_steps + terminal_reward`. Progress dương rất nhỏ vẫn có thể không đủ bù time cost. Clean success và safety failure có terminal reward riêng; timeout có terminal penalty bằng 0 nhưng vẫn kết thúc episode theo semantics timeout.”

“Reward là mục tiêu training; clean success, safety, route completion và time-to-success là metrics đánh giá. Chúng trả lời các câu hỏi liên quan nhưng khác nhau.”

**Mở code:** `src/platform/reward.py`, `src/training/stage5/env_adapter.py::Stage5TrainingEnv.step`; dùng ví dụ số trong tài liệu giải thích.

### 4. Vòng học SAC (5:00–7:45)

**Nói:** “Actor tạo action từ policy squashed Gaussian; hai critic ước lượng action-value. Môi trường trả observation và reward mới; SB3 lưu transition vào replay buffer rồi lấy batch để cập nhật. Critic tối thiểu hóa sai số Bellman; actor tối ưu Q có xét entropy; SB3 tự điều chỉnh alpha. Target critics cập nhật mềm theo tau.”

“Gamma điều khiển mức chiết khấu phần thưởng xa. Tau điều khiển tốc độ target network theo online critic. S5.3M chỉ thay tau. Loss là telemetry tối ưu hóa, không phải chỉ số route hay safety.”

**Mở code/cấu hình:** `src/training/stage5/sac_trainer.py`, `configs/experiments/stage5_tau_ab_v1.json`. Có thể chỉ `learning_starts=1000`, một gradient step mỗi env step sau warm-up, `batch_size=256`, `gamma=.99`, và hai tau.

### 5. Dữ liệu, split và ngân sách (7:45–9:15)

**Nói:** “MapSuite V1 có 240 geometry: 180 TRAIN, 48 VALIDATION, 12 TEST. Bên trong TRAIN, pilot tách 156 geometry train và 24 diagnostic, cân bằng 39/6 mỗi tier. S5.3M dùng lịch cố định tám geometry, hai mỗi tier. Geometry-generation seed và environment seed là hai đại lượng khác nhau.”

“10.000 steps là 10.000 quyết định điều khiển, không phải 10.000 episode. Mỗi nhánh S5.3M chỉ hoàn tất năm training episode. Checkpoint 0/5k/10k là các snapshot của cùng quá trình liên tục.”

**Mở:** `results/stage5/pilot_split_v1.json`, `configs/experiments/stage5_tau_ab_v1.json` và manifest S5.3M.

### 6. Phần SB3 cung cấp và phần nhóm tích hợp (9:15–10:15)

**Nói:** “Stable-Baselines3 cung cấp thuật toán SAC, networks, replay sampling, target updates và alpha tuning. Nhóm xây/integrate environment tương thích Platform, encoder, TRAIN sampler, reward/episode plumbing, callback, checkpoint integrity, inference agent đóng băng, diagnostic runners và Workbench.”

**Mở:** code walkthrough trong [tài liệu giảng giải](STAGE5_TEACHER_EXPLANATION.md#7-code-walkthrough).

### 7. Bằng chứng S5.1–S5.3M (10:15–12:15)

**Nói:** “S5.1 dựng môi trường và ranh giới dữ liệu; S5.2 tích hợp SAC và checkpoint. S5.3 đo hành vi thay vì chỉ nhìn thấy weights đổi. Pilot đầu phát hiện policy deterministic gần như đứng yên. Các chẩn đoán sau kiểm tra action mapping, reward accounting, steering/tốc độ, replay fidelity và critic.”

“S5.3L đo chênh lệch Q với soft return trên 96 rollout của policy đóng băng. Đó là dấu hiệu cần quan tâm trong mẫu ấy, chưa phải bằng chứng nguyên nhân đứng yên. S5.3M thay tau: B có route progress cao hơn 0,002675 trên tám case ghép cặp, nhưng cả hai vẫn stationary 100%, không clean success và đều censor 8/8.”

**Chiếu:** `docs/stage5/STAGE5_FINAL_EXPERIMENT.md`, sau đó A/B Comparison. Không trình bày đường cong S5.3A như kết quả ngang hàng nếu chưa nhắc rằng log đó có trước reset-reward fix.

### 8. Demo và giới hạn (12:15–14:15)

1. Mở **7. SAC Training**, chọn COMPLETE run S5.3D và checkpoint 10k. Nói đây là run lịch sử dùng để trình diễn telemetry; A/B cuối được xem riêng.
2. Mở **6. Results → Comparison**, chọn A/B 10k; giải thích provenance và lịch paired.
3. Chỉ nhãn `PROBE_CENSORED` và banner **TRAIN ONLY — NOT BENCHMARK**.
4. Tùy thời gian, bấm **Mở simulator** để xem policy đóng băng. Nếu dừng giữa episode, không có outcome record.

**Nói:** “Chỉ có một training root, tám geometry diagnostic và probe bị giới hạn 300 steps. Chưa có official Stage 5 benchmark, cũng chưa có bằng chứng policy lái thành công. Checkpoint binding và xác nhận qua các training root độc lập là điều kiện trước chứng nhận.”

### 9. Kết luận (14:15–15:00)

**Nói:** “Sản phẩm nghiên cứu hiện tại là pipeline SAC có thể kiểm tra và kết quả đo được một cách trung thực, không phải autonomous driver đã thành công. Điểm quan trọng là ta phân biệt được pipeline chạy với policy thực sự có kỹ năng.”
