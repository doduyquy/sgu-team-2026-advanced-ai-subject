# Stage 5 — giải thích từ cơ bản đến chuyên sâu

Tài liệu này dành cho sinh viên cần hiểu và trình bày được Stage 5 của XeTuHanh, kể cả khi chưa học sâu về Reinforcement Learning. Nội dung phân biệt rõ **hệ thống đã xây**, **thí nghiệm đã chạy**, **kết quả quan sát**, và **giả thuyết chưa được chứng minh**.

## 1. Stage 5 đang giải bài toán gì?

Một chiếc xe trong MetaDrive cần chọn hành động liên tục để đi theo tuyến đường, điều chỉnh hướng, tăng/giảm tốc và tránh kết thúc bằng một sự cố an toàn. Stage 5 đặt câu hỏi: **một agent có thể cải thiện quyết định lái xe bằng cách tương tác với mô phỏng và nhận tín hiệu thưởng hay không?**

Đây là một hướng khác với các Stage trước:

| Hướng | Agent nhận quyết định bằng cách nào? | Ví dụ dễ hiểu |
|---|---|---|
| Rule-Based (Stage 1) | Người viết luật điều khiển tường minh | “Nếu xe lệch trái thì bẻ phải.” |
| Planning/Search (Stage 2) | Tìm kiếm hoặc tối ưu chuỗi trạng thái/hành động | “Thử các đường đi khả dĩ rồi chọn đường có chi phí thấp.” |
| Imitation Learning (Stage 4) | Học bắt chước cặp quan sát–hành động từ dữ liệu mẫu | “Học từ cách một người hoặc baseline đã lái.” |
| Model-Free RL (Stage 5) | Thử hành động, quan sát kết quả, cập nhật chính sách bằng reward | “Tự học từ tương tác, không cần mô hình chuyển động khả vi do nhóm tự xây.” |

**Model-Free** không có nghĩa agent không dùng MetaDrive hay không có mô hình neural network. Nó có nghĩa thuật toán không cần một mô hình tường minh (s_{t+1}=f(s_t,a_t)) do nhóm cung cấp để lập kế hoạch trước. Nó học giá trị và chính sách từ transition thật do simulator sinh ra.

### Vì sao chọn SAC?

Xe có hai lệnh điều khiển liên tục: steering và throttle/brake. Soft Actor-Critic (SAC) hỗ trợ action liên tục, học off-policy từ replay buffer và giữ thành phần entropy để khuyến khích khám phá. Đây là một baseline hợp lý để thử nghiệm điều khiển kiểu này.

Việc chọn SAC **không đảm bảo SAC sẽ giải được nhiệm vụ**. Kết quả hiện tại cho thấy pipeline cập nhật được mô hình, nhưng policy 10k chưa học được chuyển động hữu ích ổn định.

## 2. Reinforcement Learning và MDP

Một cách mô tả chuẩn là Markov Decision Process (MDP):

\[
\mathcal{M}=(\mathcal S,\mathcal A,P,R,\gamma).
\]

- **State (s_t):** trạng thái đầy đủ của thế giới tại thời điểm (t), ví dụ vị trí và vận tốc xe, xe khác, làn đường và trạng thái simulator.
- **Observation (o_t):** phần thông tin agent được phép nhìn thấy. Trong dự án này agent không nhận toàn bộ nội bộ evaluator; nó nhận `AgentInputV1` rồi được mã hóa thành vector 409 chiều. Vì vậy nên gọi đây là observation, không khẳng định nó chứa đầy đủ mọi biến để tạo Markov state hoàn hảo.
- **Action (a_t):** lệnh steering/throttle-brake được gửi trong một chu kỳ điều khiển.
- **Transition (P(s_{t+1}mid s_t,a_t)):** MetaDrive cập nhật vật lý, xe khác và trạng thái tuyến đường sau khi nhận lệnh.
- **Reward (r_t):** tín hiệu scalar của RewardSpecV1, chủ yếu phản ánh tiến độ, chi phí thời gian và kết quả terminal.
- **Episode:** một lần chạy bắt đầu ở reset và kết thúc do success, safety failure hoặc hết horizon.
- **Policy (pi(amid o)):** quy tắc neural network tạo hành động từ observation.
- **Discount factor (gamma):** trọng số tương lai trong mục tiêu học.
- **Return:** tổng reward theo thời gian. Return có thể dùng bản chiết khấu (G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k}).

Ví dụ: xe đang gần mép làn là một observation. Agent chọn steering hơi sang phải và ga dương. MetaDrive phản hồi bằng tốc độ, observation mới và thay đổi route progress. Reward của bước được thêm vào kinh nghiệm; SAC dùng nhiều kinh nghiệm như vậy để điều chỉnh mạng.

## 3. Observation 409 chiều

Agent không nhận ảnh camera. Platform V1 đã định nghĩa input có cấu trúc; encoder Stage 5 chuyển cấu trúc đó thành `float32[409]` mà MLP có thể xử lý:

| Nhóm | Kích thước | Nội dung |
|---|---:|---|
| Core | 259 | 9 ego-state features, 10 thành phần hướng/checkpoint và 240 tia LiDAR 360° đã chuẩn hóa |
| Traffic | 66 | Tối đa 8 actor × 7 feature (56), validity mask 8 chiều, active-count ratio và overflow ratio |
| Task | 84 | 20 waypoint × 3 feature (60), validity mask 20 chiều, lane-width ratio, goal direction 2 chiều và cờ route-end |
| **Tổng** | **409** | 259 + 66 + 84 |

### Mask và padding

Traffic context chỉ giữ tối đa 8 actor và Task context giữ tối đa 20 waypoint. Khi ít đối tượng hơn sức chứa, các slot còn lại được đệm bằng zero. **Validity mask** báo slot nào thật sự tồn tại; zero không bị hiểu nhầm thành một xe/waypoint ở vị trí zero.

Các thành phần Core ở `[0,1]`; một số actor và waypoint dùng tọa độ/vận tốc vật lý nên có thể âm hoặc lớn hơn 1. Encoder không cắt tùy tiện các giá trị đó. Encoder có `schema fingerprint` để checkpoint không bị nạp nhầm với cách bố trí feature khác.

### Vì sao không đưa evaluator-private data vào?

`route_completion`, reward, clean-success outcome, terminal reason, case identity, environment seed và trạng thái tương lai là dữ liệu dùng cho reward, sampler hoặc đánh giá; chúng không được đưa trực tiếp thành input bí mật cho policy. Nếu policy được thấy câu trả lời của evaluator, kết quả không còn là cùng bài toán mà agent khác phải giải.

Ở cả training lẫn inference, đường đi là: tạo `AgentInputV1` công khai → dùng cùng `Stage5ObservationEncoder`. Checkpoint metadata lưu fingerprint encoder; agent từ chối checkpoint không khớp. `step_index` là trường public của AgentInput; nó không đồng nghĩa với việc cấp route completion hay outcome cho policy.

## 4. Action hai chiều

Action có dạng:

\[
a_t=[u_{steer},u_{long}],\quad u_{steer},u_{long}\in[-1,1].
\]

- `steering = -1`: bẻ hết trái; `+1`: bẻ hết phải; `0`: giữa.
- `throttle_brake = -1`: phanh mạnh; `0`: không yêu cầu ga/phanh; `+1`: ga mạnh.
- Ví dụ `[0.2, 0.6]`: đánh lái nhẹ sang phải và tăng ga.
- Ví dụ `[-0.4, -0.3]`: đánh lái trái và phanh nhẹ.

Action contract `continuous_box2_v1` kiểm tra shape, hữu hạn và miền giá trị. Adapter là identity mapping; không đổi dấu hoặc thứ tự hai thành phần.

MetaDrive chạy physics step danh nghĩa `0.02 s`, lặp mỗi quyết định 5 physics substeps. Agent ra quyết định ở `0.1 s` một lần, tức 10 Hz. Vậy một Gymnasium/SAC transition ứng với **một** action decision giữ trong năm physics substeps, không phải năm quyết định agent bị gộp thành một transition. K0 đã truy dấu điều này.

Training thường lấy action stochastic để khám phá; evaluation và Stage5SacAgent inference dùng policy deterministic để tạo kết quả ổn định hơn. Hai chế độ không có cùng phân phối hành động.

## 5. RewardSpecV1

Ở cấu hình mặc định hiện dùng:

```text
reward_t = delta_route_completion
           - 0.25 / horizon_steps
           + terminal_reward_t
```

Đầy đủ hơn, (r_t=w_p\Delta p_t-B/H+r_{terminal,t}), với `progress_weight=1`, `time_penalty_budget=0.25`, không clipping mặc định.

- **Progress:** (\Delta p_t=p_t-p_{t-1}), nên có thể âm nếu route progress giảm.
- **Time cost:** `0.25 / horizon_steps` bị trừ ở mỗi decision step.
- **Clean success:** bonus `+1.0` trên transition terminal thành công sạch.
- **Safety failure:** penalty `-1.0` cho crash/out-of-road theo phân loại Platform V1.
- **Unknown termination:** penalty `-1.0`.
- **Timeout:** penalty terminal mặc định `0.0`; hết horizon vẫn kết thúc/truncate episode nhưng không cộng safety penalty.

Ví dụ với horizon (H=1000) và bước không terminal có `delta_route=0.001`:

\[
r=0.001-0.25/1000=0.00075.
\]

Nếu cùng mức tiến độ nhưng bước cuối là safety failure, reward bước đó là `0.001 - 0.00025 - 1 = -1.00025`. Nếu bước cuối là clean success, nó là `0.001 - 0.00025 + 1 = 1.00075`. Nếu timeout, terminal component là zero.

Reward là **training signal**, không phải điểm benchmark. Benchmark dùng các metric Platform V1 như clean success rate, safety failure rate, final route completion và time-to-success. Reward có thể âm dù xe tiến một chút; ngược lại reward tăng không đủ chứng minh xe lái tốt.

## 6. SAC hoạt động thế nào?

### Actor, twin critics, replay buffer

- **Actor (\pi_\theta):** nhận observation và tạo phân phối Gaussian đã squashed qua `tanh`, nên action cuối cùng nằm trong `[-1,1]`. Khi stochastic, nó lấy mẫu nhiễu; khi deterministic, nó dùng trung tâm của phân phối đã học.
- **Twin critics (Q_{\phi_1},Q_{\phi_2}):** dự đoán giá trị dài hạn của state-action. Dùng min của hai critic trong target để giảm một số dạng overestimation; điều này không đảm bảo critic luôn calibration tốt.
- **Replay buffer:** giữ transition ((o_t,a_t,r_t,o_{t+1},done)). SB3 lấy minibatch ngẫu nhiên từ buffer để học lại nhiều lần, nên SAC là off-policy.
- **Off-policy:** dữ liệu có thể được thu từ behavior policy trước đó, không bắt buộc đúng chính policy hiện đang được cập nhật. Đây là lý do replay buffer có ích và cũng là lý do action lưu với transition phải đúng action đã thực thi.

### Soft Bellman target và entropy

SB3 SAC dùng target có dạng:

\[
y_t=r_t+\gamma(1-d_t)\left[\min_i Q_{\bar\phi_i}(o_{t+1},a_{t+1})-\alpha\log\pi(a_{t+1}\mid o_{t+1})\right].
\]

`a_{t+1}` được lấy từ policy stochastic; (d_t) dùng semantics done/timeout của replay. Vì log-probability thường âm, trừ `alpha * log_pi` góp entropy bonus vào target.

Entropy khuyến khích policy giữ một mức đa dạng hành động thay vì sụp quá sớm vào một lệnh. Nó không đồng nghĩa hành động đa dạng là hành động hữu ích: ga/phanh nhiễu vẫn có thể tạo trải nghiệm kém.

Actor được cập nhật theo loss dạng:

\[
J_\pi=\mathbb E[\alpha\log\pi(a\mid o)-\min_i Q_i(o,a)].
\]

`alpha` cân bằng entropy và giá trị. `ent_coef="auto"` yêu cầu SB3 tự điều chỉnh hệ số quanh target entropy; với action dimension 2, target mặc định SB3 là `-2`.

### Gamma và tau

- **Gamma `0.99`:** nhân giá trị tương lai theo khoảng cách. Sau 100 steps, (0.99^{100}\approx0.366); sau 300 steps còn khoảng `0.049`; xa hơn bị giảm mạnh. Gamma cao hơn giữ tín hiệu xa lâu hơn nhưng thay đổi cả độ ổn định/độ dài mục tiêu học.
- **Tau:** Polyak update target network: (\bar\phi\leftarrow\tau\phi+(1-\tau)\bar\phi). `tau=0.005` cập nhật target nhanh hơn `tau=0.001`; tau nhỏ hơn làm target chậm thay đổi, không phải thay đổi trực tiếp steering hay reward.

### Actor loss và critic loss không phải điểm lái xe

Critic loss đo sai khác với Bellman target mà critic tự xây; actor loss là objective tối ưu có Q và entropy. Cả hai có thể thay đổi hoặc giảm trong khi agent vẫn đứng yên. Chúng là dấu hiệu tối ưu hóa, không thay thế route progress, speed, safety và success.

### Phần Stable-Baselines3 và phần nhóm tự làm

Stable-Baselines3 cung cấp SAC core: actor/critic networks, action distribution, replay buffer, sampling/minibatch, Bellman/actor update, entropy coefficient auto-tuning, target network update và các hook callback.

Nhóm tự tích hợp và xây: `Stage5TrainingEnv`, encoder `AgentInputV1` 409D, TRAIN sampler, RewardSpecV1 adapter, callbacks/telemetry, checkpoint metadata/hash verification, `Stage5SacAgent`, pilot/split/evaluation runners và Workbench monitor/comparison. Nhóm **không tự viết lại** optimizer SAC.

### Cấu hình baseline

`configs/experiments/stage5_rl_baseline_v1.json` đặt `learning_rate=0.0003`, replay capacity `100000`, `batch_size=256`, `gamma=0.99`, baseline `tau=0.005`, `ent_coef="auto"`, `learning_starts=1000`, `train_freq=1`, `gradient_steps=1`. S5.3M chỉ đổi tau giữa A/B (`0.005`/`0.001`); các thiết lập khoa học khác được giữ theo protocol đăng ký.

## 7. Code walkthrough

Luồng quan sát–hành động–học:

```text
MetaDrive
  → build_agent_input() / AgentInputV1
  → Stage5ObservationEncoder.encode()  [409D]
  → SAC Actor
  → continuous_box2_v1 ActionAdapter
  → MetaDrive
  → Episode classifier + RewardSpecV1
  → replay buffer của SB3
  → critic/actor/alpha/target updates
```

| Bước | Code thật trong repository |
|---|---|
| Public input | `src/platform/agent_input.py` — `build_agent_input()` |
| 409D vector | `src/training/stage5/observation_encoder.py` — `Stage5ObservationEncoder.encode()` |
| TRAIN geometry và holdout guard | `src/training/stage5/train_sampler.py` — `TrainCaseSampler.sample_case()`, `validate_train_case()`, `get_case_for_geometry_id()` |
| Simulator reset/action/reward/observation | `src/training/stage5/env_adapter.py` — `Stage5TrainingEnv.reset()`, `step()`, `_build_agent_input()` |
| Canonical reward arithmetic | `src/launcher/executor.py` — `compute_step_route_progress_and_reward()`; `src/platform/reward.py` — `RewardSpecV1.compute_step_reward()` |
| SAC construction/training | `src/training/stage5/sac_trainer.py` — `SacBaselineTrainer.__init__()`, `train()`, `SacDiagnosticsCallback` |
| Bundle save/verify/load | `src/training/stage5/checkpoint.py` — `save_sac_checkpoint()`, `verify_checkpoint_integrity()`, `load_sac_checkpoint()` |
| Frozen inference | `src/agents/stage5_sac_agent.py` — `Stage5SacAgent.load_checkpoint()`, `reset()`, `act()` |
| S5.3M runner/report | `scripts/run_stage5_tau_ab.py`, `scripts/analyze_stage5_tau_ab.py` |
| Native frozen checkpoint viewer | `scripts/watch_stage5_sac_checkpoint.py` — `watch()` |
| GUI monitoring/comparison | `src/workbench/widgets/stage5_pilot_widget.py` — `Stage5PilotWidget`; `src/workbench/widgets/comparison_widget.py` — `ComparisonWidget`; `src/workbench/comparison_repository.py` |

Inference uses the same 409D encoder and canonical adapter as training. It loads a specific verified bundle and calls `predict(..., deterministic=True)`; inference does not call `learn()`.

## 8. MapSuite, splits, seeds and checkpoints

MapSuite V1 has 240 geometry definitions: 180 TRAIN, 48 VALIDATION, 12 TEST. Canonical protocols define 96 validation episodes and 60 test episodes. Geometry identity includes `geometry_id` and geometry SHA-256; generation seed alone is not unique. Environment seed is a separate source of simulator randomness.

For S5.3A development, the locked internal split takes 156 geometries for `PILOT_TRAIN` and 24 for `PILOT_DIAGNOSTIC`—39 and 6 per tier. The sampler only receives the 156 training IDs. S5.3M selects a fixed eight-case schedule from the 24 diagnostic IDs, two per tier, with environment seeds derived from schedule seed 5302 and geometry ID. It is still TRAIN development data, never official validation/test.

- **Training root seed 101** seeds the fresh training setup and its sampler path; it is not the same as a diagnostic environment seed.
- **Environment seed** controls simulator randomness for an episode and is recorded alongside the geometry identity.
- **10,000 steps** means ten thousand 10 Hz agent decisions/transitions, not ten thousand completed episodes. A single episode can contain many decisions; in each S5.3M arm only five episodes completed during those 10k steps.
- **0/5k/10k checkpoints** are snapshots captured from the same continuous run; they are not three independently trained agents.
- **Diagnostic** asks a bounded development question on a locked TRAIN schedule. **Benchmark** requires approved protocol, eligibility, checkpoint binding, and canonical metrics. A diagnostic is not a smaller official benchmark.

## 9. Research timeline: S5.1 through S5.3M

1. **S5.1 — training foundation:** Gymnasium wrapper around MetaDrive, 409D public observation encoder, continuous action mapping, TRAIN sampler, reset/step/episode/reward integration and lifecycle tests.
2. **S5.2 — SAC baseline:** Stable-Baselines3 SAC trainer, callbacks, metadata/hash-protected checkpoints, deterministic Platform V1 inference agent. Smoke training established that updates/checkpoints work; it did not establish good driving.
3. **S5.3A — first pilot:** fresh seed-101 learning curve at 0/5k/10k. Policy behavior was weak and stationary. An initial route-progress baseline issue was later fixed in reset handling; the old reward values remain historical and are not silently rewritten.
4. **S5.3B–D — bounded variations:** waypoint-scale and actor-activation/sampling-horizon investigations were recorded. None established clean driving; changes were treated as separate development runs rather than silently combined.
5. **S5.3E:** no matching local result artifact was verified in the final inventory. This handoff does not claim it as a completed experiment.
6. **S5.3F–H — behavior isolation:** stochastic action sampling, throttle persistence, steering and speed-control probes showed that simulator actuation works and steering can make progress with an external speed controller. Those controllers were diagnostic interventions, not SAC results or benchmark agents.
7. **S5.3I–K — policy/action/reward audits:** deterministic throttle was already negative at checkpoint 0; critic action preference and reward arithmetic were inspected; action timing/replay fidelity was checked. No implementation mismatch was found that explains the entire stationary behavior.
8. **S5.3L — soft-Q calibration:** 96 repeated frozen-policy rollouts compared critic estimates to entropy-aware soft return on fixed TRAIN states/actions. Mean min-Q was about `15.232`, mean soft return about `9.097`, gap about `+6.135`. This supports a Q calibration concern on that sample; it does not prove that this caused the policy to remain stationary.
9. **S5.3M — fresh tau A/B:** fresh shared initialization, control `tau=0.005`, treatment `tau=0.001`, 10k each, root seed 101, checkpoints 0/5k/10k and the same fixed eight-case diagnostic schedule. It is the final bounded training experiment in this sprint.

The historical sequence narrowed the observed behavior, but it did not produce a proven root cause. A stationary policy could reflect data coverage, objective/discounting, exploration, function approximation, or interactions between them; the existing evidence does not uniquely distinguish these explanations.

## 10. S5.3M result: what the evidence says

| Measure at 10k | Control A | Treatment B |
|---|---:|---:|
| SAC target coefficient | `tau=0.005` | `tau=0.001` |
| Training steps | 10,000 | 10,000 |
| Training root seed | 101 | 101 |
| Fixed internal TRAIN diagnostics | 8 | same 8 cases |
| Paired route-progress difference | — | **B − A = +0.002675 mean** |
| Stationary fraction (speed `<1 km/h`) | 100% | 100% |
| Clean successes | 0 | 0 |
| Safety failures | 0 | 0 |
| Probe outcomes | 8/8 `PROBE_CENSORED` | 8/8 `PROBE_CENSORED` |

All eight paired route-progress differences favored B in this sample, but the magnitude was small. “Stationary” is the experiment's diagnostic threshold `<1 km/h`, not a claim that every physical position was exactly constant. Censored means the bounded 300-decision probe stopped before terminal outcome. It does not establish that an episode would later succeed, fail, or time out.

The treatment had lower anchor Q scale than control. That is a change in critic numbers, not proof that B's critic is calibrated or that B's policy is better. Both policies failed to demonstrate clean driving. B is therefore a development candidate for confirmation, not a winner; A remains the reference baseline.

## 11. Limitations and current status

- There is one fresh training root seed (101), so there is no independent multi-seed estimate of run-to-run variation.
- S5.3M completed five training episodes per arm; 10k steps is a short horizon for a long-horizon driving task, and episode-level geometry experience was limited.
- The development comparison covers eight fixed TRAIN geometries and 300 decisions per probe. It cannot stand in for full canonical validation/test.
- No Stage 5 official benchmark has been run. `stage5_sac` remains `CANDIDATE`, `benchmark_eligible=false`.
- The current resolved plan does not cryptographically bind SAC checkpoint SHA. The existing checkpoint binding schema is specific to a different Stage; a reviewed versioned extension is needed before certification.
- There is no evidence that SAC has learned successful autonomous driving. The Q–return gap is a measured diagnostic but has not been shown to cause stationary behavior.

## 12. What would be required before further claims?

Future work needs approval for an independent-root confirmation protocol, a versioned checkpoint identity binding compatible with the Platform V1 plan, and frozen candidate selection before the official validation/test process. These are prerequisites, not tasks performed by this document. TEST remains a final holdout and must not be used to tune.

