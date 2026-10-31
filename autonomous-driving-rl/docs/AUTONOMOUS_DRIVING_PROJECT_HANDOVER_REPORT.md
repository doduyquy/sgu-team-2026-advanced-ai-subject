# BÁO CÁO TIẾP NHẬN DỰ ÁN XE TỰ HÀNH DÙNG HỌC TĂNG CƯỜNG

Ngày audit: **07/10/2026**  
Commit được kiểm tra: **`6a531b1`**, nhánh **`main`**  
Phạm vi: toàn bộ mã nguồn, cấu hình, dataset, test, tài liệu, script, kết quả audit và lịch sử Git trong `autonomous-driving-rl/`.

---

## Cách đọc mức độ chắc chắn của báo cáo

Để không biến suy đoán thành sự thật, báo cáo dùng các nhãn sau:

| Nhãn | Ý nghĩa |
|---|---|
| **ĐÃ XÁC MINH BẰNG CODE** | Đã đọc implementation hiện tại và trace được luồng gọi |
| **ĐÃ XÁC MINH KHI CHẠY** | Đã trực tiếp chạy lệnh/test trong lần bàn giao này |
| **ĐÃ XÁC MINH QUA ARTIFACT** | Có kết quả audit do máy tạo và được commit, nhưng chưa chạy lại trên máy hiện tại |
| **ĐƯỢC TÀI LIỆU MÔ TẢ** | Tài liệu nói có, nhưng bằng chứng runtime hiện tại chưa đủ |
| **SUY LUẬN** | Kết luận hợp lý từ nhiều bằng chứng, chưa được chứng minh hoàn toàn |
| **CHƯA HOÀN CHỈNH** | Đã có code nhưng còn thiếu một phần quan trọng |
| **KHUNG RỖNG** | Mới có thư mục, interface hoặc hướng dẫn |
| **CHƯA CÓ** | Không tìm thấy implementation |

---

# PHẦN I — HIỂU NHANH TRONG 5 PHÚT

## 1. Kết luận quan trọng nhất

Tên repository là `autonomous-driving-rl`, nhưng trạng thái thật hiện nay là:

> **Team đã xây dựng khá hoàn chỉnh một nền tảng mô phỏng, chạy thí nghiệm và đánh giá agent xe tự hành trên MetaDrive. Team chưa xây dựng thuật toán Reinforcement Learning hoàn chỉnh.**

Có thể hình dung dự án giống một phòng thí nghiệm:

```text
Đường thử, luật thi, cảm biến, bộ điều khiển,
trọng tài, bảng điểm, camera giám sát, nơi lưu kết quả
                         → ĐÃ CÓ

Chiếc xe dùng thuật toán RL đã được huấn luyện
                         → CHƯA CÓ
```

Nền tảng hiện đã định nghĩa được:

- xe được nhìn thấy thông tin gì;
- agent được phép và không được phép biết gì;
- agent điều khiển vô-lăng, ga và phanh ra sao;
- bản đồ và traffic được tạo thế nào;
- episode kết thúc khi nào;
- reward tính thế nào;
- metric đánh giá gồm những gì;
- train, validation và test phải tách biệt ra sao;
- cách chạy experiment;
- cách lưu kết quả và kiểm tra kết quả có bị sửa hay không;
- GUI để cấu hình, theo dõi và xem kết quả.

Nhưng hiện chưa có:

- PPO, SAC, DQN, TD3 hay A2C;
- policy neural network;
- optimizer và loss;
- replay buffer hoặc rollout buffer;
- vòng lặp training;
- checkpoint model;
- agent nghiên cứu thật sự có quyền chạy benchmark.

## 2. Dự án đang giải quyết bài toán gì?

Agent phải điều khiển một chiếc xe trong MetaDrive để:

1. đi theo route được chỉ định;
2. vượt qua các dạng đường từ đơn giản đến phức tạp;
3. xử lý traffic xung quanh;
4. tới đích;
5. không va chạm và không đi ra khỏi đường.

Độ khó tăng theo:

- cấu trúc đường;
- số nút giao, vòng xuyến, ramp, merge/split;
- chiều dài route;
- mật độ traffic.

Project tập trung vào **ra quyết định điều khiển xe từ state**, không phải bài toán computer vision. Agent nhận state và LiDAR đã được simulator cung cấp, chứ không nhận ảnh camera thô.

## 3. Trả lời nhanh 12 câu hỏi bàn giao

| Câu hỏi | Trả lời ngắn gọn |
|---|---|
| 1. Project thực sự xây gì? | Nền tảng nghiên cứu và đánh giá agent xe tự hành trên MetaDrive |
| 2. Kiến trúc hoạt động thế nào? | Request → Resolver → Preflight → MetaDrive → Agent → Reward/Metric → Logging/Result |
| 3. Phần nào đã hoàn thành? | Dataset, contract, environment execution, observation/action, episode, reward, metric, launcher, logging, GUI |
| 4. Phần nào mới là khung? | `src/agents`, `src/logging`, `experiments` và toàn bộ training pipeline |
| 5. Environment hoạt động ra sao? | Executor trực tiếp tạo `MetaDriveEnv` từ geometry bị khóa, seed traffic và horizon theo route |
| 6. Agent hiện có gì? | Chỉ có 4 fixture để test platform; chưa có agent nghiên cứu thật |
| 7. Observation/Action/Reward là gì? | Observation 259D + traffic + route; action là steering và ga/phanh; reward dựa trên tiến độ, thời gian và kết quả cuối |
| 8. MapSuite V1 làm gì? | Cố định 240 geometry để train/validation/test công bằng và tái lập được |
| 9. Experiment chạy thế nào? | CLI/GUI tạo plan, preflight kiểm tra, executor chạy từng case, logger lưu kết quả |
| 10. Command nào dùng để kiểm chứng? | Xem phần “Cài đặt và chạy” ở cuối báo cáo |
| 11. Còn thiếu gì? | Agent thật, training adapter, training loop, checkpoint, RL dependency, CI |
| 12. Nên bắt đầu từ đâu? | Cài đúng MetaDrive → chạy lại platform → làm Stage 0 Random/Naive baseline → xây training pipeline |

## 4. Có chạy end-to-end được chưa?

### Về mặt kiến trúc: Có

Luồng inference/evaluation đã được nối đầy đủ:

```text
CLI hoặc GUI
    ↓
Resolve experiment
    ↓
Preflight kiểm tra
    ↓
Tạo MetaDrive environment
    ↓
Agent quan sát và chọn action
    ↓
Environment chuyển trạng thái
    ↓
Tính reward và metric
    ↓
Lưu result có kiểm tra toàn vẹn
```

Committed audit cho thấy đã từng chạy thành công với simulator thật.

### Trên máy hiện tại: Chưa chạy được simulator

Python 3.10 đang có, nhưng package `metadrive` chưa được cài. Vì vậy:

- dataset verifier chạy được;
- các test thuần Python chạy được;
- CLI, Workbench và simulator chưa chạy được.

Đây là vấn đề môi trường cài đặt, chưa phải bằng chứng rằng code executor bị hỏng.

---

# PHẦN II — BẢN ĐỒ KHO MÃ NGUỒN

## 5. Cấu trúc thư mục thực tế

```text
autonomous-driving-rl/
├── configs/
│   ├── maps/                  # mô tả MapSuite lịch sử
│   ├── platform/              # contract/spec khoa học có version
│   └── templates/             # file ví dụ, chưa phải hệ thống config runtime
├── datasets/mapsuite_v1/      # dataset benchmark 240 geometry đã đóng băng
├── docs/                      # hướng dẫn, kiến trúc, roadmap, audit
├── experiments/               # chỗ dành cho training sau này; hiện là khung rỗng
├── results/audits/            # bằng chứng audit được commit
├── runs/                      # output runtime cục bộ; không commit
├── scripts/                   # audit, export, verify, inspect
├── src/
│   ├── agents/                # nơi sẽ chứa agent thật; hiện chưa có
│   ├── environments/          # CourseEnvV1 cũ, không phải benchmark hiện tại
│   ├── evaluation/            # random evaluator cũ
│   ├── launcher/              # resolver, registry, preflight, executor, CLI
│   ├── logging/               # package placeholder cũ
│   ├── platform/              # phần lõi khoa học của Platform V1
│   └── workbench/             # GUI PySide6
└── tests/                     # bộ regression test lớn
```

### Vai trò từng nhóm

| Nhóm | Vai trò thật |
|---|---|
| `src/platform` | Định nghĩa giao tiếp agent, input, action, episode, reward, metric và logging |
| `src/launcher` | Biến yêu cầu của người dùng thành experiment hợp lệ và chạy nó |
| `datasets/mapsuite_v1` | Cung cấp bộ map/case cố định cho nghiên cứu |
| `src/workbench` | Giao diện desktop gọi lại launcher qua subprocess |
| `results/audits` | Bằng chứng platform đã từng được kiểm tra như thế nào |
| `src/agents` | Nơi agent nghiên cứu sẽ nằm; hiện chưa có implementation |
| `experiments` | Nơi dự kiến chứa training config/script; hiện chưa có implementation |
| `src/environments`, `src/evaluation` | Prototype cũ được giữ lại để tham khảo/tương thích |

Không tìm thấy model hoặc checkpoint đã huấn luyện.

## 6. Nền tảng và phần học tăng cường khác nhau thế nào?

### A. Hạ tầng / nền tảng — gần hoàn chỉnh

- MapSuite và split;
- contract/spec;
- observation và action boundary;
- episode và reward;
- evaluation metric;
- launcher, resolver, registry, preflight;
- executor;
- logging cục bộ;
- W&B adapter;
- Workbench GUI;
- audit và regression test.

### B. Lõi xe tự hành / học tăng cường — mới có phần thực thi

Đã có:

- MetaDrive simulation;
- dữ liệu cho agent quan sát;
- action điều khiển;
- reward;
- episode transition;
- evaluation.

Chưa có:

- policy học được;
- neural network;
- thuật toán cập nhật weight;
- rollout/replay buffer;
- training loop;
- checkpoint;
- inference từ model đã train.

### Điểm thiếu quan trọng nhất

`AgentPolicy` hiện chỉ có:

```text
descriptor
reset(...)
act(...)
close()
```

Executor gọi `act()` rồi tự tính reward ở phía platform. Reward và transition không được đưa ngược lại agent. Vì vậy interface này phù hợp cho **inference/evaluation**, chưa đủ cho online RL training.

---

# PHẦN III — KIẾN TRÚC VÀ LUỒNG THỰC THI

## 7. Kiến trúc tổng thể đã xác minh

```text
LaunchRequestV1 / form trên Workbench
                 ↓
resolve_experiment_plan()
                 ↓
ResolvedExperimentPlanV1
                 ↓
run_preflight()
                 ↓
ExperimentExecutor.execute()
                 ↓
Đọc geometry bị khóa từ manifest
                 ↓
Tạo MetaDriveEnv
                 ↓
env.reset(environment_seed)
                 ↓
AgentPolicy.reset(agent_seed)
                 ↓
┌──────────── vòng lặp 10 Hz ────────────┐
│ raw observation + simulator state      │
│              ↓                         │
│ build_agent_input()                    │
│              ↓                         │
│ AgentInputV1                           │
│              ↓                         │
│ agent.act()                            │
│              ↓                         │
│ AgentDecision                          │
│              ↓                         │
│ ActionAdapter                          │
│              ↓                         │
│ [steering, throttle_brake]             │
│              ↓                         │
│ env.step()                             │
│              ↓                         │
│ outcome + reward + telemetry           │
└────────────────────────────────────────┘
                 ↓
EpisodeRecord + EpisodeTimingRecord
                 ↓
Aggregate metrics
                 ↓
LocalExperimentLogger + W&B tùy chọn
                 ↓
runs/<run_id>/
```

## 8. Các khối cốt lõi

| Block | File/class/function chính | Input → Output | Trạng thái |
|---|---|---|---|
| Request và plan | `launcher/models.py` | tham số CLI/GUI → request/plan bất biến | ĐÃ CÓ |
| Case resolver | `launcher/cases.py` | selector + manifest → `ResolvedCaseV1` | ĐÃ CÓ |
| Agent registry | `launcher/registry.py` | metadata + factory → agent instance | ĐÃ CÓ, nhưng chỉ fixture |
| Resolver | `launcher/resolver.py:resolve_experiment_plan` | request → plan cụ thể | ĐÃ CÓ |
| Preflight | `launcher/preflight.py:run_preflight` | plan → báo cáo pass/fail | ĐÃ CÓ |
| Executor | `launcher/executor.py:ExperimentExecutor` | plan hợp lệ → kết quả run | ĐÃ CÓ |
| Agent input | `platform/agent_input.py` | simulator state → input sạch | ĐÃ CÓ |
| Agent API | `platform/agent.py:AgentPolicy` | input → decision | API xong, agent thật chưa có |
| Action | `platform/action_adapter.py` | action agent → actuator chuẩn | ĐÃ CÓ |
| Episode | `platform/episode.py` | flag thô → outcome chuẩn | ĐÃ CÓ |
| Reward | `platform/reward.py` | tiến độ/outcome → reward | ĐÃ CÓ, chưa freeze |
| Metric | `platform/metrics.py` | episode → scorecard | ĐÃ CÓ |
| Logging | `platform/experiment_logging.py` | record → file + hash | ĐÃ CÓ |
| W&B | `platform/wandb_backend.py` | record → bản sao trên W&B | ĐÃ CÓ, tùy chọn |
| GUI | `workbench/*` | thao tác người dùng → subprocess/result | ĐÃ CÓ |

## 9. Khi chạy một command, Python đi qua đâu?

Ví dụ:

```powershell
python -m src.launcher.cli run `
  --mode SANDBOX `
  --agent fixture_seeded_random `
  --tier Easy `
  --agent-seed 101 `
  --render OFF
```

Call graph thật:

```text
src.launcher.cli.main()
└── create_parser()
└── handle_run()
    ├── build_launch_request_from_args()
    ├── resolve_experiment_plan()
    │   ├── build_canonical_agent_registry()
    │   └── resolve_sandbox_case()
    ├── run_preflight()
    └── ExperimentExecutor(...).execute()
        ├── run_preflight() lần nữa
        ├── LocalExperimentLogger(...)
        ├── WandbBackend(...)
        ├── agent_factory()
        └── với mỗi case
            ├── load_locked_geometry_block_sequence()
            ├── build_metadrive_case_config()
            ├── MetaDriveEnv(...)
            ├── env.reset()
            ├── agent.reset()
            └── vòng lặp step
                ├── build_agent_input()
                ├── agent.act()
                ├── action_adapter.to_canonical()
                ├── env.step()
                ├── classify_episode_outcome()
                └── compute_step_route_progress_and_reward()
        ├── logger.prepare_summary()
        ├── logger.finalize_run()
        └── ExecutionReportV1
```

Preflight được gọi hai lần có chủ ý: lần đầu để báo sớm cho CLI/GUI, lần hai trong executor để bảo vệ trường hợp ai đó gọi executor trực tiếp.

## 10. Các điểm bắt đầu chạy chương trình

| Cách chạy | Entry point | Mục đích |
|---|---|---|
| CLI | `python -m src.launcher.cli` | liệt kê agent/case, tạo plan, chạy experiment |
| GUI | `python -m src.workbench.app` | cấu hình, theo dõi và xem result |
| Worker | `python -m src.workbench.worker` | subprocess nội bộ của GUI |
| Dataset verify | `python scripts/verify_mapsuite_v1_dataset.py` | kiểm tra cấu trúc, nguồn và checksum |
| Dataset export | `python scripts/export_mapsuite_v1_dataset.py` | tái tạo package từ audit artifact |
| Preview export | `python scripts/export_mapsuite_v1_previews.py` | dựng preview top-down |
| Audit | `python scripts/audit_*.py` | tái tạo bằng chứng theo từng contract |
| Test | `python -m unittest discover ...` | regression test |
| Legacy evaluator | `python -m src.evaluation.evaluate_random` | prototype cũ, không phải benchmark |

Hiện **không có entry point training**.

## 11. Workbench chạy như thế nào?

```text
Người dùng bấm Resolve hoặc Run
             ↓
MainWindow
             ↓
WorkbenchProcessRunner
             ↓
QProcess mở Python subprocess
             ↓
src.workbench.worker
             ↓
resolver + preflight + executor
             ↓
Các dòng JSON có prefix @@WORKBENCH@@
             ↓
GUI cập nhật live telemetry và result
```

Workbench không gọi simulator trong UI thread. Nó không được quyền thay đổi luật khoa học. Nếu bỏ GUI hoàn toàn, mọi experiment vẫn chạy được bằng CLI.

## 12. Luồng cũ cần tránh nhầm

`CourseEnvV1` và `src/evaluation/evaluate_random.py` là prototype cũ:

- observation 35 chiều;
- action rời rạc 5 lựa chọn;
- chạy 20 episode random;
- không dùng MapSuite chuẩn;
- không dùng AgentPolicy;
- không dùng reward chuẩn;
- không dùng preflight;
- không lưu artifact chuẩn.

Do đó kết quả từ luồng này không được gọi là benchmark Platform V1.

---

# PHẦN IV — MÔI TRƯỜNG, TÁC TỬ, QUAN SÁT, HÀNH ĐỘNG VÀ PHẦN THƯỞNG

## 13. Môi trường chuẩn hiện tại

Một điểm dễ hiểu nhầm: benchmark hiện tại không có class wrapper kiểu `CourseEnvV1`. Executor tạo trực tiếp `MetaDriveEnv` cho từng case.

Config quan trọng:

| Thuộc tính | Giá trị |
|---|---|
| Map | exact block sequence bằng `PG_MAP_FILE` |
| Environment seed | lấy từ case manifest |
| Traffic density | theo tier/scenario |
| Traffic mode | `Trigger` |
| Physics step | 0,02 giây |
| Decision repeat | 5 physics step |
| Tần số agent ra quyết định | 10 Hz, tức mỗi 0,1 giây |
| Horizon | tính theo chiều dài route, 1000–4000 step |
| Timeout semantics | `truncated=True`, không ép thành terminated |
| Render | OFF hoặc NATIVE; benchmark bắt buộc OFF |

### Vòng đời episode

```text
Tạo environment
    ↓
env.reset(environment_seed)
    ↓
agent.reset(public_context, agent_seed)
    ↓
Lặp observation → action → step
    ↓
Phát hiện crash / out-of-road / arrival / timeout
    ↓
Phân loại outcome theo ưu tiên an toàn
    ↓
Tạo EpisodeRecord
    ↓
env.close()
```

Thứ tự ưu tiên kết thúc:

```text
CRASH_HUMAN
→ CRASH_VEHICLE
→ CRASH_OBJECT
→ CRASH_BUILDING
→ CRASH_SIDEWALK
→ OUT_OF_ROAD
→ SUCCESS
→ TIMEOUT
```

Nếu xe vừa tới đích vừa va chạm trong cùng step, kết quả là failure, không phải clean success.

## 14. Quan sát — xe nhìn thấy gì?

Agent nhận `AgentInputV1`, không nhận object simulator trực tiếp.

### 14.1 Quan sát lõi: 259 chiều

```text
[ 9 ego state | 10 navigation | 240 LiDAR ]
```

| Phần | Shape | Ý nghĩa | Range |
|---|---:|---|---|
| Ego state | 9 | biên đường, lệch hướng, tốc độ, steering, action trước, chuyển động quay... | chuẩn hóa `[0,1]` |
| Navigation | 10 | thông tin tương đối về hai checkpoint tiếp theo | chuẩn hóa `[0,1]` |
| LiDAR | 240 | 240 tia quét 360°, khoảng cách va chạm chuẩn hóa | `[0,1]` |

Kiểu dữ liệu là `float32[259]`, được copy phòng thủ và đặt read-only. Shape sai, NaN/Inf hoặc vượt range sẽ bị từ chối.

Chi tiết từng field nằm trong `results/audits/observation_action/observation_schema.json`. Không nên tự đoán ý nghĩa từng index chỉ từ con số 259.

### 14.2 Ngữ cảnh giao thông

Tối đa 8 xe gần ego nhất trong bán kính 50 m. Mỗi xe có 7 giá trị:

```text
relative_position_x   mét
relative_position_y   mét
relative_velocity_x   m/s
relative_velocity_y   m/s
relative_heading      radian
length                mét
width                 mét
```

Các xe được sắp theo khoảng cách. Nếu thiếu xe thì padding bằng 0 và dùng `validity_mask` để phân biệt padding với xe thật.

Lưu ý: đây là state lấy trực tiếp từ physics engine, không phải object detection từ camera/LiDAR. Điều này phù hợp vì project đang benchmark decision-making, không benchmark perception.

### 14.3 Ngữ cảnh nhiệm vụ

Agent nhận 20 waypoint phía trước:

- cách nhau 2,5 m;
- tổng lookahead 50 m;
- mỗi waypoint gồm relative x, relative y và relative heading;
- kèm validity mask;
- có lane width, hướng goal và cờ route sắp kết thúc.

### 14.4 Những gì agent không được biết

Agent bị cấm nhận:

- TRAIN/VALIDATION/TEST;
- case ID;
- geometry hash và seed;
- environment seed;
- route completion tổng thể;
- reward breakdown;
- crash/arrival flag;
- terminal reason;
- điểm đánh giá hoặc kết quả tương lai.

Mục tiêu là ngăn agent “học thuộc đề thi” hoặc dùng thông tin của evaluator.

## 15. Hành động — tác tử điều khiển xe thế nào?

Action vật lý chuẩn:

```text
[steering, throttle_brake] ∈ [-1, 1]²
```

- `steering = -1`: hết lái trái;
- `steering = +1`: hết lái phải;
- `throttle_brake > 0`: ga;
- `throttle_brake < 0`: phanh;
- `0`: trung tính.

Ba adapter được chứng nhận:

| Adapter | Action agent trả về | Chuyển đổi |
|---|---|---|
| `continuous_box2_v1` | 2 số thực | giữ nguyên sau validation |
| `discrete25_native_v1` | số nguyên 0–24 | lưới 5 mức lái × 5 mức ga/phanh |
| `discrete9_lowbranch_v1` | số nguyên 0–8 | lưới 3 × 3 cho search/MCTS |

Không có silent clipping. Sai kiểu, sai shape, NaN, vô cực hoặc vượt `[-1,1]` đều gây technical failure.

Action rời rạc 5 lựa chọn của `CourseEnvV1` là legacy và không tương thích contract hiện tại.

## 16. Phần thưởng — tác tử được khuyến khích điều gì?

Reward mỗi step:

```text
delta = route_completion hiện tại - route_completion step trước

reward = 1.0 × delta
         - 0.25 / horizon_steps
         + terminal_reward
```

Trong đó:

| Tình huống | Terminal reward |
|---|---:|
| Clean success | `+1.0` |
| Crash hoặc out-of-road | `-1.0` |
| Unknown termination | `-1.0` |
| Timeout | `0.0` |
| Chưa kết thúc | `0.0` |

Không clipping mặc định.

Reward đang khuyến khích:

- tiến về phía đích;
- không đứng yên quá lâu;
- tới đích an toàn;
- tránh collision và rời đường.

### Nguy cơ khai thác kẽ hở của phần thưởng

- Progress reward có thể khiến agent ưu tiên tiến nhanh hơn hành vi êm và an toàn.
- Timeout không có terminal penalty riêng; chỉ chịu tổng time cost khoảng `-0.25`.
- Reward chưa có penalty rõ cho comfort, steering gắt hoặc gia tốc gắt.
- Reward đang có trạng thái **NON-FROZEN**, nghĩa là cần xác nhận thêm bằng TRAIN/VALIDATION trước khi training lớn.

Vì vậy project không dùng episode return để xếp hạng agent. Return chỉ là tín hiệu training/chẩn đoán.

## 17. Tác tử hiện có gì?

Registry có bốn agent:

| Agent | Hành vi | Mục đích |
|---|---|---|
| `fixture_constant_continuous` | action liên tục cố định | test pipeline |
| `fixture_seeded_random` | action liên tục ngẫu nhiên có seed | test stochasticity |
| `fixture_stateful_counter` | giữ bộ đếm trong episode | test reset state |
| `fixture_discrete` | action discrete cố định | test adapter |

Tất cả đều:

- nằm trong `src/platform/agent.py`, không nằm trong `src/agents`;
- có `benchmark_eligible=False`;
- chỉ là fixture;
- không phải Stage 0 baseline khoa học.

## 18. Muốn viết tác tử mới thì làm gì?

Ví dụ tạo:

```text
src/agents/random_baseline.py
```

Agent cần đáp ứng `AgentPolicy`. Không bắt buộc kế thừa class, nhưng phải có đúng các thành phần:

```python
@property
def descriptor(self) -> AgentDescriptor: ...

def reset(
    self,
    public_context: AgentPublicEpisodeContext,
    agent_seed: Optional[int] = None,
) -> None: ...

def act(self, agent_input: AgentInputV1) -> AgentDecision: ...

def close(self) -> None: ...
```

Sau đó:

1. khai báo đúng `agent_id`, version, input profile, action adapter và stochasticity;
2. thêm factory và `AgentRegistrationV1` trong `src/launcher/registry.py`;
3. đảm bảo descriptor runtime khớp chính xác registration;
4. viết test lifecycle, seed, action và information leakage;
5. chạy SANDBOX trước;
6. chỉ bật benchmark eligibility sau review rõ ràng.

Luồng gọi:

```text
Platform
→ build_agent_input()
→ AgentInputV1
→ agent.act()
→ AgentDecision.action_payload
→ ActionAdapter
→ CanonicalActionV1
→ MetaDriveEnv.step()
```

---

# PHẦN V — MAPSUITE V1 VÀ HỆ THỐNG ĐÁNH GIÁ

## 19. MapSuite V1 là gì?

MapSuite V1 là bộ benchmark procedural cố định gồm:

```text
12 scenario family × 20 geometry seed = 240 geometry
```

Nó không phải dataset ảnh hoặc trajectory để supervised learning. Nó là danh mục map và case giúp mọi agent thi trên cùng một bộ điều kiện.

### Geometry

Một geometry là một instance đường cụ thể, được xác định bởi:

- sequence của các block đường;
- geometry generation seed;
- exact serialized block sequence;
- SHA-256 của geometry.

### Scenario family

Một family là một mẫu topology, ví dụ `SCS` hoặc `SCXOCS`. Hai geometry cùng family có cùng dạng block tổng quát nhưng khác chi tiết procedural do seed.

### Các tier

| Tier | Traffic density | Family |
|---|---:|---|
| Easy | 0,00 | `SCS`, `SCSS`, `SCCS` |
| Medium | 0,08 | `SCXCS`, `SCTCS`, `SCXCCS` |
| Hard | 0,15 | `SCXOCS`, `SCTXrCS`, `XTOCS` |
| Extreme | 0,25 | `CrXROSTR`, `SCXOCrTYCS`, `SCTXORyCCS` |

`family_rank_within_tier` chỉ là thứ tự trong tier, không phải điểm performance của agent.

### Preview được tạo thế nào?

Preview là ảnh top-down được render lại từ exact frozen geometry. Package chứa 12 ảnh đại diện; verifier đã xác nhận byte ảnh trùng nguồn audit.

## 20. Phân chia dữ liệu và nguy cơ rò rỉ dữ liệu

Mỗi family có:

- 15 geometry TRAIN;
- 4 geometry VALIDATION;
- 1 geometry TEST.

Toàn bộ:

```text
TRAIN       180 geometry
VALIDATION   48 geometry
TEST         12 geometry
```

Evaluation case:

```text
VALIDATION: 48 geometry × 2 environment seed = 96 case
TEST:       12 geometry × 5 environment seed = 60 case
```

Seed:

- validation environment seed: `5101`, `5102`;
- test environment seed: `9101` đến `9105`;
- agent replicate seed: `101`, `202`, `303`;
- protocol order seed: `424242`.

### Kết quả kiểm tra rò rỉ dữ liệu

**ĐÃ XÁC MINH KHI CHẠY:**

- không trùng geometry ID giữa các split;
- không trùng cặp `(scenario family, geometry seed)`;
- không trùng geometry hash;
- hợp của ba split đúng bằng toàn bộ 240 geometry;
- validation case chỉ trỏ tới validation geometry;
- test case chỉ trỏ tới test geometry.

Con số seed thô từ 0–19 có thể lặp giữa các family/split. Đây không phải leakage vì seed chỉ có ý nghĩa khi đi cùng family.

Các family cố ý xuất hiện trong cả ba split. Điều này đo khả năng tổng quát sang geometry mới trong cùng loại topology, không phải tổng quát sang family hoàn toàn chưa thấy.

## 21. Lỗi metadata lane cần đặc biệt lưu ý

Một số CSV, JSON và tài liệu cũ ghi:

```text
base_lane_num = 2
```

Nhưng geometry đã đóng băng thực tế được tạo bằng default của MetaDrive:

```text
effective_base_lane_num = 3
```

Đây là lỗi nhãn metadata lịch sử, không phải geometry hiện tại bị hỏng.

Không được “sửa” bằng cách ép `lane_num=2`, vì việc đó sẽ tạo geometry khác và làm sai hash benchmark. Hãy đọc:

`docs/environment/MAPSUITE_V1_LANE_METADATA_ERRATUM.md`.

## 22. Chỉ số đánh giá

### Chỉ số chính

| Metric | Cách tính | Tốt hơn khi |
|---|---|---|
| Clean Success Rate | số episode tới đích an toàn / tổng episode | cao hơn |
| Safety Failure Rate | số episode crash/out-of-road / tổng episode | thấp hơn |
| Mean Final Route Completion | trung bình phần route hoàn thành lúc kết thúc | cao hơn |
| Median Final Route Completion | trung vị phần route hoàn thành | cao hơn |
| Mean Time to Clean Success | thời gian trung bình của các episode clean success | thấp hơn |

Nếu không có clean success, time-to-success là `null`, không được ghi giả thành 0.

### Chỉ số phụ

- raw arrival rate;
- từng loại collision;
- timeout rate;
- max route completion;
- tốc độ trung bình/tối đa;
- episode steps;
- episode return dùng chẩn đoán.

Kết quả còn được chia theo Easy/Medium/Hard/Extreme. Macro metric cho bốn tier trọng số bằng nhau, mỗi tier 0,25. Project cố ý không gom tất cả thành một “mega-score”.

### Quy trình đánh giá

```text
Agent
× bộ case cố định
× environment seed cố định
× agent replicate seed khi stochastic
→ EpisodeRecord
→ metric theo tier
→ macro metric
→ final result
```

VALIDATION và TEST phải chạy nguyên bộ. CLI không cho lọc riêng một tier hoặc vài case trong canonical benchmark. TEST không được dùng để tune hyperparameter hoặc chọn checkpoint.

---

# PHẦN VI — BỘ KHỞI CHẠY, HỢP ĐỒNG, GHI NHẬT KÝ VÀ GIAO DIỆN

## 23. Bộ khởi chạy được thiết kế để làm gì?

| Thành phần | Nhiệm vụ |
|---|---|
| Case | một geometry + environment seed + traffic + horizon |
| Registry | biết agent nào tồn tại và factory nào tạo ra nó |
| Resolver | biến ý định người dùng thành plan đầy đủ |
| Preflight | chặn experiment sai trước khi tạo simulator/output |
| Event | telemetry về trạng thái run/episode |
| Executor | chạy simulator và điều phối toàn bộ pipeline |
| Contract | mô tả và khóa các quy tắc/interface khoa học |

Launcher tách giao diện người dùng khỏi quyền quyết định khoa học. GUI hoặc custom registry không được lách luật benchmark.

## 24. Các hợp đồng JSON có thực sự được dùng lúc chạy không?

Câu trả lời chính xác là: **phần lớn không được load như dynamic config**.

Các JSON trong `configs/platform` chủ yếu là:

- bản mô tả có version;
- snapshot contract;
- nguồn để audit hash và kiểm tra code/config có drift không;
- tài liệu máy đọc được.

Runtime rule thật thường nằm trong dataclass, constant và builder Python. Runtime chủ yếu load CSV manifest và exact geometry artifact.

| Contract/spec | Mục đích | Cách enforce thật |
|---|---|---|
| `agent_contract_v1.json` | input, action, lifecycle | dataclass/protocol/adapter và test |
| `episode_spec_v1.json` | horizon, tần số, termination | `specs.py`, `episode.py`, executor |
| `reward_spec_v1.json` | công thức reward | `RewardSpecV1()` trong code |
| `evaluation_metrics_v1.json` | metric và ý nghĩa | `metrics.py`, logger |
| `evaluation_protocol_v1.json` | suite, seed, rule | hash khóa + manifest + resolver/preflight |
| `launcher_contract_v1.json` | authority của launcher | model/resolver/preflight/executor |
| `logging_contract_v1.json` | artifact và lifecycle | logger/backend |
| `workbench*_contract_v1.json` | luật thin GUI và trust | Workbench code + test/audit |
| `scenario_split_v1.json` | kích thước split | manifest và test |
| `seed_plan_v1.json` | các kênh seed | resolver, protocol, test |

Version `v1` được quản lý bằng tên file, `spec_version`, hash và preflight. Không có framework migration schema riêng.

## 25. Thứ tự ưu tiên của cấu hình

Project không có hệ thống chung kiểu “default < JSON < env < CLI”. Luồng thật là:

```text
Default trong Python/dataclass
        ↓
CLI hoặc Workbench tạo LaunchRequestV1
        ↓
Resolver thêm default theo mode và từ chối selector trái luật
        ↓
Manifest/contract khóa phần benchmark
        ↓
Environment variable chỉ dùng cho một số cấu hình W&B/credential
```

Các file trong `configs/templates` chỉ là ví dụ, chưa có config loader dùng chúng để chạy experiment.

## 26. Ghi nhật ký và đầu ra của một lần chạy

Một run hoàn chỉnh có dạng:

```text
runs/<run_id>/
├── run_manifest.json          # experiment là gì, provenance nào
├── run_state.json             # RUNNING/COMPLETE/FAILED/INTERRUPTED
├── episodes.csv               # một dòng cho mỗi episode
├── summary.json               # metric tổng hợp
├── timing.csv                 # thời gian agent.act()
├── wandb_sync.json            # trạng thái đồng bộ W&B
├── run_integrity.json         # hash kiểm tra toàn vẹn
└── technical_failures.jsonl   # chỉ xuất hiện nếu có lỗi kỹ thuật
```

Đặc điểm:

- local artifact là source of truth;
- JSON được ghi qua file tạm rồi atomic replace;
- không cho ghi đè run ID đã tồn tại;
- kết quả cuối được hash;
- latency chỉ đo quanh `agent.act()`, không tính simulator và logging;
- failure, interruption và incomplete được phân biệt rõ.

## 27. Không dùng W&B có chạy được không?

**Có.** W&B chỉ là bản sao để theo dõi và trực quan hóa.

Mode:

- `DISABLED`;
- `OFFLINE`;
- `ONLINE`.

Nếu W&B hỏng nhưng local logging vẫn tốt, experiment khoa học vẫn có thể hoàn tất. Audit online mới nhất bị skip vì không có credential; repository chỉ có bằng chứng historical cho một online smoke trước đó.

## 28. Workbench / GUI

Workbench dùng PySide6 và cung cấp:

- form cấu hình experiment;
- xem plan/preflight;
- live telemetry;
- duyệt agent và case;
- đọc result;
- kiểm tra integrity;
- xem trạng thái W&B.

Workbench gọi launcher bằng `QProcess`, không gọi executor trực tiếp trong Qt thread. Request được truyền qua JSON tạm. Message trả về có prefix `@@WORKBENCH@@`.

Results Browser chỉ đánh dấu kết quả là authoritative nếu:

- run COMPLETE;
- đủ artifact;
- các run ID khớp nhau;
- hash integrity hợp lệ.

GUI không sửa, repair hoặc xóa result. Live telemetry chỉ là thông tin tạm; file đã verify mới là kết quả chính thức.

---

# PHẦN VII — TEST, AUDIT, DEPENDENCY VÀ REPRODUCIBILITY

## 29. Bộ kiểm thử đang kiểm tra gì?

Coverage mạnh ở:

- AgentInput và information leakage;
- action adapter;
- episode termination;
- reward và metric;
- dataset/split/checksum;
- evaluation protocol;
- launcher authority và preflight;
- logging lifecycle/integrity;
- W&B failure isolation;
- Workbench subprocess/protocol/result trust.

Coverage chưa có vì feature chưa tồn tại:

- training convergence;
- policy network;
- optimizer/update;
- replay/rollout buffer;
- checkpoint save/load;
- agent khoa học thật;
- so sánh thuật toán;
- training reproducibility dài hạn.

### Đã trực tiếp chạy trong lần audit này

| Lệnh/nhóm | Kết quả |
|---|---|
| MapSuite verifier | PASS toàn bộ |
| Dataset + preview tests | **49/49 PASS** |
| Episode + reward + agent-contract tests | **63/63 PASS** |
| Tổng test trực tiếp | **112 PASS** |
| CLI `agents`, `cases`, `plan` | bị chặn khi import vì thiếu `metadrive` |
| Full suite | chưa chạy do thiếu `metadrive` |
| Simulator smoke | chưa chạy do thiếu `metadrive` |

## 30. Hệ thống audit

| Script | Kiểm tra |
|---|---|
| `audit_agent_contract.py` | agent input, lifecycle, action adapter, leakage |
| `audit_episode_contract.py` | termination, horizon, reset, traffic lifecycle |
| `audit_reward_metrics.py` | reward candidate và metric |
| `audit_evaluation_protocol.py` | split, seed, case manifest, leakage |
| `audit_launcher_core.py` | resolver, registry, preflight, executor |
| `audit_logging_contract.py` | artifact, lifecycle, W&B, integrity |
| `audit_workbench*.py` | GUI, subprocess, result trust, hardening |
| `audit_mapsuite_candidates.py` | tạo và đo candidate map lịch sử |
| `run_observation_action_audit.py` | observation/action của MetaDrive |

Committed artifact báo cáo nhiều simulator smoke đã pass. Tuy nhiên đó là bằng chứng lịch sử. Sau khi cài MetaDrive trên máy mới vẫn nên chạy lại full suite và một sandbox run.

## 31. Kiểm tra tài liệu

### Tài liệu đang khớp code khá tốt

- `GETTING_STARTED.md`;
- `PLATFORM_V1_OVERVIEW.md`;
- `EXPERIMENT_WORKFLOW.md`;
- `AGENT_DEVELOPMENT_GUIDE.md`;
- `WORKBENCH_GUIDE.md`;
- `LEGACY_AND_BOUNDARIES.md`.

### Tài liệu mô tả tương lai, không phải implementation hiện tại

- `STAGE_ROADMAP.md`;
- `COURSE_RESEARCH_SCOPE.md`;
- các stage agent sau Stage 0.

### Chỗ có thể gây hiểu nhầm

- `src/logging/README.md` nói logging còn là placeholder, nhưng logging thật đã được xây ở `src/platform`;
- `configs/templates` trông giống config chạy được nhưng hiện chỉ là ví dụ;
- `SCHEMA.md` và một số MapSuite metadata vẫn ghi lane 2; phải đọc cùng erratum;
- root `inspect_*.py` là tool khám phá cũ, không phải canonical pipeline.

## 32. Thư viện phụ thuộc và môi trường

| Nhóm | Package |
|---|---|
| Bắt buộc cơ bản | Python 3.10, NumPy |
| Simulator | MetaDrive 0.4.3 tại commit đã pin |
| GUI | PySide6 6.11.2 |
| Plot/data | matplotlib, pandas |
| Tracking | wandb 0.30.0, tùy chọn khi chạy |
| RL framework | chưa có |

Root `requirements.txt` không cài MetaDrive và không pin Gymnasium trực tiếp. Tài liệu yêu cầu clone MetaDrive riêng và `pip install -e .` tại đúng commit.

Hệ quả thực tế: chỉ cài `requirements.txt` chưa đủ để chạy launcher. Ngay cả lệnh chỉ liệt kê agent cũng import MetaDrive sớm và bị lỗi nếu simulator chưa cài.

Đây là một gap onboarding đáng sửa sau khi platform đã được reproduce.

## 33. Khả năng tái lập

Project tách riêng năm loại seed:

| Seed | Điều khiển |
|---|---|
| Geometry generation seed | hình học đường |
| Environment seed | traffic và stochasticity simulator |
| Agent seed | RNG bên trong agent |
| Training run seed | replicate training trong tương lai |
| Protocol order seed | thứ tự case cố định |

Điểm mạnh:

- map có hash;
- manifest bị khóa;
- simulator version/commit bị pin;
- control frequency cố định;
- artifact có provenance và integrity;
- reset/action determinism có audit.

Giới hạn:

- phụ thuộc native stack, OS và simulator chính xác;
- training reproducibility chưa thể kết luận vì chưa có training;
- framework GPU tương lai có thể thêm nondeterminism;
- máy hiện tại chưa thể kiểm tra lại simulator determinism.

---

# PHẦN VIII — TEAMMATE ĐÃ LÀM ĐẾN ĐÂU?

## 34. Ma trận hoàn thiện dự án

| Component | Trạng thái | Bằng chứng | Chạy được trên máy hiện tại? |
|---|---|---|---|
| MapSuite package | **HOÀN THÀNH** | verifier và 49 test pass | Có |
| Split/evaluation manifest | **HOÀN THÀNH** | 180/48/12, 96/60, không leakage | Có ở data layer |
| MetaDrive case construction | **GẦN HOÀN THÀNH** | code + simulator audit lịch sử | Chưa, thiếu dependency |
| Observation/AgentInput | **GẦN HOÀN THÀNH** | implementation + test | Pure test chạy được |
| Action adapter | **HOÀN THÀNH** | ba adapter + validation | Có |
| Episode semantics | **HOÀN THÀNH** | classifier + horizon + test | Có |
| Reward | **GẦN HOÀN THÀNH** | code + calibration; chưa freeze | Có |
| Metrics | **HOÀN THÀNH** | implementation + test | Có |
| Agent API | **HOÀN THÀNH** | protocol, descriptor, registry | Có |
| Agent nghiên cứu thật | **CHƯA CÓ** | `src/agents` không có implementation | Không |
| RL training | **CHƯA CÓ** | không có loop/network/optimizer | Không |
| Evaluation executor | **GẦN HOÀN THÀNH** | code + historical smoke | Chưa chạy lại |
| Launcher/preflight | **GẦN HOÀN THÀNH** | code/test/audit lớn | Chưa chạy lại |
| Local logging | **GẦN HOÀN THÀNH** | artifact/integrity implementation | Chưa chạy lại full |
| W&B | **MỘT PHẦN** | local/offline tốt; online hiện chưa verify | Không rõ credential |
| Workbench | **GẦN HOÀN THÀNH** | B1/B2/B3 audit | Chưa chạy lại |
| `src/logging` | **KHUNG RỖNG** | README placeholder | Không cần dùng |
| `experiments` | **KHUNG RỖNG** | chỉ có README | Không |
| CI/package bootstrap | **CHƯA CÓ** | không thấy CI/pyproject phù hợp | Không |

### Kết luận công bằng về teammate trước

Teammate không chỉ tạo scaffold. Họ đã làm một lượng lớn platform engineering có test và audit kỹ. Tuy nhiên họ mới hoàn thành **hạ tầng nghiên cứu**, chưa hoàn thành **kết quả nghiên cứu RL**.

## 35. Lịch sử phát triển qua Git

Nhánh hiện tại: `main`, theo `origin/main`. Không thấy tag. Working tree sạch trước khi thêm báo cáo.

Các giai đoạn chính:

1. episode lifecycle và horizon;
2. reward và metric;
3. split và evaluation protocol;
4. agent input/action contract;
5. logging và W&B;
6. launcher và preflight hardening;
7. Workbench B1/B2/B3;
8. tài liệu handover;
9. đóng gói MapSuite và preview reproducible.

Không thấy commit triển khai Stage 0 hoặc agent học được.

---

# PHẦN IX — PHÂN TÍCH PHẦN CÒN THIẾU VÀ NỢ KỸ THUẬT

## 36. Những phần còn thiếu

| Gap | Bằng chứng | Ảnh hưởng | Ưu tiên | Việc nên làm |
|---|---|---|---|---|
| Chưa có baseline thật | registry chỉ có fixture | chưa có kết quả khoa học đầu tiên | P0 | làm Stage 0 Random/Naive |
| Chưa có training interface | agent không nhận transition/reward | không thể online RL | P0 | thiết kế TRAIN-only adapter |
| Chưa có RL framework | requirements không có Torch/SB3... | chưa train model | P1 | chọn và pin một framework |
| Chưa có checkpoint lifecycle | chỉ có cờ `requires_checkpoint` | không lưu/load policy học được | P1 | định nghĩa save/load/freeze/provenance |
| Setup chưa trọn gói | MetaDrive phải cài tay | onboarding dễ lỗi | P1 | health check/bootstrap và lazy import |
| Chưa chứng minh train → eval | không có model | chưa biết integration học được | P1 | training smoke nhỏ rồi load qua AgentPolicy |
| Reward chưa freeze | code/JSON ghi NON-FROZEN | training lớn có thể phải làm lại | P1 | nghiên cứu bằng TRAIN/VALIDATION |
| Chưa thấy CI | thiếu clean-machine automation | regression dễ lọt | P2 | thêm test job thuần và simulator job |
| Docs còn lệch | lane, logging README, template | người mới dễ hiểu nhầm | P2 | hợp nhất tài liệu nguồn sự thật |

## 37. Nợ kỹ thuật cụ thể

1. Launcher import MetaDrive quá sớm, làm `agents/cases/plan` cũng không chạy nếu thiếu simulator.
2. Training seam chưa tồn tại.
3. MetaDrive/Gymnasium không được khai báo đầy đủ trong requirements.
4. Executor lặp hai lần cùng nhóm kiểm tra descriptor so với registration.
5. Một số module logging/GUI dùng `except Exception`; có chỗ hợp lý để cách ly lỗi, có chỗ làm khó debug.
6. Giá trị khoa học được lặp trong code, JSON, docs và hash; audit phát hiện drift nhưng bảo trì vẫn khó.
7. Windows có vấn đề import order MetaDrive/PySide6 được tài liệu hóa.
8. Console `cp1258` lỗi với đường dẫn tiếng Việt; `PYTHONUTF8=1` giải quyết được.
9. Chưa có packaging chuẩn kiểu `pyproject.toml` cho subproject.

Search không tìm thấy TODO/FIXME/NotImplementedError đáng kể trong production core. `pass` trong abstract interface và broad exception phải đọc theo ngữ cảnh, không thể tự động gọi là code chưa làm.

---

# PHẦN X — NÊN LÀM TIẾP CÁI GÌ?

## 38. Sơ đồ phụ thuộc khi phát triển

```text
Cài đúng MetaDrive và full test xanh
                 ↓
Stage 0 baseline thật
                 ↓
TRAIN-only environment/transition adapter
                 ↓
Xác nhận và freeze reward bằng TRAIN/VALIDATION
                 ↓
Checkpoint + training reproducibility contract
                 ↓
Baseline học được đầu tiên, đề xuất PPO
                 ↓
Chọn model bằng VALIDATION
                 ↓
Freeze model rồi mới chạy TEST
                 ↓
Ablation và cải tiến thuật toán
```

Không nên code PPO trước khi có training transition boundary và checkpoint lifecycle rõ ràng. Nếu không, policy train ra sẽ khó đưa vào launcher chuẩn hoặc khó tái lập.

## 39. Thứ tự ưu tiên

### P0 — Vấn đề chặn tiến độ

- reproduce đúng môi trường MetaDrive;
- tạo Stage 0 baseline thật;
- thiết kế training seam.

### P1 — Chức năng cốt lõi

- RL training adapter;
- first learned baseline;
- checkpoint save/load;
- training seed và provenance;
- freeze reward sau validation.

### P2 — Quan trọng

- CI;
- lazy import cho CLI read-only;
- dọn tài liệu/config;
- test dài hạn và robustness.

### P3 — Nên có nếu còn thời gian

- training controls trong Workbench;
- biểu đồ nâng cao;
- packaging và tiện ích developer.

## 40. Ba nhiệm vụ đầu tiên cho người mới

### Nhiệm vụ 1 — Tái lập Platform V1 trên máy

**Mục tiêu:** cài đúng MetaDrive, chạy full test và một sandbox episode headless.

**Tại sao làm trước:** giúp phân biệt lỗi setup với lỗi code, đồng thời cho bạn nhìn thấy toàn bộ pipeline thật.

**File cần đọc:**

- `docs/GETTING_STARTED.md`;
- `docs/environment/METADRIVE_SETUP.md`;
- `src/launcher/cli.py`;
- `resolver.py`, `preflight.py`, `executor.py`.

**File có thể sửa:** lý tưởng là không sửa; nếu phát hiện setup gap thật thì chỉ cập nhật docs/bootstrap.

**Độ khó:** Trung bình.

**Đầu ra:** full test xanh và một `runs/<run_id>` có integrity VERIFIED.

### Nhiệm vụ 2 — Xây dựng mốc cơ sở Random/Naive ở Giai đoạn 0

**Mục tiêu:** tạo agent khoa học đầu tiên, tách hẳn fixture.

**Tại sao:** đây là milestone tiếp theo được project chỉ rõ; nó kiểm chứng platform trước khi thêm độ phức tạp RL.

**File cần đọc:**

- `docs/AGENT_DEVELOPMENT_GUIDE.md`;
- `src/platform/agent.py`;
- `agent_input.py`;
- `action_adapter.py`;
- `src/launcher/registry.py`;
- test resolver/preflight.

**File dự kiến thêm/sửa:**

- thêm `src/agents/random_baseline.py`;
- sửa `src/agents/__init__.py`;
- sửa `src/launcher/registry.py`;
- thêm test và tài liệu tương ứng.

**Acceptance criteria:**

- seed lặp lại được;
- descriptor khai báo stochasticity đúng;
- không dùng evaluator-private data;
- SANDBOX chạy được;
- fixture vẫn bị chặn ở benchmark;
- eligibility của baseline được review rõ;
- toàn bộ test xanh.

**Độ khó:** Trung bình.

### Nhiệm vụ 3 — Xây bộ chuyển đổi học tăng cường chỉ dùng tập TRAIN

**Mục tiêu:** đưa observation/action/reward/transition chuẩn vào một trainer RL mà không phá inference benchmark.

**Tại sao:** executor hiện chưa thể training.

**File cần đọc:** executor, agent input, action adapter, reward, episode, specs, cases và roadmap.

**File dự kiến thêm:**

- `src/training/` hoặc module tương đương;
- Gymnasium-compatible training environment/collector;
- training config;
- checkpoint contract;
- seed và integration tests.

**Acceptance criteria:**

- reset có seed tái lập được;
- chỉ lấy geometry TRAIN;
- termination/truncation đúng;
- reward khớp `RewardSpecV1`;
- không đọc TEST;
- rollout nhỏ chạy được;
- checkpoint load được vào một `AgentPolicy` dùng cho launcher.

**Độ khó:** Cao.

## 41. Những chỗ người mới chưa nên sửa

- split CSV và evaluation case;
- geometry hash và checksum;
- TEST/VALIDATION protocol;
- `AgentInputV1` public schema;
- danh sách evaluator-private field;
- canonical action schema;
- terminal precedence;
- run artifact schema và integrity;
- reward constant trước khi reproduce calibration;
- scientific authority trong launcher;
- field lane count lịch sử theo kiểu sửa trực tiếp không version.

Những phần này có ảnh hưởng dây chuyền tới hash, audit, reproducibility và khả năng so sánh kết quả cũ.

---

# PHẦN XI — CÀI ĐẶT VÀ CHẠY TỪ MÁY MỚI

## 42. Trình tự cài đặt

### Bước 1 — Clone repository

Thư mục làm việc: workspace cha.

```powershell
git clone <repository-url>
```

Thành công khi thấy thư mục `autonomous-driving-rl`.

### Bước 2 — Tạo môi trường Python

```powershell
conda create -n metadrive python=3.10 -y
conda activate metadrive
```

### Bước 3 — Cài thư viện phụ thuộc của nhóm

Thư mục: root repository, nơi có `requirements.txt`.

```powershell
python -m pip install -r requirements.txt
```

### Bước 4 — Cài đúng MetaDrive

Nên clone ở thư mục ngang hàng với repository, không clone thành package `metadrive` bên trong project.

```powershell
git clone https://github.com/metadriverse/metadrive.git metadrive-src
cd metadrive-src
git checkout 85e5dadc6c7436d324348f6e3d8f8e680c06b4db
python -m pip install -e .
```

### Bước 5 — Kiểm tra việc nhập thư viện

Trong `autonomous-driving-rl/`:

```powershell
python -c "import metadrive; print(metadrive.__file__)"
```

Đường dẫn phải trỏ tới bản pinned vừa cài, không phải một folder shadow ngoài ý muốn.

### Bước 6 — Kiểm tra dataset

Nếu đường dẫn có tiếng Việt trên Windows:

```powershell
$env:PYTHONUTF8='1'
python scripts/verify_mapsuite_v1_dataset.py
```

Điều kiện thành công: `100% PASSED`.

### Bước 7 — Chạy test nhỏ trước

```powershell
python -m unittest `
  tests.test_episode_lifecycle `
  tests.test_reward_metrics `
  tests.test_agent_contract -v
```

### Bước 8 — Tạo plan không chạy simulator

```powershell
python -m src.launcher.cli plan `
  --mode SANDBOX `
  --agent fixture_seeded_random `
  --tier Easy `
  --agent-seed 101
```

Điều kiện thành công: preflight cho phép chạy và chưa tạo run artifact.

### Bước 9 — Chạy toàn bộ kiểm thử

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Nếu gặp native crash do thứ tự import Qt/MetaDrive, dùng Qt-preload command trong `GETTING_STARTED.md`; không được bỏ bớt test.

### Bước 10 — Chạy thử simulator ở mức nhỏ nhất

Lệnh này sẽ tạo file dưới `runs/`:

```powershell
python -m src.launcher.cli run `
  --mode SANDBOX `
  --agent fixture_seeded_random `
  --tier Easy `
  --agent-seed 101 `
  --render OFF
```

Điều kiện thành công:

- status `COMPLETE`;
- có đầy đủ run artifact;
- integrity verify thành công.

### Bước 11 — Chạy Workbench

```powershell
python -m src.workbench.app
```

Hiện chưa có command training hoặc full RL experiment.

---

# PHẦN XII — FILE QUAN TRỌNG VÀ LỘ TRÌNH TIẾP QUẢN

## 43. File cần đọc theo thứ tự

### Tier 0 — Phải hiểu trước khi code

1. `README.md` — project là gì và đang ở stage nào.
2. `src/launcher/executor.py` — luồng end-to-end thật.
3. `src/platform/agent.py` — interface agent và fixture.
4. `src/platform/agent_input.py` — agent được nhìn thấy gì.
5. `src/platform/action_adapter.py` — action đi vào simulator thế nào.
6. `src/platform/episode.py` — episode kết thúc ra sao.
7. `src/platform/reward.py` — tín hiệu học.
8. `src/platform/metrics.py` — cách đánh giá.
9. `src/launcher/resolver.py`, `preflight.py`, `registry.py` — cách enforce luật.
10. `datasets/mapsuite_v1/README.md` và lane erratum — dữ liệu benchmark.

### Tier 1 — Core project

- `launcher/cases.py`;
- `launcher/models.py`;
- `launcher/cli.py`;
- `platform/protocol.py`;
- `platform/specs.py`;
- `platform/experiment_logging.py`;
- `platform/wandb_backend.py`;
- `workbench/worker.py`;
- `workbench/process_runner.py`;
- `workbench/results_repository.py`;
- `configs/platform/*`;
- các test tương ứng.

### Tier 2 và 3

- Workbench widget;
- exporter/verifier;
- audit script và artifact;
- legacy environment/evaluator;
- root inspect script;
- tài liệu lịch sử chi tiết.

## 44. Lộ trình tiếp quản

| Level | Bạn cần làm được gì để qua level? |
|---|---|
| 1 — Orientation | Giải thích được platform khác RL algorithm ở đâu |
| 2 — Run | Chạy verifier, focused test, plan và sandbox episode |
| 3 — Trace | Nói đúng từng function từ CLI tới `run_integrity.json` |
| 4 — Understand | Giải thích đúng environment, input, action, reward, termination |
| 5 — Reproduce | Chạy lại cùng config/seed và so sánh output |
| 6 — Modify | Sửa một thay đổi nhỏ có test mà không phá contract/hash |
| 7 — Extend | Thêm Stage 0 agent và tạo artifact hợp lệ |
| 8 — Research | Train trên TRAIN, chọn bằng VALIDATION, freeze rồi mới TEST |

---

# PHẦN XIII — MÔ HÌNH TƯ DUY DỄ NHỚ

## 45. Câu chuyện của một thí nghiệm

Hãy tưởng tượng launcher là ban tổ chức một cuộc thi xe tự hành.

Đầu tiên, bạn nộp đơn đăng ký: muốn chạy agent nào, ở mode nào, có render hay không, seed nào. Resolver không chạy xe ngay. Nó biến yêu cầu đó thành một kế hoạch cụ thể: thi trên case nào, theo thứ tự nào, dùng agent implementation nào và luật nào.

Preflight giống khâu kiểm tra kỹ thuật trước cuộc đua. Nó kiểm tra agent có được phép thi không, simulator có đúng version không, map có đúng hash không, seed có hợp lệ không và người dùng có đang cố chọn riêng vài đề TEST dễ hay không. Nếu sai, xe chưa được tạo và không có result giả.

Khi plan hợp lệ, executor mới mở một run và dựng MetaDrive. Map không được tạo tùy ý mà được khôi phục từ block sequence đã khóa. Geometry seed xác định con đường; environment seed xác định traffic động. Đây là hai loại seed khác nhau.

Mỗi 0,1 giây, platform lấy observation 259 chiều, thông tin xe xung quanh và waypoint phía trước. Nó đóng gói chúng thành `AgentInputV1`, loại bỏ mọi thông tin riêng của trọng tài. Agent nhìn input rồi trả action. Adapter kiểm tra action và đổi nó thành steering cùng ga/phanh. MetaDrive chạy thêm năm physics step.

Sau bước đó, trọng tài đọc route completion, collision, out-of-road và arrival. Agent không được đọc trực tiếp các giá trị này. Platform phân loại kết quả theo an toàn trước, tính reward và ghi telemetry. Quá trình lặp lại tới khi xe thành công, gặp sự cố hoặc hết thời gian.

Khi episode kết thúc, platform tạo `EpisodeRecord`. Khi toàn bộ case kết thúc, logger tổng hợp metric, ghi file local, tùy chọn gửi bản sao lên W&B và tạo hash integrity. Workbench chỉ hiển thị các dữ liệu đó; nó không tự tính lại luật thi.

Phần còn thiếu là việc học. Hiện chưa có thành phần nào thu transition, tính loss, cập nhật neural network và lưu model. Do đó việc tiếp theo là đặt một baseline thật lên đường thử, rồi xây khu vực training chỉ dùng TRAIN split.

---

# PHẦN XIV — BẢNG THUẬT NGỮ

## 46. Thuật ngữ của project

| Thuật ngữ | Là gì? | Nằm ở đâu / dùng để làm gì? |
|---|---|---|
| MapSuite | bộ benchmark map procedural bị khóa | `datasets/mapsuite_v1` |
| Geometry | một instance đường cụ thể có hash | `geometries.csv`, manifest |
| Scenario family | mẫu topology block đường | `scenario_families.csv` |
| Case | geometry + env seed + traffic + horizon | `launcher/models.py`, case manifest |
| Contract | quy tắc/interface có version và hash | `configs/platform`, Python builder |
| Spec | bộ tham số khoa học | episode/reward/metric config |
| Protocol | luật split, seed, thứ tự, báo cáo | evaluation protocol |
| Episode | một lượt lái từ reset tới kết thúc | executor/episode/logger |
| `AgentInputV1` | dữ liệu sạch được phép đưa cho agent | `platform/agent_input.py` |
| `AgentPolicy` | interface của mọi agent | `platform/agent.py` |
| `AgentDecision` | action payload và diagnostic an toàn | `platform/agent.py` |
| `ActionAdapter` | đổi action agent sang actuator chuẩn | `platform/action_adapter.py` |
| Resolver | biến request thành plan đầy đủ | `launcher/resolver.py` |
| Registry | nối agent ID với metadata và factory | `launcher/registry.py` |
| Preflight | kiểm tra chặn trước execution | `launcher/preflight.py` |
| Workbench | GUI mỏng trên launcher | `src/workbench` |
| Audit | kiểm tra có thể tái lập và tạo evidence | `scripts/audit_*`, `results/audits` |
| Fixture | agent chỉ dùng test platform | `platform/agent.py` |
| Clean success | tới đích mà không có safety violation | episode/metrics |
| Canonical run | full VALIDATION/TEST theo luật khóa | resolver/preflight/executor |

---

# PHẦN XV — CÂU HỎI TỰ KIỂM TRA

## 47. Mười câu cơ bản

1. Simulator nào đang được dùng?
2. Platform V1 có phải một RL agent đã train không?
3. Output runtime nằm ở đâu?
4. Local file hay W&B là nguồn sự thật?
5. MapSuite có bao nhiêu geometry?
6. Có bao nhiêu scenario family?
7. Số lượng TRAIN/VALIDATION/TEST là bao nhiêu?
8. `CourseEnvV1` có phải canonical environment không?
9. Milestone kế tiếp là gì?
10. Fixture có được chạy TEST benchmark không?

**Đáp án:** 1 MetaDrive 0.4.3 tại commit đã pin. 2 Không. 3 `runs/<run_id>`. 4 Local file. 5 240. 6 12. 7 180/48/12. 8 Không, đó là legacy. 9 Stage 0 Random/Naive baseline. 10 Không.

## 48. Mười câu về kiến trúc

1. Resolver trả về object gì?
2. Preflight chạy lúc nào?
3. Ai tạo `MetaDriveEnv`?
4. GUI hay launcher giữ scientific authority?
5. Registry dùng để làm gì?
6. Workbench có bắt buộc không?
7. Cái gì khóa identity của geometry?
8. Tại sao executor chạy preflight lại?
9. Metric tổng hợp nằm ở đâu?
10. Cái gì kiểm tra result có bị sửa?

**Đáp án:** 1 `ResolvedExperimentPlanV1`. 2 Trước mọi side effect và được kiểm tra lại trong executor. 3 `ExperimentExecutor`. 4 Launcher/platform. 5 Nối identity và capability với factory thật. 6 Không. 7 Exact block sequence và SHA-256. 8 Defense in depth. 9 `platform/metrics.py` và logger. 10 `run_integrity.json` cùng Results Repository.

## 49. Mười câu về môi trường và tác tử

1. Lifecycle của agent gồm những hàm nào?
2. Input profile chính là gì?
3. Agent ra quyết định ở tần số bao nhiêu?
4. Physics step và decision repeat là bao nhiêu?
5. Environment seed đưa cho ai?
6. Agent có thấy route completion không?
7. Traffic context chứa tối đa bao nhiêu xe và trong bán kính nào?
8. Task lookahead có bao nhiêu điểm?
9. Agent được tạo mỗi run hay mỗi episode?
10. Agent reset khi nào?

**Đáp án:** 1 descriptor/reset/act/close. 2 `STATE_DECISION_V1`. 3 10 Hz. 4 0,02 giây và repeat 5. 5 MetaDrive. 6 Không. 7 8 xe trong 50 m. 8 20 điểm, cách nhau 2,5 m. 9 Một lần mỗi run. 10 Trước từng episode.

## 50. Mười câu về quan sát, hành động và phần thưởng

1. Core observation có bao nhiêu chiều?
2. Ba phần của nó là gì?
3. Action vật lý chuẩn là gì?
4. Có silent clipping không?
5. Có những adapter nào?
6. Progress reward tính thế nào?
7. Tổng time cost tối đa cả horizon là bao nhiêu?
8. Clean success được cộng bao nhiêu?
9. Safety failure bị trừ bao nhiêu?
10. Episode return có phải điểm xếp hạng không?

**Đáp án:** 1 259. 2 9 ego + 10 navigation + 240 LiDAR. 3 `[steering, throttle_brake]`. 4 Không. 5 continuous2, discrete25, discrete9. 6 Chênh lệch route completion có dấu. 7 Khoảng 0,25. 8 +1. 9 -1. 10 Không, chỉ là diagnostic/training signal.

## 51. Mười câu về luồng thực thi

1. CLI entry point là gì?
2. GUI chạy core bằng cơ chế nào?
3. Ai chọn case?
4. Map được dựng lại bằng gì?
5. Hàm nào tạo input mỗi step?
6. Latency đo quanh đoạn nào?
7. Hàm nào phân loại episode outcome?
8. Ai ghi episode row?
9. Ai tạo integrity hash cuối run?
10. Action không hợp lệ dẫn tới điều gì?

**Đáp án:** 1 `src.launcher.cli:main`. 2 QProcess worker và sentinel JSON. 3 Resolver/cases. 4 Locked block sequence + `PG_MAP_FILE`. 5 `build_agent_input`. 6 Chỉ quanh `agent.act`. 7 `classify_episode_outcome`. 8 `LocalExperimentLogger.log_episode`. 9 `finalize_run`. 10 Technical execution failure, không tự clipping.

## 52. Mười câu về việc phát triển tiếp

1. Task thuật toán đầu tiên là gì?
2. Tại sao nên làm trước PPO?
3. Training không được đọc gì?
4. Split nào dùng train?
5. Split nào dùng chọn checkpoint?
6. Split nào là final holdout?
7. Training cần thêm lifecycle gì?
8. Checkpoint provenance cần ghi gì?
9. Kết quả TEST kém có được đổi map không?
10. Khi nào được thay contract?

**Đáp án:** 1 Stage 0 baseline. 2 Để chứng minh pipeline và tạo mốc tham chiếu. 3 Evaluator-private data và TEST metadata. 4 TRAIN. 5 VALIDATION. 6 TEST. 7 Collect transition, update, save và load. 8 Code/config/seed/model identity. 9 Không. 10 Chỉ khi thay đổi được version hóa, review và tái tạo evidence đầy đủ.

---

# KẾT LUẬN BÀN GIAO

Teammate trước đã hoàn thành một nền tảng nghiên cứu khá chắc, nhiều test và audit. Phần họ chưa làm không phải vài TODO nhỏ, mà là lớp thuật toán nghiên cứu: agent thật, training và checkpoint.

Hướng tiếp quản an toàn nhất:

```text
Reproduce môi trường
→ chạy platform hiện tại
→ làm Stage 0 baseline
→ xây TRAIN-only RL adapter
→ train baseline học được
→ chọn bằng VALIDATION
→ freeze
→ chạy TEST
```

Không nên thay benchmark để làm cho thuật toán có kết quả đẹp hơn. Hãy giữ Platform V1 làm “luật chơi cố định” và phát triển agent ở phía trên các interface đã có.
