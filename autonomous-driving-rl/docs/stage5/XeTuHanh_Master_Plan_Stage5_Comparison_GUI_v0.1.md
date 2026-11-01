# XeTuHanh — Master Research & Implementation Plan

**Phiên bản:** DRAFT v0.1, ngày 30/10/2026  
**Phạm vi:** Chuẩn hóa so sánh liên Stage, Stage 5 Model-Free RL, mở rộng Research Workbench; quy trình OpenCode Gemini → GitHub PR → kiểm toán độc lập.  
**Trạng thái:** KẾ HOẠCH, CHƯA ĐƯỢC PHÊ DUYỆT CHO TRIỂN KHAI. Không mã nguồn, issue, branch hay PR nào được thay đổi khi soạn kế hoạch.  
**Repository:** https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject  
**Baseline khảo sát:** `main` tại `dcd653306a0e072a2c6bee66eaeadf388101d679` (commit 23/10/2026).  
**Mốc bàn giao cũ:** PR #27 merge SHA `6a531b148b40cff0f7dd343552090aeeb5675957` (06/10/2026).

## 1. Quyết định xuất phát và bằng chứng

- Platform V1, MapSuite V1, Launcher, Workbench V1 đã có code + audit artifacts. Không xây dựng lại hệ thống khoa học chỉ để phục vụ Stage 5.
- 240 geometries; split 180 TRAIN, 48 VALIDATION, 12 TEST; canonical full suite 96 VALIDATION / 60 TEST cases. 12 sequence families trong bốn tier; MetaDrive v0.4.3 tại commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`.
- Gate 6 cung cấp `AgentInputV1` (`STATE_DECISION_V1`: Core 259D + structured TrafficContextV1 + TaskContextV1) và action adapters: `continuous_box2_v1`, `discrete25_native_v1`, `discrete9_lowbranch_v1`.
- Five primary metrics locked: clean_success_rate, safety_failure_rate, mean_final_route_completion, median_final_route_completion, mean_time_to_clean_success_s. Không gộp mega-score; episode return chỉ là learning diagnostic.
- Seed protocol: validation env seeds 5101/5102; test env seeds 9101–9105; agent/training replicate roots 101/202/303; protocol ordering 424242. Với learned methods phải có ba mô hình được huấn luyện độc lập. Không đánh đồng ba mô hình với 180 quan sát độc lập.
- Gate 7 logs local authoritative (`run_manifest`, `run_state`, `episodes`, `timing`, `summary`, `run_integrity`...), W&B optional. Results Browser chỉ tin verified COMPLETE; GUI không được tính lại kết quả canonical hay sửa artifacts.
- `src/workbench` hiện đã có sáu tab; chưa có trainer UI và multi-run scientific comparison; Workbench Results B2 có `NO_LEADERBOARDS_NO_MULTI_RUN_RANKING`.
- Stage 0 code được Khoa thêm 09/10, benchmark_eligible=False. Kiểm tra tĩnh nêu blocker: import từ `src.agents` nhưng `__init__.py` không export class; `executor.py` import `src.visualization.minimap_overlay` nhưng thư mục/module đó không có trên main; typo `_step_couter/_step_counter`; chú thích action 7 không đúng mapping. Chưa xác nhận bằng runtime.
- Sau 06/10 có 23 commit trên main, chủ yếu coursework; không phát hiện Stage 5 implementation hoặc Workbench extension trong diff đó.
- Git tree không có `.github/workflows/` ở HEAD đã kiểm tra: chưa có bằng chứng CI tự động trên GitHub. R0 nên có lệnh unit regression tái lập được; có thể đề xuất GitHub Actions cho các pure tests như một PR hạ tầng nhỏ được review riêng, nhưng không đặt integration MetaDrive nặng vào CI khi chưa có runner phù hợp.

## 2. Câu hỏi nghiên cứu và nguyên tắc

**Q1 — Scientific comparison:** Khi giữ cùng benchmark, quyền truy cập thông tin, số lần đánh giá và chuẩn báo cáo, các phương pháp 0–7 khác nhau như thế nào về success/safety/progress/efficiency và inference cost?

**Q2 — Stage 5 feasibility:** Có thể huấn luyện model-free policy trong TRAIN của MapSuite V1 và nạp checkpoint frozen qua cùng AgentPolicy/Launcher mà không phá information parity hay environment parity không?

**Q3 — Added value of learning:** Stage 5 có cải thiện trên baseline Stage 0, Stage 1 khi kiểm soát ngân sách, action representation và seed variability không? Cải thiện chưa được giả định trước.

**Q4 — Interfaces:** GUI có thể theo dõi training và so sánh verified experiments mà không trở thành nguồn sự thật thứ hai không?

Các nguyên tắc: TEST chỉ dùng cho final frozen evaluation; không tự thay đổi locked manifests / metrics / lifecycle / seed meanings; mọi thay đổi hợp đồng phải có ADR, version bump, review riêng. Tách lỗi tích hợp/hạ tầng khỏi lỗi học không tiến triển và khỏi kết luận khoa học. Mỗi gate tạo PR nhỏ, có bằng chứng và điểm dừng; không merge khi chưa review.

## 3. Kế hoạch so sánh chung: Comparison Specification V1 (C-01)

### 3.1 Bốn lớp dữ liệu

| Lớp | Nội dung tối thiểu | Vai trò |
|---|---|---|
| **Identity/provenance** | run_id, agent_id/version, method_family/stage, Git SHA/dirty state, MetaDrive pin, benchmark/agent/logging/launcher hash, input profile, action adapter, checkpoint ID/SHA, seed lineage, mode/split/case manifest | Cho biết run có hợp lệ và so sánh được không |
| **Per-episode** | case_id, tier, sequence, geom/env seeds, horizon, agent seed, terminal reason, clean success, raw arrival, final/max completion, episode steps/s, time to clean success (nullable), safety flags, episode return, technical-failure status | Chẩn đoán và kiểm tra paired cases |
| **Primary benchmark** | 5 primary metrics từ `summary.json` theo overall và 4 tier, kèm sample counts, replicate identity | Kết luận chính; chỉ dữ liệu VERIFIED + COMPLETE + canonical |
| **Supporting/training** | safety breakdown, timeout, mean/p95 act latency, training_env_steps, wall time, optimizer updates, policy loss/value loss/entropy, training seed, model/config hashes, compute hardware | Giải thích chất lượng, chi phí và tính khả thi; KHÔNG thay benchmark score |

Các training fields mới phải được quy định ở **training schema riêng**, không lén sửa EvaluationMetricsV1/episodes.csv. Không tự thêm nhãn authoritative vào dashboard chỉ vì file JSON trông hợp lệ.

### 3.2 Eligibility và comparability

Chỉ cho phép phép so sánh chính thức khi các run: (i) canonical verified COMPLETE; (ii) cùng benchmark contract + simulator pin; (iii) cùng case manifest & episode coverage; (iv) đúng seed taxonomy; (v) inference frozen, không update weights; (vi) checkpoint & preprocessing provenance đầy đủ; (vii) không technical failures bị coi nhầm là physical failure; (viii) thông báo mọi khác biệt action adapter/observation selection/training budgets. Nếu khác hợp đồng, tách bảng, không xếp hạng chung.

### 3.3 Tóm tắt thống kê

- Mỗi trained checkpoint/replicate chạy đầy đủ suite 96 VALIDATION trong giai đoạn lựa chọn và 60 TEST trong final pass; **không chọn subset canonical**.
- Ba training roots 101/202/303 tạo ba mô hình độc lập (mỗi model giữ nguyên mọi cấu hình ngoại trừ nguồn randomness). Với frozen deterministic inference, không tự nhân thêm ba inference seeds như ba mô hình mới. Với stochastic inference, định nghĩa kế hoạch replicate tương thích protocol trước khi chạy; tránh đếm trùng variation.
- Trên mỗi model/replicate, tính các primary metrics trong full suite và từng tier; sau đó báo cáo mean ± sample std qua training replicates (n=3). Báo cáo `n_train_replicates=3`, `n_cases_per_model=60` (TEST), không mô tả 180 episode là 180 training replicates.
- Đối chiếu paired case_id/geometry/env seed, dùng per-case differences để phân tích nhưng không bootstrap mỗi episode như đơn vị độc lập nếu bỏ qua model/geometry clustering. Với n=3, hạn chế tuyên bố ý nghĩa thống kê. CI/phân tích bootstrap phân tầng là mở rộng có kiểm chứng, không bắt buộc ở core.
- Không biến `mean_time_to_clean_success_s=null` khi không thành công thành 0, vô cực hoặc thắng/thua tùy tiện; luôn hiển thị số lần thành công. Không so sánh success time tách khỏi success rate.
- Không dùng aggregate reward làm chỉ số thắng. Không phát minh mega-score. Báo cáo side-by-side per-tier và trade-offs.

**Deliverables C-01:** `COMPARISON_SPEC_V1.md` (định nghĩa + invalid cases + null semantics), `comparison_fields.yaml/json` dạng **derived schema** nếu cần, `COMPARISON_TEAM_SIGNOFF.md` ghi quyết định nhóm, và 3–5 ví dụ synthetic expected outputs. Chưa sửa core metric contract.

## 4. Thiết kế Stage 5: lựa chọn kỹ thuật có điều kiện

### 4.1 MDP và ranh giới quyền

Huấn luyện tối ưu `E[Σ gamma^t r_t]`, benchmark đo năm metrics locked. Training environment có thể tính reward từ evaluator/internal telemetry, **nhưng policy/actor không được nhận hidden route completion, split, case ID, terminal verdict, future state**. Training wrapper phải bảo đảm observation của actor chỉ là biểu diễn xác định, versioned, của AgentInputV1. Không dùng trực tiếp CourseEnvV1 legacy (35D/Discrete(5)).

### 4.2 Thuật toán

**Default có điều kiện: PPO + continuous Box2**, bởi API hỗ trợ action liên tục và dễ tích hợp với SB3 để xác thực pipeline. Không có giả định PPO thắng. **SAC + cùng Box2** là extension đối chứng về sample efficiency, nếu sau pilot có ngân sách. DQN+discrete9 là một nhánh tùy chọn khác, nhưng không nên khởi động đồng thời cả ba do dễ nhiễu do khác action space và chi phí review. Khóa phiên bản PyTorch/Gymnasium/SB3 **sau dependency smoke** với pinned Python 3.10/MetaDrive 0.4.3; root requirements hiện không liệt kê SB3/Torch.

### 4.3 Observation design

- **Feasibility smoke:** chỉ `CoreObservationV1.features` 259D, là subset hợp lệ của STATE_DECISION_V1; ghi rõ đây là `core-only feature usage`, không đánh tráo profile/version.
- **Main candidate:** thử biểu diễn có cấu trúc từ full AgentInputV1: 259D + padded 8x7 actors + actor validity mask + 20x3 route waypoints + waypoint validity mask + navigation/lane metadata và các cờ được phép. Xác định thứ tự ghép, chuẩn hóa/masking trong `ObservationEncoderV1` thuần hàm; checkpoint lưu encoder version/hash.
- Chỉ fit normalization statistics trên TRAIN; inference dùng cùng immutable saved state. Nên benchmark hai biểu diễn như **ablation nội bộ Stage 5**, không coi chúng là cùng một setting. Kiểm soát nguồn các giá trị, types, NaN, buffer copies, memory, không lộ evaluator-only fields.

### 4.4 Training environment: thiết kế bắt buộc

Tạo Gymnasium adapter, không để model gọi simulator internals. Adapter phải tái sử dụng các pure functions/contract của Platform V1: MapSuite frozen TRAIN cases, deterministic seed derivation, control frequency 10Hz, traffic Trigger, route-aware horizon, safety precedence, action adapter, RewardSpecV1 và AgentInput builder. Không sao chép/cải biên công thức reward/termination một cách độc lập. Nếu code evaluator chỉ có logic embedded, thiết kế **minimal shared refactor** qua PR riêng, bảo vệ hash/evidence và backward compatibility.

Gymnasium `reset(seed)` và `step()` phải trả đúng `obs, reward, terminated, truncated, info`; timeout cần giữ `truncated=True` theo semantics. `check_env` và deterministic step-parity tests bắt buộc. Thử cùng hành động cố định trên **một TRAIN case** bằng evaluator/AUDIT và training wrapper để so sánh observation/action/reward/outcome trajectories có tolerance rõ. Dùng preflight/evidence để không sử dụng TEST trong quá trình phát triển.

### 4.5 Training scheduling

TRAIN chứa 180 geometry khác nhau. Pilot bắt đầu với vài TRAIN cases đã ghi danh để test logic, sau đó phát triển scheduler lấy các geometry trên TRAIN theo policy reproducible. Có thể curriculum từ Easy lên Hard/Extreme nhưng phải ghi schedule/mixture/sampling weights và bản version; không lấy kiến thức từ TEST để thiết kế curriculum. Tách geometry seed, env seed và training seed; không bao giờ dùng validation/test seed trong training.

**Budget theo gate, không giả định số giờ:** 1–5k env steps để chứng minh code chạy (chỉ là engineering smoke, không tính kết luận học); 10–50k cho learning-signal probe tùy measured steps/s; full budget và ba replicates chỉ khóa sau pilot. Không hứa chất lượng của từng mốc. Wall time dự báo bằng `total_env_steps / measured_env_steps_per_second` cộng training & validation overhead. Không tự chọn GPU/CPU nếu chưa đo môi trường. Bắt đầu với **một environment/worker**, chưa bật `SubprocVecEnv`/parallel MetaDrive trên Windows trước khi có integration stress test về process spawn, Panda3D teardown, CPU/RAM; sau đó so sánh throughput và ổn định trước khi quyết định vectorization.

### 4.6 Checkpoint và sự thật khoa học

Checkpoint gồm model weights, policy config, encoder config/hash, normalization state, framework/dependency versions, training seed, exact train dataset/sampling schedule, reward spec/version, total env steps, file SHA256, Git SHA, time/provenance. Training artifact và benchmark run artifact **không phải cùng một thứ**. Evaluation agent phải load checkpoint read-only, không cập nhật statistics, gradient, replay buffer hay weights. Mọi model checkpoint dùng để benchmark phải được xác thực digest. Large binaries không commit bừa vào Git; phương án lưu trữ versioned external mirrors phải được duyệt với mapping checksums. Failure/missing checkpoint → fail preflight, không tự load latest hoặc fallback random.

## 5. Giao diện theo đúng kiến trúc

**Workbench V1:** sáu tab sẵn có, keep intact. Không biến GUI thành evaluator/logger/optimizer.

**GUI-C (Comparison):** read-only, load nhiều local VERIFIED COMPLETE canonical run snapshots; kiểm tra comparability; hiển thị per-tier 5 primary metrics, N counts, replicate mean±std, paired-case diagnostics, safety breakdown, latency; nếu không tương thích, banner giải thích và disable official comparison. Derived statistics được gắn nhãn analysis-derived, không chỉnh `summary.json`. Không đưa leaderboard tùy tiện vào contract B2; xây một versioned extension riêng.

**GUI-T (Training):** xem tiến trình và artifacts do trainer phát ra (không lấy inference sampling telemetry làm training loss). Không cho phép GUI đọc/cấu hình TEST trong training. Training UI có thể bắt đầu với read-only logs; start/stop/resume/parallel workers là scope mở rộng chỉ làm sau process lifecycle audit. Existing `src/workbench/process_runner.py` chỉ có những bảo đảm đã audit cho launcher worker, không tự suy rộng cho trainer.

**UX ẩn dụ:** compare reproducible experiment bundles, không compare screenshots hoặc online W&B chart không xác minh.

## 6. Lộ trình gate và định nghĩa Done

| Gate/PR | Mục tiêu & trách nhiệm | Công việc chính | Evidence/Acceptance (bắt buộc) | Chặn gate sau khi fail |
|---|---|---|---|---|
| **R0 — Integration Recovery** | Khoa sửa Stage 0; người phụ trách nền tảng review | Fix imports & optional minimap; sửa counter/action annotation; không đụng benchmark definition | CLI `agents`, `plan SANDBOX`, runner/Workbench imports, unit suite chạy, one sandbox smoke; no contract regression; PR nhỏ | **Block C/S5/GUI runtime** nếu CLI còn hỏng |
| **C1 — Comparison Charter** | Tri chủ trì, nhóm duyệt | 5 primary locked; 4 tier; secondary; training extra fields; seeds/replicates; null; comparability; bảng biểu mẫu | Docs-only PR, ký duyệt semantics; synthetic cases test đặc biệt no success, technical failures, mismatched hash, incomplete suite | **Block benchmark comparison** nếu chưa thống nhất |
| **S5-A — Technical ADR** | Tri | Lựa chọn PPO/Box2/obs version, storage/dependencies/budget, path boundaries & parity plan | ADR được review; lock dependencies sau smoke; no code triển khai thuật toán | Block training adapter |
| **S5-B — Training Env Parity** | Tri | Gym adapter dùng TRAIN, reuse platform contracts, seed sampler, finite obs and action, terminated/truncated, reward parity | `check_env`, deterministic reproduction, fixed-action step parity trên TRAIN, tests không MetaDrive + integration pinned; memory teardown | Block real training nếu parity fail |
| **S5-C — Minimal PPO Loop** | Tri | One training seed, short budget, saved model, frozen inference adapter, sandbox launcher; debug reward/outcome trends | CPU/GPU dependency matrix; nonzero steps, finite grads/weights, successful save→load→same deterministic action; no benchmark rights yet | Block multi-seed nếu không load được |
| **S5-D — Reproducibility** | Tri | 3 training seed roots, scheduler, configs, training logs, checkpoint digest, optimizer/normalizer persistence, experiment manifests | Reproduce source/case schedule; independent model identities; negative tests for missing/tampered model and eval-time updates | Block canonical eval nếu provenance fail |
| **S5-E — VAL & Freeze** | Tri | Eligible registered policy, full 96 validation cases per checkpoint; select model/checkpoint using predeclared process; freeze training config & final set | Canonical VERIFIED COMPLETE logs, 96 cases per model, peer-reviewed selection trace, checkpoint package reviewed | **No TEST** trước freeze |
| **GUI-C — Verified Comparison** | Tri | Versioned B2-compatible extension, compare verified runs, per-tier metric tables, uncertainty display, no recompute | Synthetic fixtures: mismatched contract, nulls, incomplete, tampered, missing models; GUI smoke; regression hardening | Block official group reporting if trust gate fails |
| **GUI-T — Training Monitor** | Tri | Initially read-only trainer results, progress charts and checkpoint inspector | Observer is non-authoritative, pause/interruption truthfulness, bounded resource tests | UI launch-control optional |
| **S5-F — Final TEST & Report** | Tri + team | 3 frozen trained policies in 60 TEST full cases each, same protocol across stages; analyze trade-offs & limitations | VERIFIED artifacts; pre-registered selection; sample/seed counts, per-tier scorecard, latency, compute; signoff & no leakage | Report invalid if TEST influenced tuning |
| **S5-G — Optional SAC/Ablation** | Tri only if budget | SAC with same action/obs/reward/budget, or core-vs-full obs ablation (TRAIN/VAL) | Scientific reason + independent budget + controlled design. Separate PR; optional | Skip rather than jeopardize main result |

**Dependency graph:** R0 → S5-A → S5-B → S5-C → S5-D → S5-E → S5-F. C1 precedes any official comparison or TEST. GUI-C requires C1 + trustworthy verified results, GUI-T requires S5-D training outputs. UI visual improvement must not delay research if time constrained. Gate S5-G is OPTIONAL.

## 7. Review protocol: OpenCode Gemini → GitHub

**Mỗi gate là một nhiệm vụ riêng.** Trước khi gửi prompt, chủ dự án review acceptance criteria + no-go boundaries. OpenCode được giao base SHA, scope file paths, frozen contracts và yêu cầu không sửa ngoài phạm vi. Nếu thiếu dependency hoặc muốn thay contract, OpenCode phải báo blocker + ADR, không tự tiện bypass.

**Báo cáo bắt buộc khi agent kết thúc:** branch + HEAD full SHA; changed-file list; docs/ADR; commands exact; test outputs (exit codes); environment version (`python`, MetaDrive Git pin, SB3/Torch nếu có); audit evidence & provenance; danh sách TODO/limitations; PR link. Nêu rõ test nào chưa chạy và lý do. Không tự tuyên bố 'audit pass' nếu chưa có bằng chứng.

**Kiểm toán độc lập trên GitHub:** (a) xác minh HEAD & base/merge-base SHA; (b) fetch diff/changed files; (c) check locked hashes, no leakage/fixture promotion/fallback; (d) inspect tests + negative tests; (e) verify auditable logs not just chat claims; (f) assess theory/implementation/scientific validity separately; (g) review whether output matches scope & compatibility; (h) yêu cầu corrections trên cùng PR khi cần; (i) chỉ merge sau khi chủ dự án chấp thuận. Trước khi merge kiểm tra HEAD lần cuối để tránh code drift. Nếu reviewer không chạy được môi trường, ghi **runtime evidence not independently reproduced**.

**Yêu cầu an toàn về quy trình:** không force push main, không sửa upstream MetaDrive pin, không overwrite run artifacts, không regenerate checksums để che lỗi, không upload API keys/training private local paths, không chạy TEST smoke, không chữa bug bằng cách hạ điều kiện preflight.

### Mẫu brief (chỉ dùng sau khi phê duyệt gate)

> Task: Implement exactly Gate `<ID>` against pinned base `<SHA>` in branch `<branch>`. Read README, STAGE_ROADMAP, PLATFORM_V1_CONTRACT_INDEX, EVIDENCE_INDEX, AGENT_DEVELOPMENT_GUIDE and current code before edits. Scope: `<allowed paths>`. Frozen: benchmark hashes/splits/cases, AgentInputV1 information parity, metrics, reward/lifecycle semantics, provenance and Workbench verified-only truth. Deliver contract-aligned code, positive/negative tests, machine evidence, docs, reproducible commands and a PR. Stop and report blockers; do not silently extend scope or create results without running tests. Do not merge. At completion report exact HEAD SHA, changed files and test evidence.

## 8. Điểm quyết định trước khi phê duyệt triển khai

| Decision | Đề xuất hiện tại | Quyết định cần từ |
|---|---|---|
| Mức độ hoàn thiện 8 Stage | Stage 0/1/5 là course core; 2/3/4/6/7 ownership theo thầy nhưng mức độ làm cần xác nhận | Toàn nhóm / thầy Tài |
| Primary metrics & protocol | Giữ Gate 4/5, lập comparison charter ngoài benchmark | Toàn nhóm |
| Action representation cho Stage5 | PPO continuous Box2 first, SAC only later | Tri + nhóm |
| Stage5 observation | 259D smoke, full structured V1 encoded as main candidate sau parity | Tri |
| Dependency pins | Chọn khi test Python3.10/MetaDrive0.4.3/SB3/PyTorch trên máy thực | Tri / máy triển khai |
| Training budget | Đo throughput trước, sau đó chốt steps/hardware/replicate budget | Tri / nhóm |
| Checkpoint storage | Versioned hashed package; không lấy latest dynamic checkpoint | Tri / nhóm |
| Eval policy | Frozen deterministic inference mặc định nếu phù hợp; predeclare | Tri / nhóm |
| Workbench UI scope | Comparison read-only trước, training monitor sau | Tri |
| Stage0 runtime fixes | Khoa owner, Tri review integration | Khoa + Tri |

## 9. Risk register có phản ứng cụ thể

| Rủi ro | Tín hiệu | Giảm thiểu / hành động |
|---|---|---|
| Import regression chặn launcher | `ModuleNotFoundError`, registry import fail | R0 trước mọi runtime work; import smoke toàn bộ entrypoints |
| Training-eval divergence | Fixed-action traces/rewards/outcomes lệch | Stop S5-B; extract shared helper, regression audit |
| Leakage & unfairness | agent đọc global completion/split/future/test | Static grep, negative access tests, locked AgentInput encoder; reject PR |
| Reward hacking / stalled learning | return tăng, success/safety không tốt | Track primary VAL, reward component ablation trên TRAIN/VAL, không tune TEST |
| Missing reproducibility | same seed không ổn định, invalid checkpoint | seed DAG, save RNG/normalizer when needed, version pins, report nondeterminism |
| Over-budget compute | Throughput thấp, GPU RAM cao, 3 repeats không khả thi | Tiny smoke, benchmark steps/s, cap budgets, avoid parallel 3-algorithm start |
| Fake statistical confidence | Episode bootstrap ignores 3 trained models | Report nested units & n=3; conservative inference, no unwarranted significance |
| UI claims unverified truth | GUI recomputes canonical metric, accepts incomplete runs | Verified-only loader, derived analysis labeling, negative fixtures |
| Cross-team code collisions | multiple people edit registry/executor simultaneously | reserve shared-file ownership, sequential integration PR, mandatory rebase & review |
| Bench hash mutation by agent | hashes altered to make preflight pass | reject unless independent versioned contract migration approved |

## 10. Nguồn kỹ thuật đối chiếu

**Trong repository:**
- https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/blob/main/autonomous-driving-rl/docs/PLATFORM_V1_CONTRACT_INDEX.md
- https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/blob/main/autonomous-driving-rl/docs/EXPERIMENT_WORKFLOW.md
- https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/blob/main/autonomous-driving-rl/docs/AGENT_DEVELOPMENT_GUIDE.md
- https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/blob/main/autonomous-driving-rl/configs/platform/evaluation_protocol_v1.json
- https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/blob/main/autonomous-driving-rl/configs/platform/logging_contract_v1.json
- https://github.com/doduyquy/sgu-team-2026-advanced-ai-subject/blob/main/autonomous-driving-rl/configs/platform/workbench_results_contract_v1.json

**Khoa học / phần mềm (tra cứu 30/10/2026):**
- Stable-Baselines3, Reinforcement Learning Tips and Tricks: https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
- Stable-Baselines3, Using Custom Environments: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
- Stable-Baselines3 PPO: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- Stable-Baselines3 SAC: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
- Henderson et al. (2018), Deep Reinforcement Learning That Matters: https://ojs.aaai.org/index.php/AAAI/article/view/11694
- Agarwal et al. (2021), RLiable: https://research.google/blog/rliable-towards-reliable-evaluation-reporting-in-reinforcement-learning/

**Chú ý:** đây là bản thiết kế điều hành dự án, không phải kết quả benchmark, không phải bằng chứng chạy được code, và không khóa công cụ/thuật toán cho đến khi ADR được duyệt.
