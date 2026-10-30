# XeTuHanh Platform V1 — Open Scientific Decisions V1

- **Document ID:** `DECISIONS-OPEN-V1`
- **Version:** `1.0.0-draft`
- **Date:** 2026-10-30
- **Status:** `DRAFT / TEAM REVIEW REQUIRED` — Not an Approved Scientific Lock
- **Subproject Root:** `autonomous-driving-rl/`
- **Baseline Git Commit:** `dcd653306a0e072a2c6bee66eaeadf388101d679`

---

## 1. Governance and Sign-Off Status

This document records unresolved methodological decisions, statistical options, and interface questions required before large-scale benchmarking under Stage 5.

### Sign-Off Registry
| Role / Focus Area | Researcher | Scope / Responsibility | Review Status | Date |
|---|---|---|---|---|
| **Stages 0 / 1 / 4** | Đặng Khoa | Random Baseline, Rule-Based Agent, Imitation Learning | `PENDING INDEPENDENT REVIEW` | — |
| **Stage 5 + GUI** | Nguyễn Hữu Trí | Model-Free RL, Replicate Training, Workbench GUI Extension | `PENDING INDEPENDENT REVIEW` | — |
| **Stages 2 / 6** | Lư Hồng Phúc | Planning / Search, Search + Learned Baselines, Discrete Controls | `PENDING INDEPENDENT REVIEW` | — |
| **Stages 3 / 7** | Đỗ Duy Quý (`doduyquy`) | Simulation-Based Planning, World Models / Model-Based RL | `PENDING INDEPENDENT REVIEW` | — |

*Policy Reminder: Merging this documentation pull request into `main` creates a traceable working baseline; it does NOT constitute final scientific sign-off by team members or course instructors.*

---

## 2. Inventory of Open Scientific Decisions

---

### Decision 01: Factorial Structure for Stochastic Evaluation of Learned Policies

- **Context:** `configs/platform/seed_plan_v1.json` defines 3 training root seeds (`101, 202, 303`) and 3 agent replicate seeds (`101, 202, 303`). In Stage 5 Model-Free RL, policy networks can be evaluated either deterministically (e.g., taking the mean/argmax action $\mu(s)$) or stochastically (sampling $a \sim \pi(\cdot|s)$).
- **The Dilemma:**
  - If evaluated deterministically: Each of the 3 trained models evaluates the 60 TEST cases once ($3 \times 60 = 180$ episodes total).
  - If evaluated stochastically: Does each model evaluate the TEST suite once under a fixed inference seed ($3 \times 1 \times 60 = 180$ episodes), or does each model evaluate across all 3 agent seeds ($3 \times 3 \times 60 = 540$ episodes)?
- **Options:**
  - **Option A (3x1 Fixed Evaluation):** Standard practice in deep RL benchmarks (e.g., SB3, CleanRL). Train 3 models; evaluate each deterministically (argmax action) on the 60 TEST cases. Total: 180 episodes.
  - **Option B (3x3 Factorial Design):** Train 3 models; evaluate each stochastically with 3 seeds (`101, 202, 303`). Total: 540 episodes. Requires nested variance decomposition (training variance vs action sampling variance).
- **Recommendation:** **Option A (Recommended)** for primary benchmark reporting, reserving stochastic evaluation for exploration ablations.
- **Proposed Owner:** Nguyễn Hữu Trí (Stage 5) [PROPOSED / AWAITING TEAM AGREEMENT]
- **Target Resolution:** Gate C2 / Prior to Stage 5 RL training.

---

### Decision 02: Inferential Statistics Policy for $R=3$ Training Replicates

- **Context:** In research environments with bounded compute budgets, training 20 or 30 independent RL models per algorithm is prohibitive. Platform V1 establishes $R=3$ independent training models as the required baseline.
- **The Dilemma:** Standard parametric null-hypothesis significance testing (e.g., Student's two-sample t-test) has virtually no statistical power at $N_1=3, N_2=3$, and normality assumptions cannot be verified. Conversely, naive pooling of individual case episodes ($N=180$) violates sample independence.
- **Options:**
  - **Option A (Descriptive Dominance):** Prohibit formal $p$-value significance claims for $R=3$. Report mean $\pm$ sample std (in percentage points), individual replicate scores, and per-case paired differences descriptively.
  - **Option B (Paired Permutation / Randomization Test):** Permute paired case outcomes across models, blocking by training run identity.
  - **Option C (Stratified Bootstrap with Model-Level Resampling):** Resample models with replacement, then resample cases within models. (Requires acknowledging wide confidence intervals with $R=3$).
- **Recommendation:** **Option A (Recommended)** as the primary reporting standard, augmented with Option B for exploratory paired differences.
- **Proposed Owner:** Nguyễn Hữu Trí (Stage 5) & Lư Hồng Phúc (Stages 2/6) [PROPOSED / AWAITING TEAM AGREEMENT]
- **Target Resolution:** Prior to final benchmark report synthesis.

---

### Decision 03: Stage 5 Training Logging Schema and Checkpoint Digest Registry

- **Context:** Platform V1 (`LoggingContractV1`) strictly standardizes *evaluation* logging (`episodes.csv`, `timing.csv`, `summary.json`). It does not specify the format for intermediate training loss curves, gradient update steps, or policy weight serialization.
- **The Dilemma:** If Stage 5 training scripts invent disparate logging formats, cross-method training efficiency comparisons (e.g., sample efficiency curves) will be ad-hoc and unverified.
- **Options:**
  - **Option A:** Formalize an `ADR-Stage5-Training-Contract` introducing `training_manifest.json` and `training_metrics.jsonl` matching Gate 7 conventions.
  - **Option B:** Allow training to log freely to W&B, requiring only that the final selected model checkpoint is packaged with its SHA-256 digest before evaluation.
- **Recommendation:** **Option A (Recommended)** to preserve local-first cryptographic reproducibility.
- **Proposed Owner:** Nguyễn Hữu Trí (Stage 5) & Đỗ Duy Quý (Stages 3/7) [PROPOSED / AWAITING TEAM AGREEMENT]
- **Target Resolution:** Gate Stage 5 Kickoff.

---

### Decision 04: Benchmark Track Partitioning by Action Adapter

- **Context:** Platform V1 provides three certified action adapters (`continuous_box2_v1`, `discrete25_native_v1`, `discrete9_lowbranch_v1`). Stage 1 (rule-based) or Stage 2 (search) may operate natively over discrete grids, while Stage 5 (RL) typically optimizes continuous torques.
- **The Dilemma:** Should discrete and continuous agents compete in a single unified benchmark ranking table, or should Platform V1 establish official separate tracks?
- **Options:**
  - **Option A (Unified System Track with Mandatory Qualification):** A single benchmark table where discrete agents are marked with a mandatory `[QUALIFIED: discrete25_native_v1]` tag.
  - **Option B (Partitioned Benchmark Tracks):** Two official tracks: Track 1 (Continuous Actuation) and Track 2 (Discrete Actuation), with cross-track comparison permitted only as an end-to-end ablation.
- **Recommendation:** **Option B (Recommended)** for clarity and fairness.
- **Proposed Owner:** Đặng Khoa (Stages 0/1/4) & Nguyễn Hữu Trí (Stage 5) [PROPOSED / AWAITING TEAM AGREEMENT]
- **Target Resolution:** Prior to Stage 1 vs Stage 5 joint comparison.

---

### Decision 05: Downstream GUI V2 Comparison Module Architecture

- **Context:** The current Research Workbench (Gate 7.5B Pass B2) allows inspecting individual runs. It intentionally contains no multi-run comparison or leaderboard widgets.
- **The Dilemma:** How should GUI V2 implement comparative analysis without violating the read-only, non-mutating integrity of local run artifacts?
- **Options:**
  - **Option A (Pure In-Memory Presentation):** GUI V2 adds a 7th tab (`ComparisonWidget`) that loads two or more verified snapshots via `RunArtifactRepository`, validates Layers A and B in-memory, and displays side-by-side scorecards without writing any new files.
  - **Option B (Persisted Comparison Sessions):** GUI V2 allows exporting comparison session definitions to a `comparisons/` directory with its own schema.
- **Recommendation:** **Option A (Recommended)** to prevent file proliferation and maintain simple read-only semantics.
- **Proposed Owner:** Nguyễn Hữu Trí (Stage 5 + GUI) & Đỗ Duy Quý (Stages 3/7) [PROPOSED / AWAITING TEAM AGREEMENT]
- **Target Resolution:** GUI V2 Design Gate.

---

## 3. Team Action Items

1. **Review and Comment:** Each owner must review their designated section in `COMPARISON_SPEC_V1.md` and `COMPARABILITY_MATRIX_V1.md`.
2. **Formalize ADRs:** When an open decision is resolved, convert it into an Architectural Decision Record (ADR) under `docs/adr/`.
3. **No Premature Promotion:** Maintain document status as `DRAFT / TEAM REVIEW REQUIRED` until all 4 signatures are recorded.
