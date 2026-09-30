# Gate 7 Experiment Logging, Provenance, and W&B Integration Summary

## 1. Verified MetaDrive Source & Software Versions
- **Pinned MetaDrive Commit:** `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- **Package Version:** `0.4.3`
- **W&B SDK Version:** `0.30.0`
- **Working Tree:** Verified clean via `git status --porcelain`.

## 2. Local-First Scientific Record Architecture
- **Authoritative Source of Truth:** Local raw run directory (`runs/<run_id>/`).
- **Downstream Mirror:** Weights & Biases serves strictly as remote index, visualization, and comparison.
- **Required Local Files:** `run_manifest.json`, `episodes.csv`, `summary.json`, `timing.csv`, `run_integrity.json`, `wandb_sync.json`.
- **Duplicate Protection:** Fails loudly if `runs/<run_id>` already exists; overwriting or appending to completed scientific runs is strictly forbidden.
- **Atomic Writes:** JSON records use `.tmp` flush, fsync, and atomic rename.

## 3. Provenance & Privacy
- **Git Provenance:** Captured `git_commit_sha`, `git_branch`, and `git_worktree_dirty`. Benchmark evaluation on dirty worktrees requires prominent non-canonical tagging.
- **Ephemeral Secret Policy:** `WANDB_API_KEY` is read strictly from `os.environ`; zero credentials persisted in code, logs, or commits.
- **Machine Privacy:** Usernames, home directories, and full absolute machine paths are strictly excluded from persisted manifests.

## 4. Metric Parity & Observability Boundaries
- **Gate-4 Metrics Reused:** Primary scorecards (`clean_success_rate`, `safety_failure_rate`, `mean/median_final_route_completion`, `mean_time_to_clean_success_s`) reported overall, per-tier, and macro.
- **Diagnostic Return:** `episode_return` classified strictly as `diagnostic/episode_return`, never as primary ranking score.
- **Timing Boundary:** Latency timer wraps `agent.act()` strictly; excludes downstream logger and W&B network I/O.
- **Observational Invariance:** Verified 100% bit-for-bit identical simulator trajectories between `DISABLED` and `OFFLINE` logging modes.

## 5. Weights & Biases Online Smoke Result
- **Performed:** `True`
- **Status:** `SYNCED`
- **Run ID:** `audit_online_aaff2e38`
- **Run URL:** `https://wandb.ai/phucga15062005/sgu-autonomous-driving-rl/runs/audit_online_aaff2e38`
- **Project / Entity:** `sgu-autonomous-driving-rl / None`
- **Table & Summary Mirrored:** Verified

## 6. Additive Cryptographic Hashes
- **`gate5_benchmark_contract_sha256`:** `9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77` (locked, untouched)
- **`gate6_agent_contract_sha256`:** `53aa37079ff44afa75d9a3f921b0c1f98c4600d882fced51fc8d9197795058eb` (locked, untouched)
- **`platform_runtime_contract_sha256`:** `c7698768539a769c7b2bc6b90771ff234276a974b6e0353bbb03119faf2f79ad` (locked, untouched)
- **`logging_contract_sha256`:** `02d480cbb78e876b4c91a4eb16d831106de4199e369c9b53b50f91b142636f9f`
- **`platform_observability_contract_sha256`:** `335d96590dd9bbcaf2eb1077d497ebb155acf3abae9bf299bb47087a8dbb81d8`
