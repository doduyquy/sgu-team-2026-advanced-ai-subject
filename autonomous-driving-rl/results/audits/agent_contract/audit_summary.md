# Gate 6 Agent Interface, AgentInput, and Action Adapter Contract Summary

## 1. Verified MetaDrive Source
- **Pinned Commit:** `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- **Package Version:** `0.4.3`
- **Working Tree:** Verified clean via `git status --porcelain`.

## 2. Information Parity & AgentInput Architecture
- **Parity Mandate:** Same information rights across all agent stages (Stage 0 to Stage 7).
- **Primary Profile:** `STATE_DECISION_V1` (state-based decision making, not perception benchmark).
- **`CoreObservationV1`:** Exactly 259D float32 defensive array (`writeable=False`). Normalized range `[0.0, 1.0]` strictly enforced. Mutating exported array raises `ValueError`.
- **`TrafficContextV1`:** State-based structured local context with fixed capacity $N=8$ actors inside $50.0\text{ m}$ radius. Calibrated on TRAIN/VAL only (0 overflow events). Deterministically sorted by Euclidean distance.
- **`TaskContextV1`:** Read-only ego-relative lookahead waypoints (20 points at $2.5\text{ m}$ spacing up to $50.0\text{ m}$). Lane width $3.5\text{ m}$. Speed limit removed (no invented road speed limit). Evaluator private progress scalars (`route_completion`, arrival flags, returns) strictly excluded.

## 3. Security & Telemetry Segregation
- **Runtime Isolation:** Recursive object graph traversal verified 0 live handles to `metadrive`, `panda3d`, `direct`, engine, or vehicle objects.
- **Forbidden Field Scan:** Verified 0 leaked private evaluator fields across all 34 prohibited telemetry keys.
- **Test Metadata Segregation:** `tier`, `split`, `case_id`, `environment_seed`, `geometry_sha256` strictly excluded from AgentInput and PublicEpisodeContext.

## 4. Actuator Contract & Certified Action Adapters
- **Physical Actuator:** Continuous `Box(-1.0, 1.0, shape=(2,))` with `[steering, throttle_brake]` at nominal $10\text{ Hz}$.
- **Certified Adapters:**
  - `continuous_box2_v1`: Certified continuous identity adapter.
  - `discrete25_native_v1`: Certified native MetaDrive $5\times 5=25$ discrete grid (`EnvInputPolicy` audited).
  - `discrete9_lowbranch_v1`: Certified optional low-branching $3\times 3=9$ discrete grid for tree search / MCTS.
- **Anti-Silent Clipping:** Invalid, non-finite, out-of-range actions fail loudly with `InvalidActionError` (Technical Failure).
- **Simulator Technical Execution:** Verified representative canonical actions from all certified adapters execute cleanly in MetaDrive without clipping.

## 5. Additive Cryptographic Hashes
- **`gate5_benchmark_contract_sha256`:** `9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77` (locked, untouched)
- **`agent_contract_sha256`:** `02f9b3cfef041f5530b1e7b68a957aa96331b1b81ff4e1442e001c6f9863c03d`
- **`platform_runtime_contract_sha256`:** `da08c7e544d4df0296ac1db0e44a0bf9d564b0373c89b7bfa67907fa3d5b1b43`
