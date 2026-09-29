# Platform V1 Agent Interface, AgentInput, and Action Adapter Contract Audit

> **Status:** AUDIT & SCIENTIFIC SPECIFICATION (GATE 6)  
> **Freeze Status:** LOCKED-FOR-PLATFORM-V1 (Agent Contract & Action Adapters Certified)  
> **Target Scope:** Universal runtime contract supporting Stages 0 through 7  
> **Date:** September 2026  

---

## 1. Executive Summary & Objective

Gate 6 establishes the **canonical runtime boundary** between Research Platform V1 and every agent family across all research stages:
- **Stage 0:** Random Baseline
- **Stage 1:** Rule / Heuristic
- **Stage 2:** Planning / Search
- **Stage 3:** Simulation-Based Planning (MCTS)
- **Stage 4:** Learning from Data / Imitation Learning
- **Stage 5:** Model-Free Reinforcement Learning (PPO / SAC)
- **Stage 6:** Search + Learned Policy / Value
- **Stage 7:** Model-Based RL (World Models)

### Core Scientific Mandates:
1. **Information Parity:** Every benchmark-compliant agent receives the **exact same information rights** via `AgentInputV1` under the primary information profile (`STATE_DECISION_V1`). Heuristic planners do not receive secret simulator handles, and neural policies are not arbitrarily starved of planning context.
2. **Defensive Runtime Isolation:** `AgentInputV1` and `AgentPublicEpisodeContext` strictly contain copied, validated, serializable primitives and read-only NumPy arrays. Live simulator objects (`MetaDriveEnv`, `engine`, `vehicle`, `navigation`, `traffic_manager`, Panda3D handles) are 100% prohibited and unreachable.
3. **Private Telemetry Segregation:** Evaluator-private signals (`route_completion`, arrival flags, collision flags, rewards, returns, benchmark case identity, tier, split, environment seed) are strictly segregated from agent observation.
4. **Canonical Actuator Contract:** All environment actuation maps deterministically to a continuous `Box(-1.0, 1.0, shape=(2,))` physical actuator (`[steering, throttle_brake]`) at nominal 10 Hz.
5. **No Silent Action Clipping:** Invalid, non-finite, or out-of-bounds agent actions fail loudly as a `TechnicalFailureReason.INVALID_AGENT_ACTION` rather than being silently clipped.
6. **Additive Cryptographic Locking:** Gate-5 benchmark hash (`9ddd889b...`) remains untouched; Gate 6 defines an additive platform runtime contract hash (`platform_runtime_contract_sha256`).

---

## 2. Source Contracts from Gates 1–5

Gate 6 operationalizes the architectural foundations established in previous platform gates:
- **Gate 1:** Audited the root MetaDrive observation (259D: 9 ego + 10 navigation + 240 LiDAR) and continuous actuator (`Box(2)`). Formulated the non-frozen Candidate D structured architecture.
- **Gate 2 (`MapSuiteV1`):** 240 candidate geometries across 12 sequence families and 4 difficulty tiers.
- **Gate 3 (`EpisodeSpecV1`):** Strict termination semantics, safety-first outcome precedence, and route-aware episode horizons.
- **Gate 4 (`RewardSpecV1` & `EvaluationMetricsV1`):** Decoupled reward training signal from the evaluation scorecard.
- **Gate 5 (`EvaluationProtocolV1`):** Stratified scenario splits (180 Train / 48 Validation / 12 Test), 5-channel seed taxonomy, and locked paired test manifest (`test_manifest_sha256 = 0832c38e...`).

---

## 3. The Information Parity Principle

A foundational flaw in many autonomous driving benchmarks is **information asymmetry**: planners are given global route graphs and ground-truth oracle trajectories, while RL agents are restricted to flat sensor vectors. This invalidates cross-paradigm comparisons.

Platform V1 enforces:
$$\text{InformationRights}(\text{Stage } 0) = \text{InformationRights}(\text{Stage } 1) = \dots = \text{InformationRights}(\text{Stage } 7)$$

Under the primary benchmark profile `STATE_DECISION_V1`, all algorithms receive identical `AgentInputV1`:
```text
AgentInputV1
├── CoreObservationV1 (259D defensive read-only float32 vector)
├── TrafficContextV1   (Structured state-based local traffic actors, N=8, R=50m)
└── TaskContextV1      (Ego-relative route lookahead waypoints, K=20, spacing=2.5m)
```
An algorithm may choose to ignore fields it does not use (e.g. an RL policy may only consume `CoreObservationV1`), but the platform provides identical information to all competitors.

---

## 4. Selected Information Profiles

The platform supports two explicit, machine-readable information profiles:
1. **`STATE_DECISION_V1` (Primary Benchmark Profile):**
   - The full structured state input for autonomous driving decision making.
   - Provides `CoreObservationV1` + `TrafficContextV1` + `TaskContextV1`.
   - Distinct from perception benchmarks: Platform V1 evaluates state-based driving intelligence, not computer vision perception under sensor degradation.
2. **`CORE_ONLY_V1` (Certified Ablation Profile):**
   - Provides `CoreObservationV1` with empty/masked `TrafficContextV1` and `TaskContextV1`.
   - Used specifically for ablation studies analyzing the marginal value of structured traffic and route lookahead.

---

## 5. CoreObservationV1 Specification

`CoreObservationV1` wraps the Gate-1 259-dimensional local observation vector with defensive copying and immutability guarantees:
- **Shape:** Exactly `(259,)`
- **Dtype:** `float32`
- **Bounds:** Normalized within `[0.0, 1.0]` under audited sensor configuration.
- **Subvectors:**
  - `ego_state` (indices 0..8, 9D): heading theta diff, steering, velocity x/y, gyro, angular velocity.
  - `navigation_vector` (indices 9..18, 10D): relative directional unit vectors to upcoming checkpoints.
  - `lidar_points` (indices 19..258, 240D): 360-degree range rays.
- **Defensive Immutability:** Exported array has `flags.writeable = False`. Any attempt by an agent to modify observation memory in-place raises `ValueError: assignment destination is read-only`.

---

## 6. TrafficContextV1: Source Semantics & Calibration

### Source Semantics:
`TrafficContextV1` extracts structured state information from the simulator's traffic manager.
- **Nature:** State-based ground truth within local radius (not sensor-realistic LiDAR bounding box detections; does not simulate occlusion).
- **Coordinate Frame:** Ego-centric local frame:
  - $+x$: Longitudinal forward along ego vehicle orientation.
  - $-x$: Longitudinal backward (behind ego).
  - $+y$: Lateral left.
  - $-y$: Lateral right.
- **Actor State (7D per actor):**
  1. `relative_position_x` (m)
  2. `relative_position_y` (m)
  3. `relative_velocity_x` (m/s)
  4. `relative_velocity_y` (m/s)
  5. `relative_heading` (rad, $[-\pi, \pi]$ relative to ego heading)
  6. `length` (m)
  7. `width` (m)

### Empirical Capacity Calibration (`traffic_context_capacity.csv`):
Calibrated across 16 representative TRAIN and VALIDATION scenarios loaded directly from `geometry_split_manifest.csv` under densities $0.0$ to $0.25$ and seeds $5101/5102$:
- **Split Governance:** Zero TEST geometries used for capacity calibration (asserted `split != "TEST"` and verified `geometry_hash_match == True` on every scenario).
- **Radius:** $50.0\text{ m}$ (matches Gate-1 LiDAR range limit).
- **Route Traversal Coverage:** Used pinned MetaDrive `IDMPolicy` to traverse scenarios up to 250 steps, actively triggering road blocks and traffic spawns.
- **Observed Concurrent Actors:** Max concurrent actors within $50\text{ m}$ across calibration suite reached exactly **8 actors** (observed on Extreme `CrXROSTR` seed 12, env 5102), with peak mean local actor count of $4.33$.
- **Fixed Capacity Selected:** $\mathbf{N = 8\text{ actors}}$ (accommodates the maximum observed active traffic concurrency).
- **Overflow Rate:** **`0.0%`** (zero overflow events across all 16 scenarios and 3700+ decision cycles).
- **Ordering:** Deterministically sorted by Euclidean distance ascending, with deterministic tie-breaking on $(x_{rel}, y_{rel})$.
- **Masking:** When fewer than 8 actors are nearby, remaining slots contain zeros with `validity_mask[i] = False`.

---

## 7. TaskContextV1: Read-Only Route Lookahead

`TaskContextV1` provides ego-relative geometric reference route information required by trajectory planners and search algorithms:

### Structural Calibration on VALIDATION Geometries (`task_context_calibration.csv`):
Evaluated three candidate lookahead configurations across all road topology families on VALIDATION splits:
1. **Candidate A ($10\text{ waypoints} \times 2.5\text{ m} = 25.0\text{ m}$ lookahead, 30 floats):** Insufficient lookahead horizon; fails to preview complete curve transitions and multi-lane intersection geometries.
2. **Candidate B ($20\text{ waypoints} \times 2.5\text{ m} = 50.0\text{ m}$ lookahead, 60 floats) — SELECTED:** Structurally optimal. Lookahead range ($50.0\text{ m}$) perfectly matches the sensor LiDAR range ($50.0\text{ m}$); $2.5\text{ m}$ spacing provides high geometric fidelity to trace sharp curves ($R \approx 20\text{ m}$) without representation bloat.
3. **Candidate C ($20\text{ waypoints} \times 5.0\text{ m} = 100.0\text{ m}$ lookahead, 60 floats):** Excessive lookahead extending beyond local visibility; coarse $5.0\text{ m}$ spacing blunts curve curvature and intersection turning points.

### Schema Specification:
- **Lookahead Waypoints:** Exactly $K = 20$ waypoints sampled at $\Delta s = 2.5\text{ m}$ spacing along the designated route centerline (total lookahead range: $50.0\text{ m}$).
- **Waypoint Fields:** `(lookahead_distance_m, relative_x, relative_y, relative_heading)`.
- **Lane Width:** $3.5\text{ m}$ (audited standard).
- **Speed Semantics:** Road speed limits are omitted from `TaskContextV1` because MapSuiteV1 does not define meaningful lane speed limits (MetaDrive default unconstrained placeholder is 1000). Platform V1 does not invent synthetic road speed targets.
- **Goal Direction:** Normalized 2D vector $(\cos \Delta\theta, \sin \Delta\theta)$ towards the active checkpoint.
- **Route End Flag:** `route_end_within_lookahead` indicates only that the designated reference route ends within the lookahead window; it does **not** indicate episode termination.
- **Topological Technical Validation (`task_context_validation.csv`):** Verified 100% finite, valid construction across all road topologies and all 12 canonical test geometries (technical compatibility verification only; zero agent performance tuning).

---

## 8. Absolute Information Boundaries: Forbidden Telemetry

To eliminate benchmark gaming and data leakage, `AgentInputV1` and `AgentPublicEpisodeContext` strictly exclude the following 34 evaluator-private fields:

| Forbidden Telemetry Category | Prohibited Fields |
|---|---|
| **Evaluator Progress Scalars** | `route_completion`, `max_route_completion`, `distance_along_route`, `percent_complete` |
| **Terminal Outcomes & Flags** | `arrive_dest`, `clean_success`, `crash_human`, `crash_vehicle`, `crash_object`, `crash_building`, `crash_sidewalk`, `out_of_road`, `terminated`, `truncated`, `terminal_reason`, `primary_terminal_reason` |
| **Training Feedback & Rewards**| `reward`, `reward_breakdown`, `episode_return`, `future_outcome` |
| **Benchmark Bookkeeping** | `split` (TRAIN/VAL/TEST), `tier`, `difficulty_tier`, `candidate_role`, `case_id`, `protocol_order_index`, `test_manifest_position` |
| **Seed & Geometry Secrets** | `environment_seed`, `env_seed`, `geometry_generation_seed`, `geometry_sha256` |
| **Oracle State** | `ground_truth_trajectory`, `test_score`, `case_result` |

*Input Leakage Audit (`input_leakage_audit.json`):* Automated recursive scanning verified **0 leaked fields** across all prohibited telemetry keys.

---

## 9. Runtime Object Isolation: Zero Live Simulator Handles

To guarantee complete sandboxing, `AgentInputV1` must never allow an agent to navigate python object references back into the simulator engine.

*Runtime Type Audit (`runtime_type_audit.json`):* Recursive object graph traversal of live `AgentInputV1` instances verified:
- **`metadrive` classes:** 0 reachable handles.
- **`panda3d` classes:** 0 reachable handles.
- **`direct` classes:** 0 reachable handles.
- **Engine / Vehicle handles:** 0 reachable handles.
- All exported data consists exclusively of portable Python primitives (`bool`, `int`, `float`, `str`, `tuple`) and read-only `np.ndarray` structures.

---

## 10. Agent Policy Lifecycle Contract

The platform defines a clean, composition-friendly runtime protocol (`AgentPolicy`):
```python
class AgentPolicy(Protocol):
    @property
    def descriptor(self) -> AgentDescriptor: ...
    def reset(self, public_context: AgentPublicEpisodeContext, agent_seed: Optional[int] = None) -> None: ...
    def act(self, agent_input: AgentInputV1) -> AgentDecision: ...
    def close(self) -> None: ...
```

### Key Lifecycle Principles:
1. **Public Context at Reset:** Contains only `control_frequency_hz` (10), `control_dt_s` (0.1), `horizon_steps`, `input_profile_id`, `action_adapter_id`, and `mode` (`"INFERENCE"` or `"TRAINING"`).
2. **Episodic Reset vs. Static Parameters:** `reset()` resets recurrent internal episodic memory (e.g. RNN hidden state, search tree cache). Static learned neural network parameters remain untouched.
3. **Diagnostics Validation (Option A):** `AgentDecision.diagnostics` is validated recursively to guarantee it contains strictly JSON-safe bounded primitives (`str`, `int`, `float`, `bool`, `None`, and bounded lists/dicts up to depth 4 and 50 keys). Live simulator, engine, or model handles are rejected immediately with `TypeError`.
4. **No Online Learning in Evaluation Mode:** During benchmark evaluation (`mode = "INFERENCE"`), `act()` receives zero reward or outcome feedback. Cross-episode online adaptation is strictly prohibited.
5. **Seed Isolation:** `agent_seed` seeds agent-side stochasticity (e.g. action sampling noise) and is strictly isolated from simulator `environment_seed`.

---

## 11. Canonical Actuator Contract & Certified Adapters

### Canonical Physical Actuator:
MetaDrive physics is actuated exclusively via continuous `Box(-1.0, 1.0, shape=(2,))`:
$$\mathbf{a} = [\text{steering},\ \text{throttle\_brake}], \quad \text{steering} \in [-1.0, 1.0], \quad \text{throttle\_brake} \in [-1.0, 1.0]$$
Operates at nominal 10 Hz ($dt = 0.10\text{ s}$, decision repeat = 5 over physics $dt = 0.02\text{ s}$).

### Certified Action Adapters (`action_adapter_mappings.csv`):
1. **`continuous_box2_v1` (Certified Continuous Identity):**
   - Maps continuous 2D actions directly to physical actuators.
   - Validates dimensions and bounds; non-finite or out-of-bounds inputs raise `InvalidActionError`.
2. **`discrete25_native_v1` (Certified Native Discrete 25):**
   - Implements the exact MetaDrive `EnvInputPolicy` grid ($5 \text{ steering} \times 5 \text{ throttle} = 25\text{ actions}$):
     $$\text{steering} = (a \pmod 5) \times 0.5 - 1.0 \in \{-1.0, -0.5, 0.0, 0.5, 1.0\}$$
     $$\text{throttle} = (a \mathbin{//} 5) \times 0.5 - 1.0 \in \{-1.0, -0.5, 0.0, 0.5, 1.0\}$$
3. **`discrete9_lowbranch_v1` (Certified Optional Low-Branching Discrete 9):**
   - Designed for tree search and MCTS planners with low branching factor:
     $$\text{steering} \in \{-0.6, 0.0, 0.6\}, \quad \text{throttle} \in \{-0.8, 0.0, 0.6\}$$
   - 9 actions: combinations of Left/Straight/Right and Brake/Coast/Throttle.

### Anti-Silent Clipping Mandate:
Any invalid action (NaN, Inf, wrong dimension, or out-of-bounds such as $[1.05, 0.0]$) raises `InvalidActionError` immediately. Silent clipping is strictly forbidden in benchmark mode.

### Simulator-Backed Execution Evidence (`action_adapter_execution.csv`):
Verified that 15 representative actions across all three certified adapters execute directly into MetaDrive continuous physics (`env.step()`) without clipping or error, outputting validated 259D observations.

---

## 12. Additive Cryptographic Hashes

To ensure mutation-proof scientific provenance without invalidating Gate-5 fingerprints:
- Gate-5 `benchmark_contract_sha256` remains locked and untouched: `9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77`.
- Gate-6 defines complete additive schema and runtime contract hashes:

```json
{
  "gate5_benchmark_contract_sha256": "9ddd889b84d8705fae618879e5035556c80d0276a3dc2a58a7963937ebb59f77",
  "agent_contract_sha256": "325428fa0dd3e114b7aaa5af1c85640c3c759bce037a0e8d7effc3260fa5fe43",
  "platform_runtime_contract_sha256": "3cbe3c66bacbda3e70c6b4b9daffe4edccf2d0e09b703bf9675dec66643d8e46"
}
```

$$\text{platform\_runtime\_contract\_sha256} = \text{SHA256}(\text{canonical\_json}(\{\text{gate5\_benchmark\_contract\_sha256}, \text{agent\_contract\_sha256}, \dots\}))$$

All discrete adapter coordinate grids (25 entries for Discrete25, 9 entries for Discrete9), public context schemas, descriptor definitions, diagnostics limits, and latency boundaries are fingerprinted inside `agent_contract_sha256`. Mutating any mapping or rule immediately alters the fingerprint.

---

## 13. Unit Tests & Verification Summary

Implemented in `tests/test_agent_contract.py` (37 pure unit tests executing in $<0.04\text{ s}$ without Panda3D):
- Verified `CoreObservationV1` shape `(259,)`, float32 dtype, subvectors, and normalized `[0.0, 1.0]` bounds enforcement (-0.01 and 1.01 rejected).
- Verified non-finite rejection (NaN and Inf).
- Verified defensive immutability (mutating exported array raises `ValueError`).
- Verified `TrafficContextV1` empty state, actor sorting, active count, array consistency, and validity masking.
- Verified `TaskContextV1` lookahead waypoints, lane width, goal direction, array consistency, and route end flag.
- Verified absence of all 34 forbidden evaluator fields in `AgentInputV1` and `AgentPublicEpisodeContext`.
- Verified `AgentPublicEpisodeContext` validation (rejects non-positive frequency, dt, horizon, or invalid mode).
- Verified `ContinuousBox2Adapter` edge cases and out-of-bounds rejection without silent clipping.
- Verified `Discrete25Adapter` all 25 index mappings and invalid index rejection.
- Verified `Discrete9Adapter` all 9 index mappings and invalid index rejection.
- Verified deterministic fixture repeatability.
- Verified stochastic fixture repeatability under same `agent_seed` and variation under differing `agent_seed`.
- Verified stateful fixture episodic reset preserves static learned weights.
- Verified `AgentDecision` JSON-safe bounded diagnostics validation and deepcopy isolation in `to_dict()`.
- Verified invalid output technical failure handling (NaN, out-of-bounds, exceptions).
- Verified runtime dataclass schema introspection alignment across all 9 public classes.
- Verified runtime value and type consistency across contract schemas.
- Verified contract hash mutation sensitivity:
  - Discrete9 coordinate mutation alters hash.
  - Discrete25 coordinate mutation alters hash.
  - Public AgentInput schema mutation alters hash.
  - Top-level runtime dataclass schema field mutation alters hash.
  - Forbidden evaluator field list mutation alters hash.
  - Evaluation rule mutation alters hash.
  - Latency measurement boundary mutation alters hash.
  - Whitespace/indentation JSON formatting preserves identical hash.

---

## 14. Regression & Integrity Validation
- Pure unit tests: **83 total tests pass across repo** (`test_episode_lifecycle`: 12, `test_reward_metrics`: 14, `test_evaluation_protocol`: 20, `test_agent_contract`: 37).
- `CourseEnvV1`: Reset observation shape `(35,)` verified.
- `evaluate_random.py`: Completed 20 evaluation episodes cleanly.
- Gates 1 through 5 contracts remained completely untouched.
- Zero hardcoded drive letters in executable code.

STOP. PR remains open and unmerged.
