# Stage 5 handoff

## What is ready

- Stage 5 Gymnasium environment, 409D observation encoder, TRAIN-only sampler and RewardSpecV1 integration from S5.1.
- Stable-Baselines3 SAC training, checkpoint integrity verification and `Stage5SacAgent` deterministic inference from S5.2.
- A fixed 156/24 internal TRAIN split and compact pilot/evaluation scripts.
- A fresh S5.3M A/B result with checkpoint SHA-256, fixed paired diagnostic records and a reviewable report.
- Research Workbench **7. SAC Training** tab: per-run progress, action/speed/route charts, overall/per-tier 0/5k/10k diagnostic table, and deterministic simulator viewer.

Workbench is a useful live monitor and single-checkpoint viewer. It does not yet provide a provenance-bound side-by-side comparison of two agents or two training runs.

## Available candidate checkpoints

S5.3M produced six verified checkpoint bundles locally:

- A (`tau=.005`): 0, 5k, 10k; 10k SHA-256 `d074977e99852c796e46c7fbc1a284c1a382265e6d501e9af2812b5b53be0442`.
- B (`tau=.001`): 0, 5k, 10k; 10k SHA-256 `a50d3c8dd755b81172be1a952e5bf1d91756842e9e84d86fd1fffce46ba1398c`.

Bundles are under the ignored local `runs/stage5_tau_ab/s5_3m_tau_ab_seed101_20261119_retry1/` directory. The model/checkpoint bytes are not in GitHub. B is a development candidate for confirmation; A remains the control reference. `stage5_sac` remains `CANDIDATE`, `benchmark_eligible=false`; neither checkpoint is ready for official scoring.

## Gaps before final comparison and official evaluation

1. Confirm the `tau=.001` signal with independent training roots under the same fixed protocol before deciding it should replace the reference.
2. Design a Workbench comparison view that selects verified checkpoint identities, displays their SHA/config/provenance, and runs the exact same case/environment-seed schedule. Show per-case paired route progress, clean success, safety, timeout, censoring, speed and action behavior. Keep development diagnostic measures distinct from Platform V1 primary metrics.
3. Make checkpoint binding explicit in the frozen resolved experiment plan before canonical evaluation; the run manifest/checkpoint SHA alone is not benchmark binding.
4. Keep VALIDATION for protocol-compliant selection. Lock the candidate/config first; use TEST only for the final holdout evaluation.
5. Resolve the full-suite Workbench test mismatch with the GUI owner: the test expects 6 tabs and current GUI renders 7.

Prompt 2 can focus on the comparison interface and the preconditions for a protocol-correct final evaluation. It should not add another exploratory SAC hyperparameter sweep by default.

## Team boundary

Stage 5 may read other Stage implementations for integration and fair comparison. This handoff does not authorize changes to MCTS/Stage 2/6, Stage 1, Stage 4, frozen Platform V1 contracts, official MapSuite manifests, canonical evaluation metrics, or TEST policy.
