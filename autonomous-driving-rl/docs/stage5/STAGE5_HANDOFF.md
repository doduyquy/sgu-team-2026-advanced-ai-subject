# Stage 5 handoff

## What is ready

- Stage 5 Gymnasium environment, 409D observation encoder, TRAIN-only sampler and RewardSpecV1 integration from S5.1.
- Stable-Baselines3 SAC training, checkpoint integrity verification and `Stage5SacAgent` deterministic inference from S5.2.
- A fixed 156/24 internal TRAIN split and compact pilot/evaluation scripts.
- A fresh S5.3M A/B result with checkpoint SHA-256, fixed paired diagnostic records and a reviewable report.
- Research Workbench **7. SAC Training** tab: per-run progress, action/speed/route charts, overall/per-tier 0/5k/10k diagnostic table, and deterministic simulator viewer.
- Results → **Comparison** subtab: provenance-checked canonical track and a separate TRAIN-only S5.3M development track; matched per-case and per-tier differences; explicit censored/timeout distinction; JSON export.
- Stage 5 monitor shutdown behavior now stops its timer and disables queued UI callbacks only after MainWindow's existing process-confirmation/termination path succeeds.
- Prompt 2 regression passed: 157 Stage 5/Workbench tests; full repository suite 602 tests (601 passed, 1 skipped), exit code 0. Exact runtime and command are in [STAGE5_REPRODUCTION.md](STAGE5_REPRODUCTION.md).

See [STAGE5_COMPARISON_GUIDE.md](STAGE5_COMPARISON_GUIDE.md) for use and source trust rules. The comparison screen does not rank agents, repair artifacts, or make unpaired deltas.

## Available candidate checkpoints

S5.3M produced six verified checkpoint bundles locally:

- A (`tau=.005`): 0, 5k, 10k; 10k SHA-256 `d074977e99852c796e46c7fbc1a284c1a382265e6d501e9af2812b5b53be0442`.
- B (`tau=.001`): 0, 5k, 10k; 10k SHA-256 `a50d3c8dd755b81172be1a952e5bf1d91756842e9e84d86fd1fffce46ba1398c`.

Bundles are under the ignored local `runs/stage5_tau_ab/s5_3m_tau_ab_seed101_20261119_retry1/` directory. The model/checkpoint bytes are not in GitHub. B is a development candidate for confirmation; A remains the control reference. `stage5_sac` remains `CANDIDATE`, `benchmark_eligible=false`; neither checkpoint is ready for official scoring.

## Dependencies before official evaluation

1. Confirm the `tau=.001` signal with independent training roots under the same fixed protocol before deciding it should replace the reference.
2. Obtain a reviewed, versioned Stage 5 checkpoint-binding extension before canonical planning. Current `CheckpointBindingV1` is specialized to the learned A* workflow; reusing it for SAC would encode false semantics. See the [NO-GO proposal](STAGE5_CHECKPOINT_BINDING_PROPOSAL.md).
3. Confirm candidate B with independent training roots under a separately approved protocol before selecting it. This is future research, outside the completed bounded A/B sprint.
4. Keep VALIDATION for protocol-compliant selection. Lock candidate/config and checkpoint identity first; use TEST only for the final holdout evaluation.
5. The stale Workbench test assertion expecting 6 top-level tabs has been corrected to assert 7 and the `7. SAC Training` label. Existing Workbench/Launcher contract hashes were not changed.

Prompt 2 integration is complete. Any subsequent work should focus on reviewed checkpoint-binding governance and, if approved, final evaluation preparation. Do not repeat the completed A/B experiment or start another SAC sweep by default.

## Team boundary

Stage 5 may read other Stage implementations for integration and fair comparison. This handoff does not authorize changes to MCTS/Stage 2/6, Stage 1, Stage 4, frozen Platform V1 contracts, official MapSuite manifests, canonical evaluation metrics, or TEST policy.
