# Stage 5 P2-E — Critic Learning Signal & Action Support

**Decision: `DATA_SUPPORT_INTERVENTION` (priority for one controlled follow-up, not a confirmed root cause).**

This read-only audit used frozen P2-D checkpoints and TRAIN replay. It performed no SAC training, simulator rollout, VALIDATION/TEST access, reward change, or benchmark promotion. Stage 3 was not used as a teacher or data source.

## Scope and provenance

- Parent/source revision: `c322cb1f220d7491356a9566c3add39d49054ab9` (P2-D branch HEAD at audit start).
- P2-D protocol SHA-256: `92f6180d3217f8133d214d929fc1315eb1c24cf538b4acbad8ddd92c060bacb8`.
- F09 producer: `reference_route_station_proposal_v1`; observation transform: `pilot_groupwise_physical_v1`; AgentInputV1: 409D; action: `continuous_box2_v1`.
- Fixed anchor matrix: 201 real TRAIN observations from P2-C, SHA-256 `6ded06b27deec488dac1952b8a4a7d22755b8bbbca0d774854d53fe58ac6f13a`.
- P2-C evidence SHA-256: `489aa942d39806f5dfc33c48c6b0fcbaa7ea23dab7e687590df387be78dd00d1`.
- P2-D summary SHA-256: `d12f0e913cd8e4dd442e1caae8d6e41d1922095c4b752304573ad7728b19100b`.
- P2-D 10k model SHA-256: A `46874728028afeacc823392f934ffb95e4a6defb28efa1a765090d68ba728464`; B `7e29ec40f2269141c4fb57347bc20b4cbea7376e18f776920ce8c0d2a91581db`.
- P2-D replay SHA-256: A `b96d7799f585f128132b57c6afe61e3310e6d9cb0b0ed44fe0fb817639ef44c4`; B `8452571f0e8087bd3f3ce91eec45feba30f33da775b75df667960e8b5ec4c6ed`.
- Machine evidence: [`evidence.json`](../../results/stage5/p2e_critic_learning_signal_20261121/evidence.json), SHA-256 `1c555c2bc17715252aa1715ab37a52cd7b621a0b7c7eae6ada01a7ba509df352`.

The audit reloaded checkpoints through the project integrity loader. The 2k and 5k comparisons use the preserved 1,024-transition replay samples; only 10k has a complete replay buffer. The same 201 P2-C anchors were used at every checkpoint. Stochastic actor-gradient estimates use fixed torch seeds for repeatable comparisons. Critic target/loss calculations were checked against the installed Stable-Baselines3 2.9.2 `SAC.train()` and `ReplayBuffer` semantics; time-limit truncations bootstrap and true terminations do not.

## Findings

### Replay and action support

At 10k, the complete replay buffers aligned with their recorded action traces:

| Arm | Transitions | Action mismatches | Reward mismatches | Done mismatches | Timeout mismatches | Training episode outcomes |
|---|---:|---:|---:|---:|---:|---|
| A — ReLU actor | 10,000 | 0 | 0 | 0 | 0 | 6 TIMEOUT, 2 OUT_OF_ROAD |
| B — LeakyReLU actor | 10,000 | 0 | 0 | 0 | 0 | 7 TIMEOUT |

The replay was dominated by low speed. A had 8,320/10,000 transitions below 1 km/h; B had 8,313/10,000. Only 1,680 A and 1,687 B transitions were at or above 1 km/h, with mean speed about 1.68 km/h. Positive route-progress deltas occurred on 5,895 A and 5,936 B rows, but the mean delta among those rows was only about `4.0e-5` per decision. Task waypoint turn context (curvature score ≥0.05 rad) appeared on 4,351 A and 3,945 B transitions, mostly while stationary or slow. This is state coverage, not evidence of competent cornering.

The stochastic replay action distribution was broad, with throttle/brake sign changes on about half the transitions. On sampled replay states, the deterministic actor action was inside the axis-aligned range of its 20 nearest same-geometry replay-state action samples in all queries. At 10k, median nearest action distance was about 0.237 (A) and 0.239 (B), about 0.43 and 0.44 of local action RMS standard deviation. Moving, positive-progress, stationary, and turn-context strata had similar medians; safety-terminal queries were available only for A (two rows). This kNN summary suggests no gross *single-step marginal* action-support gap. It is not a density estimate, and it says nothing conclusive about support for a long deterministic action sequence.

### Actor objective and critic signal

For each checkpoint, the entropy and negative-Q actor-loss gradients were computed on identical frozen anchor states, separated by actor layer, with L2 norms and cosine direction. The negative-Q gradient was much smaller than the entropy gradient in the early checkpoints: roughly 1–3% in actor mean/latent layers at 2k and 5k. By 10k the ratio rose to about 10–14% in later actor layers, but remained much smaller for log-standard-deviation. A's first-layer gradient was zero at 5k/10k due to inactive ReLUs. B retained a nonzero first-layer path, but its 10k ratio was about 11%; this did not preserve practical action sensitivity.

The gradient directions also changed over time: the entropy and negative-Q terms were often opposed at 2k/5k, while B's sampled 10k terms were more aligned. These are offline, one-seed gradient snapshots; they do not establish which term caused the learned policy trajectory.

Critic action gradients at deterministic actions were small (approximately `1e-6`–`2e-5` per action dimension across checkpoints). On the same sampled states at 10k, the mean minimum-Q difference between deterministic and recorded behavior actions was near zero: A `−4.9e-6` (SD `0.00140`), B `+4.6e-5` (SD `0.00134`), with positive/negative signs close to 50/50. Q therefore showed little measured preference between these two action sets on these samples. This does not show that the critic ranking is wrong: the recorded set contains only one behavior action per state, and Q is not a measured return.

At 10k, sampled nonterminal replay strata had small self-consistency TD residuals (A roughly 0.05–0.12; B roughly 0.008–0.018 absolute, depending on stratum). A's two safety-terminal transitions had target near `−1` while both critics predicted about `+14.7`, producing absolute TD errors around `15.7`. There were only two such terminal rows in A and none in B, so the failure signal was exceptionally sparse. A small TD residual on the other samples is not critic calibration evidence; these targets bootstrap from the same learned target networks.

### Stochastic collection versus deterministic evaluation

The 10k deterministic actor was nearly constant on replay states: A's steering/throttle standard deviations were effectively zero; B's were about `1.4e-5` and `2.9e-5`. Stochastic action samples on those states still had standard deviations around 0.6 in both action dimensions. Accordingly, B's seven stochastic training episodes all timed out, while its frozen deterministic full-episode evaluation went out of road on all six diagnostic cases. A's stochastic training produced two out-of-road episodes, while deterministic evaluation timed out stationary on all six cases.

These outcomes are not contradictory: stochastic collection and deterministic evaluation are different policies. The logged behavior supports a train/evaluation trajectory-distribution gap. It does not prove that the deterministic mean action itself is unsupported at a single state, or that the critic alone caused the gap.

## Evidence classification

### CONFIRMED

- P2-D's 10k replay action, reward, done and timeout records match their training telemetry exactly in both arms.
- Replay contains broad stochastic action variation, but 83% of states are below 1 km/h and no arm has a clean training episode.
- Training terminal coverage is sparse: A has two safety failures; B has none. B's deterministic evaluation has six safety failures despite seven stochastic training timeouts.
- The actor's deterministic outputs become almost constant while stochastic action samples remain broad.
- On the sampled states, the critic's deterministic-versus-recorded-action Q gap is close to zero and changes sign across states; local Q action derivatives are small.

### SUPPORTED_HYPOTHESIS

- The dominant missing support may be *trajectory/outcome support*: replay has little sustained useful motion and no clean successful trajectory, and B has no safety terminal experience despite deterministic safety failures at evaluation.
- Weak action-discriminating Q gradients may be a mediator or a consequence of this experience distribution. P2-E cannot distinguish those explanations.
- Including data collected under the deterministic actor mode may make critic training and evaluation occupancy more consistent; this needs a controlled test.

### UNKNOWN

- Whether the critic would rank actions correctly given repeated, matched outcomes from the same state.
- Whether the deterministic action sequence is outside the replay's temporal/trajectory support.
- Whether the observed low Q action sensitivity is caused by critic learning, reward scale, sparse meaningful outcomes, actor collapse, or their interaction.
- Whether deterministic-policy data mixed into SAC replay would improve safe driving rather than merely reproduce stationary or unsafe behavior.
- Generalization beyond this one training seed and six diagnostic cases.

## Priority and bounded next experiment proposal

`DATA_SUPPORT_INTERVENTION` is the next direction to test because the clearest empirical deficit is the absence of clean trajectories and the mismatch between stochastic training outcomes and deterministic evaluation outcomes. This is a priority hypothesis, not a root-cause claim. The marginal action-kNN check did not identify a clear one-step action-support hole.

If separately authorized, run one fresh two-arm experiment with a fixed LeakyReLU-actor / ReLU-critic architecture in both arms, unchanged 409D input, F09 producer, Box2, RewardSpecV1, seed, geometry schedule and SAC hyperparameters. Change only the data-collection action mode:

- **Control:** standard stochastic SAC collection throughout.
- **Treatment:** standard stochastic collection except one preregistered slot in five uses a deterministic actor snapshot captured at that episode's start; all resulting transitions are ordinary environment experience and enter replay with the exact executed action. Keep the collection mode fixed through that episode, while the trainable SAC model continues its normal updates from replay.

Limit to 10k steps per arm for the first comparison. Track actual steps and episode outcomes by mode, replay action fidelity, terminal TD errors, actor sensitivity, and full deterministic episodes on the same six locked TRAIN diagnostic cases. Evaluate skill by clean successes across distinct cases, safety outcomes, sustained movement and route progress together. A speed or route-progress increase with safety failures is not improvement. If the treatment produces only more failure/stationary transitions or does not retain actor sensitivity, stop; do not extend the budget. This proposal is not started or preregistered by P2-E.

## Artifacts and reproduction

- Detailed machine evidence: [`evidence.json`](../../results/stage5/p2e_critic_learning_signal_20261121/evidence.json).
- Evidence checksum sidecar: [`evidence.sha256`](../../results/stage5/p2e_critic_learning_signal_20261121/evidence.sha256).
- Audit runner: [`audit_stage5_p2e_critic_learning_signal.py`](../../scripts/audit_stage5_p2e_critic_learning_signal.py).
- Large P2-D replay/checkpoints remain local under the ignored `runs/stage5_p2d/` tree; they are not copied into Git. Their SHA-256 identities and replay/counterfactual limitations are in the evidence.

Reproduce from `autonomous-driving-rl/` in the `metadrive_sb3` environment:

```powershell
conda run -n metadrive_sb3 python scripts/audit_stage5_p2e_critic_learning_signal.py
```

The audit is read-only with respect to the checkpoints and replay buffers. It overwrites only the P2-E evidence output directory.
