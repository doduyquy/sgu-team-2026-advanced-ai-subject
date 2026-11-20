# Stage 5 P2-D — Actor-Only Activation Ablation

**Status:** preregistered development experiment; TRAIN only.

**Machine protocol:** [`protocol_v3.json`](../../results/stage5/p2d_actor_activation_seed101_20261121/protocol_v3.json).

**Protocol SHA-256:** `92f6180d3217f8133d214d929fc1315eb1c24cf538b4acbad8ddd92c060bacb8`.

## Question

Does replacing only the two actor hidden-layer ReLUs with LeakyReLU(0.01) preserve actor gradient flow and observation sensitivity through 5k–10k SAC updates, and does that result in better driving?

The P2-C evidence found the ReLU actor's first layer fully inactive on fixed 201 TRAIN observations and 10k replay observations by 5k. It did not identify why the optimization entered that state or establish that activation was the only limitation.

## Registered arms

| Arm | Actor hidden activations | Twin critics and target critics |
|---|---|---|
| A | ReLU | ReLU |
| B | LeakyReLU, slope 0.01 | ReLU |

The SB3 policy-wide activation remains ReLU. A versioned custom `SACPolicy` overrides only actor construction for B. Both arms use the same parameter names, shapes, initial tensor values, entropy state, 409D observation, `pilot_groupwise_physical_v1`, corrected `reference_route_station_proposal_v1` producer, Box2 action, RewardSpecV1 and SAC settings. As the nonlinearity differs, checkpoint-0 action outputs need not match.

The sole registered treatment is actor activation. Both arms use standard SAC action collection, eight fixed Easy `PILOT_TRAIN` geometries and six disjoint fixed Easy `PILOT_DIAGNOSTIC` geometries, root seed 101 and 10,000 actual environment steps maximum. Checkpoints are 0, 2k, 5k and 10k. Evaluation runs in separate environments, deterministically, with 300-decision probes at every checkpoint and full episodes at 10k.

The fixed 201-observation matrix is the P2-C artifact with SHA-256 `6ded06b27deec488dac1952b8a4a7d22755b8bbbca0d774854d53fe58ac6f13a`. Replay-state samples at checkpoints are saved separately; the complete replay buffer is saved at 10k. LeakyReLU diagnostics report its negative-side derivative (0.01), so nonpositive preactivation is not mislabeled as a dead unit.

## Relation to S5.3C

S5.3C used the same nominal LeakyReLU slope but passed it as SB3's policy-wide `activation_fn`, changing both actor and critic. It also used `pilot_waypoint_xy_div50m_v1` rather than the P2 group-wise physical transform. Its 10k deterministic evaluation reported 24/24 timeouts, 100% stationary fraction, zero mean throttle and zero clean successes. That result is adverse historical evidence, but it does not isolate actor-only activation under the corrected P2 input protocol.

## Evaluation and decision limits

Driving outcomes, not gradient recovery alone, determine whether there is useful improvement. `PROBE_CENSORED` remains distinct from timeout and safety. A faster vehicle with increased safety failures is not a driving improvement. Six diagnostic cases and one training root seed cannot establish generalization or superiority; `benchmark_eligible` remains false.

Run after the preregistration commit:

```powershell
conda run -n metadrive_sb3 python scripts/run_stage5_p2d_actor_activation.py --arm ALL
conda run -n metadrive_sb3 python scripts/evaluate_stage5_p2d_actor_activation.py --arm BOTH
```

Do not run VALIDATION/TEST, extend beyond 10k, or treat this ablation as a confirmed SAC upgrade.
