# Stage 5 P2-G — Action Consequence Protocol

## Registered scope

P2-G is a bounded, frozen-checkpoint, TRAIN-only physical consequence audit. It performs no SAC optimizer updates and does not add any transition to a replay buffer. It does not access VALIDATION/TEST or change RewardSpecV1, gamma, alpha, action/observation contracts, or benchmark eligibility.

- Protocol ID: `stage5_p2g_action_consequence_v1`
- Protocol JSON: [`protocol.json`](../../results/stage5/p2g_action_consequence_20261121/protocol.json)
- Canonical protocol SHA-256 (sorted compact JSON): `52aeb7cf732a366fa240ea2038c2ed3cc7721b11fb3f541e4f22219d7adb8f8e`
- Source: P2-D arm A, training episode 6, `geom_easy_SCSS_seed3`, environment seed `976051`, recorded terminal outcome `OUT_OF_ROAD`.
- Source trace SHA-256 and checkpoint identities are recorded in [`evidence.json`](../../results/stage5/p2g_action_consequence_20261121/evidence.json).

## Fixed state/action design

The state selector replayed the full P2-D recorded action prefix. It selected at most six distinct decision states: reset; first state at or above 1 km/h with positive progress; first state at least 0.01 route fraction past reset; largest valid Task waypoint heading context; largest preterminal lateral error; smallest positive preterminal reference-lane clearance estimate. Selectors that chose the same decision were merged. Five unique states resulted.

The action grid was the preregistered Cartesian product:

- Steering: `-0.35`, `0`, `+0.35`.
- Throttle/brake: `-0.5`, `0`, `+0.5`.
- Each constant action was held for at most 20 decisions (2 seconds at 10 Hz), stopping on terminal outcome.
- Five states × nine actions = 45 interventions, at most 900 post-branch decisions.

Every branch restarted the same TRAIN case and replayed the exact recorded action prefix. A candidate was accepted only after three independent replays met physical tolerances: position range ≤0.02 m, heading ≤0.001 rad, speed ≤0.05 km/h, route completion range ≤1e-6, and transformed-observation range ≤1e-4. The amended observation tolerance reflects measured float32/runtime variation; physical-state tolerances were unchanged. The amendment occurred before the action grid ran and is documented in the protocol JSON.

## Measurements and interpretation boundaries

Each intervention records signed MetaDrive route-completion delta multiplied by the route's runtime `navigation.total_length`, RewardSpecV1 breakdown, speed, position, heading and reference-lane-relative lateral error. Clearance is an estimate based on lane width and vehicle width, not the simulator's safety classifier; `on_lane` and terminal flags are also recorded.

The 409D transformed observations use the same F09 producer and `pilot_groupwise_physical_v1` transform as the P2-D/P2-F checkpoint metadata. The action is passed directly through `continuous_box2_v1` semantics. No Stage 1/3 controller, teacher action, VALIDATION/TEST case, or training replay was used.

Critic Q1/Q2/min-Q and action gradients are compared over the same nine actions. Actor entropy and negative-Q objective gradients are measured separately at the selected observations. Replay support is described only by the axis-aligned action envelope among 20 nearest same-geometry replay observations in transformed 409D space. Neither this envelope nor short fixed-action returns are ground truth for long-horizon SAC soft-Q.

## Reproduction

From `autonomous-driving-rl/` in the `metadrive_sb3` environment:

```powershell
conda run -n metadrive_sb3 python scripts/run_stage5_p2g_action_consequence.py
conda run -n metadrive_sb3 python scripts/summarize_stage5_p2g_action_consequence.py
```

The runner is bounded to the registered source episode, five selected states, four existing 10k checkpoints, nine actions and a 20-decision post-branch horizon. It refuses to continue if historical trace replay or physical-state repeatability fails.
