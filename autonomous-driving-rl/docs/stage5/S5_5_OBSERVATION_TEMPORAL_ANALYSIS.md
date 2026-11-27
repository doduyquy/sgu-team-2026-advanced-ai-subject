# S5.5 Observation and Temporal Analysis

## What the policy receives

The frozen Stage5 encoder maps public `AgentInputV1` fields to a 409D float32 vector (`stage5_flat409_v1`). The slices used in this experiment are:

| Slice | Dimensions | Information |
|---|---:|---|
| `[0:9]` | 9 | Ego state from the core observation: heading deviation, steering, velocity components and angular-motion signals in the Platform core vector |
| `[9:19]` | 10 | Navigation checkpoint direction features |
| `[19:259]` | 240 | LiDAR distance rays |
| `[259:315]` | 56 | Eight traffic actors, seven relative features each |
| `[315:323]` | 8 | Traffic actor validity masks |
| `[323:325]` | 2 | Traffic count/overflow metrics |
| `[325:385]` | 60 | Twenty ego-relative task waypoints, x/y/heading |
| `[385:405]` | 20 | Task waypoint validity masks |
| `[405:409]` | 4 | Lane width, goal direction x/y, route-end flag |

Source definitions are in `src/platform/agent_input.py` and `src/training/stage5/observation_encoder.py`. The encoder excludes evaluator-private route completion, reward, case identity, terminal outcome, future simulator state, and evaluation labels. The `pilot_groupwise_physical_v1` post-transform was fixed and identical across A/B/C; input size and information rights did not change.

## Directly measured feature and policy response

The analyzer checked eight observations sampled from a complete Stage 1 reference trajectory on the same TRAIN geometry. They were analysis-only states, not demonstration replay. All 409 values were finite. In those eight anchors, ego and navigation features varied; task waypoint coordinates ranged from `-19.299` to `50.0 m`, waypoint heading from `-0.8132` to `0 rad`, and the task mask reflected the shrinking lookahead near the route end.

In that Easy no-traffic trace, all 66 traffic features were zero, all eight traffic slots were invalid, and all 240 LiDAR rays were 1.0 (no obstacle return in this scenario). This checks shape and mask behavior, but it provides no evidence about traffic avoidance or obstacle response. Easy-only training likewise cannot establish those capabilities.

Across the eight reference anchors, the 30k deterministic action standard deviations were:

| Arm | Steering std | Throttle/brake std | First actor ReLU active fraction |
|---|---:|---:|---:|
| A — V1 | 0.000584 | 0.000516 | 0.0151 |
| B — Scale | 0.000552 | 0.000459 | 0.0156 |
| C — Metre | 0.001505 | 0.000298 | 0.0166 |

On the models' own valid probe-state anchors at 5k, 10k, and 30k, deterministic action standard deviation was exactly zero for every arm, and the measured first-layer ReLU active fraction was zero. At 30k, the small nonzero response on the eight external reference anchors shows that the measured policy was not mathematically constant over every possible input. It was nevertheless nearly constant on these samples and stationary in every full evaluation episode. The 10k Metre policy moved and failed unsafely on all six full cases; action variation alone did not imply competent steering.

The report also includes local input Jacobian magnitudes by feature group. These are local derivatives at sampled points, not causal feature importance. They cannot prove that Ego, Navigation, or Task information drove correct decisions. A proper use claim would require controlled input/state interventions tied to route tracking outcomes.

## Temporal information: what is and is not established

The SAC actor is an MLP and receives the current encoded observation on each 10 Hz decision. The current Ego vector includes instantaneous velocity/angular-motion fields; traffic actors, when present, include relative velocity; Navigation and Task Context provide current route geometry ahead. The policy input has no frame stack, recurrent hidden state, or explicit previous-action history buffer. The instantaneous steering value in the core vector is part of the current sensor/state representation, not a history sequence.

At 30k, training telemetry recorded 7,652–11,481 throttle/brake sign switches cumulatively, while the longest continuous run with speed at least 1 km/h was only 15–16 decisions (1.5–1.6 seconds). All three actors then produced stationary full episodes. This shows that their collected experience had very short moving periods and that the learned deterministic actions did not sustain movement. It does not show that the task fundamentally requires observation history: an MLP that has not learned to use the present state cannot establish that memory would fix the problem.

History/memory and temporally correlated exploration are different interventions. Memory changes the information available to the policy; correlated exploration changes how sampled training actions vary over time. The present evidence points first to a possible exploration/data-coverage issue because moving runs were short, but does not prove it. It does not justify Recurrent SAC, frame stacking, or a history wrapper in this sprint.

## Evidence levels

1. **Input correctness:** finite 409D vectors and expected masks/shapes were verified for sampled TRAIN states.
2. **Policy responsiveness:** small or zero deterministic action variation was measured on the sampled anchors; this is weak/negative evidence of useful learned state dependence.
3. **Driving competence:** not demonstrated. There were zero clean successes in 36 full diagnostic episodes, and the 30k full episodes were stationary timeouts.

These observations are all TRAIN-only development evidence and do not qualify the candidate for canonical evaluation.
