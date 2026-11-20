# S5.3K — K0 action temporal fidelity

Status: **PASS**. No training or optimizer update was run.

## What the 5-step runs mean

S5.3G deliberately enabled `ThrottlePersistenceSAC(throttle_hold_steps=5)`. Its `_sample_action` calls the actor sampler every collection step, then replaces only throttle/brake before sending the action to the environment and replay. Steering remains resampled each decision. The logging callback records the post-intervention action, so the historic 10k telemetry cannot reveal the discarded raw actor throttle.

The historical log has 10,000 contiguous decision rows. When throttle runs are segmented at case/episode boundaries, run-length histogram is `{"1": 4, "2": 3, "3": 3, "4": 4, "5": 1993}`; lengths sum to 10,000. The earlier phrase ‘2,007 runs, each exactly five’ was inaccurate: 2,007 was the run count ignoring episode boundaries, and quantiles/max did not prove every run was length five. The actual distribution is recorded in `K0_audit.json`.

Within the same episode and same global 5-step hold block, unexpected throttle changes: 0. Steering changed on 7993 adjacent pairs inside such blocks, consistent with steering being sampled each decision.

## Decision-step to physics-step mapping

Runtime probe: 3 Gymnasium steps → 3 MetaDrive `step()` calls → 3 engine repeats of `[5, 5, 5]` → 15 physics-world steps. Config is `decision_repeat=5`, `physics_world_step_size=0.02` s; one policy decision is 0.10 s (10.0 Hz). Actions matched both MetaDrive input and Platform info: `True` / `True`.

SB3 source path is one `_sample_action` → one `env.step(actions)` → one `_store_transition` per collect-rollout loop iteration. `Stage5TrainingEnv.step()` calls MetaDrive once; MetaDrive calls `engine.step(decision_repeat)`; the engine loops that many physics steps. Thus the configured repeat holds an action over five 0.02 s physics substeps, not five SAC/Gymnasium decisions.

## Decision

K0 passes for the declared S5.3G throttle-hold intervention and the current Stage5 → MetaDrive temporal path. The five-repeat simulator setting is not an extra five-decision action hold. S5.3G intentionally did add a separate five-decision throttle hold; this was declared, reset at episode boundaries, and action/replay values were changed together. Therefore K1 can be designed as the next diagnostic if desired; K0 found no undeclared temporal mapping defect.

Limit: the historical S5.3G log does not preserve the actor's pre-hold throttle sample or raw replay buffer. K0 validates source/runtime semantics and recorded executed actions; it cannot retrospectively verify every historical transition array.
