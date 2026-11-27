# S5.5 Next Decision

## Decision

**Driving result:** `NO_IMPROVEMENT` in durable clean driving.

**Mechanism conclusion:** `INCONCLUSIVE`; no single root cause is established.

**Multi-map decision:** `NO_GO_MULTI_MAP` for the current candidate.

**Benchmark status:** not eligible; `benchmark_eligible=false` remains unchanged.

## Evidence behind the decision

- All three fresh reward arms completed 30,000 environment steps, 29,000 optimizer updates, and visited all eight training geometries.
- The fixed six-case full diagnostic at 10k produced no clean success: A had four safety failures and two timeouts; B had six timeouts while stationary; C had six safety failures despite higher speed and route progress.
- At 30k, all three arms were 100% stationary on the six full diagnostic episodes and all six timed out. There was no sustained improvement to carry forward.
- Deterministic actor action variation on its own sampled anchors was zero from 5k onward. Training records show repeated throttle/brake sign changes and moving runs no longer than 1.6 seconds.
- These data do not prove reward scale, route-metre units, critic error, or exploration alone caused the collapse. One training root seed and six held-out internal TRAIN cases are not enough to compare reward variants generally.

## One justified next task

If Stage 5 development continues, the next bounded experiment should test **temporally correlated exploration with the V1 reward, 409D input, and SAC settings held fixed**, against a fresh V1 control on the same locked TRAIN geometry pools. The reason is measured rather than speculative: the replay-driving experience had very short sustained movement and thousands of throttle/brake sign switches, while the frozen policy returned to stationary behavior. The experiment should test whether correlated action sampling creates longer, useful moving trajectories and whether that leads to clean deterministic driving. This changes training action sampling, not policy history; the current results do not justify recurrent memory.

Do not promote the 10k Metre arm: all six of its full diagnostic cases ended in safety failure. Do not start multi-map training or official VALIDATION/TEST evaluation from these results. First require repeated clean successes on distinct unseen internal TRAIN diagnostic geometries under native SAC actions, with exact checkpoint provenance. Any new training run needs its own frozen protocol and approval; it is not part of S5.5.

## Limits to carry forward

- Single training root seed (101); no estimate of between-seed variation.
- Easy-only, no-traffic setup; traffic and obstacle-use competence were not tested.
- Six diagnostic geometries; internal development evidence, not benchmark evaluation.
- Training per-episode outcome details were not persisted by the runner; the analysis explicitly marks them missing. Fixed evaluation records remain intact.
- Checkpoint and protocol identity are recorded, but the Stage 5 candidate remains a development artifact, not a certified benchmark agent.
