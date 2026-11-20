# Stage 5 P2-D — Next Decision

## Decision

`REPRESENTATION_RECOVERED_NO_DRIVING_GAIN` — narrowly, LeakyReLU(0.01) kept a nonzero derivative through negative first-layer preactivations, but deterministic actor sensitivity still collapsed toward a near-constant output. The treatment did not yield safe driving: 0/6 clean successes and 6/6 out-of-road outcomes in full 10k diagnostic episodes.

The control also failed to drive: it was stationary and timed out on all six full diagnostic cases. B's additional motion and mean route progress do not outweigh its safety failures.

## Candidate status

- Keep ReLU SAC as the reference baseline.
- Do not promote LeakyReLU as an improved SAC configuration.
- Preserve B as a research artifact; it is not a candidate for benchmark evaluation.
- Keep `benchmark_eligible=false` for both arms.
- Do not extend either run beyond 10k or launch another activation sweep from this result.

## Evidence boundary

This is one root seed, eight training geometries and six fixed diagnostic geometries. It shows what happened under this registered run, not that actor activation is irrelevant or that SAC cannot learn the task. The open negative-side derivative did not sustain useful observation sensitivity; whether critic/replay support, reward/entropy interaction or another mechanism limits learning remains unresolved.

## Proposed next action — not started

If Stage 5 work is authorized to continue, prioritize one bounded audit of critic/action support and replay coverage around safe versus failed moving states, using these saved 10k buffers. Do not train or change reward, gamma, actor architecture or exploration until that evidence identifies a testable single intervention. This proposal is not an automatic P2-E task and requires a separate instruction.
