# S5.3L — L0 Soft-Q / timeout semantics audit

Status: **PASS — measurement definition is consistent enough to proceed to repeated frozen-policy rollouts.** No training or simulator rollout was run for this source audit.

## Runtime and checkpoint

- Checkpoint: `runs/stage5_pilot/s5_3g_throttle_hold5_seed101_20261115/checkpoints/step_10000`
- Model SHA-256: `6347ae4d82958bba89147ec39507ce3cb6abaa7e04d82f97101d1c09c2dea073`
- Source commit recorded in checkpoint: `4f95af934f7ce9343eabdbca19e92c64f2ecb5c7`
- Runtime verified from the active `metadrive_sb3` environment: Stable-Baselines3 2.9.2, Gymnasium 1.3.0.
- Loaded SAC values: `gamma=0.99`, final frozen `alpha=0.0677141696`, target entropy `-2.0`.
- ReplayBuffer class: SB3 `ReplayBuffer`; `handle_timeout_termination=True`; VecNormalize is absent.
- The K1 return audit uses the executed initial intervention action and the same frozen stochastic policy thereafter. The initial forced action has no entropy term; subsequent stochastic actions contribute `-alpha * log_pi` under SB3's squashed-Gaussian actor log probability.

## Source trace

1. `Stage5TrainingEnv.step()` calls the wrapped MetaDrive step once, classifies outcome, computes the authoritative Platform V1 reward, encodes the final observation, then returns `(obs, reward, terminated, truncated, info)`.
2. `TIMEOUT` at the case horizon is `truncated=True`; safety outcomes such as `OUT_OF_ROAD` are terminal under the safety-first classifier. RewardSpecV1 gives timeout penalty `0.0`; safety terminal reward is `-1.0`.
3. SB3 2.9.2 `DummyVecEnv.step_wait()` maps `terminated or truncated` to `done`, sets `TimeLimit.truncated = truncated and not terminated`, saves the final observation in `info["terminal_observation"]`, then auto-resets.
4. `OffPolicyAlgorithm._store_transition()` replaces the reset observation with that `terminal_observation` before adding the transition to replay.
5. SB3 `ReplayBuffer` defaults `handle_timeout_termination=True`, stores the timeout bit, and returns `done * (1 - timeout)` to SAC training. Thus a pure timeout bootstraps from the true terminal observation; a safety termination does not.
6. SB3 SAC 2.9.2 computes `target = reward + (1-done) * gamma * (min(target_Q1,target_Q2) - alpha*log_pi(next_action|next_obs))`.

## K1 return equation check

For the fixed first intervention action \(a_0\), K1 computes

\[
G_{soft}=r_0+\sum_{t=1}^{T-1}\gamma^t\left(r_t-\alpha\log\pi(a_t\mid s_t)\right)
 + \mathbf{1}_{timeout}\gamma^T\,\mathbb{E}_{a_T\sim\pi}[\min_i Q_{i,target}(s_T,a_T)-\alpha\log\pi(a_T\mid s_T)].
\]

For safety/clean-success terminal outcomes the final bootstrap term is omitted. The terminal observation is the direct Stage5 environment observation, not an auto-reset observation. K1's implementation matches this indexing and timeout distinction. The timeout tail was only `1.2e-14` to `3.4e-6` in the 11 TIMEOUT episodes, so the timeout bootstrap convention cannot account for a Q-return gap near 6 on these cases.

## Interpretation of the proposed upper bound

With route completion bounded to `[0,1]`, a finite episode's undiscounted environment reward is at most `1 progress + 1 clean-success bonus = 2`; time costs and failure penalties only reduce it. For actions in `[-1,1]^2`, differential entropy is at most `ln(4)`. At the checkpoint's frozen alpha and gamma, the corresponding episodic soft-return bound is `2 + alpha*ln(4)/(1-gamma) = 11.3872`.

This is a bound under the finite-episode interpretation. SB3 treats pure TIMEOUT as truncation and therefore formally continues the Bellman value at the final state. The nonzero continuation tail must be acknowledged; the measured `gamma^T * V(s_T)` is negligible here. The bound also describes the expected soft value under a fixed alpha/policy, while the checkpoint critics were trained while SAC's automatic entropy coefficient and actor were changing. Therefore exceeding this bound is a strong calibration signal to replicate, not by itself proof of a code defect.

## L1 budget and isolation

- 4 fixed internal TRAIN diagnostic reset states × 3 fixed initial throttle actions × 8 independent stochastic continuation seeds = **96 complete rollouts planned**.
- Geometry, environment seed, initial reset observation, initial steering, initial throttle action, checkpoint, reward and gamma remain fixed. Future SAC samples use independent seed streams across replicates and common random numbers across the three interventions within each geometry/replicate.
- No optimizer, replay buffer, policy weight, normalization statistic, VALIDATION or TEST access.
- Measured K1 runtime: 942.5 s for 12 rollouts (913.6 s summed episode time); linear estimate for 96: about 7,540 s / 125.7 min. K1 raw trace was 93,472,992 bytes; estimated L1 trace volume about 748 MB decimal. D: had 26.29 GB free before L1.
- Per-replicate outputs are isolated under `results/stage5/s5_3l_softq_calibration_seed101_20261115/replicate_NN/` and `runs/stage5_pilot/s5_3l_softq_calibration_seed101_20261115/replicate_NN/`; the existing K1 artifacts are never overwritten.

## Sources inspected

- `src/training/stage5/env_adapter.py`: `Stage5TrainingEnv.step()`.
- `src/platform/episode.py`: safety-first terminal/truncation classification.
- `src/platform/reward.py`: RewardSpecV1 step reward and terminal components.
- `src/training/stage5/sac_trainer.py`: SAC gamma, alpha and default ReplayBuffer construction.
- Installed source inspected at runtime: SB3 2.9.2 `DummyVecEnv.step_wait`, `OffPolicyAlgorithm.collect_rollouts`, `_store_transition`, `ReplayBuffer.__init__`, `ReplayBuffer.add`, `ReplayBuffer._get_samples`, and `SAC.train`.
