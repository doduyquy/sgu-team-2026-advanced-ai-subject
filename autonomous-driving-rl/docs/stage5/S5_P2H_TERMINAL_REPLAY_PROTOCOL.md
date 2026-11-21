# Stage 5 P2-H — Terminal-Aware Critic Replay A/B

**Status:** preregistered before scientific training; TRAIN only.  
**Machine protocol:** [`protocol.json`](../../results/stage5/p2h_terminal_replay_seed101_20261121/protocol.json)  
**Protocol SHA-256:** `0f1c42c80f869bbbee801230eef722fa72afd44cb9c50c6325d185a34657f461`  
**Parent:** Draft PR #48, `48ce34a5b1e10917d99753494d5fca99fbee8f81`.

## Question and registered contrast

Does prioritizing real terminal transitions in the **critic** replay minibatch help fresh SAC learn state-dependent driving? Arm A retains standard uniform SAC replay. Arm B uses the terminal-aware distribution below for critic loss only; its actor and entropy-temperature updates sample uniformly. Both arms use the P2-D actor-only LeakyReLU(0.01), ReLU critics, and the same fresh initial tensors. Optimizer states and replay buffers are independent.

All other settings remain fixed: corrected TaskContext producer `reference_route_station_proposal_v1`, 409D observation with `pilot_groupwise_physical_v1`, Box2 actions, RewardSpecV1, gamma 0.99, tau 0.005, the locked eight Easy TRAIN training geometries, six disjoint Easy TRAIN diagnostics, root seed 101, and 10,000 actual environment decisions per arm. Checkpoints are 0/2k/5k/10k. Only TRAIN is used; benchmark eligibility remains false.

## Terminal-aware distribution

Eligible samples are confirmed `safety` or `clean_success` terminals recorded online in the replay buffer. A row is eligible only when `dones=1` and `timeouts=0`; timeout truncations remain ordinary uniform candidates and retain SB3 timeout bootstrapping.

At each critic draw, the registered distribution is

`P(i) = 0.80 / N + 0.20 * priority(i) / sum(priority(j) for eligible terminal j)`

where the terminal priority is `(min(max(|TD1|, |TD2|), 10) + 0.001)^0.60`. Draws are with replacement. If the buffer has no eligible terminal, the sampler falls back to uniform. Importance weights are `(N*P(i))^-beta`, max-normalized across the current buffer, and applied only to per-sample critic MSE. Beta increases linearly from 0.40 to 1.0 over 9,000 optimizer updates. New rows start at the current maximum priority, clipped to 1–10. No counterfactual P2-G data or teacher transitions enter the buffer.

## Evaluation and interpretation

At 0/2k/5k/10k, the frozen deterministic actor gets the same six 300-decision TRAIN diagnostic probes. At 10k, all six full diagnostic episodes run in separate environments. The report separates clean success, safety failure, timeout and `PROBE_CENSORED`, and includes route-progress delta from reset, speed, stationary fraction, action responsiveness, terminal counts and sampling exposure. Speed or Q/TD changes alone do not count as driving improvement. This one-seed, small-pool development experiment is not a benchmark or a general claim of superiority.

## Reproduction

Use the `metadrive_sb3` environment and run:

```powershell
python scripts/run_stage5_p2h_terminal_replay.py --arm ALL
python scripts/evaluate_stage5_p2h_terminal_replay.py --arm BOTH
```

Large checkpoint and replay bundles are kept under ignored `runs/stage5_p2h/`; the committed protocol, compact evidence and checksums bind the source and run artifacts.
