# Native MetaDrive SAC baseline

This experiment is intentionally independent of Platform V1 / Stage 5. It uses the installed MetaDrive 0.4.3 single-agent environment, native observation, native Box(2) action, and native reward. It does not import Stage 5, Stage 3, a teacher, or pretrained weights.

Run from `autonomous-driving-rl` in the `metadrive_sb3` conda environment:

```powershell
conda activate metadrive_sb3
python experiments/native_metadrive_sac/train.py --run-dir runs/native_metadrive_sac/native_s_20261121
```

The fixed task is MetaDrive block sequence `S`, 3 lanes, no traffic, fixed spawn lane, 1,000-decision episode horizon, and the simulator's native reward. Checkpoints are saved at 0, 100k, 300k, 500k, and 1M steps. Each scheduled evaluation loads a frozen checkpoint in a separate process and runs five complete deterministic episodes on the same map. Evaluation writes per-step CSV trajectories and episode JSON. The registered budget is 1,000,000 environment decisions; no automatic extension is allowed.

The replay capacity is 1,000,000 transitions. To fit the 259D native observation replay on this workstation, the supplied SB3-compatible replay buffer stores the same SB3 arrays as disk-backed NumPy memmaps. Sampling, timeout masks, and Bellman learning remain SB3 `ReplayBuffer` semantics. The storage is local under the run directory and must not be committed.

`protocol.json` contains the immutable experiment parameters and source/runtime identity. `report.md` is written when training reaches the budget or a scheduled checkpoint meets the registered 4/5 clean-episode criterion. A short `--smoke-steps` run validates integration only; it is never counted as a scientific result.

## Hyperparameter provenance

The requested values are gamma 0.99, learning rate 1e-4, 10k random-action warm-up, and replay capacity 1e6. The 2018 SAC Applications paper's common benchmark table reports learning rate 3e-4, replay capacity 1e6, minibatch 256, and target smoothing 0.005; therefore this experiment's 1e-4 rate is a deliberate requested deviation, not a paper-exact reproduction. SB3 2.9.2 additionally supplies automatic entropy tuning by default when `ent_coef="auto"`, uses a squashed Gaussian actor for continuous actions, and uses twin Q-targets. Exact installed implementation details are captured in `protocol.json` and the report.
