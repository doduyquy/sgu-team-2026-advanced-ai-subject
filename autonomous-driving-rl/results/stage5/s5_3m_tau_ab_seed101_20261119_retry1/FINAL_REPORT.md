# S5.3M final result

See the full methods, per-tier results and interpretation in [`STAGE5_FINAL_EXPERIMENT.md`](../../../docs/stage5/STAGE5_FINAL_EXPERIMENT.md).

- Training source: `c23f5a270114147fc39fdb508b9aed0def6d16c6`
- Protocol SHA-256: `851e740b37cbb01e30085174f32153b0a483c42c086dd4997cedb6631dc3f328`
- Arm A (`tau=.005`): 10,000 steps, 9,000 optimizer updates, 17.7 steps/s, five completed training episodes.
- Arm B (`tau=.001`): 10,000 steps, 9,000 optimizer updates, 18.2 steps/s, five completed training episodes.
- Diagnostics: 8 fixed internal TRAIN geometries × 3 checkpoints × 2 arms = 48 probes, all capped at 300 decisions.
- At 10k, B minus A reset-relative route progress averaged `+0.002675` across the eight paired cases (8/8 positive). Both arms remained 100% stationary by the `<1 km/h` rule and had no clean success or safety failure; all probes were `PROBE_CENSORED`.
- B's mean anchor min-Q was lower than A's (4.218 vs 15.238), but this does not establish improved critic calibration or explain the stationary policy.
- Decision: retain B as a development candidate for confirmation only; keep A as reference. Neither checkpoint is benchmark-eligible.

Machine-readable records, protocol, checkpoint SHA-256 values and source hashes are in `protocol.json`, `analysis.json`, `q_anchor_diagnostics.json`, `artifact_manifest.json` and `anchor_observations.json`. Raw weights and logs are in the ignored local `runs/stage5_tau_ab/` tree and are not committed.
