# Platform V1 Technical Reference

This directory contains the detailed technical specification/audit reports created while building Research Platform V1.

For normal onboarding, start with the higher-level guides one directory up:

- [`../GETTING_STARTED.md`](../GETTING_STARTED.md)
- [`../PLATFORM_V1_OVERVIEW.md`](../PLATFORM_V1_OVERVIEW.md)
- [`../EXPERIMENT_WORKFLOW.md`](../EXPERIMENT_WORKFLOW.md)
- [`../AGENT_DEVELOPMENT_GUIDE.md`](../AGENT_DEVELOPMENT_GUIDE.md)
- [`../STAGE_ROADMAP.md`](../STAGE_ROADMAP.md)
- [`../WORKBENCH_GUIDE.md`](../WORKBENCH_GUIDE.md)

Use the documents below when you need exact design rationale, contracts, calibration evidence, or audit history:

- `METADRIVE_SETUP.md` — pinned simulator setup and troubleshooting.
- `PLATFORM_V1_OBSERVATION_ACTION_AUDIT.md` — observation/action investigation and final platform direction.
- `PLATFORM_V1_MAPSUITE_AUDIT.md` — scenario suite construction and difficulty design.
- `PLATFORM_V1_EPISODE_AUDIT.md` — episode lifecycle, safety semantics, and horizon calibration.
- `PLATFORM_V1_REWARD_METRICS_AUDIT.md` — reward candidates and evaluation metrics.
- `PLATFORM_V1_EVALUATION_PROTOCOL_AUDIT.md` — TRAIN/VALIDATION/TEST split, seeds, and benchmark protocol.
- `PLATFORM_V1_AGENT_CONTRACT_AUDIT.md` — AgentInput/AgentPolicy/action-adapter contract and information parity.
- `PLATFORM_V1_LOGGING_WANDB_AUDIT.md` — local-first logging, provenance, W&B mirroring, timing, integrity.
- `PLATFORM_V1_LAUNCHER_CORE_AUDIT.md` — resolver, registry, preflight, and executor architecture.
- `PLATFORM_V1_WORKBENCH_AUDIT.md` — Workbench shell and worker protocol.
- `PLATFORM_V1_WORKBENCH_RESULTS_AUDIT.md` — result repository, integrity trust states, and live telemetry UX.
- `PLATFORM_V1_WORKBENCH_HARDENING_AUDIT.md` — final Workbench hardening and Platform V1 handoff.

These reports are evidence/reference material. Stage implementations should normally depend on the runtime contracts in `src/platform`, `src/launcher`, and `configs/platform`, not copy logic from the audit scripts.
