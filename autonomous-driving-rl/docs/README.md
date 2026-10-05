# Documentation Guide

This directory has two layers of documentation.

## 1. Team-facing guides

Read these when joining or developing the project:

- [`GETTING_STARTED.md`](GETTING_STARTED.md) — machine setup and smoke verification.
- [`PLATFORM_V1_OVERVIEW.md`](PLATFORM_V1_OVERVIEW.md) — what Platform V1 already provides.
- [`EXPERIMENT_WORKFLOW.md`](EXPERIMENT_WORKFLOW.md) — experiment modes, evaluation protocol, and result interpretation.
- [`AGENT_DEVELOPMENT_GUIDE.md`](AGENT_DEVELOPMENT_GUIDE.md) — how to implement and register agents.
- [`STAGE_ROADMAP.md`](STAGE_ROADMAP.md) — Stage 0–7 research map and expected integration boundaries.
- [`WORKBENCH_GUIDE.md`](WORKBENCH_GUIDE.md) — practical Research Workbench usage.

## 2. Detailed technical evidence

[`environment/`](environment/) contains the detailed Platform V1 audit/specification reports produced while the research foundation was built. They are valuable as technical references and evidence, but a new team member should normally read the six guides above first.

The audit sequence covers:

- observation/action design;
- scenario/map suite;
- episode lifecycle;
- reward and metrics;
- evaluation protocol;
- agent contract;
- logging/W&B/provenance;
- launcher/preflight;
- Workbench, results browser, and final hardening.

[`survey/`](survey/) contains literature-survey material and course research references. It is not an authoritative source for Platform V1 runtime behavior.
