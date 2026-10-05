# Research Agents

This package is the intended home for real Stage 0+ agent implementations.

## Current state

Platform V1 is complete, but the **canonical Stage 0 Random / Naive research baseline is the next implementation milestone**. Existing `fixture_*` policies live in the platform verification layer and are not substitutes for scientific stage implementations.

## Before adding an agent

Read:

- [`../../docs/AGENT_DEVELOPMENT_GUIDE.md`](../../docs/AGENT_DEVELOPMENT_GUIDE.md)
- [`../../docs/STAGE_ROADMAP.md`](../../docs/STAGE_ROADMAP.md)
- [`../../docs/EXPERIMENT_WORKFLOW.md`](../../docs/EXPERIMENT_WORKFLOW.md)

New agents should implement the common `AgentPolicy` lifecycle, consume `AgentInputV1`, emit actions through a certified adapter, and be registered in `src/launcher/registry.py`.

Do not implement private environment wrappers or read evaluator internals from an agent merely to simplify one method.
