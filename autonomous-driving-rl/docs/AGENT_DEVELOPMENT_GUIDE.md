# Agent Development Guide

This is the main integration guide for Stage 0 and later research methods.

The central rule is: **implement the algorithm as an agent; do not reimplement the environment or benchmark inside the agent.**

## 1. Agent runtime interface

Every research agent must provide the Platform V1 policy lifecycle:

```python
@property
def descriptor(self) -> AgentDescriptor: ...

def reset(self, public_context: AgentPublicEpisodeContext, agent_seed: int | None = None) -> None: ...

def act(self, agent_input: AgentInputV1) -> AgentDecision: ...

def close(self) -> None: ...
```

Inheritance is not required; matching the `AgentPolicy` protocol is sufficient.

Recommended implementation location:

```text
src/agents/<descriptive_agent_name>.py
```

Do not add Stage implementations to `src/platform/`. The platform package defines common contracts and should stay algorithm-neutral.

## 2. Descriptor responsibilities

The descriptor declares what the policy is and how the platform should call it:

- stable agent ID and version;
- method family / stage identity;
- input profile;
- action adapter;
- deterministic vs stochastic inference;
- whether it keeps episodic state.

The descriptor must match the registration metadata used by the launcher.

## 3. `reset()` responsibilities

`reset()` is called at the start of every episode.

It should:

- clear episodic state (recurrent hidden state, counters, temporary history);
- initialize agent-side RNG from `agent_seed` when the policy is stochastic;
- preserve trained/frozen model weights and static configuration.

Do not seed MetaDrive from the agent. Environment stochasticity is controlled independently by the platform.

## 4. `act()` responsibilities

`act()` receives only `AgentInputV1` and returns an `AgentDecision`.

An agent must not inspect the live evaluator/simulator object to obtain hidden information. In particular, do not reach around `AgentInputV1` to read global route completion, split/case identity, success/failure flags, or benchmark metrics.

The action payload must match the declared certified adapter. Invalid actions are technical failures; the platform does not silently clip them into a different decision.

Optional diagnostics are allowed for explainability/debugging, but they are observational and must not change evaluation rules.

## 5. Choose an action adapter deliberately

Current certified choices:

- `continuous_box2_v1` — direct steering + throttle/brake; natural for continuous-control RL and continuous baselines.
- `discrete25_native_v1` — 25 steering/throttle combinations.
- `discrete9_lowbranch_v1` — low-branching action set intended for search/planning experiments.

Do not invent a private action mapping inside one agent for benchmark use. If a genuinely new shared action representation is required, treat it as a platform contract change and review/version it separately.

## 6. Register the agent

The launcher obtains agents from `src/launcher/registry.py`.

A Stage implementation should:

1. define the policy class under `src/agents/`;
2. define a small factory that creates a fresh policy instance;
3. add a matching `AgentRegistrationV1` in the canonical registry;
4. initially use SANDBOX/AUDIT eligibility for development;
5. enable benchmark eligibility only after the agent integration and scientific assumptions have been reviewed/tested.

Do **not** mark an unfinished agent benchmark-eligible simply to get around a preflight block.

## 7. Minimum test expectations for a new agent

At minimum, add tests for:

- descriptor/registration parity;
- deterministic repeatability or stochastic repeatability under the same `agent_seed`;
- different stochastic seeds produce permitted variation;
- episodic state resets correctly;
- action shape/range or discrete index is always valid;
- no evaluator-private inputs are required;
- SANDBOX plan resolves and executes;
- benchmark eligibility/preflight behavior matches the intended development state.

For learned agents, also test checkpoint loading and frozen inference behavior.

## 8. Training vs evaluation

The current launcher/workbench is an execution and evaluation foundation, not a complete RL trainer UI.

For learned stages, build the training pipeline as a separate research component that uses TRAIN scenarios and produces a checkpoint. Evaluation should then load the frozen checkpoint through the normal AgentPolicy interface.

During VALIDATION/TEST inference:

- do not update model weights;
- do not learn from evaluation rewards/outcomes across episodes;
- do not tune hyperparameters from TEST.

## 9. When a stage appears to need more information

Do not access MetaDrive internals directly from the policy as a shortcut.

First ask whether the need is:

- algorithm-internal state that can be derived from existing `AgentInputV1`;
- training-only data that should never appear during benchmark inference;
- a genuinely missing public capability required by several method families.

Only the third case justifies proposing a versioned platform extension. Such an extension must preserve fairness and be available under the same rules to relevant compared methods.

## 10. Stage 0 starting point

The immediate next implementation is the real Stage 0 Random / Naive baseline.

Do not reuse `fixture_seeded_random` as the scientific Stage 0 result merely because it already emits random actions. The fixture was designed to validate infrastructure. Stage 0 should be a named, tested research baseline with explicit policy semantics, registration, reproducible seeds, documented expectations, and its own benchmark results.

See [`STAGE_ROADMAP.md`](STAGE_ROADMAP.md) for the role of every stage.
