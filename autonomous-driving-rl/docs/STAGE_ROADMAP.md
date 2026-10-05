# Research Stage Roadmap

The stage taxonomy is a research map, not a requirement to fully implement all eight stages during one course. Every stage is expected to plug into the same Platform V1 evaluation boundary.

Human-readable progression:

```text
No Intelligence
   → Human Rules
   → Search
   → Simulation / Look-ahead
   → Learn from Data
   → Learn by Interaction
   → Learn + Search
   → Learn a World Model
```

## Course-priority path

The practical core path is currently:

```text
Platform V1 (complete)
   → Stage 0 Random / Naive
   → Stage 1 Rule-Based / Heuristic
   → Stage 5 Model-Free RL
   → final controlled comparison
```

Stages 2/3/4/6/7 remain valid research directions and may be added when they answer a useful comparison question and time permits.

## Stage 0 — Random / Naive

**Purpose:** establish the minimum no-intelligence baseline and verify what uncontrolled action selection does badly.

Expected design:

- reproducibly seeded stochastic policy;
- no learning;
- same public input/action contracts as every later method;
- report success, safety failures, route completion, timeout/outcome mix, and decision latency.

A true Stage 0 agent should be implemented and registered separately from the existing infrastructure fixture.

## Stage 1 — Rule-Based / Heuristic

**Purpose:** demonstrate how far simple human-designed driving logic can improve over Random and expose the limitations of manual rules.

Possible components include lane/route following, speed control, obstacle-distance logic, and safety heuristics derived only from public AgentInput.

The exact heuristic should be explainable and parameterized. It must not use evaluator-private completion/success information.

## Stage 2 — Planning / Search

**Purpose:** choose actions by explicitly searching/planning ahead rather than applying a fixed reactive rule.

The low-branching certified discrete adapter is available for this family. Search state must still be built from allowed information.

If a planner needs a richer route graph or planning representation than the current public contract supplies, propose the representation as an explicit shared platform extension instead of reading evaluator internals.

## Stage 3 — Simulation-Based Planning / Look-ahead

**Purpose:** evaluate candidate futures using rollouts or MCTS-like reasoning.

Important current boundary: Platform V1 does **not** expose the evaluator's live MetaDrive instance as a cloneable oracle to the agent. Do not bypass this by reaching into the environment object.

A Stage 3 implementation therefore needs an explicit planning/rollout model design (analytical, learned, or a separately defined public simulator abstraction) before implementation. That design should be reviewed for information fairness.

## Stage 4 — Learning from Data / Imitation

**Purpose:** learn a policy from demonstrations rather than direct trial-and-error reward optimization.

Typical flow:

```text
training data → supervised/imitation training → frozen checkpoint → AgentPolicy inference
```

The training data source and information profile must be documented. Benchmark inference still goes through the same AgentInput/action interface.

## Stage 5 — Model-Free Reinforcement Learning

**Purpose:** learn the driving policy through environment interaction and reward.

Likely algorithm families depend on the selected action representation (for example PPO/SAC for continuous control or DQN-like methods for discrete control).

The reward revision must be frozen using TRAIN/VALIDATION before final TEST. Training infrastructure may be stage-specific, but the resulting frozen policy must be evaluated through Platform V1.

## Stage 6 — Search + Learned Policy / Value

**Purpose:** combine learned guidance/value estimation with explicit search.

This is closer to the AlphaZero-style idea: learning narrows/evaluates the search while search improves decision quality.

The learned model and search procedure may be complex internally, but they remain one AgentPolicy at the platform boundary.

## Stage 7 — Model-Based RL / World Models

**Purpose:** learn a model of environment dynamics and use it for prediction/planning/decision making.

Training data and the learned model are internal to the method. Final evaluation remains frozen inference under the same benchmark contracts.

## What may vary by stage

Allowed method-specific differences include:

- internal algorithm architecture;
- learned model/checkpoint;
- search procedure;
- declared hyperparameters;
- certified action adapter when method requirements differ;
- deterministic/stochastic inference declaration.

## What should not silently vary by stage

Do not quietly change:

- TEST cases;
- primary metric definitions;
- episode/safety semantics;
- public information rights to benefit one method;
- environment difficulty definition;
- logging truth/integrity rules.

If a research question requires a change to one of these, version it as a platform-level experiment rather than hiding it inside a stage implementation.
