# Research Agents

This package is the intended home for real Stage 0+ agent implementations.

## Current state

Only two CLI agent IDs are active research implementations:

| ID | Entry file |
|---|---|
| `stage0_random_naive` | `random_naive_agent.py` |
| `stage1_rule_based` | `rule_based_agent.py` |

Stage 1's traffic/memory/junction/lane-change files are imported helpers, not
additional agents to select. Archived `policy.py` files under `experiments/`
are reproducibility records. Source cleanup: [0.6.1](../../docs/SOURCE_CLEANUP_V0_6_1.md).

`Stage0RandomNaiveAgent` holds random controls for 2–4 simulation seconds. When a
real road junction comes within 8 m, it immediately samples uniformly among the
available left/straight/right exits and restarts the hold timer. It triggers once
per encounter, with seeded replay and full episode-state reset.

See [Stage 0 behavior and configuration](../../docs/STAGE0_RANDOM_NAIVE_AGENT.md).
The agent remains registered for SANDBOX/AUDIT development. Existing `fixture_*`
policies validate the platform and are not scientific stage implementations.

`Stage1RuleBasedAgent` v0.6.1 keeps the lane and applies adaptive speed limits
for curves, braking distance, visibility, junction approaches and front vehicles.
It matches traffic to a curved route corridor, predicts crossing/cut-in conflicts,
yields before junction entry and brakes for unclassified LiDAR obstacles.
It compensates ego motion, associates actor tracks with TTL, and evaluates
possible connected road paths without reading another vehicle's navigation.
The junction lifecycle locks a route branch through entry/traversal/exit and
waits upstream of blocked intersection exits. Full reset clears all episode memory.
Its open-road cap is 45 km/h; lane recovery remains capped at 8 km/h.
See the [usage guide](../../docs/RULE_BASED_AGENT_GUIDE.md) and
[full development plan](../../docs/RULE_BASED_AGENT_DEVELOPMENT_PLAN_VI.md).
See the [version log](../../docs/RULE_BASED_CHANGELOG.md) and
[0.4.0 experiment report](../../docs/RULE_BASED_V0_4_0_EXPERIMENT_REPORT.md).
See the [P2/P3 design](../../docs/RULE_BASED_P2_P3_DESIGN.md) and
[development work log](../../docs/RULE_BASED_V0_5_0_WORKLOG.md).
The [0.5.0 report](../../docs/RULE_BASED_V0_5_0_EXPERIMENT_REPORT.md) records
13/16 paired TRAIN successes, zero candidate crashes and three unresolved timeouts.
P4 adds mandatory adjacent changes, front/rear stopping-gap/TTC and swept-body
guards, persistent quintic paths, safe precommit abort, postcommit stop/resume,
and conditional overtaking/return. Only confirmed broken-white dividers permit
crossing. See the [P4 design](../../docs/RULE_BASED_P4_DESIGN.md) and
[0.6 work log](../../docs/RULE_BASED_V0_6_0_WORKLOG.md).
The [0.6 report](../../docs/RULE_BASED_V0_6_0_EXPERIMENT_REPORT.md) records
279 passing tests, six physical probes/929 exact archive replay steps, six
completed natural lane changes for passing, and 13/16 TRAIN successes with no safety
flags/lost successes. The three Extreme timeouts remain; source 0.6 is frozen.
Complete legal right-of-way handling and P5 remain pending; the agent is still
SANDBOX/AUDIT only with a pending-audit 1.3.0 public schema extension.
Runs preserve source/configuration hashes and per-step public decisions.

## Before adding an agent

Read:

- [`../../docs/AGENT_DEVELOPMENT_GUIDE.md`](../../docs/AGENT_DEVELOPMENT_GUIDE.md)
- [`../../docs/STAGE_ROADMAP.md`](../../docs/STAGE_ROADMAP.md)
- [`../../docs/EXPERIMENT_WORKFLOW.md`](../../docs/EXPERIMENT_WORKFLOW.md)

New agents should implement the common `AgentPolicy` lifecycle, consume `AgentInputV1`, emit actions through a certified adapter, and be registered in `src/launcher/registry.py`.

Do not implement private environment wrappers or read evaluator internals from an agent merely to simplify one method.
