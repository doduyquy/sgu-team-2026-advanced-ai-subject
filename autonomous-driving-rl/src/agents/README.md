# Research Agents

This package is the intended home for real Stage 0+ agent implementations.

## Current state

The launcher registers these research agents for SANDBOX and AUDIT.

| ID | Entry file | Model setup |
|---|---|---|
| `stage0_random_naive` | `random_naive_agent.py` | None |
| `stage1_rule_based` | `rule_based_agent.py` | None |
| `stage2_astar` | `astar_agent.py` | None |
| `stage2_bfs`, `stage2_dijkstra` | `graph_search_agents.py` | None |
| `stage2_hybrid_astar` | `hybrid_astar_agent.py` | None |
| `stage3_mcts` | `MCTS_agent.py` | None |
| `stage3_mppi` | `MPPI.py` | None |
| `stage4_pdd_hybrid` | `agent4_pdd_agent.py` | Bundled PDD release, verified and loaded automatically |
| `stage5_sac` | `stage5_sac_agent.py` | CLI `--checkpoint` selects a verified SAC bundle |
| `stage6_astar_learned` | `astar_agent.py` | CLI `--training-manifest` and `--training-seed` select a trained heuristic |

See the [Agent 4 PDD release guide](../../docs/AGENT4_PDD_AGENT.md) for the frozen model and launcher command.

`MCTSAgent` v3.0.0 uses UCT, guided rollouts and a local model of vehicle motion.
`MPPIAgent` optimizes continuous steering and throttle sequences using NumPy
kinematic bicycle rollouts, moving actors and public LiDAR. See the
[project pipeline and CLI examples](../../docs/PROJECT_AI_PIPELINE_VI.md).

A*, BFS, Dijkstra and Hybrid A* use a shared `discrete9_lowbranch_v1` path
follower. See the [search planner comparison](../../docs/SEARCH_PLANNERS.md)
and the [learned A* training workflow](../../docs/ASTAR_LEARNED_HEURISTIC.md).

Stage 1's traffic/memory/junction/lane-change files are imported helpers, not
additional agents to select. Archived `policy.py` files under `experiments/`
are reproducibility records. Source cleanup: [0.6.1](../../docs/SOURCE_CLEANUP_V0_6_1.md).

`Stage0RandomNaiveAgent` v0.2.0 samples its first action immediately and holds
random controls for 2–4 simulation seconds. When a real road junction comes within
8 m, it immediately samples uniformly among the available left/straight/right
exits and restarts the hold timer. It triggers once per encounter, with seeded
replay and full episode-state reset.

See [Stage 0 behavior and configuration](../../docs/STAGE0_RANDOM_NAIVE_AGENT.md).
The historical `fixture_*` policies exercise the launcher infrastructure.

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

## Verify Stage 0 locally

From `autonomous-driving-rl`, run the tests and preview a fixed TRAIN case with the
project's MetaDrive Python interpreter.

```powershell
python -X utf8 -B -m unittest tests.test_random_naive_agent
python -X utf8 -B -m src.launcher.cli plan --mode SANDBOX --agent stage0_random_naive --tier Easy --sequence SCS --geom-seed 0 --env-seed 0 --agent-seed 101 --wandb DISABLED
```

## Before adding an agent

Read:

- [`../../docs/AGENT_DEVELOPMENT_GUIDE.md`](../../docs/AGENT_DEVELOPMENT_GUIDE.md)
- [`../../docs/STAGE_ROADMAP.md`](../../docs/STAGE_ROADMAP.md)
- [`../../docs/EXPERIMENT_WORKFLOW.md`](../../docs/EXPERIMENT_WORKFLOW.md)

New agents should implement the common `AgentPolicy` lifecycle, consume `AgentInputV1`, emit actions through a certified adapter, and be registered in `src/launcher/registry.py`.

Do not implement private environment wrappers or read evaluator internals from an agent merely to simplify one method.
