# Experiment Workflow

This guide describes how research experiments should be run after Platform V1. It is deliberately algorithm-neutral: Random, rule-based, planning, imitation, and RL agents should all go through the same scientific boundaries.

## 1. Four launcher modes

### SANDBOX

Use for development, learning, visual inspection, and debugging.

- resolves one TRAIN case;
- can render natively;
- non-canonical;
- safe place to inspect an agent before benchmark eligibility.

Typical use:

```powershell
python -m src.launcher.cli plan --mode SANDBOX --agent fixture_seeded_random --tier Easy --agent-seed 101
python -m src.launcher.cli run --mode SANDBOX --agent fixture_seeded_random --tier Easy --agent-seed 101 --render NATIVE
```

### AUDIT

Use for platform verification and contract/runtime checks. This is not an algorithm ranking mode.

### VALIDATION

Use for model selection, checkpoint selection, ablation, and research decisions.

- canonical full suite: **96 cases**;
- benchmark-eligible agent required;
- headless execution;
- never treat it as the final held-out result.

### TEST

Use only after the method/configuration is frozen.

- canonical full suite: **60 cases**;
- benchmark-eligible agent required;
- headless execution;
- final holdout;
- do not tune from TEST outcomes.

## 2. Normal research loop

```text
Implement/change method
        ↓
Unit tests
        ↓
SANDBOX sanity runs on TRAIN
        ↓
Training / development on TRAIN
        ↓
VALIDATION for selection/tuning
        ↓
Freeze method + checkpoint + config
        ↓
TEST full suite once for final comparison
        ↓
Failure analysis and report
```

If TEST exposes a weakness, report/analyze it. Do not silently alter the TEST suite to make the method look better.

## 3. How do we decide whether an agent is better?

Do not rank agents by cumulative RL return. Reward is a learning signal whose scale depends on reward design.

The primary scorecard is:

| Metric | Preferred direction | What it answers |
|---|---|---|
| Clean Success Rate | Higher | How often does the agent reach the destination safely? |
| Safety Failure Rate | Lower | How often does it crash or leave the road? |
| Mean Final Route Completion | Higher | How far does it get on average? |
| Median Final Route Completion | Higher | Is progress robust rather than driven by a few good episodes? |
| Mean Time to Clean Success | Lower | When successful, how efficiently does it reach the goal? |

Interpret these together and by tier (Easy/Medium/Hard/Extreme). Example: a method that gains success by sharply increasing crashes has a real trade-off; do not hide it in a weighted mega-score.

Secondary diagnostics such as timeout rate, outcome breakdown, speed, episode return, and decision latency help explain *why* a method behaves as it does.

## 4. Fair-comparison rules

When comparing Agent A and Agent B, keep fixed:

- case manifest;
- environment seeds;
- information rights;
- action contract;
- episode/safety definitions;
- primary metric definitions;
- benchmark environment version.

Only the method and explicitly declared method-specific configuration should change.

## 5. Plan before run

Use `plan` before `run` when developing a new agent:

```powershell
python -m src.launcher.cli plan --mode SANDBOX --agent <agent_id> --tier Easy
```

The resolved plan shows what the platform will actually execute. Preflight then decides whether that plan is allowed.

A `BLOCKED` verdict is not something to bypass. Read the failed check and fix the cause.

## 6. Fixture agents and expected benchmark blocking

The current registry includes fixture agents for infrastructure testing. They are intentionally marked non-benchmark-eligible.

Therefore this is expected to fail preflight:

```powershell
python -m src.launcher.cli plan --mode VALIDATION --agent fixture_constant_continuous
```

Once a true research agent is implemented, tested, and registered with the appropriate eligibility, it can enter canonical suites.

## 7. Run outputs

Canonical run data is stored under `runs/<run_id>/` and includes, at a high level:

- experiment/provenance information;
- lifecycle state;
- per-episode outcomes and metrics;
- aggregate summary metrics;
- agent decision timing;
- integrity information;
- W&B synchronization state when used.

Do not edit scientific run outputs to make a run pass verification. If a run is invalid, rerun the experiment from a valid configuration.

## 8. W&B policy

W&B is optional tracking/visualization, not the sole scientific record. Local persistence is authoritative.

For ordinary SANDBOX development, the default can remain disabled. Benchmark modes may use offline tracking by default. Online mode requires the credential to be supplied in the environment; never commit credentials.

## 9. Workbench workflow

The GUI implements the same launcher rules:

```text
Experiment Setup
   → Resolve / Preflight
   → Run Monitor
   → persisted artifacts
   → Results integrity verification
```

Live monitor values are provisional. Trust benchmark metrics from the Results view only after the run is verified.
