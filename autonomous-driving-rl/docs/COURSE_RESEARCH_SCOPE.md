# Course research scope

The eight stages are a conceptual taxonomy of decision-making methods. They describe research possibilities, not eight mandatory implementation commitments. Platform V1 supplies the common scientific foundation beneath the taxonomy.

| Stage | Conceptual method | Current course role |
|---|---|---|
| 0 | Random / Naive | Immediate next algorithm milestone; canonical scientific agent is not implemented. |
| 1 | Rule / Heuristic | Practical core comparison after Stage 0. |
| 2 | Planning / Search | Optional research direction. |
| 3 | Simulation-Based Planning | Optional look-ahead/rollout research; using MetaDrive alone does not constitute this stage. |
| 4 | Learning from Data / Imitation | Optional data-driven research direction. |
| 5 | Model-Free RL | Practical core learning milestone. |
| 6 | Search + Learned Policy/Value | Optional combined search/learning direction. |
| 7 | Model-Based RL / World Models | Optional learned-dynamics/planning direction. |

The practical current priority is:

**Platform V1 → Stage 0 → Stage 1 → Stage 5 → controlled comparison.**

Platform V1 is complete; MapSuite V1 is packaged; Workbench is hardened. Stage 0 is outside this documentation and preview-export branch. Existing random/constant/discrete fixtures verify infrastructure and are not the Stage 0 scientific baseline. Stages 2/3/4/6/7 remain valid options when justified by course time, research questions and available resources.

## Controlled comparison and future extensions

Each implemented stage must use the same frozen scenario universe, splits, canonical cases, input rights, action adapters, lifecycle, reward and evaluation metrics. TRAIN supports development; VALIDATION supports tuning/model selection; TEST remains the final holdout. Compare eligible verified local runs with per-agent provenance and replicate seeds. Reward is a training signal; benchmark rankings use the locked evaluation metrics.

An algorithm may change its internal computation and training schedule. It may not read evaluator-only fields, gain hidden map/future state through previews, use TEST for tuning, bypass launcher/preflight, or claim fixture results as scientific evidence. A stage that requires new simulator access, information rights or benchmark capabilities must first propose a reviewed and versioned extension. The taxonomy alone authorizes none of those changes.

Read the [stage roadmap](STAGE_ROADMAP.md) for method context, [agent guide](AGENT_DEVELOPMENT_GUIDE.md) for integration, [boundaries](LEGACY_AND_BOUNDARIES.md) for authority, and [contract registry](PLATFORM_V1_CONTRACT_INDEX.md) for the frozen identities.
