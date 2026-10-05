# Results and Audit Evidence

This tracked directory contains committed platform audit evidence and, in the future, selected lightweight research summaries that are useful to the team/report.

`results/audits/` is the machine-derived evidence used to verify Platform V1 contracts and infrastructure.

Raw experiment runs should **not** be copied here by default. The authoritative runtime layout is `runs/<run_id>/`, which is local and git-ignored except for the placeholder file.

When course algorithms begin producing benchmark results, commit only intentional summary tables/figures needed for collaboration or reporting. Never edit a run's persisted metrics to manufacture a desired result; rerun from a valid configuration instead.
