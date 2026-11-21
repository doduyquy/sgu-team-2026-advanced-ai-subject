# Stage 5 P2-H — Next Decision

**P2-H outcome:** `P2H_INCONCLUSIVE`.

The terminal-aware sampler passed unit and integration checks, but neither online training arm produced a clean-success or safety terminal in 10,000 steps. Arm B therefore ran the uniform fallback for all 2.304 million critic draws. Its deterministic policy and Arm A both failed all six full diagnostic episodes at 10k, with `OUT_OF_ROAD`; neither showed clean driving. The 5k short-probe movement in B did not persist to full-episode competence.

The experiment provides no evidence that terminal-aware replay improves learning because it had zero eligible sample exposure. It also does not establish that terminal-aware replay cannot help when such transitions exist. The A/B result is further non-causal because the treatment used an independent actor/temperature minibatch while the control shared one uniform batch across losses.

**Stop condition:** do not extend these runs or tune priority parameters. Keep both candidates as development evidence only; neither is benchmark eligible.

**Single next decision:** if Stage 5 work resumes, preregister one objective-level comparison that isolates the reward–entropy–critic update balance, with identical minibatch coupling between arms and a driving-based stop rule. Do not combine it with replay prioritization or activation changes. P2-H does not implement or start that work.
