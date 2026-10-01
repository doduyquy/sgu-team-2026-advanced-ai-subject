"""
Unit tests for Gate 7.5A: Research Platform V1 Launcher Core.

Pure unit test suite running in <0.50s without network, W&B subprocesses,
or physical MetaDrive simulator instantiations.
"""

import copy
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.launcher.cases import (
    LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256,
    LOCKED_TEST_CASE_MANIFEST_SHA256,
    LOCKED_VALIDATION_CASE_MANIFEST_SHA256,
    load_locked_geometry_block_sequence,
    load_manifest_csv,
    resolve_audit_cases,
    resolve_sandbox_case,
    resolve_test_cases,
    resolve_validation_cases,
)
from src.launcher.cli import (
    build_launch_request_from_args,
    create_parser,
    handle_agents,
    handle_cases,
    handle_plan,
)
from src.launcher.contracts import (
    GATE7_LOCKED_OBSERVABILITY_HASH,
    build_launcher_contract_core,
    compute_launcher_contract_sha256,
    compute_platform_execution_contract_sha256,
)
from src.launcher.events import LauncherEventType, LauncherEventV1
from src.launcher.executor import (
    ExperimentExecutor,
    build_metadrive_case_config,
    resolve_manifest_name_for_mode,
)
from src.launcher.models import (
    AgentRegistrationV1,
    LaunchRequestV1,
    LauncherMode,
    PreflightBlockedError,
    ResolvedCaseV1,
    ResolvedExperimentPlanV1,
    compute_resolved_plan_sha256,
    describe_plan,
    mode_to_run_kind,
)
from src.launcher.preflight import PreflightCheckV1, PreflightReportV1, run_preflight
from src.launcher.registry import (
    AgentRegistryV1,
    build_canonical_agent_registry,
    build_default_agent_registry,
)
from src.launcher.resolver import (
    DEFAULT_LAUNCHER_CONTRACT_HASH,
    DEFAULT_PLATFORM_EXECUTION_HASH,
    GATE5_LOCKED_BENCHMARK_HASH,
    GATE6_LOCKED_AGENT_HASH,
    GATE6_LOCKED_RUNTIME_HASH,
    GATE7_LOCKED_LOGGING_HASH,
    resolve_experiment_plan,
)
from src.platform import (
    PINNED_METADRIVE_COMMIT,
    PINNED_METADRIVE_VERSION,
    RewardSpecV1,
    RunKind,
    WandbMode,
    canonical_json_sha256,
)


class TestLauncherModelsAndEnums(unittest.TestCase):
    def test_launcher_modes_and_run_kind_mapping(self):
        self.assertEqual(mode_to_run_kind(LauncherMode.SANDBOX), RunKind.AUDIT)
        self.assertEqual(mode_to_run_kind(LauncherMode.AUDIT), RunKind.AUDIT)
        self.assertEqual(mode_to_run_kind(LauncherMode.VALIDATION), RunKind.VALIDATION_EVALUATION)
        self.assertEqual(mode_to_run_kind(LauncherMode.TEST), RunKind.TEST_EVALUATION)

    def test_launch_request_serialization(self):
        req = LaunchRequestV1(
            mode=LauncherMode.SANDBOX,
            agent_id="fixture_seeded_random",
            tier="Medium",
            agent_seed=101,
            render_mode="OFF",
            wandb_mode=WandbMode.DISABLED
        )
        d = req.to_dict()
        self.assertEqual(d["mode"], "SANDBOX")
        self.assertEqual(d["agent_id"], "fixture_seeded_random")
        self.assertEqual(d["tier"], "Medium")
        self.assertEqual(d["agent_seed"], 101)
        self.assertEqual(d["wandb_mode"], "DISABLED")

    def test_describe_plan_output(self):
        registry = build_default_agent_registry()
        req = LaunchRequestV1(
            mode=LauncherMode.SANDBOX,
            agent_id="fixture_seeded_random",
            tier="Easy",
            render_mode="OFF",
            wandb_mode=WandbMode.DISABLED
        )
        plan = resolve_experiment_plan(req, registry)
        desc = describe_plan(plan)
        self.assertIn("EXPERIMENT PLAN PREVIEW: SANDBOX", desc)
        self.assertIn("fixture_seeded_random", desc)
        self.assertIn("TRAIN only guaranteed", desc)
        self.assertIn("10.0 Hz", desc)


class TestAgentRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = build_default_agent_registry()

    def test_initial_fixture_agents_registered(self):
        agents = self.registry.list_all()
        agent_ids = [a.agent_id for a in agents]
        expected = [
            "fixture_constant_continuous",
            "fixture_discrete",
            "fixture_seeded_random",
            "fixture_stateful_counter"
        ]
        self.assertEqual(sorted(agent_ids), sorted(expected))

    def test_fixtures_not_benchmark_eligible(self):
        for a in self.registry.list_all():
            self.assertFalse(a.benchmark_eligible, f"Fixture agent {a.agent_id} must not be benchmark eligible!")
            self.assertTrue(a.sandbox_eligible)
            self.assertTrue(a.audit_eligible)

    def test_duplicate_registration_rejected(self):
        dup = AgentRegistrationV1(agent_id="fixture_seeded_random")
        with self.assertRaises(ValueError):
            self.registry.register(dup, lambda: None)

    def test_unknown_agent_lookup_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.registry.get("non_existent_agent")
        with self.assertRaises(KeyError):
            self.registry.get_factory("non_existent_agent")


class TestCaseResolutionAndHoldoutSafety(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parent.parent

    def test_sandbox_resolves_train_case_only(self):
        req = LaunchRequestV1(
            mode=LauncherMode.SANDBOX,
            agent_id="fixture_seeded_random",
            tier="Easy"
        )
        case = resolve_sandbox_case(req, self.project_root)
        self.assertEqual(case.split, "TRAIN")
        self.assertEqual(case.tier, "Easy")
        self.assertEqual(case.case_index, 1)
        self.assertEqual(case.protocol_order_index, 1)

    def test_sandbox_rejects_test_geometry_loudly(self):
        # In Gate-5 canonical manifest, canonical test geometries have known seeds (e.g. seed 11 or 12 for canonical TEST)
        # Load test manifest to pick a known TEST geometry
        test_p = self.project_root / "results" / "audits" / "evaluation_protocol" / "test_case_manifest.csv"
        test_recs = load_manifest_csv(test_p)
        test_seed = int(test_recs[0]["geometry_generation_seed"])
        test_seq = test_recs[0]["sequence"]

        req = LaunchRequestV1(
            mode=LauncherMode.SANDBOX,
            agent_id="fixture_seeded_random",
            sequence=test_seq,
            geometry_generation_seed=test_seed
        )
        with self.assertRaises(ValueError) as ctx:
            resolve_sandbox_case(req, self.project_root)
        self.assertIn("HOLDOUT VIOLATION", str(ctx.exception))

    def test_sandbox_rejects_validation_geometry_loudly(self):
        val_p = self.project_root / "results" / "audits" / "evaluation_protocol" / "validation_case_manifest.csv"
        val_recs = load_manifest_csv(val_p)
        val_seed = int(val_recs[0]["geometry_generation_seed"])
        val_seq = val_recs[0]["sequence"]

        req = LaunchRequestV1(
            mode=LauncherMode.SANDBOX,
            agent_id="fixture_seeded_random",
            sequence=val_seq,
            geometry_generation_seed=val_seed
        )
        with self.assertRaises(ValueError) as ctx:
            resolve_sandbox_case(req, self.project_root)
        self.assertIn("HOLDOUT VIOLATION", str(ctx.exception))

    def test_validation_exact_manifest_parity(self):
        cases = resolve_validation_cases(self.project_root)
        self.assertEqual(len(cases), 96)
        manifest_p = self.project_root / "results" / "audits" / "evaluation_protocol" / "validation_case_manifest.csv"
        manifest_recs = load_manifest_csv(manifest_p)

        for idx, (c, m) in enumerate(zip(cases, manifest_recs)):
            self.assertEqual(c.case_id, m["case_id"])
            self.assertEqual(c.protocol_order_index, int(m["protocol_order_index"]))
            self.assertEqual(c.split, "VALIDATION")
            self.assertEqual(c.geometry_generation_seed, int(m["geometry_generation_seed"]))
            self.assertEqual(c.environment_seed, int(m["environment_seed"]))
            self.assertEqual(c.geometry_sha256, m["geometry_sha256"])
            self.assertEqual(c.horizon_steps, int(m["horizon_steps"]))

    def test_test_exact_manifest_parity(self):
        cases = resolve_test_cases(self.project_root)
        self.assertEqual(len(cases), 60)
        manifest_p = self.project_root / "results" / "audits" / "evaluation_protocol" / "test_case_manifest.csv"
        manifest_recs = load_manifest_csv(manifest_p)

        for idx, (c, m) in enumerate(zip(cases, manifest_recs)):
            self.assertEqual(c.case_id, m["case_id"])
            self.assertEqual(c.protocol_order_index, int(m["protocol_order_index"]))
            self.assertEqual(c.split, "TEST")
            self.assertEqual(c.geometry_generation_seed, int(m["geometry_generation_seed"]))
            self.assertEqual(c.environment_seed, int(m["environment_seed"]))
            self.assertEqual(c.geometry_sha256, m["geometry_sha256"])
            self.assertEqual(c.horizon_steps, int(m["horizon_steps"]))


class TestPlanResolverAndHashDeterminism(unittest.TestCase):
    def setUp(self):
        self.registry = build_default_agent_registry()
        self.project_root = Path(__file__).resolve().parent.parent

    def test_plan_preview_has_zero_side_effects(self):
        runs_dir = self.project_root / "runs"
        existing_runs = set(runs_dir.iterdir()) if runs_dir.exists() else set()

        req = LaunchRequestV1(
            mode=LauncherMode.SANDBOX,
            agent_id="fixture_seeded_random",
            tier="Easy",
            agent_seed=101
        )
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        preflight = run_preflight(plan, self.project_root)

        # Assert no new run directory created
        current_runs = set(runs_dir.iterdir()) if runs_dir.exists() else set()
        self.assertEqual(existing_runs, current_runs)
        self.assertTrue(preflight.can_execute)

    def test_plan_hash_determinism(self):
        req1 = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Easy", agent_seed=101)
        req2 = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Easy", agent_seed=101)

        p1 = resolve_experiment_plan(req1, self.registry, self.project_root)
        p2 = resolve_experiment_plan(req2, self.registry, self.project_root)

        self.assertEqual(p1.resolved_plan_sha256, p2.resolved_plan_sha256)

    def test_plan_hash_mutation_sensitivity(self):
        req_base = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Easy", agent_seed=101)
        p_base = resolve_experiment_plan(req_base, self.registry, self.project_root)

        # Mutate agent seed
        req_seed = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Easy", agent_seed=202)
        p_seed = resolve_experiment_plan(req_seed, self.registry, self.project_root)
        self.assertNotEqual(p_base.resolved_plan_sha256, p_seed.resolved_plan_sha256)

        # Mutate tier
        req_tier = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Medium", agent_seed=101)
        p_tier = resolve_experiment_plan(req_tier, self.registry, self.project_root)
        self.assertNotEqual(p_base.resolved_plan_sha256, p_tier.resolved_plan_sha256)

        # Mutate agent
        req_agent = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy")
        p_agent = resolve_experiment_plan(req_agent, self.registry, self.project_root)
        self.assertNotEqual(p_base.resolved_plan_sha256, p_agent.resolved_plan_sha256)


class TestPreflightMatrix(unittest.TestCase):
    def setUp(self):
        self.registry = build_default_agent_registry()
        self.project_root = Path(__file__).resolve().parent.parent
        self.clean_git = {"git_commit_sha": "abc12345", "git_worktree_dirty": False}
        self.dirty_git = {"git_commit_sha": "abc12345", "git_worktree_dirty": True}
        self.verified_env = {
            "metadrive_version": PINNED_METADRIVE_VERSION,
            "metadrive_commit": PINNED_METADRIVE_COMMIT
        }
        self.unverified_env = {
            "metadrive_version": "0.4.2",
            "metadrive_commit": "unknown"
        }

    def test_valid_sandbox_fixture_passes(self):
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Easy", agent_seed=101)
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        rep = run_preflight(plan, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        self.assertTrue(rep.can_execute)
        self.assertEqual(rep.fail_count, 0)

    def test_stochastic_fixture_on_validation_fails(self):
        req = LaunchRequestV1(mode=LauncherMode.VALIDATION, agent_id="fixture_seeded_random", agent_seed=101)
        plan = resolve_experiment_plan(req, project_root=self.project_root)
        rep = run_preflight(plan, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "agent_benchmark_eligibility")
        self.assertEqual(check.status, "FAIL")

    def test_fixture_on_test_fails(self):
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_seeded_random", agent_seed=101)
        plan = resolve_experiment_plan(req, project_root=self.project_root)
        rep = run_preflight(plan, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "agent_benchmark_eligibility")
        self.assertEqual(check.status, "FAIL")

    def test_deterministic_fixture_on_validation_fails(self):
        req = LaunchRequestV1(mode=LauncherMode.VALIDATION, agent_id="fixture_constant_continuous")
        plan = resolve_experiment_plan(req, project_root=self.project_root)
        rep = run_preflight(plan, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "agent_benchmark_eligibility")
        self.assertEqual(check.status, "FAIL")

    def test_benchmark_render_native_fails(self):
        # 1. Core resolver must reject NATIVE render mode for benchmark suites
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous", render_mode="NATIVE")
        with self.assertRaises(ValueError) as ctx:
            resolve_experiment_plan(req, project_root=self.project_root)
        self.assertIn("requires render_mode='OFF'", str(ctx.exception))

        # 2. Preflight check also rejects NATIVE render mode if present on plan
        valid_req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous", render_mode="OFF")
        valid_plan = resolve_experiment_plan(valid_req, project_root=self.project_root)
        bad_plan = dataclasses.replace(valid_plan, render_mode="NATIVE")
        rep = run_preflight(bad_plan, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "benchmark_render_policy")
        self.assertEqual(check.status, "FAIL")

    def test_stochastic_benchmark_seed_validation(self):
        # Missing seed -> FAIL
        req_missing = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_seeded_random", agent_seed=None)
        p_missing = resolve_experiment_plan(req_missing, project_root=self.project_root)
        rep_missing = run_preflight(p_missing, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        self.assertFalse(rep_missing.can_execute)
        check_missing = next(c for c in rep_missing.checks if c.check_id == "stochastic_seed_policy")
        self.assertEqual(check_missing.status, "FAIL")

        # Invalid seed (999) -> FAIL
        req_invalid = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_seeded_random", agent_seed=999)
        p_invalid = resolve_experiment_plan(req_invalid, project_root=self.project_root)
        rep_invalid = run_preflight(p_invalid, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        self.assertFalse(rep_invalid.can_execute)
        check_invalid = next(c for c in rep_invalid.checks if c.check_id == "stochastic_seed_policy")
        self.assertEqual(check_invalid.status, "FAIL")

        # Valid declared replicate seed (101) -> PASS on stochastic_seed_policy
        req_valid = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_seeded_random", agent_seed=101)
        p_valid = resolve_experiment_plan(req_valid, project_root=self.project_root)
        rep_valid = run_preflight(p_valid, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        check_seed = next(c for c in rep_valid.checks if c.check_id == "stochastic_seed_policy")
        self.assertEqual(check_seed.status, "PASS")

    def test_deterministic_benchmark_seed_must_be_none(self):
        req_with_seed = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous", agent_seed=101)
        p = resolve_experiment_plan(req_with_seed, project_root=self.project_root)
        rep = run_preflight(p, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "deterministic_seed_policy")
        self.assertEqual(check.status, "FAIL")

        req_no_seed = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous", agent_seed=None)
        p_no_seed = resolve_experiment_plan(req_no_seed, project_root=self.project_root)
        rep_no_seed = run_preflight(p_no_seed, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        check_ok = next(c for c in rep_no_seed.checks if c.check_id == "deterministic_seed_policy")
        self.assertEqual(check_ok.status, "PASS")

    def test_dirty_worktree_on_benchmark_fails(self):
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous")
        p = resolve_experiment_plan(req, project_root=self.project_root)
        rep = run_preflight(p, self.project_root, custom_git_provenance=self.dirty_git, custom_environment_provenance=self.verified_env)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "git_cleanliness")
        self.assertEqual(check.status, "FAIL")

    def test_wandb_online_without_key_fails(self):
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", wandb_mode=WandbMode.ONLINE)
        p = resolve_experiment_plan(req, self.registry, self.project_root)

        with patch.dict("os.environ", {}, clear=True):
            rep = run_preflight(p, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
            self.assertFalse(rep.can_execute)
            check = next(c for c in rep.checks if c.check_id == "wandb_online_credential")
            self.assertEqual(check.status, "FAIL")


class TestLauncherCLI(unittest.TestCase):
    def test_argparse_creation(self):
        parser = create_parser()
        args = parser.parse_args(["plan", "--mode", "SANDBOX", "--agent", "fixture_seeded_random", "--tier", "Medium"])
        self.assertEqual(args.command, "plan")
        self.assertEqual(args.mode, "SANDBOX")
        self.assertEqual(args.agent, "fixture_seeded_random")
        self.assertEqual(args.tier, "Medium")

    def test_cli_request_builder(self):
        parser = create_parser()
        args = parser.parse_args(["plan", "--mode", "SANDBOX", "--agent", "fixture_seeded_random", "--tier", "Easy", "--agent-seed", "101"])
        req = build_launch_request_from_args(args)
        self.assertEqual(req.mode, LauncherMode.SANDBOX)
        self.assertEqual(req.agent_id, "fixture_seeded_random")
        self.assertEqual(req.tier, "Easy")
        self.assertEqual(req.agent_seed, 101)


class TestLauncherContractAndHashes(unittest.TestCase):
    def test_contract_dataclasses_and_enums_introspection(self):
        core = build_launcher_contract_core()
        expected_classes = [
            "AgentRegistrationV1",
            "LaunchRequestV1",
            "ResolvedCaseV1",
            "ResolvedExperimentPlanV1",
            "LauncherEventV1"
        ]
        for name in expected_classes:
            self.assertIn(name, core["runtime_dataclass_schemas"])

    def test_contract_hash_mutation_sensitivity(self):
        core_default = build_launcher_contract_core()
        h_default = compute_launcher_contract_sha256(core_default)

        # Mutate sandbox policy
        core_mut = build_launcher_contract_core(custom_sandbox_policy={"split_restriction": "MUTATED_SPLIT"})
        h_mut = compute_launcher_contract_sha256(core_mut)
        self.assertNotEqual(h_default, h_mut)

        # Mutate test policy
        core_test = build_launcher_contract_core(custom_test_policy={"expected_case_count": 999})
        h_test = compute_launcher_contract_sha256(core_test)
        self.assertNotEqual(h_default, h_test)

    def test_execution_hash_links_observability_and_launcher(self):
        h_launch = DEFAULT_LAUNCHER_CONTRACT_HASH
        h_exec = compute_platform_execution_contract_sha256(h_launch)
        self.assertEqual(h_exec, DEFAULT_PLATFORM_EXECUTION_HASH)


class TestExecutorGuaranteesAndScientificContracts(unittest.TestCase):
    """Verifies scientific execution contracts, signed rewards, and preflight enforcement."""
    def setUp(self):
        self.registry = build_default_agent_registry()
        self.project_root = Path(__file__).resolve().parent.parent

    def test_signed_negative_progress_reward_without_clamping(self):
        from src.launcher.executor import compute_step_route_progress_and_reward
        reward_spec = RewardSpecV1()

        # Progress backwards: RC from 0.50 to 0.45
        delta_route, new_rc, breakdown = compute_step_route_progress_and_reward(
            reward_spec=reward_spec,
            current_route_completion=0.45,
            last_route_completion=0.50,
            horizon_steps=1000
        )
        self.assertAlmostEqual(delta_route, -0.05)
        self.assertEqual(new_rc, 0.45)
        # Gate 4 progress component must be strictly negative
        self.assertLess(breakdown.progress_reward, 0.0)
        self.assertAlmostEqual(breakdown.progress_reward, reward_spec.progress_weight * -0.05)

        # Mutation regression test: if a helper clamped delta to 0.0, breakdown.progress_reward would be 0.0
        def clamped_delta_helper(curr, last):
            d = max(0.0, curr - last)
            return d, curr, reward_spec.compute_step_reward(delta_route_completion=d, horizon_steps=1000)

        clamped_d, _, clamped_bd = clamped_delta_helper(0.45, 0.50)
        self.assertEqual(clamped_d, 0.0)
        self.assertEqual(clamped_bd.progress_reward, 0.0)
        # Proves that production compute_step_route_progress_and_reward behaves differently from clamped helper
        self.assertNotEqual(delta_route, clamped_d)
        self.assertNotEqual(breakdown.progress_reward, clamped_bd.progress_reward)

    def test_executor_blocks_invalid_plan_without_caller_preflight(self):
        # Plan for TEST with fixture agent (benchmark_eligible=False)
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_seeded_random", agent_seed=101)
        plan = resolve_experiment_plan(req, project_root=self.project_root)

        executor = ExperimentExecutor(plan=plan, project_root=self.project_root)

        # Calling execute directly without preflight MUST raise PreflightBlockedError specifically for benchmark eligibility
        with self.assertRaises(PreflightBlockedError) as ctx:
            executor.execute()
        err_msg = str(ctx.exception)
        self.assertIn("benchmark_eligible=False", err_msg)
        # Invariant: Canonical executor internally-created canonical registry must NEVER be misclassified as external
        self.assertNotIn("canonical_external_registry_forbidden", err_msg)

    def test_canonical_executor_passes_none_registry_to_preflight(self):
        # Cross-layer proof: canonical executor passes registry=None to run_preflight()
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous")
        plan = resolve_experiment_plan(req, project_root=self.project_root)

        executor = ExperimentExecutor(plan=plan, registry=None, project_root=self.project_root)
        self.assertIsNotNone(executor.registry)

        with patch("src.launcher.executor.run_preflight") as mock_pf:
            mock_pf.side_effect = RuntimeError("Preflight intercept")
            try:
                executor.execute()
            except RuntimeError as e:
                if "Preflight intercept" not in str(e):
                    raise
            self.assertTrue(mock_pf.called)
            _, kwargs = mock_pf.call_args
            self.assertIsNone(kwargs.get("registry"), "Canonical executor must pass registry=None to run_preflight()")

    def test_noncanonical_executor_passes_development_registry_to_preflight(self):
        dev_reg = AgentRegistryV1()
        dev_reg.register(AgentRegistrationV1(agent_id="dev_agent", sandbox_eligible=True), lambda: None)
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="dev_agent", tier="Easy")
        plan = resolve_experiment_plan(req, registry=dev_reg, project_root=self.project_root)

        executor = ExperimentExecutor(plan=plan, registry=dev_reg, project_root=self.project_root)
        with patch("src.launcher.executor.run_preflight") as mock_pf:
            mock_pf.side_effect = RuntimeError("Preflight intercept")
            try:
                executor.execute()
            except RuntimeError as e:
                if "Preflight intercept" not in str(e):
                    raise
            self.assertTrue(mock_pf.called)
            _, kwargs = mock_pf.call_args
            self.assertIs(kwargs.get("registry"), dev_reg)

    def test_preflight_blocks_contract_hash_chain_mismatch(self):
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)

        # Mutate launcher_contract_sha256 in plan
        bad_plan = dataclasses.replace(plan, launcher_contract_sha256="corrupted_hash_12345")
        rep = run_preflight(bad_plan, self.project_root)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "contract_hash_chain")
        self.assertEqual(check.status, "FAIL")

    def test_ambiguous_sandbox_seed_rejected(self):
        # In Gate-5 universe, seed 0 exists in multiple sequences (SCS, SCXCS, etc.)
        # Requesting seed 0 without sequence must be rejected as ambiguous
        req_ambiguous = LaunchRequestV1(
            mode=LauncherMode.SANDBOX,
            agent_id="fixture_seeded_random",
            geometry_generation_seed=0
        )
        with self.assertRaises(ValueError) as ctx:
            resolve_sandbox_case(req_ambiguous, self.project_root)
        self.assertIn("Ambiguous geometry request", str(ctx.exception))

    def test_tier_selector_respected_with_explicit_seed(self):
        # Seed 2 with tier Medium must resolve to Medium, never Easy
        req_med = LaunchRequestV1(
            mode=LauncherMode.SANDBOX,
            agent_id="fixture_seeded_random",
            tier="Medium",
            geometry_generation_seed=2,
            sequence="SCTCS"
        )
        case = resolve_sandbox_case(req_med, self.project_root)
        self.assertEqual(case.tier, "Medium")
        self.assertEqual(case.sequence, "SCTCS")
        self.assertEqual(case.geometry_generation_seed, 2)

    def test_runtime_agent_descriptor_mismatch_rejected(self):
        # Mock factory that returns agent with mismatched agent_id
        class MismatchedAgent:
            @property
            def descriptor(self):
                from src.platform import AgentDescriptor
                return AgentDescriptor(agent_id="different_agent_id", method_family="fixture")
            def close(self): pass

        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        executor = ExperimentExecutor(plan=plan, agent_factory=lambda: MismatchedAgent(), project_root=self.project_root)

        # Execution must fail before any environment is created
        with self.assertRaises(RuntimeError) as ctx:
            executor.execute()
        self.assertIn("agent_id mismatch", str(ctx.exception))

    def test_runtime_method_family_mismatch_rejected(self):
        class WrongFamilyAgent:
            @property
            def descriptor(self):
                from src.platform import AgentDescriptor
                return AgentDescriptor(
                    agent_id="fixture_constant_continuous",
                    agent_version="1.0.0",
                    input_profile_id="STATE_DECISION_V1",
                    action_adapter_id="continuous_box2_v1",
                    inference_stochasticity="deterministic",
                    stateful_within_episode=False,
                    method_family="WRONG_FAMILY"
                )
            def reset(self, ctx, agent_seed=None): pass
            def act(self, inp): pass
            def close(self): pass

        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        executor = ExperimentExecutor(plan=plan, agent_factory=lambda: WrongFamilyAgent(), project_root=self.project_root)

        with self.assertRaises(RuntimeError) as ctx:
            executor.execute()
        self.assertIn("method_family mismatch", str(ctx.exception))

    def test_agent_lifecycle_one_instance_per_run(self):
        calls = {"factory": 0, "reset": 0, "close": 0}
        class TrackingAgent:
            @property
            def descriptor(self):
                from src.platform import AgentDescriptor
                return AgentDescriptor(agent_id="fixture_constant_continuous", action_adapter_id="continuous_box2_v1", method_family="fixture")
            def reset(self, ctx, agent_seed=None):
                calls["reset"] += 1
            def act(self, inp):
                from src.platform import AgentDecision
                return AgentDecision(action_payload=[0.0, 0.0])
            def close(self):
                calls["close"] += 1

        def tracking_factory():
            calls["factory"] += 1
            return TrackingAgent()

        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        temp_d = tempfile.TemporaryDirectory()
        try:
            executor = ExperimentExecutor(plan=plan, agent_factory=tracking_factory, runs_root=Path(temp_d.name), project_root=self.project_root)
            with patch("src.launcher.executor.build_agent_input") as mock_input, \
                 patch("src.launcher.executor.MetaDriveEnv") as mock_env_cls:
                mock_env = mock_env_cls.return_value
                mock_env.reset.return_value = (np.zeros(259, dtype=np.float32), {})
                mock_env.step.return_value = (np.zeros(259, dtype=np.float32), 0.0, True, False, {"route_completion": 1.0, "arrive_dest": True})
                executor.execute()

            self.assertEqual(calls["factory"], 1)
            self.assertEqual(calls["reset"], 1)
            self.assertEqual(calls["close"], 1)
        finally:
            temp_d.cleanup()

    def test_explicit_metadrive_control_config(self):
        case = ResolvedCaseV1(
            case_id="test_case", case_index=1, protocol_order_index=1, split="TRAIN",
            tier="Easy", sequence="SCS", geometry_generation_seed=0, geometry_sha256="hash",
            environment_seed=101, traffic_density=0.1, horizon_steps=1000
        )
        cfg = build_metadrive_case_config(case, blocks=[], render_mode="OFF")
        self.assertEqual(cfg["physics_world_step_size"], 0.02)
        self.assertEqual(cfg["decision_repeat"], 5)
        self.assertFalse(cfg["truncate_as_terminate"])
        self.assertEqual(cfg["horizon"], 1000)
        self.assertEqual(cfg["start_seed"], 101)

    def test_manifest_name_mapping(self):
        self.assertEqual(resolve_manifest_name_for_mode(LauncherMode.SANDBOX), "geometry_split_manifest.csv")
        self.assertEqual(resolve_manifest_name_for_mode(LauncherMode.AUDIT), "geometry_split_manifest.csv")
        self.assertEqual(resolve_manifest_name_for_mode(LauncherMode.VALIDATION), "validation_case_manifest.csv")
        self.assertEqual(resolve_manifest_name_for_mode(LauncherMode.TEST), "test_case_manifest.csv")

    def test_wandb_mode_auto_vs_explicit_disabled(self):
        # AUTO (None) defaults to OFFLINE for TEST
        req_auto = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_seeded_random", wandb_mode=None)
        p_auto = resolve_experiment_plan(req_auto, project_root=self.project_root)
        self.assertEqual(p_auto.wandb_mode, WandbMode.OFFLINE)

        # Explicit DISABLED remains DISABLED in TEST
        req_dis = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_seeded_random", wandb_mode=WandbMode.DISABLED)
        p_dis = resolve_experiment_plan(req_dis, project_root=self.project_root)
        self.assertEqual(p_dis.wandb_mode, WandbMode.DISABLED)

        # AUTO defaults to DISABLED for SANDBOX
        req_sb = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", wandb_mode=None)
        p_sb = resolve_experiment_plan(req_sb, self.registry, self.project_root)
        self.assertEqual(p_sb.wandb_mode, WandbMode.DISABLED)


class TestPlanDeepIntegrityAndTampering(unittest.TestCase):
    """Blocker 1: Preflight recomputation and deep row-by-row manifest verification."""
    def setUp(self):
        self.registry = build_default_agent_registry()
        self.project_root = Path(__file__).resolve().parent.parent

    def test_valid_untampered_plan_passes_deep_preflight(self):
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        rep = run_preflight(plan, self.project_root)
        check_hash = next(c for c in rep.checks if c.check_id == "plan_hash_integrity")
        self.assertEqual(check_hash.status, "PASS")
        check_struct = next(c for c in rep.checks if c.check_id == "plan_structural_consistency")
        self.assertEqual(check_struct.status, "PASS")

    def test_stale_or_tampered_plan_hash_fails_preflight(self):
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        tampered_plan = dataclasses.replace(plan, resolved_plan_sha256="corrupted_hash_value")
        rep = run_preflight(tampered_plan, self.project_root)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "plan_hash_integrity")
        self.assertEqual(check.status, "FAIL")

    def test_tampered_run_kind_fails_preflight(self):
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        # Tamper run_kind to TEST_EVALUATION
        bad_plan = dataclasses.replace(plan, run_kind=RunKind.TEST_EVALUATION)
        rep = run_preflight(bad_plan, self.project_root)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "plan_structural_consistency")
        self.assertEqual(check.status, "FAIL")

    def test_tampered_canonical_run_fails_preflight(self):
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        bad_plan = dataclasses.replace(plan, canonical_run=True)
        rep = run_preflight(bad_plan, self.project_root)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "plan_structural_consistency")
        self.assertEqual(check.status, "FAIL")

    def test_tampered_protocol_scope_fails_preflight(self):
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        bad_plan = dataclasses.replace(plan, protocol_scope="CORRUPTED_SCOPE")
        rep = run_preflight(bad_plan, self.project_root)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "plan_structural_consistency")
        self.assertEqual(check.status, "FAIL")

    def test_tampered_control_frequency_fails_preflight(self):
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        bad_plan = dataclasses.replace(plan, control_frequency_hz=20.0)
        rep = run_preflight(bad_plan, self.project_root)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "plan_structural_consistency")
        self.assertEqual(check.status, "FAIL")

    def test_tampered_validation_cases_row_diff_fails_preflight(self):
        req = LaunchRequestV1(mode=LauncherMode.VALIDATION, agent_id="fixture_constant_continuous")
        plan = resolve_experiment_plan(req, project_root=self.project_root)

        # Tamper environment_seed in case 0
        c0 = plan.resolved_cases[0]
        tampered_c0 = dataclasses.replace(c0, environment_seed=9999)
        new_cases = (tampered_c0,) + plan.resolved_cases[1:]
        bad_plan = dataclasses.replace(plan, resolved_cases=new_cases)

        rep = run_preflight(bad_plan, self.project_root, custom_git_provenance={"git_worktree_dirty": False}, custom_environment_provenance={"metadrive_version": PINNED_METADRIVE_VERSION, "metadrive_commit": PINNED_METADRIVE_COMMIT})
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "validation_manifest_deep_parity")
        self.assertEqual(check.status, "FAIL")

    def test_tampered_test_cases_traffic_density_fails_preflight(self):
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous")
        plan = resolve_experiment_plan(req, project_root=self.project_root)

        # Tamper traffic_density in case 5
        c5 = plan.resolved_cases[5]
        tampered_c5 = dataclasses.replace(c5, traffic_density=0.99)
        new_cases = plan.resolved_cases[:5] + (tampered_c5,) + plan.resolved_cases[6:]
        bad_plan = dataclasses.replace(plan, resolved_cases=new_cases)

        rep = run_preflight(bad_plan, self.project_root, custom_git_provenance={"git_worktree_dirty": False}, custom_environment_provenance={"metadrive_version": PINNED_METADRIVE_VERSION, "metadrive_commit": PINNED_METADRIVE_COMMIT})
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "test_manifest_deep_parity")
        self.assertEqual(check.status, "FAIL")

    def test_tiny_traffic_density_mutation_fails_preflight(self):
        req = LaunchRequestV1(mode=LauncherMode.VALIDATION, agent_id="fixture_constant_continuous")
        plan = resolve_experiment_plan(req, project_root=self.project_root)

        # Mutate traffic_density by a tiny +1e-6
        c0 = plan.resolved_cases[0]
        tampered_c0 = dataclasses.replace(c0, traffic_density=float(c0.traffic_density) + 1e-6)
        new_cases = (tampered_c0,) + plan.resolved_cases[1:]

        # Recompute plan hash so failure demonstrably comes from deep manifest parity, NOT stale hash!
        temp_dict = plan.to_dict()
        temp_dict["resolved_cases"] = [c.to_dict() for c in new_cases]
        recomputed_hash = compute_resolved_plan_sha256(temp_dict)

        bad_plan = dataclasses.replace(plan, resolved_cases=new_cases, resolved_plan_sha256=recomputed_hash)
        rep = run_preflight(bad_plan, self.project_root, custom_git_provenance={"git_worktree_dirty": False}, custom_environment_provenance={"metadrive_version": PINNED_METADRIVE_VERSION, "metadrive_commit": PINNED_METADRIVE_COMMIT})
        self.assertFalse(rep.can_execute)
        # Demonstrably passes hash integrity but fails deep manifest parity specifically
        check_hash = next(c for c in rep.checks if c.check_id == "plan_hash_integrity")
        self.assertEqual(check_hash.status, "PASS")
        check_parity = next(c for c in rep.checks if c.check_id == "validation_manifest_deep_parity")
        self.assertEqual(check_parity.status, "FAIL")
        self.assertIn("traffic_density mismatch", check_parity.message)


class TestCustomRunIdSanitization(unittest.TestCase):
    """Blocker 3: custom_run_id validation and containment check."""
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_custom_run_ids_accepted(self):
        from src.launcher.models import validate_custom_run_id
        self.assertEqual(validate_custom_run_id("run_valid_01", self.runs_root), "run_valid_01")
        self.assertEqual(validate_custom_run_id("run-abc-123.test", self.runs_root), "run-abc-123.test")
        self.assertIsNone(validate_custom_run_id(None, self.runs_root))

    def test_directory_traversal_run_ids_rejected(self):
        from src.launcher.models import validate_custom_run_id
        with self.assertRaises(ValueError):
            validate_custom_run_id("../escape_dir", self.runs_root)
        with self.assertRaises(ValueError):
            validate_custom_run_id("..\\escape_dir", self.runs_root)
        with self.assertRaises(ValueError):
            validate_custom_run_id("..", self.runs_root)
        with self.assertRaises(ValueError):
            validate_custom_run_id(".", self.runs_root)

    def test_path_separators_and_absolute_paths_rejected(self):
        from src.launcher.models import validate_custom_run_id
        with self.assertRaises(ValueError):
            validate_custom_run_id("/absolute/path", self.runs_root)
        with self.assertRaises(ValueError):
            validate_custom_run_id("C:\\Windows", self.runs_root)
        with self.assertRaises(ValueError):
            validate_custom_run_id("nested/dir/id", self.runs_root)

    def test_empty_or_invalid_grammar_rejected(self):
        from src.launcher.models import validate_custom_run_id
        with self.assertRaises(ValueError):
            validate_custom_run_id("", self.runs_root)
        with self.assertRaises(ValueError):
            validate_custom_run_id("   ", self.runs_root)
        with self.assertRaises(ValueError):
            validate_custom_run_id("run id with spaces", self.runs_root)
        with self.assertRaises(ValueError):
            validate_custom_run_id("run*id$special", self.runs_root)

    def test_leading_and_trailing_whitespace_run_ids_rejected(self):
        from src.launcher.models import validate_custom_run_id
        with self.assertRaises(ValueError):
            validate_custom_run_id(" run_01", self.runs_root)
        with self.assertRaises(ValueError):
            validate_custom_run_id("run_01 ", self.runs_root)
        with self.assertRaises(ValueError):
            validate_custom_run_id("\trun_01", self.runs_root)


class TestBenchmarkFilterRejection(unittest.TestCase):
    """Blocker 4: Rejection of illegal benchmark filtering intent."""
    def setUp(self):
        self.registry = build_default_agent_registry()
        self.project_root = Path(__file__).resolve().parent.parent

    def test_validation_rejects_tier_selector(self):
        req = LaunchRequestV1(mode=LauncherMode.VALIDATION, agent_id="fixture_seeded_random", tier="Medium")
        with self.assertRaises(ValueError) as ctx:
            resolve_experiment_plan(req, project_root=self.project_root)
        self.assertIn("Illegal case selector", str(ctx.exception))
        self.assertIn("VALIDATION", str(ctx.exception))

    def test_validation_rejects_sequence_selector(self):
        req = LaunchRequestV1(mode=LauncherMode.VALIDATION, agent_id="fixture_seeded_random", sequence="SCS")
        with self.assertRaises(ValueError) as ctx:
            resolve_experiment_plan(req, project_root=self.project_root)
        self.assertIn("Illegal case selector", str(ctx.exception))

    def test_test_rejects_geom_seed_selector(self):
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_seeded_random", geometry_generation_seed=0)
        with self.assertRaises(ValueError) as ctx:
            resolve_experiment_plan(req, project_root=self.project_root)
        self.assertIn("Illegal case selector", str(ctx.exception))
        self.assertIn("TEST", str(ctx.exception))

    def test_test_rejects_env_seed_selector(self):
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_seeded_random", environment_seed=9101)
        with self.assertRaises(ValueError) as ctx:
            resolve_experiment_plan(req, project_root=self.project_root)
        self.assertIn("Illegal case selector", str(ctx.exception))


class TestRenderModeCoreValidation(unittest.TestCase):
    """Blocker 5: Core-level render mode validation."""
    def test_invalid_render_mode_string_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_seeded_random", render_mode="GUI_WINDOW")
        self.assertIn("Invalid render_mode", str(ctx.exception))

    def test_benchmark_native_render_rejected_by_resolver(self):
        # Even if lowercase native is supplied to LaunchRequest
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_seeded_random", render_mode="NATIVE")
        project_root = Path(__file__).resolve().parent.parent
        with self.assertRaises(ValueError) as ctx:
            resolve_experiment_plan(req, project_root=project_root)
        self.assertIn("requires render_mode='OFF'", str(ctx.exception))


class TestFailureLifecycleSemantics(unittest.TestCase):
    """Blocker 2: Technical failures must be FAILED, never INTERRUPTED."""
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.temp_dir.name)
        self.project_root = Path(__file__).resolve().parent.parent
        self.registry = build_default_agent_registry()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_technical_exception_in_act_transitions_to_failed(self):
        # Create an agent whose act() raises an exception
        class CrashingAgent:
            @property
            def descriptor(self):
                from src.platform import AgentDescriptor
                return AgentDescriptor(agent_id="fixture_constant_continuous", action_adapter_id="continuous_box2_v1", method_family="fixture")
            def reset(self, ctx, agent_seed=None): pass
            def act(self, inp): raise RuntimeError("Simulated agent neural network crash")
            def close(self): pass

        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy", runs_root=self.runs_root)
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        executor = ExperimentExecutor(plan=plan, agent_factory=lambda: CrashingAgent(), runs_root=self.runs_root, project_root=self.project_root)

        with self.assertRaises(RuntimeError) as ctx:
            executor.execute()
        self.assertIn("Simulated agent neural network crash", str(ctx.exception))

        # Check durable run_state.json: MUST be FAILED, never INTERRUPTED
        run_dirs = list(self.runs_root.iterdir())
        self.assertEqual(len(run_dirs), 1)
        with open(run_dirs[0] / "run_state.json") as f:
            st = json.load(f)
        self.assertEqual(st["status"], "FAILED")
        self.assertEqual(st["failure_category"], "TECHNICAL_EXECUTION_ERROR")

    def test_pre_existing_failed_logger_not_overwritten_by_interrupted(self):
        from src.platform import LocalExperimentLogger, ExperimentRunConfig, RunKind
        config = ExperimentRunConfig(
            run_kind=RunKind.AUDIT, benchmark_contract_sha256="d", agent_contract_sha256="d",
            platform_runtime_contract_sha256="d", logging_contract_sha256="d", platform_observability_contract_sha256="d",
            agent_id="a", agent_version="1", input_profile_id="STATE_DECISION_V1", action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic", stateful_within_episode=False, allow_dirty_worktree_override=True
        )
        logger = LocalExperimentLogger(config, runs_root=self.runs_root)
        # Explicitly fail the logger (e.g. from local log write error)
        logger.mark_failed("LOCAL_LOG_WRITE_ERROR", "Disk full")
        self.assertEqual(logger.status.value, "FAILED")

        # Now attempt mark_interrupted() on the already-failed logger
        logger.mark_interrupted("User hit Ctrl+C")
        # Invariant: Status must remain FAILED, never overwritten as INTERRUPTED
        self.assertEqual(logger.status.value, "FAILED")
        with open(logger.run_state_path) as f:
            st = json.load(f)
        self.assertEqual(st["status"], "FAILED")
        self.assertEqual(st["failure_category"], "LOCAL_LOG_WRITE_ERROR")

    def test_keyboard_interrupt_transitions_to_interrupted(self):
        class InterruptingAgent:
            @property
            def descriptor(self):
                from src.platform import AgentDescriptor
                return AgentDescriptor(agent_id="fixture_constant_continuous", action_adapter_id="continuous_box2_v1", method_family="fixture")
            def reset(self, ctx, agent_seed=None): pass
            def act(self, inp): raise KeyboardInterrupt()
            def close(self): pass

        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy", runs_root=self.runs_root)
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        executor = ExperimentExecutor(plan=plan, agent_factory=lambda: InterruptingAgent(), runs_root=self.runs_root, project_root=self.project_root)

        report = executor.execute()
        self.assertEqual(report.status, "INTERRUPTED")

        run_dirs = list(self.runs_root.iterdir())
        with open(run_dirs[0] / "run_state.json") as f:
            st = json.load(f)
        self.assertEqual(st["status"], "INTERRUPTED")
        self.assertEqual(st["failure_category"], "INTERRUPTED")

    def test_action_adapter_failure_transitions_to_failed(self):
        class OutOfBoundsActionAgent:
            @property
            def descriptor(self):
                from src.platform import AgentDescriptor
                return AgentDescriptor(agent_id="fixture_constant_continuous", action_adapter_id="continuous_box2_v1", method_family="fixture")
            def reset(self, ctx, agent_seed=None): pass
            def act(self, inp):
                from src.platform import AgentDecision
                return AgentDecision(action_payload=[10.0, 0.0])  # Out of bounds!
            def close(self): pass

        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy", runs_root=self.runs_root)
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        executor = ExperimentExecutor(plan=plan, agent_factory=lambda: OutOfBoundsActionAgent(), runs_root=self.runs_root, project_root=self.project_root)

        with self.assertRaises(RuntimeError) as ctx:
            executor.execute()
        self.assertIn("violates canonical bounds", str(ctx.exception))

        run_dirs = list(self.runs_root.iterdir())
        with open(run_dirs[0] / "run_state.json") as f:
            st = json.load(f)
        self.assertEqual(st["status"], "FAILED")
        self.assertEqual(st["failure_category"], "TECHNICAL_EXECUTION_ERROR")
        self.assertNotEqual(st["status"], "COMPLETE")
        self.assertNotEqual(st["status"], "INTERRUPTED")

    def test_actual_logger_write_failure_remains_failed_and_not_overwritten(self):
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy", runs_root=self.runs_root)
        plan = resolve_experiment_plan(req, self.registry, self.project_root)

        # Injected failure: make episodes.csv append write raise IOError
        original_open = open
        def failing_open(path, mode="r", *args, **kwargs):
            if "episodes.csv" in str(path) and "a" in mode:
                raise IOError("Simulated disk I/O write failure in episodes.csv")
            return original_open(path, mode, *args, **kwargs)

        executor = ExperimentExecutor(
            plan=plan,
            agent_factory=self.registry.get_factory("fixture_constant_continuous"),
            runs_root=self.runs_root,
            project_root=self.project_root
        )

        with patch("builtins.open", side_effect=failing_open):
            with self.assertRaises(RuntimeError):
                executor.execute()

        run_dirs = list(self.runs_root.iterdir())
        self.assertEqual(len(run_dirs), 1)
        with open(run_dirs[0] / "run_state.json") as f:
            st = json.load(f)
        # Must be FAILED with category LOCAL_LOG_WRITE_ERROR, NEVER overwritten as INTERRUPTED!
        self.assertEqual(st["status"], "FAILED")
        self.assertEqual(st["failure_category"], "LOCAL_LOG_WRITE_ERROR")
        self.assertNotEqual(st["status"], "COMPLETE")
        self.assertNotEqual(st["status"], "INTERRUPTED")


class TestCaseInspectionTampering(unittest.TestCase):
    def test_cli_case_inspection_tampered_manifest_rejected(self):
        from src.launcher.cli import handle_cases, create_parser
        parser = create_parser()
        args = parser.parse_args(["cases", "--split", "VALIDATION"])

        real_project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temp_d:
            temp_root = Path(temp_d)
            manifest_dir = temp_root / "results" / "audits" / "evaluation_protocol"
            manifest_dir.mkdir(parents=True, exist_ok=True)

            src_manifest = real_project_root / "results" / "audits" / "evaluation_protocol" / "validation_case_manifest.csv"
            dest_manifest = manifest_dir / "validation_case_manifest.csv"

            # Read real manifest content and tamper with it
            content = src_manifest.read_text(encoding="utf-8")
            tampered_content = content.replace("pool_candidate", "tampered_candidate", 1)
            self.assertNotEqual(content, tampered_content)
            dest_manifest.write_text(tampered_content, encoding="utf-8")

            args.project_root = temp_root
            with self.assertRaises(ValueError) as ctx:
                handle_cases(args)
            self.assertIn("Gate-5 manifest tampering detected", str(ctx.exception))


class TestLauncherCoreBlockerRegressions(unittest.TestCase):
    """Verifies all Gate 7.5A audit blocker resolutions and scientific invariants."""
    def setUp(self):
        self.registry = build_canonical_agent_registry()
        self.project_root = Path(__file__).resolve().parent.parent
        self.clean_git = {"git_worktree_dirty": False, "git_commit_sha": "abc12345"}
        self.dirty_git = {"git_worktree_dirty": True, "git_commit_sha": "abc12345"}
        self.verified_env = {"metadrive_version": PINNED_METADRIVE_VERSION, "metadrive_commit": PINNED_METADRIVE_COMMIT}

    def test_preflight_authoritative_registry_unregistered_agent_rejected(self):
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)

        bad_reg = dataclasses.replace(plan.agent_registration, agent_id="unregistered_fake_agent")
        bad_plan = dataclasses.replace(plan, agent_registration=bad_reg)

        rep = run_preflight(bad_plan, self.project_root, registry=self.registry)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "agent_registry_binding")
        self.assertEqual(check.status, "FAIL")
        self.assertIn("not present in authoritative AgentRegistryV1", check.message)

    def test_preflight_authoritative_registry_tampered_metadata_rejected(self):
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)

        # Tamper with benchmark_eligible flag on plan
        tampered_reg = dataclasses.replace(plan.agent_registration, benchmark_eligible=True)
        bad_plan = dataclasses.replace(plan, agent_registration=tampered_reg)

        rep = run_preflight(bad_plan, self.project_root, registry=self.registry)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "agent_registry_binding")
        self.assertEqual(check.status, "FAIL")
        self.assertIn("metadata tampering detected", check.message)

    def test_preflight_agent_descriptor_projection_parity_rejected(self):
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)

        # Tamper with agent descriptor on plan
        tampered_desc = dataclasses.replace(plan.agent_descriptor, method_family="corrupted_family")
        bad_plan = dataclasses.replace(plan, agent_descriptor=tampered_desc)

        rep = run_preflight(bad_plan, self.project_root, registry=self.registry)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "agent_descriptor_projection_parity")
        self.assertEqual(check.status, "FAIL")
        self.assertIn("method_family", check.message)

    def test_executor_resolves_factory_from_registry_when_omitted(self):
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)

        executor = ExperimentExecutor(plan=plan, registry=self.registry, project_root=self.project_root)
        self.assertIsNotNone(executor.agent_factory)
        agent = executor.agent_factory()
        self.assertEqual(agent.descriptor.agent_id, "fixture_constant_continuous")

    def test_executor_blocks_unauthorized_custom_factory_on_canonical_run(self):
        req = LaunchRequestV1(mode=LauncherMode.VALIDATION, agent_id="fixture_constant_continuous")
        plan = resolve_experiment_plan(req, project_root=self.project_root)

        with self.assertRaises(ValueError) as ctx:
            ExperimentExecutor(plan=plan, agent_factory=lambda: None, project_root=self.project_root)
        self.assertIn("Explicit agent_factory override is strictly forbidden", str(ctx.exception))

    def test_clean_sandbox_audit_runs_persist_noncanonical_in_run_state(self):
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)

        with tempfile.TemporaryDirectory() as temp_d:
            runs_dir = Path(temp_d)
            executor = ExperimentExecutor(
                plan=plan,
                registry=self.registry,
                runs_root=runs_dir,
                project_root=self.project_root
            )
            with patch("src.launcher.executor.build_agent_input"), \
                 patch("src.launcher.executor.MetaDriveEnv") as mock_env_cls:
                mock_env = mock_env_cls.return_value
                mock_env.reset.return_value = (np.zeros(259, dtype=np.float32), {})
                mock_env.step.return_value = (np.zeros(259, dtype=np.float32), 0.0, True, False, {"route_completion": 1.0, "arrive_dest": True})
                executor.execute()

            run_dirs = list(runs_dir.iterdir())
            self.assertEqual(len(run_dirs), 1)
            with open(run_dirs[0] / "run_state.json", "r", encoding="utf-8") as f:
                st = json.load(f)
            self.assertFalse(st["canonical_run"], "Sandbox runs must always persist canonical_run=False")

    def test_resolved_cases_tuple_immutability(self):
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        self.assertIsInstance(plan.resolved_cases, tuple)
        with self.assertRaises(TypeError):
            plan.resolved_cases[0] = None  # Tuples cannot be assigned

    def test_custom_run_id_whitespace_rejected(self):
        from src.launcher.models import validate_custom_run_id
        with tempfile.TemporaryDirectory() as temp_d:
            with self.assertRaises(ValueError):
                validate_custom_run_id(" run_with_space ", Path(temp_d))
            with self.assertRaises(ValueError):
                validate_custom_run_id("run_with_newline\n", Path(temp_d))

    def test_canonical_resolution_external_registry_rejected(self):
        # R1: Canonical resolution + external custom registry -> rejected
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous")
        custom_reg = AgentRegistryV1()
        with self.assertRaises(ValueError) as ctx:
            resolve_experiment_plan(req, registry=custom_reg, project_root=self.project_root)
        self.assertIn("External registry is forbidden for canonical benchmark resolution", str(ctx.exception))

    def test_canonical_resolution_external_registry_identical_contents_rejected(self):
        # R2: Canonical resolution + external registry with identical canonical contents -> rejected
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous")
        identical_reg = build_canonical_agent_registry()
        with self.assertRaises(ValueError) as ctx:
            resolve_experiment_plan(req, registry=identical_reg, project_root=self.project_root)
        self.assertIn("External registry is forbidden for canonical benchmark resolution", str(ctx.exception))

    def test_canonical_resolution_mutated_existing_fixture_rejected(self):
        # R3 & C1: Mutated existing fixture registry rejected at resolution
        base_reg = self.registry.get("fixture_seeded_random")
        forged_reg = dataclasses.replace(base_reg, benchmark_eligible=True)
        custom_reg = AgentRegistryV1()
        for a in self.registry.list_all():
            if a.agent_id != "fixture_seeded_random":
                custom_reg.register(a, self.registry.get_factory(a.agent_id))
            else:
                custom_reg.register(forged_reg, self.registry.get_factory(a.agent_id))

        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_seeded_random", agent_seed=101)
        with self.assertRaises(ValueError) as ctx:
            resolve_experiment_plan(req, registry=custom_reg, project_root=self.project_root)
        self.assertIn("External registry is forbidden for canonical benchmark resolution", str(ctx.exception))

    def test_canonical_resolution_new_custom_agent_rejected(self):
        # R4 & C2: Completely new custom benchmark agent rejected at resolution
        custom_reg = AgentRegistryV1()
        custom_reg.register(
            AgentRegistrationV1(agent_id="custom_fake_benchmark_agent", benchmark_eligible=True),
            lambda: None
        )
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="custom_fake_benchmark_agent")
        with self.assertRaises(ValueError) as ctx:
            resolve_experiment_plan(req, registry=custom_reg, project_root=self.project_root)
        self.assertIn("External registry is forbidden for canonical benchmark resolution", str(ctx.exception))

    def test_canonical_resolution_unknown_agent_rejected_by_canonical_membership(self):
        # R5: Unknown agent ID without external registry rejected by canonical membership
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="unknown_fake_benchmark_agent")
        with self.assertRaises(KeyError) as ctx:
            resolve_experiment_plan(req, project_root=self.project_root)
        self.assertIn("does not exist in canonical platform registry", str(ctx.exception))

    def test_noncanonical_sandbox_and_audit_custom_registry_allowed(self):
        # R6 & R7: Noncanonical SANDBOX and AUDIT allow custom development registry
        dev_reg = AgentRegistryV1()
        dev_reg.register(
            AgentRegistrationV1(agent_id="custom_dev_agent", sandbox_eligible=True, audit_eligible=True),
            lambda: None
        )
        p_sb = resolve_experiment_plan(
            LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="custom_dev_agent", tier="Easy"),
            registry=dev_reg,
            project_root=self.project_root
        )
        self.assertEqual(p_sb.agent_registration.agent_id, "custom_dev_agent")

        p_aud = resolve_experiment_plan(
            LaunchRequestV1(mode=LauncherMode.AUDIT, agent_id="custom_dev_agent"),
            registry=dev_reg,
            project_root=self.project_root
        )
        self.assertEqual(p_aud.agent_registration.agent_id, "custom_dev_agent")

    def test_canonical_manually_forged_unknown_agent_plan_fails_preflight(self):
        # R8 & C3: Manually forged canonical plan with unknown agent and valid recomputed hash fails preflight
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous")
        valid_plan = resolve_experiment_plan(req, project_root=self.project_root)

        forged_reg = dataclasses.replace(
            valid_plan.agent_registration,
            agent_id="custom_fake_benchmark_agent",
            benchmark_eligible=True
        )
        temp_dict = valid_plan.to_dict()
        temp_dict["agent_registration"] = forged_reg.to_dict()
        recomputed_hash = compute_resolved_plan_sha256(temp_dict)

        forged_plan = dataclasses.replace(valid_plan, agent_registration=forged_reg, resolved_plan_sha256=recomputed_hash)

        rep = run_preflight(
            forged_plan,
            self.project_root,
            custom_git_provenance=self.clean_git,
            custom_environment_provenance=self.verified_env
        )
        self.assertFalse(rep.can_execute)
        fail_ids = [c.check_id for c in rep.checks if c.status == "FAIL"]
        self.assertIn("canonical_registry_agent_membership", fail_ids)

    def test_canonical_preflight_external_registry_rejected(self):
        # R9 & C4: Canonical preflight with external registry argument produces blocking check
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous")
        plan = resolve_experiment_plan(req, project_root=self.project_root)

        rep = run_preflight(
            plan,
            self.project_root,
            custom_git_provenance=self.clean_git,
            custom_environment_provenance=self.verified_env,
            registry=AgentRegistryV1()
        )
        self.assertFalse(rep.can_execute)
        fail_ids = [c.check_id for c in rep.checks if c.status == "FAIL"]
        self.assertIn("canonical_external_registry_forbidden", fail_ids)

    def test_canonical_executor_external_registry_rejected(self):
        # R10 & C5: Canonical ExperimentExecutor with external registry rejected with ValueError
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous")
        plan = resolve_experiment_plan(req, project_root=self.project_root)

        with self.assertRaises(ValueError) as ctx:
            ExperimentExecutor(plan=plan, registry=AgentRegistryV1(), project_root=self.project_root)
        self.assertIn("External registry is forbidden for canonical benchmark execution", str(ctx.exception))

    def test_canonical_executor_external_factory_rejected(self):
        # R11: Canonical ExperimentExecutor with external agent_factory rejected with ValueError
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous")
        plan = resolve_experiment_plan(req, project_root=self.project_root)

        with self.assertRaises(ValueError) as ctx:
            ExperimentExecutor(plan=plan, agent_factory=lambda: None, project_root=self.project_root)
        self.assertIn("Explicit agent_factory override is strictly forbidden", str(ctx.exception))

    def test_factory_identity_substitution_rejected(self):
        # R12: Unauthorized post-construction factory substitution rejected with factory identity mismatch
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous", wandb_mode=WandbMode.DISABLED)
        plan = resolve_experiment_plan(req, project_root=self.project_root)

        executor = ExperimentExecutor(plan=plan, project_root=self.project_root)
        object.__setattr__(executor, "agent_factory", lambda: None)

        with patch("src.launcher.executor.run_preflight") as mock_pf, \
             patch("src.platform.experiment_logging.capture_git_provenance", return_value={"git_worktree_dirty": False, "git_commit_sha": "abc12345"}), \
             patch("src.platform.experiment_logging.capture_environment_provenance", return_value={"metadrive_version": PINNED_METADRIVE_VERSION, "metadrive_commit": PINNED_METADRIVE_COMMIT}):
            mock_pf.return_value = PreflightReportV1(can_execute=True, fail_count=0, warning_count=0, pass_count=10, summary_verdict="PASSED", checks=[])
            with self.assertRaises(RuntimeError) as ctx:
                executor.execute()
            self.assertIn("Agent factory identity mismatch", str(ctx.exception))

    def test_same_descriptor_different_runtime_implementation_rejected(self):
        # R13: Same canonical factory identity + different runtime class + identical AgentDescriptor rejected
        class ImposterAgent:
            @property
            def descriptor(self):
                from src.platform import AgentDescriptor
                return AgentDescriptor(
                    agent_id="fixture_constant_continuous",
                    agent_version="1.0.0",
                    input_profile_id="STATE_DECISION_V1",
                    action_adapter_id="continuous_box2_v1",
                    inference_stochasticity="deterministic",
                    stateful_within_episode=False,
                    method_family="fixture"
                )
            def reset(self, ctx, agent_seed=None): pass
            def act(self, inp): pass
            def close(self): pass

        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous", wandb_mode=WandbMode.DISABLED)
        plan = resolve_experiment_plan(req, project_root=self.project_root)

        executor = ExperimentExecutor(plan=plan, project_root=self.project_root)

        with patch("src.launcher.registry.DeterministicConstantFixtureAgent", ImposterAgent), \
             patch("src.launcher.executor.run_preflight") as mock_pf, \
             patch("src.platform.experiment_logging.capture_git_provenance", return_value={"git_worktree_dirty": False, "git_commit_sha": "abc12345"}), \
             patch("src.platform.experiment_logging.capture_environment_provenance", return_value={"metadrive_version": PINNED_METADRIVE_VERSION, "metadrive_commit": PINNED_METADRIVE_COMMIT}):
            mock_pf.return_value = PreflightReportV1(can_execute=True, fail_count=0, warning_count=0, pass_count=10, summary_verdict="PASSED", checks=[])
            with self.assertRaises(RuntimeError) as ctx:
                executor.execute()
            self.assertIn("Agent runtime implementation mismatch", str(ctx.exception))

    def test_canonical_registry_fingerprint_deterministic_and_sensitive(self):
        # R14-R17: Canonical registry SHA determinism, 14-field sensitivity, factory_ref sensitivity, description neutrality
        from src.launcher.registry import compute_canonical_registry_sha256
        h1 = compute_canonical_registry_sha256(self.registry)
        h2 = compute_canonical_registry_sha256(self.registry)
        self.assertEqual(h1, h2)

        # Mutate benchmark_eligible on an entry (R15)
        custom_reg = AgentRegistryV1()
        for a in self.registry.list_all():
            if a.agent_id == "fixture_constant_continuous":
                mut_a = dataclasses.replace(a, benchmark_eligible=True)
                custom_reg.register(mut_a, self.registry.get_factory(a.agent_id))
            else:
                custom_reg.register(a, self.registry.get_factory(a.agent_id))
        h_mut_eligible = compute_canonical_registry_sha256(custom_reg)
        self.assertNotEqual(h1, h_mut_eligible)

        # Mutate factory_ref on an entry (R16)
        custom_reg_f = AgentRegistryV1()
        def _other_factory(): return None
        for a in self.registry.list_all():
            if a.agent_id == "fixture_constant_continuous":
                custom_reg_f.register(a, _other_factory)
            else:
                custom_reg_f.register(a, self.registry.get_factory(a.agent_id))
        h_mut_f = compute_canonical_registry_sha256(custom_reg_f)
        self.assertNotEqual(h1, h_mut_f)

        # Mutate description only on an entry (R17)
        custom_reg_desc = AgentRegistryV1()
        for a in self.registry.list_all():
            if a.agent_id == "fixture_constant_continuous":
                mut_a = dataclasses.replace(a, description="Cosmetic description only")
                custom_reg_desc.register(mut_a, self.registry.get_factory(a.agent_id))
            else:
                custom_reg_desc.register(a, self.registry.get_factory(a.agent_id))
        h_desc = compute_canonical_registry_sha256(custom_reg_desc)
        self.assertEqual(h1, h_desc)

    def test_plan_hash_14_scientific_fields_sensitivity_and_description_exclusion(self):
        # R18 & R21: Plan hash sensitivity across all 14 scientific fields and description neutrality
        from src.launcher.models import SCIENTIFIC_REGISTRATION_FIELDS
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        h0 = plan.resolved_plan_sha256

        mutations = {
            "agent_id": "mutated_id",
            "agent_version": "2.0.0",
            "stage_label": "STAGE_99",
            "method_family": "mutated_family",
            "purpose": "MUTATED_PURPOSE",
            "implementation_ref": "mutated.impl:Ref",
            "input_profile_id": "MUTATED_PROFILE",
            "action_adapter_id": "mutated_adapter",
            "inference_stochasticity": "stochastic",
            "stateful_within_episode": True,
            "benchmark_eligible": True,
            "sandbox_eligible": False,
            "audit_eligible": False,
            "requires_checkpoint": True,
        }
        for field_name in SCIENTIFIC_REGISTRATION_FIELDS:
            mut_val = mutations[field_name]
            mut_reg = dataclasses.replace(plan.agent_registration, **{field_name: mut_val})
            mut_plan = dataclasses.replace(plan, agent_registration=mut_reg)
            h_mut = compute_resolved_plan_sha256(mut_plan.to_dict())
            self.assertNotEqual(h0, h_mut, f"Field '{field_name}' did not alter resolved plan hash!")

        # Verify description does NOT alter the plan hash
        desc_reg = dataclasses.replace(plan.agent_registration, description="Brand new cosmetic description.")
        desc_plan = dataclasses.replace(plan, agent_registration=desc_reg)
        h_desc = compute_resolved_plan_sha256(desc_plan.to_dict())
        self.assertEqual(h0, h_desc, "Cosmetic description altered resolved plan hash!")

    def test_plan_hash_canonical_registry_sha_and_factory_ref_sensitivity(self):
        # R19 & R20: Plan hash sensitivity to canonical_agent_registry_sha256 and factory_ref
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        h0 = plan.resolved_plan_sha256

        # Mutate canonical_agent_registry_sha256
        p_dict1 = plan.to_dict()
        p_dict1["canonical_agent_registry_sha256"] = "mutated_registry_hash_abc123"
        self.assertNotEqual(h0, compute_resolved_plan_sha256(p_dict1))

        # Mutate factory_ref
        p_dict2 = plan.to_dict()
        p_dict2["factory_ref"] = "src.other.module:other_factory"
        self.assertNotEqual(h0, compute_resolved_plan_sha256(p_dict2))

    def test_sandbox_execution_persists_noncanonical(self):
        # R22: Sandbox successful execution persists canonical false
        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        with tempfile.TemporaryDirectory() as td:
            ex = ExperimentExecutor(plan=plan, registry=self.registry, runs_root=Path(td), project_root=self.project_root)
            with patch("src.launcher.executor.build_agent_input"), patch("src.launcher.executor.MetaDriveEnv") as mock_env_cls:
                m = mock_env_cls.return_value
                m.reset.return_value = (np.zeros(259, dtype=np.float32), {})
                m.step.return_value = (np.zeros(259, dtype=np.float32), 0.0, True, False, {"route_completion": 1.0, "arrive_dest": True})
                ex.execute()
            rdir = list(Path(td).iterdir())[0]
            with open(rdir / "run_state.json", "r", encoding="utf-8") as f:
                st = json.load(f)
            self.assertFalse(st["canonical_run"])

    def test_audit_execution_persists_noncanonical(self):
        # R23: Audit successful execution persists canonical false
        req = LaunchRequestV1(mode=LauncherMode.AUDIT, agent_id="fixture_constant_continuous")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        with tempfile.TemporaryDirectory() as td:
            ex = ExperimentExecutor(plan=plan, registry=self.registry, runs_root=Path(td), project_root=self.project_root)
            with patch("src.launcher.executor.build_agent_input"), patch("src.launcher.executor.MetaDriveEnv") as mock_env_cls:
                m = mock_env_cls.return_value
                m.reset.return_value = (np.zeros(259, dtype=np.float32), {})
                m.step.return_value = (np.zeros(259, dtype=np.float32), 0.0, True, False, {"route_completion": 1.0, "arrive_dest": True})
                ex.execute()
            rdir = list(Path(td).iterdir())[0]
            with open(rdir / "run_state.json", "r", encoding="utf-8") as f:
                st = json.load(f)
            self.assertFalse(st["canonical_run"])

    def test_validation_plan_is_canonical_but_fixture_is_benchmark_blocked(self):
        # R24 & R26: Validation plan semantics is canonical true, but fixture is benchmark blocked
        req = LaunchRequestV1(mode=LauncherMode.VALIDATION, agent_id="fixture_constant_continuous")
        plan = resolve_experiment_plan(req, project_root=self.project_root)
        self.assertTrue(plan.canonical_run)
        rep = run_preflight(plan, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        self.assertFalse(rep.can_execute)
        chk = next(c for c in rep.checks if c.check_id == "agent_benchmark_eligibility")
        self.assertEqual(chk.status, "FAIL")

    def test_test_plan_is_canonical_but_fixture_is_benchmark_blocked(self):
        # R25 & R27: Test plan semantics is canonical true, but fixture is benchmark blocked
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous")
        plan = resolve_experiment_plan(req, project_root=self.project_root)
        self.assertTrue(plan.canonical_run)
        rep = run_preflight(plan, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        self.assertFalse(rep.can_execute)
        chk = next(c for c in rep.checks if c.check_id == "agent_benchmark_eligibility")
        self.assertEqual(chk.status, "FAIL")

    def test_canonical_dirty_git_is_blocked(self):
        # R28: Dirty canonical benchmark preflight fails git cleanliness
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous")
        plan = resolve_experiment_plan(req, project_root=self.project_root)
        rep = run_preflight(plan, self.project_root, custom_git_provenance=self.dirty_git, custom_environment_provenance=self.verified_env)
        self.assertFalse(rep.can_execute)
        chk = next(c for c in rep.checks if c.check_id == "git_cleanliness")
        self.assertEqual(chk.status, "FAIL")

    def test_canonical_unverified_metadrive_is_blocked(self):
        # R29: Unverified canonical benchmark preflight fails exact MetaDrive pin
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_constant_continuous")
        plan = resolve_experiment_plan(req, project_root=self.project_root)
        rep = run_preflight(plan, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance={"metadrive_version": "0.1.0", "metadrive_commit": "wrong"})
        self.assertFalse(rep.can_execute)
        chk = next(c for c in rep.checks if c.check_id == "metadrive_exact_pin")
        self.assertEqual(chk.status, "FAIL")


if __name__ == "__main__":
    unittest.main()
