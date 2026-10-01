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

from src.launcher.cases import (
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
from src.launcher.models import (
    AgentRegistrationV1,
    LaunchRequestV1,
    LauncherMode,
    ResolvedCaseV1,
    ResolvedExperimentPlanV1,
    compute_resolved_plan_sha256,
    describe_plan,
    mode_to_run_kind,
)
from src.launcher.preflight import PreflightCheckV1, PreflightReportV1, run_preflight
from src.launcher.registry import AgentRegistryV1, build_default_agent_registry
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

    def test_fixture_on_validation_fails(self):
        req = LaunchRequestV1(mode=LauncherMode.VALIDATION, agent_id="fixture_seeded_random", agent_seed=101)
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        rep = run_preflight(plan, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "agent_benchmark_eligibility")
        self.assertEqual(check.status, "FAIL")

    def test_fixture_on_test_fails(self):
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_seeded_random", agent_seed=101)
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        rep = run_preflight(plan, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "agent_benchmark_eligibility")
        self.assertEqual(check.status, "FAIL")

    def test_benchmark_render_native_fails(self):
        # Register a mock benchmark eligible agent for preflight testing
        mock_reg = AgentRegistrationV1(
            agent_id="mock_benchmark_agent",
            benchmark_eligible=True,
            inference_stochasticity="deterministic"
        )
        reg_suite = AgentRegistryV1()
        reg_suite.register(mock_reg, lambda: None)

        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="mock_benchmark_agent", render_mode="NATIVE")
        plan = resolve_experiment_plan(req, reg_suite, self.project_root)
        rep = run_preflight(plan, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "benchmark_render_policy")
        self.assertEqual(check.status, "FAIL")

    def test_stochastic_benchmark_seed_validation(self):
        stoch_reg = AgentRegistrationV1(
            agent_id="mock_stoch_benchmark",
            benchmark_eligible=True,
            inference_stochasticity="stochastic"
        )
        reg_suite = AgentRegistryV1()
        reg_suite.register(stoch_reg, lambda: None)

        # Missing seed -> FAIL
        req_missing = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="mock_stoch_benchmark", agent_seed=None)
        p_missing = resolve_experiment_plan(req_missing, reg_suite, self.project_root)
        rep_missing = run_preflight(p_missing, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        self.assertFalse(rep_missing.can_execute)

        # Invalid seed (999) -> FAIL
        req_invalid = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="mock_stoch_benchmark", agent_seed=999)
        p_invalid = resolve_experiment_plan(req_invalid, reg_suite, self.project_root)
        rep_invalid = run_preflight(p_invalid, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        self.assertFalse(rep_invalid.can_execute)

        # Valid declared replicate seed (101) -> PASS
        req_valid = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="mock_stoch_benchmark", agent_seed=101)
        p_valid = resolve_experiment_plan(req_valid, reg_suite, self.project_root)
        rep_valid = run_preflight(p_valid, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        check_seed = next(c for c in rep_valid.checks if c.check_id == "stochastic_seed_policy")
        self.assertEqual(check_seed.status, "PASS")

    def test_deterministic_benchmark_seed_must_be_none(self):
        det_reg = AgentRegistrationV1(
            agent_id="mock_det_benchmark",
            benchmark_eligible=True,
            inference_stochasticity="deterministic"
        )
        reg_suite = AgentRegistryV1()
        reg_suite.register(det_reg, lambda: None)

        req_with_seed = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="mock_det_benchmark", agent_seed=101)
        p = resolve_experiment_plan(req_with_seed, reg_suite, self.project_root)
        rep = run_preflight(p, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "deterministic_seed_policy")
        self.assertEqual(check.status, "FAIL")

    def test_dirty_worktree_on_benchmark_fails(self):
        det_reg = AgentRegistrationV1(agent_id="mock_det_benchmark", benchmark_eligible=True, inference_stochasticity="deterministic")
        reg_suite = AgentRegistryV1()
        reg_suite.register(det_reg, lambda: None)

        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="mock_det_benchmark")
        p = resolve_experiment_plan(req, reg_suite, self.project_root)
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


if __name__ == "__main__":
    unittest.main()
