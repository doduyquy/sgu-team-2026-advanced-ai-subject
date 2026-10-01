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
        # 1. Core resolver must reject NATIVE render mode for benchmark suites
        mock_reg = AgentRegistrationV1(
            agent_id="mock_benchmark_agent",
            benchmark_eligible=True,
            inference_stochasticity="deterministic"
        )
        reg_suite = AgentRegistryV1()
        reg_suite.register(mock_reg, lambda: None)

        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="mock_benchmark_agent", render_mode="NATIVE")
        with self.assertRaises(ValueError) as ctx:
            resolve_experiment_plan(req, reg_suite, self.project_root)
        self.assertIn("requires render_mode='OFF'", str(ctx.exception))

        # 2. Preflight check also rejects NATIVE render mode if present on plan
        valid_req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="mock_benchmark_agent", render_mode="OFF")
        valid_plan = resolve_experiment_plan(valid_req, reg_suite, self.project_root)
        bad_plan = dataclasses.replace(valid_plan, render_mode="NATIVE")
        rep = run_preflight(bad_plan, self.project_root, custom_git_provenance=self.clean_git, custom_environment_provenance=self.verified_env)
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


class TestExecutorGuaranteesAndScientificContracts(unittest.TestCase):
    """Verifies scientific execution contracts, signed rewards, and preflight enforcement."""
    def setUp(self):
        self.registry = build_default_agent_registry()
        self.project_root = Path(__file__).resolve().parent.parent

    def test_signed_negative_progress_reward_without_clamping(self):
        # Progress backwards: RC from 0.50 to 0.45
        last_rc = 0.50
        curr_rc = 0.45
        delta_route = curr_rc - last_rc
        self.assertAlmostEqual(delta_route, -0.05)

        reward_spec = RewardSpecV1()
        breakdown = reward_spec.compute_step_reward(
            delta_route_completion=delta_route,
            horizon_steps=1000
        )
        # Gate 4 progress component must be negative
        self.assertLess(breakdown.progress_reward, 0.0)
        self.assertAlmostEqual(breakdown.progress_reward, reward_spec.progress_weight * -0.05)

    def test_executor_blocks_invalid_plan_without_caller_preflight(self):
        # Plan for TEST with fixture agent (benchmark_eligible=False)
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_seeded_random", agent_seed=101)
        plan = resolve_experiment_plan(req, self.registry, self.project_root)

        factory = self.registry.get_factory("fixture_seeded_random")
        executor = ExperimentExecutor(plan=plan, agent_factory=factory, project_root=self.project_root)

        # Calling execute directly without preflight MUST raise PreflightBlockedError
        with self.assertRaises(PreflightBlockedError) as ctx:
            executor.execute()
        self.assertIn("benchmark_eligible=False", str(ctx.exception))

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
                return AgentDescriptor(agent_id="different_agent_id")

        req = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="fixture_constant_continuous", tier="Easy")
        plan = resolve_experiment_plan(req, self.registry, self.project_root)
        executor = ExperimentExecutor(plan=plan, agent_factory=lambda: MismatchedAgent(), project_root=self.project_root)

        # Execution must fail before any environment is created
        with self.assertRaises(RuntimeError) as ctx:
            executor.execute()
        self.assertIn("agent_id mismatch", str(ctx.exception))

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
        p_auto = resolve_experiment_plan(req_auto, self.registry, self.project_root)
        self.assertEqual(p_auto.wandb_mode, WandbMode.OFFLINE)

        # Explicit DISABLED remains DISABLED in TEST
        req_dis = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_seeded_random", wandb_mode=WandbMode.DISABLED)
        p_dis = resolve_experiment_plan(req_dis, self.registry, self.project_root)
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
        # Create a mock benchmark eligible agent for validation plan
        bm_agent = AgentRegistrationV1(agent_id="bm_val_agent", benchmark_eligible=True, inference_stochasticity="deterministic")
        reg = AgentRegistryV1()
        reg.register(bm_agent, lambda: None)

        req = LaunchRequestV1(mode=LauncherMode.VALIDATION, agent_id="bm_val_agent")
        plan = resolve_experiment_plan(req, reg, self.project_root)

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
        bm_agent = AgentRegistrationV1(agent_id="bm_test_agent", benchmark_eligible=True, inference_stochasticity="deterministic")
        reg = AgentRegistryV1()
        reg.register(bm_agent, lambda: None)

        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="bm_test_agent")
        plan = resolve_experiment_plan(req, reg, self.project_root)

        # Tamper traffic_density in case 5
        c5 = plan.resolved_cases[5]
        tampered_c5 = dataclasses.replace(c5, traffic_density=0.99)
        new_cases = plan.resolved_cases[:5] + (tampered_c5,) + plan.resolved_cases[6:]
        bad_plan = dataclasses.replace(plan, resolved_cases=new_cases)

        rep = run_preflight(bad_plan, self.project_root, custom_git_provenance={"git_worktree_dirty": False}, custom_environment_provenance={"metadrive_version": PINNED_METADRIVE_VERSION, "metadrive_commit": PINNED_METADRIVE_COMMIT})
        self.assertFalse(rep.can_execute)
        check = next(c for c in rep.checks if c.check_id == "test_manifest_deep_parity")
        self.assertEqual(check.status, "FAIL")


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


class TestBenchmarkFilterRejection(unittest.TestCase):
    """Blocker 4: Rejection of illegal benchmark filtering intent."""
    def setUp(self):
        self.registry = build_default_agent_registry()
        self.project_root = Path(__file__).resolve().parent.parent

    def test_validation_rejects_tier_selector(self):
        req = LaunchRequestV1(mode=LauncherMode.VALIDATION, agent_id="fixture_seeded_random", tier="Medium")
        with self.assertRaises(ValueError) as ctx:
            resolve_experiment_plan(req, self.registry, self.project_root)
        self.assertIn("Illegal case selector", str(ctx.exception))
        self.assertIn("VALIDATION", str(ctx.exception))

    def test_validation_rejects_sequence_selector(self):
        req = LaunchRequestV1(mode=LauncherMode.VALIDATION, agent_id="fixture_seeded_random", sequence="SCS")
        with self.assertRaises(ValueError) as ctx:
            resolve_experiment_plan(req, self.registry, self.project_root)
        self.assertIn("Illegal case selector", str(ctx.exception))

    def test_test_rejects_geom_seed_selector(self):
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_seeded_random", geometry_generation_seed=0)
        with self.assertRaises(ValueError) as ctx:
            resolve_experiment_plan(req, self.registry, self.project_root)
        self.assertIn("Illegal case selector", str(ctx.exception))
        self.assertIn("TEST", str(ctx.exception))

    def test_test_rejects_env_seed_selector(self):
        req = LaunchRequestV1(mode=LauncherMode.TEST, agent_id="fixture_seeded_random", environment_seed=9101)
        with self.assertRaises(ValueError) as ctx:
            resolve_experiment_plan(req, self.registry, self.project_root)
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
        registry = build_default_agent_registry()
        project_root = Path(__file__).resolve().parent.parent
        with self.assertRaises(ValueError) as ctx:
            resolve_experiment_plan(req, registry, project_root)
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
                return AgentDescriptor(agent_id="fixture_constant_continuous", action_adapter_id="continuous_box2_v1")
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


if __name__ == "__main__":
    unittest.main()
