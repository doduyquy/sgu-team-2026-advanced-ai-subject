"""Research baseline semantics and launcher integration, independent of rendering."""

import builtins
import unittest
from unittest.mock import patch

import numpy as np

from src.agents import Stage0RandomNaiveAgent
from src.launcher.executor import create_native_minimap, read_vehicle_speed_kmh
from src.launcher.models import LaunchRequestV1, LauncherMode
from src.launcher.preflight import run_preflight
from src.launcher.registry import build_canonical_agent_registry
from src.launcher.resolver import resolve_experiment_plan
from src.platform import (
    AgentInputV1, AgentPublicEpisodeContext, CoreObservationV1, InputProfileId,
    TaskContextV1, TrafficContextV1, get_action_adapter,
    PINNED_METADRIVE_COMMIT, PINNED_METADRIVE_VERSION,
)


class TestStage0RandomNaive(unittest.TestCase):
    def setUp(self):
        self.context = AgentPublicEpisodeContext(action_adapter_id="discrete9_lowbranch_v1")
        self.input = AgentInputV1(
            profile_id=InputProfileId.STATE_DECISION_V1,
            core_observation=CoreObservationV1(np.zeros(259, dtype=np.float32)),
            traffic_context=TrafficContextV1.empty(), task_context=TaskContextV1.empty(), step_index=0,
        )
        self.agent = Stage0RandomNaiveAgent()

    def tearDown(self):
        self.agent.close()

    def sequence(self, seed, count=500):
        self.agent.reset(self.context, agent_seed=seed)
        return [self.agent.act(self.input).to_dict() for _ in range(count)]

    def test_reset_is_required_and_close_invalidates_policy(self):
        with self.assertRaises(RuntimeError):
            self.agent.act(self.input)
        self.agent.reset(self.context, agent_seed=101)
        self.agent.act(self.input)
        self.agent.close()
        with self.assertRaises(RuntimeError):
            self.agent.act(self.input)
        self.agent.reset(self.context, agent_seed=101)
        self.assertEqual(self.agent.act(self.input).diagnostics["step_counter"], 1)

    def test_same_seed_restarts_actions_and_diagnostics(self):
        first = self.sequence(101)
        self.sequence(202, 75)
        self.assertEqual(first, self.sequence(101))
        self.assertEqual(first[0]["diagnostics"]["step_counter"], 1)

    def test_different_seeds_produce_different_actions(self):
        self.assertNotEqual(
            [d["action_payload"] for d in self.sequence(101)],
            [d["action_payload"] for d in self.sequence(202)],
        )

    def test_initial_action_and_resampling_durations(self):
        decisions = self.sequence(101, 1000)
        self.assertEqual([d["action_payload"] for d in decisions[:20]], [7] * 20)
        self.assertFalse(any(d["diagnostics"]["action_resample"] for d in decisions[:20]))
        resamples = [i for i, d in enumerate(decisions) if d["diagnostics"]["action_resample"]]
        self.assertEqual(resamples[0], 20)
        self.assertTrue(all(20 <= b - a <= 40 for a, b in zip(resamples, resamples[1:])))
        for i, decision in enumerate(decisions):
            self.assertEqual(decision["diagnostics"]["step_counter"], i + 1)
            if i and not decision["diagnostics"]["action_resample"]:
                self.assertEqual(decision["action_payload"], decisions[i - 1]["action_payload"])

    def test_action_mapping_is_valid_and_initial_action_is_throttle(self):
        adapter = get_action_adapter(self.agent.descriptor.action_adapter_id)
        for decision in self.sequence(303, 1000):
            action = decision["action_payload"]
            self.assertIsInstance(action, int)
            self.assertTrue(0 <= action <= 8)
            adapter.to_canonical(action)
        initial = adapter.to_canonical(7)
        self.assertEqual((initial.steering, initial.throttle_brake), (0.0, 0.6))

    def test_no_evaluator_or_observation_access_is_required(self):
        class ForbiddenInput:
            def __getattribute__(self, name):
                raise AssertionError(f"Random baseline accessed input: {name}")
        self.agent.reset(self.context, agent_seed=101)
        for _ in range(100):
            self.agent.act(ForbiddenInput())

    def test_descriptor_and_registration_match(self):
        registry = build_canonical_agent_registry()
        registration = registry.get(self.agent.descriptor.agent_id)
        for name, value in self.agent.descriptor.to_dict().items():
            self.assertEqual(value, getattr(registration, name))
        self.assertFalse(registration.benchmark_eligible)
        self.assertEqual(registration.stage_label, "STAGE_0")
        self.assertEqual(registration.purpose, "RESEARCH_BASELINE")

    def test_sandbox_plan_passes_and_benchmark_modes_remain_blocked(self):
        git = {"git_commit_sha": "test-commit", "git_worktree_dirty": False}
        environment = {"metadrive_version": PINNED_METADRIVE_VERSION,
                       "metadrive_commit": PINNED_METADRIVE_COMMIT}
        for mode in (LauncherMode.SANDBOX, LauncherMode.VALIDATION, LauncherMode.TEST):
            request = LaunchRequestV1(mode=mode, agent_id=self.agent.descriptor.agent_id, agent_seed=101)
            plan = resolve_experiment_plan(request)
            report = run_preflight(plan, custom_git_provenance=git,
                                   custom_environment_provenance=environment)
            if mode == LauncherMode.SANDBOX:
                self.assertTrue(report.can_execute)
                self.assertEqual(plan.resolved_cases[0].split, "TRAIN")
            else:
                self.assertFalse(report.can_execute)
                self.assertIn("agent_benchmark_eligibility", [c.check_id for c in report.checks if c.status == "FAIL"])


class TestOptionalNativeMinimap(unittest.TestCase):
    def test_missing_overlay_is_reported_and_native_window_remains_usable(self):
        missing = ModuleNotFoundError("missing overlay", name="src.visualization")
        original_import = builtins.__import__
        def import_without_overlay(name, *args, **kwargs):
            if name == "src.visualization.minimap_overlay":
                raise missing
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=import_without_overlay):
            with self.assertWarnsRegex(RuntimeWarning, "Optional minimap"):
                self.assertIsNone(create_native_minimap(object()))

    def test_missing_dependency_inside_overlay_is_not_hidden(self):
        missing = ModuleNotFoundError("missing dependency", name="unrelated_dependency")
        with patch("builtins.__import__", side_effect=missing):
            with self.assertRaises(ModuleNotFoundError):
                create_native_minimap(object())


class TestSimulatorSpeedAudit(unittest.TestCase):
    def test_speed_uses_the_pinned_metadrive_property(self):
        class Vehicle:
            speed_km_h = 36.0
            speed_kmh = 0.0
        self.assertEqual(read_vehicle_speed_kmh(Vehicle()), 36.0)

    def test_missing_or_nonfinite_speed_is_not_logged_as_zero(self):
        with self.assertRaises(AttributeError):
            read_vehicle_speed_kmh(object())
        class Vehicle:
            speed_km_h = float("nan")
        with self.assertRaises(ValueError):
            read_vehicle_speed_kmh(Vehicle())


if __name__ == "__main__":
    unittest.main()
