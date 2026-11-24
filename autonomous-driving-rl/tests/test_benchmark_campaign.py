"""Fixed protocols, queue lifecycle, result trust, and real isolated-worker coverage."""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from src.launcher.cases import get_default_project_root, resolve_validation_cases
from src.launcher.registry import build_canonical_agent_registry
from src.platform import PINNED_METADRIVE_COMMIT, PINNED_METADRIVE_VERSION, WandbMode
from src.workbench.benchmark_campaign import (
    build_evaluation_campaign, inspect_campaign_results, load_evaluation_profiles,
)
from src.workbench.campaign_controller import CampaignController
from src.workbench.main_window import MainWindow
from src.workbench.process_runner import WorkbenchProcessRunner
from src.launcher.models import LaunchRequestV1, LauncherMode
from src.workbench.results_repository import IntegrityDisplayStatus
from src.workbench.widgets.run_monitor_widget import RunMonitorWidget
from src.launcher.events import LauncherEventType, LauncherEventV1


class FakeRunner:
    def __init__(self):
        self.is_running = False
        self.starts = []

    def start_operation(self, operation, request):
        if self.is_running:
            raise RuntimeError("Two concurrent workers")
        self.is_running = True
        self.starts.append((operation, request))


class TestBenchmarkCampaign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.root = get_default_project_root()
        cls.profiles = load_evaluation_profiles(cls.root)
        cls.agent = build_canonical_agent_registry().get("stage0_random_naive")

    def campaign(self, profile=0, **changes):
        return build_evaluation_campaign(self.profiles[profile], [replace(self.agent, **changes)], show_demo=False)

    def controller(self):
        runner = FakeRunner()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return CampaignController(runner, Path(temp.name)), runner

    def finish_plan_job(self, controller, runner, allowed=True):
        controller.preflight_received({"report": {"can_execute": allowed, "checks": []}})
        controller.worker_done({"operation": "PLAN", "success": True})
        runner.is_running = False
        controller.process_finished(0)

    def test_fixed_schedule_has_full_suites_and_no_case_overrides(self):
        for profile, count in ((0, 96), (1, 60)):
            campaign = self.campaign(profile)
            self.assertEqual([j.request.agent_seed for j in campaign.jobs], [101, 202, 303])
            self.assertEqual(sum(j.case_count for j in campaign.jobs), count * 3)
            for job in campaign.jobs:
                request = job.request
                self.assertIsNone(request.tier)
                self.assertIsNone(request.sequence)
                self.assertIsNone(request.geometry_generation_seed)
                self.assertIsNone(request.environment_seed)
                self.assertEqual(request.render_mode, "OFF")
                self.assertIs(request.wandb_mode, WandbMode.DISABLED)
            self.assertTrue(campaign.blocking_reasons)

    def test_deterministic_agent_runs_once_and_versions_bind_schedule(self):
        campaign = self.campaign(inference_stochasticity="deterministic")
        self.assertEqual(len(campaign.jobs), 1)
        self.assertIsNone(campaign.jobs[0].request.agent_seed)
        self.assertNotEqual(campaign.fingerprint, self.campaign(agent_version="changed").fingerprint)

    def test_checkpoint_methods_are_not_misrepresented_as_training_replicates(self):
        campaign = self.campaign(benchmark_eligible=True, requires_checkpoint=True)
        self.assertTrue(any("checkpoint" in r for r in campaign.blocking_reasons))

    def test_different_information_or_actuation_blocks_direct_comparison(self):
        other = replace(self.agent, agent_id="other", action_adapter_id="continuous_box2_v1")
        campaign = build_evaluation_campaign(self.profiles[0], [self.agent, other])
        self.assertTrue(any("action adapter" in r for r in campaign.blocking_reasons))

    def test_profile_file_cannot_change_locked_case_count_or_seeds(self):
        source = self.root / "configs/experiments/workbench_profiles_v1.json"
        for field, value in (("case_count", 1), ("replicate_seeds", [42]), ("tier", "Easy")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                target = root / "configs/experiments/workbench_profiles_v1.json"
                target.parent.mkdir(parents=True)
                data = json.loads(source.read_text(encoding="utf-8"))
                data["profiles"][0][field] = value
                target.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_evaluation_profiles(root)

    def test_guided_page_is_default_and_manual_form_remains_available(self):
        window = MainWindow()
        self.addCleanup(window.close)
        widget = window.benchmark_widget
        self.assertEqual(window.tab_widget.count(), 6)
        self.assertEqual(window.setup_pages.currentIndex(), 0)
        self.assertEqual(widget.preview.profile.mode.value, "VALIDATION")
        self.assertEqual(len(widget.preview.jobs), 1)
        self.assertEqual(widget.preview.jobs[0].request.agent_id, "stage2_astar")
        self.assertFalse(widget.btn_run.isEnabled())
        self.assertTrue(all(a.purpose != "AUDIT_FIXTURE" for a in widget.agents))
        widget.combo_profile.setCurrentIndex(2)
        request = widget.preview.jobs[0].request
        self.assertEqual((request.tier, request.sequence, request.geometry_generation_seed,
                          request.environment_seed, request.agent_seed), ("Easy", "SCS", 0, 0, None))
        self.assertEqual(window.setup_pages.widget(1), window.setup_widget)

    def test_blocked_plan_never_enables_run(self):
        controller, runner = self.controller()
        campaign = self.campaign(inference_stochasticity="deterministic")
        controller.start("PLAN", campaign)
        self.finish_plan_job(controller, runner, allowed=False)
        self.assertEqual(controller.status, "BLOCKED")
        self.assertIsNone(controller.validated_fingerprint)
        with self.assertRaises(ValueError):
            controller.start("RUN", campaign)
        self.assertEqual(len(runner.starts), 1)

    def test_queue_waits_for_process_exit_and_cancellation_stops_next_job(self):
        controller, runner = self.controller()
        controller.start("PLAN", self.campaign(benchmark_eligible=True))
        controller.preflight_received({"can_execute": True, "checks": []})
        controller.worker_done({"operation": "PLAN", "success": True})
        self.app.processEvents()
        self.assertEqual(len(runner.starts), 1)
        runner.is_running = False
        controller.process_finished(0)
        controller.cancel()
        self.app.processEvents()
        self.assertEqual(len(runner.starts), 1)
        self.assertEqual(controller.status, "CANCELLED")
        self.assertEqual(controller.rows[1]["status"], "NOT_RUN")

    def test_all_preflights_are_required_before_run_and_selection_change_invalidates(self):
        controller, runner = self.controller()
        campaign = self.campaign(benchmark_eligible=True)
        with self.assertRaises(ValueError):
            controller.start("RUN", campaign)
        controller.start("PLAN", campaign)
        for _ in range(3):
            self.finish_plan_job(controller, runner)
            self.app.processEvents()
        self.assertEqual(controller.status, "READY")
        self.assertEqual(controller.validated_fingerprint, campaign.fingerprint)
        controller.invalidate()
        with self.assertRaises(ValueError):
            controller.start("RUN", campaign)

    def test_worker_without_done_stops_campaign(self):
        controller, runner = self.controller()
        controller.start("PLAN", self.campaign(benchmark_eligible=True))
        runner.is_running = False
        controller.process_finished(0)
        self.app.processEvents()
        self.assertEqual(controller.status, "FAILED")
        self.assertEqual(len(runner.starts), 1)

    def test_request_serialization_failure_does_not_leave_runner_active(self):
        runner = WorkbenchProcessRunner()
        invalid = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id="stage0_random_naive", wandb_mode="DISABLED")
        with self.assertRaises(AttributeError):
            runner.start_operation("PLAN", invalid)
        self.assertFalse(runner.is_running)
        self.assertIsNone(runner.current_operation)
        self.assertIsNone(runner._temp_request_file)

    def test_run_finish_preserves_final_episode_progress_in_monitor(self):
        monitor = RunMonitorWidget()
        self.addCleanup(monitor.close)
        for event_type, fields in (
            (LauncherEventType.RUN_STARTED, {}),
            (LauncherEventType.EPISODE_STARTED, {"episode_index": 1, "total_episodes": 3}),
            (LauncherEventType.EPISODE_PROGRESS, {"episode_index": 1, "total_episodes": 3,
                                                "step_index": 40, "route_completion": .3, "speed_kmh": 20}),
        ):
            monitor.handle_launcher_event(LauncherEventV1.create(event_type, "run_test", **fields).to_dict())
        self.assertEqual(monitor.lbl_episode.text(), "Episode: 1 / 3")
        self.assertEqual(monitor.progress_bar.value(), 10)
        monitor.handle_launcher_event(LauncherEventV1.create(
            LauncherEventType.EPISODE_FINISHED, "run_test", episode_index=1, total_episodes=3,
            step_index=43, route_completion=.31,
        ).to_dict())
        monitor.handle_launcher_event(LauncherEventV1.create(
            LauncherEventType.RUN_FINISHED, "run_test", total_episodes=3,
        ).to_dict())
        self.assertEqual(monitor.lbl_step.text(), "Step: 43")
        self.assertEqual(monitor.lbl_route.text(), "Route Completion: 31.0%")
        self.assertEqual(monitor.lbl_speed.text(), "Speed: 20.0 km/h")

    def snapshot(self, job, run_id):
        cases = resolve_validation_cases(self.root)
        rows = [dict(case_id=c.case_id, split=c.split, episode_index=str(i),
                     protocol_order_index=str(c.protocol_order_index),
                     geometry_generation_seed=str(c.geometry_generation_seed),
                     environment_seed=str(c.environment_seed), horizon_steps=str(c.horizon_steps))
                for i, c in enumerate(cases, 1)]
        return SimpleNamespace(
            run_id=run_id, status="COMPLETE", integrity_status=IntegrityDisplayStatus.VERIFIED,
            agent_id=job.request.agent_id, agent_version=job.agent_version, agent_seed=job.request.agent_seed,
            input_profile_id=job.input_profile_id, action_adapter_id=job.action_adapter_id,
            episode_rows=rows, recorded_episode_count=96,
            summary_payload={"technical_failure_count": 0, "overall_metrics": {
                "clean_success_rate": .2, "safety_failure_rate": .8,
                "mean_final_route_completion": .3, "median_final_route_completion": .1,
                "mean_time_to_clean_success_s": None,
            }}, technical_failures_payload=None, canonical_run=True,
            git_worktree_dirty=False, dirty_override=False, unverified_env_override=False,
            environment_verification_status="VERIFIED", metadrive_version=PINNED_METADRIVE_VERSION,
            metadrive_commit=PINNED_METADRIVE_COMMIT, git_commit_sha="same-code",
            platform_contracts={"benchmark": "same-contract"},
        )

    def test_result_inspection_requires_complete_verified_matching_cohort(self):
        campaign = self.campaign(benchmark_eligible=True)
        reports = [{"run_id": f"run_{i}"} for i in range(3)]
        with patch("src.workbench.benchmark_campaign.RunArtifactRepository") as repo:
            snapshots = [self.snapshot(j, r["run_id"]) for j, r in zip(campaign.jobs, reports)]
            repo.return_value.load_run_snapshot.side_effect = snapshots
            rows = inspect_campaign_results(campaign, reports, self.root)
            self.assertEqual([r["agent_seed"] for r in rows], [101, 202, 303])
            self.assertEqual(rows[0]["mean_final_route_completion"], .3)
            self.assertIsNone(rows[0]["mean_time_to_clean_success_s"])
        for mutation in ("integrity", "canonical", "environment", "case", "technical", "seed", "code"):
            with self.subTest(mutation=mutation), patch("src.workbench.benchmark_campaign.RunArtifactRepository") as repo:
                snapshots = [self.snapshot(j, r["run_id"]) for j, r in zip(campaign.jobs, reports)]
                bad = snapshots[-1]
                if mutation == "integrity": bad.integrity_status = IntegrityDisplayStatus.FAILED
                elif mutation == "canonical": bad.canonical_run = False
                elif mutation == "environment": bad.environment_verification_status = "UNKNOWN"
                elif mutation == "case": bad.episode_rows[0]["environment_seed"] = "wrong"
                elif mutation == "technical": bad.summary_payload["technical_failure_count"] = 1
                elif mutation == "seed": bad.agent_seed = 999
                elif mutation == "code": bad.git_commit_sha = "different-code"
                repo.return_value.load_run_snapshot.side_effect = snapshots
                with self.assertRaises(ValueError):
                    inspect_campaign_results(campaign, reports, self.root)

    def wait_until(self, predicate, timeout=90):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertTrue(predicate(), "Timed out waiting for isolated worker")

    def test_real_gui_campaign_plan_is_blocked_by_existing_benchmark_rules(self):
        window = MainWindow()
        self.addCleanup(window.close)
        for row, agent in enumerate(window.benchmark_widget.agents):
            window.benchmark_widget.table_agents.item(row, 0).setCheckState(
                Qt.CheckState.Checked if agent.agent_id == "stage0_random_naive" else Qt.CheckState.Unchecked)
        controller = window.campaign_controller
        window.benchmark_widget.btn_plan.click()
        self.wait_until(lambda: not controller.active)
        self.assertEqual(controller.status, "BLOCKED")
        self.assertEqual(len(controller.rows), 3)
        self.assertTrue(all(r["status"] == "BLOCKED" for r in controller.rows))
        self.assertFalse(window.benchmark_widget.btn_run.isEnabled())

    def test_real_fixed_train_demo_runs_and_verifies_saved_results(self):
        window = MainWindow()
        self.addCleanup(window.close)
        widget = window.benchmark_widget
        widget.combo_profile.setCurrentIndex(2)
        widget.check_show_demo.setChecked(False)
        widget.btn_plan.click()
        controller = window.campaign_controller
        self.wait_until(lambda: not controller.active)
        self.assertEqual(controller.status, "READY")
        self.assertTrue(widget.btn_run.isEnabled())
        widget.btn_run.click()
        self.wait_until(lambda: not controller.active)
        self.assertEqual(controller.status, "COMPLETE", controller.rows)
        self.assertEqual(widget.table_results.rowCount(), 1)
        self.assertFalse(widget._result_rows[0]["canonical"])


if __name__ == "__main__":
    unittest.main()
