"""Real isolated TRAIN worker, checkpoint provenance, and paired GUI execution."""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from src.agents.astar_agent import LearnedAStarAgent
from src.agents.heuristic_model import HeuristicMLP, file_sha256
from src.launcher.cases import get_default_project_root
from src.launcher.checkpoints import bindings_from_training_manifest, validate_checkpoint_binding
from src.launcher.executor import ExperimentExecutor
from src.launcher.models import CheckpointLaunchRequestV1, LauncherMode, PreflightBlockedError
from src.launcher.registry import build_canonical_agent_registry
from src.launcher.resolver import resolve_experiment_plan
from src.platform import WandbMode
from src.training.protocol import parse_training_line, training_line
from src.training.runtime import local_training_runtime
from src.workbench.benchmark_campaign import build_evaluation_campaign, load_evaluation_profiles
from src.workbench.main_window import MainWindow
from src.workbench.process_runner import WorkbenchProcessRunner
from src.workbench.protocol import deserialize_launch_request, serialize_launch_request
from src.workbench.widgets.training_widget import TrainingWidget


def wait_for(app, predicate, timeout=120):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    if not predicate():
        raise AssertionError("Timed out waiting for isolated training/evaluation worker")


class TestAStarWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.root = get_default_project_root()
        cls.runner = WorkbenchProcessRunner()
        cls.messages, cls.done, cls.exits = [], [], []
        cls.runner.rawOutputReceived.connect(lambda line: cls.messages.append(parse_training_line(line)))
        cls.runner.workerDone.connect(cls.done.append)
        cls.runner.processFinished.connect(cls.exits.append)
        pointer = cls.root / "artifacts/astar_learning/latest.json"
        before = pointer.read_bytes() if pointer.exists() else None
        cls.runner.start_training("smoke", publish_latest=False)
        try:
            wait_for(cls.app, lambda: bool(cls.exits))
        finally:
            if cls.runner.is_running:
                cls.runner.force_terminate_and_wait()
        after = pointer.read_bytes() if pointer.exists() else None
        if before != after:
            raise AssertionError("Integration check changed the user's selected checkpoint")
        cls.messages = [m for m in cls.messages if m is not None]
        reports = [m["payload"] for m in cls.messages if m["type"] == "REPORT"]
        if cls.exits != [0] or not cls.done or not cls.done[-1].get("success") or not reports:
            raise AssertionError(f"Real TRAIN worker failed: {cls.exits}, {cls.messages[-3:]}")
        cls.report = reports[-1]
        cls.bindings = bindings_from_training_manifest(cls.root, cls.report["manifest_path"])

    def request(self, binding=None):
        return CheckpointLaunchRequestV1(
            mode=LauncherMode.SANDBOX, agent_id="stage6_astar_learned", tier="Easy", sequence="SCS",
            geometry_generation_seed=0, environment_seed=0, wandb_mode=WandbMode.DISABLED,
            checkpoint_binding=binding or self.bindings[0],
        )

    def test_real_training_has_optimizer_evidence_and_train_only_normalization(self):
        self.assertEqual(self.report["status"], "COMPLETE")
        model_record = self.report["models"][0]
        self.assertEqual(len(model_record["collected_geometries"]), 8)
        self.assertTrue(all(g["split"] == "TRAIN" for g in model_record["collected_geometries"]))
        self.assertGreater(model_record["optimizer_steps"], 0)
        self.assertLess(model_record["best_holdout_mse"], model_record["initial_holdout_mse"])
        epochs = [m["payload"]["epoch"] for m in self.messages if m["type"] == "PROGRESS" and m["payload"].get("phase") == "FIT"]
        self.assertEqual(epochs, list(range(41)))
        binding = self.bindings[0]
        path = validate_checkpoint_binding(binding, self.root)
        model = HeuristicMLP.load(path, binding.model_sha256, binding.planner_fingerprint)
        data_path = path.parent / "dataset.npz"
        self.assertEqual(file_sha256(data_path), binding.data_sha256)
        with np.load(data_path, allow_pickle=False) as data:
            mask = np.isin(data["geometry_ids"], model.metadata["train_geometry_ids"])
            np.testing.assert_allclose(model.mean, data["features"][mask].mean(axis=0))
            self.assertEqual(int(mask.sum()), model.metadata["train_samples"])
        self.assertFalse(set(model.metadata["train_geometry_ids"]) & set(model.metadata["internal_holdout_geometry_ids"]))
        agent = LearnedAStarAgent()
        agent.load_checkpoint(path, binding.model_sha256)
        self.assertEqual(agent.descriptor.action_adapter_id, "discrete9_lowbranch_v1")

    def test_checkpoint_roundtrip_and_plan_identity_bind_content_not_local_path(self):
        request = self.request()
        self.assertEqual(deserialize_launch_request(serialize_launch_request(request)), request)
        plan = resolve_experiment_plan(request, project_root=self.root)
        moved = replace(request, checkpoint_binding=replace(self.bindings[0], model_path="artifacts/moved-model.npz"))
        other = resolve_experiment_plan(moved, project_root=self.root)
        self.assertEqual(plan.resolved_plan_sha256, other.resolved_plan_sha256)
        changed = replace(request, checkpoint_binding=replace(self.bindings[0], model_sha256="0" * 64))
        self.assertNotEqual(plan.resolved_plan_sha256,
                            resolve_experiment_plan(changed, project_root=self.root).resolved_plan_sha256)

    def test_checkpoint_tampering_blocks_before_logger_and_simulator(self):
        bad = replace(self.bindings[0], model_sha256="0" * 64)
        plan = resolve_experiment_plan(self.request(bad), project_root=self.root)
        with local_training_runtime(self.root), patch("src.launcher.executor.LocalExperimentLogger") as logger:
            with self.assertRaisesRegex(PreflightBlockedError, "Checkpoint SHA-256 mismatch"):
                ExperimentExecutor(plan, project_root=self.root).execute()
            logger.assert_not_called()

    def test_checkpoint_rejects_train_leakage_empty_optimization_and_path_escape(self):
        binding = self.bindings[0]
        with tempfile.TemporaryDirectory(dir=self.root / "artifacts") as temp:
            target = Path(temp) / "tampered.npz"
            for updates in (
                {"internal_holdout_geometry_ids": ["test/Easy/SCS/geom-11"]},
                {"internal_holdout_geometry_ids": self.report["models"][0]["train_geometry_ids"]},
                {"optimizer_steps": 0},
            ):
                with self.subTest(updates=updates):
                    model = HeuristicMLP.load(self.root / binding.model_path, binding.model_sha256, binding.planner_fingerprint)
                    digest = model.save(target, updates)
                    bad = replace(binding, model_path=str(target.relative_to(self.root)), model_sha256=digest)
                    with self.assertRaises(ValueError):
                        validate_checkpoint_binding(bad, self.root)
            with self.assertRaises(ValueError):
                validate_checkpoint_binding(replace(binding, model_path="../outside.npz"), self.root)
            with self.assertRaises(ValueError):
                validate_checkpoint_binding(replace(binding, planner_fingerprint="0" * 64), self.root)

    def test_schedule_separates_training_seeds_from_inference_seeds(self):
        registry = build_canonical_agent_registry()
        agents = [registry.get("stage2_astar"), registry.get("stage6_astar_learned")]
        validation, _, demo = load_evaluation_profiles(self.root)
        campaign = build_evaluation_campaign(demo, agents, checkpoint_bindings=self.bindings, show_demo=False)
        self.assertFalse(campaign.blocking_reasons)
        self.assertEqual(len(campaign.jobs), 2)
        self.assertTrue(all(j.request.agent_seed is None for j in campaign.jobs))
        self.assertEqual(campaign.jobs[1].request.checkpoint_binding.training_run_seed, 101)
        validation_campaign = build_evaluation_campaign(validation, agents, checkpoint_bindings=self.bindings)
        self.assertTrue(any("101, 202, 303" in reason for reason in validation_campaign.blocking_reasons))
        self.assertTrue(any("benchmark" in reason for reason in validation_campaign.blocking_reasons))

    def test_training_ui_accepts_real_epochs_and_requires_successful_completion(self):
        widget = TrainingWidget(self.root)
        self.addCleanup(widget.close)
        completed = []
        widget.trainingComplete.connect(completed.append)
        widget.begin()
        for message in self.messages:
            widget.consume_line(training_line(message["type"], message["payload"]))
        self.assertEqual(len(widget._curves[101]), 41)
        widget.finish(0, None)
        self.assertFalse(completed)
        self.assertLess(widget.progress.value(), 100)
        widget.finish(0, {"operation": "TRAIN", "success": True})
        self.assertEqual(completed, [self.report])
        self.assertEqual(widget.progress.value(), 100)

    def test_main_window_locks_during_training_and_selects_pair_after_smoke(self):
        window = MainWindow()
        self.addCleanup(window.close)
        with patch.object(window._runner, "start_training") as start:
            window._on_training_requested("smoke")
            start.assert_called_once_with("smoke")
            self.assertTrue(window._training_active)
            self.assertFalse(window.benchmark_widget.btn_plan.isEnabled())
            self.assertFalse(window.setup_pages.isTabEnabled(1))
            window._on_training_requested("smoke")
            self.assertEqual(start.call_count, 1)
        for message in self.messages:
            window._on_raw_worker_output(training_line(message["type"], message["payload"]))
        window._on_worker_done({"operation": "TRAIN", "success": True})
        with patch("src.workbench.widgets.benchmark_widget.latest_training_bindings", return_value=self.bindings):
            window._on_process_finished(0)
        self.assertFalse(window._training_active)
        self.assertEqual(window.setup_pages.currentIndex(), 0)
        self.assertEqual(window.benchmark_widget.preview.profile.profile_id, "train_demo_v1")
        self.assertEqual([j.request.agent_id for j in window.benchmark_widget.preview.jobs],
                         ["stage2_astar", "stage6_astar_learned"])
        self.assertFalse(window.benchmark_widget.btn_run.isEnabled())  # Must preflight the new models.

    def test_real_gui_pair_runs_same_train_case_and_checks_saved_checkpoint_provenance(self):
        window = MainWindow()
        self.addCleanup(window.close)
        widget = window.benchmark_widget
        widget.combo_profile.setCurrentIndex(2)
        widget.check_show_demo.setChecked(False)
        widget.checkpoint_bindings = self.bindings
        for row, agent in enumerate(widget.agents):
            widget.table_agents.item(row, 0).setCheckState(
                Qt.CheckState.Checked if agent.agent_id in ("stage2_astar", "stage6_astar_learned") else Qt.CheckState.Unchecked)
        widget._selection_changed()
        widget.btn_plan.click()
        controller = window.campaign_controller
        wait_for(self.app, lambda: not controller.active)
        self.assertEqual(controller.status, "READY", controller.rows)
        widget.btn_run.click()
        wait_for(self.app, lambda: not controller.active)
        self.assertEqual(controller.status, "COMPLETE", controller.rows)
        self.assertEqual(len(widget._result_rows), 2)
        self.assertEqual(widget._result_rows[1]["training_run_seed"], 101)
        self.assertTrue(all(not r["canonical"] for r in widget._result_rows))
        run_id = widget._result_rows[1]["run_id"]
        window._open_campaign_run(run_id)
        self.assertIn(self.bindings[0].model_sha256, window.results_widget.text_provenance.toPlainText())


if __name__ == "__main__":
    unittest.main()
