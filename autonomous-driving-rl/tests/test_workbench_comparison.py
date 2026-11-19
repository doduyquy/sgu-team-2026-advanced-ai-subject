"""Integrity and offscreen GUI tests for the additive Results Comparison view."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from src.workbench.comparison_repository import (
    ComparisonRun,
    _checkpoint_bundle_status,
    compare_runs,
    discover_comparison_runs,
)
from src.workbench.widgets.comparison_widget import ComparisonWidget


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTUAL_RESULT_ID = "s5_3m_tau_ab_seed101_20261119_retry1"


def _sample_run(**changes):
    defaults = dict(
        key="a", label="A", track="CANONICAL_EVALUATION", run_id="run-a",
        agent_id="agent_a", agent_version="1", method_family="SEARCH",
        checkpoint_step=None, checkpoint_sha256=None, checkpoint_bundle_status="N/A",
        source_revision="abc", dataset_split="VALIDATION", schedule=(("case-1", 1),),
        input_profile_id="STATE_DECISION_V1", action_adapter_id="continuous_box2_v1",
        inference_mode="deterministic", metric_contracts=(("metric", "v1"),),
        summary={"mean_final_route_completion": .2}, per_tier={},
        records=({"case_id": "case-1", "environment_seed": 1, "final_route_completion": .2},),
        provenance={}, benchmark_eligible=True,
    )
    defaults.update(changes)
    return ComparisonRun(**defaults)


class TestComparisonRepository(unittest.TestCase):
    def test_actual_stage5_ab_artifacts_load_as_train_only_diagnostics(self):
        runs = discover_comparison_runs(PROJECT_ROOT)
        stage5 = [run for run in runs if run.track == "DEVELOPMENT_DIAGNOSTICS"]
        self.assertEqual(len(stage5), 6)
        self.assertTrue(all("TRAIN ONLY" in run.dataset_split for run in stage5))
        self.assertTrue(all(not run.benchmark_eligible for run in stage5))

    def test_actual_a_b_10k_results_are_paired_and_preserve_censoring(self):
        runs = discover_comparison_runs(PROJECT_ROOT)
        left = next(run for run in runs if run.key.endswith(":A:step10000"))
        right = next(run for run in runs if run.key.endswith(":B:step10000"))
        result = compare_runs(left, right)
        self.assertTrue(result.directly_comparable)
        self.assertEqual(len(result.paired_records), 8)
        self.assertEqual(left.summary["probe_censored"], 8)
        self.assertEqual(right.summary["official_timeout_count"], 0)
        self.assertEqual(left.records[0]["outcome"], "PROBE_CENSORED")
        self.assertIn("mean_deterministic_min_q", left.summary)

    def test_missing_checkpoint_bundle_is_reported_without_rejecting_artifact_record(self):
        status = _checkpoint_bundle_status(Path(tempfile.gettempdir()), "runs/missing", "A", 10000,
                                           "a" * 64, "b" * 64, "c" * 64)
        self.assertIn("BUNDLE MISSING", status)

    def test_missing_checkpoint_sha_suppresses_pairing(self):
        left = _sample_run(track="DEVELOPMENT_DIAGNOSTICS", checkpoint_sha256=None,
                           dataset_split="PILOT_DIAGNOSTIC_INTERNAL_TRAIN",
                           provenance={"protocol_sha256": "same"})
        right = replace(left, key="b", checkpoint_sha256="a" * 64)
        self.assertFalse(compare_runs(left, right).directly_comparable)
        self.assertIn("SHA-256 is missing", compare_runs(left, right).reason)

    def test_case_id_or_environment_seed_mismatch_prevents_pairing(self):
        left = _sample_run()
        for schedule in ((("different-case", 1),), (("case-1", 2),)):
            with self.subTest(schedule=schedule):
                result = compare_runs(left, replace(left, key="b", schedule=schedule))
                self.assertFalse(result.directly_comparable)
                self.assertIn("Case IDs or environment seeds differ", result.reason)

    def test_incompatible_metric_and_action_adapter_tracks_prevent_pairing(self):
        left = _sample_run()
        self.assertFalse(compare_runs(left, replace(left, key="b", action_adapter_id="other_adapter")).directly_comparable)
        self.assertFalse(compare_runs(
            left, replace(left, key="b", metric_contracts=(("metric", "v2"),))
        ).directly_comparable)

    def test_cross_track_comparison_is_rejected(self):
        left = _sample_run()
        right = replace(left, key="diagnostic", track="DEVELOPMENT_DIAGNOSTICS")
        self.assertIn("different tracks", compare_runs(left, right).reason)

    def test_noneligible_canonical_run_is_not_admitted_to_track_a(self):
        left = _sample_run(benchmark_eligible=False)
        self.assertIn("not benchmark eligible", compare_runs(left, left).reason)

    def test_corrupted_stage5_artifact_fails_closed(self):
        source = PROJECT_ROOT / "results" / "stage5" / ACTUAL_RESULT_ID
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "results" / "stage5" / ACTUAL_RESULT_ID
            shutil.copytree(source, target)
            analysis = json.loads((target / "analysis.json").read_text(encoding="utf-8"))
            analysis["tampered_fixture"] = True
            (target / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity failed"):
                discover_comparison_runs(root)


class TestComparisonWidget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_actual_diagnostic_artifact_renders_paired_gui(self):
        widget = ComparisonWidget(PROJECT_ROOT)
        try:
            runs = widget._runs
            a = next(run for run in runs if run.key.endswith(":A:step10000"))
            b = next(run for run in runs if run.key.endswith(":B:step10000"))
            widget.combo_left.setCurrentIndex(widget.combo_left.findData(a.key))
            widget.combo_right.setCurrentIndex(widget.combo_right.findData(b.key))
            self.assertTrue(widget._last_result.directly_comparable)
            self.assertEqual(widget.table_cases.rowCount(), 8)
            self.assertIn("TRAIN ONLY — NOT BENCHMARK", widget.lbl_status.text())
            self.assertEqual(widget.table_cases.item(0, 10).text(), "PROBE_CENSORED")
            self.assertEqual(widget.table_cases.item(0, 11).text(), "PROBE_CENSORED")
        finally:
            widget.close()
            self.app.processEvents()

    def test_no_data_state_and_repeated_open_close(self):
        for _ in range(3):
            with tempfile.TemporaryDirectory() as temp:
                widget = ComparisonWidget(Path(temp))
                self.assertIn("No data", widget.lbl_status.text())
                widget.close()
                self.app.processEvents()

    def test_corrupt_artifact_is_reported_not_rendered_as_a_result(self):
        source = PROJECT_ROOT / "results" / "stage5" / ACTUAL_RESULT_ID
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "results" / "stage5" / ACTUAL_RESULT_ID
            shutil.copytree(source, target)
            with (target / "analysis.json").open("ab") as stream:
                stream.write(b" ")
            widget = ComparisonWidget(root)
            try:
                self.assertIn("SOURCE VALIDATION FAILED", widget.lbl_status.text())
                self.assertEqual(widget.combo_left.count(), 0)
                self.assertEqual(widget.table_cases.rowCount(), 0)
            finally:
                widget.close()


if __name__ == "__main__":
    unittest.main()
