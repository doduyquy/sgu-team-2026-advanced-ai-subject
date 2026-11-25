"""Incomplete runs and modified source evidence must not become release reports."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_rule_based_experiment.py"
spec = importlib.util.spec_from_file_location("rule_based_report", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class TestRuleBasedReportIntegrity(unittest.TestCase):
    def test_shard_order_may_differ_but_physical_case_must_match(self):
        case = {"case_id": "sandbox/Hard/x/0/0", "split": "TRAIN", "geometry_sha256": "same", "environment_seed": 0,
                "traffic_density": .2, "horizon_steps": 1000, "case_index": 7}
        report.validate_paired_definitions(case, case | {"case_index": 1})
        for changed in ({"environment_seed": 1}, {"traffic_density": .5}, {"geometry_sha256": "other"}, {"horizon_steps": 2}):
            with self.assertRaisesRegex(ValueError, "Mismatched paired case"):
                report.validate_paired_definitions(case, case | changed)

    def test_unfinished_run_cannot_be_published_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_state.json").write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run status RUNNING"):
                report.load_run(root, "current")

    def test_modified_source_snapshot_is_rejected_before_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "current/sources").mkdir(parents=True)
            (root / "run_state.json").write_text(json.dumps({"status": "COMPLETED"}), encoding="utf-8")
            (root / "resolved_cases.json").write_text(json.dumps({"cases": []}), encoding="utf-8")
            (root / "current/agent_provenance.json").write_text(json.dumps({
                "source_sha256": {"policy.py": "original_hash"},
            }), encoding="utf-8")
            (root / "current/sources/policy.py").write_text("modified code", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Source snapshot was modified"):
                report.load_run(root, "current")


if __name__ == "__main__":
    unittest.main()
