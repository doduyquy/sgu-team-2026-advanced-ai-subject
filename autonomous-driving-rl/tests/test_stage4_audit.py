"Kiểm thử G0 phát hiện hỏng dữ liệu, rò rỉ tập đánh giá và bảo toàn lần thử thất bại."

from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import stage4
from src.imitation.audit import (
    EvidenceError, audit_episode, audit_legacy_run, audit_snapshot, load_json,
    project_path, propose_partition, validate_case, validate_trace_row, verify_expert_sources,
)
from tests.test_rule_based_agent import make_input


def row(index=0, action=(0.125, 0.55)):
    return {"case_id": "TRAIN/example", "step_index": index,
            "agent_input": replace(make_input(), step_index=index).to_dict(),
            "decision": {"action_payload": list(action), "diagnostics": {"behavior_state": "YIELD"}},
            "canonical_action": np.asarray(action, dtype=np.float32).tolist(), "act_latency_ms": 0.1}


class TestStage4TraceAudit(unittest.TestCase):
    def test_float32_rounding_is_accepted_without_relabeling(self):
        example = row()
        self.assertNotEqual(example["canonical_action"], example["decision"]["action_payload"])
        inp, decision = validate_trace_row(example, "TRAIN/example", 0)
        self.assertEqual(inp.to_dict(), example["agent_input"])
        self.assertEqual(decision.action_payload, example["decision"]["action_payload"])

    def test_wrong_case_or_missing_step_is_rejected(self):
        for example in [row(1), {**row(), "case_id": "other"}, {**row(), "step_index": True}]:
            with self.subTest(example=example["step_index"]), self.assertRaises(EvidenceError):
                validate_trace_row(example, "TRAIN/example", 0)

    def test_input_step_mismatch_is_rejected(self):
        example = row()
        example["agent_input"]["step_index"] = 1
        with self.assertRaises(EvidenceError):
            validate_trace_row(example, "TRAIN/example", 0)

    def test_one_step_shifted_action_is_rejected(self):
        example = row()
        example["canonical_action"] = row(1, (-0.2, -0.8))["canonical_action"]
        with self.assertRaises(EvidenceError):
            validate_trace_row(example, "TRAIN/example", 0)

    def test_unrounded_payload_cannot_masquerade_as_actual_action(self):
        example = row()
        example["canonical_action"] = example["decision"]["action_payload"]
        with self.assertRaises(EvidenceError):
            validate_trace_row(example, "TRAIN/example", 0)

    def test_invalid_action_is_not_silently_clipped(self):
        with self.assertRaises(ValueError):
            validate_trace_row(row(action=(1.2, 0)), "TRAIN/example", 0)

    def test_forbidden_and_unknown_nested_features_are_rejected(self):
        for field in ["reward", "future_outcome", "expert_hidden_state"]:
            example = row()
            example["agent_input"][field] = 1
            with self.subTest(field=field), self.assertRaises(EvidenceError):
                validate_trace_row(example, "TRAIN/example", 0)
        example = row()
        example["agent_input"]["task_context"]["private_route_completion"] = 1
        with self.assertRaises(EvidenceError):
            validate_trace_row(example, "TRAIN/example", 0)

    def test_old_schema_and_nonfinite_features_are_rejected(self):
        for modify in [lambda d: d.pop("ego_motion"),
                       lambda d: d["core_observation"]["features"].__setitem__(0, float("nan")),
                       lambda d: d["traffic_context"].pop("motion_paths")]:
            example = row()
            modify(example["agent_input"])
            with self.assertRaises(ValueError):
                validate_trace_row(example, "TRAIN/example", 0)

    def test_nonzero_invalid_padding_is_rejected(self):
        example = row()
        example["agent_input"]["traffic_context"]["actors"][7]["length"] = 4.5
        with self.assertRaises(ValueError):
            validate_trace_row(example, "TRAIN/example", 0)


class TestStage4EpisodeAudit(unittest.TestCase):
    def fixture(self, root, outcome="TIMEOUT", expected_steps=3):
        case = {"case_id": "TRAIN/example", "horizon_steps": 10}
        result = {"case_id": case["case_id"], "agent_version": "0.6.0", "steps": expected_steps,
                  "terminal_reason": outcome, "clean_success": outcome == "SUCCESS",
                  "crash": False, "out_of_road": False, "boundary_contact": False}
        (root / "result.json").write_text(json.dumps(result), encoding="utf8")
        (root / "environment.json").write_text(json.dumps({"case": case}), encoding="utf8")
        with gzip.open(root / "decisions.jsonl.gz", "wt", encoding="utf8") as stream:
            for index in range(3):
                stream.write(json.dumps(row(index)) + "\n")
        return case, result

    def test_timeout_segments_are_review_items_not_accepted_data(self):
        with tempfile.TemporaryDirectory() as name:
            case, result = self.fixture(Path(name))
            summary = audit_episode(name, case, result)
            self.assertEqual(summary["steps"], 3)
            segment, = summary["timeout_review_segments"]
            self.assertEqual((segment["start_step"], segment["end_step_exclusive"]), (0, 3))
            self.assertEqual(segment["review_status"], "UNREVIEWED")
            self.assertIsNone(segment["quality_class"])
            self.assertIsNone(segment["min_observed_speed_mps"])
            self.assertFalse(segment["approved_for_bc"])

    def test_success_is_not_automatically_quality_certified(self):
        with tempfile.TemporaryDirectory() as name:
            case, result = self.fixture(Path(name), outcome="SUCCESS")
            summary = audit_episode(name, case, result)
            self.assertEqual(summary["timeout_review_segments"], [])
            self.assertIn("UNREVIEWED", summary["trajectory_quality_status"])

    def test_partial_episode_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            case, result = self.fixture(Path(name), expected_steps=4)
            with self.assertRaises(EvidenceError):
                audit_episode(name, case, result)

    def test_false_clean_success_and_technical_errors_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            case, result = self.fixture(Path(name), outcome="SUCCESS")
            for change in [{"crash": True}, {"technical_error": "bước mô phỏng thất bại"}]:
                with self.subTest(change=change), self.assertRaises(EvidenceError):
                    audit_episode(name, case, {**result, **change})

    def test_failed_legacy_run_is_rejected_before_reading_episodes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "attempt").mkdir()
            (root / "attempt/run_state.json").write_text(json.dumps({"status": "INTERRUPTED"}), encoding="utf8")
            with self.assertRaises(EvidenceError):
                audit_legacy_run(root, "attempt", {}, {})


class TestStage4SplitAndProvenance(unittest.TestCase):
    def test_aliases_bind_to_p4_and_auxiliary_copies_still_require_exact_bytes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            names = ["policy.py", "agent_input.py", "action_adapter.py", "agent_experiment.py",
                     "agents/rule_based_agent.py", "agents/rule_based_traffic.py",
                     "agents/rule_based_memory.py", "agents/rule_based_junction.py",
                     "agents/rule_based_lane_change.py", "agents/random_naive_agent.py", "agents/__init__.py"]
            digest = hashlib.sha256(b"source bytes").hexdigest()
            recorded = {name: digest for name in names}
            reference = {name.removeprefix("agents/"): digest for name in names
                         if name not in {"agents/random_naive_agent.py", "agents/__init__.py", "agents/rule_based_agent.py"}}
            for path in names:
                target = root / path
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(b"source bytes")
            bindings = verify_expert_sources(root, recorded, reference)
            self.assertEqual(bindings["agents/rule_based_agent.py"]["reference_key"], "policy.py")
            self.assertEqual(bindings["agents/random_naive_agent.py"]["role"], "AUXILIARY_RECORDER_COPY_NOT_P4_REFERENCE")
            with self.assertRaisesRegex(EvidenceError, "chưa đầy đủ"):
                verify_expert_sources(root, {k: v for k, v in recorded.items() if k != "agent_input.py"}, reference)
            bad_reference = {**reference, "rule_based_memory.py": "bad"}
            with self.assertRaisesRegex(EvidenceError, "P4 đã đóng băng"):
                verify_expert_sources(root, recorded, bad_reference)
            (root / "agents/random_naive_agent.py").write_bytes(b"tampered")
            with self.assertRaisesRegex(EvidenceError, "không khớp"):
                verify_expert_sources(root, recorded, reference)

    def test_partition_stays_geometry_grouped_and_order_independent(self):
        geometries = {(s, i): {"sequence": s, "geometry_generation_seed": str(i),
                              "geometry_id": f"{s}/{i}", "geometry_sha256": f"hash-{s}-{i}"}
                      for s in ("SCS", "SCXCS") for i in range(15)}
        partition = propose_partition(geometries)
        self.assertEqual(partition["counts"], {"DEV": 6, "FIT": 24})
        self.assertEqual(partition, propose_partition(dict(reversed(list(geometries.items())))))
        self.assertEqual(len({r["geometry_id"] for r in partition["geometries"]}), 30)
        geometries.pop(("SCS", 0))
        with self.assertRaises(EvidenceError):
            propose_partition(geometries)

    def test_holdout_wrong_geometry_and_reserved_env_seeds_are_rejected(self):
        case = {"sequence": "SCS", "geometry_generation_seed": 0, "geometry_sha256": "abc",
                "split": "TRAIN", "tier": "Easy", "environment_seed": 0}
        train = {("SCS", 0): case}
        validate_case(case, train)
        for change in [{"split": "TEST"}, {"geometry_generation_seed": 1},
                       {"geometry_sha256": "wrong"}, {"environment_seed": 5101}]:
            with self.subTest(change=change), self.assertRaises(EvidenceError):
                validate_case({**case, **change}, train)

    def test_newline_equivalence_does_not_repair_or_certify_archive(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            content = b"a\r\nb\r\n"
            (root / "policy.py").write_bytes(content)
            (root / "snapshot.json").write_text(json.dumps({"version": "0.6.1", "source_sha256": {
                "policy.py": hashlib.sha256(b"a\nb\n").hexdigest()}}), encoding="utf8")
            result = audit_snapshot(root, "snapshot.json")
            self.assertEqual(result["counts"], {"LF_CONVERSION_MATCH_ONLY": 1})
            self.assertEqual((root / "policy.py").read_bytes(), content)
            (root / "policy.py").write_bytes(b"changed")
            self.assertEqual(audit_snapshot(root, "snapshot.json")["counts"], {"MISMATCH": 1})

    def test_paths_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as name:
            for value in ["../outside", "a/../../outside", "C:/outside", "/outside", ""]:
                with self.subTest(value=value), self.assertRaises(EvidenceError):
                    project_path(name, value)

    def test_nonfinite_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "bad.json"
            path.write_text('{"x":NaN}', encoding="utf8")
            with self.assertRaises(EvidenceError):
                load_json(path)


class TestStage4AttemptLifecycle(unittest.TestCase):
    def test_integrity_write_failure_does_not_leave_complete_state(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            state = {"run_id": "example", "status": "COMPLETE", "gate_approved": False}
            stage4.write_record(output / "run_state.json", state)
            original = stage4.write_record

            def fail_integrity(path, payload, replace=False):
                if Path(path).name == "run_integrity.json":
                    raise OSError("giả lập lỗi lưu hồ sơ toàn vẹn")
                return original(path, payload, replace)

            with patch.object(stage4, "write_record", side_effect=fail_integrity):
                with self.assertRaises(OSError):
                    stage4.seal_attempt(output, state)
            self.assertEqual(load_json(output / "run_state.json")["status"], "FAILED")
            self.assertEqual(load_json(output / "failure.json")["type"], "INTEGRITY_FINALIZATION_ERROR")
            self.assertFalse((output / "run_integrity.json").exists())

    def test_failed_attempt_is_preserved_and_duplicate_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "runs").mkdir()
            with patch.object(stage4, "ROOT", root):
                with self.assertRaises(FileNotFoundError):
                    stage4.run_audit("missing.json", "s4_failed")
                attempt = root / "runs/s4_failed"
                self.assertEqual(load_json(attempt / "run_state.json")["status"], "FAILED")
                self.assertTrue((attempt / "failure.json").exists())
                self.assertEqual(load_json(attempt / "run_integrity.json")["files"], stage4.fingerprints(attempt))
                state = (attempt / "run_state.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    stage4.run_audit("missing.json", "s4_failed")
                self.assertEqual((attempt / "run_state.json").read_bytes(), state)
                with self.assertRaises(EvidenceError):
                    stage4.publish_audit("s4_failed", "results/audits/stage4/g0/example")

    def test_publication_rejects_tampered_or_unverified_reports(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            attempt = root / "runs/example"
            attempt.mkdir(parents=True)
            stage4.write_record(attempt / "audit_summary.json", {"tests_status": "NOT_RUN"})
            stage4.write_record(attempt / "run_integrity.json", {
                "run_id": "example", "status": "COMPLETE", "files": stage4.fingerprints(attempt)})
            with patch.object(stage4, "ROOT", root):
                with self.assertRaisesRegex(EvidenceError, "bằng chứng kiểm thử hồi quy"):
                    stage4.publish_audit("example", "results/audits/stage4/g0/example")
                (attempt / "audit_summary.json").write_bytes(b"{}")
                with self.assertRaisesRegex(EvidenceError, "mã băm không khớp"):
                    stage4.publish_audit("example", "results/audits/stage4/g0/example")

    def test_records_do_not_overwrite_existing_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "record.json"
            stage4.write_record(path, {"attempt": 1})
            with self.assertRaises(FileExistsError):
                stage4.write_record(path, {"attempt": 2})
            self.assertEqual(load_json(path), {"attempt": 1})


if __name__ == "__main__":
    unittest.main()
