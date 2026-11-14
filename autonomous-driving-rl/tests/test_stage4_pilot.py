"""Bắt lệch thời điểm, sai nhãn hành động, sửa bằng chứng và nhận nhầm lượt thu lỗi."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import collect_stage4_pilot as collector
from src.imitation.audit import EvidenceError, load_json, sha256
from src.imitation.pilot import check_pair, encode_runtime_config, observation_digest, verify_seal
from src.platform.agent_input import EgoMotionV1
from tests.test_rule_based_agent import make_input


def pair():
    inp = replace(make_input(), ego_motion=EgoMotionV1(0, 0, 0, 15 / 3.6, 0, 0))
    action = np.asarray([.2, .3], dtype=np.float32).tolist()
    decision = {"case_id": "case", "step_index": 0, "agent_input": inp.to_dict(),
                "decision": {"action_payload": [.2, .3], "diagnostics": None}, "canonical_action": action}
    transition = {"case_id": "case", "step_index": 0, "next_step_index": 1,
                  "pre_observation_sha256": observation_digest(inp.core_observation.features),
                  "canonical_action": action.copy(), "post_observation": inp.core_observation.features.tolist(),
                  "simulation_time_before_s": 0, "simulation_time_after_s": .1,
                  "pre_vehicle": {"body_velocity_mps": [15 / 3.6, 0]},
                  "post_vehicle": {"applied_steering": action[0], "body_velocity_mps": [5, 0], "speed_km_h": 18}}
    return decision, transition, {"case_id": "case"}, inp.core_observation.features.copy()


class TestPilotPairing(unittest.TestCase):
    def test_valid_pair_preserves_float32_label(self):
        inp, speed = check_pair(*pair()[:3], 0, pair()[3])
        self.assertEqual(speed, 18)
        self.assertEqual(inp.step_index, 0)

    def assert_rejected(self, change):
        decision, transition, case, previous = pair()
        change(decision, transition, previous)
        with self.assertRaises((EvidenceError, ValueError)):
            check_pair(decision, transition, case, 0, previous)

    def test_shifted_transition_index_is_rejected(self):
        self.assert_rejected(lambda d, t, p: t.update(step_index=1))

    def test_wrong_successor_index_is_rejected(self):
        self.assert_rejected(lambda d, t, p: t.update(next_step_index=2))

    def test_cross_case_pair_is_rejected(self):
        self.assert_rejected(lambda d, t, p: t.update(case_id="other"))

    def test_label_different_from_submitted_action_is_rejected(self):
        self.assert_rejected(lambda d, t, p: t.update(canonical_action=[-.2, .3]))

    def test_next_observation_cannot_replace_current_observation(self):
        self.assert_rejected(lambda d, t, p: p.__setitem__(19, .3))

    def test_pre_observation_digest_mismatch_is_rejected(self):
        self.assert_rejected(lambda d, t, p: t.update(pre_observation_sha256="0" * 64))

    def test_timestep_drift_is_rejected(self):
        self.assert_rejected(lambda d, t, p: t.update(simulation_time_after_s=.2))

    def test_applied_steering_sign_flip_is_rejected(self):
        self.assert_rejected(lambda d, t, p: t["post_vehicle"].update(applied_steering=-.2))

    def test_mps_value_cannot_be_logged_as_kmh(self):
        self.assert_rejected(lambda d, t, p: t["post_vehicle"].update(speed_km_h=5))

    def test_nonfinite_logged_speed_is_rejected(self):
        self.assert_rejected(lambda d, t, p: t["post_vehicle"].update(speed_km_h=float("nan")))

    def test_invalid_post_observation_is_rejected(self):
        self.assert_rejected(lambda d, t, p: t.update(post_observation=[0] * 258))


class TestPilotAttemptIntegrity(unittest.TestCase):
    def assert_partition_rejected_before_simulation(self, bad_hash=False):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            definition=root/"experiments/stage4/custom.json"
            definition.parent.mkdir(parents=True)
            partition=root/"datasets/partition/manifest.json"
            partition.parent.mkdir(parents=True)
            assignments=partition.parent/"geometry_assignments.csv"
            assignments.write_text("geometry_sha256,partition\nmap,DEV\n",encoding="utf8")
            partition.write_text(json.dumps({"status":"APPROVED_FROZEN","files":{"geometry_assignments.csv":sha256(assignments)}}))
            definition.write_text(json.dumps({"scope":"FIT_CALIBRATION","agent_seed":42,"prerequisite":"unused","cases":[{}],
                "required_partition":"FIT","partition_manifest":"datasets/partition/manifest.json",
                "partition_manifest_sha256":"0"*64 if bad_hash else sha256(partition)}))
            case=SimpleNamespace(geometry_sha256="map",to_dict=lambda:{"split":"TRAIN","geometry_sha256":"map"})
            plan=SimpleNamespace(resolved_cases=[case])
            with patch.object(collector,"ROOT",root),patch.object(collector,"verify_prerequisite"), \
                 patch.object(collector,"resolve",return_value=plan),patch.object(collector,"train_geometries",return_value=([],None)), \
                 patch.object(collector,"validate_case"),patch.object(collector.subprocess,"run") as launch:
                with self.assertRaises(EvidenceError): collector.collect("rejected","experiments/stage4/custom.json")
                launch.assert_not_called()
            self.assertEqual(load_json(root/"runs/rejected/run_state.json")["status"],"FAILED")

    def test_fit_collection_rejects_dev_before_launch(self):
        self.assert_partition_rejected_before_simulation()

    def test_custom_collection_rejects_modified_partition(self):
        self.assert_partition_rejected_before_simulation(bad_hash=True)

    def test_runtime_configuration_keeps_class_reference_and_numpy_values(self):
        encoded = encode_runtime_config({"policy": EgoMotionV1, "nested": (np.array([1, 2]), np.float32(.5))})
        self.assertEqual(encoded["policy"], {"python_type": "src.platform.agent_input:EgoMotionV1"})
        self.assertEqual(encoded["nested"], [[1, 2], .5])
        json.dumps(encoded, allow_nan=False)

    def test_unknown_runtime_configuration_value_is_not_silently_stringified(self):
        with self.assertRaises(TypeError):
            encode_runtime_config({"unexpected": object()})

    def test_sealed_attempt_detects_tampering_and_extra_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "sample").write_bytes(b"original")
            (folder / "run_integrity.json").write_text(json.dumps(
                {"status": "COMPLETE", "files": {"sample": sha256(folder / "sample")}}))
            verify_seal(folder)
            (folder / "extra").write_bytes(b"injected")
            with self.assertRaises(EvidenceError):
                verify_seal(folder)
            (folder / "sample").write_bytes(b"edited")
            with self.assertRaises(EvidenceError):
                verify_seal(folder)

    def test_failed_setup_is_preserved_and_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(collector, "ROOT", Path(temp)):
            with self.assertRaises(FileNotFoundError):
                collector.collect("failed")
            output = Path(temp) / "runs/failed"
            self.assertEqual(load_json(output / "run_state.json")["status"], "FAILED")
            self.assertEqual(load_json(output / "run_integrity.json")["status"], "FAILED")
            with self.assertRaises(EvidenceError):
                collector.publish("failed", "results/audits/stage4/g1/new")
            self.assertFalse((Path(temp) / "results").exists())

    def test_existing_attempt_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(collector, "ROOT", Path(temp)):
            output = Path(temp) / "runs/existing"
            output.mkdir(parents=True)
            (output / "keep").write_bytes(b"evidence")
            with self.assertRaises(FileExistsError):
                collector.collect("existing")
            self.assertEqual((output / "keep").read_bytes(), b"evidence")


if __name__ == "__main__":
    unittest.main()
