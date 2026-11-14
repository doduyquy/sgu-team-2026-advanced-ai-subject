"""Kiểm tra cách cộng RAM, phân loại dung lượng và ngoại suy theo mức khó."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import profile_stage4_resources as profiler
from src.imitation.audit import EvidenceError, load_json
from src.imitation.resources import disk_inventory, project_costs, summarize_phases, summarize_samples


def sample(elapsed, values, episode=None):
    return {"elapsed_s": elapsed, "episode_id": episode, "tree_rss_bytes": sum(values),
            "processes": [{"pid": i + 1, "create_time": 1, "rss_bytes": value} for i, value in enumerate(values)],
            "profiler_rss_bytes": 20, "system_available_bytes": 1000, "access_denied": 0, "vanished": 0}


def projection_fixture():
    episodes = [{"episode_id": "e0", "status": "COMPLETE", "case": {"split": "TRAIN", "tier": "Easy", "sequence": "SCS"}, "steps": 10},
                {"episode_id": "e1", "status": "COMPLETE", "case": {"split": "TRAIN", "tier": "Easy", "sequence": "SCS"}, "steps": 10},
                {"episode_id": "h0", "status": "COMPLETE", "case": {"split": "TRAIN", "tier": "Hard", "sequence": "SCTXrCS"}, "steps": 20}]
    timing = {"case_wall_seconds": {"e0": 1, "e1": 1, "h0": 10}, "batch_setup_and_finalize_seconds": 2}
    disk = {"total_bytes": 410, "bytes_by_episode": {"e0": 100, "e1": 100, "h0": 200},
            "bytes_by_category": {"sample_index_gzip": 0}}
    geometries = [{"split": "TRAIN", "tier": tier, "route_length_m": 100} for tier in ("Easy", "Hard")]
    return episodes, timing, disk, geometries


class TestResourceAccounting(unittest.TestCase):
    def test_peak_is_simultaneous_sum_not_sum_of_individual_peaks(self):
        result = summarize_samples([sample(0, [100, 10]), sample(.1, [10, 100])])
        self.assertEqual(result["peak_tree_rss_bytes"], 110)
        self.assertEqual(result["peak_profiler_rss_bytes"], 20)

    def test_duplicate_process_identity_is_rejected(self):
        row = sample(0, [10, 20])
        row["processes"][1].update(pid=1)
        with self.assertRaises(EvidenceError):
            summarize_samples([row])

    def test_reused_pid_across_samples_is_allowed(self):
        a, b = sample(0, [10]), sample(.1, [20])
        b["processes"][0]["create_time"] = 2
        self.assertEqual(summarize_samples([a, b])["peak_tree_rss_bytes"], 20)

    def test_missing_samples_and_clock_reversal_are_rejected(self):
        for rows in ([], [sample(1, [10]), sample(0, [10])]):
            with self.assertRaises(EvidenceError):
                summarize_samples(rows)

    def test_incorrect_sum_is_rejected(self):
        row = sample(0, [10])
        row["tree_rss_bytes"] = 20
        with self.assertRaises(EvidenceError):
            summarize_samples([row])

    def test_access_denied_is_exposed_instead_of_silently_omitted(self):
        row = sample(0, [10])
        row["access_denied"] = 1
        self.assertEqual(summarize_samples([row])["access_denied_observations"], 1)

    def test_disk_categories_and_case_bytes_preserve_total(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {"source_snapshot/a.py": b"12", "episodes/e0/launcher/stage1/sources/a.py": b"345",
                     "episodes/e0/launcher/stage1/agent_decisions.jsonl.gz": b"6789",
                     "episodes/e0/transitions.jsonl.gz": b"abcde", "raw_sample_index.jsonl.gz": b"f"}
            for name, payload in files.items():
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(payload)
            result = disk_inventory(root)
            self.assertEqual(result["total_bytes"], 15)
            self.assertEqual(result["bytes_by_episode"]["e0"], 12)
            self.assertEqual(result["bytes_by_category"]["shared_source_and_expert"], 2)
            self.assertEqual(result["bytes_by_category"]["per_episode_source_copies"], 3)


class TestResourceProjection(unittest.TestCase):
    def test_tier_weighting_does_not_overweight_easy_repeated_seed(self):
        result = project_costs(*projection_fixture(), [1, 2])
        observed, two_seeds = result["scenarios"][0], result["scenarios"][2]
        self.assertEqual(observed["estimated_wall_seconds"], 13)  # 2 + 1 + 10
        self.assertEqual(two_seeds["estimated_wall_seconds"], 24)  # shared setup counted once
        self.assertEqual(observed["estimated_total_bytes"], 310)
        self.assertEqual(two_seeds["estimated_total_bytes"], 610)

    def test_full_horizon_is_separate_scenario_not_a_confidence_bound(self):
        result = project_costs(*projection_fixture(), [1])
        self.assertGreater(result["scenarios"][1]["estimated_steps"], result["scenarios"][0]["estimated_steps"])
        self.assertIn("not a guaranteed upper bound", " ".join(result["limitations"]))

    def test_holdout_geometry_cannot_enter_projection(self):
        inputs = list(projection_fixture())
        inputs[3][0]["split"] = "TEST"
        with self.assertRaises(EvidenceError):
            project_costs(*inputs, [1])

    def test_missing_tier_cannot_be_silently_averaged(self):
        inputs = list(projection_fixture())
        inputs[0] = inputs[0][:2]
        with self.assertRaises(EvidenceError):
            project_costs(*inputs, [1])

    def test_invalid_seed_counts_are_rejected(self):
        for seeds in ([0], [-1], [True]):
            with self.assertRaises(EvidenceError):
                project_costs(*projection_fixture(), seeds)

    def test_missing_or_failed_phase_prevents_complete_measurement(self):
        episodes = [{"episode_id": "e0"}]
        events = [{"phase": "collecting", "episode_id": "e0", "observed_elapsed_s": 1}]
        with self.assertRaises(EvidenceError):
            summarize_phases(events, episodes, 10)
        events.append({"phase": "verified", "episode_id": "e0", "status": "FAILED", "observed_elapsed_s": 5})
        with self.assertRaises(EvidenceError):
            summarize_phases(events, episodes, 10)

    def test_case_timing_includes_replay_and_separates_batch_overhead(self):
        events = [{"phase": "collecting", "episode_id": "e0", "observed_elapsed_s": 2},
                  {"phase": "verified", "episode_id": "e0", "status": "COMPLETE", "observed_elapsed_s": 8}]
        result = summarize_phases(events, [{"episode_id": "e0"}], 10)
        self.assertEqual(result["case_wall_seconds"]["e0"], 6)
        self.assertEqual(result["batch_setup_and_finalize_seconds"], 4)


class TestResourceAttempt(unittest.TestCase):
    def test_failed_setup_is_durable_and_not_publishable(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(profiler, "ROOT", Path(temp)):
            with self.assertRaises(FileNotFoundError):
                profiler.profile("failed")
            output = Path(temp) / "runs/failed"
            self.assertEqual(load_json(output / "run_state.json")["status"], "FAILED")
            self.assertEqual(load_json(output / "run_integrity.json")["status"], "FAILED")
            with self.assertRaises(EvidenceError):
                profiler.publish("failed", "results/audits/stage4/g1/out")

    def test_existing_workload_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(profiler, "ROOT", Path(temp)):
            (Path(temp) / "runs/task_collect").mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                profiler.profile("task")
            self.assertFalse((Path(temp) / "runs/task").exists())

    def test_missing_pytorch_is_reported_without_installation_or_training(self):
        with tempfile.TemporaryDirectory() as temp:
            def fake_command(args):
                return {"command": args, "exit_code": 0, "stdout": "{}" if args[0] == "powershell" else "", "stderr": ""}
            with patch.object(profiler, "command_record", side_effect=fake_command) as invoked, \
                 patch.object(profiler, "package_version", return_value=None):
                _, result = profiler.inspect_environment(Path(temp))
            self.assertEqual(result["tensor_smoke"]["status"], "NOT_INSTALLED")
            self.assertEqual(result["training_tensor_environment_status"], "NOT_READY")
            self.assertEqual(result["training_updates"], 0)
            self.assertEqual(result["dependency_installations"], 0)
            self.assertFalse(any("install" in c.args[0] or "-c" in c.args[0] for c in invoked.call_args_list))


if __name__ == "__main__":
    unittest.main()
