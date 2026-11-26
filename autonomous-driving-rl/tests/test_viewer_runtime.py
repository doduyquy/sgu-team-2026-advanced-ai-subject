"""Fast viewing must preserve step semantics and recover only finished cases."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

from src.visualization.viewer_runtime import PlaybackSpeed, FrameRate, set_native_view_mode, load_viewer_progress


class PlaybackTests(unittest.TestCase):
    def test_smooth_renders_physics_substeps_at_requested_fps_and_fast_disables_limiter(self):
        limiter = SimpleNamespace(fps=None, enable=Mock(), disable=Mock())
        engine = SimpleNamespace(force_fps=limiter)
        set_native_view_mode(engine, "smooth", 60)
        limiter.enable.assert_called_once()
        self.assertEqual(limiter.fps, 60)
        set_native_view_mode(engine, "fast", 60)
        limiter.disable.assert_called_once()

    def test_fps_counts_drawn_frames_not_decision_steps(self):
        rate = FrameRate(0.)
        for frame in range(1, 61):
            rate.record(frame / 60.)
        self.assertEqual(rate.fps, 60.)

    def test_speed_changes_waiting_without_skipping_control_steps(self):
        playback = PlaybackSpeed(1)
        self.assertAlmostEqual(playback.delay(.01), .09)
        playback.set_speed(5)
        self.assertAlmostEqual(playback.delay(.01), .01)
        self.assertEqual(playback.delay(.03), 0)
        self.assertEqual(playback.label, "5×")
        playback.set_speed(0)
        self.assertEqual(playback.delay(0), 0)
        self.assertEqual(playback.label, "MAX")

    def test_invalid_speeds_are_rejected(self):
        for value in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                PlaybackSpeed(value)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.schedule = [{"agent_id": "stage2_bfs", "case_id": f"val/{i}", "split": "VALIDATION"} for i in range(4)]
        self.source = [{"path": "planner.py", "sha256": "same"}]
        self.manifest = {"kind": "NONCANONICAL_VISUAL_RERUN", "status": "INTERRUPTED",
                         "schedule": self.schedule, "source": self.source,
                         "carried_episode_statuses": {"1": "COMPLETE"}}

    def write(self, statuses):
        (self.path / "session.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        rows = [{"episode": index, "agent_id": "stage2_bfs",
                 "case": {k: v for k, v in self.schedule[index - 1].items() if k != "agent_id"},
                 "status": status} for index, status in statuses]
        (self.path / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def test_carries_previous_resume_complete_and_skipped_but_restarts_interrupted(self):
        self.write([(2, "COMPLETE"), (3, "SKIPPED"), (4, "INTERRUPTED")])
        self.assertEqual(load_viewer_progress(self.path, self.schedule, self.source),
                         {1: "COMPLETE", 2: "COMPLETE", 3: "SKIPPED"})

    def test_preview_limit_is_not_carried_as_completed(self):
        self.write([(2, "PREVIEW_LIMIT")])
        self.assertEqual(load_viewer_progress(self.path, self.schedule, self.source), {1: "COMPLETE"})

    def test_live_source_and_schedule_mismatches_are_rejected(self):
        self.write([])
        with self.assertRaises(ValueError):
            load_viewer_progress(self.path, self.schedule[::-1], self.source)
        with self.assertRaises(ValueError):
            load_viewer_progress(self.path, self.schedule, [{"sha256": "changed"}])
        self.manifest["status"] = "RUNNING"
        self.write([])
        with self.assertRaises(ValueError):
            load_viewer_progress(self.path, self.schedule, self.source)

    def test_mismatched_case_cannot_be_marked_finished(self):
        self.write([(2, "COMPLETE")])
        path = self.path / "episodes.jsonl"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["case"]["case_id"] = "val/wrong-case"
        path.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_viewer_progress(self.path, self.schedule, self.source)

    def test_old_next_agent_gap_is_preserved_as_skipped(self):
        self.schedule[3]["agent_id"] = "stage2_astar"
        self.write([(2, "SKIPPED"), (4, "COMPLETE")])
        path = self.path / "episodes.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[-1]["agent_id"] = "stage2_astar"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.assertEqual(load_viewer_progress(self.path, self.schedule, self.source),
                         {1: "COMPLETE", 2: "SKIPPED", 3: "SKIPPED", 4: "COMPLETE"})


if __name__ == "__main__":
    unittest.main()
