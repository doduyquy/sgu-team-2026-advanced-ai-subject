import json
import tempfile
import unittest
from pathlib import Path

from scripts.map_stage5_task_alignment_features import rows
from scripts.run_stage5_s55_training import write_completed_episodes


class TestStage5TaskAlignmentP0(unittest.TestCase):
    def test_feature_manifest_covers_masks_and_native_fields(self):
        features = list(rows())
        self.assertEqual(len(features), 409)
        self.assertEqual(len({feature[0] for feature in features}), 409)
        self.assertEqual(features[7][0], "yaw_rate_magnitude")
        self.assertEqual(features[11][0], "nav.current.bend_radius")
        self.assertEqual(features[259][-1], 315)
        self.assertEqual(features[314][-1], 322)
        self.assertEqual(features[325][-1], 385)
        self.assertEqual(features[384][-1], 404)

    def test_episode_ledger_survives_compact_summary_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            episodes = [{"outcome": "OUT_OF_ROAD", "clean_success": False}]
            write_completed_episodes(run, episodes)
            (run / "training_summary.json").write_text('{"status":"COMPLETE"}')
            path = run / "completed_episodes.json"
            self.assertEqual(json.loads(path.read_text()), episodes)
            with self.assertRaises(FileExistsError):
                write_completed_episodes(run, [])
            self.assertEqual(json.loads(path.read_text()), episodes)
