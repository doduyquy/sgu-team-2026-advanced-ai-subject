from __future__ import annotations

import unittest

from scripts.evaluate_stage5_s55 import checkpoint_transform_id


class TestStage5S55EvaluationMetadata(unittest.TestCase):
    def test_reads_versioned_checkpoint_transform_object(self) -> None:
        metadata = {"observation_transform": {"transform_id": "pilot_groupwise_physical_v1"}}
        self.assertEqual(checkpoint_transform_id(metadata), "pilot_groupwise_physical_v1")

    def test_reads_legacy_transform_string(self) -> None:
        metadata = {"observation_transform": "waypoint_xy_div50m_v1"}
        self.assertEqual(checkpoint_transform_id(metadata), "waypoint_xy_div50m_v1")

    def test_missing_transform_is_not_accepted(self) -> None:
        self.assertIsNone(checkpoint_transform_id({}))


if __name__ == "__main__":
    unittest.main()
