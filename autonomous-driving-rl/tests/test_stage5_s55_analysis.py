import unittest

from scripts.summarize_stage5_s55_results import training_episode_outcomes


class Stage5S55AnalysisTests(unittest.TestCase):
    def test_missing_episode_records_are_not_reported_as_zero_outcomes(self):
        counts, status = training_episode_outcomes({"completed_episodes": 24})
        self.assertIsNone(counts)
        self.assertEqual(status, "NOT_PERSISTED_BY_RUNNER")

    def test_persisted_episode_records_are_counted(self):
        counts, status = training_episode_outcomes({
            "episodes": [
                {"outcome": "OUT_OF_ROAD"},
                {"outcome": "TIMEOUT"},
                {"outcome": "OUT_OF_ROAD"},
            ]
        })
        self.assertEqual(counts, {"OUT_OF_ROAD": 2, "TIMEOUT": 1})
        self.assertEqual(status, "PERSISTED")


if __name__ == "__main__":
    unittest.main()
