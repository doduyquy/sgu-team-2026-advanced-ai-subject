"""Coverage, holdout isolation and frozen-source checks for candidate evaluation."""

from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_candidate_test import check_snapshot, make_jobs


class CandidateTestSchedule(unittest.TestCase):
    def cases(self):
        return [{"case_id": f"test/{i}", "split": "TEST", "protocol_order_index": i}
                for i in range(1, 61)]

    def test_both_agents_use_the_same_full_ordered_cases_for_three_seeds(self):
        jobs = make_jobs(["random", "mppi"], self.cases())
        self.assertEqual(len(jobs), 6)
        self.assertEqual(sum(len(j["cases"]) for j in jobs), 360)
        for agent in ("stage0_random_naive", "stage3_mppi"):
            cohort = [j for j in jobs if j["agent_id"] == agent]
            self.assertEqual([j["agent_seed"] for j in cohort], [101, 202, 303])
            self.assertTrue(all(j["cases"] == self.cases() for j in cohort))

    def test_partial_reordered_duplicate_or_wrong_split_test_suites_are_rejected(self):
        cases = self.cases()
        wrong_split = [{**c, "split": "TRAIN"} for c in cases]
        for invalid in (cases[:1], list(reversed(cases)), [cases[0]] * 60, wrong_split):
            with self.subTest(invalid=invalid[:1]), self.assertRaises(ValueError):
                make_jobs(["random"], invalid)

    def test_smoke_cannot_be_used_to_shorten_test_cases(self):
        with self.assertRaises(ValueError):
            make_jobs(["mppi"], self.cases()[:1], smoke=True)
        jobs = make_jobs(["random", "mppi"], [{"case_id": "train/1", "split": "TRAIN"}], smoke=True)
        self.assertEqual([j["agent_seed"] for j in jobs], [101, 101])

    def test_frozen_source_tampering_is_rejected(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            config = source / "config.json"
            config.write_text('{"horizon":30}', encoding="utf-8")
            hashes = {"config.json": hashlib.sha256(config.read_bytes()).hexdigest()}
            check_snapshot(source, hashes)
            config.write_text('{"horizon":20}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Source snapshot changed"):
                check_snapshot(source, hashes)


if __name__ == "__main__":
    unittest.main()
