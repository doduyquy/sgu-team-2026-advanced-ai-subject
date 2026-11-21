import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/stage5/p2g_action_consequence_20261121/evidence.json"


class P2GActionConsequenceEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_scope_is_read_only_train_diagnostic(self):
        self.assertFalse(self.evidence["limits"]["teacher_or_external_controller_used"])
        self.assertFalse(self.evidence["limits"]["validation_test_accessed"])
        self.assertTrue(self.evidence["limits"]["counterfactuals_are_not_training_data"])
        self.assertFalse(self.evidence["limits"]["benchmark_eligible"])
        self.assertEqual(self.evidence["source_geometry"]["split"], "TRAIN/PILOT_TRAIN")

    def test_physical_states_pass_replay_validation(self):
        self.assertEqual(len(self.evidence["states"]), 5)
        self.assertEqual(len(self.evidence["state_replay_validation"]), 5)
        self.assertTrue(all(row["passed"] for row in self.evidence["state_replay_validation"]))
        labels = " ".join(row["label"] for row in self.evidence["states"])
        for expected in ("reset_stationary", "moving_positive_progress", "already_progressed",
                         "near_curve", "lateral_heading_error", "reduced_safety_margin"):
            self.assertIn(expected, labels)

    def test_action_grid_is_complete_and_censoring_is_not_outcome(self):
        rows = self.evidence["counterfactuals"]
        self.assertEqual(len(rows), 45)
        for state in self.evidence["states"]:
            current = [r for r in rows if r["state_label"] == state["label"]]
            self.assertEqual(len(current), 9)
            actions = {(r["action"][0], r["action"][1]) for r in current}
            self.assertEqual(actions, {(s, t) for s in (-0.35, 0.0, 0.35) for t in (-0.5, 0.0, 0.5)})
        for row in rows:
            if row["probe_censored"]:
                self.assertIsNone(row["terminal_outcome"])
            else:
                self.assertIsNotNone(row["terminal_outcome"])

    def test_checkpoint_identity_and_action_support_are_present(self):
        checkpoints = self.evidence["checkpoint_provenance"]
        self.assertEqual(set(checkpoints), {"P2D_A_10000", "P2D_B_10000", "P2F_A_10000", "P2F_B_10000"})
        self.assertTrue(all(len(row["model_sha256"]) == 64 for row in checkpoints.values()))
        diag = self.evidence["critic_actor_diagnostics"]
        self.assertEqual(len(diag["state_model_diagnostics"]), 20)
        self.assertEqual(set(diag["replay_action_support"]), set(checkpoints))


if __name__ == "__main__":
    unittest.main()
