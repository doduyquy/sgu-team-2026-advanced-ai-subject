import unittest

import numpy as np

from src.training.stage5.terminal_aware_replay import TerminalAwareSampler


class TerminalAwareReplayTests(unittest.TestCase):
    def test_zero_terminal_falls_back_to_uniform(self):
        sampler = TerminalAwareSampler(seed=4)
        batch = sampler.sample(size=20, priorities=np.ones(8), true_terminal=np.zeros(8, bool),
                               kinds=np.zeros(8, np.int8), optimizer_update=0)
        np.testing.assert_allclose(batch.probabilities, np.full(8, 1 / 8))
        self.assertEqual(batch.terminal_candidates, 0)

    def test_one_two_and_multiple_terminals_are_oversampled_without_reweighting_timeout(self):
        for count in (1, 2, 4):
            n = 8
            terminal = np.zeros(n, bool); terminal[:count] = True
            kinds = np.zeros(n, np.int8); kinds[:count] = 1
            kinds[-1] = 3  # truncation is never eligible
            batch = TerminalAwareSampler(seed=12).sample(size=20000, priorities=np.ones(n),
                true_terminal=terminal, kinds=kinds, optimizer_update=100)
            self.assertEqual(batch.terminal_candidates, count)
            expected_terminal_probability = .20 + .80 * count / n
            self.assertAlmostEqual(float(np.mean(batch.terminal_mask)), expected_terminal_probability, delta=.015)
            self.assertAlmostEqual(float(batch.probabilities.sum()), 1.0, places=6)
            self.assertTrue(np.all(batch.normalized_importance_weights <= 1.0))
            self.assertEqual(float(batch.probabilities[-1]), (1.0 - .20) / n)

    def test_terminal_priority_is_capped_and_beta_schedule_is_frozen(self):
        sampler = TerminalAwareSampler(seed=8, td_error_cap=10.0)
        terminal = np.array([True, True, False])
        kinds = np.array([1, 2, 0], np.int8)
        batch = sampler.sample(size=5000, priorities=np.array([100.0, 2.0, 1.0]),
                               true_terminal=terminal, kinds=kinds, optimizer_update=9000)
        self.assertGreater(batch.probabilities[0], batch.probabilities[1])
        capped_batch = sampler.sample(size=10, priorities=np.array([10000.0, 10.0, 1.0]),
                                      true_terminal=terminal, kinds=kinds, optimizer_update=9000)
        np.testing.assert_allclose(capped_batch.probabilities[0], capped_batch.probabilities[1])
        self.assertEqual(sampler.beta(0), .4)
        self.assertEqual(sampler.beta(9000), 1.0)
        self.assertGreater(batch.effective_sample_size, 0)


if __name__ == "__main__":
    unittest.main()
