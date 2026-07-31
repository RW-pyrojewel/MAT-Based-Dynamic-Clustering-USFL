import unittest

import numpy as np

from utils.channel_diagnostics import (
    channel_bandwidth_metrics,
    max_transmission_delay,
    normalized_channel_quality,
    oracle_bandwidth,
    required_airtime,
    spearman_correlation,
)


class ChannelDiagnosticsTests(unittest.TestCase):
    def test_channel_transform_handles_zero_and_extreme_gain(self):
        transformed = normalized_channel_quality(np.asarray([0.0, 0.1, 1.0, 1e6]))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue(np.all(np.diff(transformed) > 0.0))
        self.assertEqual(float(transformed[0]), 0.0)
        self.assertAlmostEqual(float(transformed[2]), 1.0)

    def test_oracle_minimizes_max_delay_with_floor(self):
        costs = np.asarray([1.0, 2.0, 8.0, 0.5])
        oracle = oracle_bandwidth(costs, min_share=0.05)
        equal = np.full(4, 0.25)
        self.assertAlmostEqual(float(oracle.sum()), 1.0, places=10)
        self.assertGreaterEqual(float(oracle.min()), 0.05 - 1e-10)
        self.assertLess(
            max_transmission_delay(costs, oracle, 1.0),
            max_transmission_delay(costs, equal, 1.0),
        )
        # Random feasible alternatives must not beat the water-filling solution.
        rng = np.random.default_rng(9)
        for _ in range(200):
            residual = rng.dirichlet(np.ones(4)) * 0.8
            candidate = residual + 0.05
            self.assertLessEqual(
                max_transmission_delay(costs, oracle, 1.0),
                max_transmission_delay(costs, candidate, 1.0) + 1e-8,
            )

    def test_metrics_reward_bandwidth_for_required_airtime(self):
        gains = np.asarray([0.1, 0.5, 1.0, 2.0])
        payload = np.full(4, 4096.0)
        airtime = required_airtime(payload, gains)
        oracle = oracle_bandwidth(airtime, min_share=0.01)
        metrics = channel_bandwidth_metrics(gains, payload, oracle, 100e6)
        self.assertGreater(spearman_correlation(airtime, oracle), 0.99)
        self.assertLess(metrics["channel_bandwidth_spearman"], -0.99)
        self.assertAlmostEqual(metrics["oracle_gap_closure"], 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
