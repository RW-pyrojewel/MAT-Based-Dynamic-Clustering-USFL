import unittest

import numpy as np

from envs.liquid_airan_env import LiquidAIRANEnv
from utils.channel_diagnostics import (
    channel_bandwidth_metrics,
    max_transmission_delay,
    normalized_channel_quality,
    oracle_bandwidth,
    required_airtime,
    spearman_correlation,
)


class ChannelDiagnosticsTests(unittest.TestCase):
    def test_bidirectional_usfl_matches_section_3_1_1(self):
        env = object.__new__(LiquidAIRANEnv)
        env.max_migs = 3
        env.current_migs = 2
        env.current_bandwidth = 20e6
        clusters = np.asarray([0, 0, 1])
        uplink = np.asarray([0.2, 0.3, 0.5])
        l1 = np.asarray([1000.0, 2000.0, 3000.0])
        l2 = np.asarray([4000.0, 5000.0, 6000.0])
        gains = np.asarray([0.2, 0.5, 1.0])
        downlink_rates = np.asarray([5e6, 6e6, 7e6])
        result = env.calc_bidirectional_transmission_delay(
            clusters, uplink, l1, l2, gains, available_migs=2,
            bandwidth_hz=20e6, downlink_rates_bps=downlink_rates)
        payload = l1 + l2
        expected_ul_rates = uplink * 20e6 * np.log2(1.0 + 10.0 * gains)
        expected_ul = payload * 8.0 / expected_ul_rates
        expected_dl = payload * 8.0 / downlink_rates
        np.testing.assert_allclose(result["uplink_delays"], expected_ul)
        np.testing.assert_allclose(result["downlink_delays"], expected_dl)
        np.testing.assert_allclose(result["client_delays"], expected_ul + expected_dl)
        self.assertAlmostEqual(result["cluster_delays"][0], float((expected_ul + expected_dl)[:2].max()))
        self.assertAlmostEqual(result["cluster_delays"][1], float((expected_ul + expected_dl)[2]))

        larger_l2 = env.calc_bidirectional_transmission_delay(
            clusters, uplink, l1, l2 * 2.0, gains, available_migs=2,
            bandwidth_hz=20e6, downlink_rates_bps=downlink_rates)
        self.assertTrue(np.all(larger_l2["client_delays"] > result["client_delays"]))

    def test_base_station_downlink_is_equal_and_finite_at_zero_gain(self):
        env = object.__new__(LiquidAIRANEnv)
        env.current_bandwidth = 30e6
        rates = env.base_station_downlink_rates(np.asarray([0.0, 0.5, 2.0]))
        self.assertTrue(np.isfinite(rates).all())
        self.assertTrue((rates > 0.0).all())
        expected = 10e6 * np.log2(1.0 + 10.0 * np.asarray([0.0, 0.5, 2.0]))
        np.testing.assert_allclose(rates[1:], expected[1:])

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
