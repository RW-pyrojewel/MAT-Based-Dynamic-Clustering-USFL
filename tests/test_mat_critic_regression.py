import json
import unittest

from mat_critic_regression_runner import run_critic_regression


class MATCriticRegressionTests(unittest.TestCase):
    def test_requires_at_least_three_cycles(self):
        with self.assertRaises(ValueError):
            run_critic_regression(cycles=2, write_report=False)

    def test_repeated_collect_update_sync_stays_stable(self):
        report = run_critic_regression(
            cycles=3,
            train_rounds=4,
            holdout_rounds=3,
            client_count=10,
            hidden_dim=16,
            ppo_epochs=1,
            minibatch_size=8,
            station_2_scale=100.0,
            device="cpu",
            write_report=False,
        )
        self.assertTrue(report["passed"])
        json.dumps(report)
        self.assertEqual(len(report["cycles"]), 3)
        for expected_version, cycle in enumerate(report["cycles"]):
            self.assertEqual(cycle["policy_version_before"], expected_version)
            self.assertEqual(cycle["policy_version_after"], expected_version + 1)
            self.assertTrue(cycle["passed"])
            self.assertTrue(all(cycle["gates"].values()))
            self.assertEqual(cycle["rollout_target_drift"], 0.0)
            self.assertLessEqual(cycle["diagnostics"]["grad_norm_post_max"], 0.500001)
            self.assertEqual(cycle["diagnostics"]["target_drift_during_update"], 0.0)


if __name__ == "__main__":
    unittest.main()
