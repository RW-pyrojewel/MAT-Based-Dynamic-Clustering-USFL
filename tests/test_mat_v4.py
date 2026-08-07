import unittest

import numpy as np
import torch

from models.mat_agent import MATAgent
from utils.mat_reward import MATRewardConfig, compute_mat_reward
from utils.runtime_profile import CausalRuntimeProfile


class MATV4Tests(unittest.TestCase):
    def setUp(self):
        np.random.seed(81)
        torch.manual_seed(81)
        labels = np.random.default_rng(81).dirichlet(np.full(100, 0.2), size=10)
        self.state = np.column_stack((
            np.geomspace(0.05, 2.0, 10), np.ones(10), labels)).astype(np.float32)
        self.edge = np.asarray([3.0, 100e6], dtype=np.float32)

    def agent(self, **kwargs):
        return MATAgent(
            state_dim=102, hidden_dim=32, ppo_epochs=1, minibatch_size=8,
            bandwidth_policy="hybrid_water_filling", policy_schema="hierarchical_rgs_v4",
            policy_state_mode="physical_runtime", device="cpu", **kwargs)

    def test_delay_reward_and_policy_ignore_label_distribution(self):
        clusters = np.asarray([0, 1] * 5)
        bandwidth = np.full(10, 0.1)
        reward_a, terms_a = compute_mat_reward(
            2.5, self.state[:, 2:], clusters, bandwidth, MATRewardConfig())
        changed = self.state.copy()
        changed[:, 2:] = np.roll(changed[:, 2:], 17, axis=1)
        reward_b, terms_b = compute_mat_reward(
            2.5, changed[:, 2:], clusters, bandwidth, MATRewardConfig())
        self.assertEqual(reward_a, -2.5)
        self.assertEqual(reward_a, reward_b)
        self.assertEqual(terms_a["reward_objective_mode"], "delay_only")
        agent = self.agent()
        first, _ = agent.act(self.state, 3, self.edge, client_ids=np.arange(10), deterministic=True)
        second, _ = agent.act(changed, 3, self.edge, client_ids=np.arange(10), deterministic=True)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertAlmostEqual(agent.get_value(self.state, self.edge), agent.get_value(changed, self.edge), places=7)
        self.assertTrue(np.isfinite(terms_b["label_kl_weighted_to_station"]))

    def test_one_step_credit_does_not_use_future_reward(self):
        agent = self.agent()
        infos = [{"value": 0.25}, {"value": 0.25}]
        common = dict(
            next_states=[self.state, self.state], dones=np.asarray([False, True]),
            edge_states=[self.edge, self.edge], next_edge_states=[self.edge, self.edge],
            policy_infos=infos, station_ids=[1, 1], epochs=[1, 2], one_step=True)
        first = agent._compute_gae(np.asarray([1.0, 2.0]), **common)
        second = agent._compute_gae(np.asarray([1.0, 2000.0]), **common)
        self.assertEqual(float(first[1][0]), float(second[1][0]))
        self.assertEqual(float(first[2][0]), 1.0)

    def test_restricted_growth_and_immediate_mig_response(self):
        agent = self.agent()
        counts = []
        for migs in (2, 5, 7):
            edge = np.asarray([migs, 100e6], dtype=np.float32)
            action, info = agent.act(
                self.state, migs, edge, client_ids=np.arange(10), deterministic=True)
            ordered = action["cluster"][info["decision_order"]]
            self.assertEqual(int(ordered[0]), 0)
            for index in range(1, len(ordered)):
                self.assertLessEqual(int(ordered[index]), int(ordered[:index].max()) + 1)
            self.assertEqual(sorted(np.unique(action["cluster"]).tolist()),
                             list(range(len(np.unique(action["cluster"])))))
            counts.append(len(np.unique(action["cluster"])))
        self.assertEqual(counts, [2, 5, 7])

    def test_split_prior_uses_real_payload_and_runtime_profile(self):
        agent = self.agent()
        action, info = agent.act(
            self.state, 3, self.edge, client_ids=np.arange(10), deterministic=True)
        for cluster in np.unique(action["cluster"]):
            members = action["cluster"] == cluster
            selected = int(info["split_action_ids"][cluster])
            self.assertEqual(selected, int(np.argmin(info["split_predicted_delays"][cluster])))
            self.assertTrue(np.all(action["l1"][members] == action["l1"][members][0]))
            self.assertTrue(np.all(action["l2"][members] == action["l2"][members][0]))
        before = agent.runtime_profile.diagnostics()["runtime_profile_observations"]
        replay_before = agent._evaluate_action(
            self.state, self.edge, action, 3, info["decision_order"], policy_info=info)
        agent.observe_runtime(action, {int(c): 0.1 + 0.01 * int(c)
                                      for c in np.unique(action["cluster"])}, epoch=1)
        self.assertGreater(agent.runtime_profile.diagnostics()["runtime_profile_observations"], before)
        replay_after = agent._evaluate_action(
            self.state, self.edge, action, 3, info["decision_order"], policy_info=info)
        torch.testing.assert_close(replay_before["cluster_log_probs"], replay_after["cluster_log_probs"])
        torch.testing.assert_close(replay_before["split_log_probs"], replay_after["split_log_probs"])

    def test_v4_batched_ppo_is_finite_and_keeps_target_frozen(self):
        agent = self.agent()
        actions, infos = [], []
        for _ in range(16):
            action, info = agent.act(self.state, 3, self.edge, client_ids=np.arange(10))
            actions.append(action)
            infos.append(info)
        targets_before = [value.detach().clone() for value in agent.target_encoder.parameters()]
        diagnostics = agent.update_policy(
            np.linspace(-0.2, -2.0, 16), [self.state] * 16, np.ones(16, dtype=bool),
            states=[self.state] * 16, actions=actions, edge_states=[self.edge] * 16,
            next_edge_states=[self.edge] * 16, available_migs=[3] * 16,
            policy_infos=infos, station_ids=[1, 2, 3, 1] * 4, epochs=list(range(16)),
            trajectory_ids=[(index, index % 3) for index in range(16)],
            policy_versions=[0] * 16,
        )
        self.assertTrue(all(np.isfinite(float(value)) for value in diagnostics.values()
                            if isinstance(value, (int, float))))
        self.assertEqual(diagnostics["target_drift_during_update"], 0.0)
        self.assertLessEqual(diagnostics["grad_norm_post_max"], 0.500001)
        self.assertTrue(any(not torch.equal(before, after) for before, after in zip(
            targets_before, agent.target_encoder.parameters())))

    def test_runtime_profile_is_finite_and_round_trippable(self):
        profile = CausalRuntimeProfile(max_clients=10, num_cut_layers=7, alpha=0.2)
        action = {
            "cluster": np.asarray([0, 0, 1]), "l1": np.asarray([3, 3, 5]),
            "l2": np.asarray([4, 4, 6]),
        }
        profile.update(action, {0: 0.3, 1: 0.1}, epoch=4)
        copied = CausalRuntimeProfile(max_clients=10, num_cut_layers=7, alpha=0.2)
        copied.load_state_dict(profile.state_dict())
        self.assertAlmostEqual(copied.predict(2, 3, 4), 0.3)
        self.assertTrue(np.isfinite(copied.predict(3, 2, 5)))


if __name__ == "__main__":
    unittest.main()
