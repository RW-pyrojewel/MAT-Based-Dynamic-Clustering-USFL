import unittest

import numpy as np
import torch

from models.mat_agent import MATAgent
from models.mat_components import CausalDeviceDecoder, ClusterSplitHead
from utils.mat_reward import MATRewardConfig, compute_mat_reward
from utils.trajectory_buffer import MATTrajectoryBuffer


class MATRefactorTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(11)
        torch.manual_seed(11)
        self.state = np.concatenate(
            [
                np.linspace(0.2, 2.0, 10)[:, None],
                np.linspace(1.0, 2.5, 10)[:, None],
                np.full((10, 100), 0.01),
            ],
            axis=1,
        ).astype(np.float32)
        self.edge = np.asarray([2.0, 100e6], dtype=np.float32)

    def test_deterministic_bandwidth_is_equal_and_replay_is_exact(self):
        agent = MATAgent(state_dim=102, device="cpu")
        deterministic_action, _ = agent.act(
            self.state,
            2,
            self.edge,
            client_ids=np.arange(10),
            deterministic=True,
        )
        np.testing.assert_allclose(deterministic_action["bw"], np.full(10, 0.1), atol=1e-6)
        action, policy_info = agent.act(self.state, 2, self.edge, deterministic=False)
        evaluated = agent._evaluate_action(
            self.state,
            self.edge,
            action,
            2,
            policy_info["decision_order"],
        )
        np.testing.assert_allclose(
            evaluated["cluster_log_probs"].detach().numpy(),
            policy_info["cluster_log_probs"],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            evaluated["bandwidth_log_probs"].detach().numpy(),
            policy_info["bandwidth_log_probs"],
            atol=1e-4,
        )
        np.testing.assert_allclose(
            evaluated["split_log_probs"].detach().numpy(),
            policy_info["split_log_probs"],
            atol=1e-5,
        )
        self.assertAlmostEqual(float(action["bw"].sum()), 1.0, places=6)
        self.assertGreaterEqual(float(action["bw"].min()), 0.01 - 1e-6)

    def test_deterministic_actions_are_equivariant_to_input_permutation(self):
        agent = MATAgent(state_dim=102, device="cpu")
        client_ids = np.arange(10)
        base_action, _ = agent.act(
            self.state,
            2,
            self.edge,
            client_ids=client_ids,
            deterministic=True,
        )
        permutation = np.asarray([7, 2, 9, 0, 4, 1, 8, 6, 3, 5])
        permuted_action, _ = agent.act(
            self.state[permutation],
            2,
            self.edge,
            client_ids=client_ids[permutation],
            deterministic=True,
        )
        inverse = np.argsort(permutation)
        for key in ("cluster", "l1", "l2"):
            np.testing.assert_array_equal(base_action[key], permuted_action[key][inverse])
        np.testing.assert_allclose(base_action["bw"], permuted_action["bw"][inverse], atol=1e-6)

    def test_component_ppo_update_is_finite(self):
        agent = MATAgent(state_dim=102, device="cpu", ppo_epochs=1, minibatch_size=1)
        action, policy_info = agent.act(self.state, 2, self.edge, deterministic=False)
        buffer = MATTrajectoryBuffer()
        buffer.append(
            self.state,
            self.edge,
            action,
            -0.5,
            self.state,
            self.edge,
            True,
            policy_info,
            2,
            1,
            1,
        )
        kwargs = buffer.as_ppo_kwargs()
        diagnostics = agent.update_policy(
            kwargs.pop("rewards"),
            kwargs.pop("next_states"),
            kwargs.pop("dones"),
            **kwargs,
        )
        self.assertTrue(diagnostics)
        self.assertTrue(all(np.isfinite(value) for value in diagnostics.values()))
        self.assertEqual(diagnostics["grad_norm"], diagnostics["grad_norm_pre"])
        self.assertLessEqual(diagnostics["grad_norm_post"], agent.max_grad_norm + 1e-5)
        self.assertIn("station_1_return_mean", diagnostics)
        self.assertIn("station_1_explained_variance", diagnostics)

    def test_future_action_does_not_change_earlier_log_probs(self):
        decoder = CausalDeviceDecoder(hidden_dim=16, num_migs=3, num_heads=4, num_layers=1)
        encoded = torch.randn(1, 6, 16)
        clusters, bandwidths, _, _, _ = decoder.act(encoded, available_migs=3, deterministic=True)
        original, _, _ = decoder.evaluate_actions(encoded, clusters, bandwidths, 3)
        changed_clusters = clusters.clone()
        changed_clusters[:, 4] = (changed_clusters[:, 4] + 1) % 3
        changed, _, _ = decoder.evaluate_actions(encoded, changed_clusters, bandwidths, 3)
        torch.testing.assert_close(original[:, :4], changed[:, :4])

    def test_cluster_split_attention_is_isolated_by_membership(self):
        head = ClusterSplitHead(hidden_dim=16, num_migs=2, num_cut_layers=7, num_heads=4)
        encoded = torch.randn(1, 4, 16)
        clusters = torch.tensor([[0, 0, 1, 1]])
        bandwidths = torch.full((1, 4), 0.25)
        l1, l2, original, _, _ = head.act(encoded, clusters, bandwidths, deterministic=True)
        changed_encoded = encoded.clone()
        changed_encoded[:, 2:] += 100.0
        changed, _, _ = head.evaluate_actions(changed_encoded, clusters, bandwidths, l1, l2)
        replay, _, _ = head.evaluate_actions(encoded, clusters, bandwidths, l1, l2)
        torch.testing.assert_close(replay[:, 0], original[:, 0])
        torch.testing.assert_close(changed[:, 0], original[:, 0])

    def test_edge_state_normalization_uses_physical_references(self):
        agent = MATAgent(state_dim=102, device="cpu", nominal_bandwidth_hz=100e6)
        normalized = agent._normalise_edge_state(np.asarray([7.0, 20e6], dtype=np.float32))
        torch.testing.assert_close(normalized, torch.tensor([[1.0, 0.2]]))

    def test_reward_components_are_dimensionless(self):
        labels = np.full((4, 100), 0.01)
        clusters = np.asarray([0, 0, 1, 1])
        bandwidths = np.full(4, 0.25)
        reward, terms = compute_mat_reward(0.5, labels, clusters, bandwidths, MATRewardConfig())
        self.assertAlmostEqual(terms["delay_normalized"], 0.5)
        self.assertAlmostEqual(terms["kl_normalized"], 0.0)
        self.assertAlmostEqual(reward, -0.25)
        self.assertEqual(terms["cluster_size_violation"], 0.0)

    def test_gae_does_not_cross_station_boundaries(self):
        agent = MATAgent(state_dim=102, device="cpu")
        agent.get_value = lambda state, edge: 0.0
        policy_infos = [{"value": 0.0} for _ in range(4)]
        advantages, returns, station_stats = agent._compute_gae(
            rewards=np.asarray([1.0, 10.0, 1.0, 10.0]),
            next_states=[self.state] * 4,
            dones=np.asarray([False, False, True, True]),
            edge_states=[self.edge] * 4,
            next_edge_states=[self.edge] * 4,
            policy_infos=policy_infos,
            station_ids=[1, 2, 1, 2],
            epochs=[1, 1, 2, 2],
        )
        self.assertTrue(np.isfinite(advantages).all())
        self.assertAlmostEqual(float(returns[0]), 1.0 + 0.99 * 0.95, places=5)
        self.assertAlmostEqual(float(returns[1]), 10.0 + 0.99 * 0.95 * 10.0, places=5)
        for indices in ([0, 2], [1, 3]):
            self.assertAlmostEqual(float(advantages[indices].mean()), 0.0, places=6)
            self.assertAlmostEqual(float(advantages[indices].std()), 1.0, places=6)
        self.assertAlmostEqual(station_stats[1]["return_mean"], float(returns[[0, 2]].mean()), places=6)
        self.assertAlmostEqual(station_stats[2]["return_std"], float(returns[[1, 3]].std()), places=6)

    def test_single_sample_station_normalization_is_finite(self):
        agent = MATAgent(state_dim=102, device="cpu")
        policy_infos = [{"value": 0.0} for _ in range(3)]
        advantages, returns, station_stats = agent._compute_gae(
            rewards=np.asarray([1.0, 10.0, 100.0]),
            next_states=[self.state] * 3,
            dones=np.asarray([True, True, True]),
            edge_states=[self.edge] * 3,
            next_edge_states=[self.edge] * 3,
            policy_infos=policy_infos,
            station_ids=[1, 2, 3],
            epochs=[1, 1, 1],
        )
        np.testing.assert_array_equal(advantages, np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(returns, np.asarray([1.0, 10.0, 100.0]))
        for station_id in (1, 2, 3):
            self.assertEqual(station_stats[station_id]["return_std"], 0.0)
            self.assertEqual(station_stats[station_id]["return_scale"], 1.0)
        self.assertEqual(agent._explained_variance([1.0], [50.0]), 0.0)

    def test_clipped_huber_value_loss_has_finite_gradient(self):
        new_value = torch.tensor(3.0, requires_grad=True)
        value_loss, clip_fraction = MATAgent._value_loss(
            new_value,
            torch.tensor(0.0),
            torch.tensor(1.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
            clip_ratio=0.2,
            huber_delta=1.0,
        )
        value_loss.backward()
        self.assertTrue(torch.isfinite(value_loss))
        self.assertTrue(torch.isfinite(new_value.grad))
        self.assertEqual(float(clip_fraction), 1.0)

    def test_extreme_station_return_scale_keeps_ppo_finite(self):
        agent = MATAgent(
            state_dim=102,
            hidden_dim=32,
            device="cpu",
            ppo_epochs=2,
            minibatch_size=6,
            max_grad_norm=0.5,
        )
        buffer = MATTrajectoryBuffer()
        rewards = {1: (-1.0, -1.2), 2: (-1e6, -1.2e6), 3: (-0.5, -0.7)}
        for epoch in (1, 2):
            for station_id in (1, 2, 3):
                edge = self.edge.copy()
                if station_id == 2:
                    edge[1] *= 0.2
                action, policy_info = agent.act(self.state, 2, edge, deterministic=False)
                buffer.append(
                    self.state,
                    edge,
                    action,
                    rewards[station_id][epoch - 1],
                    self.state,
                    edge,
                    epoch == 2,
                    policy_info,
                    2,
                    station_id,
                    epoch,
                )
        kwargs = buffer.as_ppo_kwargs()
        diagnostics = agent.update_policy(
            kwargs.pop("rewards"), kwargs.pop("next_states"), kwargs.pop("dones"), **kwargs,
        )
        self.assertTrue(all(np.isfinite(value) for value in diagnostics.values()))
        self.assertLessEqual(diagnostics["grad_norm_post"], agent.max_grad_norm + 1e-5)
        self.assertEqual(diagnostics["grad_norm"], diagnostics["grad_norm_pre"])
        self.assertGreater(
            abs(diagnostics["station_2_return_mean"]),
            1e4 * abs(diagnostics["station_1_return_mean"]),
        )
        for station_id in (1, 2, 3):
            self.assertIn(f"station_{station_id}_return_std", diagnostics)
            self.assertIn(f"station_{station_id}_explained_variance", diagnostics)

    def test_buffer_preserves_structured_policy_information(self):
        agent = MATAgent(state_dim=102, device="cpu")
        action, policy_info = agent.act(self.state, 2, self.edge, deterministic=False)
        buffer = MATTrajectoryBuffer()
        buffer.append(
            self.state,
            self.edge,
            action,
            -0.5,
            self.state,
            self.edge,
            True,
            policy_info,
            2,
            1,
            1,
        )
        exported = buffer.as_ppo_kwargs()
        self.assertEqual(exported["station_ids"], [1])
        self.assertEqual(exported["epochs"], [1])
        np.testing.assert_array_equal(
            exported["policy_infos"][0]["decision_order"],
            policy_info["decision_order"],
        )


if __name__ == "__main__":
    unittest.main()
