import copy
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
        agent._target_value = lambda state, edge: 0.0
        policy_infos = [{"value": 0.0} for _ in range(4)]
        advantages, returns, _, _ = agent._compute_gae(
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

    def test_shared_encoder_target_is_frozen_and_synced(self):
        agent = MATAgent(state_dim=102, hidden_dim=32, device="cpu", ppo_epochs=1)
        self.assertFalse(hasattr(agent, "critic_encoder"))
        self.assertTrue(all(not p.requires_grad for p in agent.target_encoder.parameters()))
        self.assertTrue(all(not p.requires_grad for p in agent.target_value_head.parameters()))
        with torch.no_grad():
            next(agent.encoder.parameters()).add_(1.0)
        self.assertFalse(torch.equal(next(agent.encoder.parameters()), next(agent.target_encoder.parameters())))
        agent.sync_target()
        for online, target in zip(agent.encoder.parameters(), agent.target_encoder.parameters()):
            torch.testing.assert_close(online, target)

    def test_tokenwise_value_aggregation_is_permutation_invariant(self):
        agent = MATAgent(state_dim=102, hidden_dim=32, device="cpu")
        encoded = torch.randn(2, 10, 32)
        value = agent._aggregate_token_values(agent.value_head, encoded)
        permuted = agent._aggregate_token_values(agent.value_head, encoded[:, torch.randperm(10)])
        torch.testing.assert_close(value, permuted)

    def test_gae_does_not_cross_episode_boundaries(self):
        agent = MATAgent(state_dim=102, hidden_dim=32, device="cpu")
        agent._target_value = lambda state, edge: 0.0
        advantages, returns, targets, residuals = agent._compute_gae(
            np.asarray([1.0, 100.0, 1.0, 100.0]), [self.state] * 4,
            np.asarray([False, False, True, True]), [self.edge] * 4, [self.edge] * 4,
            [{"value": 0.0}] * 4, [1, 1, 1, 1], [1, 1, 2, 2],
            [(0, 1), (1, 1), (0, 1), (1, 1)])
        self.assertAlmostEqual(float(returns[0]), 1.0 + 0.99 * 0.95, places=5)
        self.assertAlmostEqual(float(returns[1]), 100.0 + 0.99 * 0.95 * 100.0, places=4)
        self.assertTrue(np.isfinite(np.concatenate((advantages, returns, targets, residuals))).all())

    def test_buffer_rejects_mixed_policy_versions(self):
        agent = MATAgent(state_dim=102, hidden_dim=32, device="cpu")
        action, info = agent.act(self.state, 2, self.edge)
        buffer = MATTrajectoryBuffer()
        buffer.append(self.state, self.edge, action, 0.0, self.state, self.edge, True, info, 2, 1, 1,
                      trajectory_id=(0, 1), policy_version=0)
        with self.assertRaises(ValueError):
            buffer.append(self.state, self.edge, action, 0.0, self.state, self.edge, True, info, 2, 1, 2,
                          trajectory_id=(0, 1), policy_version=1)

    def test_full_batch_gradient_accumulation_matches_direct_batch(self):
        base = MATAgent(state_dim=102, hidden_dim=32, device="cpu", ppo_epochs=1, minibatch_size=2)
        action1, info1 = base.act(self.state, 2, self.edge)
        action2, info2 = base.act(self.state * 0.9, 2, self.edge)
        buffer = MATTrajectoryBuffer()
        for epoch, (state, action, info, reward) in enumerate(((self.state, action1, info1, -0.5),
                                                               (self.state * 0.9, action2, info2, 2.0)), 1):
            buffer.append(state, self.edge, action, reward, state, self.edge, True, info, 2, 1, epoch,
                          trajectory_id=(0, 1), policy_version=0)
        accumulated = copy.deepcopy(base)
        accumulated.minibatch_size = 1
        direct = copy.deepcopy(base)
        kwargs1, kwargs2 = buffer.as_ppo_kwargs(), buffer.as_ppo_kwargs()
        np.random.seed(5)
        accumulated.update_policy(kwargs1.pop("rewards"), kwargs1.pop("next_states"), kwargs1.pop("dones"), **kwargs1)
        np.random.seed(5)
        direct.update_policy(kwargs2.pop("rewards"), kwargs2.pop("next_states"), kwargs2.pop("dones"), **kwargs2)
        for left, right in zip(accumulated._parameters(), direct._parameters()):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)

    def test_extreme_station_reward_stays_finite_and_target_does_not_drift(self):
        agent = MATAgent(state_dim=102, hidden_dim=32, device="cpu", ppo_epochs=1, minibatch_size=2)
        buffer = MATTrajectoryBuffer()
        for station, reward in ((1, -1.0), (2, -1e6), (3, -2.0)):
            action, info = agent.act(self.state, 2, self.edge)
            buffer.append(self.state, self.edge, action, reward, self.state, self.edge, True, info, 2,
                          station, 1, trajectory_id=(0, station), policy_version=0)
        kwargs = buffer.as_ppo_kwargs()
        diagnostics = agent.update_policy(kwargs.pop("rewards"), kwargs.pop("next_states"), kwargs.pop("dones"), **kwargs)
        numeric = [value for value in diagnostics.values() if isinstance(value, (int, float, np.number))]
        self.assertTrue(np.isfinite(numeric).all())
        self.assertLessEqual(diagnostics["grad_norm_post_max"], 0.500001)
        self.assertEqual(diagnostics["target_drift_during_update"], 0.0)
        for online, target in zip(agent.encoder.parameters(), agent.target_encoder.parameters()):
            torch.testing.assert_close(online, target)

    def test_zero_variance_single_sample_is_finite(self):
        agent = MATAgent(state_dim=102, hidden_dim=32, device="cpu")
        agent._target_value = lambda state, edge: 0.0
        values = agent._compute_gae(np.asarray([0.0]), [self.state], np.asarray([True]), [self.edge], [self.edge],
                                    [{"value": 0.0}], [1], [1], [(0, 1)])
        self.assertTrue(all(np.isfinite(item).all() for item in values))
        self.assertEqual(float(values[0][0]), 0.0)

if __name__ == "__main__":
    unittest.main()
