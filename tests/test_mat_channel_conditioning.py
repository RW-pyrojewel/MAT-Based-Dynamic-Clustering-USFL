import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models.mat_agent import MATAgent
from mat_channel_probe_runner import _trace
from utils.channel_diagnostics import spearman_correlation
from utils.trajectory_buffer import MATTrajectoryBuffer


class MATChannelConditioningTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(23)
        torch.manual_seed(23)
        self.state = np.column_stack(
            (
                np.geomspace(0.01, 3.0, 10),
                np.linspace(1.0, 2.5, 10),
                np.full((10, 4), 0.25),
            )
        ).astype(np.float32)
        self.edge = np.asarray([3.0, 100e6], dtype=np.float32)

    def test_explicit_transform_and_legacy_compatibility(self):
        explicit = MATAgent(state_dim=6, hidden_dim=32, channel_conditioning="explicit")
        prepared = explicit._prepare_client_state(self.state).squeeze(0).cpu().numpy()
        expected = np.log1p(10.0 * self.state[:, 0]) / np.log(11.0)
        np.testing.assert_allclose(prepared[:, 0], expected, rtol=1e-6)
        np.testing.assert_allclose(prepared[:, 1], self.state[:, 1] / 2.5)

        legacy = MATAgent(state_dim=6, hidden_dim=32, channel_conditioning="legacy")
        np.testing.assert_allclose(
            legacy._prepare_client_state(self.state).squeeze(0).cpu().numpy(),
            self.state,
        )
        self.assertIsNone(legacy._channel_features(legacy._prepare_client_state(self.state)))

    def test_probe_channel_quality_is_not_bound_to_client_position(self):
        correlations = []
        for seed in (449, 457, 461):
            traces = _trace(seed)
            for station in (1, 2, 3):
                mean_channel = np.mean(
                    [traces[station][round_index][:, 0] for round_index in range(20)], axis=0,
                )
                correlations.append(spearman_correlation(np.arange(10, dtype=float), mean_channel))
        self.assertLess(max(abs(value) for value in correlations), 0.40)
    def test_channel_sidecar_has_gradient_and_counterfactual_response(self):
        agent = MATAgent(state_dim=6, hidden_dim=32, channel_conditioning="explicit",
                         bandwidth_policy="joint_dirichlet")
        action, info = agent.act(
            self.state, 3, self.edge, client_ids=np.arange(10), deterministic=True
        )
        with torch.no_grad():
            agent.bandwidth_head.physical_weight.fill_(2.0)
        original = agent.evaluate_bandwidth_prefix_means(
            self.state, self.edge, action, 3, info["decision_order"]
        )
        permuted_state = self.state.copy()
        permuted_state[:, 0] = self.state[::-1, 0]
        changed = agent.evaluate_bandwidth_prefix_means(
            permuted_state, self.edge, action, 3, info["decision_order"]
        )
        required_delta = (
            1.0 / np.log2(1.0 + 10.0 * permuted_state[:, 0])
            - 1.0 / np.log2(1.0 + 10.0 * self.state[:, 0])
        )
        # The shared encoder also reacts to the counterfactual channel; the explicit scalar path must dominate.
        self.assertGreater(spearman_correlation(required_delta, changed - original), 0.80)

        prepared, _ = agent._encode(self.state, self.edge)
        context = agent._bandwidth_context(prepared, self.edge)
        alpha, _, _, feature = agent.bandwidth_head.parameters_for(
            context, torch.as_tensor(action["cluster"]).reshape(1, -1), prepared[..., :1]
        )
        means = agent.bandwidth_head._to_bandwidth(alpha / alpha.sum(dim=-1, keepdim=True))
        gradient = torch.autograd.grad((means * feature).sum(), agent.bandwidth_head.physical_weight)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().item()), 0.0)

    def test_legacy_ignores_sidecar_weight(self):
        agent = MATAgent(state_dim=6, hidden_dim=32, channel_conditioning="legacy",
                         bandwidth_policy="sequential_gaussian")
        action, info = agent.act(
            self.state, 3, self.edge, client_ids=np.arange(10), deterministic=True
        )
        before = agent.evaluate_bandwidth_prefix_means(
            self.state, self.edge, action, 3, info["decision_order"]
        )
        with torch.no_grad():
            agent.device_decoder.bandwidth_channel_head.weight.fill_(100.0)
        after = agent.evaluate_bandwidth_prefix_means(
            self.state, self.edge, action, 3, info["decision_order"]
        )
        np.testing.assert_allclose(before, after, atol=1e-7)

    def test_checkpoint_round_trip(self):
        agent = MATAgent(
            state_dim=6,
            hidden_dim=32,
            channel_conditioning="explicit",
            component_balanced_ppo=True,
            bandwidth_policy="joint_dirichlet",
        )
        agent.policy_version = 4
        agent.update_count = 3
        with torch.no_grad():
            agent.bandwidth_head.physical_weight.fill_(0.75)
        expected_action, expected_info = agent.act(
            self.state, 3, self.edge, client_ids=np.arange(10), deterministic=True
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mat.pt"
            agent.save_checkpoint(path)
            loaded = MATAgent.load_checkpoint(path)
        actual_action, actual_info = loaded.act(
            self.state, 3, self.edge, client_ids=np.arange(10), deterministic=True
        )
        self.assertEqual(loaded.policy_version, 4)
        self.assertEqual(loaded.update_count, 3)
        self.assertTrue(loaded.component_balanced_ppo)
        for key in expected_action:
            np.testing.assert_allclose(expected_action[key], actual_action[key])
        np.testing.assert_allclose(
            expected_info["bandwidth_latent_means"],
            actual_info["bandwidth_latent_means"],
        )
        for online, target in zip(loaded.encoder.parameters(), loaded.target_encoder.parameters()):
            self.assertTrue(torch.isfinite(online).all())
            self.assertTrue(torch.isfinite(target).all())
        self.assertTrue(all(not parameter.requires_grad for parameter in loaded.target_encoder.parameters()))

    def test_joint_dirichlet_constraints_masking_and_order_invariance(self):
        agent = MATAgent(state_dim=6, hidden_dim=32, bandwidth_policy="joint_dirichlet")
        action, info = agent.act(self.state, 3, self.edge, deterministic=True)
        self.assertAlmostEqual(float(action["bw"].sum()), 1.0, places=6)
        self.assertGreaterEqual(float(action["bw"].min()), 0.01)
        self.assertEqual(np.asarray(info["bandwidth_log_probs"]).shape, (1,))
        prepared = agent._prepare_client_state(self.state)
        context = agent._bandwidth_context(prepared, self.edge)
        changed = prepared.clone()
        changed[..., 0] = torch.flip(changed[..., 0], dims=(1,))
        torch.testing.assert_close(context, agent._bandwidth_context(changed, self.edge))
        reverse = agent.evaluate_bandwidth_prefix_means(
            self.state, self.edge, action, 3, info["decision_order"][::-1].copy())
        np.testing.assert_allclose(reverse, info["bandwidth_latent_means"], atol=1e-7)

    def test_joint_log_prob_is_finite_for_extreme_channels(self):
        agent = MATAgent(state_dim=6, hidden_dim=32)
        extreme = self.state.copy()
        extreme[:, 0] = np.geomspace(0.0 + 1e-12, 1e6, 10)
        action, info = agent.act(extreme, 3, self.edge)
        evaluated = agent._evaluate_action(extreme, self.edge, action, 3, info["decision_order"])
        self.assertTrue(torch.isfinite(evaluated["bandwidth_log_probs"]).all())
        self.assertTrue(torch.isfinite(evaluated["bandwidth_entropy"]).all())
        self.assertAlmostEqual(float(action["bw"].sum()), 1.0, places=6)

    def test_bandwidth_only_one_step_does_not_update_cluster_or_split_heads(self):
        agent = MATAgent(state_dim=6, hidden_dim=32, ppo_epochs=1, minibatch_size=1,
                         bandwidth_policy="joint_dirichlet")
        action, info = agent.act(self.state, 3, self.edge)
        buffer = MATTrajectoryBuffer()
        buffer.append(self.state, self.edge, action, -1.0, self.state, self.edge, False,
                      info, 3, 1, 1, trajectory_id=(0, 1), policy_version=0)
        cluster_before = [parameter.detach().clone() for parameter in agent.device_decoder.cluster_head.parameters()]
        split_before = [parameter.detach().clone() for parameter in agent.split_head.parameters()]
        kwargs = buffer.as_ppo_kwargs()
        diagnostics = agent.update_policy(
            kwargs.pop("rewards"), kwargs.pop("next_states"), kwargs.pop("dones"),
            actor_components=("bandwidth",), one_step_advantage=True, **kwargs)
        for before, after in zip(cluster_before, agent.device_decoder.cluster_head.parameters()):
            torch.testing.assert_close(before, after)
        for before, after in zip(split_before, agent.split_head.parameters()):
            torch.testing.assert_close(before, after)
        self.assertEqual(diagnostics["cluster_policy_grad_norm"], 0.0)
        self.assertEqual(diagnostics["split_policy_grad_norm"], 0.0)
    def test_component_balanced_diagnostics_are_finite(self):
        agent = MATAgent(
            state_dim=6,
            hidden_dim=32,
            ppo_epochs=1,
            minibatch_size=1,
            component_balanced_ppo=True,
            bandwidth_policy="joint_dirichlet",
        )
        action, info = agent.act(self.state, 3, self.edge)
        buffer = MATTrajectoryBuffer()
        buffer.append(
            self.state,
            self.edge,
            action,
            -1.0,
            self.state,
            self.edge,
            True,
            info,
            3,
            1,
            1,
            trajectory_id=(0, 1),
            policy_version=0,
        )
        kwargs = buffer.as_ppo_kwargs()
        diagnostics = agent.update_policy(
            kwargs.pop("rewards"), kwargs.pop("next_states"), kwargs.pop("dones"), **kwargs
        )
        for key in (
            "cluster_policy_loss",
            "bandwidth_policy_loss",
            "split_policy_loss",
            "cluster_policy_grad_norm",
            "bandwidth_policy_grad_norm",
            "split_policy_grad_norm",
            "bandwidth_actor_gradient_share",
        ):
            self.assertIn(key, diagnostics)
            self.assertTrue(np.isfinite(diagnostics[key]))


if __name__ == "__main__":
    unittest.main()
