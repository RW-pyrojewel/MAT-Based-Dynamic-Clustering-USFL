import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from baselines.cpsl.cpsl_agent import CPSLAgent
from models.mat_agent import MATAgent
from models.usfl_networks import ResNet18_USFL
from scenario_a_runner import _physical_channel_diagnostics
from utils.channel_diagnostics import offset_aware_oracle_bandwidth
from utils.trajectory_buffer import MATTrajectoryBuffer


class MATHybridAllocatorTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(71)
        torch.manual_seed(71)
        self.state = np.column_stack((
            np.geomspace(0.01, 3.0, 10), np.linspace(1.0, 2.5, 10), np.full((10, 4), 0.25),
        )).astype(np.float32)
        self.edge = np.asarray([3.0, 100e6], dtype=np.float32)
        self.clusters = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        self.payload = np.asarray([262144, 131072, 65536, 32768, 262144, 65536, 32768, 131072, 65536, 32768], dtype=float)
        self.compute = {0: 0.12, 1: 0.08, 2: 0.15}

    @staticmethod
    def objective(costs, clusters, compute, bandwidth, total_bandwidth=100e6):
        offsets = np.asarray([compute[int(cluster)] for cluster in clusters])
        return float(np.max(offsets + costs / (total_bandwidth * bandwidth)))

    def test_offset_aware_solution_is_feasible_and_beats_equal_and_random(self):
        spectral = np.log2(1.0 + 10.0 * self.state[:, 0])
        costs = self.payload * 8.0 / spectral
        result = offset_aware_oracle_bandwidth(costs, self.clusters, self.compute, 100e6, 0.01)
        self.assertAlmostEqual(float(result.sum()), 1.0, places=10)
        self.assertGreaterEqual(float(result.min()), 0.01 - 1e-10)
        equal = np.full(10, 0.1)
        optimum = self.objective(costs, self.clusters, self.compute, result)
        self.assertLessEqual(optimum, self.objective(costs, self.clusters, self.compute, equal) + 1e-10)
        rng = np.random.default_rng(9)
        for _ in range(2000):
            candidate = 0.01 + 0.9 * rng.dirichlet(np.ones(10))
            self.assertLessEqual(optimum, self.objective(costs, self.clusters, self.compute, candidate) + 1e-7)

    def test_zero_payload_and_permutation_equivariance(self):
        zero = offset_aware_oracle_bandwidth(np.zeros(10), self.clusters, self.compute, 100e6, 0.01)
        np.testing.assert_allclose(zero, 0.1)
        costs = self.payload * 8.0 / np.log2(1.0 + 10.0 * self.state[:, 0])
        original = offset_aware_oracle_bandwidth(costs, self.clusters, self.compute, 100e6, 0.01)
        permutation = np.asarray([7, 1, 9, 0, 5, 2, 8, 4, 6, 3])
        changed = offset_aware_oracle_bandwidth(
            costs[permutation], self.clusters[permutation], self.compute, 100e6, 0.01)
        np.testing.assert_allclose(changed, original[permutation], rtol=1e-9, atol=1e-10)

    def test_hybrid_action_replay_allocator_and_no_bandwidth_actor(self):
        agent = MATAgent(state_dim=6, hidden_dim=32, bandwidth_policy="hybrid_water_filling", ppo_epochs=1)
        self.assertIsNone(agent.bandwidth_head)
        action, info = agent.act(self.state, 3, self.edge, deterministic=True)
        np.testing.assert_allclose(action["bw"], 0.1)
        np.testing.assert_allclose(info["split_conditioning_bandwidth"], 0.1)
        self.assertFalse(np.asarray(info["bandwidth_mask"]).any())
        allocation, diagnostics = agent.allocate_bandwidth(
            action, self.state, self.payload, self.compute, self.edge[1])
        action["bw"] = allocation
        replay = agent._evaluate_action(self.state, self.edge, action, 3, info["decision_order"])
        np.testing.assert_allclose(replay["split_log_probs"].detach().numpy(), info["split_log_probs"], atol=1e-6)
        self.assertLessEqual(diagnostics["hybrid_total_delay_ms"], diagnostics["equal_total_delay_ms"] + 1e-6)
        buffer = MATTrajectoryBuffer()
        buffer.append(self.state, self.edge, action, -1.0, self.state, self.edge, True, info, 3, 1, 1,
                      trajectory_id=(0, 1), policy_version=0)
        kwargs = buffer.as_ppo_kwargs()
        update = agent.update_policy(kwargs.pop("rewards"), kwargs.pop("next_states"), kwargs.pop("dones"), **kwargs)
        self.assertEqual(update["bandwidth_policy_grad_norm"], 0.0)
        self.assertEqual(update["bandwidth_head_grad_norm_pre"], 0.0)

    def test_allocator_treats_downlink_as_client_offset(self):
        agent = MATAgent(state_dim=6, hidden_dim=32, bandwidth_policy="hybrid_water_filling")
        action, _ = agent.act(self.state, 3, self.edge, deterministic=True)
        downlink = np.linspace(0.01, 0.1, len(self.payload))
        allocation, diagnostics = agent.allocate_bandwidth(
            action, self.state, self.payload, self.compute, self.edge[1],
            downlink_delays=downlink)
        costs = self.payload * 8.0 / np.log2(1.0 + 10.0 * self.state[:, 0])
        offsets = np.asarray([self.compute[int(cluster)] for cluster in self.clusters]) + downlink
        equal = np.full(len(self.payload), 0.1)
        optimum = float(np.max(offsets + costs / (self.edge[1] * allocation)))
        equal_value = float(np.max(offsets + costs / (self.edge[1] * equal)))
        self.assertLessEqual(optimum, equal_value + 1e-9)
        np.testing.assert_allclose(diagnostics["client_downlink_delays"], downlink)

    def test_schema_v4_hybrid_round_trip(self):
        agent = MATAgent(state_dim=6, hidden_dim=32, bandwidth_policy="hybrid_water_filling")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid.pt"
            agent.save_checkpoint(path)
            loaded = MATAgent.load_checkpoint(path)
        self.assertEqual(loaded.bandwidth_policy, "hybrid_water_filling")
        self.assertIsNone(loaded.bandwidth_head)
        expected, _ = agent.act(self.state, 3, self.edge, deterministic=True)
        actual, _ = loaded.act(self.state, 3, self.edge, deterministic=True)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key])

    def test_schema_v2_joint_and_schema_v1_legacy_load(self):
        joint = MATAgent(
            state_dim=6, hidden_dim=32, bandwidth_policy="joint_dirichlet",
            policy_schema="legacy_v3", policy_state_mode="legacy_full")
        with tempfile.TemporaryDirectory() as directory:
            v3_path = Path(directory) / "joint_v3.pt"
            v2_path = Path(directory) / "joint_v2.pt"
            v1_path = Path(directory) / "legacy_v1.pt"
            joint.save_checkpoint(v3_path)
            payload = torch.load(v3_path, map_location="cpu", weights_only=False)
            for key in (
                "policy_schema", "policy_state_mode", "max_clients", "runtime_profile_alpha",
                "physics_prior_scale", "execution_batch_size", "execution_local_steps", "physics_only"):
                payload["config"].pop(key, None)
            payload.pop("runtime_profile", None)
            payload["schema_version"] = 2
            torch.save(payload, v2_path)
            loaded_v2 = MATAgent.load_checkpoint(v2_path, load_optimizer=False)
            payload["schema_version"] = 1
            payload.pop("bandwidth_head")
            torch.save(payload, v1_path)
            loaded_v1 = MATAgent.load_checkpoint(v1_path, load_optimizer=False)
        self.assertEqual(loaded_v2.bandwidth_policy, "joint_dirichlet")
        self.assertIsNotNone(loaded_v2.bandwidth_head)
        self.assertEqual(loaded_v1.bandwidth_policy, "sequential_gaussian")
        self.assertIsNone(loaded_v1.bandwidth_head)

    def test_split_payload_matches_real_smashed_tensor(self):
        model = ResNet18_USFL(num_classes=100)
        images = torch.zeros(1, 3, 32, 32)
        actual = np.asarray([
            model.forward_partA(images, layer).numel() * images.element_size()
            for layer in range(6)
        ])
        np.testing.assert_array_equal(
            actual, np.asarray([12288, 262144, 262144, 131072, 65536, 32768]))

    def test_baseline_none_policy_info_remains_compatible(self):
        baseline = CPSLAgent()
        action, info = baseline.act(self.state, 3, self.edge)
        scalars, clients = _physical_channel_diagnostics(
            baseline, self.state, self.edge, action, info, 3,
            np.full(10, 4096.0), float(self.edge[1]), 0.01)
        self.assertEqual(len(clients), 10)
        self.assertTrue(np.isfinite(list(scalars.values())).all())


if __name__ == "__main__":
    unittest.main()
