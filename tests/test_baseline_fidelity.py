import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from baselines import CPSLAgent, ClusterSFLAgent, PCSFLAgent
from scenario_a_runner import (
    _compress_feature_tensor, _fedavg, _model_pca_embedding, _run_clustersfl_round,
    _run_pcsfl_round,
)


def synthetic_state(seed=3):
    rng = np.random.default_rng(seed)
    labels = rng.dirichlet(np.full(100, 0.2), size=10)
    return np.concatenate([
        rng.uniform(0.05, 2.0, size=(10, 1)),
        rng.uniform(1.0, 2.5, size=(10, 1)), labels,
    ], axis=1).astype(np.float32)


class TinyUSFL(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2)])

    def forward_partA(self, inputs, l1):
        for layer in range(l1):
            inputs = self.layers[layer](inputs)
        return inputs

    def forward_partB(self, inputs, l1, l2):
        for layer in range(l1, l2):
            inputs = self.layers[layer](inputs)
        return inputs

    def forward_partC(self, inputs, l2):
        for layer in range(l2, len(self.layers)):
            inputs = self.layers[layer](inputs)
        return inputs


class TinyProvider:
    def __init__(self):
        self.client_indices = {index: list(range(index + 2)) for index in range(4)}
        generator = torch.Generator().manual_seed(4)
        self.datasets = {
            index: TensorDataset(torch.randn(8, 4, generator=generator),
                                 torch.randint(0, 2, (8,), generator=generator))
            for index in range(4)
        }

    def get_client_dataloader(self, client_id, batch_size, shuffle=True, drop_last=True):
        return DataLoader(self.datasets[int(client_id)], batch_size=batch_size,
                          shuffle=shuffle, drop_last=drop_last)


class BaselineFidelityTests(unittest.TestCase):
    def test_cpsl_uses_capacity_gibbs_and_discrete_subchannels(self):
        state = synthetic_state()
        agent = CPSLAgent(seed=9, gibbs_iterations=8)
        action, info = agent.act(state, 3, np.asarray([3, 100e6]), deterministic=True)
        self.assertLessEqual(np.bincount(action["cluster"]).max(), 5)
        self.assertAlmostEqual(float(action["bw"].sum()), 1.0)
        np.testing.assert_allclose(action["bw"] * agent.subchannels,
                                   np.round(action["bw"] * agent.subchannels))
        self.assertGreater(np.std(action["bw"]), 0.0)
        self.assertIn("Gibbs", info["paper_algorithm"])
        calibration = agent.calibrate_split([state], [100e6])
        self.assertEqual(calibration["sample_count"], 1)
        self.assertEqual(agent.fixed_l2, agent.fixed_l1 + 1)

    def test_clustersfl_keeps_fixed_split_and_exposes_paper_controls(self):
        state = synthetic_state()
        volumes = np.arange(1, 11, dtype=np.float64)
        action, _ = ClusterSFLAgent().act(
            state, 3, np.asarray([3, 100e6]), data_volumes=volumes)
        self.assertTrue(np.all(action["l1"] == 3))
        self.assertTrue(np.all(action["l2"] == 4))
        self.assertLessEqual(np.bincount(action["cluster"]).max(), 5)
        self.assertEqual(int(action["top_worker"].sum()), len(np.unique(action["cluster"])))
        self.assertTrue(np.all((action["feature_compression"] >= 0.05)
                               & (action["feature_compression"] <= 1.0)))
        self.assertAlmostEqual(float(action["aggregation_weight"].sum()), 1.0)

    def test_real_topk_compression_changes_payload_and_is_finite(self):
        tensor = torch.arange(1.0, 101.0, requires_grad=True)
        compressed, retained = _compress_feature_tensor(tensor, 0.2)
        self.assertEqual(retained, 20)
        self.assertEqual(int(torch.count_nonzero(compressed)), 20)
        compressed.sum().backward()
        self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_pcsfl_has_recurrent_encoder_independent_heads_and_rewards(self):
        agent = PCSFLAgent(state_dim=102, batch_size=2, max_clients=10, max_migs=7)
        self.assertIsInstance(agent.online.encoder, nn.LSTM)
        self.assertIsNot(agent.online.cluster_head, agent.online.split_head)
        self.assertAlmostEqual(agent.paper_reward(0.0, 0.0), 0.0, places=7)
        self.assertLess(agent.paper_reward(0.05, 1.0), 0.0)
        action, _ = agent.act(
            synthetic_state(), 3, np.asarray([3, 100e6]), deterministic=True,
            data_volumes=np.ones(10), model_embedding=np.zeros(8, dtype=np.float32))
        self.assertTrue(np.all(action["cluster"] < 3))

    def test_pcsfl_cluster_training_and_weighted_hierarchy(self):
        model = TinyUSFL()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        action = {
            "cluster": np.asarray([0, 0, 1, 1]),
            "l1": np.asarray([1, 1, 1, 1]),
            "l2": np.asarray([2, 2, 2, 2]),
            "bw": np.full(4, 0.25),
        }
        result = _run_pcsfl_round(
            model, optimizer, TinyProvider(), {"batch_size": 2}, np.arange(4), action,
            torch.device("cpu"), local_steps=1, learning_rate=0.01)
        self.assertEqual(result["smashed_sizes"].shape, (4,))
        self.assertTrue(np.isfinite(result["pcsfl_clustering_factor"]))
        self.assertTrue(any(not torch.equal(before[key], model.state_dict()[key]) for key in before))

    def test_clustersfl_cluster_models_are_trained_and_aggregated(self):
        model = TinyUSFL()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        action = {
            "cluster": np.asarray([0, 0, 1, 1]),
            "l1": np.asarray([1, 1, 1, 1]),
            "l2": np.asarray([2, 2, 2, 2]),
            "bw": np.full(4, 0.25),
            "feature_compression": np.full(4, 0.5),
            "local_update_frequency": {0: 1, 1: 1},
            "aggregation_weight": np.asarray([0.1, 0.2, 0.3, 0.4]),
        }
        result = _run_clustersfl_round(
            model, optimizer, TinyProvider(), {"batch_size": 2}, np.arange(4), action,
            torch.device("cpu"), local_steps=2, learning_rate=0.01)
        self.assertTrue(np.all(result["smashed_sizes"] > 0.0))
        self.assertTrue(np.isfinite(result["train_loss"]))
        self.assertTrue(any(not torch.equal(before[key], model.state_dict()[key]) for key in before))

    def test_weighted_cloud_aggregation_and_model_pca_are_finite(self):
        first, second = TinyUSFL(), TinyUSFL()
        with torch.no_grad():
            for parameter in first.parameters():
                parameter.zero_()
            for parameter in second.parameters():
                parameter.fill_(2.0)
        optimizers = [torch.optim.SGD(first.parameters(), lr=0.1),
                      torch.optim.SGD(second.parameters(), lr=0.1)]
        _fedavg([first, second], optimizers, weights=[1.0, 3.0])
        for parameter in first.parameters():
            np.testing.assert_allclose(parameter.detach().numpy(), 1.5)
        embedding = _model_pca_embedding(first, dimensions=8)
        self.assertEqual(embedding.shape, (8,))
        self.assertTrue(np.isfinite(embedding).all())


if __name__ == "__main__":
    unittest.main()
