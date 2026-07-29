"""Adapted ClusterSFL baseline with MIG-aware K-Means and greedy split projection."""
import numpy as np

from interfaces.base_agent import BaseAgent


class ClusterSFLAgent(BaseAgent):
    """Re-cluster channel/compute features greedily at every communication round."""

    _UPLINK_BYTES = np.asarray([262144, 262144, 131072, 65536, 32768, 32768], dtype=np.float64)

    def __init__(self, num_cut_layers=7, kmeans_iterations=12):
        super().__init__(agent_name="Adapted-ClusterSFL")
        if num_cut_layers < 2:
            raise ValueError("num_cut_layers must be at least two")
        self.num_cut_layers = int(num_cut_layers)
        self.kmeans_iterations = int(kmeans_iterations)

    @staticmethod
    def _normalise(features):
        scale = features.std(axis=0, keepdims=True)
        return (features - features.mean(axis=0, keepdims=True)) / np.maximum(scale, 1e-6)

    def _kmeans(self, features, cluster_count):
        client_count = len(features)
        if cluster_count == 1:
            return np.zeros(client_count, dtype=np.int64)
        normalised = self._normalise(features)
        order = np.argsort(normalised[:, 0] + 0.1 * normalised[:, 1], kind="stable")
        initial = np.linspace(0, client_count - 1, cluster_count, dtype=np.int64)
        centroids = normalised[order[initial]].copy()
        labels = np.zeros(client_count, dtype=np.int64)
        for _ in range(self.kmeans_iterations):
            distances = ((normalised[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            for cluster_id in range(cluster_count):
                members = labels == cluster_id
                if members.any():
                    centroids[cluster_id] = normalised[members].mean(axis=0)
                else:
                    farthest = distances.min(axis=1).argmax()
                    centroids[cluster_id] = normalised[farthest]
                    labels[farthest] = cluster_id
        return labels

    def _preferred_pair(self, channel_gain, compute_capacity, client_count, bandwidth_hz):
        candidate_count = min(self.num_cut_layers - 1, len(self._UPLINK_BYTES))
        indices = np.arange(candidate_count)
        spectral_efficiency = np.log2(1.0 + 10.0 * max(float(channel_gain), 1e-6))
        transmission = self._UPLINK_BYTES[indices] * 8.0 * client_count / max(bandwidth_hz * spectral_efficiency, 1.0)
        client_compute = (indices + 1.0) / max(float(compute_capacity), 1e-6) * 1e-3
        l1 = int(np.argmin(transmission + client_compute))
        return l1, l1 + 1

    def act(self, active_clients_state, available_migs, edge_state=None, deterministic=True):
        state = np.asarray(active_clients_state, dtype=np.float64)
        client_count = len(state)
        if client_count == 0 or available_migs < 1:
            raise ValueError("ClusterSFL requires at least one client and one available MIG")
        cluster_count = min(client_count, int(available_migs))
        clusters = self._kmeans(state[:, :2], cluster_count)
        bandwidth_hz = float(edge_state[1]) if edge_state is not None else 1.0
        l1 = np.empty(client_count, dtype=np.int64)
        l2 = np.empty(client_count, dtype=np.int64)
        for cluster_id in range(cluster_count):
            members = np.flatnonzero(clusters == cluster_id)
            bottleneck = members[np.argmin(state[members, 0])]
            pair = self._preferred_pair(state[bottleneck, 0], state[bottleneck, 1], client_count, bandwidth_hz)
            l1[members], l2[members] = pair
        return {
            "cluster": clusters,
            "l1": l1,
            "l2": l2,
            "bw": np.full(client_count, 1.0 / client_count, dtype=np.float64),
        }, None

    def step(self, active_clients_state, available_migs):
        action, _ = self.act(active_clients_state, available_migs)
        return action["cluster"], action["l1"], action["l2"], action["bw"]
