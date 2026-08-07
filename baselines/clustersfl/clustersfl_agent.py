"""Adapted ClusterSFL baseline with MIG-aware K-Means and greedy split projection."""
import numpy as np

from interfaces.base_agent import BaseAgent


class ClusterSFLAgent(BaseAgent):
    """Label-aware worker clustering, top-worker selection and feature compression."""

    DEFAULT_PAYLOAD_BYTES = np.asarray(
        [12288.0, 262144.0, 262144.0, 131072.0, 65536.0, 32768.0, 2048.0], dtype=np.float64)

    def __init__(self, num_cut_layers=7, fixed_l1=3, fixed_l2=4,
                 max_workers_per_cluster=5, minimum_compression_ratio=0.05,
                 payload_bytes_by_l1=None, max_local_frequency=4):
        super().__init__(agent_name="PaperAdapted-ClusterSFL")
        if num_cut_layers < 3 or not 1 <= fixed_l1 < fixed_l2 < num_cut_layers:
            raise ValueError("invalid fixed ClusterSFL split")
        if max_workers_per_cluster < 1 or not 0.0 < minimum_compression_ratio <= 1.0:
            raise ValueError("invalid cluster capacity or compression floor")
        payloads = self.DEFAULT_PAYLOAD_BYTES if payload_bytes_by_l1 is None else payload_bytes_by_l1
        self.payload_bytes_by_l1 = np.asarray(payloads, dtype=np.float64)
        self.num_cut_layers = int(num_cut_layers)
        self.fixed_l1, self.fixed_l2 = int(fixed_l1), int(fixed_l2)
        self.max_workers_per_cluster = int(max_workers_per_cluster)
        self.minimum_compression_ratio = float(minimum_compression_ratio)
        self.max_local_frequency = int(max_local_frequency)

    @staticmethod
    def _distribution(values):
        values = np.maximum(np.asarray(values, dtype=np.float64), 1e-12)
        return values / values.sum(axis=-1, keepdims=True)

    @classmethod
    def _symmetric_kl(cls, first, second):
        first, second = cls._distribution(first), cls._distribution(second)
        return 0.5 * float(np.sum(first * np.log(first / second) + second * np.log(second / first)))

    def _cluster_workers(self, state, cluster_count):
        labels = self._distribution(state[:, 2:])
        count = len(state)
        if cluster_count == 1:
            return np.zeros(count, dtype=np.int64)
        pairwise = np.asarray([[self._symmetric_kl(labels[i], labels[j]) for j in range(count)]
                               for i in range(count)])
        seeds = [int(np.argmax(pairwise.mean(axis=1)))]
        while len(seeds) < cluster_count:
            distance = np.min(pairwise[:, seeds], axis=1)
            distance[seeds] = -1.0
            seeds.append(int(np.argmax(distance)))
        clusters = np.full(count, -1, dtype=np.int64)
        for cluster, client in enumerate(seeds):
            clusters[client] = cluster
        ideal = np.full(labels.shape[1], 1.0 / labels.shape[1])
        remaining = [index for index in range(count) if clusters[index] < 0]
        for client in sorted(remaining, key=lambda index: -pairwise[index, seeds].mean()):
            costs = np.full(cluster_count, np.inf)
            for cluster in range(cluster_count):
                members = np.flatnonzero(clusters == cluster)
                if len(members) >= self.max_workers_per_cluster:
                    continue
                aggregate = labels[np.append(members, client)].mean(axis=0)
                label_cost = self._symmetric_kl(aggregate, ideal)
                timing = np.std(1.0 / np.maximum(state[np.append(members, client), 1], 1e-6))
                costs[cluster] = label_cost + 0.02 * timing
            clusters[client] = int(np.argmin(costs))
        return clusters

    def _compression_and_frequency(self, state, clusters, bandwidth_hz):
        count = len(state)
        share = 1.0 / count
        efficiency = np.maximum(np.log2(1.0 + 10.0 * np.maximum(state[:, 0], 0.0)), 1e-9)
        compute = 1e-3 * (self.fixed_l1 + 1.0) / np.maximum(state[:, 1], 1e-6)
        # Compression changes the l1 boundary only; l2 remains present in both
        # directions.  The factor two accounts for symmetric UL and DL.
        l1_payload = self.payload_bytes_by_l1[self.fixed_l1]
        l2_payload = self.payload_bytes_by_l1[self.fixed_l2]
        tx_full = (16.0 * (l1_payload + l2_payload)
                   / np.maximum(share * bandwidth_hz * efficiency, 1e-12))
        ratios = np.ones(count, dtype=np.float64)
        cluster_time = {}
        for cluster in np.unique(clusters):
            members = np.flatnonzero(clusters == cluster)
            target = float(np.min(compute[members] + tx_full[members]))
            fixed_l2_tx = 16.0 * l2_payload / np.maximum(
                share * bandwidth_hz * efficiency[members], 1e-12)
            scalable_l1_tx = 16.0 * l1_payload / np.maximum(
                share * bandwidth_hz * efficiency[members], 1e-12)
            ratios[members] = np.clip(
                (target - compute[members] - fixed_l2_tx) / np.maximum(scalable_l1_tx, 1e-12),
                self.minimum_compression_ratio, 1.0)
            cluster_time[int(cluster)] = float(np.max(
                compute[members] + fixed_l2_tx + ratios[members] * scalable_l1_tx))
        slowest = max(cluster_time.values())
        frequency = {cluster: int(np.clip(np.floor(slowest / max(delay, 1e-12)), 1,
                                          self.max_local_frequency))
                     for cluster, delay in cluster_time.items()}
        return ratios, frequency, cluster_time

    def act(self, active_clients_state, available_migs, edge_state=None, deterministic=True,
            data_volumes=None, **_):
        state = np.asarray(active_clients_state, dtype=np.float64)
        count = len(state)
        if state.ndim != 2 or state.shape[1] < 3 or count == 0 or available_migs < 1:
            raise ValueError("ClusterSFL requires channel, compute and label state")
        desired = int(np.ceil(count / self.max_workers_per_cluster))
        cluster_count = min(desired, int(available_migs), count)
        clusters = self._cluster_workers(state, cluster_count)
        bandwidth_hz = float(edge_state[1]) if edge_state is not None else 1.0
        ratios, frequencies, cluster_times = self._compression_and_frequency(state, clusters, bandwidth_hz)
        top_workers = np.zeros(count, dtype=bool)
        for cluster in np.unique(clusters):
            members = np.flatnonzero(clusters == cluster)
            top_workers[members[np.argmax(state[members, 0])]] = True
        volumes = np.ones(count, dtype=np.float64) if data_volumes is None else np.asarray(data_volumes, dtype=np.float64)
        # The paper control remains available as a diagnostic, but the primary
        # fixed-round comparison gives every active client the same mini-batch
        # budget. Aggregation therefore follows samples actually consumed.
        aggregation = np.ones(count, dtype=np.float64)
        cloud_mass = float(aggregation.sum())
        aggregation /= max(aggregation.sum(), 1e-12)
        return {
            "cluster": clusters,
            "virtual_cluster": clusters.copy(),
            "l1": np.full(count, self.fixed_l1, dtype=np.int64),
            "l2": np.full(count, self.fixed_l2, dtype=np.int64),
            "bw": np.full(count, 1.0 / count, dtype=np.float64),
            "feature_compression": ratios,
            "top_worker": top_workers,
            "local_update_frequency": frequencies,
            "aggregation_weight": aggregation,
            "cloud_aggregation_mass": cloud_mass,
        }, {
            "paper_algorithm": "ClusterSFL-KL/top-worker/compression/local-frequency",
            "feature_compression_mean": float(ratios.mean()),
            "top_worker_count": int(top_workers.sum()),
            "cluster_completion_proxy": float(max(cluster_times.values())),
        }

    def step(self, active_clients_state, available_migs):
        action, _ = self.act(active_clients_state, available_migs)
        return action["cluster"], action["l1"], action["l2"], action["bw"]
