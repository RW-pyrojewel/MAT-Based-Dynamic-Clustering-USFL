"""Adapted CPSL baseline with a static virtual-cluster topology."""
from __future__ import annotations

import numpy as np

from interfaces.base_agent import BaseAgent


class CPSLAgent(BaseAgent):
    """Fixed split with Gibbs clustering and greedy subchannel allocation.

    CPSL reuses all subchannels for clusters that execute sequentially. Scenario A has
    concurrent MIGs and one global spectrum budget, so its marginal-delay allocation
    rule is applied to that global pool. The controller otherwise follows Algorithms
    2--4 of the paper rather than the former round-robin/equal-bandwidth surrogate.
    """

    DEFAULT_PAYLOAD_BYTES = np.asarray(
        [12288.0, 262144.0, 262144.0, 131072.0, 65536.0, 32768.0, 2048.0], dtype=np.float64)

    def __init__(self, fixed_clusters=2, fixed_l1=3, fixed_l2=4, cluster_capacity=5,
                 subchannels=100, gibbs_iterations=40, gibbs_temperature=0.05,
                 payload_bytes_by_l1=None, seed=0):
        super().__init__(agent_name="PaperAdapted-CPSL")
        if fixed_clusters < 1 or cluster_capacity < 1 or subchannels < 1:
            raise ValueError("cluster and subchannel counts must be positive")
        if not 1 <= fixed_l1 < fixed_l2:
            raise ValueError("fixed split pair must keep Part A non-empty")
        payloads = self.DEFAULT_PAYLOAD_BYTES if payload_bytes_by_l1 is None else payload_bytes_by_l1
        payloads = np.asarray(payloads, dtype=np.float64)
        if fixed_l2 >= len(payloads) or not np.isfinite(payloads).all() or (payloads < 0).any():
            raise ValueError("payload profile does not cover the fixed split pair")
        self.fixed_clusters = int(fixed_clusters)
        self.fixed_l1 = int(fixed_l1)
        self.fixed_l2 = int(fixed_l2)
        self.cluster_capacity = int(cluster_capacity)
        self.subchannels = int(subchannels)
        self.gibbs_iterations = int(gibbs_iterations)
        self.gibbs_temperature = float(gibbs_temperature)
        self.payload_bytes_by_l1 = payloads.copy()
        self.rng = np.random.default_rng(seed)
        self.saa_diagnostics = None

    @staticmethod
    def _spectral_efficiency(channel_gain):
        return np.log2(1.0 + 10.0 * np.maximum(np.asarray(channel_gain, dtype=np.float64), 0.0))

    def _objective(self, state, bandwidth_hz, clusters, allocation):
        # Section 3.1.1: both UL and DL carry v(l1)+v(l2).
        payload_bits = 8.0 * (
            self.payload_bytes_by_l1[self.fixed_l1]
            + self.payload_bytes_by_l1[self.fixed_l2])
        rate_unit = max(float(bandwidth_hz), 1e-9) / self.subchannels
        efficiency = np.maximum(self._spectral_efficiency(state[:, 0]), 1e-9)
        uplink = payload_bits / np.maximum(allocation * rate_unit * efficiency, 1e-12)
        downlink = payload_bits / np.maximum(
            (float(self.subchannels) / len(state)) * rate_unit * efficiency, 1e-12)
        tx = uplink + downlink
        compute = (self.fixed_l1 + 1.0) * 1e-3 / np.maximum(state[:, 1], 1e-6)
        return max(float(np.max(tx[clusters == group] + compute[clusters == group]))
                   for group in np.unique(clusters))

    def _greedy_subchannels(self, state, bandwidth_hz, clusters):
        count = len(state)
        if self.subchannels < count:
            raise ValueError("subchannels must be at least the active-client count")
        allocation = np.ones(count, dtype=np.int64)
        current = self._objective(state, bandwidth_hz, clusters, allocation)
        for _ in range(self.subchannels - count):
            candidates = np.empty(count, dtype=np.float64)
            for client in range(count):
                trial = allocation.copy()
                trial[client] += 1
                candidates[client] = self._objective(state, bandwidth_hz, clusters, trial)
            chosen = int(np.argmin(candidates))
            allocation[chosen] += 1
            current = float(candidates[chosen])
        return allocation.astype(np.float64) / float(self.subchannels), current

    def _initial_clusters(self, state, cluster_count):
        hardness = 1.0 / np.maximum(self._spectral_efficiency(state[:, 0]), 1e-9)
        hardness += 0.1 / np.maximum(state[:, 1], 1e-6)
        order = np.argsort(-hardness, kind="stable")
        clusters = np.empty(len(state), dtype=np.int64)
        for rank, client in enumerate(order):
            clusters[client] = rank % cluster_count
        return clusters

    def calibrate_split(self, historical_states, historical_bandwidths):
        """Long-timescale sample-average split selection on excluded warm-up states."""
        states = [np.asarray(state, dtype=np.float64) for state in historical_states]
        bandwidths = np.asarray(historical_bandwidths, dtype=np.float64)
        if not states or len(states) != len(bandwidths):
            raise ValueError("CPSL SAA needs matching historical states and bandwidths")
        scores = {}
        original = self.fixed_l1
        # l1=0 uploads raw inputs and is not a valid split-learning action.
        for l1 in range(1, len(self.payload_bytes_by_l1) - 1):
            self.fixed_l1 = l1
            self.fixed_l2 = l1 + 1
            samples = []
            for state, bandwidth in zip(states, bandwidths):
                clusters = self._initial_clusters(state, min(self.fixed_clusters, len(state)))
                _, delay = self._greedy_subchannels(state, bandwidth, clusters)
                samples.append(delay)
            scores[l1] = float(np.mean(samples))
        self.fixed_l1 = min(scores, key=scores.get)
        self.fixed_l2 = self.fixed_l1 + 1
        self.saa_diagnostics = {
            "selected_l1": self.fixed_l1,
            "selected_l2": self.fixed_l2,
            "sample_count": len(states),
            "candidate_scores": scores,
            "previous_l1": original,
        }
        return dict(self.saa_diagnostics)

    def _gibbs_clusters(self, state, bandwidth_hz, cluster_count, deterministic):
        clusters = self._initial_clusters(state, cluster_count)
        _, objective = self._greedy_subchannels(state, bandwidth_hz, clusters)
        best, best_objective = clusters.copy(), objective
        if cluster_count == 1:
            return clusters
        for _ in range(self.gibbs_iterations):
            first, second = self.rng.choice(len(state), size=2, replace=False)
            if clusters[first] == clusters[second]:
                continue
            proposal = clusters.copy()
            proposal[first], proposal[second] = proposal[second], proposal[first]
            if max(np.bincount(proposal, minlength=cluster_count)) > self.cluster_capacity:
                continue
            _, proposal_objective = self._greedy_subchannels(state, bandwidth_hz, proposal)
            delta = proposal_objective - objective
            if deterministic:
                accept = delta < 0.0
            else:
                exponent = np.clip(delta / max(self.gibbs_temperature, 1e-12), -60.0, 60.0)
                accept = self.rng.random() < 1.0 / (1.0 + np.exp(exponent))
            if accept:
                clusters, objective = proposal, proposal_objective
                if objective < best_objective:
                    best, best_objective = clusters.copy(), objective
        return best

    def act(self, active_clients_state, available_migs, edge_state=None, deterministic=True, **_):
        state = np.asarray(active_clients_state, dtype=np.float64)
        count = len(state)
        if state.ndim != 2 or state.shape[1] < 2 or count == 0 or available_migs < 1:
            raise ValueError("CPSL requires a non-empty channel/compute state and one MIG")
        cluster_count = min(self.fixed_clusters, int(available_migs), count)
        if count > cluster_count * self.cluster_capacity:
            raise ValueError("active clients exceed the configured CPSL cluster capacity")
        bandwidth_hz = float(edge_state[1]) if edge_state is not None else 1.0
        clusters = self._gibbs_clusters(state, bandwidth_hz, cluster_count, deterministic)
        bandwidth, proxy_delay = self._greedy_subchannels(state, bandwidth_hz, clusters)
        return {
            "cluster": clusters,
            "virtual_cluster": clusters.copy(),
            "l1": np.full(count, self.fixed_l1, dtype=np.int64),
            "l2": np.full(count, self.fixed_l2, dtype=np.int64),
            "bw": bandwidth,
        }, {
            "paper_algorithm": "CPSL-SAA/Gibbs/greedy-subchannel",
            "cpsl_proxy_delay": float(proxy_delay),
            "cpsl_subchannels": self.subchannels,
            "cpsl_saa_sample_count": 0 if self.saa_diagnostics is None else self.saa_diagnostics["sample_count"],
        }

    def step(self, active_clients_state, available_migs):
        action, _ = self.act(active_clients_state, available_migs)
        return action["cluster"], action["l1"], action["l2"], action["bw"]
