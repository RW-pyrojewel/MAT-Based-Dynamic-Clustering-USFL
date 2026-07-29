"""Adapted CPSL baseline with a static virtual-cluster topology."""
import numpy as np

from interfaces.base_agent import BaseAgent


class CPSLAgent(BaseAgent):
    """Keep a fixed cluster topology and a global split pair throughout a run."""

    def __init__(self, fixed_clusters=2, fixed_l1=3, fixed_l2=4):
        super().__init__(agent_name="Adapted-CPSL")
        if fixed_clusters < 1:
            raise ValueError("fixed_clusters must be positive")
        if not 0 <= fixed_l1 < fixed_l2:
            raise ValueError("fixed split pair must satisfy 0 <= l1 < l2")
        self.fixed_clusters = int(fixed_clusters)
        self.fixed_l1 = int(fixed_l1)
        self.fixed_l2 = int(fixed_l2)

    def act(self, active_clients_state, available_migs, edge_state=None, deterministic=True):
        client_count = len(active_clients_state)
        if client_count == 0 or available_migs < 1:
            raise ValueError("CPSL requires at least one client and one available MIG")
        virtual_cluster = np.arange(client_count, dtype=np.int64) % self.fixed_clusters
        physical_cluster = virtual_cluster % int(available_migs)
        return {
            "cluster": physical_cluster,
            "virtual_cluster": virtual_cluster,
            "l1": np.full(client_count, self.fixed_l1, dtype=np.int64),
            "l2": np.full(client_count, self.fixed_l2, dtype=np.int64),
            "bw": np.full(client_count, 1.0 / client_count, dtype=np.float64),
        }, None

    def step(self, active_clients_state, available_migs):
        action, _ = self.act(active_clients_state, available_migs)
        return action["cluster"], action["l1"], action["l2"], action["bw"]
