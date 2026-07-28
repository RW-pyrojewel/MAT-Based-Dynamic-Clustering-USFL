"""Station-local liquid AI-RAN environment with research-plan scenario support."""
import numpy as np

from .scenario_specs import ScenarioSpec, get_scenario_spec


class LiquidAIRANEnv:
    """Generates one base station's state and resource tide for a named scenario.

    One environment instance represents one of the three stations. Coordinators
    should create three instances with identical scenario names and station IDs
    1, 2 and 3. The returned vehicle IDs are stable in fixed-participation
    scenarios and may vary only in scenarios C and D.
    """

    def __init__(
        self,
        data_provider,
        max_vehicles=None,
        base_bandwidth=100e6,
        max_migs=None,
        scenario: str | ScenarioSpec = "A",
        station_id=1,
        seed=None,
    ):
        self.data_provider = data_provider
        self.scenario = get_scenario_spec(scenario)
        self.station_id = int(station_id)
        self.rng = np.random.default_rng(seed)
        self.num_classes = data_provider.num_classes
        self.base_bandwidth = float(base_bandwidth)
        self.max_vehicles = int(max_vehicles or self.scenario.max_clients)
        self.max_migs = int(max_migs or self.scenario.max_migs)
        if self.max_vehicles < self.scenario.max_clients:
            raise ValueError("max_vehicles cannot be smaller than the scenario maximum")
        if self.max_migs < self.scenario.max_migs:
            raise ValueError("max_migs cannot be smaller than the scenario maximum")
        if self.base_bandwidth <= 0.0:
            raise ValueError("base_bandwidth must be positive")

        self.current_epoch = 0
        self.max_epochs = self.scenario.total_epochs
        self.current_N = 0
        self.current_migs = 2
        self.current_bandwidth = self.base_bandwidth
        self.active_vehicle_ids = np.empty(0, dtype=np.int64)
        self.vehicle_pool_computes = self.rng.uniform(1.0, 2.5, size=self.max_vehicles)
        self.vehicle_pool_labels = np.asarray(
            [data_provider.get_client_label_dist(vehicle_id) for vehicle_id in range(self.max_vehicles)],
            dtype=np.float64,
        )
        if self.vehicle_pool_labels.shape != (self.max_vehicles, self.num_classes):
            raise ValueError("data_provider label distributions must have shape (num_clients, num_classes)")
        self._fixed_vehicle_ids = self.rng.choice(
            self.max_vehicles, size=self.scenario.fixed_client_count or 1, replace=False
        )

    def reset(self):
        """Reset to the pre-episode observation at epoch zero."""
        self.current_epoch = 0
        return self._set_epoch_state()

    def step(self):
        """Advance one communication round and return state, MIGs, bandwidth, and IDs."""
        if self.current_epoch >= self.max_epochs:
            raise RuntimeError("scenario horizon exhausted; call reset() before stepping again")
        self.current_epoch += 1
        return self._set_epoch_state()

    def _set_epoch_state(self):
        resources = self.scenario.resources_for(self.station_id, self.current_epoch)
        self.current_migs = resources.available_migs
        self.current_bandwidth = self.base_bandwidth * resources.bandwidth_scale
        self.active_vehicle_ids = self._select_active_vehicle_ids()
        self.current_N = len(self.active_vehicle_ids)
        return self._generate_state_for_active_vehicles(), self.current_migs, self.current_bandwidth, self.active_vehicle_ids.copy()

    def _select_active_vehicle_ids(self):
        if not self.scenario.dynamic_participation:
            return self._fixed_vehicle_ids.copy()
        low, high = self.scenario.dynamic_client_range
        client_count = int(self.rng.integers(low, high + 1))
        active_ids = self.rng.choice(self.max_vehicles, size=client_count, replace=False)
        if self.scenario.name == "D" and self.current_epoch >= self.scenario.rare_label_departure_epoch:
            if self.station_id == 1:
                active_ids = active_ids[active_ids != 0]
                if len(active_ids) < client_count:
                    candidates = np.setdiff1d(np.arange(1, self.max_vehicles), active_ids, assume_unique=False)
                    active_ids = np.append(active_ids, self.rng.choice(candidates))
            elif self.station_id == 2 and 0 not in active_ids:
                active_ids[0] = 0
        return np.asarray(active_ids, dtype=np.int64)

    def _generate_state_for_active_vehicles(self):
        client_count = self.current_N
        channel_gains = self.rng.rayleigh(scale=1.0, size=(client_count, 1))
        computes = self.vehicle_pool_computes[self.active_vehicle_ids].reshape(client_count, 1)
        labels = self.vehicle_pool_labels[self.active_vehicle_ids]
        return np.concatenate([channel_gains, computes, labels], axis=1).astype(np.float32)

    def calc_wireless_transmission_delay(self, cluster_choices, bw_weights, smashed_data_sizes_bytes, channel_gains):
        """Compute each active cluster's synchronized uplink bottleneck delay."""
        clusters = np.asarray(cluster_choices)
        bandwidths = np.asarray(bw_weights, dtype=np.float64)
        sizes = np.asarray(smashed_data_sizes_bytes, dtype=np.float64)
        client_count = len(clusters)
        gains = np.asarray(channel_gains, dtype=np.float64)
        if clusters.shape != (client_count,) or bandwidths.shape != (client_count,) or sizes.shape != (client_count,):
            raise ValueError("cluster choices, bandwidth weights and data sizes must have shape (N,)")
        if gains.shape != (client_count,) or not np.isfinite(gains).all() or (gains < 0.0).any():
            raise ValueError("channel_gains must be a finite non-negative array of shape (N,)")
        if (clusters < 0).any() or (clusters >= self.current_migs).any():
            raise ValueError("cluster choices must reference currently available MIGs")
        if (bandwidths < 0.0).any() or (sizes < 0.0).any():
            raise ValueError("bandwidth weights and data sizes must be non-negative")

        cluster_delays = np.zeros(self.max_migs, dtype=np.float64)
        for mig_id in range(self.current_migs):
            members = clusters == mig_id
            if not members.any():
                continue
            weights = bandwidths[members]
            weight_sum = weights.sum()
            allocation = np.full(len(weights), 1.0 / len(weights)) if weight_sum <= 1e-12 else weights / weight_sum
            rates_bps = allocation * (self.current_bandwidth / self.current_migs) * np.log2(1.0 + 10.0 * gains[members])
            cluster_delays[mig_id] = np.max((sizes[members] * 8.0) / np.maximum(rates_bps, 1e-9))
        return cluster_delays
