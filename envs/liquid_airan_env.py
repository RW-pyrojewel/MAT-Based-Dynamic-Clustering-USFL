"""Station-local liquid AI-RAN environment with research-plan scenario support."""
import numpy as np

from .scenario_specs import ScenarioSpec, get_scenario_spec


class LiquidAIRANEnv:
    """Generate one base station's state and resource tide for a named scenario."""

    def __init__(
        self,
        data_provider,
        max_vehicles=None,
        base_bandwidth=100e6,
        max_migs=None,
        scenario: str | ScenarioSpec = "A",
        station_id=1,
        seed=None,
        vehicle_id_offset=0,
    ):
        self.data_provider = data_provider
        self.scenario = get_scenario_spec(scenario)
        self.station_id = int(station_id)
        self.rng = np.random.default_rng(seed)
        self.num_classes = data_provider.num_classes
        self.base_bandwidth = float(base_bandwidth)
        self.max_vehicles = int(max_vehicles or self.scenario.max_clients)
        self.max_migs = int(max_migs or self.scenario.max_migs)
        self.vehicle_id_offset = int(vehicle_id_offset)
        if self.max_vehicles < self.scenario.max_clients:
            raise ValueError("max_vehicles cannot be smaller than the scenario maximum")
        if self.max_migs < self.scenario.max_migs:
            raise ValueError("max_migs cannot be smaller than the scenario maximum")
        if self.base_bandwidth <= 0.0:
            raise ValueError("base_bandwidth must be positive")
        if self.vehicle_id_offset < 0 or self.vehicle_id_offset + self.max_vehicles > data_provider.num_clients:
            raise ValueError("vehicle ID range must be available from data_provider")

        self.current_epoch = 0
        self.max_epochs = self.scenario.total_epochs
        self.current_N = 0
        self.current_migs = 2
        self.current_bandwidth = self.base_bandwidth
        self.vehicle_pool_ids = np.arange(
            self.vehicle_id_offset, self.vehicle_id_offset + self.max_vehicles, dtype=np.int64
        )
        self.active_vehicle_slots = np.empty(0, dtype=np.int64)
        self.active_vehicle_ids = np.empty(0, dtype=np.int64)
        self.vehicle_pool_computes = self.rng.uniform(1.0, 2.5, size=self.max_vehicles)
        self.vehicle_pool_labels = np.asarray(
            [data_provider.get_client_label_dist(vehicle_id) for vehicle_id in self.vehicle_pool_ids],
            dtype=np.float64,
        )
        if self.vehicle_pool_labels.shape != (self.max_vehicles, self.num_classes):
            raise ValueError("data_provider label distributions must have shape (num_clients, num_classes)")
        self._fixed_vehicle_slots = self.rng.choice(
            self.max_vehicles, size=self.scenario.fixed_client_count or 1, replace=False
        )

    def reset(self):
        """Reset to the pre-episode observation at epoch zero."""
        self.current_epoch = 0
        return self._set_epoch_state()

    def step(self):
        """Advance one communication round and return state, MIGs, bandwidth, and global IDs."""
        if self.current_epoch >= self.max_epochs:
            raise RuntimeError("scenario horizon exhausted; call reset() before stepping again")
        self.current_epoch += 1
        return self._set_epoch_state()

    def _set_epoch_state(self):
        resources = self.scenario.resources_for(self.station_id, self.current_epoch)
        self.current_migs = resources.available_migs
        self.current_bandwidth = self.base_bandwidth * resources.bandwidth_scale
        self.active_vehicle_slots = self._select_active_vehicle_slots()
        self.active_vehicle_ids = self.vehicle_pool_ids[self.active_vehicle_slots]
        self.current_N = len(self.active_vehicle_ids)
        return self._generate_state_for_active_vehicles(), self.current_migs, self.current_bandwidth, self.active_vehicle_ids.copy()

    def _select_active_vehicle_slots(self):
        if not self.scenario.dynamic_participation:
            return self._fixed_vehicle_slots.copy()
        low, high = self.scenario.dynamic_client_range
        client_count = int(self.rng.integers(low, high + 1))
        active_slots = self.rng.choice(self.max_vehicles, size=client_count, replace=False)
        if self.scenario.name == "D" and self.current_epoch >= self.scenario.rare_label_departure_epoch:
            if self.station_id == 1:
                active_slots = active_slots[active_slots != 0]
                if len(active_slots) < client_count:
                    candidates = np.setdiff1d(np.arange(1, self.max_vehicles), active_slots, assume_unique=False)
                    active_slots = np.append(active_slots, self.rng.choice(candidates))
            elif self.station_id == 2 and 0 not in active_slots:
                active_slots[0] = 0
        return np.asarray(active_slots, dtype=np.int64)

    def _generate_state_for_active_vehicles(self):
        client_count = self.current_N
        channel_gains = self.rng.rayleigh(scale=1.0, size=(client_count, 1))
        computes = self.vehicle_pool_computes[self.active_vehicle_slots].reshape(client_count, 1)
        labels = self.vehicle_pool_labels[self.active_vehicle_slots]
        return np.concatenate([channel_gains, computes, labels], axis=1).astype(np.float32)

    def calc_wireless_transmission_delay(
        self,
        cluster_choices,
        bw_weights,
        smashed_data_sizes_bytes,
        channel_gains,
        available_migs=None,
        bandwidth_hz=None,
    ):
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
        current_migs = self.current_migs if available_migs is None else int(available_migs)
        current_bandwidth = self.current_bandwidth if bandwidth_hz is None else float(bandwidth_hz)
        if current_migs < 1 or current_migs > self.max_migs:
            raise ValueError("available_migs must be in [1, max_migs]")
        if current_bandwidth <= 0.0:
            raise ValueError("bandwidth_hz must be positive")
        if (clusters < 0).any() or (clusters >= current_migs).any():
            raise ValueError("cluster choices must reference currently available MIGs")
        if (bandwidths < 0.0).any() or (sizes < 0.0).any():
            raise ValueError("bandwidth weights and data sizes must be non-negative")
        if bandwidths.sum() > 1.0 + 1e-6:
            raise ValueError("bandwidth allocations must not exceed the global budget")

        cluster_delays = np.zeros(self.max_migs, dtype=np.float64)
        for mig_id in range(current_migs):
            members = clusters == mig_id
            if not members.any():
                continue
            rates_bps = bandwidths[members] * current_bandwidth * np.log2(1.0 + 10.0 * gains[members])
            cluster_delays[mig_id] = np.max((sizes[members] * 8.0) / np.maximum(rates_bps, 1e-9))
        return cluster_delays
