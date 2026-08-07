"""Reusable exogenous state traces for fair Scenario-A algorithm comparisons."""
from dataclasses import dataclass
import hashlib

import numpy as np

from envs import LiquidAIRANEnv


@dataclass(frozen=True)
class ScenarioATrace:
    """Immutable per-round station observations shared by every algorithm."""

    observations: tuple
    trace_id: str

    @property
    def total_epochs(self):
        return len(self.observations)

    def get(self, epoch, station_id):
        state, available_migs, bandwidth, client_ids = self.observations[epoch - 1][station_id]
        return state.copy(), int(available_migs), float(bandwidth), client_ids.copy()


def build_scenario_a_trace(provider, seed, total_epochs):
    """Pre-generate a reproducible Scenario-A trajectory without policy feedback."""
    if not 1 <= total_epochs <= 150:
        raise ValueError("total_epochs must be in [1, 150]")
    environments = {
        station_id: LiquidAIRANEnv(
            provider,
            max_vehicles=10,
            max_migs=7,
            scenario="A",
            station_id=station_id,
            vehicle_id_offset=(station_id - 1) * 10,
            seed=seed + station_id,
            # Scenario A uses measured accelerator wall-clock for compute delay.
            # Keep the legacy state column as a constant schema placeholder rather
            # than inventing unobserved heterogeneous client capacities.
            compute_profile="uniform",
        )
        for station_id in (1, 2, 3)
    }
    digest = hashlib.sha256()
    observations = []
    for _ in range(total_epochs):
        by_station = {}
        for station_id, environment in environments.items():
            state, available_migs, bandwidth, client_ids = environment.step()
            state.setflags(write=False)
            client_ids.setflags(write=False)
            by_station[station_id] = (state, available_migs, bandwidth, client_ids)
            digest.update(state.tobytes())
            digest.update(client_ids.tobytes())
            digest.update(f"{available_migs}:{bandwidth:.6f}".encode("ascii"))
        observations.append(by_station)
    return ScenarioATrace(tuple(observations), digest.hexdigest()[:16])
