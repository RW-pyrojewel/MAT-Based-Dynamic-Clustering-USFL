"""Experiment scenarios defined by the research plan's sections 5.3 and 5.4."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ResourceState:
    """Resources exposed by one base station in a communication round."""

    available_migs: int
    bandwidth_scale: float = 1.0


@dataclass(frozen=True)
class ScenarioSpec:
    """A reproducible station-local view of one plan experiment scenario."""

    name: str
    total_epochs: int
    dynamic_participation: bool
    fixed_client_count: Optional[int] = None
    dynamic_client_range: Optional[tuple[int, int]] = None
    rare_label_departure_epoch: Optional[int] = None

    @property
    def max_clients(self) -> int:
        if self.dynamic_participation:
            assert self.dynamic_client_range is not None
            return self.dynamic_client_range[1]
        assert self.fixed_client_count is not None
        return self.fixed_client_count

    @property
    def max_migs(self) -> int:
        return 7

    def resources_for(self, station_id: int, epoch: int) -> ResourceState:
        """Return the prescribed per-station resource condition for one epoch."""
        if station_id not in (1, 2, 3):
            raise ValueError("station_id must be 1, 2, or 3")
        if epoch < 0 or epoch > self.total_epochs:
            raise ValueError("epoch is outside the scenario horizon")

        if self.name == "A" and epoch >= 100:
            if station_id == 1:
                return ResourceState(available_migs=7)
            if station_id == 2:
                return ResourceState(available_migs=2, bandwidth_scale=0.2)
        return ResourceState(available_migs=2)


SCENARIOS = {
    "A": ScenarioSpec("A", total_epochs=150, dynamic_participation=False, fixed_client_count=10),
    "B": ScenarioSpec("B", total_epochs=150, dynamic_participation=False, fixed_client_count=10),
    "C": ScenarioSpec("C", total_epochs=150, dynamic_participation=True, dynamic_client_range=(5, 30)),
    "D": ScenarioSpec(
        "D",
        total_epochs=150,
        dynamic_participation=True,
        dynamic_client_range=(5, 30),
        rare_label_departure_epoch=100,
    ),
}


def get_scenario_spec(scenario: str | ScenarioSpec) -> ScenarioSpec:
    """Resolve a scenario name while allowing callers to pass custom specs."""
    if isinstance(scenario, ScenarioSpec):
        return scenario
    try:
        return SCENARIOS[scenario.upper()]
    except (AttributeError, KeyError) as exc:
        raise ValueError("scenario must be one of A, B, C, D or a ScenarioSpec") from exc
