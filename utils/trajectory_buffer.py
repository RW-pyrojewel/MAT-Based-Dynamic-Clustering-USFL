"""In-memory transition buffer for one-version on-policy MAT PPO updates."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MATTransition:
    state: np.ndarray
    edge_state: np.ndarray
    action: dict
    reward: float
    next_state: np.ndarray
    next_edge_state: np.ndarray
    done: bool
    policy_info: dict
    available_migs: int
    station_id: int
    epoch: int
    trajectory_id: tuple
    policy_version: int


class MATTrajectoryBuffer:
    """Store complete MAT transitions without crossing policy/trajectory boundaries."""

    _ACTION_KEYS = {"cluster", "l1", "l2", "bw"}
    _POLICY_KEYS = {
        "decision_order", "cluster_log_probs", "bandwidth_log_probs", "bandwidth_mask",
        "split_log_probs", "split_mask", "value", "device_entropy", "split_entropy",
    }

    def __init__(self):
        self._transitions = []
        self._policy_version = None

    def __len__(self):
        return len(self._transitions)

    @property
    def policy_version(self):
        return self._policy_version

    def clear(self):
        self._transitions.clear()
        self._policy_version = None

    @staticmethod
    def _copy_policy_info(policy_info, client_count, available_migs):
        if set(policy_info) != MATTrajectoryBuffer._POLICY_KEYS:
            raise ValueError("policy_info does not match the MAT rollout contract")
        copied = {key: np.asarray(value).copy() if key != "value" else float(value) for key, value in policy_info.items()}
        if copied["decision_order"].shape != (client_count,):
            raise ValueError("decision_order must have shape (N,)")
        if sorted(copied["decision_order"].tolist()) != list(range(client_count)):
            raise ValueError("decision_order must be a permutation of active clients")
        for key in ("cluster_log_probs", "bandwidth_log_probs", "bandwidth_mask", "device_entropy"):
            if copied[key].shape != (client_count,):
                raise ValueError(f"{key} must have shape (N,)")
        for key in ("split_log_probs", "split_mask", "split_entropy"):
            if copied[key].ndim != 1 or len(copied[key]) < available_migs:
                raise ValueError(f"{key} must cover every available MIG")
        if not np.isfinite(copied["value"]):
            raise ValueError("rollout value must be finite")
        return copied

    def append(self, state, edge_state, action, reward, next_state, next_edge_state, done,
               policy_info, available_migs, station_id, epoch, trajectory_id=None, policy_version=0):
        state = np.asarray(state, dtype=np.float32).copy()
        next_state = np.asarray(next_state, dtype=np.float32).copy()
        edge_state = np.asarray(edge_state, dtype=np.float32).copy()
        next_edge_state = np.asarray(next_edge_state, dtype=np.float32).copy()
        if state.ndim != 2 or next_state.ndim != 2 or edge_state.shape != (2,) or next_edge_state.shape != (2,):
            raise ValueError("states must be 2D and edge states must have shape (2,)")
        if int(available_migs) < 1:
            raise ValueError("available_migs must be positive")
        if int(station_id) < 1 or int(epoch) < 1:
            raise ValueError("station_id and epoch must be positive")
        if set(action) != self._ACTION_KEYS:
            raise ValueError("action must contain cluster, l1, l2 and bw")
        if any(np.asarray(action[key]).shape != (state.shape[0],) for key in self._ACTION_KEYS):
            raise ValueError("every action component must have shape (N,)")
        version = int(policy_version)
        if self._policy_version is not None and version != self._policy_version:
            raise ValueError("cannot mix different policy versions in one on-policy buffer")
        self._policy_version = version
        trajectory = (0, int(station_id)) if trajectory_id is None else tuple(trajectory_id)
        if len(trajectory) != 2:
            raise ValueError("trajectory_id must be (episode_id, station_id)")
        self._transitions.append(MATTransition(
            state, edge_state, {key: np.asarray(value).copy() for key, value in action.items()}, float(reward),
            next_state, next_edge_state, bool(done), self._copy_policy_info(policy_info, state.shape[0], int(available_migs)),
            int(available_migs), int(station_id), int(epoch), trajectory, version,
        ))

    def extend(self, other):
        if not isinstance(other, MATTrajectoryBuffer):
            raise TypeError("other must be MATTrajectoryBuffer")
        if not len(other):
            return
        if self._policy_version is not None and self._policy_version != other.policy_version:
            raise ValueError("cannot merge buffers from different policy versions")
        self._policy_version = other.policy_version
        self._transitions.extend(other._transitions)

    def as_ppo_kwargs(self):
        if not self._transitions:
            raise ValueError("cannot export an empty trajectory buffer")
        transitions = self._transitions
        return {
            "rewards": np.asarray([item.reward for item in transitions], dtype=np.float32),
            "next_states": [item.next_state for item in transitions],
            "dones": np.asarray([item.done for item in transitions], dtype=bool),
            "states": [item.state for item in transitions],
            "actions": [item.action for item in transitions],
            "edge_states": [item.edge_state for item in transitions],
            "next_edge_states": [item.next_edge_state for item in transitions],
            "available_migs": [item.available_migs for item in transitions],
            "policy_infos": [item.policy_info for item in transitions],
            "station_ids": [item.station_id for item in transitions],
            "epochs": [item.epoch for item in transitions],
            "trajectory_ids": [item.trajectory_id for item in transitions],
            "policy_versions": [item.policy_version for item in transitions],
        }