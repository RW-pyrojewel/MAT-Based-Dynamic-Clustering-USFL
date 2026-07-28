"""In-memory transition buffer for on-policy MAT PPO updates."""
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
    old_log_prob: float
    available_migs: int


class MATTrajectoryBuffer:
    """Stores complete, immutable-equivalent on-policy transitions."""
    def __init__(self):
        self._transitions = []

    def __len__(self):
        return len(self._transitions)

    def clear(self):
        self._transitions.clear()

    def append(self, state, edge_state, action, reward, next_state, next_edge_state, done, old_log_prob, available_migs):
        state = np.asarray(state, dtype=np.float32).copy()
        next_state = np.asarray(next_state, dtype=np.float32).copy()
        edge_state = np.asarray(edge_state, dtype=np.float32).copy()
        next_edge_state = np.asarray(next_edge_state, dtype=np.float32).copy()
        if state.ndim != 2 or next_state.ndim != 2 or edge_state.shape != (2,) or next_edge_state.shape != (2,):
            raise ValueError("states must be 2D and edge states must have shape (2,)")
        if not 1 <= int(available_migs):
            raise ValueError("available_migs must be positive")
        required = {"cluster", "l1", "l2", "bw"}
        if set(action) != required or any(np.asarray(action[key]).shape != (state.shape[0],) for key in required):
            raise ValueError("action must contain cluster, l1, l2 and bw arrays of shape (N,)")
        stored_action = {key: np.asarray(value).copy() for key, value in action.items()}
        self._transitions.append(MATTransition(state, edge_state, stored_action, float(reward), next_state, next_edge_state, bool(done), float(old_log_prob), int(available_migs)))

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
            "old_log_probs": np.asarray([item.old_log_prob for item in transitions], dtype=np.float32),
            "edge_states": [item.edge_state for item in transitions],
            "next_edge_states": [item.next_edge_state for item in transitions],
            "available_migs": [item.available_migs for item in transitions],
        }
