"""Causal GPU runtime profile used by the physics-guided MAT policy."""
from __future__ import annotations

import numpy as np


class CausalRuntimeProfile:
    """EMA timings indexed by (cluster size, legal U-split pair)."""

    def __init__(self, max_clients=10, num_cut_layers=7, alpha=0.2):
        if max_clients < 1 or num_cut_layers < 3 or not 0.0 < alpha <= 1.0:
            raise ValueError("invalid runtime-profile configuration")
        self.max_clients = int(max_clients)
        self.num_cut_layers = int(num_cut_layers)
        self.alpha = float(alpha)
        self.pairs = tuple(
            (l1, l2) for l1 in range(1, self.num_cut_layers - 1)
            for l2 in range(l1 + 1, self.num_cut_layers)
        )
        shape = (self.max_clients + 1, len(self.pairs))
        self.ema_seconds = np.full(shape, np.nan, dtype=np.float64)
        self.counts = np.zeros(shape, dtype=np.int64)
        self.ema_abs_error = np.zeros(shape, dtype=np.float64)
        self.last_epoch = np.zeros(shape, dtype=np.int64)
        self._pair_index = {pair: index for index, pair in enumerate(self.pairs)}

    def predict(self, cluster_size, l1, l2):
        size = int(np.clip(cluster_size, 1, self.max_clients))
        pair = (int(l1), int(l2))
        if pair not in self._pair_index:
            raise ValueError("runtime profile received an illegal split pair")
        pair_index = self._pair_index[pair]
        if self.counts[size, pair_index] > 0:
            return float(self.ema_seconds[size, pair_index])
        observed = np.argwhere(self.counts > 0)
        if not len(observed):
            return 0.0
        distance = (
            np.abs(observed[:, 0] - size)
            + np.abs(np.asarray([self.pairs[index][0] for index in observed[:, 1]]) - pair[0])
            + np.abs(np.asarray([self.pairs[index][1] for index in observed[:, 1]]) - pair[1])
        )
        nearest_size, nearest_pair = observed[int(np.argmin(distance))]
        # Scale the nearest observation by cluster size; this is a bounded,
        # causal interpolation rather than a synthetic compute-capacity claim.
        return float(self.ema_seconds[nearest_size, nearest_pair] * size / max(int(nearest_size), 1))

    def update(self, action, cluster_compute_delays, epoch):
        clusters = np.asarray(action["cluster"], dtype=np.int64)
        l1_values = np.asarray(action["l1"], dtype=np.int64)
        l2_values = np.asarray(action["l2"], dtype=np.int64)
        for cluster in np.unique(clusters):
            members = np.flatnonzero(clusters == cluster)
            l1, l2 = int(l1_values[members[0]]), int(l2_values[members[0]])
            if not np.all(l1_values[members] == l1) or not np.all(l2_values[members] == l2):
                raise ValueError("runtime-profile clusters require one shared split pair")
            observed = float(cluster_compute_delays.get(int(cluster), 0.0))
            if not np.isfinite(observed) or observed < 0.0:
                raise ValueError("runtime observations must be finite and non-negative")
            size, pair_index = min(len(members), self.max_clients), self._pair_index[(l1, l2)]
            previous = self.ema_seconds[size, pair_index]
            if self.counts[size, pair_index] == 0:
                updated, error = observed, 0.0
            else:
                error = abs(observed - previous)
                updated = (1.0 - self.alpha) * previous + self.alpha * observed
            self.ema_seconds[size, pair_index] = updated
            self.ema_abs_error[size, pair_index] = (
                error if self.counts[size, pair_index] == 0 else
                (1.0 - self.alpha) * self.ema_abs_error[size, pair_index] + self.alpha * error
            )
            self.counts[size, pair_index] += 1
            self.last_epoch[size, pair_index] = int(epoch)

    def diagnostics(self):
        observed = self.counts > 0
        return {
            "runtime_profile_coverage": float(observed.mean()),
            "runtime_profile_observations": int(self.counts.sum()),
            "runtime_profile_mean_seconds": float(np.nanmean(self.ema_seconds)) if observed.any() else 0.0,
            "runtime_profile_mean_abs_error": float(self.ema_abs_error[observed].mean()) if observed.any() else 0.0,
        }

    def state_dict(self):
        return {
            "max_clients": self.max_clients,
            "num_cut_layers": self.num_cut_layers,
            "alpha": self.alpha,
            "pairs": self.pairs,
            "ema_seconds": self.ema_seconds.copy(),
            "counts": self.counts.copy(),
            "ema_abs_error": self.ema_abs_error.copy(),
            "last_epoch": self.last_epoch.copy(),
        }

    def load_state_dict(self, state):
        if (int(state["max_clients"]) != self.max_clients
                or int(state["num_cut_layers"]) != self.num_cut_layers
                or tuple(map(tuple, state["pairs"])) != self.pairs):
            raise ValueError("runtime-profile checkpoint configuration mismatch")
        for name in ("ema_seconds", "counts", "ema_abs_error", "last_epoch"):
            value = np.asarray(state[name])
            if value.shape != getattr(self, name).shape:
                raise ValueError("runtime-profile checkpoint shape mismatch")
            getattr(self, name)[:] = value
