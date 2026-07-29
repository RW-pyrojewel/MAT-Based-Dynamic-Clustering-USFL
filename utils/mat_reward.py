"""Dimensionless reward calculation for MAT resource allocation."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class MATRewardConfig:
    """Coefficients and physical references for the scheme-defined reward."""

    delay_weight: float = 0.5
    delay_reference_seconds: float = 1.0
    bandwidth_penalty_weight: float = 1.0
    cluster_size_penalty_weight: float = 0.0
    cluster_size_limit: int | None = None
    epsilon: float = 1e-12

    def __post_init__(self):
        if not 0.0 <= self.delay_weight <= 1.0:
            raise ValueError("delay_weight must be in [0, 1]")
        if self.delay_reference_seconds <= 0.0:
            raise ValueError("delay_reference_seconds must be positive")
        if self.cluster_size_limit is not None and self.cluster_size_limit < 1:
            raise ValueError("cluster_size_limit must be positive when enabled")
        if min(self.bandwidth_penalty_weight, self.cluster_size_penalty_weight, self.epsilon) < 0.0:
            raise ValueError("penalty weights and epsilon must be non-negative")


def compute_mat_reward(system_max_delay, label_distributions, cluster_choices, bandwidths, config, ideal_distribution=None):
    """Compute the normalized delay/Non-IID objective and physical penalties."""
    labels = np.asarray(label_distributions, dtype=np.float64)
    clusters = np.asarray(cluster_choices)
    bandwidths = np.asarray(bandwidths, dtype=np.float64)
    if labels.ndim != 2 or labels.shape[0] == 0:
        raise ValueError("label_distributions must have shape (N, num_classes) with N > 0")
    if clusters.shape != (labels.shape[0],) or bandwidths.shape != (labels.shape[0],):
        raise ValueError("cluster_choices and bandwidths must have shape (N,)")
    if not np.isfinite(system_max_delay) or system_max_delay < 0.0:
        raise ValueError("system_max_delay must be finite and non-negative")
    if not np.isfinite(labels).all() or (labels < 0.0).any() or not np.isfinite(bandwidths).all():
        raise ValueError("labels and bandwidths must be finite; labels must be non-negative")

    class_count = labels.shape[1]
    if ideal_distribution is None:
        ideal = np.full(class_count, 1.0 / class_count)
    else:
        ideal = np.asarray(ideal_distribution, dtype=np.float64)
        if ideal.shape != (class_count,) or not np.isfinite(ideal).all() or (ideal < 0.0).any() or ideal.sum() <= 0.0:
            raise ValueError("ideal_distribution must be a non-negative vector matching num_classes")
        ideal = ideal / ideal.sum()
    ideal = np.clip(ideal, config.epsilon, None)
    ideal = ideal / ideal.sum()

    cluster_ids = np.unique(clusters)
    cluster_sizes = np.asarray([np.sum(clusters == cluster_id) for cluster_id in cluster_ids], dtype=np.float64)
    kl_sum = 0.0
    for cluster_id in cluster_ids:
        members = labels[clusters == cluster_id]
        cluster_distribution = members.mean(axis=0)
        if cluster_distribution.sum() <= 0.0:
            raise ValueError("each non-empty cluster must have a positive label-distribution mass")
        cluster_distribution = cluster_distribution / cluster_distribution.sum()
        positive = cluster_distribution > 0.0
        kl_sum += float(np.sum(cluster_distribution[positive] * np.log(cluster_distribution[positive] / ideal[positive])))

    delay_normalized = float(system_max_delay) / config.delay_reference_seconds
    kl_reference = max(len(cluster_ids) * math.log(class_count), config.epsilon)
    kl_normalized = kl_sum / kl_reference
    bandwidth_violation = max(float(bandwidths.sum()) - 1.0, 0.0)
    if config.cluster_size_limit is None:
        cluster_size_violation = 0.0
    else:
        excess = np.maximum(cluster_sizes - config.cluster_size_limit, 0.0).sum()
        cluster_size_violation = float(excess / labels.shape[0])
    penalty = (
        config.bandwidth_penalty_weight * bandwidth_violation
        + config.cluster_size_penalty_weight * cluster_size_violation
    )
    objective = config.delay_weight * delay_normalized + (1.0 - config.delay_weight) * kl_normalized
    mean_cluster_size = float(cluster_sizes.mean())
    cluster_size_std = float(cluster_sizes.std())
    cluster_load_cv = cluster_size_std / max(mean_cluster_size, config.epsilon)
    return -(objective + penalty), {
        "delay": float(system_max_delay),
        "delay_normalized": delay_normalized,
        "kl_sum": kl_sum,
        "kl_normalized": kl_normalized,
        "nonempty_cluster_count": int(len(cluster_ids)),
        "max_cluster_size": int(cluster_sizes.max()),
        "cluster_size_std": cluster_size_std,
        "cluster_load_cv": cluster_load_cv,
        "bandwidth_violation": bandwidth_violation,
        "cluster_size_violation": cluster_size_violation,
        "penalty": penalty,
    }
