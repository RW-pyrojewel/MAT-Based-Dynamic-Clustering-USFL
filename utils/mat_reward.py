"""Scheme-aligned reward calculation for MAT resource allocation."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MATRewardConfig:
    """Explicit coefficients for the reward in research-plan section 4.1.3."""

    delay_weight: float = 0.5
    bandwidth_penalty_weight: float = 1.0
    cluster_size_penalty_weight: float = 1.0
    cluster_size_limit: int = 8
    epsilon: float = 1e-12

    def __post_init__(self):
        if not 0.0 <= self.delay_weight <= 1.0:
            raise ValueError("delay_weight must be in [0, 1]")
        if self.cluster_size_limit < 1:
            raise ValueError("cluster_size_limit must be positive")
        if min(self.bandwidth_penalty_weight, self.cluster_size_penalty_weight, self.epsilon) < 0.0:
            raise ValueError("penalty weights and epsilon must be non-negative")


def compute_mat_reward(system_max_delay, label_distributions, cluster_choices, bandwidths, config, ideal_distribution=None):
    """Compute ``-[lambda*T + (1-lambda)*Gamma] - Upsilon``.

    ``Gamma`` is the sum of KL(cluster label distribution || ideal label
    distribution) over non-empty clusters. ``Upsilon`` applies the plan's
    ReLU penalties to the global bandwidth request and cluster sizes.
    """
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

    kl_sum = 0.0
    for cluster_id in np.unique(clusters):
        members = labels[clusters == cluster_id]
        cluster_distribution = members.mean(axis=0)
        if cluster_distribution.sum() <= 0.0:
            raise ValueError("each non-empty cluster must have a positive label-distribution mass")
        cluster_distribution = cluster_distribution / cluster_distribution.sum()
        positive = cluster_distribution > 0.0
        kl_sum += float(np.sum(cluster_distribution[positive] * np.log(cluster_distribution[positive] / ideal[positive])))

    bandwidth_violation = max(float(bandwidths.sum()) - 1.0, 0.0)
    cluster_size_violation = sum(max(int(np.sum(clusters == cluster_id)) - config.cluster_size_limit, 0) for cluster_id in np.unique(clusters))
    penalty = (config.bandwidth_penalty_weight * bandwidth_violation + config.cluster_size_penalty_weight * cluster_size_violation)
    objective = config.delay_weight * float(system_max_delay) + (1.0 - config.delay_weight) * kl_sum
    return -(objective + penalty), {"delay": float(system_max_delay), "kl_sum": kl_sum, "bandwidth_violation": bandwidth_violation, "cluster_size_violation": float(cluster_size_violation), "penalty": penalty}
