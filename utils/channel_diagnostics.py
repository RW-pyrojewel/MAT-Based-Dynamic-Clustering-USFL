"""Physical channel/bandwidth diagnostics that never alter environment actions."""
import numpy as np


def spectral_efficiency(channel_gains, snr_scale=10.0):
    gains = np.asarray(channel_gains, dtype=np.float64)
    if not np.isfinite(gains).all() or (gains < 0.0).any():
        raise ValueError("channel gains must be finite and non-negative")
    return np.log2(1.0 + float(snr_scale) * gains)


def normalized_channel_quality(channel_gains, snr_scale=10.0):
    reference = np.log2(1.0 + float(snr_scale))
    if reference <= 0.0:
        raise ValueError("snr_scale must be positive")
    return spectral_efficiency(channel_gains, snr_scale) / reference


def required_airtime(payload_bytes, channel_gains, snr_scale=10.0, epsilon=1e-12):
    payload = np.asarray(payload_bytes, dtype=np.float64)
    gains = np.asarray(channel_gains, dtype=np.float64)
    if payload.shape != gains.shape or (payload < 0.0).any() or not np.isfinite(payload).all():
        raise ValueError("payload and channel gains must be matching finite vectors")
    return payload * 8.0 / np.maximum(spectral_efficiency(gains, snr_scale), epsilon)


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        return 0.0
    left_rank, right_rank = _rankdata(left), _rankdata(right)
    if left_rank.std() <= 1e-12 or right_rank.std() <= 1e-12:
        return 0.0
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return value if np.isfinite(value) else 0.0


def oracle_bandwidth(required_airtimes, min_share=0.01):
    costs = np.asarray(required_airtimes, dtype=np.float64)
    if costs.ndim != 1 or len(costs) == 0 or not np.isfinite(costs).all() or (costs < 0.0).any():
        raise ValueError("required airtimes must be a finite non-negative vector")
    if min_share < 0.0 or len(costs) * min_share >= 1.0:
        raise ValueError("minimum share is infeasible")
    if costs.sum() <= 1e-15:
        return np.full(len(costs), 1.0 / len(costs), dtype=np.float64)
    low = np.finfo(np.float64).tiny
    high = max(float(costs.max() / max(min_share, 1e-12)), 1.0)
    for _ in range(160):
        threshold = 0.5 * (low + high)
        allocation = np.maximum(min_share, costs / threshold)
        if allocation.sum() > 1.0:
            low = threshold
        else:
            high = threshold
    allocation = np.maximum(min_share, costs / high)
    residual = 1.0 - allocation.sum()
    if residual > 0.0:
        allocation += residual * costs / costs.sum()
    return allocation / allocation.sum()


def offset_aware_oracle_bandwidth(required_airtimes, cluster_choices, cluster_compute_delays,
                                  bandwidth_hz, min_share=0.01, iterations=96,
                                  client_offsets=None):
    """Minimize max_i(compute[cluster_i]+offset_i+airtime_i/(B*b_i))."""
    costs = np.asarray(required_airtimes, dtype=np.float64)
    clusters = np.asarray(cluster_choices, dtype=np.int64)
    if costs.ndim != 1 or clusters.shape != costs.shape or len(costs) == 0:
        raise ValueError("airtimes and cluster choices must be matching non-empty vectors")
    if not np.isfinite(costs).all() or (costs < 0.0).any() or bandwidth_hz <= 0.0:
        raise ValueError("airtimes must be finite/non-negative and bandwidth_hz positive")
    if min_share < 0.0 or len(costs) * min_share >= 1.0:
        raise ValueError("minimum bandwidth share is infeasible")
    compute = np.asarray([float(cluster_compute_delays.get(int(cluster), 0.0)) for cluster in clusters], dtype=np.float64)
    if client_offsets is not None:
        offsets = np.asarray(client_offsets, dtype=np.float64)
        if offsets.shape != costs.shape or not np.isfinite(offsets).all() or (offsets < 0.0).any():
            raise ValueError("client offsets must be a matching finite non-negative vector")
        compute = compute + offsets
    if not np.isfinite(compute).all() or (compute < 0.0).any():
        raise ValueError("cluster compute delays must be finite and non-negative")
    equal = np.full(len(costs), 1.0 / len(costs), dtype=np.float64)
    if costs.sum() <= 1e-15:
        return equal
    low = float(compute.max())
    high = float(np.max(compute + costs / (float(bandwidth_hz) * equal)))
    tiny = np.finfo(np.float64).tiny
    for _ in range(int(iterations)):
        threshold = 0.5 * (low + high)
        denominators = float(bandwidth_hz) * np.maximum(threshold - compute, tiny)
        allocation = np.maximum(min_share, costs / denominators)
        if allocation.sum() > 1.0:
            low = threshold
        else:
            high = threshold
    denominators = float(bandwidth_hz) * np.maximum(high - compute, tiny)
    allocation = np.maximum(min_share, costs / denominators)
    residual = max(1.0 - float(allocation.sum()), 0.0)
    if residual:
        allocation += residual * costs / max(float(costs.sum()), tiny)
    allocation /= allocation.sum()
    if (allocation < min_share - 1e-10).any() or not np.isfinite(allocation).all():
        raise FloatingPointError("offset-aware water filling produced an invalid allocation")
    return allocation

def max_transmission_delay(required_airtimes, bandwidth_shares, bandwidth_hz):
    costs = np.asarray(required_airtimes, dtype=np.float64)
    shares = np.asarray(bandwidth_shares, dtype=np.float64)
    if costs.shape != shares.shape or bandwidth_hz <= 0.0 or (shares <= 0.0).any():
        raise ValueError("invalid airtime, bandwidth shares, or physical bandwidth")
    return float(np.max(costs / shares) / float(bandwidth_hz))


def channel_bandwidth_metrics(channel_gains, payload_bytes, bandwidth_shares, bandwidth_hz, min_share=0.01):
    gains = np.asarray(channel_gains, dtype=np.float64)
    shares = np.asarray(bandwidth_shares, dtype=np.float64)
    airtimes = required_airtime(payload_bytes, gains)
    equal = np.full(len(shares), 1.0 / len(shares), dtype=np.float64)
    oracle = oracle_bandwidth(airtimes, min_share=min_share)
    actual_delay = max_transmission_delay(airtimes, shares, bandwidth_hz)
    equal_delay = max_transmission_delay(airtimes, equal, bandwidth_hz)
    oracle_delay = max_transmission_delay(airtimes, oracle, bandwidth_hz)
    opportunity = max(equal_delay - oracle_delay, 0.0)
    closure = (equal_delay - actual_delay) / opportunity if opportunity > 1e-12 else 0.0
    return {
        "channel_bandwidth_spearman": spearman_correlation(gains, shares),
        "required_airtime_bandwidth_spearman": spearman_correlation(airtimes, shares),
        "actual_bandwidth_tx_delay_ms": actual_delay * 1000.0,
        "equal_bandwidth_tx_delay_ms": equal_delay * 1000.0,
        "oracle_bandwidth_tx_delay_ms": oracle_delay * 1000.0,
        "equal_bandwidth_improvement": (equal_delay - actual_delay) / max(equal_delay, 1e-12),
        "oracle_opportunity": opportunity / max(equal_delay, 1e-12),
        "oracle_gap_closure": closure,
        "oracle_bandwidth": oracle,
        "required_airtimes": airtimes,
    }
