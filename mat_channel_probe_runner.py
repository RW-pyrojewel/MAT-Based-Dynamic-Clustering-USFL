"""Equal-payload channel probe for MAT bandwidth credit assignment.

The probe deliberately excludes split payload and compute latency.  It trains on
wireless max-delay only, then evaluates physical bandwidth allocation metrics on
fixed holdout traces.
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from models.mat_agent import MATAgent
from utils.channel_diagnostics import channel_bandwidth_metrics, required_airtime, spearman_correlation
from utils.trajectory_buffer import MATTrajectoryBuffer


TRAIN_SEEDS = (401, 409, 419, 421, 431, 433, 439, 443)
HOLDOUT_SEEDS = (449, 457, 461)
STATIONS = (1, 2, 3)
CLIENTS = 10
ROUNDS = 20
PAYLOAD_BYTES = 4096.0
BANDWIDTH_HZ = 100e6
MIN_SHARE = 0.01


def _trace(seed, state_dim=6):
    """Return a reproducible 3-station tidal trace with equal client payload."""
    rng = np.random.default_rng(seed)
    traces = {}
    base = np.geomspace(0.025, 2.5, CLIENTS)
    for station in STATIONS:
        rows = []
        client_base = base[rng.permutation(CLIENTS)]
        station_scale = {1: 1.0, 2: 0.72, 3: 1.18}[station]
        phases = rng.uniform(0.0, 2.0 * np.pi, size=CLIENTS)
        for round_index in range(ROUNDS):
            tide = 1.0 + 0.52 * np.sin(2.0 * np.pi * round_index / ROUNDS + phases)
            fading = rng.lognormal(mean=0.0, sigma=0.20, size=CLIENTS)
            channels = np.clip(client_base * station_scale * tide * fading, 1e-4, 5.0)
            compute = rng.uniform(1.0, 2.5, size=CLIENTS)
            label_count = state_dim - 2
            labels = np.full((CLIENTS, label_count), 1.0 / label_count)
            rows.append(np.column_stack((channels, compute, labels)).astype(np.float32))
        traces[station] = rows
    return traces


def _wireless_reward(channels, bandwidths):
    airtime = required_airtime(np.full(CLIENTS, PAYLOAD_BYTES), channels)
    max_delay = float(np.max(airtime / (np.asarray(bandwidths) * BANDWIDTH_HZ)))
    return -max_delay


def _collect_cycle(agent, seeds):
    buffer = MATTrajectoryBuffer()
    for seed in seeds:
        traces = _trace(seed, agent.state_dim)
        for station in STATIONS:
            states = traces[station]
            for round_index, state in enumerate(states):
                edge = np.asarray([3.0, BANDWIDTH_HZ], dtype=np.float32)
                action, info = agent.act(
                    state, 3, edge, client_ids=np.arange(CLIENTS), deterministic=False
                )
                done = round_index == ROUNDS - 1
                next_state = state if done else states[round_index + 1]
                buffer.append(
                    state,
                    edge,
                    action,
                    _wireless_reward(state[:, 0], action["bw"]),
                    next_state,
                    edge,
                    done,
                    info,
                    3,
                    station,
                    round_index + 1,
                    trajectory_id=(seed, station),
                    policy_version=agent.policy_version,
                )
    kwargs = buffer.as_ppo_kwargs()
    diagnostics = agent.update_policy(
        kwargs.pop("rewards"), kwargs.pop("next_states"), kwargs.pop("dones"), **kwargs
    )
    buffer.clear()
    return diagnostics


def _evaluate(agent, seeds):
    records = []
    seed_summaries = []
    for seed in seeds:
        correlations = []
        counterfactual_correlations = []
        equal_delays, actual_delays, oracle_delays = [], [], []
        traces = _trace(seed, agent.state_dim)
        for station in STATIONS:
            for round_index, state in enumerate(traces[station]):
                edge = np.asarray([3.0, BANDWIDTH_HZ], dtype=np.float32)
                action, info = agent.act(
                    state, 3, edge, client_ids=np.arange(CLIENTS), deterministic=True
                )
                metrics = channel_bandwidth_metrics(
                    state[:, 0],
                    np.full(CLIENTS, PAYLOAD_BYTES),
                    action["bw"],
                    BANDWIDTH_HZ,
                    MIN_SHARE,
                )
                permutation = np.random.default_rng(seed * 1000 + station * 100 + round_index).permutation(CLIENTS)
                counterfactual_state = state.copy()
                counterfactual_state[:, 0] = state[permutation, 0]
                before = agent.evaluate_bandwidth_prefix_means(
                    state, edge, action, 3, info["decision_order"]
                )
                after = agent.evaluate_bandwidth_prefix_means(
                    counterfactual_state, edge, action, 3, info["decision_order"]
                )
                delta_required = required_airtime(
                    np.full(CLIENTS, PAYLOAD_BYTES), counterfactual_state[:, 0]
                ) - required_airtime(np.full(CLIENTS, PAYLOAD_BYTES), state[:, 0])
                counterfactual_rho = spearman_correlation(delta_required, after - before)
                correlations.append(metrics["required_airtime_bandwidth_spearman"])
                counterfactual_correlations.append(counterfactual_rho)
                equal_delays.append(metrics["equal_bandwidth_tx_delay_ms"])
                actual_delays.append(metrics["actual_bandwidth_tx_delay_ms"])
                oracle_delays.append(metrics["oracle_bandwidth_tx_delay_ms"])
                records.append(
                    {
                        "seed": seed,
                        "station_id": station,
                        "round": round_index + 1,
                        **{
                            key: value
                            for key, value in metrics.items()
                            if np.isscalar(value)
                        },
                        "counterfactual_delta_spearman": counterfactual_rho,
                        "bandwidth_latent_mean": float(np.mean(info["bandwidth_latent_means"])),
                    }
                )
        equal_delay = float(np.mean(equal_delays))
        actual_delay = float(np.mean(actual_delays))
        oracle_delay = float(np.mean(oracle_delays))
        opportunity = max(equal_delay - oracle_delay, 0.0)
        seed_summaries.append(
            {
                "seed": seed,
                "required_airtime_bandwidth_spearman": float(np.median(correlations)),
                "counterfactual_delta_spearman": float(np.median(counterfactual_correlations)),
                "equal_max_tx_delay_ms": equal_delay,
                "actual_max_tx_delay_ms": actual_delay,
                "oracle_max_tx_delay_ms": oracle_delay,
                "equal_delay_improvement": (equal_delay - actual_delay) / max(equal_delay, 1e-12),
                "oracle_gap_closure": (equal_delay - actual_delay) / max(opportunity, 1e-12),
                "oracle_regret_ms": actual_delay - oracle_delay,
            }
        )
    summary = {
        "seed_summaries": seed_summaries,
        "median_required_airtime_bandwidth_spearman": float(
            np.median([item["required_airtime_bandwidth_spearman"] for item in seed_summaries])
        ),
        "all_seed_correlations_positive": bool(
            all(item["required_airtime_bandwidth_spearman"] > 0.0 for item in seed_summaries)
        ),
        "median_counterfactual_delta_spearman": float(
            np.median([item["counterfactual_delta_spearman"] for item in seed_summaries])
        ),
        "equal_delay_improvement": float(
            np.mean([item["equal_delay_improvement"] for item in seed_summaries])
        ),
        "oracle_gap_closure": float(
            np.mean([item["oracle_gap_closure"] for item in seed_summaries])
        ),
        "oracle_regret_ms": float(np.mean([item["oracle_regret_ms"] for item in seed_summaries])),
    }
    return summary, records


def _numerically_stable(diagnostics):
    numeric = [
        value
        for value in diagnostics.values()
        if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))
    ]
    return (
        bool(np.isfinite(numeric).all())
        and diagnostics["grad_norm_post_max"] <= 0.500001
        and diagnostics["target_drift_during_update"] == 0.0
    )


def _gates(candidate, legacy, diagnostics):
    regret_reduction = (
        legacy["oracle_regret_ms"] - candidate["oracle_regret_ms"]
    ) / max(abs(legacy["oracle_regret_ms"]), 1e-12)
    gates = {
        "required_airtime_spearman": candidate["median_required_airtime_bandwidth_spearman"] >= 0.50,
        "all_holdout_seeds_positive": candidate["all_seed_correlations_positive"],
        "counterfactual_spearman": candidate["median_counterfactual_delta_spearman"] >= 0.50,
        "equal_delay_improvement": candidate["equal_delay_improvement"] >= 0.10,
        "oracle_gap_closure": candidate["oracle_gap_closure"] >= 0.30,
        "legacy_regret_reduction": regret_reduction >= 0.25,
        "numerical_stability": _numerically_stable(diagnostics),
    }
    return gates, float(regret_reduction)


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_probe(output_dir="logs", max_candidate_cycles=10, device="cpu"):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(output_dir) / f"mat_channel_probe_{stamp}"
    root.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(20260731)
    legacy = MATAgent(
        state_dim=6, hidden_dim=32, ppo_epochs=10, minibatch_size=256,
        channel_conditioning="legacy", device=device,
    )
    legacy_updates = []
    initial_legacy_summary, initial_legacy_records = _evaluate(legacy, HOLDOUT_SEEDS)
    for cycle in range(1, 4):
        np.random.seed(7000 + cycle)
        torch.manual_seed(7000 + cycle)
        diagnostics = _collect_cycle(legacy, TRAIN_SEEDS)
        diagnostics["cycle"] = cycle
        legacy_updates.append(diagnostics)
    legacy_summary, legacy_records = _evaluate(legacy, HOLDOUT_SEEDS)
    print("legacy probe complete", flush=True)
    bandwidth_share = float(np.median([item["bandwidth_actor_gradient_share"] for item in legacy_updates]))
    opportunity = float(
        np.mean(
            [
                (
                    item["equal_max_tx_delay_ms"] - item["oracle_max_tx_delay_ms"]
                ) / max(item["equal_max_tx_delay_ms"], 1e-12)
                for item in legacy_summary["seed_summaries"]
            ]
        )
    )
    latent_change = abs(
        float(np.mean([row["bandwidth_latent_mean"] for row in legacy_records]))
        - float(np.mean([row["bandwidth_latent_mean"] for row in initial_legacy_records]))
    )
    starvation = bandwidth_share < 0.20 or (opportunity >= 0.10 and latent_change < 1e-3)

    torch.manual_seed(20260731)
    candidate = MATAgent(
        state_dim=6, hidden_dim=32, ppo_epochs=10, minibatch_size=256,
        channel_conditioning="explicit", component_balanced_ppo=starvation, device=device,
    )
    candidate_updates, candidate_records = [], []
    candidate_summary = None
    final_gates, regret_reduction = {}, 0.0
    for cycle in range(1, max_candidate_cycles + 1):
        np.random.seed(7000 + cycle)
        torch.manual_seed(7000 + cycle)
        diagnostics = _collect_cycle(candidate, TRAIN_SEEDS)
        diagnostics["cycle"] = cycle
        candidate_updates.append(diagnostics)
        candidate_summary, candidate_records = _evaluate(candidate, HOLDOUT_SEEDS)
        final_gates, regret_reduction = _gates(candidate_summary, legacy_summary, diagnostics)
        candidate.save_checkpoint(root / f"candidate_cycle_{cycle}.pt")
        print(f"candidate cycle={cycle} gates={sum(final_gates.values())}/{len(final_gates)}", flush=True)
        if cycle >= 5 and all(final_gates.values()):
            break

    report = {
        "schema_version": 1,
        "configuration": {
            "train_seeds": TRAIN_SEEDS,
            "holdout_seeds": HOLDOUT_SEEDS,
            "stations": STATIONS,
            "clients": CLIENTS,
            "rounds": ROUNDS,
            "payload_bytes": PAYLOAD_BYTES,
        },
        "legacy": legacy_summary,
        "candidate": candidate_summary,
        "legacy_updates": legacy_updates,
        "candidate_updates": candidate_updates,
        "starvation": {
            "triggered": starvation,
            "bandwidth_actor_gradient_share": bandwidth_share,
            "oracle_improvement_opportunity": opportunity,
            "latent_mean_change": latent_change,
        },
        "candidate_component_balanced_ppo": candidate.component_balanced_ppo,
        "legacy_regret_reduction": regret_reduction,
        "gates": final_gates,
        "passed": bool(final_gates and all(final_gates.values())),
    }
    (root / "probe_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_csv(root / "legacy_holdout.csv", legacy_records)
    _write_csv(root / "candidate_holdout.csv", candidate_records)
    _write_csv(root / "legacy_updates.csv", legacy_updates)
    _write_csv(root / "candidate_updates.csv", candidate_updates)
    return root, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--max-candidate-cycles", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root, report = run_probe(args.output_dir, args.max_candidate_cycles, args.device)
    print(json.dumps({"output_dir": str(root), **report}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
