"""Fixed-150-round online-adaptation evaluation for hybrid MAT and Scenario-A baselines."""
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from models.mat_agent import MATAgent
from scenario_a_runner import run_scenario_a
from utils.trajectory_buffer import MATTrajectoryBuffer


SEEDS = (7, 17, 29)
BASELINES = ("cpsl", "clustersfl", "pcsfl")
SNAPSHOT_EPOCHS = (99, 104, 119, 150)
PHASES = {
    "full_post_warmup": (6, 150),
    "pre_tide": (6, 99),
    "immediate_tide": (100, 104),
    "adaptation": (105, 119),
    "adapted": (120, 150),
}
PAYLOAD_BY_L1 = np.asarray([49152.0, 1048576.0, 1048576.0, 524288.0, 262144.0, 131072.0])


def _records(result, start=1, end=150):
    rows = next(iter(result["logger"].records.values()))
    return [row for row in rows if start <= int(row["epoch"]) <= end]


def _summary(rows):
    def mean(key):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else 0.0

    accuracy = [float(row["test_accuracy"]) for row in rows if row.get("test_accuracy") is not None]
    return {
        "total_delay_ms": mean("total_delay_ms"),
        "tx_delay_ms": mean("tx_delay_ms"),
        "compute_delay_ms": mean("compute_delay_ms"),
        "reward": mean("reward"),
        "physical_cluster_count": mean("physical_cluster_count"),
        "available_migs": mean("available_migs"),
        "mean_l1": mean("mean_l1"),
        "mean_l2": mean("mean_l2"),
        "payload_bytes_per_client": mean("smashed_data_bytes_per_client_mean"),
        "final_test_accuracy": accuracy[-1] if accuracy else 0.0,
    }


def _aggregate(by_seed, start, end):
    summaries = {str(seed): _summary([row for row in rows if start <= int(row["epoch"]) <= end])
                 for seed, rows in by_seed.items()}
    pooled = _summary(sum(([row for row in rows if start <= int(row["epoch"]) <= end]
                           for rows in by_seed.values()), []))
    pooled["final_test_accuracy"] = float(np.mean(
        [summary["final_test_accuracy"] for summary in summaries.values()]))
    return {"overall": pooled, "seeds": summaries}


def _paired_vectors(candidate, reference, start, end):
    output = {}
    for seed in SEEDS:
        left = {(int(row["epoch"]), int(row["station_id"])): row for row in candidate[seed]
                if start <= int(row["epoch"]) <= end}
        right = {(int(row["epoch"]), int(row["station_id"])): row for row in reference[seed]
                 if start <= int(row["epoch"]) <= end}
        if left.keys() != right.keys():
            raise RuntimeError("paired Scenario-A rows do not align")
        output[seed] = np.asarray([
            (float(right[key]["total_delay_ms"]) - float(left[key]["total_delay_ms"]))
            / max(float(right[key]["total_delay_ms"]), 1e-12)
            for key in sorted(left)
        ])
    return output


def _bootstrap_lower(vectors, samples=10000):
    rng = np.random.default_rng(20260806)
    estimates = []
    seeds = np.asarray(sorted(vectors))
    for _ in range(samples):
        values = []
        for seed in rng.choice(seeds, size=len(seeds), replace=True):
            vector = vectors[int(seed)]
            values.extend(rng.choice(vector, size=len(vector), replace=True))
        estimates.append(float(np.mean(values)))
    return float(np.quantile(estimates, 0.025))


def _probe_policy(agent, state, bandwidth_hz, client_ids, samples, seed):
    output = {}
    for available_migs in (2, 5, 7):
        edge = np.asarray([available_migs, bandwidth_hz], dtype=np.float32)
        deterministic, _ = agent.act(
            state, available_migs, edge, client_ids=client_ids, deterministic=True)
        torch.manual_seed(seed + available_migs)
        np.random.seed(seed + available_migs)
        stochastic = [agent.act(
            state, available_migs, edge, client_ids=client_ids, deterministic=False)[0]
                      for _ in range(samples)]

        def action_metrics(actions):
            cluster_counts = [len(np.unique(action["cluster"])) for action in actions]
            l1 = np.concatenate([np.asarray(action["l1"], dtype=np.int64) for action in actions])
            l2 = np.concatenate([np.asarray(action["l2"], dtype=np.int64) for action in actions])
            return {
                "physical_cluster_count": float(np.mean(cluster_counts)),
                "mean_l1": float(np.mean(l1)),
                "mean_l2": float(np.mean(l2)),
                "estimated_payload_bytes_per_client": float(np.mean(PAYLOAD_BY_L1[l1])),
            }

        output[str(available_migs)] = {
            "deterministic": action_metrics([deterministic]),
            "stochastic": action_metrics(stochastic),
        }
    return output


def _finite_updates(updates):
    for update in updates:
        for value in update.values():
            if isinstance(value, (int, float, np.integer, np.floating)) and not np.isfinite(float(value)):
                return False
        if float(update["target_drift_during_update"]) != 0.0:
            return False
        if float(update["grad_norm_post_max"]) > 0.500001:
            return False
        if int(update["policy_version_count"]) != 1:
            return False
    return True


def run_online_adaptation(data_dir="../Data", log_dir="logs", device=None, batch_size=16,
                          local_steps=4, evaluation_batches=10, ppo_update_interval=48,
                          shadow_samples=32):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(log_dir) / f"mat_hybrid_online_{datetime.now():%Y%m%d_%H%M%S}"
    root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20260806)
    np.random.seed(20260806)
    initial = MATAgent(
        state_dim=102, hidden_dim=128, ppo_epochs=10, minibatch_size=256,
        bandwidth_policy="hybrid_water_filling", component_balanced_ppo=True, device=device)
    initial_checkpoint = root / "hybrid_initial.pt"
    initial.save_checkpoint(initial_checkpoint)
    initial_sha256 = hashlib.sha256(initial_checkpoint.read_bytes()).hexdigest()
    del initial

    common = dict(
        data_dir=data_dir, log_dir=str(root), total_epochs=150, batch_size=batch_size,
        local_steps=local_steps, evaluation_batches=evaluation_batches, device=device,
        create_plots=False, ppo_update_interval=ppo_update_interval,
    )
    candidate, baselines, traces, updates, snapshots = {}, {name: {} for name in BASELINES}, {}, {}, {}
    elapsed = {}
    baseline_elapsed = {name: {} for name in BASELINES}
    trace_objects = {}
    for seed in SEEDS:
        print(f"[MAT] seed={seed}: starting 150-round online run", flush=True)
        agent = MATAgent.load_checkpoint(initial_checkpoint, device=device)
        seed_snapshots = {}

        def save_snapshot(*, epoch, agent, logger, trace, seed=seed):
            if epoch in SNAPSHOT_EPOCHS:
                path = root / f"hybrid_seed_{seed}_epoch_{epoch}.pt"
                agent.save_checkpoint(path)
                seed_snapshots[epoch] = str(path)

        started = time.time()
        result = run_scenario_a(
            algorithm="mat", seed=seed, run_name=f"hybrid_online_seed_{seed}", mat_agent=agent,
            trajectory_buffer=MATTrajectoryBuffer(), mat_training_mode="online", mat_deterministic=False,
            client_diagnostics_path=root / f"hybrid_online_seed_{seed}_clients.json",
            epoch_callback=save_snapshot, **common)
        elapsed[str(seed)] = time.time() - started
        print(f"[MAT] seed={seed}: completed in {elapsed[str(seed)]:.1f}s", flush=True)
        candidate[seed] = _records(result)
        traces[seed] = result["trace_id"]
        trace_objects[seed] = result["trace"]
        updates[seed] = result["ppo_updates"]
        snapshots[seed] = seed_snapshots
        if sum(int(item["transition_count"]) for item in updates[seed]) != 435:
            raise RuntimeError("each online seed must consume exactly 435 transitions")
        versions = [int(item["policy_version_after"]) for item in updates[seed]]
        if versions != list(range(1, len(versions) + 1)):
            raise RuntimeError("online PPO policy versions are not sequential")

    for name in BASELINES:
        for seed in SEEDS:
            print(f"[{name}] seed={seed}: starting paper-adapted baseline", flush=True)
            started = time.time()
            result = run_scenario_a(
                algorithm=name, seed=seed, run_name=f"{name}_online_protocol_seed_{seed}", **common)
            if result["trace_id"] != traces[seed]:
                raise RuntimeError("comparison traces differ")
            baselines[name][seed] = _records(result)
            baseline_elapsed[name][str(seed)] = time.time() - started
            print(f"[{name}] seed={seed}: completed in "
                  f"{baseline_elapsed[name][str(seed)]:.1f}s", flush=True)

    shadow = {}
    for seed in SEEDS:
        shadow[str(seed)] = {"initial": {}, "snapshots": {}}
        trace = trace_objects[seed]
        initial_agent = MATAgent.load_checkpoint(initial_checkpoint, device=device, load_optimizer=False)
        for epoch in SNAPSHOT_EPOCHS:
            observation_epoch = min(epoch + 1, 150)
            state, _, bandwidth, client_ids = trace.get(observation_epoch, 1)
            if epoch == SNAPSHOT_EPOCHS[0]:
                shadow[str(seed)]["initial"] = _probe_policy(
                    initial_agent, state, bandwidth, client_ids, shadow_samples, 10000 + seed)
            learned = MATAgent.load_checkpoint(
                snapshots[seed][epoch], device=device, load_optimizer=False)
            shadow[str(seed)]["snapshots"][str(epoch)] = _probe_policy(
                learned, state, bandwidth, client_ids, shadow_samples, epoch * 100 + seed)
        del initial_agent
    (root / "cluster_split_shadow_probe.json").write_text(
        json.dumps(shadow, indent=2, allow_nan=False), encoding="utf-8")

    summaries = {
        "candidate": {phase: _aggregate(candidate, *bounds) for phase, bounds in PHASES.items()},
        "baselines": {
            name: {phase: _aggregate(rows, *bounds) for phase, bounds in PHASES.items()}
            for name, rows in baselines.items()
        },
    }
    best = min(BASELINES, key=lambda name:
               summaries["baselines"][name]["full_post_warmup"]["overall"]["total_delay_ms"])
    comparisons = {}
    for phase, bounds in PHASES.items():
        vectors = _paired_vectors(candidate, baselines[best], *bounds)
        relative = float(np.mean(np.concatenate(list(vectors.values()))))
        comparisons[phase] = {
            "best_baseline": best,
            "mean_paired_relative_delay_reduction": relative,
            "ratio_of_mean_delay_reduction": 1.0 - (
                summaries["candidate"][phase]["overall"]["total_delay_ms"]
                / summaries["baselines"][best][phase]["overall"]["total_delay_ms"]),
            "bootstrap_95pct_lower": _bootstrap_lower(vectors),
            "per_seed": {str(seed): float(np.mean(vector)) for seed, vector in vectors.items()},
        }

    candidate_rows = sum(candidate.values(), [])
    equal_changes = [
        (float(row["allocator_equal_total_delay_ms"]) - float(row["total_delay_ms"]))
        / max(float(row["allocator_equal_total_delay_ms"]), 1e-12)
        for row in candidate_rows if int(row["epoch"]) >= 6
    ]
    allocator_times = [float(row["allocator_elapsed_ms"]) for row in candidate_rows if int(row["epoch"]) >= 6]
    no_degradation = all(float(row["total_delay_ms"]) <=
                         float(row["allocator_equal_total_delay_ms"]) + 1e-6
                         for row in candidate_rows if int(row["epoch"]) >= 6)
    cluster_expansion = {}
    for seed, rows in candidate.items():
        pre = [float(row["physical_cluster_count"]) for row in rows
               if int(row["station_id"]) == 1 and 6 <= int(row["epoch"]) <= 99]
        immediate = [float(row["physical_cluster_count"]) for row in rows
                     if int(row["station_id"]) == 1 and 100 <= int(row["epoch"]) <= 104]
        cluster_expansion[str(seed)] = float(np.mean(immediate) - np.mean(pre))

    all_updates = sum(updates.values(), [])
    full_comparison = comparisons["full_post_warmup"]
    candidate_accuracy = summaries["candidate"]["full_post_warmup"]["overall"]["final_test_accuracy"]
    best_accuracy = max(summaries["baselines"][name]["full_post_warmup"]["overall"]["final_test_accuracy"]
                        for name in BASELINES)
    gates = {
        "three_independent_150_round_online_runs": all(
            sum(int(item["transition_count"]) for item in updates[seed]) == 435 for seed in SEEDS),
        "same_initial_checkpoint": bool(initial_sha256),
        "online_ppo_numerically_stable": _finite_updates(all_updates),
        "hybrid_vs_equal_at_least_0_03": float(np.mean(equal_changes)) >= 0.03,
        "no_station_round_degrades_vs_equal": no_degradation,
        "allocator_median_below_1_ms": float(np.median(allocator_times)) < 1.0,
        "allocator_p99_below_5_ms": float(np.quantile(allocator_times, 0.99)) < 5.0,
        "station_1_expands_after_tide_each_seed": all(value > 0.0 for value in cluster_expansion.values()),
        "candidate_beats_best_baseline_by_0_10":
            full_comparison["mean_paired_relative_delay_reduction"] >= 0.10,
        "each_seed_beats_best_baseline": all(
            value > 0.0 for value in full_comparison["per_seed"].values()),
        "paired_bootstrap_lower_above_zero": full_comparison["bootstrap_95pct_lower"] > 0.0,
        "reward_above_best_baseline":
            summaries["candidate"]["full_post_warmup"]["overall"]["reward"]
            > summaries["baselines"][best]["full_post_warmup"]["overall"]["reward"],
        "accuracy_within_two_percentage_points": candidate_accuracy >= best_accuracy - 0.02,
    }
    report = {
        "schema_version": 5,
        "status": "completed",
        "passed": all(gates.values()),
        "protocol": {
            "seeds": list(SEEDS), "rounds_per_seed": 150, "warmup_rounds": 5,
            "ppo_update_interval_transitions": ppo_update_interval,
            "ppo_update_interval_rounds_nominal": ppo_update_interval / 3.0,
            "online_behavior": "stochastic",
            "causality": "record reward, close transition with next observation, update, then act",
            "cross_seed_parameter_carry": False,
            "initial_checkpoint": str(initial_checkpoint), "initial_checkpoint_sha256": initial_sha256,
            "baseline_fidelity": "paper-adapted-v2",
            "baseline_contract": "baselines/FIDELITY.md",
            "baseline_external_metrics": "shared Scenario-A delay/reward/accuracy",
            "baseline_internal_objectives": {
                "cpsl": "fixed split + Gibbs clustering + greedy subchannels",
                "clustersfl": "KL clustering + top-worker compression + local frequency",
                "pcsfl": "model-PCA recurrent dual-DDQN + paper factor reward + hierarchy",
            },
        },
        "elapsed_seconds_by_seed": elapsed,
        "baseline_elapsed_seconds": baseline_elapsed,
        "trace_ids": {str(seed): trace_id for seed, trace_id in traces.items()},
        "ppo_updates": {str(seed): value for seed, value in updates.items()},
        "snapshots": {str(seed): {str(epoch): path for epoch, path in value.items()}
                      for seed, value in snapshots.items()},
        "summaries": summaries,
        "best_baseline": best,
        "comparisons": comparisons,
        "bandwidth": {
            "mean_paired_equal_delay_reduction": float(np.mean(equal_changes)),
            "allocator_median_ms": float(np.median(allocator_times)),
            "allocator_p99_ms": float(np.quantile(allocator_times, 0.99)),
        },
        "station_1_immediate_cluster_expansion": cluster_expansion,
        "gates": gates,
        "claim": ("online-adaptive hybrid MAT versus paper-adapted baselines"
                  if all(gates.values()) else "no superiority claim"),
        "shadow_probe_role": "diagnostic only; excluded from performance gates",
    }
    (root / "online_adaptation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return root, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../Data")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-steps", type=int, default=4)
    parser.add_argument("--evaluation-batches", type=int, default=10)
    parser.add_argument("--ppo-update-interval", type=int, default=48)
    parser.add_argument("--shadow-samples", type=int, default=32)
    args = parser.parse_args()
    root, report = run_online_adaptation(
        args.data_dir, args.log_dir, args.device, args.batch_size, args.local_steps,
        args.evaluation_batches, args.ppo_update_interval, args.shadow_samples)
    print(json.dumps({"output_dir": str(root), **report}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
