"""One-episode training and frozen three-seed comparison for hybrid MAT."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from models.mat_agent import MATAgent
from scenario_a_runner import run_scenario_a
from utils.trajectory_buffer import MATTrajectoryBuffer

TRAIN_SEED, VALIDATION_SEEDS = 41, (7, 17, 29)
BASELINES = ("cpsl", "clustersfl", "pcsfl")


def _rows(result):
    return [row for row in next(iter(result["logger"].records.values())) if int(row["epoch"]) > 5]


def _summary(rows):
    def mean(key):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else 0.0
    accuracies = [float(row["test_accuracy"]) for row in rows if row.get("test_accuracy") is not None]
    return {"total_delay_ms": mean("total_delay_ms"), "tx_delay_ms": mean("tx_delay_ms"),
            "compute_delay_ms": mean("compute_delay_ms"), "reward": mean("reward"),
            "final_test_accuracy": accuracies[-1] if accuracies else 0.0}


def _bootstrap_lower(differences, samples=10000):
    rng, seeds = np.random.default_rng(20260805), sorted(differences)
    estimates = []
    for _ in range(samples):
        values = []
        for seed in rng.choice(seeds, size=len(seeds), replace=True):
            vector = np.asarray(differences[int(seed)])
            values.extend(rng.choice(vector, size=len(vector), replace=True))
        estimates.append(np.mean(values))
    return float(np.quantile(estimates, 0.025))


def run_fixed_budget(data_dir="../Data", log_dir="logs", device=None,
                     batch_size=16, local_steps=4, evaluation_batches=10):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(log_dir) / f"mat_hybrid_fixed_budget_{datetime.now():%Y%m%d_%H%M%S}"
    root.mkdir(parents=True, exist_ok=False)
    np.random.seed(20260805)
    torch.manual_seed(20260805)
    agent = MATAgent(state_dim=102, hidden_dim=128, ppo_epochs=10, minibatch_size=256,
                     bandwidth_policy="hybrid_water_filling", component_balanced_ppo=True, device=device)
    common = dict(data_dir=data_dir, log_dir=str(root), total_epochs=150, batch_size=batch_size,
                  local_steps=local_steps, evaluation_batches=evaluation_batches,
                  device=device, create_plots=False)
    buffer, started = MATTrajectoryBuffer(), time.time()
    train = run_scenario_a(algorithm="mat", seed=TRAIN_SEED, run_name="hybrid_train_seed_41",
                           mat_agent=agent, trajectory_buffer=buffer, mat_training_mode="collect",
                           client_diagnostics_path=root / "train_seed_41_clients.json", **common)
    if len(buffer) != 435 or buffer.policy_version != 0:
        raise RuntimeError("training budget must be exactly 435 version-0 transitions")
    kwargs, update_started = buffer.as_ppo_kwargs(), time.time()
    update = agent.update_policy(kwargs.pop("rewards"), kwargs.pop("next_states"), kwargs.pop("dones"), **kwargs)
    update["ppo_update_elapsed_seconds"] = time.time() - update_started
    checkpoint = root / "hybrid_after_one_episode.pt"
    agent.save_checkpoint(checkpoint)
    training_seconds = time.time() - started

    candidate, baselines, traces = {}, {name: {} for name in BASELINES}, {}
    for seed in VALIDATION_SEEDS:
        result = run_scenario_a(
            algorithm="mat", seed=seed, run_name=f"hybrid_validation_seed_{seed}", mat_agent=agent,
            trajectory_buffer=MATTrajectoryBuffer(), mat_training_mode="collect", mat_deterministic=True,
            client_diagnostics_path=root / f"hybrid_validation_seed_{seed}_clients.json", **common)
        candidate[seed], traces[seed] = _rows(result), result["trace_id"]
    for name in BASELINES:
        for seed in VALIDATION_SEEDS:
            result = run_scenario_a(algorithm=name, seed=seed, run_name=f"{name}_seed_{seed}", **common)
            if result["trace_id"] != traces[seed]:
                raise RuntimeError("comparison traces differ")
            baselines[name][seed] = _rows(result)

    candidate_summary = _summary(sum(candidate.values(), []))
    baseline_summary = {name: _summary(sum(rows.values(), [])) for name, rows in baselines.items()}
    best = min(BASELINES, key=lambda name: baseline_summary[name]["total_delay_ms"])
    differences, equal_changes, allocator_times, closures, per_seed = {}, [], [], [], {}
    no_degradation = True
    for seed in VALIDATION_SEEDS:
        hybrid, baseline = candidate[seed], baselines[best][seed]
        actual = np.asarray([float(row["total_delay_ms"]) for row in hybrid])
        equal = np.asarray([float(row["allocator_equal_total_delay_ms"]) for row in hybrid])
        reference = np.asarray([float(row["total_delay_ms"]) for row in baseline])
        differences[seed] = ((reference - actual) / np.maximum(reference, 1e-12)).tolist()
        equal_changes.extend(((equal - actual) / np.maximum(equal, 1e-12)).tolist())
        allocator_times.extend(float(row["allocator_elapsed_ms"]) for row in hybrid)
        closures.extend(float(row["allocator_oracle_gap_closure"]) for row in hybrid
                        if float(row["allocator_equal_total_delay_ms"]) >
                        float(row["allocator_hybrid_total_delay_ms"]) + 1e-9)
        no_degradation &= bool(np.all(actual <= equal + 1e-6))
        per_seed[str(seed)] = {"candidate": _summary(hybrid), "best_baseline": _summary(baseline),
                               "relative_delay_reduction": float(np.mean(differences[seed]))}
    reduction, ci_lower = float(np.mean(sum(differences.values(), []))), _bootstrap_lower(differences)
    best_accuracy = max(value["final_test_accuracy"] for value in baseline_summary.values())
    finite_update = all(np.isfinite(float(value)) for value in update.values()
                        if isinstance(value, (int, float, np.number)))
    gates = {
        "single_150_round_training_episode": len(buffer) == 435 and update["episode_count"] == 1,
        "hybrid_vs_equal_at_least_0_03": float(np.mean(equal_changes)) >= 0.03,
        "no_station_round_degrades_vs_equal": no_degradation,
        "oracle_gap_closure_at_least_0_999": (float(np.mean(closures)) if closures else 1.0) >= 0.999,
        "allocator_median_below_1_ms": float(np.median(allocator_times)) < 1.0,
        "allocator_p99_below_5_ms": float(np.quantile(allocator_times, 0.99)) < 5.0,
        "candidate_beats_best_baseline_by_0_10": reduction >= 0.10,
        "each_seed_beats_best_baseline": all(np.mean(value) > 0.0 for value in differences.values()),
        "paired_bootstrap_lower_above_zero": ci_lower > 0.0,
        "reward_above_best_baseline": candidate_summary["reward"] > baseline_summary[best]["reward"],
        "accuracy_within_two_percentage_points": candidate_summary["final_test_accuracy"] >= best_accuracy - 0.02,
        "ppo_critic_numerically_stable": bool(finite_update and update["target_drift_during_update"] == 0.0
                                               and update["grad_norm_post_max"] <= 0.500001),
    }
    report = {"schema_version": 3, "status": "completed", "passed": all(gates.values()),
              "training": {"seed": TRAIN_SEED, "rounds": 150, "transitions": len(buffer),
                           "elapsed_seconds": training_seconds, "checkpoint": str(checkpoint),
                           "trace_id": train["trace_id"], "update": update},
              "validation_seeds": list(VALIDATION_SEEDS), "trace_ids": traces,
              "candidate": candidate_summary, "baselines": baseline_summary, "best_baseline": best,
              "per_seed": per_seed,
              "bandwidth": {"equal_delay_reduction": float(np.mean(equal_changes)),
                            "oracle_gap_closure": float(np.mean(closures)) if closures else 1.0,
                            "allocator_median_ms": float(np.median(allocator_times)),
                            "allocator_p99_ms": float(np.quantile(allocator_times, 0.99))},
              "comparison": {"relative_delay_reduction": reduction,
                             "paired_bootstrap_95pct_lower": ci_lower}, "gates": gates,
              "claim": "fixed-budget hybrid MAT" if all(gates.values()) else "no superiority claim"}
    (root / "fixed_budget_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return root, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../Data")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    root, report = run_fixed_budget(args.data_dir, args.log_dir, args.device)
    print(json.dumps({"output_dir": str(root), **report}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
