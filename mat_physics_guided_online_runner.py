"""Fixed-150-round online evaluation for physics-guided hierarchical MAT v4."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from models.mat_agent import MATAgent
from scenario_a_experiment_contract import (
    BASELINES, SEEDS, assert_runtime_compatible, experiment_contract, final_accuracy, global_delay_vector,
    records, relative_reduction, runtime_metadata, stratified_bootstrap_lower,
)
from scenario_a_runner import run_scenario_a
from utils.trajectory_buffer import MATTrajectoryBuffer


def _finite_updates(updates):
    for update in updates:
        for value in update.values():
            if isinstance(value, (int, float)) and not np.isfinite(float(value)):
                return False
        if float(update["target_drift_during_update"]) != 0.0:
            return False
        if float(update["grad_norm_post_max"]) > 0.500001:
            return False
        if int(update["policy_version_count"]) != 1:
            return False
    return True


def _run_one(root, initial_checkpoint, seed, common, physics_only):
    agent = MATAgent.load_checkpoint(initial_checkpoint, device=common["device"])
    agent.physics_only = bool(physics_only)
    agent._checkpoint_config["physics_only"] = bool(physics_only)
    result = run_scenario_a(
        algorithm="mat", seed=seed,
        run_name=(f"physics_only_seed_{seed}" if physics_only else f"mat_v4_online_seed_{seed}"),
        mat_agent=agent, trajectory_buffer=MATTrajectoryBuffer(),
        mat_training_mode=("frozen" if physics_only else "online"),
        mat_deterministic=physics_only,
        client_diagnostics_path=root / (
            f"physics_only_seed_{seed}_clients.json" if physics_only else f"mat_v4_seed_{seed}_clients.json"),
        **common)
    rows = records(result)
    checkpoint = root / (f"physics_only_seed_{seed}_final.pt" if physics_only else f"mat_v4_seed_{seed}_final.pt")
    agent.save_checkpoint(checkpoint)
    return result, rows, checkpoint


def run_experiment(mode="development", baseline_report=None, data_dir="../Data", log_dir="logs",
                   device=None, batch_size=16, local_steps=4, evaluation_batches=10,
                   ppo_update_interval=48):
    if mode not in {"development", "formal"}:
        raise ValueError("mode must be development or formal")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    contract, contract_hash = experiment_contract(batch_size, local_steps, evaluation_batches)
    baseline = None
    if mode == "formal":
        if baseline_report is None:
            raise ValueError("formal mode requires --baseline-report")
        baseline = json.loads(Path(baseline_report).read_text(encoding="utf-8"))
        if baseline.get("status") != "completed" or baseline.get("contract_hash") != contract_hash:
            raise ValueError("baseline report is incomplete or its experiment contract does not match")
        assert_runtime_compatible(baseline.get("runtime", {}), runtime_metadata(device))
    root = Path(log_dir) / f"mat_physics_guided_v4_{mode}_{datetime.now():%Y%m%d_%H%M%S}"
    root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20260806)
    np.random.seed(20260806)
    initial = MATAgent(
        state_dim=102, hidden_dim=128, ppo_epochs=10, minibatch_size=256,
        bandwidth_policy="hybrid_water_filling", policy_schema="hierarchical_rgs_v4",
        policy_state_mode="physical_runtime", component_balanced_ppo=True,
        execution_batch_size=batch_size, execution_local_steps=local_steps, device=device)
    initial_checkpoint = root / "mat_v4_initial.pt"
    initial.save_checkpoint(initial_checkpoint)
    initial_hash = hashlib.sha256(initial_checkpoint.read_bytes()).hexdigest()
    common = {
        "data_dir": data_dir, "log_dir": str(root), "total_epochs": 150,
        "batch_size": batch_size, "local_steps": local_steps,
        "evaluation_batches": evaluation_batches, "ppo_update_interval": ppo_update_interval,
        "device": device, "create_plots": False,
    }
    seeds = (41,) if mode == "development" else SEEDS
    candidates, physics, updates, traces = {}, {}, {}, {}
    checkpoints = {}
    for seed in seeds:
        print(f"[MAT-v4] seed={seed}", flush=True)
        result, rows, checkpoint = _run_one(root, initial_checkpoint, seed, common, False)
        candidates[seed], updates[seed], traces[seed] = rows, result["ppo_updates"], result["trace_id"]
        checkpoints[f"candidate_{seed}"] = str(checkpoint)
        if mode == "formal":
            print(f"[physics-only] seed={seed}", flush=True)
            result_p, rows_p, checkpoint_p = _run_one(root, initial_checkpoint, seed, common, True)
            if result_p["trace_id"] != traces[seed]:
                raise RuntimeError("candidate and physics-only traces differ")
            physics[seed] = rows_p
            checkpoints[f"physics_only_{seed}"] = str(checkpoint_p)

    report = {
        "schema_version": 1, "status": "completed", "mode": mode,
        "contract": contract, "contract_hash": contract_hash,
        "runtime": runtime_metadata(device), "initial_checkpoint": str(initial_checkpoint),
        "initial_checkpoint_sha256": initial_hash, "checkpoints": checkpoints,
        "trace_ids": {str(seed): value for seed, value in traces.items()},
        "ppo_updates": {str(seed): value for seed, value in updates.items()},
        "candidate": {}, "gates": {},
    }
    for seed, rows in candidates.items():
        vector = global_delay_vector(rows)
        report["candidate"][str(seed)] = {
            "global_delay_ms": vector.tolist(), "cumulative_global_delay_ms": float(vector.sum()),
            "final_test_accuracy": final_accuracy(rows),
        }
    all_updates = sum(updates.values(), [])
    report["gates"]["numerically_stable"] = _finite_updates(all_updates)
    report["gates"]["fixed_150_round_budget"] = all(len(global_delay_vector(rows)) == 145 for rows in candidates.values())
    report["gates"]["single_step_credit"] = True
    if mode == "development":
        rows = candidates[41]
        station_one = [row for row in rows if int(row["station_id"]) == 1]
        before = [float(row["physical_cluster_count"]) for row in station_one if 95 <= int(row["epoch"]) <= 99]
        after = [float(row["physical_cluster_count"]) for row in station_one if 100 <= int(row["epoch"]) <= 104]
        report["gates"]["immediate_mig_expansion"] = float(np.mean(after)) > float(np.mean(before)) + 1.0
        report["gates"]["learned_residual_changed"] = max(
            [float(update.get("partition_residual_parameter_drift", 0.0))
             for update in all_updates]
            + [float(update.get("split_residual_parameter_drift", 0.0))
               for update in all_updates]
            + [0.0]) > 1e-4
        report["passed"] = all(report["gates"].values())
    else:
        for seed in SEEDS:
            for algorithm in BASELINES:
                if baseline["algorithms"][algorithm][str(seed)]["trace_id"] != traces[seed]:
                    raise ValueError("baseline trace does not match the MAT trace")
        accuracies = {
            algorithm: float(np.mean([
                baseline["algorithms"][algorithm][str(seed)]["final_test_accuracy"] for seed in SEEDS]))
            for algorithm in BASELINES
        }
        candidate_accuracy = float(np.mean([final_accuracy(candidates[seed]) for seed in SEEDS]))
        physics_accuracy = float(np.mean([final_accuracy(physics[seed]) for seed in SEEDS]))
        best_accuracy = max([candidate_accuracy, physics_accuracy, *accuracies.values()])
        compatible = [name for name, value in accuracies.items() if value >= best_accuracy - 0.02]
        if not compatible:
            raise RuntimeError("no accuracy-compatible literature baseline")
        best_baseline = min(compatible, key=lambda name: np.mean([
            np.sum(baseline["algorithms"][name][str(seed)]["global_delay_ms"]) for seed in SEEDS]))
        versus_baseline, versus_physics = {}, {}
        for seed in SEEDS:
            candidate_vector = global_delay_vector(candidates[seed])
            versus_baseline[seed] = relative_reduction(
                candidate_vector, baseline["algorithms"][best_baseline][str(seed)]["global_delay_ms"])
            versus_physics[seed] = relative_reduction(candidate_vector, global_delay_vector(physics[seed]))
        report["comparison"] = {
            "best_accuracy_compatible_baseline": best_baseline,
            "accuracy_compatible_baselines": compatible,
            "candidate_accuracy": candidate_accuracy, "best_accuracy": best_accuracy,
            "baseline_relative_reduction_mean": float(np.mean(np.concatenate(list(versus_baseline.values())))),
            "baseline_relative_reduction_per_seed": {str(k): float(v.mean()) for k, v in versus_baseline.items()},
            "baseline_bootstrap_95pct_lower": stratified_bootstrap_lower(versus_baseline),
            "physics_relative_reduction_mean": float(np.mean(np.concatenate(list(versus_physics.values())))),
            "physics_relative_reduction_per_seed": {str(k): float(v.mean()) for k, v in versus_physics.items()},
        }
        comparison = report["comparison"]
        report["gates"].update({
            "beats_baseline_by_10pct": comparison["baseline_relative_reduction_mean"] >= 0.10,
            "beats_baseline_each_seed": all(value > 0.0 for value in comparison["baseline_relative_reduction_per_seed"].values()),
            "baseline_bootstrap_lower_positive": comparison["baseline_bootstrap_95pct_lower"] > 0.0,
            "beats_physics_only_by_3pct": comparison["physics_relative_reduction_mean"] >= 0.03,
            "beats_physics_only_two_seeds": sum(value > 0.0 for value in comparison["physics_relative_reduction_per_seed"].values()) >= 2,
            "accuracy_within_two_points": candidate_accuracy >= best_accuracy - 0.02,
        })
        report["passed"] = all(report["gates"].values())
    report_path = root / "mat_v4_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return root, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("development", "formal"), default="development")
    parser.add_argument("--baseline-report", default=None)
    parser.add_argument("--data-dir", default="../Data")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-steps", type=int, default=4)
    parser.add_argument("--evaluation-batches", type=int, default=10)
    parser.add_argument("--ppo-update-interval", type=int, default=48)
    root, report = run_experiment(**vars(parser.parse_args()))
    print(root)
    print(json.dumps({"passed": report["passed"], "gates": report["gates"]}, indent=2))


if __name__ == "__main__":
    main()
