"""Staged Scenario-A training and frozen validation for channel-aware MAT."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from mat_channel_probe_runner import _evaluate as evaluate_probe
from mat_channel_probe_runner import _gates as probe_gates
from models.mat_agent import MATAgent
from scenario_a_runner import run_scenario_a
from utils.trajectory_buffer import MATTrajectoryBuffer


CYCLE_SEEDS = (
    (41, 53, 67, 79, 97, 113, 127, 139),
    (151, 163, 179, 191, 211, 223, 239, 251),
    (263, 277, 293, 307, 317, 331, 347, 359),
    (367, 373, 379, 383, 389, 397, 467, 479),
    (487, 491, 499, 503, 509, 521, 523, 541),
    (547, 557, 563, 569, 571, 577, 587, 593),
    (599, 601, 607, 613, 617, 619, 631, 641),
    (643, 647, 653, 659, 661, 673, 677, 683),
)
VALIDATION_SEEDS = (7, 17, 29)


def _load_probe_report(path):
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    required = {"legacy", "candidate", "passed"}
    if not required.issubset(report):
        raise ValueError("probe report has an incompatible schema")
    return report


def _bootstrap_lower(values, seed=20260731, samples=10000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return 0.0
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(samples, len(values)), replace=True), axis=1)
    return float(np.quantile(means, 0.025))


def _frozen_summary(results):
    seed_summaries = {}
    all_correlations = []
    all_closures = []
    station_seed_improvements = []
    for seed, result in results.items():
        rows = [
            row for row in result["logger"].records["MAT-RL"]
            if not row["is_warmup"]
        ]
        correlations = [row["required_airtime_bandwidth_spearman"] for row in rows]
        counterfactual = [row["channel_permutation_delta_spearman"] for row in rows]
        closures = [row["oracle_gap_closure"] for row in rows]
        all_correlations.extend(correlations)
        all_closures.extend(closures)
        by_station = {}
        for station in (1, 2, 3):
            station_rows = [row for row in rows if row["station_id"] == station]
            improvement = float(np.mean([
                row["equal_bandwidth_improvement"] for row in station_rows
            ]))
            by_station[str(station)] = {"equal_bandwidth_improvement": improvement}
            station_seed_improvements.append(improvement)
        seed_summaries[str(seed)] = {
            "required_airtime_bandwidth_spearman": float(np.median(correlations)),
            "counterfactual_delta_spearman": float(np.median(counterfactual)),
            "oracle_gap_closure": float(np.mean(closures)),
            "stations": by_station,
        }
    overall_rho = float(np.median(all_correlations))
    overall_closure = float(np.mean(all_closures))
    gates = {
        "each_seed_required_airtime_correlation_positive": all(
            item["required_airtime_bandwidth_spearman"] > 0.0
            for item in seed_summaries.values()
        ),
        "overall_median_required_airtime_correlation_at_least_0_30": overall_rho >= 0.30,
        "bootstrap_95pct_lower_above_zero": _bootstrap_lower(all_correlations) > 0.0,
        "each_seed_counterfactual_at_least_0_30": all(
            item["counterfactual_delta_spearman"] >= 0.30
            for item in seed_summaries.values()
        ),
        "overall_equal_delay_improvement_at_least_0_03": float(
            np.mean(station_seed_improvements)
        ) >= 0.03,
        "no_station_seed_degrades_more_than_0_01": min(station_seed_improvements) >= -0.01,
        "overall_oracle_gap_closure_at_least_0_20": overall_closure >= 0.20,
    }
    return {
        "seeds": seed_summaries,
        "overall_median_required_airtime_bandwidth_spearman": overall_rho,
        "bootstrap_95pct_lower": _bootstrap_lower(all_correlations),
        "overall_oracle_gap_closure": overall_closure,
        "gates": gates,
        "passed": all(gates.values()),
    }


def run_channel_scenario(
    probe_report_path,
    data_dir="../Data",
    log_dir="logs",
    device=None,
    total_epochs=150,
    warmup_epochs=5,
    batch_size=16,
    local_steps=4,
    evaluation_batches=10,
    resume_checkpoint=None,
    start_cycle=1,
):
    probe_report = _load_probe_report(probe_report_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(log_dir) / f"mat_channel_scenario_{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    if not probe_report["passed"]:
        report = {
            "schema_version": 2,
            "status": "blocked_by_isolated_probe",
            "probe_report_path": str(probe_report_path),
            "reason": "Scenario A training is not permitted until all isolated channel gates pass.",
        }
        (root / "scenario_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return root, report

    execution_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260731)
    if resume_checkpoint:
        agent = MATAgent.load_checkpoint(resume_checkpoint, device=execution_device)
        if not 1 <= int(start_cycle) <= len(CYCLE_SEEDS):
            raise ValueError("start_cycle must select one of the configured Scenario A cycles")
        if agent.policy_version != int(start_cycle) - 1:
            raise ValueError("resume checkpoint policy_version does not match start_cycle")
    else:
        if int(start_cycle) != 1:
            raise ValueError("start_cycle greater than one requires resume_checkpoint")
        agent = MATAgent(
            state_dim=102, hidden_dim=128, num_migs=7, num_cut_layers=7,
            ppo_epochs=10, minibatch_size=256, channel_conditioning="explicit",
            bandwidth_policy="joint_dirichlet", component_balanced_ppo=True,
            device=execution_device,
        )
    cycles = []
    consecutive_probe_passes = 0
    for cycle_index in range(int(start_cycle), len(CYCLE_SEEDS) + 1):
        seeds = CYCLE_SEEDS[cycle_index - 1]
        buffer = MATTrajectoryBuffer()
        for episode_index, seed in enumerate(seeds):
            episode_started = time.time()
            print(f"scenario cycle={cycle_index} episode={episode_index + 1}/{len(seeds)} seed={seed} start", flush=True)
            run_scenario_a(
                algorithm="mat", data_dir=data_dir, log_dir=str(root), total_epochs=total_epochs,
                seed=seed, batch_size=batch_size, local_steps=local_steps,
                warmup_epochs=warmup_epochs, evaluation_batches=evaluation_batches,
                device=execution_device, create_plots=False, export_results=False,
                mat_agent=agent, trajectory_buffer=buffer, mat_training_mode="collect",
                episode_id=(cycle_index - 1) * len(seeds) + episode_index,
            )
            print(
                f"scenario cycle={cycle_index} episode={episode_index + 1}/{len(seeds)} seed={seed} "
                f"complete elapsed_minutes={(time.time() - episode_started) / 60.0:.1f}", flush=True)
        kwargs = buffer.as_ppo_kwargs()
        transition_count = len(buffer)
        update_started = time.time()
        diagnostics = agent.update_policy(
            kwargs.pop("rewards"), kwargs.pop("next_states"), kwargs.pop("dones"), **kwargs
        )
        diagnostics["ppo_update_elapsed_seconds"] = float(time.time() - update_started)
        print(
            f"scenario cycle={cycle_index} PPO update complete "
            f"elapsed_minutes={diagnostics['ppo_update_elapsed_seconds'] / 60.0:.1f}", flush=True)
        checkpoint = root / f"candidate_cycle_{cycle_index}.pt"
        agent.save_checkpoint(checkpoint)
        probe_summary, _ = evaluate_probe(agent, (449, 457, 461))
        gates, regret_reduction = probe_gates(
            probe_summary, probe_report["legacy"], diagnostics
        )
        cycle_passed = all(gates.values())
        consecutive_probe_passes = consecutive_probe_passes + 1 if cycle_passed else 0
        cycle_report = {
            "cycle": cycle_index,
            "seeds": seeds,
            "transition_count": transition_count,
            "policy_version": agent.policy_version,
            "checkpoint": str(checkpoint),
            "update": diagnostics,
            "probe": probe_summary,
            "legacy_regret_reduction": regret_reduction,
            "gates": gates,
            "passed": cycle_passed,
        }
        cycles.append(cycle_report)
        (root / f"cycle_{cycle_index}_report.json").write_text(
            json.dumps(cycle_report, indent=2, allow_nan=False), encoding="utf-8"
        )
        if consecutive_probe_passes >= 2:
            break

    validation_results = {}
    for seed in VALIDATION_SEEDS:
        sidecar = root / f"validation_seed_{seed}_clients.json"
        validation_results[seed] = run_scenario_a(
            algorithm="mat", data_dir=data_dir, log_dir=str(root), total_epochs=total_epochs,
            seed=seed, batch_size=batch_size, local_steps=local_steps,
            warmup_epochs=warmup_epochs, evaluation_batches=evaluation_batches,
            device=execution_device, create_plots=False, export_results=True,
            run_name=f"mat_channel_validation_seed_{seed}", mat_agent=agent,
            trajectory_buffer=MATTrajectoryBuffer(), mat_training_mode="collect",
            episode_id=1000 + seed, mat_deterministic=True,
            client_diagnostics_path=sidecar,
        )
    validation = _frozen_summary(validation_results)
    numerical_gate = bool(cycles) and all(
        np.isfinite([
            value for value in cycle["update"].values()
            if isinstance(value, (int, float, np.number))
        ]).all()
        and cycle["update"]["grad_norm_post_max"] <= 0.500001
        and cycle["update"]["target_drift_during_update"] == 0.0
        for cycle in cycles
    )
    report = {
        "schema_version": 2,
        "status": "completed",
        "probe_report_path": str(probe_report_path),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "start_cycle": int(start_cycle),
        "cycles": cycles,
        "validation": validation,
        "numerical_gate": numerical_gate,
        "passed": validation["passed"] and numerical_gate,
        "interpretation_boundary": {
            "channel_awareness": "hard gate",
            "payload_visibility": "recorded separately; action order unchanged",
            "raw_congestion_latency": "descriptive only",
        },
    }
    (root / "scenario_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return root, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-report", required=True)
    parser.add_argument("--data-dir", default="../Data")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-steps", type=int, default=4)
    parser.add_argument("--evaluation-batches", type=int, default=10)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--start-cycle", type=int, default=1)
    args = parser.parse_args()
    root, report = run_channel_scenario(
        args.probe_report, args.data_dir, args.log_dir, args.device,
        args.epochs, 5, args.batch_size, args.local_steps, args.evaluation_batches,
        args.resume_checkpoint, args.start_cycle,
    )
    print(json.dumps({"output_dir": str(root), **report}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


