"""One-shot 8+1 Scenario-A gate for the shared-encoder MAT critic."""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch

from models.mat_agent import MATAgent
from scenario_a_runner import run_scenario_a
from utils.trajectory_buffer import MATTrajectoryBuffer

TRAIN_SEEDS = (7, 17, 29, 41, 53, 67, 79, 97)
HOLDOUT_SEED = 113


def _critic_snapshot(agent, kwargs, td_targets):
    return agent.evaluate_critic_targets(kwargs["states"], kwargs["edge_states"], td_targets, kwargs["station_ids"])


def _latency_summary(results, warmup_epochs):
    summary = {}
    for seed, result in results.items():
        rows = [row for row in result["logger"].records["MAT-RL"] if row["epoch"] > warmup_epochs]
        summary[str(seed)] = {
            f"station_{station}_delay_mean_ms": float(np.mean([row["total_delay_ms"] for row in rows if row["station_id"] == station]))
            for station in (1, 2, 3)
        }
    return summary


def run_large_batch_gate(data_dir="../Data", log_dir="logs", device=None, total_epochs=150,
                         warmup_epochs=5, batch_size=16, local_steps=4, evaluation_batches=10):
    execution_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260730)
    agent = MATAgent(state_dim=102, hidden_dim=128, num_migs=7, num_cut_layers=7,
                     ppo_epochs=10, minibatch_size=256, actor_learning_rate=1e-4,
                     critic_learning_rate=1e-4, max_grad_norm=0.5, huber_delta=1.0,
                     target_kl=0.03, device=execution_device)
    train_buffer = MATTrajectoryBuffer()
    episode_results = {}
    for episode_id, seed in enumerate(TRAIN_SEEDS):
        result = run_scenario_a(
            algorithm="mat", data_dir=data_dir, log_dir=log_dir, total_epochs=total_epochs, seed=seed,
            batch_size=batch_size, local_steps=local_steps, warmup_epochs=warmup_epochs,
            evaluation_batches=evaluation_batches, device=execution_device, create_plots=False,
            mat_agent=agent, trajectory_buffer=train_buffer, mat_training_mode="collect",
            episode_id=episode_id, export_results=False,
        )
        episode_results[seed] = result
        print(f"collected seed={seed}: transitions={len(train_buffer)}, policy_version={agent.policy_version}", flush=True)
    holdout_buffer = MATTrajectoryBuffer()
    holdout_result = run_scenario_a(
        algorithm="mat", data_dir=data_dir, log_dir=log_dir, total_epochs=total_epochs, seed=HOLDOUT_SEED,
        batch_size=batch_size, local_steps=local_steps, warmup_epochs=warmup_epochs,
        evaluation_batches=evaluation_batches, device=execution_device, create_plots=False,
        mat_agent=agent, trajectory_buffer=holdout_buffer, mat_training_mode="collect",
        episode_id=len(TRAIN_SEEDS), export_results=False,
    )
    episode_results[HOLDOUT_SEED] = holdout_result
    train_kwargs = train_buffer.as_ppo_kwargs()
    holdout_kwargs = holdout_buffer.as_ppo_kwargs()
    _, _, holdout_targets, _ = agent._compute_gae(
        holdout_kwargs["rewards"], holdout_kwargs["next_states"], holdout_kwargs["dones"],
        holdout_kwargs["edge_states"], holdout_kwargs["next_edge_states"], holdout_kwargs["policy_infos"],
        holdout_kwargs["station_ids"], holdout_kwargs["epochs"], holdout_kwargs["trajectory_ids"])
    holdout_before = _critic_snapshot(agent, holdout_kwargs, holdout_targets)
    rewards, next_states, dones = train_kwargs.pop("rewards"), train_kwargs.pop("next_states"), train_kwargs.pop("dones")
    diagnostics = agent.update_policy(rewards, next_states, dones, **train_kwargs)
    holdout_after = _critic_snapshot(agent, holdout_kwargs, holdout_targets)
    before_loss, after_loss = holdout_before["normalized_td_huber_loss"], holdout_after["normalized_td_huber_loss"]
    improvement = (before_loss - after_loss) / max(before_loss, 1e-12)
    holdout_evs = [holdout_after[f"station_{station}_explained_variance"] for station in (1, 2, 3)]
    finite_values = [value for section in (diagnostics, holdout_before, holdout_after)
                     for value in section.values() if isinstance(value, (int, float, np.number))]
    gates = {
        "at_least_3200_transitions": len(train_buffer) >= 3200,
        "all_diagnostics_finite": bool(np.isfinite(finite_values).all()),
        "grad_norm_pre_max_below_50": diagnostics["grad_norm_pre_max"] < 50.0,
        "grad_norm_post_max_at_most_0_5": diagnostics["grad_norm_post_max"] <= 0.500001,
        "kl_within_limit_or_early_stopped": diagnostics["approx_kl"] <= 0.03 or diagnostics["kl_early_stop"],
        "holdout_td_huber_improved_10_percent": improvement >= 0.10,
        "holdout_mean_ev_nonnegative": float(np.mean(holdout_evs)) >= 0.0,
        "holdout_each_ev_above_minus_0_05": min(holdout_evs) >= -0.05,
    }
    report = {
        "configuration": {"train_seeds": TRAIN_SEEDS, "holdout_seed": HOLDOUT_SEED,
                          "total_epochs": total_epochs, "warmup_epochs": warmup_epochs,
                          "device": execution_device, "training_semantics": "single-version on-policy full-batch-equivalent"},
        "counts": {"train_transitions": len(train_buffer), "holdout_transitions": len(holdout_buffer),
                   "train_trajectories": len(set(train_kwargs["trajectory_ids"])),
                   "policy_versions": len(set(train_kwargs["policy_versions"]))},
        "update": diagnostics, "holdout_before": holdout_before, "holdout_after": holdout_after,
        "holdout_normalized_td_huber_improvement": float(improvement), "latency_descriptive_only": _latency_summary(episode_results, warmup_epochs),
        "gates": gates, "passed": all(gates.values()),
    }
    os.makedirs(log_dir, exist_ok=True)
    output = os.path.join(log_dir, f"mat_shared_encoder_8plus1_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(output, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    report["report_path"] = output
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../Data")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-steps", type=int, default=4)
    parser.add_argument("--evaluation-batches", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(run_large_batch_gate(data_dir=args.data_dir, log_dir=args.log_dir, device=args.device,
                                           batch_size=args.batch_size, local_steps=args.local_steps,
                                           evaluation_batches=args.evaluation_batches), indent=2))


if __name__ == "__main__":
    main()