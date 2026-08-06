"""Multi-cycle synthetic regression gate for MAT critic stability."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from envs.liquid_airan_env import LiquidAIRANEnv
from mat_tide_runner import SyntheticCIFAR100Provider, build_state
from models.mat_agent import MATAgent
from utils.mat_reward import MATRewardConfig, compute_mat_reward
from utils.trajectory_buffer import MATTrajectoryBuffer


def _target_parameters(agent):
    return list(agent.target_encoder.parameters()) + list(agent.target_value_head.parameters())


def _online_critic_parameters(agent):
    return list(agent.encoder.parameters()) + list(agent.value_head.parameters())


def _max_drift(before, after):
    return max((float((left - right).abs().max()) for left, right in zip(before, after)), default=0.0)


def _resources(round_index, rounds):
    first = max(rounds // 3, 1)
    second = max(2 * rounds // 3, first + 1)
    return (5 if first < round_index <= second else 2), (0.2 if round_index > second else 1.0)


def _collect(agent, provider, envs, rng, rounds, episode_id, station_2_scale):
    buffer = MATTrajectoryBuffer()
    config = MATRewardConfig()
    constraints_hold = True
    rewards = {station: [] for station in envs}
    states = {station: build_state(rng, provider, provider.num_clients) for station in envs}
    for round_index in range(1, rounds + 1):
        mig_count, bandwidth_scale = _resources(round_index, rounds)
        next_migs, next_scale = _resources(min(round_index + 1, rounds), rounds)
        for station, env in envs.items():
            env.current_migs = mig_count
            env.current_bandwidth = env.base_bandwidth * bandwidth_scale
            state = states[station]
            edge = np.asarray([mig_count, env.current_bandwidth], dtype=np.float32)
            action, policy_info = agent.act(state, mig_count, edge)
            constraints_hold &= bool(
                np.all((action["cluster"] >= 0) & (action["cluster"] < mig_count))
                and np.all(action["l1"] < action["l2"])
                and action["bw"].sum() <= 1.000001
                and action["bw"].min() >= agent.min_bandwidth_share - 1e-6
            )
            delays = env.calc_wireless_transmission_delay(
                action["cluster"], action["bw"], np.full(provider.num_clients, 4096.0), state[:, 0]
            )
            reward, _ = compute_mat_reward(
                float(delays.max()), state[:, 2:], action["cluster"], action["bw"], config
            )
            if station == 2 and round_index > 2 * rounds // 3:
                reward *= station_2_scale
            next_state = build_state(rng, provider, provider.num_clients)
            next_edge = np.asarray([next_migs, env.base_bandwidth * next_scale], dtype=np.float32)
            buffer.append(
                state, edge, action, reward, next_state, next_edge, round_index == rounds,
                policy_info, mig_count, station, round_index,
                trajectory_id=(episode_id, station), policy_version=agent.policy_version,
            )
            rewards[station].append(reward)
            states[station] = next_state
    return buffer, constraints_hold, {str(key): float(np.mean(value)) for key, value in rewards.items()}


def _critic_targets(agent, kwargs):
    return agent._compute_gae(
        kwargs["rewards"], kwargs["next_states"], kwargs["dones"], kwargs["edge_states"],
        kwargs["next_edge_states"], kwargs["policy_infos"], kwargs["station_ids"],
        kwargs["epochs"], kwargs["trajectory_ids"],
    )[2]


def _finite(mapping):
    values = [value for value in mapping.values() if isinstance(value, (int, float, np.number))]
    return bool(np.isfinite(values).all())


def run_critic_regression(
    cycles=5, train_rounds=20, holdout_rounds=12, client_count=10, seed=20260731,
    hidden_dim=32, ppo_epochs=3, minibatch_size=16, station_2_scale=1000.0,
    device=None, log_dir="logs", write_report=True,
):
    """Run repeated collect/update/sync cycles and return a machine-readable report."""
    if cycles < 3:
        raise ValueError("critic regression protection requires at least three cycles")
    execution_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    provider = SyntheticCIFAR100Provider(client_count, seed)
    envs = {
        station: LiquidAIRANEnv(provider, max_vehicles=client_count, max_migs=7,
                                station_id=station, seed=seed + station)
        for station in (1, 2, 3)
    }
    agent = MATAgent(
        state_dim=2 + provider.num_classes, hidden_dim=hidden_dim, ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size, max_grad_norm=0.5, target_kl=0.03,
        bandwidth_policy="joint_dirichlet", device=execution_device,
    )
    train_rng = np.random.default_rng(seed + 1000)
    holdout_rng = np.random.default_rng(seed + 2000)
    reports = []
    for cycle in range(cycles):
        version = agent.policy_version
        target_before = [parameter.detach().clone() for parameter in _target_parameters(agent)]
        train_buffer, train_actions, train_rewards = _collect(
            agent, provider, envs, train_rng, train_rounds, cycle * 2, station_2_scale)
        holdout_buffer, holdout_actions, holdout_rewards = _collect(
            agent, provider, envs, holdout_rng, holdout_rounds, cycle * 2 + 1, station_2_scale)
        rollout_drift = _max_drift(target_before, _target_parameters(agent))
        train = train_buffer.as_ppo_kwargs()
        holdout = holdout_buffer.as_ppo_kwargs()
        targets = _critic_targets(agent, holdout)
        before = agent.evaluate_critic_targets(holdout["states"], holdout["edge_states"], targets, holdout["station_ids"])
        diagnostics = agent.update_policy(
            train.pop("rewards"), train.pop("next_states"), train.pop("dones"), **train)
        after = agent.evaluate_critic_targets(holdout["states"], holdout["edge_states"], targets, holdout["station_ids"])
        before_loss = before["normalized_td_huber_loss"]
        after_loss = after["normalized_td_huber_loss"]
        relative_change = (after_loss - before_loss) / max(before_loss, 1e-12)
        evs = [after[f"station_{station}_explained_variance"] for station in (1, 2, 3)]
        synced = all(torch.equal(online, target) for online, target in zip(
            _online_critic_parameters(agent), _target_parameters(agent)))
        gates = {
            "action_constraints_hold": train_actions and holdout_actions,
            "single_current_policy_version": (
                set(train["policy_versions"]) == {version}
                and set(holdout["policy_versions"]) == {version}
                and agent.policy_version == version + 1),
            "all_metrics_finite": bool(_finite(diagnostics) and _finite(before) and _finite(after) and np.isfinite(relative_change)),
            "target_frozen_during_rollouts": rollout_drift == 0.0,
            "target_frozen_during_update": diagnostics["target_drift_during_update"] == 0.0,
            "target_synced_after_update": synced,
            "gradient_pre_below_50": diagnostics["grad_norm_pre_max"] < 50.0,
            "gradient_post_at_most_0_5": diagnostics["grad_norm_post_max"] <= 0.500001,
            "kl_within_limit_or_early_stopped": diagnostics["approx_kl"] <= 0.03 or diagnostics["kl_early_stop"],
            "holdout_loss_not_worse": bool(after_loss <= before_loss + 1e-12),
        }
        reports.append({
            "cycle": cycle + 1, "policy_version_before": version,
            "policy_version_after": agent.policy_version, "train_transitions": len(train_buffer),
            "holdout_transitions": len(holdout_buffer), "rollout_target_drift": rollout_drift,
            "train_reward_means": train_rewards, "holdout_reward_means": holdout_rewards,
            "holdout_before": before, "holdout_after": after,
            "holdout_loss_relative_change": float(relative_change),
            "holdout_mean_ev_after": float(np.mean(evs)), "diagnostics": diagnostics,
            "gates": gates, "passed": all(gates.values()),
        })
    before_losses = [item["holdout_before"]["normalized_td_huber_loss"] for item in reports]
    after_losses = [item["holdout_after"]["normalized_td_huber_loss"] for item in reports]
    final_evs = [reports[-1]["holdout_after"][f"station_{station}_explained_variance"] for station in (1, 2, 3)]
    mean_evs = [item["holdout_mean_ev_after"] for item in reports]
    aggregate_gates = {
        "all_cycles_pass": all(item["passed"] for item in reports),
        "completed_requested_cycles": len(reports) == cycles,
        "policy_versions_advanced_once_per_cycle": agent.policy_version == cycles,
        "median_holdout_loss_not_worse": bool(float(np.median(after_losses)) <= float(np.median(before_losses))),
        "every_cycle_holdout_loss_not_worse": all(after <= before + 1e-12 for before, after in zip(before_losses, after_losses)),
        "final_mean_ev_improved_from_first_cycle": bool(cycles < 5 or mean_evs[-1] >= mean_evs[0]),
        "last_two_mean_ev_above_minus_0_10": bool(min(mean_evs[-2:]) >= -0.10),
        "final_holdout_mean_ev_above_minus_0_05": bool(float(np.mean(final_evs)) >= -0.05),
        "final_each_station_ev_above_minus_0_10": bool(min(final_evs) >= -0.10),
    }
    report = {
        "configuration": {"cycles": cycles, "train_rounds": train_rounds,
                          "holdout_rounds": holdout_rounds, "stations": 3,
                          "client_count": client_count, "station_2_scale": station_2_scale,
                          "seed": seed, "device": execution_device},
        "cycles": reports,
        "aggregate": {
            "median_holdout_loss_before": float(np.median(before_losses)),
            "median_holdout_loss_after": float(np.median(after_losses)),
            "final_station_evs": {str(key): float(value) for key, value in zip((1, 2, 3), final_evs)},
            "gates": aggregate_gates,
        },
        "passed": all(aggregate_gates.values()),
    }
    if write_report:
        output_dir = Path(log_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"mat_critic_regression_{datetime.now():%Y%m%d_%H%M%S}.json"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()
    print(json.dumps(run_critic_regression(cycles=args.cycles, device=args.device, log_dir=args.log_dir), indent=2))


if __name__ == "__main__":
    main()