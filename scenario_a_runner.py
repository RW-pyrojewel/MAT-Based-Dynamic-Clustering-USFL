"""MAT-only runner for research-plan scenario A.

Future baselines can be passed through agents. They must expose either an act
method with state, available_migs and edge_state, or a legacy step method.
Only MATAgent instances are updated.
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from envs import LiquidAIRANEnv
from models.mat_agent import MATAgent
from utils.mat_reward import MATRewardConfig, compute_mat_reward
from utils.trajectory_buffer import MATTrajectoryBuffer


class SyntheticCIFAR100Provider:
    """Small deterministic provider for structural runner validation."""

    num_classes = 100

    def __init__(self, num_clients=30, alpha=0.1, seed=7):
        rng = np.random.default_rng(seed)
        self.distributions = rng.dirichlet(np.full(self.num_classes, alpha), size=num_clients)

    def get_client_label_dist(self, client_id):
        return self.distributions[client_id]


def _select_action(agent, state, available_migs, edge_state, deterministic):
    if hasattr(agent, "act"):
        outcome = agent.act(state, available_migs, edge_state, deterministic=deterministic)
        if isinstance(outcome, tuple) and len(outcome) == 2 and isinstance(outcome[0], dict):
            return outcome
    clusters, l1, l2, bandwidths = agent.step(state, available_migs)
    return {
        "cluster": np.asarray(clusters),
        "l1": np.asarray(l1),
        "l2": np.asarray(l2),
        "bw": np.asarray(bandwidths),
    }, None


def _estimate_round_delay(env, state, action):
    """Use a deterministic lightweight USFL timing proxy until profiling is wired in."""
    clusters = action["cluster"]
    l1 = action["l1"]
    l2 = action["l2"]
    smashed_sizes = 16384.0 / np.power(2.0, np.clip(l1, 0, 6))
    tx_delays = env.calc_wireless_transmission_delay(clusters, action["bw"], smashed_sizes, state[:, 0])
    cluster_delays = np.zeros(env.current_migs, dtype=np.float64)
    for mig_id in range(env.current_migs):
        members = clusters == mig_id
        if not members.any():
            continue
        depth = float(np.mean(l2[members] - l1[members] + 1))
        effective_compute = float(np.mean(state[members, 1]))
        compute_delay = 0.02 + 0.004 * members.sum() * depth / max(effective_compute, 1e-6)
        cluster_delays[mig_id] = tx_delays[mig_id] + compute_delay
    return float(cluster_delays.max()), float(tx_delays.max())


def _flush_mat_buffer(agent, buffer):
    if not isinstance(agent, MATAgent) or not len(buffer):
        return
    kwargs = buffer.as_ppo_kwargs()
    agent.update_policy(kwargs.pop("rewards"), kwargs.pop("next_states"), kwargs.pop("dones"), **kwargs)
    buffer.clear()


def run_scenario_a(agents, data_provider, total_epochs=150, seed=7, update_interval=16, deterministic=False):
    """Run scenario A for supplied policies and update MAT policies online."""
    if total_epochs < 1 or total_epochs > 150:
        raise ValueError("total_epochs must be in [1, 150]")
    if not agents:
        raise ValueError("at least one agent is required")

    environments = {
        station_id: LiquidAIRANEnv(
            data_provider,
            max_vehicles=10,
            max_migs=7,
            scenario="A",
            station_id=station_id,
            seed=seed + station_id,
        )
        for station_id in (1, 2, 3)
    }
    reward_config = MATRewardConfig(cluster_size_limit=10)
    buffers = {name: MATTrajectoryBuffer() for name in agents}
    pending = {name: {} for name in agents}
    reports = []

    for epoch in range(1, total_epochs + 1):
        observations = {station_id: env.step() for station_id, env in environments.items()}
        for name, agent in agents.items():
            for station_id, (state, available_migs, bandwidth, _) in observations.items():
                edge_state = np.asarray([available_migs, bandwidth], dtype=np.float32)
                prior = pending[name].pop(station_id, None)
                if prior is not None and prior["log_prob"] is not None:
                    buffers[name].append(
                        prior["state"], prior["edge_state"], prior["action"], prior["reward"],
                        state, edge_state, False, prior["log_prob"], prior["available_migs"],
                    )

                action, log_prob = _select_action(agent, state, available_migs, edge_state, deterministic)
                if (action["cluster"] < 0).any() or (action["cluster"] >= available_migs).any() or not np.all(action["l1"] < action["l2"]):
                    raise ValueError(f"{name} produced an invalid scenario-A action")
                total_delay, tx_delay = _estimate_round_delay(environments[station_id], state, action)
                reward, reward_terms = compute_mat_reward(
                    total_delay, state[:, 2:], action["cluster"], action["bw"], reward_config
                )
                pending[name][station_id] = {
                    "state": state, "edge_state": edge_state, "action": action, "reward": reward,
                    "log_prob": log_prob, "available_migs": available_migs,
                }
                reports.append(
                    {
                        "epoch": epoch, "agent": name, "station_id": station_id,
                        "available_migs": available_migs, "bandwidth": bandwidth,
                        "vehicle_count": len(state), "reward": reward, "total_delay": total_delay,
                        "tx_delay": tx_delay, **reward_terms,
                    }
                )
            if len(buffers[name]) >= update_interval:
                _flush_mat_buffer(agent, buffers[name])

    for name, agent in agents.items():
        for terminal in pending[name].values():
            if terminal["log_prob"] is not None:
                buffers[name].append(
                    terminal["state"], terminal["edge_state"], terminal["action"], terminal["reward"],
                    terminal["state"], terminal["edge_state"], True, terminal["log_prob"],
                    terminal["available_migs"],
                )
        _flush_mat_buffer(agent, buffers[name])
    return reports


def run_mat_scenario_a(total_epochs=150, seed=7, update_interval=16, deterministic=False):
    """Construct and run the currently available MAT policy in scenario A."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    provider = SyntheticCIFAR100Provider(seed=seed)
    agent = MATAgent(state_dim=2 + provider.num_classes, hidden_dim=32, num_migs=7, num_cut_layers=7)
    return run_scenario_a(
        {"MAT-RL": agent}, provider, total_epochs=total_epochs, seed=seed,
        update_interval=update_interval, deterministic=deterministic,
    )


def main():
    parser = argparse.ArgumentParser(description="Run MAT in scenario A with synthetic CIFAR-100 labels.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--update-interval", type=int, default=16)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    reports = run_mat_scenario_a(args.epochs, args.seed, args.update_interval, args.deterministic)
    by_epoch = defaultdict(list)
    for report in reports:
        by_epoch[report["epoch"]].append(report)
    summary = [
        {
            "epoch": epoch,
            "mean_reward": float(np.mean([item["reward"] for item in values])),
            "station_resources": {
                str(item["station_id"]): {
                    "migs": item["available_migs"], "bandwidth": item["bandwidth"],
                }
                for item in values
            },
        }
        for epoch, values in by_epoch.items()
    ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
