"""Twenty-round, three-station synthetic MAT stability validation."""
import json

import numpy as np
import torch

from envs.liquid_airan_env import LiquidAIRANEnv
from models.mat_agent import MATAgent
from utils.mat_reward import MATRewardConfig, compute_mat_reward
from utils.trajectory_buffer import MATTrajectoryBuffer


class SyntheticCIFAR100Provider:
    num_classes = 100

    def __init__(self, num_clients, seed):
        self.num_clients = num_clients
        rng = np.random.default_rng(seed)
        self.distributions = rng.dirichlet(np.full(self.num_classes, 0.1), size=num_clients)

    def get_client_label_dist(self, client_id):
        return self.distributions[client_id]


def build_state(rng, provider, client_count):
    channels = rng.rayleigh(scale=1.0, size=(client_count, 1)).astype(np.float32)
    computes = rng.uniform(1.0, 2.5, size=(client_count, 1)).astype(np.float32)
    return np.concatenate((channels, computes, provider.distributions[:client_count].astype(np.float32)), axis=1)


def run_tide_validation(rounds=20, client_count=10, seed=7):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    provider = SyntheticCIFAR100Provider(client_count, seed)
    envs = {station: LiquidAIRANEnv(provider, max_vehicles=client_count, max_migs=7, station_id=station,
                                    seed=seed + station) for station in (1, 2, 3)}
    agent = MATAgent(state_dim=102, hidden_dim=32, ppo_epochs=3, minibatch_size=16)
    reward_config = MATRewardConfig()
    buffer = MATTrajectoryBuffer()
    target_before = [p.detach().clone() for p in list(agent.target_encoder.parameters()) + list(agent.target_value_head.parameters())]
    reward_by_station = {station: [] for station in envs}
    action_constraints_hold = True
    for round_index in range(1, rounds + 1):
        available_migs = 2 if round_index <= 6 or round_index > 14 else 5
        bandwidth_scale = 0.2 if round_index > 14 else 1.0
        for station, env in envs.items():
            env.current_migs = available_migs
            env.current_bandwidth = env.base_bandwidth * bandwidth_scale
            state = build_state(rng, provider, client_count)
            edge = np.asarray([available_migs, env.current_bandwidth], dtype=np.float32)
            action, info = agent.act(state, available_migs, edge)
            action_constraints_hold &= bool(np.all((action["cluster"] >= 0) & (action["cluster"] < available_migs)))
            action_constraints_hold &= bool(np.all(action["l1"] < action["l2"]) and action["bw"].sum() <= 1.000001)
            delays = env.calc_wireless_transmission_delay(action["cluster"], action["bw"],
                                                          np.full(client_count, 4096.0), state[:, 0])
            reward, _ = compute_mat_reward(float(delays.max()), state[:, 2:], action["cluster"], action["bw"], reward_config)
            if station == 2 and round_index > 14:
                reward *= 1000.0
            next_state = build_state(rng, provider, client_count)
            done = round_index == rounds
            buffer.append(state, edge, action, reward, next_state, edge, done, info, available_migs,
                          station, round_index, trajectory_id=(0, station), policy_version=agent.policy_version)
            reward_by_station[station].append(reward)
    target_rollout_drift = max(float((left - right).abs().max()) for left, right in zip(
        target_before, list(agent.target_encoder.parameters()) + list(agent.target_value_head.parameters())))
    kwargs = buffer.as_ppo_kwargs()
    diagnostics = agent.update_policy(kwargs.pop("rewards"), kwargs.pop("next_states"), kwargs.pop("dones"), **kwargs)
    numeric = [value for value in diagnostics.values() if isinstance(value, (int, float, np.number))]
    target_synced = all(torch.equal(online, target) for online, target in zip(
        list(agent.encoder.parameters()) + list(agent.value_head.parameters()),
        list(agent.target_encoder.parameters()) + list(agent.target_value_head.parameters())))
    report = {
        "rounds": rounds, "stations": 3, "transitions": len(buffer),
        "action_constraints_hold": action_constraints_hold,
        "all_diagnostics_finite": bool(np.isfinite(numeric).all()),
        "target_rollout_drift": target_rollout_drift,
        "target_synced_after_update": target_synced,
        "post_clip_gradient_within_limit": diagnostics["grad_norm_post_max"] <= 0.500001,
        "station_reward_means": {str(key): float(np.mean(value)) for key, value in reward_by_station.items()},
        "diagnostics": diagnostics,
    }
    assert report["action_constraints_hold"]
    assert report["all_diagnostics_finite"]
    assert report["target_rollout_drift"] == 0.0
    assert report["target_synced_after_update"]
    assert report["post_clip_gradient_within_limit"]
    return report


if __name__ == "__main__":
    print(json.dumps(run_tide_validation(), indent=2))