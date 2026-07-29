"""Minimal MAT online-interaction smoke test for the 2->5->2 MIG tide."""
import json

import numpy as np
import torch

from envs.liquid_airan_env import LiquidAIRANEnv
from models.mat_agent import MATAgent
from utils.mat_reward import MATRewardConfig, compute_mat_reward
from utils.trajectory_buffer import MATTrajectoryBuffer


class SyntheticCIFAR100Provider:
    """Deterministic label-distribution provider; it never downloads a dataset."""
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
    labels = provider.distributions[:client_count].astype(np.float32)
    return np.concatenate((channels, computes, labels), axis=1)


def run_tide_validation(rounds_per_phase=4, client_count=10, seed=7):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    provider = SyntheticCIFAR100Provider(client_count, seed)
    env = LiquidAIRANEnv(provider, max_vehicles=client_count, max_migs=7)
    agent = MATAgent(state_dim=2 + provider.num_classes, hidden_dim=32, num_migs=env.max_migs, num_cut_layers=7)
    reward_config = MATRewardConfig()
    buffer = MATTrajectoryBuffer()
    reports = []
    phases = (("baseline", 2, 1.0), ("compute_windfall", 5, 1.0), ("recovery_congestion", 2, 0.2))

    for phase_index, (phase_name, available_migs, bandwidth_scale) in enumerate(phases):
        env.current_migs = available_migs
        env.current_bandwidth = env.base_bandwidth * bandwidth_scale
        phase_rewards = []
        for round_index in range(rounds_per_phase):
            state = build_state(rng, provider, client_count)
            edge_state = np.asarray([available_migs, env.current_bandwidth], dtype=np.float32)
            action, policy_info = agent.act(state, available_migs, edge_state, deterministic=False)
            assert np.all((action["cluster"] >= 0) & (action["cluster"] < available_migs))
            assert np.all(action["l1"] < action["l2"])
            tx_delays = env.calc_wireless_transmission_delay(action["cluster"], action["bw"], np.full(client_count, 4096.0), state[:, 0])
            reward, _ = compute_mat_reward(float(tx_delays.max()), state[:, 2:], action["cluster"], action["bw"], reward_config)
            next_state = build_state(rng, provider, client_count)
            next_edge_state = np.asarray([available_migs, env.current_bandwidth], dtype=np.float32)
            done = phase_index == len(phases) - 1 and round_index == rounds_per_phase - 1
            epoch = phase_index * rounds_per_phase + round_index + 1
            buffer.append(
                state, edge_state, action, reward, next_state, next_edge_state, done,
                policy_info, available_migs, station_id=1, epoch=epoch,
            )
            phase_rewards.append(reward)
        kwargs = buffer.as_ppo_kwargs()
        agent.update_policy(kwargs.pop("rewards"), kwargs.pop("next_states"), kwargs.pop("dones"), **kwargs)
        reports.append({"phase": phase_name, "available_migs": available_migs, "mean_reward": float(np.mean(phase_rewards)), "transitions": len(buffer)})
        buffer.clear()
    return reports


if __name__ == "__main__":
    print(json.dumps(run_tide_validation(), indent=2))
