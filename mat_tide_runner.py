"""Twenty-round, three-station MAT stability validation without dataset downloads."""
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


def _station_resources(station_id, round_index, shock_round):
    after_shock = round_index >= shock_round
    if station_id == 1:
        return (5 if after_shock else 2), 1.0
    if station_id == 2:
        return 2, (0.2 if after_shock else 1.0)
    return 2, 1.0


def run_tide_validation(total_rounds=20, client_count=10, seed=7):
    """Stress station-local critic scaling across Scenario-A-like resource changes."""
    if total_rounds < 4:
        raise ValueError("total_rounds must be at least four")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    provider = SyntheticCIFAR100Provider(client_count, seed)
    environments = {
        station_id: LiquidAIRANEnv(provider, max_vehicles=client_count, max_migs=7, seed=seed + station_id)
        for station_id in (1, 2, 3)
    }
    agent = MATAgent(
        state_dim=2 + provider.num_classes,
        hidden_dim=32,
        num_migs=7,
        num_cut_layers=7,
        ppo_epochs=1,
        minibatch_size=12,
    )
    reward_config = MATRewardConfig()
    buffer = MATTrajectoryBuffer()
    shock_round = total_rounds // 2
    station_rewards = {station_id: [] for station_id in environments}

    for round_index in range(total_rounds):
        for station_id, environment in environments.items():
            available_migs, bandwidth_scale = _station_resources(station_id, round_index, shock_round)
            bandwidth_hz = environment.base_bandwidth * bandwidth_scale
            state = build_state(rng, provider, client_count)
            edge_state = np.asarray([available_migs, bandwidth_hz], dtype=np.float32)
            action, policy_info = agent.act(state, available_migs, edge_state, deterministic=False)
            if not np.all((action["cluster"] >= 0) & (action["cluster"] < available_migs)):
                raise AssertionError("cluster action exceeded available MIGs")
            if not np.all(action["l1"] < action["l2"]):
                raise AssertionError("invalid U-shaped split action")
            if not np.isclose(action["bw"].sum(), 1.0, atol=1e-6):
                raise AssertionError("bandwidth action violated the global budget")
            tx_delays = environment.calc_wireless_transmission_delay(
                action["cluster"], action["bw"], np.full(client_count, 4096.0), state[:, 0],
                available_migs=available_migs, bandwidth_hz=bandwidth_hz,
            )
            reward, _ = compute_mat_reward(
                float(tx_delays.max()), state[:, 2:], action["cluster"], action["bw"], reward_config,
            )
            if station_id == 2 and round_index >= shock_round:
                reward *= 50.0
            next_state = build_state(rng, provider, client_count)
            next_edge_state = np.asarray([available_migs, bandwidth_hz], dtype=np.float32)
            done = round_index == total_rounds - 1
            buffer.append(
                state, edge_state, action, reward, next_state, next_edge_state, done,
                policy_info, available_migs, station_id=station_id, epoch=round_index + 1,
            )
            station_rewards[station_id].append(reward)

    transition_count = len(buffer)
    kwargs = buffer.as_ppo_kwargs()
    diagnostics = agent.update_policy(
        kwargs.pop("rewards"), kwargs.pop("next_states"), kwargs.pop("dones"), **kwargs,
    )
    if not diagnostics or not all(np.isfinite(value) for value in diagnostics.values()):
        raise AssertionError("PPO stability diagnostics must all be finite")
    if diagnostics["grad_norm_post"] > agent.max_grad_norm + 1e-5:
        raise AssertionError("post-clipping gradient norm exceeded the configured limit")
    return {
        "rounds": total_rounds,
        "stations": len(environments),
        "transitions": transition_count,
        "shock_round": shock_round + 1,
        "station_reward_mean": {
            str(station_id): float(np.mean(rewards))
            for station_id, rewards in station_rewards.items()
        },
        "diagnostics": diagnostics,
    }


if __name__ == "__main__":
    print(json.dumps(run_tide_validation(), indent=2))
