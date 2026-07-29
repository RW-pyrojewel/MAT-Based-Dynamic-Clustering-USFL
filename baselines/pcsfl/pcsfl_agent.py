"""Adapted PCSFL baseline using a fixed-width, factorised DDQN controller."""
from collections import deque
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional

from interfaces.base_agent import BaseAgent


class _PCSFLQNetwork(nn.Module):
    def __init__(self, input_dim, max_clients, max_migs, split_count):
        super().__init__()
        self.max_clients = max_clients
        self.max_migs = max_migs
        self.split_count = split_count
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        )
        self.cluster_head = nn.Linear(256, max_clients * max_migs)
        self.split_head = nn.Linear(256, max_clients * split_count)

    def forward(self, state):
        hidden = self.trunk(state)
        clusters = self.cluster_head(hidden).view(-1, self.max_clients, self.max_migs)
        splits = self.split_head(hidden).view(-1, self.max_clients, self.split_count)
        return clusters, splits


class PCSFLAgent(BaseAgent):
    """Fixed-dimensional MLP DDQN with independent parallel client decisions."""

    def __init__(
        self,
        state_dim,
        max_clients=10,
        max_migs=7,
        num_cut_layers=7,
        learning_rate=3e-4,
        gamma=0.95,
        epsilon_start=0.30,
        epsilon_end=0.05,
        epsilon_decay=300,
        replay_size=2048,
        batch_size=32,
        target_update_interval=32,
        device="cpu",
    ):
        super().__init__(agent_name="Adapted-PCSFL")
        self.state_dim = int(state_dim)
        self.max_clients = int(max_clients)
        self.max_migs = int(max_migs)
        self.split_pairs = [(layer, layer + 1) for layer in range(int(num_cut_layers) - 1)]
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = int(epsilon_decay)
        self.batch_size = int(batch_size)
        self.target_update_interval = int(target_update_interval)
        self.device = torch.device(device)
        input_dim = self.max_clients * self.state_dim + 2
        self.online = _PCSFLQNetwork(input_dim, self.max_clients, self.max_migs, len(self.split_pairs)).to(self.device)
        self.target = _PCSFLQNetwork(input_dim, self.max_clients, self.max_migs, len(self.split_pairs)).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay = deque(maxlen=int(replay_size))
        self.action_steps = 0
        self.update_steps = 0

    def _encode(self, state, edge_state):
        state = np.asarray(state, dtype=np.float32)
        if state.ndim != 2 or state.shape[1] != self.state_dim or len(state) > self.max_clients:
            raise ValueError("PCSFL received an incompatible active-client state")
        padded = np.zeros((self.max_clients, self.state_dim), dtype=np.float32)
        padded[:len(state)] = state
        edge = np.asarray(edge_state, dtype=np.float32).reshape(2)
        return np.concatenate([padded.ravel(), edge])

    def _epsilon(self, deterministic):
        if deterministic:
            return 0.0
        progress = min(1.0, self.action_steps / max(self.epsilon_decay, 1))
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    @torch.no_grad()
    def act(self, active_clients_state, available_migs, edge_state, deterministic=False):
        client_count = len(active_clients_state)
        if client_count == 0 or client_count > self.max_clients or not 1 <= available_migs <= self.max_migs:
            raise ValueError("PCSFL received an unsupported client or MIG count")
        encoded = self._encode(active_clients_state, edge_state)
        tensor = torch.as_tensor(encoded, dtype=torch.float32, device=self.device).unsqueeze(0)
        cluster_q, split_q = self.online(tensor)
        clusters = cluster_q[0, :client_count, :available_migs].argmax(dim=1).cpu().numpy()
        split_ids = split_q[0, :client_count].argmax(dim=1).cpu().numpy()
        epsilon = self._epsilon(deterministic)
        if epsilon > 0.0:
            random_cluster = np.random.randint(0, available_migs, size=client_count)
            random_split = np.random.randint(0, len(self.split_pairs), size=client_count)
            clusters = np.where(np.random.random(client_count) < epsilon, random_cluster, clusters)
            split_ids = np.where(np.random.random(client_count) < epsilon, random_split, split_ids)
        self.action_steps += 1
        pairs = np.asarray([self.split_pairs[index] for index in split_ids], dtype=np.int64)
        return {
            "cluster": clusters.astype(np.int64),
            "virtual_cluster": np.arange(client_count, dtype=np.int64),
            "l1": pairs[:, 0],
            "l2": pairs[:, 1],
            "bw": np.full(client_count, 1.0 / client_count, dtype=np.float64),
        }, None

    def step(self, active_clients_state, available_migs):
        action, _ = self.act(active_clients_state, available_migs, np.asarray([available_migs, 1.0], dtype=np.float32))
        return action["cluster"], action["l1"], action["l2"], action["bw"]

    def observe(self, state, edge_state, action, reward, next_state, next_edge_state, done):
        split_lookup = {pair: index for index, pair in enumerate(self.split_pairs)}
        split_ids = np.asarray([split_lookup[(int(first), int(second))] for first, second in zip(action["l1"], action["l2"])])
        clusters = np.zeros(self.max_clients, dtype=np.int64)
        padded_splits = np.zeros(self.max_clients, dtype=np.int64)
        clusters[:len(action["cluster"])] = action["cluster"]
        padded_splits[:len(split_ids)] = split_ids
        self.replay.append((
            self._encode(state, edge_state), clusters, padded_splits, float(reward),
            self._encode(next_state, next_edge_state), float(done), len(action["cluster"]),
        ))
        self._learn()

    def _learn(self):
        if len(self.replay) < self.batch_size:
            return
        batch = random.sample(self.replay, self.batch_size)
        states = torch.as_tensor(np.stack([item[0] for item in batch]), dtype=torch.float32, device=self.device)
        clusters = torch.as_tensor(np.stack([item[1] for item in batch]), dtype=torch.long, device=self.device)
        splits = torch.as_tensor(np.stack([item[2] for item in batch]), dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([item[3] for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([item[4] for item in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([item[5] for item in batch], dtype=torch.float32, device=self.device)
        counts = torch.as_tensor([item[6] for item in batch], dtype=torch.float32, device=self.device)
        positions = torch.arange(self.max_clients, device=self.device).unsqueeze(0) < counts.unsqueeze(1)
        cluster_q, split_q = self.online(states)
        chosen = cluster_q.gather(2, clusters.unsqueeze(-1)).squeeze(-1) + split_q.gather(2, splits.unsqueeze(-1)).squeeze(-1)
        prediction = (chosen * positions).sum(dim=1) / counts
        with torch.no_grad():
            next_cluster_q, next_split_q = self.online(next_states)
            next_clusters = next_cluster_q.argmax(dim=2, keepdim=True)
            next_splits = next_split_q.argmax(dim=2, keepdim=True)
            target_cluster_q, target_split_q = self.target(next_states)
            next_value = target_cluster_q.gather(2, next_clusters).squeeze(-1) + target_split_q.gather(2, next_splits).squeeze(-1)
            bootstrap = (next_value * positions).sum(dim=1) / counts
            target = rewards + self.gamma * (1.0 - dones) * bootstrap
        loss = functional.smooth_l1_loss(prediction, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        self.update_steps += 1
        if self.update_steps % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
