"""Adapted PCSFL baseline using a fixed-width, factorised DDQN controller."""
from collections import deque
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional

from interfaces.base_agent import BaseAgent


class _PCSFLQNetwork(nn.Module):
    """Paper-style recurrent state encoder with independent cluster/split Q heads."""

    def __init__(self, token_dim, max_clients, max_migs, split_count, hidden_dim=128):
        super().__init__()
        self.max_clients = max_clients
        self.encoder = nn.LSTM(token_dim, hidden_dim, batch_first=True)
        self.cluster_head = nn.Linear(hidden_dim, max_migs)
        self.split_head = nn.Linear(hidden_dim, split_count)

    def forward(self, state):
        hidden, _ = self.encoder(state)
        return self.cluster_head(hidden), self.split_head(hidden)


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
        model_pca_dim=8,
        clustering_factor_reference=0.05,
        waiting_factor_reference=1.0,
        device="cpu",
    ):
        super().__init__(agent_name="PaperAdapted-PCSFL")
        self.state_dim = int(state_dim)
        self.max_clients = int(max_clients)
        self.max_migs = int(max_migs)
        if int(num_cut_layers) < 3:
            raise ValueError("num_cut_layers must keep both client-side parts non-empty")
        self.split_pairs = [(layer, layer + 1) for layer in range(1, int(num_cut_layers) - 1)]
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = int(epsilon_decay)
        self.batch_size = int(batch_size)
        self.target_update_interval = int(target_update_interval)
        self.model_pca_dim = int(model_pca_dim)
        self.clustering_factor_reference = float(clustering_factor_reference)
        self.waiting_factor_reference = float(waiting_factor_reference)
        self.device = torch.device(device)
        token_dim = self.state_dim + 1 + self.model_pca_dim + 2
        self.online = _PCSFLQNetwork(token_dim, self.max_clients, self.max_migs, len(self.split_pairs)).to(self.device)
        self.target = _PCSFLQNetwork(token_dim, self.max_clients, self.max_migs, len(self.split_pairs)).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay = deque(maxlen=int(replay_size))
        self.action_steps = 0
        self.update_steps = 0
        self.last_diagnostics = {}
        self.client_model_embeddings = {}

    def embeddings_for(self, client_ids, global_fallback):
        fallback = np.asarray(global_fallback, dtype=np.float32)
        return np.asarray([
            self.client_model_embeddings.get(int(client_id), fallback) for client_id in client_ids
        ], dtype=np.float32)

    def update_client_embeddings(self, client_ids, embeddings):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.shape != (len(client_ids), self.model_pca_dim):
            raise ValueError("PCSFL client model embeddings have an invalid shape")
        for client_id, embedding in zip(client_ids, embeddings):
            self.client_model_embeddings[int(client_id)] = embedding.copy()

    def _encode(self, state, edge_state, data_volumes=None, model_embedding=None):
        state = np.asarray(state, dtype=np.float32)
        if state.ndim != 2 or state.shape[1] != self.state_dim or len(state) > self.max_clients:
            raise ValueError("PCSFL received an incompatible active-client state")
        count = len(state)
        volumes = np.ones(count, dtype=np.float32) if data_volumes is None else np.asarray(data_volumes, dtype=np.float32)
        if volumes.shape != (count,):
            raise ValueError("data_volumes must have shape (N,)")
        volumes = volumes / max(float(volumes.max()), 1.0)
        embedding = np.zeros((count, self.model_pca_dim), dtype=np.float32) if model_embedding is None else np.asarray(model_embedding, dtype=np.float32)
        if embedding.shape == (self.model_pca_dim,):
            embedding = np.repeat(embedding[None], count, axis=0)
        if embedding.shape != (count, self.model_pca_dim):
            raise ValueError("model_embedding has the wrong PCSFL PCA dimension")
        edge = np.asarray(edge_state, dtype=np.float32).reshape(2)
        tokens = np.concatenate([
            state, volumes[:, None], embedding,
            np.repeat(edge[None], count, axis=0),
        ], axis=1)
        padded = np.zeros((self.max_clients, tokens.shape[1]), dtype=np.float32)
        padded[:count] = tokens
        return padded

    def _epsilon(self, deterministic):
        if deterministic:
            return 0.0
        progress = min(1.0, self.action_steps / max(self.epsilon_decay, 1))
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    @torch.no_grad()
    def act(self, active_clients_state, available_migs, edge_state, deterministic=False,
            data_volumes=None, model_embedding=None, **_):
        client_count = len(active_clients_state)
        if client_count == 0 or client_count > self.max_clients or not 1 <= available_migs <= self.max_migs:
            raise ValueError("PCSFL received an unsupported client or MIG count")
        encoded = self._encode(active_clients_state, edge_state, data_volumes, model_embedding)
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
            "virtual_cluster": clusters.astype(np.int64).copy(),
            "l1": pairs[:, 0],
            "l2": pairs[:, 1],
            "bw": np.full(client_count, 1.0 / client_count, dtype=np.float64),
        }, {"paper_algorithm": "PCSFL-LSTM/dual-DDQN", "epsilon": float(epsilon)}

    def step(self, active_clients_state, available_migs):
        action, _ = self.act(active_clients_state, available_migs, np.asarray([available_migs, 1.0], dtype=np.float32))
        return action["cluster"], action["l1"], action["l2"], action["bw"]

    @staticmethod
    def _factor_map(value, reference):
        scaled = np.clip(float(value) / max(float(reference), 1e-12), 0.0, 3.0)
        return 0.5 * np.sin((np.pi / 3.0) * scaled - np.pi / 2.0) + 0.5

    def paper_reward(self, clustering_factor, waiting_factor):
        cluster_term = self._factor_map(clustering_factor, self.clustering_factor_reference)
        waiting_term = self._factor_map(waiting_factor, self.waiting_factor_reference)
        return -float(cluster_term + waiting_term)

    def observe(self, state, edge_state, action, reward, next_state, next_edge_state, done,
                data_volumes=None, next_data_volumes=None, model_embedding=None,
                next_model_embedding=None, clustering_factor=None, waiting_factor=None):
        split_lookup = {pair: index for index, pair in enumerate(self.split_pairs)}
        split_ids = np.asarray([split_lookup[(int(first), int(second))] for first, second in zip(action["l1"], action["l2"])])
        clusters = np.zeros(self.max_clients, dtype=np.int64)
        padded_splits = np.zeros(self.max_clients, dtype=np.int64)
        clusters[:len(action["cluster"])] = action["cluster"]
        padded_splits[:len(split_ids)] = split_ids
        learning_reward = (float(reward) if clustering_factor is None or waiting_factor is None else
                           self.paper_reward(clustering_factor, waiting_factor))
        self.replay.append((
            self._encode(state, edge_state, data_volumes, model_embedding), clusters,
            padded_splits, learning_reward,
            self._encode(next_state, next_edge_state, next_data_volumes, next_model_embedding),
            float(done), int(np.asarray(next_edge_state).reshape(2)[0]), len(action["cluster"]),
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
        next_migs = torch.as_tensor([item[6] for item in batch], dtype=torch.long, device=self.device)
        counts = torch.as_tensor([item[7] for item in batch], dtype=torch.float32, device=self.device)
        positions = torch.arange(self.max_clients, device=self.device).unsqueeze(0) < counts.unsqueeze(1)
        cluster_q, split_q = self.online(states)
        chosen_cluster = cluster_q.gather(2, clusters.unsqueeze(-1)).squeeze(-1)
        chosen_split = split_q.gather(2, splits.unsqueeze(-1)).squeeze(-1)
        prediction_cluster = (chosen_cluster * positions).sum(dim=1) / counts
        prediction_split = (chosen_split * positions).sum(dim=1) / counts
        with torch.no_grad():
            next_cluster_q, next_split_q = self.online(next_states)
            mig_ids = torch.arange(self.max_migs, device=self.device).view(1, 1, -1)
            next_cluster_q = next_cluster_q.masked_fill(
                mig_ids >= next_migs.view(-1, 1, 1), torch.finfo(next_cluster_q.dtype).min)
            next_clusters = next_cluster_q.argmax(dim=2, keepdim=True)
            next_splits = next_split_q.argmax(dim=2, keepdim=True)
            target_cluster_q, target_split_q = self.target(next_states)
            next_cluster_value = target_cluster_q.gather(2, next_clusters).squeeze(-1)
            next_split_value = target_split_q.gather(2, next_splits).squeeze(-1)
            cluster_bootstrap = (next_cluster_value * positions).sum(dim=1) / counts
            split_bootstrap = (next_split_value * positions).sum(dim=1) / counts
            cluster_target = rewards + self.gamma * (1.0 - dones) * cluster_bootstrap
            split_target = rewards + self.gamma * (1.0 - dones) * split_bootstrap
        cluster_loss = functional.smooth_l1_loss(prediction_cluster, cluster_target)
        split_loss = functional.smooth_l1_loss(prediction_split, split_target)
        loss = cluster_loss + split_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        self.last_diagnostics = {
            "pcsfl_cluster_q_loss": float(cluster_loss.detach()),
            "pcsfl_split_q_loss": float(split_loss.detach()),
            "pcsfl_learning_reward_mean": float(rewards.mean()),
        }
        self.update_steps += 1
        if self.update_steps % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
