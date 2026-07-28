"""PPO agent implementing the hierarchical MAT policy for liquid AI-RAN."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from interfaces.base_agent import BaseAgent
from models.mat_components import AutoregressiveDecoder, HeterogeneousEncoder


class MATAgent(BaseAgent):
    def __init__(self, state_dim, hidden_dim=128, num_migs=7, num_cut_layers=7, edge_state_dim=2, device="cpu"):
        super().__init__(agent_name="MAT-RL Agent (Proposed)")
        if num_cut_layers < 2:
            raise ValueError("num_cut_layers must be at least two")
        self.device, self.num_migs, self.num_cut_layers = torch.device(device), num_migs, num_cut_layers
        self.encoder = HeterogeneousEncoder(state_dim, hidden_dim, edge_state_dim=edge_state_dim).to(self.device)
        self.decoder = AutoregressiveDecoder(hidden_dim, num_migs).to(self.device)
        self.cluster_l1_head = nn.Linear(hidden_dim, num_cut_layers - 1).to(self.device)
        self.cluster_l2_head = nn.Linear(hidden_dim, num_cut_layers).to(self.device)
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1)).to(self.device)
        self.optimizer = torch.optim.Adam(self._parameters(), lr=3e-4)
        self.gamma, self.clip_ratio, self.value_coef, self.entropy_coef = 0.99, 0.2, 0.5, 0.01
        self.last_action_log_prob = None

    def _parameters(self):
        for module in (self.encoder, self.decoder, self.cluster_l1_head, self.cluster_l2_head, self.value_head):
            yield from module.parameters()

    def _cluster_actions(self, encoded, clusters, deterministic=False, supplied_l1=None, supplied_l2=None):
        batch_size, client_count, _ = encoded.shape
        l1_all = torch.zeros((batch_size, client_count), dtype=torch.long, device=self.device)
        l2_all = torch.ones((batch_size, client_count), dtype=torch.long, device=self.device)
        total_log_prob, total_entropy = encoded.new_zeros(batch_size), encoded.new_zeros(batch_size)
        for mig in range(self.num_migs):
            members = clusters.eq(mig)
            present = members.any(1)
            if not present.any():
                continue
            context = (encoded * members.unsqueeze(-1)).sum(1) / members.sum(1, keepdim=True).clamp_min(1)
            l1_dist = Categorical(logits=self.cluster_l1_head(context))
            l1 = l1_dist.probs.argmax(-1) if supplied_l1 is None and deterministic else (l1_dist.sample() if supplied_l1 is None else supplied_l1[:, members[0]].mode(1).values.long())
            valid_l2 = torch.arange(self.num_cut_layers, device=self.device).unsqueeze(0) > l1.unsqueeze(1)
            l2_dist = Categorical(logits=self.cluster_l2_head(context).masked_fill(~valid_l2, -torch.inf))
            l2 = l2_dist.probs.argmax(-1) if supplied_l2 is None and deterministic else (l2_dist.sample() if supplied_l2 is None else supplied_l2[:, members[0]].mode(1).values.long())
            total_log_prob += present.float() * (l1_dist.log_prob(l1) + l2_dist.log_prob(l2))
            total_entropy += present.float() * (l1_dist.entropy() + l2_dist.entropy())
            l1_all, l2_all = torch.where(members, l1.unsqueeze(1), l1_all), torch.where(members, l2.unsqueeze(1), l2_all)
        return l1_all, l2_all, total_log_prob, total_entropy

    @torch.no_grad()
    def act(self, active_clients_state: np.ndarray, available_migs: int, edge_state: np.ndarray, deterministic=False):
        """Generate one joint action and its behavior-policy log probability."""
        state = torch.as_tensor(active_clients_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        edge = torch.as_tensor(edge_state, dtype=torch.float32, device=self.device).reshape(1, -1)
        encoded = self.encoder(state, edge)
        clusters, bandwidths, device_log_prob, _ = self.decoder.act(encoded, available_migs, deterministic=deterministic)
        l1, l2, cluster_log_prob, _ = self._cluster_actions(encoded, clusters, deterministic=deterministic)
        action = {
            "cluster": clusters.squeeze(0).cpu().numpy(),
            "l1": l1.squeeze(0).cpu().numpy(),
            "l2": l2.squeeze(0).cpu().numpy(),
            "bw": bandwidths.squeeze(0).cpu().numpy(),
        }
        log_prob = (device_log_prob + cluster_log_prob).item()
        self.last_action_log_prob = log_prob
        return action, log_prob

    @torch.no_grad()
    def step(self, active_clients_state: np.ndarray, available_migs: int, edge_state: np.ndarray):
        """Deterministic evaluation action required by the BaseAgent interface."""
        action, _ = self.act(active_clients_state, available_migs, edge_state, deterministic=True)
        return action["cluster"], action["l1"], action["l2"], action["bw"]

    @torch.no_grad()
    def get_value(self, state, edge_state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        edge = torch.as_tensor(edge_state, dtype=torch.float32, device=self.device).reshape(1, -1)
        return self.value_head(self.encoder(state, edge).mean(1)).item()

    def _evaluate_action(self, state, edge_state, action, available_migs):
        encoded = self.encoder(state, edge_state)
        clusters = torch.as_tensor(action["cluster"], device=self.device).reshape(1, -1)
        bandwidths = torch.as_tensor(action["bw"], dtype=torch.float32, device=self.device).reshape(1, -1)
        l1 = torch.as_tensor(action["l1"], device=self.device).reshape(1, -1)
        l2 = torch.as_tensor(action["l2"], device=self.device).reshape(1, -1)
        device_lp, device_entropy = self.decoder.evaluate_actions(encoded, clusters, bandwidths, available_migs)
        _, _, cluster_lp, cluster_entropy = self._cluster_actions(encoded, clusters, supplied_l1=l1, supplied_l2=l2)
        return device_lp + cluster_lp, device_entropy + cluster_entropy, self.value_head(encoded.mean(1)).squeeze(-1)

    def update_policy(self, rewards, next_states, dones, **kwargs):
        states, actions, old_log_probs = kwargs.get("states"), kwargs.get("actions"), kwargs.get("old_log_probs")
        edge_states, next_edge_states, available_migs = kwargs.get("edge_states"), kwargs.get("next_edge_states"), kwargs.get("available_migs")
        if states is None or actions is None or old_log_probs is None or edge_states is None or next_edge_states is None or available_migs is None or len(states) == 0:
            return
        with torch.no_grad():
            values = torch.tensor([self.get_value(state, edge) for state, edge in zip(states, edge_states)], device=self.device)
            next_values = torch.tensor([0.0 if done else self.get_value(next_state, next_edge) for next_state, next_edge, done in zip(next_states, next_edge_states, dones)], device=self.device)
            returns = torch.as_tensor(rewards, dtype=torch.float32, device=self.device) + self.gamma * next_values
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        for _ in range(4):
            for index, state in enumerate(states):
                state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                edge = torch.as_tensor(edge_states[index], dtype=torch.float32, device=self.device).reshape(1, -1)
                step_migs = available_migs[index] if isinstance(available_migs, (list, tuple, np.ndarray)) else available_migs
                log_prob, entropy, value = self._evaluate_action(state, edge, actions[index], step_migs)
                ratio = (log_prob - old_log_probs[index]).exp()
                objective = torch.min(ratio * advantages[index], ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * advantages[index])
                loss = -objective + self.value_coef * F.mse_loss(value, returns[index].unsqueeze(0)) - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self._parameters()), 0.5)
                self.optimizer.step()
