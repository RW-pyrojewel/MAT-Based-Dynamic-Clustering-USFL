"""PPO agent implementing the hierarchical MAT policy for liquid AI-RAN."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from interfaces.base_agent import BaseAgent
from models.mat_components import CausalDeviceDecoder, ClusterSplitHead, HeterogeneousEncoder


class MATAgent(BaseAgent):
    def __init__(
        self,
        state_dim,
        hidden_dim=128,
        num_migs=7,
        num_cut_layers=7,
        edge_state_dim=2,
        min_bandwidth_share=0.01,
        nominal_bandwidth_hz=100e6,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=4,
        minibatch_size=8,
        value_clip_ratio=0.2,
        max_grad_norm=0.5,
        huber_delta=1.0,
        device="cpu",
    ):
        super().__init__(agent_name="MAT-RL Agent (Proposed)")
        if num_cut_layers < 2:
            raise ValueError("num_cut_layers must be at least two")
        if nominal_bandwidth_hz <= 0.0:
            raise ValueError("nominal_bandwidth_hz must be positive")
        if value_clip_ratio <= 0.0:
            raise ValueError("value_clip_ratio must be positive")
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        self.device = torch.device(device)
        self.num_migs = int(num_migs)
        self.num_cut_layers = int(num_cut_layers)
        self.min_bandwidth_share = float(min_bandwidth_share)
        self.nominal_bandwidth_hz = float(nominal_bandwidth_hz)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.ppo_epochs = int(ppo_epochs)
        self.minibatch_size = int(minibatch_size)
        self.clip_ratio = 0.2
        self.value_clip_ratio = float(value_clip_ratio)
        self.max_grad_norm = float(max_grad_norm)
        self.huber_delta = float(huber_delta)
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.encoder = HeterogeneousEncoder(state_dim, hidden_dim, edge_state_dim=edge_state_dim).to(self.device)
        self.device_decoder = CausalDeviceDecoder(
            hidden_dim,
            num_migs,
            min_bandwidth_share=self.min_bandwidth_share,
        ).to(self.device)
        self.split_head = ClusterSplitHead(hidden_dim, num_migs, num_cut_layers).to(self.device)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1)
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self._parameters(), lr=3e-4)
        self.last_policy_info = None

    def _parameters(self):
        for module in (self.encoder, self.device_decoder, self.split_head, self.value_head):
            yield from module.parameters()

    def _normalise_edge_state(self, edge_state):
        edge = torch.as_tensor(edge_state, dtype=torch.float32, device=self.device).reshape(-1, 2).clone()
        edge[:, 0] = edge[:, 0] / float(self.num_migs)
        edge[:, 1] = edge[:, 1] / self.nominal_bandwidth_hz
        return edge

    @staticmethod
    def _decision_order(client_count, client_ids, deterministic, device):
        if deterministic:
            if client_ids is None:
                return torch.arange(client_count, device=device)
            ids = np.asarray(client_ids)
            if ids.shape != (client_count,):
                raise ValueError("client_ids must have shape (N,)")
            return torch.as_tensor(np.argsort(ids, kind="stable"), dtype=torch.long, device=device)
        return torch.randperm(client_count, device=device)

    @staticmethod
    def _restore_original_order(ordered_values, order):
        restored = torch.empty_like(ordered_values)
        restored[:, order] = ordered_values
        return restored

    def _encode(self, state, edge_state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(0)
        return state_tensor, self.encoder(state_tensor, self._normalise_edge_state(edge_state))

    @torch.no_grad()
    def act(
        self,
        active_clients_state: np.ndarray,
        available_migs: int,
        edge_state: np.ndarray,
        client_ids=None,
        deterministic=False,
    ):
        """Generate one joint action and the component-wise behavior-policy information."""
        _, encoded = self._encode(active_clients_state, edge_state)
        client_count = encoded.shape[1]
        order = self._decision_order(client_count, client_ids, deterministic, self.device)
        ordered_encoded = encoded[:, order]
        ordered_clusters, ordered_bandwidths, cluster_lp, bandwidth_lp, device_entropy = self.device_decoder.act(
            ordered_encoded,
            available_migs,
            deterministic=deterministic,
        )
        clusters = self._restore_original_order(ordered_clusters, order)
        bandwidths = self._restore_original_order(ordered_bandwidths, order)
        l1, l2, split_lp, split_entropy, split_mask = self.split_head.act(
            encoded,
            clusters,
            bandwidths,
            deterministic=deterministic,
        )
        value = self.value_head(encoded.mean(dim=1)).squeeze(-1)
        action = {
            "cluster": clusters.squeeze(0).cpu().numpy(),
            "l1": l1.squeeze(0).cpu().numpy(),
            "l2": l2.squeeze(0).cpu().numpy(),
            "bw": bandwidths.squeeze(0).cpu().numpy(),
        }
        bandwidth_mask = np.ones(client_count, dtype=bool)
        bandwidth_mask[-1] = False
        policy_info = {
            "decision_order": order.cpu().numpy(),
            "cluster_log_probs": cluster_lp.squeeze(0).cpu().numpy(),
            "bandwidth_log_probs": bandwidth_lp.squeeze(0).cpu().numpy(),
            "bandwidth_mask": bandwidth_mask,
            "split_log_probs": split_lp.squeeze(0).cpu().numpy(),
            "split_mask": split_mask.squeeze(0).cpu().numpy(),
            "value": float(value.item()),
            "device_entropy": device_entropy.squeeze(0).cpu().numpy(),
            "split_entropy": split_entropy.squeeze(0).cpu().numpy(),
        }
        self.last_policy_info = policy_info
        return action, policy_info

    @torch.no_grad()
    def step(self, active_clients_state, available_migs, edge_state, client_ids=None):
        action, _ = self.act(
            active_clients_state,
            available_migs,
            edge_state,
            client_ids=client_ids,
            deterministic=True,
        )
        return action["cluster"], action["l1"], action["l2"], action["bw"]

    @torch.no_grad()
    def get_value(self, state, edge_state):
        _, encoded = self._encode(state, edge_state)
        return self.value_head(encoded.mean(dim=1)).squeeze(-1).item()

    def _evaluate_action(self, state, edge_state, action, available_migs, decision_order):
        _, encoded = self._encode(state, edge_state)
        order = torch.as_tensor(decision_order, dtype=torch.long, device=self.device)
        clusters = torch.as_tensor(action["cluster"], dtype=torch.long, device=self.device).reshape(1, -1)
        bandwidths = torch.as_tensor(action["bw"], dtype=torch.float32, device=self.device).reshape(1, -1)
        l1 = torch.as_tensor(action["l1"], dtype=torch.long, device=self.device).reshape(1, -1)
        l2 = torch.as_tensor(action["l2"], dtype=torch.long, device=self.device).reshape(1, -1)
        ordered_encoded = encoded[:, order]
        ordered_clusters = clusters[:, order]
        ordered_bandwidths = bandwidths[:, order]
        cluster_lp, bandwidth_lp, device_entropy = self.device_decoder.evaluate_actions(
            ordered_encoded,
            ordered_clusters,
            ordered_bandwidths,
            available_migs,
        )
        split_lp, split_entropy, split_mask = self.split_head.evaluate_actions(
            encoded,
            clusters,
            bandwidths,
            l1,
            l2,
        )
        value = self.value_head(encoded.mean(dim=1)).squeeze(-1)
        return {
            "cluster_log_probs": cluster_lp.squeeze(0),
            "bandwidth_log_probs": bandwidth_lp.squeeze(0),
            "split_log_probs": split_lp.squeeze(0),
            "split_mask": split_mask.squeeze(0),
            "device_entropy": device_entropy.squeeze(0),
            "split_entropy": split_entropy.squeeze(0),
            "value": value,
        }

    def _compute_gae(self, rewards, next_states, dones, edge_states, next_edge_states, policy_infos, station_ids, epochs):
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=bool)
        values = np.asarray([info["value"] for info in policy_infos], dtype=np.float32)
        with torch.no_grad():
            next_values = np.asarray([
                0.0 if done else self.get_value(next_state, next_edge)
                for next_state, next_edge, done in zip(next_states, next_edge_states, dones)
            ], dtype=np.float32)
        station_ids = np.asarray(station_ids, dtype=np.int64)
        epochs = np.asarray(epochs, dtype=np.int64)
        raw_advantages = np.zeros_like(rewards)
        for station_id in np.unique(station_ids):
            indices = np.flatnonzero(station_ids == station_id).tolist()
            indices.sort(key=lambda index: epochs[index], reverse=True)
            next_advantage = 0.0
            for index in indices:
                nonterminal = 0.0 if dones[index] else 1.0
                delta = rewards[index] + self.gamma * next_values[index] * nonterminal - values[index]
                raw_advantages[index] = delta + self.gamma * self.gae_lambda * nonterminal * next_advantage
                next_advantage = raw_advantages[index]
        returns = raw_advantages + values
        advantages = np.zeros_like(raw_advantages)
        station_stats = {}
        for station_id in np.unique(station_ids):
            indices = np.flatnonzero(station_ids == station_id)
            station_advantages = raw_advantages[indices]
            advantage_mean = float(station_advantages.mean())
            advantage_std = float(station_advantages.std())
            if advantage_std > 1e-8:
                advantages[indices] = (station_advantages - advantage_mean) / advantage_std
            else:
                advantages[indices] = 0.0
            station_returns = returns[indices]
            return_std = float(station_returns.std())
            station_stats[int(station_id)] = {
                "indices": indices,
                "return_mean": float(station_returns.mean()),
                "return_std": return_std,
                "return_scale": return_std if return_std > 1e-6 else 1.0,
            }
        return advantages, returns, station_stats

    @staticmethod
    def _component_loss(new_log_probs, old_log_probs, advantage, clip_ratio):
        log_ratio = (new_log_probs - old_log_probs).clamp(-20.0, 20.0)
        ratio = log_ratio.exp()
        unclipped = ratio * advantage
        clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        approx_kl = (old_log_probs - new_log_probs).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
        return policy_loss, approx_kl, clip_fraction

    @staticmethod
    def _value_loss(new_value, old_value, target, return_mean, return_scale, clip_ratio, huber_delta):
        """Clipped Huber objective in station-local normalized return coordinates."""
        new_normalized = (new_value - return_mean) / return_scale
        old_normalized = (old_value - return_mean) / return_scale
        target_normalized = (target - return_mean) / return_scale
        clipped_value = old_normalized + (new_normalized - old_normalized).clamp(-clip_ratio, clip_ratio)
        unclipped_loss = functional.smooth_l1_loss(
            new_normalized, target_normalized, reduction="none", beta=huber_delta,
        )
        clipped_loss = functional.smooth_l1_loss(
            clipped_value, target_normalized, reduction="none", beta=huber_delta,
        )
        value_loss = torch.maximum(unclipped_loss, clipped_loss).mean()
        clip_fraction = ((new_normalized - old_normalized).abs() > clip_ratio).float().mean()
        return value_loss, clip_fraction

    @staticmethod
    def _gradient_norm(parameters):
        squared_norms = [
            parameter.grad.detach().float().norm(2).square()
            for parameter in parameters
            if parameter.grad is not None
        ]
        if not squared_norms:
            return 0.0
        return float(torch.stack(squared_norms).sum().sqrt().item())

    @staticmethod
    def _explained_variance(targets, predictions):
        targets = np.asarray(targets, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)
        target_variance = float(np.var(targets))
        if target_variance <= 1e-12:
            return 0.0
        value = 1.0 - float(np.var(targets - predictions)) / target_variance
        return value if np.isfinite(value) else 0.0

    def update_policy(self, rewards, next_states, dones, **kwargs):
        states = kwargs.get("states")
        actions = kwargs.get("actions")
        edge_states = kwargs.get("edge_states")
        next_edge_states = kwargs.get("next_edge_states")
        available_migs = kwargs.get("available_migs")
        policy_infos = kwargs.get("policy_infos")
        station_ids = kwargs.get("station_ids")
        epochs = kwargs.get("epochs")
        required = (states, actions, edge_states, next_edge_states, available_migs, policy_infos, station_ids, epochs)
        if any(value is None for value in required) or len(states) == 0:
            return {}
        advantages, returns, station_stats = self._compute_gae(
            rewards,
            next_states,
            dones,
            edge_states,
            next_edge_states,
            policy_infos,
            station_ids,
            epochs,
        )
        diagnostics = {
            key: []
            for key in (
                "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                "value_clip_fraction", "grad_norm_pre", "grad_norm_post",
            )
        }
        indices = np.arange(len(states))
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.minibatch_size):
                minibatch = indices[start:start + self.minibatch_size]
                sample_losses = []
                batch_policy_losses = []
                batch_value_losses = []
                batch_entropies = []
                batch_kls = []
                batch_clips = []
                batch_value_clips = []
                for index in minibatch:
                    evaluated = self._evaluate_action(
                        states[index],
                        edge_states[index],
                        actions[index],
                        available_migs[index],
                        policy_infos[index]["decision_order"],
                    )
                    old_cluster = torch.as_tensor(policy_infos[index]["cluster_log_probs"], dtype=torch.float32, device=self.device)
                    old_bandwidth = torch.as_tensor(policy_infos[index]["bandwidth_log_probs"], dtype=torch.float32, device=self.device)
                    bandwidth_mask = torch.as_tensor(policy_infos[index]["bandwidth_mask"], dtype=torch.bool, device=self.device)
                    old_split = torch.as_tensor(policy_infos[index]["split_log_probs"], dtype=torch.float32, device=self.device)
                    split_mask = torch.as_tensor(policy_infos[index]["split_mask"], dtype=torch.bool, device=self.device)
                    new_components = [evaluated["cluster_log_probs"]]
                    old_components = [old_cluster]
                    entropy_components = [evaluated["device_entropy"]]
                    if bandwidth_mask.any():
                        new_components.append(evaluated["bandwidth_log_probs"][bandwidth_mask])
                        old_components.append(old_bandwidth[bandwidth_mask])
                    if split_mask.any():
                        new_components.append(evaluated["split_log_probs"][split_mask])
                        old_components.append(old_split[split_mask])
                        entropy_components.append(evaluated["split_entropy"][split_mask])
                    new_log_probs = torch.cat(new_components)
                    old_log_probs = torch.cat(old_components)
                    advantage = torch.tensor(advantages[index], dtype=torch.float32, device=self.device)
                    policy_loss, approx_kl, clip_fraction = self._component_loss(
                        new_log_probs,
                        old_log_probs,
                        advantage,
                        self.clip_ratio,
                    )
                    value_target = torch.tensor(returns[index], dtype=torch.float32, device=self.device)
                    old_value = torch.tensor(policy_infos[index]["value"], dtype=torch.float32, device=self.device)
                    station_stat = station_stats[int(station_ids[index])]
                    return_mean = torch.tensor(station_stat["return_mean"], dtype=torch.float32, device=self.device)
                    return_scale = torch.tensor(station_stat["return_scale"], dtype=torch.float32, device=self.device)
                    value_loss, value_clip_fraction = self._value_loss(
                        evaluated["value"].squeeze(0), old_value, value_target, return_mean,
                        return_scale, self.value_clip_ratio, self.huber_delta,
                    )
                    entropy = torch.cat(entropy_components).mean()
                    sample_losses.append(policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy)
                    batch_policy_losses.append(policy_loss.detach())
                    batch_value_losses.append(value_loss.detach())
                    batch_entropies.append(entropy.detach())
                    batch_kls.append(approx_kl.detach())
                    batch_clips.append(clip_fraction.detach())
                    batch_value_clips.append(value_clip_fraction.detach())
                loss = torch.stack(sample_losses).mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                parameters = list(self._parameters())
                grad_norm_pre = float(torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm))
                grad_norm_post = self._gradient_norm(parameters)
                self.optimizer.step()
                diagnostics["policy_loss"].append(torch.stack(batch_policy_losses).mean().item())
                diagnostics["value_loss"].append(torch.stack(batch_value_losses).mean().item())
                diagnostics["entropy"].append(torch.stack(batch_entropies).mean().item())
                diagnostics["approx_kl"].append(torch.stack(batch_kls).mean().item())
                diagnostics["clip_fraction"].append(torch.stack(batch_clips).mean().item())
                diagnostics["value_clip_fraction"].append(torch.stack(batch_value_clips).mean().item())
                diagnostics["grad_norm_pre"].append(grad_norm_pre)
                diagnostics["grad_norm_post"].append(grad_norm_post)
        result = {key: float(np.mean(values)) for key, values in diagnostics.items() if values}
        result["grad_norm"] = result["grad_norm_pre"]
        with torch.no_grad():
            updated_values = np.asarray([
                self.get_value(state, edge_state)
                for state, edge_state in zip(states, edge_states)
            ], dtype=np.float64)
        for station_id, station_stat in station_stats.items():
            station_indices = station_stat["indices"]
            prefix = f"station_{station_id}"
            result[f"{prefix}_return_mean"] = station_stat["return_mean"]
            result[f"{prefix}_return_std"] = station_stat["return_std"]
            result[f"{prefix}_explained_variance"] = self._explained_variance(
                returns[station_indices], updated_values[station_indices],
            )
        return result
