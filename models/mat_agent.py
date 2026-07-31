"""Shared-encoder MAT PPO with a frozen target critic and full-batch-equivalent updates."""
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from interfaces.base_agent import BaseAgent
from models.mat_components import CausalDeviceDecoder, ClusterSplitHead, HeterogeneousEncoder


class MATAgent(BaseAgent):
    def __init__(self, state_dim, hidden_dim=128, num_migs=7, num_cut_layers=7, edge_state_dim=2,
                 min_bandwidth_share=0.01, nominal_bandwidth_hz=100e6, gamma=0.99, gae_lambda=0.95,
                 ppo_epochs=10, minibatch_size=256, actor_learning_rate=1e-4,
                 critic_learning_rate=1e-4, max_grad_norm=0.5, huber_delta=1.0,
                 target_kl=0.03, device="cpu"):
        super().__init__(agent_name="MAT-RL Agent (Proposed)")
        if num_cut_layers < 2 or nominal_bandwidth_hz <= 0:
            raise ValueError("invalid MAT dimensions")
        if min(actor_learning_rate, critic_learning_rate, max_grad_norm, huber_delta) <= 0:
            raise ValueError("learning rates and loss/gradient limits must be positive")
        self.device = torch.device(device)
        self.num_migs, self.num_cut_layers = int(num_migs), int(num_cut_layers)
        self.min_bandwidth_share = float(min_bandwidth_share)
        self.nominal_bandwidth_hz = float(nominal_bandwidth_hz)
        self.gamma, self.gae_lambda = float(gamma), float(gae_lambda)
        self.ppo_epochs, self.minibatch_size = int(ppo_epochs), int(minibatch_size)
        self.clip_ratio, self.value_coef, self.entropy_coef = 0.2, 0.5, 0.01
        self.max_grad_norm, self.huber_delta, self.target_kl = float(max_grad_norm), float(huber_delta), float(target_kl)
        self.encoder = HeterogeneousEncoder(state_dim, hidden_dim, edge_state_dim=edge_state_dim).to(self.device)
        self.device_decoder = CausalDeviceDecoder(hidden_dim, num_migs, min_bandwidth_share=self.min_bandwidth_share).to(self.device)
        self.split_head = ClusterSplitHead(hidden_dim, num_migs, num_cut_layers).to(self.device)
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1)).to(self.device)
        self.target_encoder = deepcopy(self.encoder).to(self.device)
        self.target_value_head = deepcopy(self.value_head).to(self.device)
        self._freeze_targets()
        self.optimizer = torch.optim.Adam([
            {"params": list(self.encoder.parameters()) + list(self.value_head.parameters()), "lr": critic_learning_rate},
            {"params": list(self.device_decoder.parameters()) + list(self.split_head.parameters()), "lr": actor_learning_rate},
        ])
        self.policy_version = 0
        self.update_count = 0
        self.last_policy_info = None

    def _parameters(self):
        for module in (self.encoder, self.device_decoder, self.split_head, self.value_head):
            yield from module.parameters()

    def _freeze_targets(self):
        for module in (self.target_encoder, self.target_value_head):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None

    def sync_target(self):
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_value_head.load_state_dict(self.value_head.state_dict())
        self._freeze_targets()

    def _normalise_edge_state(self, edge_state):
        edge = torch.as_tensor(edge_state, dtype=torch.float32, device=self.device).reshape(-1, 2).clone()
        edge[:, 0] /= float(self.num_migs)
        edge[:, 1] /= self.nominal_bandwidth_hz
        return edge

    @staticmethod
    def _decision_order(client_count, client_ids, deterministic, device):
        if not deterministic:
            return torch.randperm(client_count, device=device)
        if client_ids is None:
            return torch.arange(client_count, device=device)
        ids = np.asarray(client_ids)
        if ids.shape != (client_count,):
            raise ValueError("client_ids must have shape (N,)")
        return torch.as_tensor(np.argsort(ids, kind="stable"), dtype=torch.long, device=device)

    @staticmethod
    def _restore_original_order(ordered_values, order):
        restored = torch.empty_like(ordered_values)
        restored[:, order] = ordered_values
        return restored

    @staticmethod
    def _aggregate_token_values(value_head, encoded):
        return value_head(encoded).squeeze(-1).mean(dim=-1)

    def _encode(self, state, edge_state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(0)
        return state_tensor, self.encoder(state_tensor, self._normalise_edge_state(edge_state))

    @torch.no_grad()
    def _target_value(self, state, edge_state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(0)
        encoded = self.target_encoder(state_tensor, self._normalise_edge_state(edge_state))
        return float(self._aggregate_token_values(self.target_value_head, encoded).item())

    @torch.no_grad()
    def act(self, active_clients_state, available_migs, edge_state, client_ids=None, deterministic=False):
        _, encoded = self._encode(active_clients_state, edge_state)
        client_count = encoded.shape[1]
        order = self._decision_order(client_count, client_ids, deterministic, self.device)
        ordered_encoded = encoded[:, order]
        ordered_clusters, ordered_bandwidths, cluster_lp, bandwidth_lp, device_entropy = self.device_decoder.act(
            ordered_encoded, available_migs, deterministic=deterministic)
        clusters = self._restore_original_order(ordered_clusters, order)
        bandwidths = self._restore_original_order(ordered_bandwidths, order)
        l1, l2, split_lp, split_entropy, split_mask = self.split_head.act(
            encoded, clusters, bandwidths, deterministic=deterministic)
        value = self._aggregate_token_values(self.value_head, encoded)
        action = {"cluster": clusters.squeeze(0).cpu().numpy(), "l1": l1.squeeze(0).cpu().numpy(),
                  "l2": l2.squeeze(0).cpu().numpy(), "bw": bandwidths.squeeze(0).cpu().numpy()}
        bandwidth_mask = np.ones(client_count, dtype=bool)
        bandwidth_mask[-1] = False
        policy_info = {
            "decision_order": order.cpu().numpy(), "cluster_log_probs": cluster_lp.squeeze(0).cpu().numpy(),
            "bandwidth_log_probs": bandwidth_lp.squeeze(0).cpu().numpy(), "bandwidth_mask": bandwidth_mask,
            "split_log_probs": split_lp.squeeze(0).cpu().numpy(), "split_mask": split_mask.squeeze(0).cpu().numpy(),
            "value": float(value.item()), "device_entropy": device_entropy.squeeze(0).cpu().numpy(),
            "split_entropy": split_entropy.squeeze(0).cpu().numpy(),
        }
        self.last_policy_info = policy_info
        return action, policy_info

    @torch.no_grad()
    def step(self, active_clients_state, available_migs, edge_state, client_ids=None):
        action, _ = self.act(active_clients_state, available_migs, edge_state, client_ids=client_ids, deterministic=True)
        return action["cluster"], action["l1"], action["l2"], action["bw"]

    @torch.no_grad()
    def get_value(self, state, edge_state):
        _, encoded = self._encode(state, edge_state)
        return float(self._aggregate_token_values(self.value_head, encoded).item())

    def _evaluate_action(self, state, edge_state, action, available_migs, decision_order):
        _, encoded = self._encode(state, edge_state)
        order = torch.as_tensor(decision_order, dtype=torch.long, device=self.device)
        clusters = torch.as_tensor(action["cluster"], dtype=torch.long, device=self.device).reshape(1, -1)
        bandwidths = torch.as_tensor(action["bw"], dtype=torch.float32, device=self.device).reshape(1, -1)
        l1 = torch.as_tensor(action["l1"], dtype=torch.long, device=self.device).reshape(1, -1)
        l2 = torch.as_tensor(action["l2"], dtype=torch.long, device=self.device).reshape(1, -1)
        cluster_lp, bandwidth_lp, device_entropy = self.device_decoder.evaluate_actions(
            encoded[:, order], clusters[:, order], bandwidths[:, order], available_migs)
        split_lp, split_entropy, split_mask = self.split_head.evaluate_actions(encoded, clusters, bandwidths, l1, l2)
        return {"cluster_log_probs": cluster_lp.squeeze(0), "bandwidth_log_probs": bandwidth_lp.squeeze(0),
                "split_log_probs": split_lp.squeeze(0), "split_mask": split_mask.squeeze(0),
                "device_entropy": device_entropy.squeeze(0), "split_entropy": split_entropy.squeeze(0),
                "value": self._aggregate_token_values(self.value_head, encoded)}

    def _compute_gae(self, rewards, next_states, dones, edge_states, next_edge_states, policy_infos,
                     station_ids, epochs, trajectory_ids=None):
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=bool)
        values = np.asarray([info["value"] for info in policy_infos], dtype=np.float32)
        next_values = np.asarray([0.0 if done else self._target_value(state, edge)
                                  for state, edge, done in zip(next_states, next_edge_states, dones)], dtype=np.float32)
        td_targets = rewards + self.gamma * next_values * (~dones)
        td_residuals = td_targets - values
        raw_advantages = np.zeros_like(rewards)
        trajectories = list(zip(np.zeros(len(rewards), dtype=int), station_ids)) if trajectory_ids is None else [tuple(x) for x in trajectory_ids]
        for trajectory in dict.fromkeys(trajectories):
            indices = [i for i, item in enumerate(trajectories) if item == trajectory]
            indices.sort(key=lambda i: epochs[i], reverse=True)
            next_advantage = 0.0
            for index in indices:
                nonterminal = 0.0 if dones[index] else 1.0
                raw_advantages[index] = td_residuals[index] + self.gamma * self.gae_lambda * nonterminal * next_advantage
                next_advantage = raw_advantages[index]
        returns = raw_advantages + values
        std = float(raw_advantages.std())
        advantages = (raw_advantages - raw_advantages.mean()) / std if std > 1e-8 else np.zeros_like(raw_advantages)
        return advantages.astype(np.float32), returns.astype(np.float32), td_targets.astype(np.float32), td_residuals.astype(np.float32)

    @staticmethod
    def _component_loss(new_log_probs, old_log_probs, advantage, clip_ratio):
        log_ratio = (new_log_probs - old_log_probs).clamp(-20.0, 20.0)
        ratio = log_ratio.exp()
        policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage).mean()
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
        return policy_loss, approx_kl, clip_fraction

    @staticmethod
    def _gradient_norm(parameters):
        values = [p.grad.detach().float().norm(2).square() for p in parameters if p.grad is not None]
        return float(torch.stack(values).sum().sqrt().item()) if values else 0.0

    @staticmethod
    def _explained_variance(targets, predictions):
        targets, predictions = np.asarray(targets, dtype=np.float64), np.asarray(predictions, dtype=np.float64)
        variance = float(np.var(targets))
        if variance <= 1e-12:
            return 0.0
        result = 1.0 - float(np.var(targets - predictions)) / variance
        return float(result) if np.isfinite(result) else 0.0

    def _sample_terms(self, index, states, actions, edge_states, available_migs, policy_infos, advantages, td_targets):
        evaluated = self._evaluate_action(states[index], edge_states[index], actions[index], available_migs[index],
                                          policy_infos[index]["decision_order"])
        info = policy_infos[index]
        old_cluster = torch.as_tensor(info["cluster_log_probs"], dtype=torch.float32, device=self.device)
        old_bandwidth = torch.as_tensor(info["bandwidth_log_probs"], dtype=torch.float32, device=self.device)
        bandwidth_mask = torch.as_tensor(info["bandwidth_mask"], dtype=torch.bool, device=self.device)
        old_split = torch.as_tensor(info["split_log_probs"], dtype=torch.float32, device=self.device)
        split_mask = torch.as_tensor(info["split_mask"], dtype=torch.bool, device=self.device)
        new_parts, old_parts = [evaluated["cluster_log_probs"]], [old_cluster]
        entropy_parts = [evaluated["device_entropy"]]
        if bandwidth_mask.any():
            new_parts.append(evaluated["bandwidth_log_probs"][bandwidth_mask]); old_parts.append(old_bandwidth[bandwidth_mask])
        if split_mask.any():
            new_parts.append(evaluated["split_log_probs"][split_mask]); old_parts.append(old_split[split_mask])
            entropy_parts.append(evaluated["split_entropy"][split_mask])
        advantage = torch.as_tensor(advantages[index], dtype=torch.float32, device=self.device)
        policy_loss, kl, clip = self._component_loss(torch.cat(new_parts), torch.cat(old_parts), advantage, self.clip_ratio)
        target = torch.as_tensor(td_targets[index], dtype=torch.float32, device=self.device)
        value_loss = functional.smooth_l1_loss(evaluated["value"].squeeze(0), target, beta=self.huber_delta)
        entropy = torch.cat(entropy_parts).mean()
        return policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy, policy_loss, value_loss, entropy, kl, clip

    def evaluate_critic_targets(self, states, edge_states, targets, station_ids):
        predictions = np.asarray([self.get_value(s, e) for s, e in zip(states, edge_states)], dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        result = {"value_mean": float(predictions.mean()) if len(predictions) else 0.0}
        scaled_losses = []
        for station in sorted(set(station_ids)):
            index = np.flatnonzero(np.asarray(station_ids) == station)
            station_targets, station_predictions = targets[index], predictions[index]
            scale = max(float(station_targets.std()), 1.0)
            errors = (station_predictions - station_targets) / scale
            losses = np.where(np.abs(errors) < self.huber_delta, 0.5 * errors ** 2,
                              self.huber_delta * (np.abs(errors) - 0.5 * self.huber_delta))
            scaled_losses.extend(losses.tolist())
            prefix = f"station_{station}"
            result[f"{prefix}_td_huber"] = float(np.mean(losses)) if len(losses) else 0.0
            result[f"{prefix}_explained_variance"] = self._explained_variance(station_targets, station_predictions)
        result["normalized_td_huber_loss"] = float(np.mean(scaled_losses)) if scaled_losses else 0.0
        return result

    def update_policy(self, rewards, next_states, dones, **kwargs):
        names = ("states", "actions", "edge_states", "next_edge_states", "available_migs", "policy_infos", "station_ids", "epochs")
        data = {name: kwargs.get(name) for name in names}
        if any(data[name] is None for name in names) or not len(data["states"]):
            return {}
        versions = kwargs.get("policy_versions", [self.policy_version] * len(data["states"]))
        if set(map(int, versions)) != {self.policy_version}:
            raise ValueError("PPO batch must contain exactly the current policy_version")
        trajectory_ids = kwargs.get("trajectory_ids")
        advantages, returns, td_targets, td_residuals = self._compute_gae(
            rewards, next_states, dones, data["edge_states"], data["next_edge_states"], data["policy_infos"],
            data["station_ids"], data["epochs"], trajectory_ids)
        count = len(data["states"])
        diagnostics = {key: [] for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                                            "grad_norm_pre", "grad_norm_post")}
        module_pre = {"encoder": [], "value_head": [], "decoder": [], "split_head": []}
        module_post = {key: [] for key in module_pre}
        target_before = [p.detach().clone() for p in list(self.target_encoder.parameters()) + list(self.target_value_head.parameters())]
        epochs_run, early_stop = 0, False
        indices = np.arange(count)
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            self.optimizer.zero_grad(set_to_none=True)
            epoch_metrics = {key: [] for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")}
            for start in range(0, count, self.minibatch_size):
                microbatch = indices[start:start + self.minibatch_size]
                terms = [self._sample_terms(i, data["states"], data["actions"], data["edge_states"], data["available_migs"],
                                            data["policy_infos"], advantages, td_targets) for i in microbatch]
                (torch.stack([item[0] for item in terms]).sum() / count).backward()
                for key, position in zip(epoch_metrics, range(1, 6)):
                    epoch_metrics[key].extend(float(item[position].detach()) for item in terms)
            groups = {"encoder": list(self.encoder.parameters()), "value_head": list(self.value_head.parameters()),
                      "decoder": list(self.device_decoder.parameters()), "split_head": list(self.split_head.parameters())}
            for key, params in groups.items(): module_pre[key].append(self._gradient_norm(params))
            pre = float(torch.nn.utils.clip_grad_norm_(list(self._parameters()), self.max_grad_norm))
            post = self._gradient_norm(list(self._parameters()))
            for key, params in groups.items(): module_post[key].append(self._gradient_norm(params))
            self.optimizer.step()
            epochs_run += 1
            for key, values in epoch_metrics.items(): diagnostics[key].append(float(np.mean(values)))
            diagnostics["grad_norm_pre"].append(pre); diagnostics["grad_norm_post"].append(post)
            if diagnostics["approx_kl"][-1] > self.target_kl:
                early_stop = True
                break
        target_drift_during_update = max((float((a - b).abs().max()) for a, b in zip(
            target_before, list(self.target_encoder.parameters()) + list(self.target_value_head.parameters()))), default=0.0)
        updated = np.asarray([self.get_value(s, e) for s, e in zip(data["states"], data["edge_states"])])
        target_values = np.asarray([self._target_value(s, e) for s, e in zip(data["states"], data["edge_states"])])
        self.update_count += 1
        self.policy_version += 1
        self.sync_target()
        result = {key: float(np.mean(values)) for key, values in diagnostics.items()}
        result.update({"grad_norm": result["grad_norm_pre"], "grad_norm_pre_max": float(max(diagnostics["grad_norm_pre"])),
                       "grad_norm_post_max": float(max(diagnostics["grad_norm_post"])), "ppo_epochs_run": epochs_run,
                       "kl_early_stop": bool(early_stop), "episode_count": len({tuple(item)[0] for item in trajectory_ids}) if trajectory_ids else 1,
                       "transition_count": count, "policy_version_count": len(set(versions)),
                       "target_drift_during_update": target_drift_during_update})
        for key in module_pre:
            result[f"{key}_grad_norm_pre"] = float(np.mean(module_pre[key])); result[f"{key}_grad_norm_post"] = float(np.mean(module_post[key]))
        updated = np.asarray([self.get_value(s, e) for s, e in zip(data["states"], data["edge_states"])])
        target_values = np.asarray([self._target_value(s, e) for s, e in zip(data["states"], data["edge_states"])])
        result["online_value_mean"] = float(updated.mean()); result["target_value_mean"] = float(target_values.mean())
        result["online_target_value_drift"] = float(np.mean(np.abs(updated - target_values)))
        station_array = np.asarray(data["station_ids"])
        for station in sorted(set(data["station_ids"])):
            index = np.flatnonzero(station_array == station); prefix = f"station_{station}"
            result[f"{prefix}_return_mean"] = float(returns[index].mean()); result[f"{prefix}_return_std"] = float(returns[index].std())
            result[f"{prefix}_td_residual_mean"] = float(td_residuals[index].mean())
            result[f"{prefix}_td_residual_std"] = float(td_residuals[index].std())
            result[f"{prefix}_explained_variance"] = self._explained_variance(td_targets[index], updated[index])
        if not all(np.isfinite(float(value)) for value in result.values() if isinstance(value, (int, float, np.number))):
            raise FloatingPointError("non-finite MAT diagnostic")
        return result