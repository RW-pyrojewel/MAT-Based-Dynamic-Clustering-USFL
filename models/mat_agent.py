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
                 target_kl=0.03, channel_conditioning="explicit", component_balanced_ppo=False,
                 compute_reference=2.5, snr_scale=10.0, device="cpu"):
        super().__init__(agent_name="MAT-RL Agent (Proposed)")
        if num_cut_layers < 2 or nominal_bandwidth_hz <= 0:
            raise ValueError("invalid MAT dimensions")
        if min(actor_learning_rate, critic_learning_rate, max_grad_norm, huber_delta) <= 0:
            raise ValueError("learning rates and loss/gradient limits must be positive")
        if channel_conditioning not in {"legacy", "explicit"}:
            raise ValueError("channel_conditioning must be legacy or explicit")
        if compute_reference <= 0.0 or snr_scale <= 0.0:
            raise ValueError("channel/compute normalization references must be positive")
        self.device = torch.device(device)
        self.num_migs, self.num_cut_layers = int(num_migs), int(num_cut_layers)
        self.min_bandwidth_share = float(min_bandwidth_share)
        self.nominal_bandwidth_hz = float(nominal_bandwidth_hz)
        self.gamma, self.gae_lambda = float(gamma), float(gae_lambda)
        self.ppo_epochs, self.minibatch_size = int(ppo_epochs), int(minibatch_size)
        self.clip_ratio, self.value_coef, self.entropy_coef = 0.2, 0.5, 0.01
        self.max_grad_norm, self.huber_delta, self.target_kl = float(max_grad_norm), float(huber_delta), float(target_kl)
        self.channel_conditioning = channel_conditioning
        self.component_balanced_ppo = bool(component_balanced_ppo)
        self.compute_reference, self.snr_scale = float(compute_reference), float(snr_scale)
        self.state_dim, self.hidden_dim, self.edge_state_dim = int(state_dim), int(hidden_dim), int(edge_state_dim)
        self.actor_learning_rate, self.critic_learning_rate = float(actor_learning_rate), float(critic_learning_rate)
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
        self._checkpoint_config = {
            "state_dim": self.state_dim, "hidden_dim": self.hidden_dim, "num_migs": self.num_migs,
            "num_cut_layers": self.num_cut_layers, "edge_state_dim": self.edge_state_dim,
            "min_bandwidth_share": self.min_bandwidth_share, "nominal_bandwidth_hz": self.nominal_bandwidth_hz,
            "gamma": self.gamma, "gae_lambda": self.gae_lambda, "ppo_epochs": self.ppo_epochs,
            "minibatch_size": self.minibatch_size, "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate, "max_grad_norm": self.max_grad_norm,
            "huber_delta": self.huber_delta, "target_kl": self.target_kl,
            "channel_conditioning": self.channel_conditioning,
            "component_balanced_ppo": self.component_balanced_ppo,
            "compute_reference": self.compute_reference, "snr_scale": self.snr_scale,
        }

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

    def save_checkpoint(self, path):
        payload = {
            "schema_version": 1,
            "config": self._checkpoint_config,
            "encoder": self.encoder.state_dict(),
            "device_decoder": self.device_decoder.state_dict(),
            "split_head": self.split_head.state_dict(),
            "value_head": self.value_head.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "target_value_head": self.target_value_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "policy_version": self.policy_version,
            "update_count": self.update_count,
        }
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path, device="cpu", load_optimizer=True):
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported MAT checkpoint schema")
        agent = cls(**payload["config"], device=device)
        for name in ("encoder", "device_decoder", "split_head", "value_head", "target_encoder", "target_value_head"):
            getattr(agent, name).load_state_dict(payload[name])
        if load_optimizer:
            agent.optimizer.load_state_dict(payload["optimizer"])
        agent.policy_version = int(payload["policy_version"])
        agent.update_count = int(payload["update_count"])
        agent._freeze_targets()
        return agent

    def _prepare_client_state(self, state):
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[-1] != self.state_dim:
            raise ValueError("client state must have shape (N, state_dim) or (B, N, state_dim)")
        prepared = tensor.clone()
        if self.channel_conditioning == "explicit":
            prepared[..., 0] = torch.log1p(self.snr_scale * prepared[..., 0].clamp_min(0.0)) / np.log1p(self.snr_scale)
            prepared[..., 1] = prepared[..., 1] / self.compute_reference
        return prepared

    def _channel_features(self, prepared_state):
        if self.channel_conditioning == "legacy":
            return None
        return prepared_state[..., :1]
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
        state_tensor = self._prepare_client_state(state)
        return state_tensor, self.encoder(state_tensor, self._normalise_edge_state(edge_state))

    @torch.no_grad()
    def _target_value(self, state, edge_state):
        state_tensor = self._prepare_client_state(state)
        encoded = self.target_encoder(state_tensor, self._normalise_edge_state(edge_state))
        return float(self._aggregate_token_values(self.target_value_head, encoded).item())

    @torch.no_grad()
    def act(self, active_clients_state, available_migs, edge_state, client_ids=None, deterministic=False):
        prepared_state, encoded = self._encode(active_clients_state, edge_state)
        client_count = encoded.shape[1]
        order = self._decision_order(client_count, client_ids, deterministic, self.device)
        ordered_encoded = encoded[:, order]
        channel_features = self._channel_features(prepared_state)
        ordered_channel = None if channel_features is None else channel_features[:, order]
        (ordered_clusters, ordered_bandwidths, cluster_lp, bandwidth_lp, device_entropy,
         bandwidth_means, cluster_entropy, bandwidth_entropy) = self.device_decoder.act(
            ordered_encoded, available_migs, deterministic=deterministic,
            channel_features=ordered_channel, return_diagnostics=True)
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
            "cluster_entropy": cluster_entropy.squeeze(0).cpu().numpy(),
            "bandwidth_entropy": bandwidth_entropy.squeeze(0).cpu().numpy(),
            "bandwidth_latent_means": bandwidth_means.squeeze(0).cpu().numpy(),
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

    @torch.no_grad()
    def evaluate_bandwidth_prefix_means(
        self, state, edge_state, action, available_migs, decision_order,
    ):
        """Replay a fixed action prefix and return bandwidth means in client order."""
        evaluated = self._evaluate_action(
            state, edge_state, action, available_migs, decision_order,
        )
        order = torch.as_tensor(decision_order, dtype=torch.long, device=self.device)
        ordered_means = evaluated["bandwidth_latent_means"].reshape(1, -1)
        return self._restore_original_order(ordered_means, order).squeeze(0).cpu().numpy()

    def _evaluate_action(self, state, edge_state, action, available_migs, decision_order):
        prepared_state, encoded = self._encode(state, edge_state)
        order = torch.as_tensor(decision_order, dtype=torch.long, device=self.device)
        clusters = torch.as_tensor(action["cluster"], dtype=torch.long, device=self.device).reshape(1, -1)
        bandwidths = torch.as_tensor(action["bw"], dtype=torch.float32, device=self.device).reshape(1, -1)
        l1 = torch.as_tensor(action["l1"], dtype=torch.long, device=self.device).reshape(1, -1)
        l2 = torch.as_tensor(action["l2"], dtype=torch.long, device=self.device).reshape(1, -1)
        channel_features = self._channel_features(prepared_state)
        ordered_channel = None if channel_features is None else channel_features[:, order]
        (cluster_lp, bandwidth_lp, device_entropy, bandwidth_means,
         cluster_entropy, bandwidth_entropy) = self.device_decoder.evaluate_actions(
            encoded[:, order], clusters[:, order], bandwidths[:, order], available_migs,
            channel_features=ordered_channel, return_diagnostics=True)
        split_lp, split_entropy, split_mask = self.split_head.evaluate_actions(encoded, clusters, bandwidths, l1, l2)
        return {"cluster_log_probs": cluster_lp.squeeze(0), "bandwidth_log_probs": bandwidth_lp.squeeze(0),
                "split_log_probs": split_lp.squeeze(0), "split_mask": split_mask.squeeze(0),
                "device_entropy": device_entropy.squeeze(0), "cluster_entropy": cluster_entropy.squeeze(0),
                "bandwidth_entropy": bandwidth_entropy.squeeze(0),
                "bandwidth_latent_means": bandwidth_means.squeeze(0),
                "split_entropy": split_entropy.squeeze(0),
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
    def _autograd_gradient_norm(loss, parameters):
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        values = [gradient.detach().float().norm(2).square() for gradient in gradients if gradient is not None]
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
        bandwidth_mask = torch.as_tensor(info["bandwidth_mask"], dtype=torch.bool, device=self.device)
        split_mask = torch.as_tensor(info["split_mask"], dtype=torch.bool, device=self.device)
        advantage = torch.as_tensor(advantages[index], dtype=torch.float32, device=self.device)
        component_pairs = {
            "cluster": (
                evaluated["cluster_log_probs"],
                torch.as_tensor(info["cluster_log_probs"], dtype=torch.float32, device=self.device),
            ),
        }
        if bandwidth_mask.any():
            component_pairs["bandwidth"] = (
                evaluated["bandwidth_log_probs"][bandwidth_mask],
                torch.as_tensor(info["bandwidth_log_probs"], dtype=torch.float32, device=self.device)[bandwidth_mask],
            )
        if split_mask.any():
            component_pairs["split"] = (
                evaluated["split_log_probs"][split_mask],
                torch.as_tensor(info["split_log_probs"], dtype=torch.float32, device=self.device)[split_mask],
            )
        component_results = {
            name: self._component_loss(new, old, advantage, self.clip_ratio)
            for name, (new, old) in component_pairs.items()
        }
        if self.component_balanced_ppo:
            policy_loss = torch.stack([value[0] for value in component_results.values()]).mean()
            kl = torch.stack([value[1] for value in component_results.values()]).mean()
            clip = torch.stack([value[2] for value in component_results.values()]).mean()
        else:
            new_joint = torch.cat([value[0] for value in component_pairs.values()])
            old_joint = torch.cat([value[1] for value in component_pairs.values()])
            policy_loss, kl, clip = self._component_loss(new_joint, old_joint, advantage, self.clip_ratio)
        target = torch.as_tensor(td_targets[index], dtype=torch.float32, device=self.device)
        value_loss = functional.smooth_l1_loss(evaluated["value"].squeeze(0), target, beta=self.huber_delta)
        entropy_parts = [evaluated["device_entropy"]]
        if split_mask.any():
            entropy_parts.append(evaluated["split_entropy"][split_mask])
        entropy = torch.cat(entropy_parts).mean()
        zero = policy_loss.detach() * 0.0
        output = {
            "total_loss": policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy,
            "policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy,
            "approx_kl": kl, "clip_fraction": clip,
            "cluster_entropy": evaluated["cluster_entropy"].mean(),
            "bandwidth_entropy": evaluated["bandwidth_entropy"][bandwidth_mask].mean() if bandwidth_mask.any() else zero,
            "split_entropy_component": evaluated["split_entropy"][split_mask].mean() if split_mask.any() else zero,
        }
        for name in ("cluster", "bandwidth", "split"):
            result = component_results.get(name)
            output[f"{name}_policy_loss"] = result[0] if result is not None else zero
            output[f"{name}_approx_kl"] = result[1] if result is not None else zero
            output[f"{name}_clip_fraction"] = result[2] if result is not None else zero
        return output
    def _microbatch_terms(self, indices, states, actions, edge_states, available_migs,
                          policy_infos, advantages, td_targets):
        """Evaluate fixed-size samples in one Transformer pass; fall back for mixed shapes."""
        indices = list(map(int, indices))
        counts = {np.asarray(states[index]).shape[0] for index in indices}
        mig_counts = {int(available_migs[index]) for index in indices}
        if len(indices) < 16 or len(counts) != 1 or len(mig_counts) != 1:
            return [self._sample_terms(index, states, actions, edge_states, available_migs,
                                       policy_infos, advantages, td_targets) for index in indices]
        prepared = torch.cat([self._prepare_client_state(states[index]) for index in indices], dim=0)
        edges = torch.cat([self._normalise_edge_state(edge_states[index]) for index in indices], dim=0)
        encoded = self.encoder(prepared, edges)
        orders = torch.stack([torch.as_tensor(policy_infos[index]["decision_order"], dtype=torch.long,
                                               device=self.device) for index in indices])
        ordered_encoded = torch.gather(encoded, 1, orders.unsqueeze(-1).expand(-1, -1, encoded.shape[-1]))
        clusters = torch.stack([torch.as_tensor(actions[index]["cluster"], dtype=torch.long,
                                                 device=self.device) for index in indices])
        bandwidths = torch.stack([torch.as_tensor(actions[index]["bw"], dtype=torch.float32,
                                                   device=self.device) for index in indices])
        channels = self._channel_features(prepared)
        ordered_channels = None if channels is None else torch.gather(channels, 1, orders.unsqueeze(-1))
        cluster_lp, bandwidth_lp, device_entropy, _, cluster_entropy, bandwidth_entropy = (
            self.device_decoder.evaluate_actions(
                ordered_encoded, torch.gather(clusters, 1, orders), torch.gather(bandwidths, 1, orders),
                next(iter(mig_counts)), channel_features=ordered_channels, return_diagnostics=True))
        l1 = torch.stack([torch.as_tensor(actions[index]["l1"], dtype=torch.long,
                                           device=self.device) for index in indices])
        l2 = torch.stack([torch.as_tensor(actions[index]["l2"], dtype=torch.long,
                                           device=self.device) for index in indices])
        split_lp, split_entropy, _ = self.split_head.evaluate_actions(encoded, clusters, bandwidths, l1, l2)
        values = self._aggregate_token_values(self.value_head, encoded)
        outputs = []
        for row, index in enumerate(indices):
            info = policy_infos[index]
            bandwidth_mask = torch.as_tensor(info["bandwidth_mask"], dtype=torch.bool, device=self.device)
            split_mask = torch.as_tensor(info["split_mask"], dtype=torch.bool, device=self.device)
            advantage = torch.as_tensor(advantages[index], dtype=torch.float32, device=self.device)
            pairs = {"cluster": (cluster_lp[row], torch.as_tensor(info["cluster_log_probs"], dtype=torch.float32,
                                                                    device=self.device))}
            if bandwidth_mask.any():
                pairs["bandwidth"] = (bandwidth_lp[row][bandwidth_mask],
                                      torch.as_tensor(info["bandwidth_log_probs"], dtype=torch.float32,
                                                      device=self.device)[bandwidth_mask])
            if split_mask.any():
                pairs["split"] = (split_lp[row][split_mask],
                                  torch.as_tensor(info["split_log_probs"], dtype=torch.float32,
                                                  device=self.device)[split_mask])
            results = {name: self._component_loss(new, old, advantage, self.clip_ratio)
                       for name, (new, old) in pairs.items()}
            if self.component_balanced_ppo:
                policy_loss = torch.stack([value[0] for value in results.values()]).mean()
                kl = torch.stack([value[1] for value in results.values()]).mean()
                clip = torch.stack([value[2] for value in results.values()]).mean()
            else:
                policy_loss, kl, clip = self._component_loss(
                    torch.cat([value[0] for value in pairs.values()]),
                    torch.cat([value[1] for value in pairs.values()]), advantage, self.clip_ratio)
            target = torch.as_tensor(td_targets[index], dtype=torch.float32, device=self.device)
            value_loss = functional.smooth_l1_loss(values[row], target, beta=self.huber_delta)
            entropy_parts = [device_entropy[row]]
            if split_mask.any():
                entropy_parts.append(split_entropy[row][split_mask])
            entropy = torch.cat(entropy_parts).mean()
            zero = policy_loss.detach() * 0.0
            output = {"total_loss": policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy,
                      "policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy,
                      "approx_kl": kl, "clip_fraction": clip,
                      "cluster_entropy": cluster_entropy[row].mean(),
                      "bandwidth_entropy": bandwidth_entropy[row][bandwidth_mask].mean() if bandwidth_mask.any() else zero,
                      "split_entropy_component": split_entropy[row][split_mask].mean() if split_mask.any() else zero}
            for name in ("cluster", "bandwidth", "split"):
                result = results.get(name)
                output[f"{name}_policy_loss"] = result[0] if result is not None else zero
                output[f"{name}_approx_kl"] = result[1] if result is not None else zero
                output[f"{name}_clip_fraction"] = result[2] if result is not None else zero
            outputs.append(output)
        return outputs
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
        component_names = ("cluster", "bandwidth", "split")
        metric_keys = ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"]
        for component in component_names:
            metric_keys.extend((f"{component}_policy_loss", f"{component}_approx_kl", f"{component}_clip_fraction"))
        metric_keys.extend(("cluster_entropy", "bandwidth_entropy", "split_entropy_component"))
        diagnostics = {key: [] for key in metric_keys + ["grad_norm_pre", "grad_norm_post"]}
        component_gradients = {component: [] for component in component_names}
        module_pre = {"encoder": [], "value_head": [], "decoder": [], "split_head": []}
        module_post = {key: [] for key in module_pre}
        target_before = [p.detach().clone() for p in list(self.target_encoder.parameters()) + list(self.target_value_head.parameters())]
        channel_head_before = self.device_decoder.bandwidth_channel_head.weight.detach().clone()
        bandwidth_head_before = [parameter.detach().clone() for parameter in self.device_decoder.bandwidth_mean_head.parameters()]
        actor_parameters = list(self.encoder.parameters()) + list(self.device_decoder.parameters()) + list(self.split_head.parameters())
        epochs_run, early_stop = 0, False
        indices = np.arange(count)
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            self.optimizer.zero_grad(set_to_none=True)
            epoch_metrics = {key: [] for key in metric_keys}
            for start in range(0, count, self.minibatch_size):
                microbatch = indices[start:start + self.minibatch_size]
                terms = self._microbatch_terms(
                    microbatch, data["states"], data["actions"], data["edge_states"], data["available_migs"],
                    data["policy_infos"], advantages, td_targets)
                if start == 0:
                    for component in component_names:
                        component_loss = torch.stack([item[f"{component}_policy_loss"] for item in terms]).mean()
                        component_gradients[component].append(
                            self._autograd_gradient_norm(component_loss, actor_parameters))
                (torch.stack([item["total_loss"] for item in terms]).sum() / count).backward()
                for key in metric_keys:
                    epoch_metrics[key].extend(float(item[key].detach()) for item in terms)
            groups = {"encoder": list(self.encoder.parameters()), "value_head": list(self.value_head.parameters()),
                      "decoder": list(self.device_decoder.parameters()), "split_head": list(self.split_head.parameters())}
            for key, parameters in groups.items():
                module_pre[key].append(self._gradient_norm(parameters))
            pre = float(torch.nn.utils.clip_grad_norm_(list(self._parameters()), self.max_grad_norm))
            post = self._gradient_norm(list(self._parameters()))
            for key, parameters in groups.items():
                module_post[key].append(self._gradient_norm(parameters))
            self.optimizer.step()
            epochs_run += 1
            for key, values in epoch_metrics.items():
                diagnostics[key].append(float(np.mean(values)))
            diagnostics["grad_norm_pre"].append(pre)
            diagnostics["grad_norm_post"].append(post)
            if diagnostics["approx_kl"][-1] > self.target_kl:
                early_stop = True
                break
        target_drift_during_update = max((float((before - after).abs().max()) for before, after in zip(
            target_before, list(self.target_encoder.parameters()) + list(self.target_value_head.parameters()))), default=0.0)
        updated_values = np.asarray([self.get_value(state, edge) for state, edge in zip(data["states"], data["edge_states"])])
        old_target_values = np.asarray([self._target_value(state, edge) for state, edge in zip(data["states"], data["edge_states"])])
        self.update_count += 1
        self.policy_version += 1
        self.sync_target()
        result = {key: float(np.mean(values)) for key, values in diagnostics.items()}
        result.update({
            "grad_norm": result["grad_norm_pre"],
            "grad_norm_pre_max": float(max(diagnostics["grad_norm_pre"])),
            "grad_norm_post_max": float(max(diagnostics["grad_norm_post"])),
            "ppo_epochs_run": epochs_run, "kl_early_stop": bool(early_stop),
            "episode_count": len({tuple(item)[0] for item in trajectory_ids}) if trajectory_ids else 1,
            "transition_count": count, "policy_version_count": len(set(versions)),
            "target_drift_during_update": target_drift_during_update,
            "component_balanced_ppo": self.component_balanced_ppo,
        })
        for key in module_pre:
            result[f"{key}_grad_norm_pre"] = float(np.mean(module_pre[key]))
            result[f"{key}_grad_norm_post"] = float(np.mean(module_post[key]))
        gradient_total = sum(float(np.mean(component_gradients[name])) for name in component_names)
        for component in component_names:
            result[f"{component}_policy_grad_norm"] = float(np.mean(component_gradients[component]))
        result["bandwidth_actor_gradient_share"] = (
            result["bandwidth_policy_grad_norm"] / gradient_total if gradient_total > 1e-12 else 0.0)
        rollout_means = []
        for info in data["policy_infos"]:
            mask = np.asarray(info["bandwidth_mask"], dtype=bool)
            rollout_means.extend(np.asarray(info["bandwidth_latent_means"])[mask].tolist())
        result["bandwidth_latent_mean_abs"] = float(np.mean(np.abs(rollout_means))) if rollout_means else 0.0
        result["bandwidth_latent_mean_std"] = float(np.std(rollout_means)) if rollout_means else 0.0
        result["bandwidth_channel_weight"] = float(self.device_decoder.bandwidth_channel_head.weight.item())
        result["bandwidth_channel_weight_drift"] = float(
            (self.device_decoder.bandwidth_channel_head.weight.detach() - channel_head_before).abs().max())
        head_drift = [float((after.detach() - before).float().norm()) for after, before in zip(
            self.device_decoder.bandwidth_mean_head.parameters(), bandwidth_head_before)]
        result["bandwidth_mean_head_parameter_drift"] = float(np.sqrt(np.square(head_drift).sum()))
        result["online_value_mean"] = float(updated_values.mean())
        result["target_value_mean"] = float(old_target_values.mean())
        result["online_target_value_drift"] = float(np.mean(np.abs(updated_values - old_target_values)))
        station_array = np.asarray(data["station_ids"])
        for station in sorted(set(data["station_ids"])):
            index = np.flatnonzero(station_array == station)
            prefix = f"station_{station}"
            result[f"{prefix}_return_mean"] = float(returns[index].mean())
            result[f"{prefix}_return_std"] = float(returns[index].std())
            result[f"{prefix}_td_residual_mean"] = float(td_residuals[index].mean())
            result[f"{prefix}_td_residual_std"] = float(td_residuals[index].std())
            result[f"{prefix}_explained_variance"] = self._explained_variance(td_targets[index], updated_values[index])
        if not all(np.isfinite(float(value)) for value in result.values() if isinstance(value, (int, float, np.number))):
            raise FloatingPointError("non-finite MAT diagnostic")
        return result
