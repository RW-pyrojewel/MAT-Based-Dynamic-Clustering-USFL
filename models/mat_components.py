"""Neural building blocks for the hierarchical multi-agent Transformer policy."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.distributions import Categorical, Dirichlet, Normal


class HeterogeneousEncoder(nn.Module):
    """Encode client tokens together with the normalized edge-resource token."""

    def __init__(self, state_dim, hidden_dim, edge_state_dim=2, num_heads=4, num_layers=2):
        super().__init__()
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_state_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim * 4, batch_first=True, activation="gelu", norm_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, client_states, edge_state, padding_mask=None):
        if edge_state.ndim != 2 or edge_state.shape[0] != client_states.shape[0]:
            raise ValueError("edge_state must have shape (batch_size, edge_state_dim)")
        client_tokens = functional.gelu(self.state_embed(client_states))
        edge_token = functional.gelu(self.edge_embed(edge_state)).unsqueeze(1)
        tokens = torch.cat((edge_token, client_tokens), dim=1)
        if padding_mask is not None:
            edge_mask = torch.zeros((padding_mask.shape[0], 1), dtype=torch.bool, device=padding_mask.device)
            padding_mask = torch.cat((edge_mask, padding_mask), dim=1)
        return self.transformer(tokens, src_key_padding_mask=padding_mask)[:, 1:]


class CausalDeviceDecoder(nn.Module):
    """Generate per-device MIG and bandwidth actions from a causal action prefix."""

    _EPS = 1e-6

    def __init__(self, hidden_dim, num_migs, num_heads=4, num_layers=2, min_bandwidth_share=0.01, initial_bandwidth_log_std=-1.5):
        super().__init__()
        if not 0.0 <= min_bandwidth_share < 1.0:
            raise ValueError("min_bandwidth_share must be in [0, 1)")
        self.hidden_dim = int(hidden_dim)
        self.num_migs = int(num_migs)
        self.min_bandwidth_share = float(min_bandwidth_share)
        self.start_token = nn.Parameter(torch.zeros(hidden_dim))
        self.cluster_embedding = nn.Embedding(num_migs, hidden_dim)
        self.bandwidth_embedding = nn.Linear(1, hidden_dim)
        layer = nn.TransformerDecoderLayer(
            hidden_dim, num_heads, hidden_dim * 4, batch_first=True, activation="gelu", norm_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.cluster_head = nn.Linear(hidden_dim, num_migs)
        self.bandwidth_mean_head = nn.Linear(hidden_dim, 1)
        self.bandwidth_channel_head = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.bandwidth_mean_head.weight)
        nn.init.zeros_(self.bandwidth_mean_head.bias)
        nn.init.zeros_(self.bandwidth_channel_head.weight)
        self.bandwidth_log_std = nn.Parameter(torch.tensor(float(initial_bandwidth_log_std)))

    @staticmethod
    def _causal_mask(length, device):
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)

    def _prefix_inputs(self, ordered_states, clusters=None, bandwidths=None):
        batch_size, client_count, _ = ordered_states.shape
        shifted_actions = ordered_states.new_zeros((batch_size, client_count, self.hidden_dim))
        shifted_actions[:, 0] = self.start_token
        if client_count > 1 and clusters is not None and bandwidths is not None:
            action_tokens = self.cluster_embedding(clusters[:, :-1].long())
            action_tokens = action_tokens + self.bandwidth_embedding(bandwidths[:, :-1].unsqueeze(-1))
            shifted_actions[:, 1:] = action_tokens
        return ordered_states + shifted_actions

    def _decode_prefix(self, ordered_states, clusters=None, bandwidths=None):
        target = self._prefix_inputs(ordered_states, clusters, bandwidths)
        decoded = self.transformer(target, ordered_states, tgt_mask=self._causal_mask(target.shape[1], target.device))
        return self.output_norm(decoded)

    def _cluster_dist(self, features, available_migs):
        if not 1 <= int(available_migs) <= self.num_migs:
            raise ValueError("available_migs must be in [1, num_migs]")
        logits = self.cluster_head(features)
        valid = torch.arange(self.num_migs, device=features.device) < int(available_migs)
        return Categorical(logits=logits.masked_fill(~valid, -torch.inf))

    def act_clusters(self, ordered_states, available_migs, deterministic=False):
        """Generate clusters before bandwidth, conditioned only on the cluster prefix."""
        batch_size, client_count, _ = ordered_states.shape
        clusters = torch.zeros((batch_size, client_count), dtype=torch.long, device=ordered_states.device)
        log_probs = ordered_states.new_zeros((batch_size, client_count))
        entropies = ordered_states.new_zeros((batch_size, client_count))
        empty_bandwidths = ordered_states.new_zeros((batch_size, client_count))
        for index in range(client_count):
            decoded = self._decode_prefix(ordered_states, clusters, empty_bandwidths)
            distribution = self._cluster_dist(decoded[:, index], available_migs)
            cluster = distribution.probs.argmax(-1) if deterministic else distribution.sample()
            clusters[:, index] = cluster
            log_probs[:, index] = distribution.log_prob(cluster)
            entropies[:, index] = distribution.entropy()
        return clusters, log_probs, entropies

    def evaluate_clusters(self, ordered_states, clusters, available_migs):
        empty_bandwidths = ordered_states.new_zeros(clusters.shape)
        decoded = self._decode_prefix(ordered_states, clusters, empty_bandwidths)
        log_probs, entropies = [], []
        for index in range(ordered_states.shape[1]):
            distribution = self._cluster_dist(decoded[:, index], available_migs)
            log_probs.append(distribution.log_prob(clusters[:, index].long()))
            entropies.append(distribution.entropy())
        return torch.stack(log_probs, dim=1), torch.stack(entropies, dim=1)
    def _bandwidth_dist(self, features, channel_features=None):
        mean = self.bandwidth_mean_head(features).squeeze(-1)
        if channel_features is not None:
            mean = mean + self.bandwidth_channel_head(channel_features).squeeze(-1)
        return Normal(mean, self.bandwidth_log_std.exp().clamp(1e-3, 1.0))

    def _allocation_parameters(self, remaining, remaining_clients):
        center = remaining / float(remaining_clients)
        lower = remaining.new_full(remaining.shape, self.min_bandwidth_share)
        upper = remaining - float(remaining_clients - 1) * self.min_bandwidth_share
        if torch.any(upper < lower - self._EPS):
            raise ValueError("minimum bandwidth share is infeasible for this client count")
        half_width = torch.minimum(center - lower, upper - center).clamp_min(0.0)
        return center, half_width

    @classmethod
    def _transform_bandwidth(cls, latent, center, half_width):
        return center + half_width * torch.tanh(latent)

    @classmethod
    def _bandwidth_log_prob(cls, distribution, allocation, center, half_width):
        if torch.all(half_width <= cls._EPS):
            return allocation.new_zeros(allocation.shape)
        normalized = ((allocation - center) / half_width.clamp_min(cls._EPS)).clamp(-1.0 + cls._EPS, 1.0 - cls._EPS)
        latent = torch.atanh(normalized)
        jacobian = half_width.clamp_min(cls._EPS) * (1.0 - normalized.square()).clamp_min(cls._EPS)
        return distribution.log_prob(latent) - torch.log(jacobian)

    def act(self, ordered_states, available_migs, deterministic=False, channel_features=None, return_diagnostics=False):
        batch_size, client_count, _ = ordered_states.shape
        if channel_features is not None and channel_features.shape != (batch_size, client_count, 1):
            raise ValueError("channel_features must have shape (B, N, 1)")
        if client_count * self.min_bandwidth_share >= 1.0:
            raise ValueError("minimum bandwidth share is infeasible for the active client count")
        clusters = torch.zeros((batch_size, client_count), dtype=torch.long, device=ordered_states.device)
        bandwidths = ordered_states.new_zeros((batch_size, client_count))
        cluster_log_probs = ordered_states.new_zeros((batch_size, client_count))
        bandwidth_log_probs = ordered_states.new_zeros((batch_size, client_count))
        cluster_entropies = ordered_states.new_zeros((batch_size, client_count))
        bandwidth_entropies = ordered_states.new_zeros((batch_size, client_count))
        bandwidth_means = ordered_states.new_zeros((batch_size, client_count))
        remaining = ordered_states.new_ones(batch_size)
        for index in range(client_count):
            decoded = self._decode_prefix(ordered_states, clusters, bandwidths)
            features = decoded[:, index]
            cluster_dist = self._cluster_dist(features, available_migs)
            cluster = cluster_dist.probs.argmax(-1) if deterministic else cluster_dist.sample()
            remaining_clients = client_count - index
            if remaining_clients == 1:
                bandwidth = remaining
                bandwidth_log_prob = remaining.new_zeros(remaining.shape)
                bandwidth_entropy = remaining.new_zeros(remaining.shape)
            else:
                center, half_width = self._allocation_parameters(remaining, remaining_clients)
                current_channel = None if channel_features is None else channel_features[:, index]
                bandwidth_dist = self._bandwidth_dist(features, current_channel)
                latent = bandwidth_dist.mean if deterministic else bandwidth_dist.rsample()
                bandwidth = self._transform_bandwidth(latent, center, half_width)
                bandwidth_log_prob = self._bandwidth_log_prob(bandwidth_dist, bandwidth, center, half_width)
                bandwidth_entropy = -bandwidth_log_prob
            clusters[:, index] = cluster
            bandwidths[:, index] = bandwidth
            cluster_log_probs[:, index] = cluster_dist.log_prob(cluster)
            bandwidth_log_probs[:, index] = bandwidth_log_prob
            cluster_entropies[:, index] = cluster_dist.entropy()
            bandwidth_entropies[:, index] = bandwidth_entropy
            if remaining_clients > 1:
                bandwidth_means[:, index] = bandwidth_dist.mean
            remaining = (remaining - bandwidth).clamp_min(0.0)
        device_entropies = cluster_entropies + bandwidth_entropies
        base = (clusters, bandwidths, cluster_log_probs, bandwidth_log_probs, device_entropies)
        return base + (bandwidth_means, cluster_entropies, bandwidth_entropies) if return_diagnostics else base

    def evaluate_actions(self, ordered_states, clusters, bandwidths, available_migs, channel_features=None, return_diagnostics=False):
        batch_size, client_count, _ = ordered_states.shape
        if channel_features is not None and channel_features.shape != (batch_size, client_count, 1):
            raise ValueError("channel_features must have shape (B, N, 1)")
        if clusters.shape != (batch_size, client_count) or bandwidths.shape != (batch_size, client_count):
            raise ValueError("clusters and bandwidths must match ordered_states")
        decoded = self._decode_prefix(ordered_states, clusters, bandwidths)
        cluster_log_probs = ordered_states.new_zeros((batch_size, client_count))
        bandwidth_log_probs = ordered_states.new_zeros((batch_size, client_count))
        cluster_entropies = ordered_states.new_zeros((batch_size, client_count))
        bandwidth_entropies = ordered_states.new_zeros((batch_size, client_count))
        bandwidth_means = ordered_states.new_zeros((batch_size, client_count))
        remaining = ordered_states.new_ones(batch_size)
        for index in range(client_count):
            features = decoded[:, index]
            cluster_dist = self._cluster_dist(features, available_migs)
            cluster = clusters[:, index].long()
            bandwidth = bandwidths[:, index]
            remaining_clients = client_count - index
            if remaining_clients == 1:
                if torch.any((bandwidth - remaining).abs() > 1e-4):
                    raise ValueError("the final client must receive all remaining bandwidth")
                bandwidth_log_prob = remaining.new_zeros(remaining.shape)
                bandwidth_entropy = remaining.new_zeros(remaining.shape)
            else:
                center, half_width = self._allocation_parameters(remaining, remaining_clients)
                lower, upper = center - half_width, center + half_width
                if torch.any(bandwidth < lower - 1e-5) or torch.any(bandwidth > upper + 1e-5):
                    raise ValueError("bandwidth allocation is outside its feasible interval")
                current_channel = None if channel_features is None else channel_features[:, index]
                bandwidth_dist = self._bandwidth_dist(features, current_channel)
                bandwidth_log_prob = self._bandwidth_log_prob(bandwidth_dist, bandwidth, center, half_width)
                bandwidth_entropy = -bandwidth_log_prob
            cluster_log_probs[:, index] = cluster_dist.log_prob(cluster)
            bandwidth_log_probs[:, index] = bandwidth_log_prob
            cluster_entropies[:, index] = cluster_dist.entropy()
            bandwidth_entropies[:, index] = bandwidth_entropy
            if remaining_clients > 1:
                bandwidth_means[:, index] = bandwidth_dist.mean
            remaining = (remaining - bandwidth).clamp_min(0.0)
        device_entropies = cluster_entropies + bandwidth_entropies
        base = (cluster_log_probs, bandwidth_log_probs, device_entropies)
        return base + (bandwidth_means, cluster_entropies, bandwidth_entropies) if return_diagnostics else base


class JointDirichletBandwidthHead(nn.Module):
    """Permutation-equivariant simplex policy with an isolated physical CSI path."""

    def __init__(self, hidden_dim, num_migs, min_bandwidth_share=0.01,
                 alpha_floor=1.0, alpha_init=22.5):
        super().__init__()
        if alpha_floor <= 0.0 or alpha_init <= alpha_floor:
            raise ValueError("Dirichlet alpha_init must exceed a positive alpha_floor")
        self.min_bandwidth_share = float(min_bandwidth_share)
        self.alpha_floor = float(alpha_floor)
        self.alpha_init = float(alpha_init)
        self.context_head = nn.Linear(hidden_dim, 1)
        self.cluster_embedding = nn.Embedding(num_migs, hidden_dim)
        self.cluster_head = nn.Linear(hidden_dim, 1, bias=False)
        self.physical_weight = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.context_head.weight)
        nn.init.zeros_(self.context_head.bias)
        nn.init.zeros_(self.cluster_head.weight)
        initial_softplus = self.alpha_init - self.alpha_floor
        self.register_buffer("concentration_bias", torch.tensor(math.log(math.expm1(initial_softplus))))

    @staticmethod
    def physical_feature(normalized_channel):
        feature = -torch.log(normalized_channel.squeeze(-1).clamp_min(1e-6)).clamp(-8.0, 8.0)
        return feature - feature.mean(dim=1, keepdim=True)

    def parameters_for(self, context, clusters, normalized_channel):
        if context.shape[:2] != clusters.shape or normalized_channel.shape != (*clusters.shape, 1):
            raise ValueError("bandwidth context, clusters and channel must share (B, N)")
        context_score = self.context_head(context).squeeze(-1)
        cluster_score = self.cluster_head(self.cluster_embedding(clusters.long())).squeeze(-1)
        physical_feature = self.physical_feature(normalized_channel)
        physical_score = self.physical_weight * physical_feature
        score = context_score + cluster_score + physical_score
        # Separate the allocation mean from exploration concentration.  A symmetric
        # score still yields alpha_init per client, while score differences act
        # directly on the simplex mean instead of being divided by alpha_init.
        mean = torch.softmax(score, dim=-1)
        excess_concentration = float(score.shape[-1]) * self.concentration_bias
        alpha = self.alpha_floor + excess_concentration * mean
        return alpha, context_score + cluster_score, physical_score, physical_feature

    def _scale(self, client_count):
        scale = 1.0 - float(client_count) * self.min_bandwidth_share
        if scale <= 0.0:
            raise ValueError("minimum bandwidth share is infeasible for the active client count")
        return scale

    def _to_bandwidth(self, simplex):
        return self.min_bandwidth_share + self._scale(simplex.shape[-1]) * simplex

    def _to_simplex(self, bandwidth):
        simplex = (bandwidth - self.min_bandwidth_share) / self._scale(bandwidth.shape[-1])
        if torch.any(simplex <= 0.0) or torch.any(~torch.isfinite(simplex)):
            raise ValueError("bandwidth allocation is outside the open constrained simplex")
        return simplex / simplex.sum(dim=-1, keepdim=True)

    def _log_prob_entropy(self, distribution, simplex):
        dimension = simplex.shape[-1] - 1
        log_scale = math.log(self._scale(simplex.shape[-1]))
        return (distribution.log_prob(simplex) - dimension * log_scale,
                distribution.entropy() + dimension * log_scale)

    def act(self, context, clusters, normalized_channel, deterministic=False):
        alpha, context_score, physical_score, physical_feature = self.parameters_for(
            context, clusters, normalized_channel)
        distribution = Dirichlet(alpha)
        simplex = alpha / alpha.sum(dim=-1, keepdim=True) if deterministic else distribution.rsample()
        bandwidth = self._to_bandwidth(simplex)
        log_prob, entropy = self._log_prob_entropy(distribution, simplex)
        return bandwidth, log_prob, entropy, alpha, context_score, physical_score, physical_feature

    def evaluate_actions(self, context, clusters, normalized_channel, bandwidth):
        alpha, context_score, physical_score, physical_feature = self.parameters_for(
            context, clusters, normalized_channel)
        distribution = Dirichlet(alpha)
        simplex = self._to_simplex(bandwidth)
        log_prob, entropy = self._log_prob_entropy(distribution, simplex)
        return log_prob, entropy, alpha, context_score, physical_score, physical_feature

class ClusterSplitHead(nn.Module):
    """Choose one shared U-shaped split pair for each non-empty cluster."""

    def __init__(self, hidden_dim, num_migs, num_cut_layers, num_heads=4):
        super().__init__()
        if num_cut_layers < 2:
            raise ValueError("num_cut_layers must be at least two")
        self.num_migs = int(num_migs)
        self.num_cut_layers = int(num_cut_layers)
        self.cluster_tokens = nn.Parameter(torch.empty(num_migs, hidden_dim))
        nn.init.normal_(self.cluster_tokens, std=1.0 / math.sqrt(hidden_dim))
        self.resource_embed = nn.Linear(2, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.l1_head = nn.Linear(hidden_dim, num_cut_layers - 1)
        self.l2_head = nn.Linear(hidden_dim, num_cut_layers)

    def _cluster_context(self, encoded_states, clusters, bandwidths, mig_id):
        members = clusters.eq(mig_id)
        member_count = members.sum(dim=1)
        present = member_count > 0
        query = self.cluster_tokens[mig_id].reshape(1, 1, -1).expand(encoded_states.shape[0], -1, -1)
        total_bandwidth = (bandwidths * members.float()).sum(dim=1)
        resources = torch.stack((member_count.float() / float(encoded_states.shape[1]), total_bandwidth), dim=1)
        query = query + self.resource_embed(resources).unsqueeze(1)
        safe_members = members.clone()
        safe_members[~present, 0] = True
        context, _ = self.attention(query, encoded_states, encoded_states, key_padding_mask=~safe_members, need_weights=False)
        return self.output_norm(context.squeeze(1)), present

    def act(self, encoded_states, clusters, bandwidths, deterministic=False):
        batch_size, client_count, _ = encoded_states.shape
        l1_all = torch.zeros((batch_size, client_count), dtype=torch.long, device=encoded_states.device)
        l2_all = torch.ones((batch_size, client_count), dtype=torch.long, device=encoded_states.device)
        split_log_probs = encoded_states.new_zeros((batch_size, self.num_migs))
        split_entropies = encoded_states.new_zeros((batch_size, self.num_migs))
        split_mask = torch.zeros((batch_size, self.num_migs), dtype=torch.bool, device=encoded_states.device)
        for mig_id in range(self.num_migs):
            context, present = self._cluster_context(encoded_states, clusters, bandwidths, mig_id)
            if not present.any():
                continue
            l1_dist = Categorical(logits=self.l1_head(context))
            l1 = l1_dist.probs.argmax(-1) if deterministic else l1_dist.sample()
            valid_l2 = torch.arange(self.num_cut_layers, device=encoded_states.device).unsqueeze(0) > l1.unsqueeze(1)
            l2_dist = Categorical(logits=self.l2_head(context).masked_fill(~valid_l2, -torch.inf))
            l2 = l2_dist.probs.argmax(-1) if deterministic else l2_dist.sample()
            members = clusters.eq(mig_id)
            l1_all = torch.where(members, l1.unsqueeze(1), l1_all)
            l2_all = torch.where(members, l2.unsqueeze(1), l2_all)
            split_log_probs[:, mig_id] = l1_dist.log_prob(l1) + l2_dist.log_prob(l2)
            split_entropies[:, mig_id] = l1_dist.entropy() + l2_dist.entropy()
            split_mask[:, mig_id] = present
        return l1_all, l2_all, split_log_probs, split_entropies, split_mask

    def evaluate_actions(self, encoded_states, clusters, bandwidths, l1, l2):
        batch_size = encoded_states.shape[0]
        split_log_probs = encoded_states.new_zeros((batch_size, self.num_migs))
        split_entropies = encoded_states.new_zeros((batch_size, self.num_migs))
        split_mask = torch.zeros((batch_size, self.num_migs), dtype=torch.bool, device=encoded_states.device)
        for mig_id in range(self.num_migs):
            members = clusters.eq(mig_id)
            present = members.any(dim=1)
            if not present.any():
                continue
            context, _ = self._cluster_context(encoded_states, clusters, bandwidths, mig_id)
            selected_l1 = torch.where(members, l1, torch.zeros_like(l1)).max(dim=1).values
            selected_l2 = torch.where(members, l2, torch.zeros_like(l2)).max(dim=1).values
            if torch.any(torch.where(members, l1, selected_l1.unsqueeze(1)) != selected_l1.unsqueeze(1)):
                raise ValueError("all clients in a cluster must share l1")
            if torch.any(torch.where(members, l2, selected_l2.unsqueeze(1)) != selected_l2.unsqueeze(1)):
                raise ValueError("all clients in a cluster must share l2")
            l1_dist = Categorical(logits=self.l1_head(context))
            valid_l2 = torch.arange(self.num_cut_layers, device=encoded_states.device).unsqueeze(0) > selected_l1.unsqueeze(1)
            l2_dist = Categorical(logits=self.l2_head(context).masked_fill(~valid_l2, -torch.inf))
            split_log_probs[:, mig_id] = l1_dist.log_prob(selected_l1) + l2_dist.log_prob(selected_l2)
            split_entropies[:, mig_id] = l1_dist.entropy() + l2_dist.entropy()
            split_mask[:, mig_id] = present
        return split_log_probs, split_entropies, split_mask


AutoregressiveDecoder = CausalDeviceDecoder

