"""Neural building blocks for the multi-agent Transformer policy."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class HeterogeneousEncoder(nn.Module):
    """Encodes client tokens and the scheme-defined global edge token."""
    def __init__(self, state_dim, hidden_dim, edge_state_dim=2, num_heads=4, num_layers=2):
        super().__init__()
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_state_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, batch_first=True, activation="gelu", norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, client_states, edge_state, padding_mask=None):
        if edge_state.ndim != 2 or edge_state.shape[0] != client_states.shape[0]:
            raise ValueError("edge_state must have shape (batch_size, edge_state_dim)")
        client_tokens = F.gelu(self.state_embed(client_states))
        edge_token = F.gelu(self.edge_embed(edge_state)).unsqueeze(1)
        tokens = torch.cat((edge_token, client_tokens), dim=1)
        if padding_mask is not None:
            edge_mask = torch.zeros((padding_mask.shape[0], 1), dtype=torch.bool, device=padding_mask.device)
            padding_mask = torch.cat((edge_mask, padding_mask), dim=1)
        return self.transformer(tokens, src_key_padding_mask=padding_mask)[:, 1:]


class AutoregressiveDecoder(nn.Module):
    """Autoregressive device policy with exact replayable action likelihoods."""
    _EPS = 1e-6

    def __init__(self, hidden_dim, num_migs, num_cut_layers=None, num_heads=4):
        super().__init__()
        self.num_migs = num_migs
        self.context_embed = nn.Linear(num_migs * 2, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.cluster_head = nn.Linear(hidden_dim, num_migs)
        self.bandwidth_mean_head = nn.Linear(hidden_dim, 1)
        self.bandwidth_log_std = nn.Parameter(torch.tensor(-0.5))

    def _fuse(self, token, history):
        context = F.gelu(self.context_embed(history)).unsqueeze(1)
        attended, _ = self.attention(token, context, context, need_weights=False)
        return self.fusion_norm((token + attended).squeeze(1))

    def _cluster_dist(self, features, available_migs):
        if not 1 <= available_migs <= self.num_migs:
            raise ValueError("available_migs must be in [1, num_migs]")
        logits = self.cluster_head(features)
        logits[:, available_migs:] = -torch.inf
        return Categorical(logits=logits)

    def _bandwidth_dist(self, features):
        return Normal(self.bandwidth_mean_head(features).squeeze(-1), self.bandwidth_log_std.exp().clamp(1e-3, 2.0))

    @classmethod
    def _squashed_log_prob(cls, distribution, bandwidth):
        bandwidth = bandwidth.clamp(cls._EPS, 1.0 - cls._EPS)
        return distribution.log_prob(torch.logit(bandwidth)) - torch.log(bandwidth * (1.0 - bandwidth))

    def _update_history(self, history, choices, bandwidth):
        updated = history.clone()
        batch = torch.arange(history.size(0), device=history.device)
        updated[batch, choices * 2] += 1.0
        updated[batch, choices * 2 + 1] += bandwidth
        return updated

    def act(self, encoded_states, available_migs, deterministic=False):
        batch_size, client_count, _ = encoded_states.shape
        history = encoded_states.new_zeros(batch_size, self.num_migs * 2)
        choices, bandwidths, log_probs, entropies = [], [], [], []
        for index in range(client_count):
            features = self._fuse(encoded_states[:, index:index + 1], history)
            cluster_dist = self._cluster_dist(features, available_migs)
            choice = cluster_dist.probs.argmax(-1) if deterministic else cluster_dist.sample()
            bandwidth_dist = self._bandwidth_dist(features)
            latent = bandwidth_dist.mean if deterministic else bandwidth_dist.rsample()
            bandwidth = torch.sigmoid(latent)
            log_probs.append(cluster_dist.log_prob(choice) + self._squashed_log_prob(bandwidth_dist, bandwidth))
            entropies.append(cluster_dist.entropy() + bandwidth_dist.entropy())
            history = self._update_history(history, choice, bandwidth)
            choices.append(choice)
            bandwidths.append(bandwidth)
        return torch.stack(choices, 1), torch.stack(bandwidths, 1), torch.stack(log_probs, 1).sum(1), torch.stack(entropies, 1).sum(1)

    def evaluate_actions(self, encoded_states, cluster_choices, bandwidths, available_migs):
        batch_size, client_count, _ = encoded_states.shape
        history = encoded_states.new_zeros(batch_size, self.num_migs * 2)
        log_probs, entropies = [], []
        for index in range(client_count):
            features = self._fuse(encoded_states[:, index:index + 1], history)
            cluster_dist = self._cluster_dist(features, available_migs)
            choice, bandwidth = cluster_choices[:, index].long(), bandwidths[:, index]
            bandwidth_dist = self._bandwidth_dist(features)
            log_probs.append(cluster_dist.log_prob(choice) + self._squashed_log_prob(bandwidth_dist, bandwidth))
            entropies.append(cluster_dist.entropy() + bandwidth_dist.entropy())
            history = self._update_history(history, choice, bandwidth)
        return torch.stack(log_probs, 1).sum(1), torch.stack(entropies, 1).sum(1)
