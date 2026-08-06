"""Real CIFAR-100 and USFL training runner for research-plan scenario A."""
import argparse
import copy
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as functional

from baselines import CPSLAgent, ClusterSFLAgent, PCSFLAgent
from data import CIFAR100NonIIDProvider
from envs import LiquidAIRANEnv
from models.mat_agent import MATAgent
from models.usfl_networks import ResNet18_USFL
from scenario_a_trace import build_scenario_a_trace
from utils.logger import SimulationLogger
from utils.mat_reward import MATRewardConfig, compute_mat_reward
from utils.trajectory_buffer import MATTrajectoryBuffer
from utils.channel_diagnostics import channel_bandwidth_metrics, required_airtime, spearman_correlation


def _next_batch(provider, iterators, client_id, batch_size):
    iterator = iterators.get(client_id)
    if iterator is None:
        iterator = iter(provider.get_client_dataloader(client_id, batch_size=batch_size, shuffle=True, drop_last=True))
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(provider.get_client_dataloader(client_id, batch_size=batch_size, shuffle=True, drop_last=True))
        batch = next(iterator)
    iterators[client_id] = iterator
    return batch


def _client_data_volumes(provider, client_ids):
    return np.asarray([len(provider.client_indices[int(client_id)]) for client_id in client_ids],
                      dtype=np.float64)


@torch.no_grad()
def _model_pca_embedding(model, dimensions=8, samples_per_tensor=64):
    """Bounded PCA summary used by the paper-adapted PCSFL state."""
    rows = []
    for parameter in model.parameters():
        flat = parameter.detach().float().reshape(-1)
        if not flat.numel():
            continue
        indices = torch.linspace(0, flat.numel() - 1,
                                 min(samples_per_tensor, flat.numel()), device=flat.device).long()
        values = flat[indices]
        if values.numel() < samples_per_tensor:
            values = functional.pad(values, (0, samples_per_tensor - values.numel()))
        rows.append(values)
    if not rows:
        return np.zeros(dimensions, dtype=np.float32)
    matrix = torch.stack(rows)
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(matrix)[:dimensions]
    singular = singular / torch.clamp(torch.linalg.vector_norm(singular), min=1e-12)
    output = functional.pad(singular, (0, max(0, dimensions - singular.numel())))[:dimensions]
    return output.cpu().numpy().astype(np.float32)


def _baseline_context(agent, provider, client_ids, model):
    context = {"data_volumes": _client_data_volumes(provider, client_ids)}
    if isinstance(agent, PCSFLAgent):
        global_embedding = _model_pca_embedding(model, agent.model_pca_dim)
        context["model_embedding"] = agent.embeddings_for(client_ids, global_embedding)
    return context


def _act(agent, state, available_migs, edge_state, deterministic, client_ids=None,
         baseline_context=None):
    if isinstance(agent, MATAgent):
        action, policy_info = agent.act(
            state,
            available_migs,
            edge_state,
            client_ids=client_ids,
            deterministic=deterministic,
        )
    else:
        action, policy_info = agent.act(
            state, available_migs, edge_state, deterministic=deterministic,
            **(baseline_context or {}))
    if (action["cluster"] < 0).any() or (action["cluster"] >= available_migs).any():
        raise ValueError("agent produced a cluster outside the available MIG range")
    if not np.all(action["l1"] < action["l2"]):
        raise ValueError("agent produced an invalid USFL split pair")
    if action["bw"].sum() > 1.0 + 1e-6:
        raise ValueError("agent exceeded the global bandwidth budget")
    return action, policy_info


def _bandwidth_diagnostics(weights, minimum_share):
    weights = np.asarray(weights, dtype=np.float64)
    mean = float(weights.mean())
    pairwise_difference = np.abs(weights[:, None] - weights[None, :]).mean()
    gini = pairwise_difference / max(2.0 * mean, 1e-12)
    return {
        "bandwidth_gini": float(gini),
        "bandwidth_cv": float(weights.std() / max(mean, 1e-12)),
        "bandwidth_floor_hit_rate": float(np.mean(np.isclose(weights, minimum_share, atol=1e-4))),
    }


def _physical_channel_diagnostics(agent, state, edge_state, action, policy_info, available_migs,
                                  payload_bytes, bandwidth_hz, minimum_share):
    policy_info = policy_info or {}
    metrics = channel_bandwidth_metrics(
        state[:, 0], payload_bytes, action["bw"], bandwidth_hz, minimum_share,
    )
    counterfactual_rho = 0.0
    latent_means = np.zeros(len(state), dtype=np.float64)
    if isinstance(agent, MATAgent):
        latent_means = agent.evaluate_bandwidth_prefix_means(
            state, edge_state, action, available_migs, policy_info["decision_order"],
        )
        counterfactual_state = np.asarray(state).copy()
        counterfactual_state[:, 0] = np.roll(counterfactual_state[:, 0], 1)
        changed_means = agent.evaluate_bandwidth_prefix_means(
            counterfactual_state, edge_state, action, available_migs, policy_info["decision_order"],
        )
        original_airtime = required_airtime(payload_bytes, state[:, 0])
        changed_airtime = required_airtime(payload_bytes, counterfactual_state[:, 0])
        counterfactual_rho = spearman_correlation(changed_airtime - original_airtime,
                                                  changed_means - latent_means)
    scalars = {key: float(value) for key, value in metrics.items() if np.isscalar(value)}
    scalars["channel_permutation_delta_spearman"] = float(counterfactual_rho)
    scalars["bandwidth_latent_mean"] = float(np.mean(latent_means))
    scalars["bandwidth_alpha_mean"] = float(np.mean(policy_info.get("bandwidth_alpha", 0.0)))
    scalars["bandwidth_alpha_min"] = float(np.min(policy_info.get("bandwidth_alpha", [0.0])))
    scalars["bandwidth_alpha_max"] = float(np.max(policy_info.get("bandwidth_alpha", [0.0])))
    scalars["bandwidth_context_score_std"] = float(np.std(policy_info.get("bandwidth_context_scores", 0.0)))
    scalars["bandwidth_physical_score_std"] = float(np.std(policy_info.get("bandwidth_physical_scores", 0.0)))
    oracle = np.asarray(metrics["oracle_bandwidth"], dtype=np.float64)
    airtimes = np.asarray(metrics["required_airtimes"], dtype=np.float64)
    clients = [{
        "client_index": int(index), "channel_gain": float(state[index, 0]),
        "payload_bytes": float(payload_bytes[index]), "required_airtime": float(airtimes[index]),
        "bandwidth_share": float(action["bw"][index]), "oracle_bandwidth_share": float(oracle[index]),
        "bandwidth_latent_mean": float(latent_means[index]),
        "bandwidth_alpha": float(np.asarray(policy_info.get("bandwidth_alpha", np.zeros(len(state))))[index]),
        "bandwidth_context_score": float(np.asarray(policy_info.get("bandwidth_context_scores", np.zeros(len(state))))[index]),
        "bandwidth_physical_score": float(np.asarray(policy_info.get("bandwidth_physical_scores", np.zeros(len(state))))[index]),
    } for index in range(len(state))]
    return scalars, clients

def _compress_feature_tensor(tensor, ratio):
    ratio = float(np.clip(ratio, 0.0, 1.0))
    if ratio >= 1.0:
        return tensor, tensor.numel()
    flat = tensor.detach().abs().reshape(-1)
    keep = max(1, int(np.ceil(ratio * flat.numel())))
    if keep >= flat.numel():
        return tensor, tensor.numel()
    threshold = torch.topk(flat, keep, sorted=False).values.min()
    mask = (tensor.detach().abs() >= threshold).to(dtype=tensor.dtype)
    return tensor * mask, int(mask.sum().item())


def _run_usfl_round(model, optimizer, provider, iterators, client_ids, action, device, local_steps):
    """Execute actual U-shaped split training for several local mini-batches."""
    model.train()
    cluster_compute_delays = {}
    smashed_sizes = np.zeros(len(client_ids), dtype=np.float64)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    frequency = action.get("local_update_frequency", {})
    compression = np.asarray(action.get("feature_compression", np.ones(len(client_ids))), dtype=np.float64)
    aggregation = np.asarray(action.get("aggregation_weight", np.full(len(client_ids), 1.0 / len(client_ids))),
                             dtype=np.float64)
    for local_step in range(local_steps):
        optimizer.zero_grad(set_to_none=True)
        execution_groups = np.asarray(action.get("virtual_cluster", action["cluster"]), dtype=np.int64)
        for group_id in np.unique(execution_groups):
            members = np.flatnonzero(execution_groups == group_id)
            if frequency and local_step >= int(frequency[int(group_id)]):
                continue
            mig_id = int(action["cluster"][members[0]])
            if not np.all(action["cluster"][members] == mig_id):
                raise ValueError("an execution group must map to one physical MIG")
            l1 = int(action["l1"][members[0]])
            l2 = int(action["l2"][members[0]])
            if not np.all(action["l1"][members] == l1) or not np.all(action["l2"][members] == l2):
                raise ValueError("USFL requires a shared split pair within each execution group")

            batches = [
                _next_batch(provider, iterators, int(client_ids[index]), batch_size=iterators["batch_size"])
                for index in members
            ]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            compute_started = time.perf_counter()
            image_parts = []
            label_parts = []
            part_sizes = []
            for images, targets in batches:
                image_parts.append(images.to(device))
                label_parts.append(targets.to(device))
                part_sizes.append(len(targets))
            input_batch = torch.cat(image_parts, dim=0)
            target_batch = torch.cat(label_parts, dim=0)
            smashed_batch = model.forward_partA(input_batch, l1)
            compressed_parts = []
            offset = 0
            retained = []
            for local_index, size in zip(members, part_sizes):
                part, nonzero = _compress_feature_tensor(
                    smashed_batch[offset:offset + size], compression[local_index])
                compressed_parts.append(part)
                retained.append(nonzero / size * smashed_batch.element_size())
                offset += size
            smashed_batch = torch.cat(compressed_parts, dim=0)
            advanced_batch = model.forward_partB(smashed_batch, l1, l2)
            logits = model.forward_partC(advanced_batch, l2)
            loss = functional.cross_entropy(logits, target_batch)
            (loss * float(aggregation[members].sum())).backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            cluster_compute_delays[int(mig_id)] = cluster_compute_delays.get(int(mig_id), 0.0) + time.perf_counter() - compute_started

            total_loss += float(loss.detach()) * len(target_batch)
            total_correct += int((logits.detach().argmax(dim=1) == target_batch).sum())
            total_examples += len(target_batch)
            for local_index, retained_bytes in zip(members, retained):
                smashed_sizes[local_index] += retained_bytes
        optimizer.step()

    return {
        "smashed_sizes": smashed_sizes,
        "cluster_compute_delays": cluster_compute_delays,
        "train_loss": total_loss / max(total_examples, 1),
        "train_accuracy": total_correct / max(total_examples, 1),
    }


def _run_clustersfl_round(model, optimizer, provider, iterators, client_ids, action, device,
                          local_steps, learning_rate):
    """Train independent ClusterSFL models and aggregate the complete cluster models."""
    base_state = copy.deepcopy(model.state_dict())
    volumes = _client_data_volumes(provider, client_ids)
    aggregate = {key: torch.zeros_like(value, dtype=torch.float32) if torch.is_floating_point(value)
                 else value.clone() for key, value in base_state.items()}
    smashed_sizes = np.zeros(len(client_ids), dtype=np.float64)
    cluster_compute = {}
    total_loss = total_correct = total_examples = 0
    for cluster in np.unique(action["cluster"]):
        members = np.flatnonzero(action["cluster"] == cluster)
        cluster_weight = float(np.asarray(action["aggregation_weight"])[members].sum())
        frequency = min(local_steps, int(action["local_update_frequency"][int(cluster)]))
        cluster_model = copy.deepcopy(model)
        cluster_model.load_state_dict(base_state)
        cluster_model.train()
        cluster_optimizer = torch.optim.SGD(
            cluster_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        started = time.perf_counter()
        for _ in range(frequency):
            cluster_optimizer.zero_grad(set_to_none=True)
            batches = [_next_batch(provider, iterators, int(client_ids[index]),
                                   batch_size=iterators["batch_size"]) for index in members]
            images = torch.cat([batch[0].to(device) for batch in batches], dim=0)
            targets = torch.cat([batch[1].to(device) for batch in batches], dim=0)
            sizes = [len(batch[1]) for batch in batches]
            l1, l2 = int(action["l1"][members[0]]), int(action["l2"][members[0]])
            smashed = cluster_model.forward_partA(images, l1)
            parts, offset = [], 0
            for index, size in zip(members, sizes):
                part, retained = _compress_feature_tensor(
                    smashed[offset:offset + size], action["feature_compression"][index])
                parts.append(part)
                smashed_sizes[index] += retained / size * smashed.element_size()
                offset += size
            smashed = torch.cat(parts, dim=0)
            logits = cluster_model.forward_partC(cluster_model.forward_partB(smashed, l1, l2), l2)
            loss = functional.cross_entropy(logits, targets)
            loss.backward()
            cluster_optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            total_correct += int((logits.detach().argmax(dim=1) == targets).sum())
            total_examples += len(targets)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        cluster_compute[int(cluster)] = time.perf_counter() - started
        state = cluster_model.state_dict()
        for key, value in aggregate.items():
            if torch.is_floating_point(base_state[key]):
                value.add_(state[key].detach().float(), alpha=cluster_weight)
        del cluster_model, cluster_optimizer
    for key, value in aggregate.items():
        aggregate[key] = (value.to(dtype=base_state[key].dtype)
                          if torch.is_floating_point(base_state[key]) else base_state[key])
    model.load_state_dict(aggregate)
    optimizer.state.clear()
    return {
        "smashed_sizes": smashed_sizes,
        "cluster_compute_delays": cluster_compute,
        "train_loss": total_loss / max(total_examples, 1),
        "train_accuracy": total_correct / max(total_examples, 1),
    }


@torch.no_grad()
def _parameter_sample(model, samples_per_tensor=32, layer_indices=None):
    values = []
    modules = [model] if layer_indices is None else [model.layers[int(index)] for index in layer_indices]
    for parameter in (parameter for module in modules for parameter in module.parameters()):
        flat = parameter.detach().float().reshape(-1)
        if flat.numel():
            index = torch.linspace(0, flat.numel() - 1,
                                   min(samples_per_tensor, flat.numel()), device=flat.device).long()
            values.append(flat[index])
    return torch.cat(values) if values else torch.zeros(1, device=next(model.parameters()).device)


def _run_pcsfl_round(model, optimizer, provider, iterators, client_ids, action, device,
                     local_steps, learning_rate):
    """PCSFL's per-cluster local training followed by data-weighted edge aggregation."""
    base_state = copy.deepcopy(model.state_dict())
    volumes = _client_data_volumes(provider, client_ids)
    cluster_ids = np.unique(action["cluster"])
    cluster_weights = np.asarray([volumes[action["cluster"] == cluster].sum()
                                  for cluster in cluster_ids], dtype=np.float64)
    cluster_weights /= max(cluster_weights.sum(), 1e-12)
    aggregate = {key: torch.zeros_like(value, dtype=torch.float32) if torch.is_floating_point(value)
                 else value.clone() for key, value in base_state.items()}
    smashed_sizes = np.zeros(len(client_ids), dtype=np.float64)
    client_compute = np.zeros(len(client_ids), dtype=np.float64)
    cluster_compute = {}
    total_loss = total_correct = total_examples = 0
    wasserstein = []
    client_embeddings = np.zeros((len(client_ids), 8), dtype=np.float32)

    for cluster_weight, cluster in zip(cluster_weights, cluster_ids):
        cluster_model = copy.deepcopy(model)
        cluster_model.load_state_dict(base_state)
        cluster_model.train()
        cluster_optimizer = torch.optim.SGD(
            cluster_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        members = np.flatnonzero(action["cluster"] == cluster)
        cut_layers = np.unique(np.concatenate([action["l1"][members], action["l2"][members]]))
        base_sample = torch.sort(_parameter_sample(model, layer_indices=cut_layers))[0]
        member_weights = volumes[members] / max(volumes[members].sum(), 1e-12)
        started = time.perf_counter()
        for _ in range(local_steps):
            cluster_optimizer.zero_grad(set_to_none=True)
            for member_weight, index in zip(member_weights, members):
                images, targets = _next_batch(
                    provider, iterators, int(client_ids[index]), batch_size=iterators["batch_size"])
                images, targets = images.to(device), targets.to(device)
                client_started = time.perf_counter()
                l1, l2 = int(action["l1"][index]), int(action["l2"][index])
                smashed = cluster_model.forward_partA(images, l1)
                logits = cluster_model.forward_partC(cluster_model.forward_partB(smashed, l1, l2), l2)
                loss = functional.cross_entropy(logits, targets)
                (loss * float(member_weight)).backward()
                client_compute[index] += time.perf_counter() - client_started
                smashed_sizes[index] += smashed.numel() / len(targets) * smashed.element_size()
                total_loss += float(loss.detach()) * len(targets)
                total_correct += int((logits.detach().argmax(dim=1) == targets).sum())
                total_examples += len(targets)
            cluster_optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        cluster_compute[int(cluster)] = time.perf_counter() - started
        current_state = cluster_model.state_dict()
        for key, value in aggregate.items():
            if torch.is_floating_point(base_state[key]):
                value.add_(current_state[key].detach().float(), alpha=float(cluster_weight))
        current_sample = torch.sort(_parameter_sample(
            cluster_model, layer_indices=cut_layers))[0]
        size = min(base_sample.numel(), current_sample.numel())
        wasserstein.append(float(torch.mean(torch.abs(base_sample[:size] - current_sample[:size])).cpu()))
        embedding = _model_pca_embedding(cluster_model, dimensions=8)
        client_embeddings[members] = embedding
        del cluster_model, cluster_optimizer

    for key, value in aggregate.items():
        if not torch.is_floating_point(base_state[key]):
            aggregate[key] = base_state[key]
        else:
            aggregate[key] = value.to(dtype=base_state[key].dtype)
    model.load_state_dict(aggregate)
    optimizer.state.clear()
    return {
        "smashed_sizes": smashed_sizes,
        "cluster_compute_delays": cluster_compute,
        "client_compute_delays": client_compute,
        "train_loss": total_loss / max(total_examples, 1),
        "train_accuracy": total_correct / max(total_examples, 1),
        "pcsfl_clustering_factor": float(np.mean(wasserstein)) if wasserstein else 0.0,
        "pcsfl_client_model_embeddings": client_embeddings,
    }


@torch.no_grad()
def _warm_up_models(models, device, batch_size):
    """Initialize CUDA kernels without contaminating measured communication rounds."""
    sample = torch.randn(batch_size, 3, 32, 32, device=device)
    for model in models.values():
        was_training = model.training
        model.eval()
        logits = model.forward_partC(model.forward_partB(model.forward_partA(sample, 0), 0, 6), 6)
        del logits
        model.train(was_training)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _evaluate_global_model(model, provider, device, max_batches):
    model.eval()
    correct = 0
    total = 0
    for batch_index, (images, targets) in enumerate(provider.get_test_dataloader()):
        if batch_index >= max_batches:
            break
        images, targets = images.to(device), targets.to(device)
        logits = model.forward_partC(model.forward_partB(model.forward_partA(images, 0), 0, 6), 6)
        correct += int((logits.argmax(dim=1) == targets).sum())
        total += len(targets)
    return correct / max(total, 1)


def _fedavg(models, optimizers, weights=None):
    if weights is None:
        weights = np.full(len(models), 1.0 / len(models), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights /= max(weights.sum(), 1e-12)
    reference_state = copy.deepcopy(models[0].state_dict())
    for key, tensor in reference_state.items():
        values = [model.state_dict()[key] for model in models]
        if torch.is_floating_point(tensor):
            averaged = sum(value.detach().float() * float(weight)
                           for value, weight in zip(values, weights))
            tensor.copy_(averaged.to(dtype=tensor.dtype))
        else:
            tensor.copy_(values[0])
    for model, optimizer in zip(models, optimizers):
        model.load_state_dict(reference_state)
        optimizer.state.clear()


def _flush_ppo(agent, buffer):
    if not len(buffer):
        return {}
    kwargs = buffer.as_ppo_kwargs()
    diagnostics = agent.update_policy(
        kwargs.pop("rewards"),
        kwargs.pop("next_states"),
        kwargs.pop("dones"),
        **kwargs,
    )
    buffer.clear()
    return diagnostics


def _build_agent(algorithm, state_dim, device, min_bandwidth_share, nominal_bandwidth_hz=100e6,
                 seed=0):
    algorithm = algorithm.lower()
    if algorithm == "mat":
        return "MAT-RL", MATAgent(
            state_dim=state_dim, hidden_dim=128, num_migs=7, num_cut_layers=7,
            min_bandwidth_share=min_bandwidth_share, nominal_bandwidth_hz=nominal_bandwidth_hz, device=device,
        )
    if algorithm == "cpsl":
        return "PaperAdapted-CPSL", CPSLAgent(
            fixed_clusters=2, fixed_l1=3, fixed_l2=4, seed=seed)
    if algorithm == "clustersfl":
        return "PaperAdapted-ClusterSFL", ClusterSFLAgent(
            num_cut_layers=7, fixed_l1=3, fixed_l2=4)
    if algorithm == "pcsfl":
        return "PaperAdapted-PCSFL", PCSFLAgent(
            state_dim=state_dim, max_clients=10, max_migs=7, num_cut_layers=7, device=device,
        )
    raise ValueError("algorithm must be one of: mat, cpsl, clustersfl, pcsfl")


def _append_transition(agent, buffer, previous, state, edge_state, done, station_id, epoch,
                       episode_id=0, next_context=None):
    if isinstance(agent, MATAgent):
        buffer.append(
            previous["state"],
            previous["edge_state"],
            previous["action"],
            previous["reward"],
            state,
            edge_state,
            done,
            previous["policy_info"],
            previous["available_migs"],
            station_id,
            previous["epoch"],
            trajectory_id=(int(episode_id), int(station_id)),
            policy_version=previous["policy_version"],
        )
    elif isinstance(agent, PCSFLAgent):
        context = previous.get("baseline_context", {})
        next_context = next_context or context
        agent.observe(
            previous["state"], previous["edge_state"], previous["action"], previous["reward"],
            state, edge_state, done,
            data_volumes=context.get("data_volumes"),
            next_data_volumes=next_context.get("data_volumes"),
            model_embedding=context.get("model_embedding"),
            next_model_embedding=next_context.get("model_embedding"),
            clustering_factor=previous.get("pcsfl_clustering_factor"),
            waiting_factor=previous.get("pcsfl_waiting_factor"),
        )

def run_scenario_a(
    algorithm="mat",
    data_dir="../Data",
    log_dir="logs",
    total_epochs=150,
    seed=7,
    batch_size=16,
    local_steps=4,
    warmup_epochs=5,
    min_bandwidth_share=0.01,
    learning_rate=0.01,
    ppo_update_interval=16,
    fedavg_interval=10,
    evaluation_batches=10,
    device=None,
    create_plots=True,
    run_name=None,
    trace=None,
    mat_agent=None,
    trajectory_buffer=None,
    mat_training_mode="online",
    episode_id=0,
    export_results=True,
    mat_deterministic=False,
    client_diagnostics_path=None,
    epoch_callback=None,
):
    """Run one controller against a shared exogenous Scenario-A trace."""
    algorithm = algorithm.lower()
    if mat_training_mode not in {"online", "collect"}:
        raise ValueError("mat_training_mode must be online or collect")
    if mat_agent is not None and algorithm != "mat":
        raise ValueError("mat_agent can only be supplied for algorithm=mat")
    if not 1 <= total_epochs <= 150:
        raise ValueError("total_epochs must be in [1, 150]")
    if min(batch_size, local_steps, ppo_update_interval, fedavg_interval, evaluation_batches) < 1:
        raise ValueError("training intervals and batch size must be positive")
    if not 0 <= warmup_epochs < total_epochs:
        raise ValueError("warmup_epochs must be non-negative and smaller than total_epochs")
    if not 0.0 < min_bandwidth_share < 0.1:
        raise ValueError("min_bandwidth_share must be in (0, 0.1) for scenario A")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    execution_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    provider = CIFAR100NonIIDProvider(num_clients=30, alpha=0.1, data_dir=data_dir, seed=seed)
    trace = trace or build_scenario_a_trace(provider, seed=seed, total_epochs=total_epochs)
    if trace.total_epochs < total_epochs:
        raise ValueError("trace is shorter than total_epochs")
    calculators = {
        station_id: LiquidAIRANEnv(
            provider, max_vehicles=10, max_migs=7, scenario="A", station_id=station_id,
            vehicle_id_offset=(station_id - 1) * 10, seed=seed + station_id,
        )
        for station_id in (1, 2, 3)
    }
    models = {station_id: ResNet18_USFL(num_classes=provider.num_classes).to(execution_device) for station_id in calculators}
    optimizers = {
        station_id: torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        for station_id, model in models.items()
    }
    _warm_up_models(models, execution_device, batch_size)
    iterators = {station_id: {"batch_size": batch_size} for station_id in calculators}
    if mat_agent is None:
        algorithm_name, agent = _build_agent(
            algorithm, 2 + provider.num_classes, execution_device, min_bandwidth_share,
            nominal_bandwidth_hz=calculators[1].base_bandwidth, seed=seed,
        )
    else:
        algorithm_name, agent = "MAT-RL", mat_agent
    cpsl_saa = None
    if isinstance(agent, CPSLAgent) and warmup_epochs > 0:
        historical_states, historical_bandwidths = [], []
        for calibration_epoch in range(1, warmup_epochs + 1):
            for station_id in calculators:
                calibration_state, _, calibration_bandwidth, _ = trace.get(
                    calibration_epoch, station_id)
                historical_states.append(calibration_state)
                historical_bandwidths.append(calibration_bandwidth)
        cpsl_saa = agent.calibrate_split(historical_states, historical_bandwidths)
    reward_config = MATRewardConfig()
    buffer = trajectory_buffer if trajectory_buffer is not None else MATTrajectoryBuffer()
    pending = {}
    ppo_updates = []
    client_diagnostics = []
    logger = SimulationLogger(log_dir=log_dir, run_name=run_name or f"scenario_a_{algorithm}_seed_{seed}")
    logger.set_metadata(
        scenario="A", algorithm=algorithm_name, dataset="CIFAR-100", num_stations=3, clients_per_station=10,
        total_epochs=total_epochs, batch_size=batch_size, local_steps=local_steps, warmup_epochs=warmup_epochs,
        client_gradient_normalization=True, optimizer_momentum_reset_on_fedavg=True,
        min_bandwidth_share=min_bandwidth_share,
        baseline_fidelity_schema="paper-adapted-v2",
        baseline_external_budget="same Scenario-A clients/MIGs/bandwidth/data/rounds",
        baseline_cpsl_adaptation="global subchannel pool for concurrent Scenario-A MIGs",
        baseline_clustersfl_adaptation="top worker mapped to edge-cluster coordinator; fixed U-split",
        baseline_pcsfl_adaptation="shared station model represented by PCA summary; cluster-edge/cloud hierarchy",
        baseline_cpsl_saa=cpsl_saa,
        mat_decision_order="random-rollout/stable-client-id-deterministic",
        edge_state_normalization=(
            f"available_migs/7, bandwidth_hz/{calculators[1].base_bandwidth:g}"
        ),
        reward_delay_weight=reward_config.delay_weight,
        reward_delay_reference_seconds=reward_config.delay_reference_seconds,
        reward_cluster_capacity_enabled=reward_config.cluster_size_limit is not None,
        fedavg_interval=fedavg_interval, seed=seed, device=str(execution_device), trace_id=trace.trace_id,
    )

    for epoch in range(1, total_epochs + 1):
        is_warmup = epoch <= warmup_epochs
        observations = {}
        station_cloud_weights = {}
        for station_id in calculators:
            state, available_migs, bandwidth, client_ids = trace.get(epoch, station_id)
            edge_state = np.asarray([available_migs, bandwidth], dtype=np.float32)
            baseline_context = _baseline_context(agent, provider, client_ids, models[station_id])
            observations[station_id] = (
                state, available_migs, bandwidth, client_ids, edge_state, baseline_context)
            previous = pending.pop(station_id, None)
            if previous is not None:
                _append_transition(
                    agent, buffer, previous, state, edge_state, False, station_id, epoch,
                    episode_id, next_context=baseline_context)

        update_diagnostics = None
        if algorithm == "mat" and mat_training_mode == "online" and len(buffer) >= ppo_update_interval:
            update_started = time.perf_counter()
            update_diagnostics = _flush_ppo(agent, buffer)
            update_diagnostics["ppo_update_elapsed_seconds"] = time.perf_counter() - update_started
            update_diagnostics["update_epoch"] = int(epoch)
            update_diagnostics["policy_version_after"] = int(agent.policy_version)
            ppo_updates.append(dict(update_diagnostics))

        for station_id in calculators:
            state, available_migs, bandwidth, client_ids, edge_state, baseline_context = observations[station_id]
            action, policy_info = _act(
                agent, state, available_migs, edge_state,
                deterministic=(mat_deterministic and algorithm == "mat"), client_ids=client_ids,
                baseline_context=baseline_context,
            )
            if isinstance(agent, PCSFLAgent):
                training = _run_pcsfl_round(
                    models[station_id], optimizers[station_id], provider, iterators[station_id],
                    client_ids, action, execution_device, local_steps, learning_rate)
                agent.update_client_embeddings(
                    client_ids, training["pcsfl_client_model_embeddings"])
                station_cloud_weights[station_id] = float(baseline_context["data_volumes"].sum())
            elif isinstance(agent, ClusterSFLAgent):
                training = _run_clustersfl_round(
                    models[station_id], optimizers[station_id], provider, iterators[station_id],
                    client_ids, action, execution_device, local_steps, learning_rate)
                station_cloud_weights[station_id] = float(action["cloud_aggregation_mass"])
            else:
                training = _run_usfl_round(
                    models[station_id], optimizers[station_id], provider, iterators[station_id],
                    client_ids, action, execution_device, local_steps)
            allocator_diagnostics = None
            if algorithm == "mat" and agent.bandwidth_policy == "hybrid_water_filling":
                allocation, allocator_diagnostics = agent.allocate_bandwidth(
                    action, state, training["smashed_sizes"], training["cluster_compute_delays"], bandwidth)
                action["bw"] = allocation
            tx_delays = calculators[station_id].calc_wireless_transmission_delay(
                action["cluster"], action["bw"], training["smashed_sizes"], state[:, 0],
                available_migs=available_migs, bandwidth_hz=bandwidth,
            )
            physical_metrics, physical_clients = _physical_channel_diagnostics(
                agent, state, edge_state, action, policy_info, available_migs,
                training["smashed_sizes"], bandwidth, min_bandwidth_share,
            )
            if client_diagnostics_path is not None:
                for client_index, (client, client_id) in enumerate(zip(physical_clients, client_ids)):
                    allocator_client = {}
                    if allocator_diagnostics is not None:
                        allocator_client = {
                            "allocator_required_airtime": float(allocator_diagnostics["required_airtimes"][client_index]),
                            "allocator_equal_bandwidth": float(allocator_diagnostics["equal_bandwidth"][client_index]),
                            "allocator_hybrid_bandwidth": float(allocator_diagnostics["hybrid_bandwidth"][client_index]),
                            "allocator_compute_delay": float(allocator_diagnostics["client_compute_delays"][client_index]),
                        }
                    client_diagnostics.append({
                        "seed": int(seed), "episode_id": int(episode_id), "epoch": int(epoch),
                        "station_id": int(station_id), "client_id": int(client_id), **client, **allocator_client,
                    })
            total_delay = max(
                tx_delays[mig_id] + training["cluster_compute_delays"].get(mig_id, 0.0)
                for mig_id in range(available_migs)
            )
            compute_delay = max(training["cluster_compute_delays"].values(), default=0.0)
            pcsfl_waiting_factor = None
            if isinstance(agent, PCSFLAgent):
                efficiency = np.maximum(np.log2(1.0 + 10.0 * state[:, 0]), 1e-9)
                client_tx = (8.0 * training["smashed_sizes"]
                             / np.maximum(action["bw"] * bandwidth * efficiency, 1e-12))
                client_times = training["client_compute_delays"] + client_tx
                waiting = []
                for cluster in np.unique(action["cluster"]):
                    members = np.flatnonzero(action["cluster"] == cluster)
                    waiting.extend(float(client_times[members].max() - client_times[index])
                                   for index in members)
                pcsfl_waiting_factor = float(np.mean(waiting)) if waiting else 0.0
            reward, reward_terms = compute_mat_reward(
                total_delay, state[:, 2:], action["cluster"], action["bw"], reward_config
            )
            pcsfl_learning_reward = None
            if isinstance(agent, PCSFLAgent):
                pcsfl_learning_reward = agent.paper_reward(
                    training["pcsfl_clustering_factor"], pcsfl_waiting_factor)
            updates_online = algorithm == "pcsfl" or (algorithm == "mat" and not is_warmup)
            if updates_online:
                pending[station_id] = {
                    "state": state, "edge_state": edge_state, "action": action, "reward": reward,
                    "policy_info": policy_info, "available_migs": available_migs, "epoch": epoch,
                    "baseline_context": baseline_context,
                }
                if isinstance(agent, PCSFLAgent):
                    pending[station_id]["pcsfl_clustering_factor"] = training["pcsfl_clustering_factor"]
                    pending[station_id]["pcsfl_waiting_factor"] = pcsfl_waiting_factor
                if isinstance(agent, MATAgent):
                    pending[station_id]["policy_version"] = agent.policy_version
            allocator_scalars = {} if allocator_diagnostics is None else {
                (key if key.startswith("allocator_") else f"allocator_{key}"): value
                for key, value in allocator_diagnostics.items() if np.isscalar(value)
            }
            logger.log_metrics(
                algorithm_name,
                epoch=epoch, station_id=station_id, is_warmup=is_warmup, vehicle_count=len(state),
                available_migs=available_migs, bandwidth_hz=bandwidth,
                physical_cluster_count=int(len(np.unique(action["cluster"]))),
                virtual_cluster_count=int(len(np.unique(action.get("virtual_cluster", action["cluster"])))),
                bandwidth_allocation_sum=float(action["bw"].sum()), bandwidth_unused=float(1.0 - action["bw"].sum()),
                min_bandwidth_share=float(action["bw"].min()), max_bandwidth_share=float(action["bw"].max()),
                **_bandwidth_diagnostics(action["bw"], min_bandwidth_share), **physical_metrics,
                mean_l1=float(action["l1"].mean()), mean_l2=float(action["l2"].mean()),
                smashed_data_bytes_total=float(training["smashed_sizes"].sum()),
                smashed_data_bytes_per_client_mean=float(training["smashed_sizes"].mean()),
                total_delay_ms=total_delay * 1000.0, tx_delay_ms=float(tx_delays.max()) * 1000.0,
                compute_delay_ms=compute_delay * 1000.0, reward=reward, train_loss=training["train_loss"],
                **allocator_scalars,
                train_accuracy=training["train_accuracy"], test_accuracy=None, **reward_terms,
                feature_compression_mean=float(np.mean(action.get("feature_compression", 1.0))),
                top_worker_count=int(np.sum(action.get("top_worker", 0))),
                pcsfl_clustering_factor=training.get("pcsfl_clustering_factor"),
                pcsfl_waiting_factor=pcsfl_waiting_factor,
                pcsfl_learning_reward=pcsfl_learning_reward,
                cpsl_proxy_delay=(policy_info or {}).get("cpsl_proxy_delay"),
                baseline_exploration_epsilon=(policy_info or {}).get("epsilon"),
                **(agent.last_diagnostics if isinstance(agent, PCSFLAgent) else {}),
            )

        if epoch % fedavg_interval == 0:
            cloud_weights = None
            if isinstance(agent, (PCSFLAgent, ClusterSFLAgent)):
                cloud_weights = [station_cloud_weights[station_id] for station_id in models]
            _fedavg(list(models.values()), list(optimizers.values()), weights=cloud_weights)
            test_accuracy = _evaluate_global_model(models[1], provider, execution_device, evaluation_batches)
            for row in logger.records[algorithm_name][-3:]:
                row["test_accuracy"] = test_accuracy
        if update_diagnostics:
            for row in logger.records[algorithm_name][-3:]:
                row.update({f"ppo_{key}": value for key, value in update_diagnostics.items()})
        if epoch_callback is not None and epoch < total_epochs:
            epoch_callback(epoch=epoch, agent=agent, logger=logger, trace=trace)

    for station_id, terminal in pending.items():
        _append_transition(
            agent,
            buffer,
            terminal,
            terminal["state"],
            terminal["edge_state"],
            True,
            station_id,
            total_epochs,
            episode_id,
        )
    if algorithm == "mat" and mat_training_mode == "online":
        update_started = time.perf_counter()
        diagnostics = _flush_ppo(agent, buffer)
        if diagnostics:
            diagnostics["ppo_update_elapsed_seconds"] = time.perf_counter() - update_started
            diagnostics["update_epoch"] = int(total_epochs)
            diagnostics["policy_version_after"] = int(agent.policy_version)
            ppo_updates.append(dict(diagnostics))
            for row in logger.records[algorithm_name][-3:]:
                row.update({f"ppo_{key}": value for key, value in diagnostics.items()})
    if epoch_callback is not None:
        epoch_callback(epoch=total_epochs, agent=agent, logger=logger, trace=trace)
    if client_diagnostics_path is not None:
        sidecar_path = str(client_diagnostics_path)
        with open(sidecar_path, "w", encoding="utf-8") as handle:
            json.dump(client_diagnostics, handle, indent=2, allow_nan=False)
    else:
        sidecar_path = None
    csv_paths = logger.export_to_csv() if export_results else []
    json_path = logger.export_to_json() if export_results else None
    plot_paths = logger.plot_scenario_a(algorithm_name, warmup_epochs=warmup_epochs) if create_plots and export_results else []
    return {
        "logger": logger, "csv_paths": csv_paths, "json_path": json_path, "plot_paths": plot_paths,
        "device": str(execution_device), "trace_id": trace.trace_id, "agent": agent,
        "trajectory_buffer": buffer, "client_diagnostics_path": sidecar_path,
        "trace": trace, "ppo_updates": ppo_updates,
    }


def run_mat_scenario_a(**kwargs):
    """Compatibility wrapper for the original MAT-only Scenario-A entry point."""
    return run_scenario_a(algorithm="mat", **kwargs)


def run_baseline_scenario_a(algorithm, **kwargs):
    """Run one adapted CPSL, ClusterSFL, or PCSFL baseline in Scenario A."""
    if algorithm.lower() == "mat":
        raise ValueError("use run_mat_scenario_a for MAT-RL")
    return run_scenario_a(algorithm=algorithm, **kwargs)
def main():
    parser = argparse.ArgumentParser(description="Run one Scenario-A controller with real CIFAR-100 USFL training.")
    parser.add_argument("--algorithm", choices=("mat", "cpsl", "clustersfl", "pcsfl"), default="mat")
    parser.add_argument("--data-dir", default="../Data")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-steps", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-bandwidth-share", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--ppo-update-interval", type=int, default=16)
    parser.add_argument("--fedavg-interval", type=int, default=10)
    parser.add_argument("--evaluation-batches", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    result = run_scenario_a(
        algorithm=args.algorithm,
        data_dir=args.data_dir, log_dir=args.log_dir, total_epochs=args.epochs, seed=args.seed,
        batch_size=args.batch_size, local_steps=args.local_steps, warmup_epochs=args.warmup_epochs,
        min_bandwidth_share=args.min_bandwidth_share,
        learning_rate=args.learning_rate, ppo_update_interval=args.ppo_update_interval,
        fedavg_interval=args.fedavg_interval, evaluation_batches=args.evaluation_batches,
        device=args.device, create_plots=not args.no_plots, run_name=args.run_name,
    )
    print(json.dumps({key: value for key, value in result.items() if key not in {"logger", "agent", "trajectory_buffer"}}, indent=2))


if __name__ == "__main__":
    main()
