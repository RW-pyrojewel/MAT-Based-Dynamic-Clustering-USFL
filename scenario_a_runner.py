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


def _act(agent, state, available_migs, edge_state, deterministic, client_ids=None):
    if isinstance(agent, MATAgent):
        action, policy_info = agent.act(
            state,
            available_migs,
            edge_state,
            client_ids=client_ids,
            deterministic=deterministic,
        )
    else:
        action, policy_info = agent.act(state, available_migs, edge_state, deterministic=deterministic)
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


def _run_usfl_round(model, optimizer, provider, iterators, client_ids, action, device, local_steps):
    """Execute actual U-shaped split training for several local mini-batches."""
    model.train()
    cluster_compute_delays = {}
    smashed_sizes = np.zeros(len(client_ids), dtype=np.float64)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for _ in range(local_steps):
        optimizer.zero_grad(set_to_none=True)
        execution_groups = np.asarray(action.get("virtual_cluster", action["cluster"]), dtype=np.int64)
        for group_id in np.unique(execution_groups):
            members = np.flatnonzero(execution_groups == group_id)
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
            advanced_batch = model.forward_partB(smashed_batch, l1, l2)
            logits = model.forward_partC(advanced_batch, l2)
            loss = functional.cross_entropy(logits, target_batch)
            (loss * (len(members) / len(client_ids))).backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            cluster_compute_delays[int(mig_id)] = cluster_compute_delays.get(int(mig_id), 0.0) + time.perf_counter() - compute_started

            total_loss += float(loss.detach()) * len(target_batch)
            total_correct += int((logits.detach().argmax(dim=1) == target_batch).sum())
            total_examples += len(target_batch)
            offset = 0
            for local_index, size in zip(members, part_sizes):
                smashed_sizes[local_index] += smashed_batch[offset:offset + size].numel() / size * smashed_batch.element_size()
                offset += size
        optimizer.step()

    return {
        "smashed_sizes": smashed_sizes,
        "cluster_compute_delays": cluster_compute_delays,
        "train_loss": total_loss / max(total_examples, 1),
        "train_accuracy": total_correct / max(total_examples, 1),
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


def _fedavg(models, optimizers):
    reference_state = copy.deepcopy(models[0].state_dict())
    for key, tensor in reference_state.items():
        values = [model.state_dict()[key] for model in models]
        if torch.is_floating_point(tensor):
            tensor.copy_(torch.stack([value.detach().float() for value in values]).mean(dim=0).to(dtype=tensor.dtype))
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


def _build_agent(algorithm, state_dim, device, min_bandwidth_share, nominal_bandwidth_hz=100e6):
    algorithm = algorithm.lower()
    if algorithm == "mat":
        return "MAT-RL", MATAgent(
            state_dim=state_dim, hidden_dim=128, num_migs=7, num_cut_layers=7,
            min_bandwidth_share=min_bandwidth_share, nominal_bandwidth_hz=nominal_bandwidth_hz, device=device,
        )
    if algorithm == "cpsl":
        return "Adapted-CPSL", CPSLAgent(fixed_clusters=2, fixed_l1=3, fixed_l2=4)
    if algorithm == "clustersfl":
        return "Adapted-ClusterSFL", ClusterSFLAgent(num_cut_layers=7)
    if algorithm == "pcsfl":
        return "Adapted-PCSFL", PCSFLAgent(
            state_dim=state_dim, max_clients=10, max_migs=7, num_cut_layers=7, device=device,
        )
    raise ValueError("algorithm must be one of: mat, cpsl, clustersfl, pcsfl")


def _append_transition(agent, buffer, previous, state, edge_state, done, station_id, epoch, episode_id=0):
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
        agent.observe(
            previous["state"], previous["edge_state"], previous["action"], previous["reward"],
            state, edge_state, done,
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
            nominal_bandwidth_hz=calculators[1].base_bandwidth,
        )
    else:
        algorithm_name, agent = "MAT-RL", mat_agent
    reward_config = MATRewardConfig()
    buffer = trajectory_buffer if trajectory_buffer is not None else MATTrajectoryBuffer()
    pending = {}
    logger = SimulationLogger(log_dir=log_dir, run_name=run_name or f"scenario_a_{algorithm}_seed_{seed}")
    logger.set_metadata(
        scenario="A", algorithm=algorithm_name, dataset="CIFAR-100", num_stations=3, clients_per_station=10,
        total_epochs=total_epochs, batch_size=batch_size, local_steps=local_steps, warmup_epochs=warmup_epochs,
        client_gradient_normalization=True, optimizer_momentum_reset_on_fedavg=True,
        min_bandwidth_share=min_bandwidth_share, baseline_bandwidth_policy="equal-global-share",
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
        for station_id in calculators:
            state, available_migs, bandwidth, client_ids = trace.get(epoch, station_id)
            edge_state = np.asarray([available_migs, bandwidth], dtype=np.float32)
            previous = pending.pop(station_id, None)
            if previous is not None:
                _append_transition(agent, buffer, previous, state, edge_state, False, station_id, epoch, episode_id)

            action, policy_info = _act(
                agent, state, available_migs, edge_state, deterministic=False, client_ids=client_ids
            )
            training = _run_usfl_round(
                models[station_id], optimizers[station_id], provider, iterators[station_id], client_ids,
                action, execution_device, local_steps,
            )
            tx_delays = calculators[station_id].calc_wireless_transmission_delay(
                action["cluster"], action["bw"], training["smashed_sizes"], state[:, 0],
                available_migs=available_migs, bandwidth_hz=bandwidth,
            )
            total_delay = max(
                tx_delays[mig_id] + training["cluster_compute_delays"].get(mig_id, 0.0)
                for mig_id in range(available_migs)
            )
            compute_delay = max(training["cluster_compute_delays"].values(), default=0.0)
            reward, reward_terms = compute_mat_reward(
                total_delay, state[:, 2:], action["cluster"], action["bw"], reward_config
            )
            updates_online = algorithm == "pcsfl" or (algorithm == "mat" and not is_warmup)
            if updates_online:
                pending[station_id] = {
                    "state": state, "edge_state": edge_state, "action": action, "reward": reward,
                    "policy_info": policy_info, "available_migs": available_migs, "epoch": epoch,
                    "policy_version": agent.policy_version,
                }
            logger.log_metrics(
                algorithm_name,
                epoch=epoch, station_id=station_id, is_warmup=is_warmup, vehicle_count=len(state),
                available_migs=available_migs, bandwidth_hz=bandwidth,
                physical_cluster_count=int(len(np.unique(action["cluster"]))),
                virtual_cluster_count=int(len(np.unique(action.get("virtual_cluster", action["cluster"])))),
                bandwidth_allocation_sum=float(action["bw"].sum()), bandwidth_unused=float(1.0 - action["bw"].sum()),
                min_bandwidth_share=float(action["bw"].min()), max_bandwidth_share=float(action["bw"].max()),
                **_bandwidth_diagnostics(action["bw"], min_bandwidth_share),
                mean_l1=float(action["l1"].mean()), mean_l2=float(action["l2"].mean()),
                smashed_data_bytes_total=float(training["smashed_sizes"].sum()),
                smashed_data_bytes_per_client_mean=float(training["smashed_sizes"].mean()),
                total_delay_ms=total_delay * 1000.0, tx_delay_ms=float(tx_delays.max()) * 1000.0,
                compute_delay_ms=compute_delay * 1000.0, reward=reward, train_loss=training["train_loss"],
                train_accuracy=training["train_accuracy"], test_accuracy=None, **reward_terms,
            )

        if epoch % fedavg_interval == 0:
            _fedavg(list(models.values()), list(optimizers.values()))
            test_accuracy = _evaluate_global_model(models[1], provider, execution_device, evaluation_batches)
            for row in logger.records[algorithm_name][-3:]:
                row["test_accuracy"] = test_accuracy
        if algorithm == "mat" and mat_training_mode == "online" and not is_warmup and len(buffer) >= ppo_update_interval:
            diagnostics = _flush_ppo(agent, buffer)
            for row in logger.records[algorithm_name][-3:]:
                row.update({f"ppo_{key}": value for key, value in diagnostics.items()})

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
        diagnostics = _flush_ppo(agent, buffer)
        if diagnostics:
            for row in logger.records[algorithm_name][-3:]:
                row.update({f"ppo_{key}": value for key, value in diagnostics.items()})
    csv_paths = logger.export_to_csv() if export_results else []
    json_path = logger.export_to_json() if export_results else None
    plot_paths = logger.plot_scenario_a(algorithm_name, warmup_epochs=warmup_epochs) if create_plots and export_results else []
    return {
        "logger": logger, "csv_paths": csv_paths, "json_path": json_path, "plot_paths": plot_paths,
        "device": str(execution_device), "trace_id": trace.trace_id, "agent": agent,
        "trajectory_buffer": buffer,
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
