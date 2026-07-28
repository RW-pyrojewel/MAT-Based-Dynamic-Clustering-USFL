"""Real CIFAR-100 and USFL training runner for research-plan scenario A."""
import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn.functional as functional

from data import CIFAR100NonIIDProvider
from envs import LiquidAIRANEnv
from models.mat_agent import MATAgent
from models.usfl_networks import ResNet18_USFL
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


def _act(agent, state, available_migs, edge_state, deterministic):
    action, log_prob = agent.act(state, available_migs, edge_state, deterministic=deterministic)
    if (action["cluster"] < 0).any() or (action["cluster"] >= available_migs).any():
        raise ValueError("MAT produced a cluster outside the available MIG range")
    if not np.all(action["l1"] < action["l2"]):
        raise ValueError("MAT produced an invalid USFL split pair")
    if action["bw"].sum() > 1.0 + 1e-6:
        raise ValueError("MAT exceeded the global bandwidth budget")
    return action, log_prob


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
        for mig_id in np.unique(action["cluster"]):
            members = np.flatnonzero(action["cluster"] == mig_id)
            l1 = int(action["l1"][members[0]])
            l2 = int(action["l2"][members[0]])
            if not np.all(action["l1"][members] == l1) or not np.all(action["l2"][members] == l2):
                raise ValueError("USFL requires a shared split pair within each cluster")

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
        return
    kwargs = buffer.as_ppo_kwargs()
    agent.update_policy(kwargs.pop("rewards"), kwargs.pop("next_states"), kwargs.pop("dones"), **kwargs)
    buffer.clear()


def run_mat_scenario_a(
    data_dir="../Data",
    log_dir="logs",
    total_epochs=150,
    seed=7,
    batch_size=16,
    local_steps=4,
    warmup_epochs=5,
    learning_rate=0.01,
    ppo_update_interval=16,
    fedavg_interval=10,
    evaluation_batches=10,
    device=None,
    create_plots=True,
):
    """Run online MAT control with actual CIFAR-100 USFL training in scenario A."""
    if not 1 <= total_epochs <= 150:
        raise ValueError("total_epochs must be in [1, 150]")
    if min(batch_size, local_steps, ppo_update_interval, fedavg_interval, evaluation_batches) < 1:
        raise ValueError("training intervals and batch size must be positive")
    if not 0 <= warmup_epochs < total_epochs:
        raise ValueError("warmup_epochs must be non-negative and smaller than total_epochs")

    np.random.seed(seed)
    torch.manual_seed(seed)
    execution_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    provider = CIFAR100NonIIDProvider(num_clients=30, alpha=0.1, data_dir=data_dir, seed=seed)
    environments = {
        station_id: LiquidAIRANEnv(
            provider, max_vehicles=10, max_migs=7, scenario="A", station_id=station_id,
            vehicle_id_offset=(station_id - 1) * 10, seed=seed + station_id,
        )
        for station_id in (1, 2, 3)
    }
    models = {station_id: ResNet18_USFL(num_classes=provider.num_classes).to(execution_device) for station_id in environments}
    optimizers = {
        station_id: torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        for station_id, model in models.items()
    }
    _warm_up_models(models, execution_device, batch_size)
    iterators = {station_id: {"batch_size": batch_size} for station_id in environments}
    agent = MATAgent(state_dim=2 + provider.num_classes, hidden_dim=128, num_migs=7, num_cut_layers=7, device=execution_device)
    reward_config = MATRewardConfig(cluster_size_limit=10)
    buffer = MATTrajectoryBuffer()
    pending = {}
    logger = SimulationLogger(log_dir=log_dir, run_name="scenario_a_mat")
    logger.set_metadata(
        scenario="A", algorithm="MAT-RL", dataset="CIFAR-100", num_stations=3, clients_per_station=10,
        total_epochs=total_epochs, batch_size=batch_size, local_steps=local_steps, warmup_epochs=warmup_epochs,
        client_gradient_normalization=True, optimizer_momentum_reset_on_fedavg=True,
        fedavg_interval=fedavg_interval, seed=seed, device=str(execution_device),
    )

    for epoch in range(1, total_epochs + 1):
        is_warmup = epoch <= warmup_epochs
        observations = {station_id: environment.step() for station_id, environment in environments.items()}
        for station_id, (state, available_migs, bandwidth, client_ids) in observations.items():
            edge_state = np.asarray([available_migs, bandwidth], dtype=np.float32)
            previous = pending.pop(station_id, None)
            if previous is not None and not is_warmup:
                buffer.append(
                    previous["state"], previous["edge_state"], previous["action"], previous["reward"],
                    state, edge_state, False, previous["log_prob"], previous["available_migs"],
                )

            action, log_prob = _act(agent, state, available_migs, edge_state, deterministic=False)
            training = _run_usfl_round(
                models[station_id], optimizers[station_id], provider, iterators[station_id], client_ids,
                action, execution_device, local_steps,
            )
            tx_delays = environments[station_id].calc_wireless_transmission_delay(
                action["cluster"], action["bw"], training["smashed_sizes"], state[:, 0]
            )
            total_delay = max(
                tx_delays[mig_id] + training["cluster_compute_delays"].get(mig_id, 0.0)
                for mig_id in range(available_migs)
            )
            compute_delay = max(training["cluster_compute_delays"].values(), default=0.0)
            reward, reward_terms = compute_mat_reward(
                total_delay, state[:, 2:], action["cluster"], action["bw"], reward_config
            )
            if not is_warmup:
                pending[station_id] = {
                    "state": state, "edge_state": edge_state, "action": action, "reward": reward,
                    "log_prob": log_prob, "available_migs": available_migs,
                }
            logger.log_metrics(
                "MAT-RL",
                epoch=epoch, station_id=station_id, is_warmup=is_warmup, vehicle_count=len(state),
                available_migs=available_migs, bandwidth_hz=bandwidth,
                bandwidth_allocation_sum=float(action["bw"].sum()),
                bandwidth_unused=float(1.0 - action["bw"].sum()),
                total_delay_ms=total_delay * 1000.0, tx_delay_ms=float(tx_delays.max()) * 1000.0,
                compute_delay_ms=compute_delay * 1000.0, reward=reward, train_loss=training["train_loss"],
                train_accuracy=training["train_accuracy"], test_accuracy=None, **reward_terms,
            )

        if epoch % fedavg_interval == 0:
            _fedavg(list(models.values()), list(optimizers.values()))
            test_accuracy = _evaluate_global_model(models[1], provider, execution_device, evaluation_batches)
            for row in logger.records["MAT-RL"][-3:]:
                row["test_accuracy"] = test_accuracy
        if not is_warmup and len(buffer) >= ppo_update_interval:
            _flush_ppo(agent, buffer)

    for terminal in pending.values():
        buffer.append(
            terminal["state"], terminal["edge_state"], terminal["action"], terminal["reward"],
            terminal["state"], terminal["edge_state"], True, terminal["log_prob"], terminal["available_migs"],
        )
    _flush_ppo(agent, buffer)
    csv_paths = logger.export_to_csv()
    json_path = logger.export_to_json()
    plot_paths = logger.plot_scenario_a("MAT-RL", warmup_epochs=warmup_epochs) if create_plots else []
    return {"logger": logger, "csv_paths": csv_paths, "json_path": json_path, "plot_paths": plot_paths, "device": str(execution_device)}


def main():
    parser = argparse.ArgumentParser(description="Run real CIFAR-100 USFL training with MAT in scenario A.")
    parser.add_argument("--data-dir", default="../Data")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-steps", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--ppo-update-interval", type=int, default=16)
    parser.add_argument("--fedavg-interval", type=int, default=10)
    parser.add_argument("--evaluation-batches", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run_mat_scenario_a(
        data_dir=args.data_dir, log_dir=args.log_dir, total_epochs=args.epochs, seed=args.seed,
        batch_size=args.batch_size, local_steps=args.local_steps, warmup_epochs=args.warmup_epochs,
        learning_rate=args.learning_rate, ppo_update_interval=args.ppo_update_interval,
        fedavg_interval=args.fedavg_interval, evaluation_batches=args.evaluation_batches,
        device=args.device, create_plots=not args.no_plots,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "logger"}, indent=2))


if __name__ == "__main__":
    main()
