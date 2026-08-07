"""Shared contract and metrics for reusable Scenario-A experiment artifacts."""
from __future__ import annotations

import hashlib
import json
import platform

import numpy as np
import torch


SEEDS = (7, 17, 29)
BASELINES = ("cpsl", "clustersfl", "pcsfl")


def experiment_contract(batch_size=16, local_steps=4, evaluation_batches=10):
    contract = {
        "schema_version": 1,
        "scenario": "A",
        "dataset": "CIFAR-100-noniid-alpha-0.1",
        "data_partition": "dirichlet-alpha-0.1-seed-matched-10-clients-per-station",
        "trace_schema": "scenario-a-resource-tide-v2-seed-matched",
        "station_count": 3,
        "clients_per_station": 10,
        "model": "CIFAR-ResNet18-USFL-GroupNorm",
        "rounds": 150,
        "warmup_rounds": 5,
        "batch_size": int(batch_size),
        "local_steps": int(local_steps),
        "evaluation_batches": int(evaluation_batches),
        "communication_schema": "section-3.1.1-bidirectional-v2",
        "uplink": "beta*B*log2(1+10g)",
        "downlink": "base-station-equal-share",
        "payload": "each-direction-v(l1)+v(l2)-actual-batched-bytes",
        "global_round_metric": "max-station-delay",
        "accuracy_gate_pp": 2.0,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return contract, hashlib.sha256(encoded).hexdigest()[:16]


def records(result):
    return next(iter(result["logger"].records.values()))


def global_delay_vector(rows, start=6, end=150):
    by_epoch = {}
    for row in rows:
        epoch = int(row["epoch"])
        if start <= epoch <= end:
            by_epoch.setdefault(epoch, []).append(float(row["total_delay_ms"]))
    return np.asarray([max(by_epoch[epoch]) for epoch in sorted(by_epoch)], dtype=np.float64)


def final_accuracy(rows):
    values = [float(row["test_accuracy"]) for row in rows if row.get("test_accuracy") is not None]
    return values[-1] if values else 0.0


def runtime_metadata(device):
    metadata = {
        "device": str(device), "torch": torch.__version__, "python": platform.python_version(),
        "cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        metadata["gpu"] = torch.cuda.get_device_name(torch.device(device))
        metadata["cudnn"] = torch.backends.cudnn.version()
    else:
        metadata["gpu"] = None
        metadata["cudnn"] = None
    return metadata


def assert_runtime_compatible(reference, current):
    """Reject cross-hardware/software comparisons of measured GPU wall-clock."""
    keys = ("gpu", "torch", "cuda", "cudnn")
    mismatch = {key: (reference.get(key), current.get(key))
                for key in keys if reference.get(key) != current.get(key)}
    if mismatch:
        raise ValueError(f"baseline runtime contract differs from MAT runtime: {mismatch}")


def relative_reduction(candidate, reference):
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if candidate.shape != reference.shape:
        raise ValueError("paired global-delay vectors do not align")
    return (reference - candidate) / np.maximum(reference, 1e-12)


def stratified_bootstrap_lower(vectors, samples=10000, seed=20260806):
    rng = np.random.default_rng(seed)
    keys = np.asarray(sorted(vectors))
    estimates = []
    for _ in range(int(samples)):
        values = []
        for key in rng.choice(keys, size=len(keys), replace=True):
            vector = np.asarray(vectors[int(key)])
            values.extend(rng.choice(vector, size=len(vector), replace=True))
        estimates.append(float(np.mean(values)))
    return float(np.quantile(estimates, 0.025))
