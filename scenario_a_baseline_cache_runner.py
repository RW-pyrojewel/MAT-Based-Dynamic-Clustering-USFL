"""Run each paper-adapted Scenario-A baseline once and cache its contract."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from scenario_a_experiment_contract import (
    BASELINES, SEEDS, experiment_contract, final_accuracy, global_delay_vector,
    records, runtime_metadata,
)
from scenario_a_runner import run_scenario_a


def run_baseline_cache(data_dir="../Data", log_dir="logs", device=None,
                       batch_size=16, local_steps=4, evaluation_batches=10):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    contract, contract_hash = experiment_contract(batch_size, local_steps, evaluation_batches)
    root = Path(log_dir) / f"scenario_a_baselines_v4_{contract_hash}_{datetime.now():%Y%m%d_%H%M%S}"
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "status": "running", "contract": contract,
        "contract_hash": contract_hash, "runtime": runtime_metadata(device),
        "seeds": list(SEEDS), "algorithms": {},
    }
    for algorithm in BASELINES:
        report["algorithms"][algorithm] = {}
        for seed in SEEDS:
            print(f"[{algorithm}] seed={seed}", flush=True)
            result = run_scenario_a(
                algorithm=algorithm, seed=seed, data_dir=data_dir, log_dir=str(root),
                total_epochs=150, batch_size=batch_size, local_steps=local_steps,
                evaluation_batches=evaluation_batches, create_plots=False,
                device=device,
                run_name=f"{algorithm}_contract_{contract_hash}_seed_{seed}")
            rows = records(result)
            report["algorithms"][algorithm][str(seed)] = {
                "trace_id": result["trace_id"],
                "global_delay_ms": global_delay_vector(rows).tolist(),
                "cumulative_global_delay_ms": float(global_delay_vector(rows).sum()),
                "final_test_accuracy": final_accuracy(rows),
                "csv_paths": [str(path) for path in result["csv_paths"]],
                "json_path": str(result["json_path"]),
            }
            (root / "baseline_contract_report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    report["status"] = "completed"
    report_path = root / "baseline_contract_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return root, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../Data")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-steps", type=int, default=4)
    parser.add_argument("--evaluation-batches", type=int, default=10)
    args = parser.parse_args()
    root, report = run_baseline_cache(**vars(args))
    print(root)
    print(json.dumps({"contract_hash": report["contract_hash"], "status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
