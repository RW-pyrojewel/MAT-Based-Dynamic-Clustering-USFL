"""Four-controller, multi-seed Scenario-A comparison entry point."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scenario_a_multiseed_runner import run_multiseed_scenario_a


ALGORITHMS = ("mat", "cpsl", "clustersfl", "pcsfl")
METRICS = (
    ("total_delay_ms", "System delay (ms)", "Scenario A: System Delay"),
    ("reward", "Reward", "Scenario A: Online Reward"),
    ("train_loss", "Cross-entropy loss", "Scenario A: USFL Training Loss"),
    ("test_accuracy", "Test accuracy", "Scenario A: Global Test Accuracy"),
)
COLORS = {
    "MAT-RL": "#1f77b4",
    "Adapted-CPSL": "#ff7f0e",
    "Adapted-ClusterSFL": "#2ca02c",
    "Adapted-PCSFL": "#9467bd",
}


def _plot_comparison(results, output_dir, run_name):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for axis, (metric, ylabel, title) in zip(axes.flat, METRICS):
        for result in results:
            rows = [row for row in result["summary"] if not row["is_warmup"]]
            epochs = np.asarray([row["epoch"] for row in rows])
            means = np.asarray([
                row[f"{metric}_mean"] if row[f"{metric}_mean"] is not None else np.nan
                for row in rows
            ])
            cis = np.asarray([
                row[f"{metric}_ci95"] if row[f"{metric}_ci95"] is not None else np.nan
                for row in rows
            ])
            algorithm_name = result["algorithm"]
            color = COLORS[algorithm_name]
            axis.plot(epochs, means, color=color, linewidth=2, label=algorithm_name)
            axis.fill_between(epochs, means - cis, means + cis, color=color, alpha=0.12)
        axis.axvline(100, color="#d62728", linestyle="--", linewidth=1, label="Resource shock")
        axis.set_title(title)
        axis.set_xlabel("Communication round")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    path = Path(output_dir) / f"{run_name}_comparison.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return str(path)


def _write_comparison_summary(results, output_dir, run_name, seeds):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    rows = []
    for result in results:
        for row in result["summary"]:
            rows.append({"algorithm": result["algorithm"], **row})
    csv_path = output_dir / f"{run_name}_comparison_summary_{timestamp}.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    trace_ids = {
        result["algorithm"]: [raw["trace_id"] for raw in result["raw_results"]]
        for result in results
    }
    json_path = output_dir / f"{run_name}_comparison_summary_{timestamp}.json"
    json_path.write_text(
        json.dumps({"seeds": list(seeds), "trace_ids": trace_ids, "results": results}, indent=2),
        encoding="utf-8",
    )
    return str(json_path), str(csv_path)


def run_scenario_a_comparison(seeds=(7, 17, 29), log_dir="logs/comparison", run_name="scenario_a_comparison", **kwargs):
    """Run all four methods serially with matching seed-specific Scenario-A traces."""
    seeds = tuple(int(seed) for seed in seeds)
    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    expected_trace_ids = None
    for algorithm in ALGORITHMS:
        result = run_multiseed_scenario_a(
            seeds,
            algorithm=algorithm,
            log_dir=str(output_dir),
            run_name=f"{run_name}_{algorithm}",
            **kwargs,
        )
        trace_ids = [raw["trace_id"] for raw in result["raw_results"]]
        if expected_trace_ids is None:
            expected_trace_ids = trace_ids
        elif trace_ids != expected_trace_ids:
            raise RuntimeError("algorithms did not receive identical Scenario-A traces")
        results.append(result)
    summary_json, summary_csv = _write_comparison_summary(results, output_dir, run_name, seeds)
    comparison_plot = _plot_comparison(results, output_dir, run_name)
    return {
        "algorithm_results": results,
        "summary_json": summary_json,
        "summary_csv": summary_csv,
        "comparison_plot": comparison_plot,
        "trace_ids": expected_trace_ids,
    }


def main():
    parser = argparse.ArgumentParser(description="Run a fair four-controller multi-seed Scenario-A comparison.")
    parser.add_argument("--seeds", nargs="+", type=int, default=(7, 17, 29))
    parser.add_argument("--data-dir", default="../Data")
    parser.add_argument("--log-dir", default="logs/comparison")
    parser.add_argument("--run-name", default="scenario_a_comparison")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-steps", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-bandwidth-share", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--ppo-update-interval", type=int, default=16)
    parser.add_argument("--fedavg-interval", type=int, default=10)
    parser.add_argument("--evaluation-batches", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    result = run_scenario_a_comparison(
        args.seeds,
        log_dir=args.log_dir,
        run_name=args.run_name,
        data_dir=args.data_dir,
        total_epochs=args.epochs,
        batch_size=args.batch_size,
        local_steps=args.local_steps,
        warmup_epochs=args.warmup_epochs,
        min_bandwidth_share=args.min_bandwidth_share,
        learning_rate=args.learning_rate,
        ppo_update_interval=args.ppo_update_interval,
        fedavg_interval=args.fedavg_interval,
        evaluation_batches=args.evaluation_batches,
        device=args.device,
        create_plots=False,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "algorithm_results"}, indent=2))


if __name__ == "__main__":
    main()
