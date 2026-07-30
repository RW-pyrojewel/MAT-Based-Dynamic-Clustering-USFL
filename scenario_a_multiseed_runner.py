"""Sequential multi-seed Scenario-A runner with reproducible aggregate reports."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scenario_a_runner import run_scenario_a


METRICS = (
    "total_delay_ms",
    "tx_delay_ms",
    "compute_delay_ms",
    "reward",
    "delay_normalized",
    "kl_sum",
    "kl_normalized",
    "train_loss",
    "train_accuracy",
    "test_accuracy",
    "physical_cluster_count",
    "max_cluster_size",
    "cluster_size_std",
    "cluster_load_cv",
    "min_bandwidth_share",
    "max_bandwidth_share",
    "bandwidth_gini",
    "bandwidth_cv",
    "bandwidth_floor_hit_rate",
    "mean_l1",
    "mean_l2",
    "smashed_data_bytes_total",
    "ppo_policy_loss",
    "ppo_value_loss",
    "ppo_entropy",
    "ppo_approx_kl",
    "ppo_clip_fraction",
    "ppo_value_clip_fraction",
    "ppo_grad_norm",
    "ppo_grad_norm_pre",
    "ppo_grad_norm_post",
    "ppo_station_1_return_mean",
    "ppo_station_1_return_std",
    "ppo_station_1_explained_variance",
    "ppo_station_2_return_mean",
    "ppo_station_2_return_std",
    "ppo_station_2_explained_variance",
    "ppo_station_3_return_mean",
    "ppo_station_3_return_std",
    "ppo_station_3_explained_variance",
)


def _seed_epoch_records(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["epoch"], []).append(row)
    output = {}
    for epoch, station_rows in grouped.items():
        values = {"epoch": epoch, "is_warmup": bool(station_rows[0].get("is_warmup", False))}
        for metric in METRICS:
            samples = [row[metric] for row in station_rows if row.get(metric) is not None]
            values[metric] = float(np.mean(samples)) if samples else None
        output[epoch] = values
    return output


def _confidence_interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def _summarize(seed_loggers, seeds, algorithm_name):
    per_seed = {
        seed: _seed_epoch_records(logger.records[algorithm_name])
        for seed, logger in zip(seeds, seed_loggers)
    }
    epochs = sorted(set.intersection(*(set(records) for records in per_seed.values())))
    summary = []
    for epoch in epochs:
        record = {"epoch": epoch, "is_warmup": per_seed[seeds[0]][epoch]["is_warmup"], "seed_count": len(seeds)}
        for metric in METRICS:
            values = [per_seed[seed][epoch][metric] for seed in seeds if per_seed[seed][epoch][metric] is not None]
            if values:
                record[f"{metric}_mean"] = float(np.mean(values))
                record[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                record[f"{metric}_ci95"] = _confidence_interval(values)
            else:
                record[f"{metric}_mean"] = None
                record[f"{metric}_std"] = None
                record[f"{metric}_ci95"] = None
        summary.append(record)
    return summary


def _write_summary(summary, seeds, output_dir, run_name, metadata):
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{run_name}_summary_{timestamp}.json"
    csv_path = output_dir / f"{run_name}_summary_{timestamp}.csv"
    json_path.write_text(
        json.dumps({"seeds": list(seeds), "metadata": metadata, "epochs": summary}, indent=2),
        encoding="utf-8",
    )
    fieldnames = sorted({key for row in summary for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    return str(json_path), str(csv_path)


def _plot_summary(summary, output_dir, run_name):
    rows = [row for row in summary if not row["is_warmup"]]
    epochs = np.asarray([row["epoch"] for row in rows])
    plots = (
        ("total_delay_ms", "System delay (ms)", "Scenario A: Multi-seed System Delay", "#1f77b4"),
        ("reward", "Reward", "Scenario A: Multi-seed Reward", "#2ca02c"),
        ("train_loss", "Cross-entropy loss", "Scenario A: Multi-seed USFL Loss", "#8c564b"),
        ("test_accuracy", "Test accuracy", "Scenario A: Multi-seed Test Accuracy", "#9467bd"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, (metric, ylabel, title, color) in zip(axes.flat, plots):
        means = np.asarray([row[f"{metric}_mean"] if row[f"{metric}_mean"] is not None else np.nan for row in rows])
        cis = np.asarray([row[f"{metric}_ci95"] if row[f"{metric}_ci95"] is not None else np.nan for row in rows])
        axis.plot(epochs, means, color=color, linewidth=2, label="Seed mean")
        axis.fill_between(epochs, means - cis, means + cis, color=color, alpha=0.2, label="95% seed CI")
        axis.axvline(100, color="#d62728", linestyle="--", linewidth=1, label="Resource shock")
        axis.set_title(title)
        axis.set_xlabel("Communication round")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    path = Path(output_dir) / f"{run_name}_summary.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return str(path)


def run_multiseed_scenario_a(seeds, algorithm="mat", log_dir="logs/multiseed", run_name=None, **kwargs):
    """Run Scenario A serially for every seed and save raw and aggregate results."""
    seeds = tuple(int(seed) for seed in seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("provide at least two distinct random seeds")
    algorithm = algorithm.lower()
    run_name = run_name or f"scenario_a_{algorithm}_multiseed"
    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loggers = []
    raw_results = []
    for seed in seeds:
        result = run_scenario_a(
            algorithm=algorithm,
            log_dir=str(output_dir),
            seed=seed,
            run_name=f"{run_name}_seed_{seed}",
            **kwargs,
        )
        loggers.append(result["logger"])
        raw_results.append({key: value for key, value in result.items() if key != "logger"})
    algorithm_name = loggers[0].metadata["algorithm"]
    if any(logger.metadata["algorithm"] != algorithm_name for logger in loggers):
        raise RuntimeError("multi-seed run mixed algorithm identities")
    summary = _summarize(loggers, seeds, algorithm_name)
    metadata = dict(loggers[0].metadata)
    metadata["seeds"] = list(seeds)
    metadata["run_name"] = run_name
    summary_json, summary_csv = _write_summary(summary, seeds, output_dir, run_name, metadata)
    summary_plot = _plot_summary(summary, output_dir, run_name)
    return {
        "raw_results": raw_results,
        "summary_json": summary_json,
        "summary_csv": summary_csv,
        "summary_plot": summary_plot,
        "summary": summary,
        "algorithm": algorithm_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Run reproducible multi-seed Scenario-A experiments.")
    parser.add_argument("--seeds", nargs="+", type=int, default=(7, 17, 29))
    parser.add_argument("--algorithm", choices=("mat", "cpsl", "clustersfl", "pcsfl"), default="mat")
    parser.add_argument("--data-dir", default="../Data")
    parser.add_argument("--log-dir", default="logs/multiseed")
    parser.add_argument("--run-name", default="scenario_a_mat_multiseed")
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
    result = run_multiseed_scenario_a(
        args.seeds,
        algorithm=args.algorithm,
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
        create_plots=True,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
