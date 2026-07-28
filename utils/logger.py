"""Structured logging and English chart generation for AI-RAN experiments."""
import csv
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class SimulationLogger:
    """Collect experiment records and export portable data and publication-ready figures."""

    def __init__(self, log_dir="logs", run_name="scenario_a"):
        self.log_dir = log_dir
        self.run_name = run_name
        os.makedirs(self.log_dir, exist_ok=True)
        self.records = {}
        self.metadata = {}

    def set_metadata(self, **metadata):
        self.metadata.update(metadata)

    def log_metrics(self, algo_name, **metrics):
        if "epoch" not in metrics:
            raise ValueError("every log record must include epoch")
        self.records.setdefault(algo_name, []).append(dict(metrics))

    def log_step(self, algo_name, epoch, n_clients, n_migs, bandwidth, total_delay, tx_delay=0.0, comp_delay=0.0):
        """Compatibility wrapper for the original simulation runner."""
        self.log_metrics(
            algo_name,
            epoch=int(epoch),
            vehicle_count=int(n_clients),
            available_migs=int(n_migs),
            bandwidth_hz=float(bandwidth),
            total_delay_ms=float(total_delay) * 1000.0,
            tx_delay_ms=float(tx_delay) * 1000.0,
            compute_delay_ms=float(comp_delay) * 1000.0,
        )

    def _timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def export_to_csv(self, filename_prefix=None):
        timestamp = self._timestamp()
        prefix = filename_prefix or self.run_name
        exported = []
        for algo_name, rows in self.records.items():
            if not rows:
                continue
            fieldnames = sorted({key for row in rows for key in row})
            safe_name = algo_name.replace(" ", "_").replace("-", "_")
            path = os.path.join(self.log_dir, f"{prefix}_{safe_name}_{timestamp}.csv")
            with open(path, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            exported.append(path)
        return exported

    def export_to_json(self, filename_prefix=None):
        timestamp = self._timestamp()
        prefix = filename_prefix or self.run_name
        path = os.path.join(self.log_dir, f"{prefix}_{timestamp}.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump({"metadata": self.metadata, "records": self.records}, file, indent=2)
        return path

    @staticmethod
    def _rolling_mean(values, window):
        return np.asarray([
            np.mean(values[max(0, index - window + 1):index + 1])
            for index in range(len(values))
        ])

    def plot_scenario_a(self, algo_name, filename_prefix=None, warmup_epochs=0, rolling_window=5):
        """Create English Scenario-A figures excluding configured warmup rounds."""
        rows = self.records.get(algo_name, [])
        if not rows:
            raise ValueError(f"no records available for {algo_name}")
        rows = [
            row for row in rows
            if not row.get("is_warmup", False) and row["epoch"] > warmup_epochs
        ]
        if not rows:
            raise ValueError("no post-warmup records available for plotting")
        prefix = filename_prefix or self.run_name
        by_epoch = {}
        for row in rows:
            by_epoch.setdefault(row["epoch"], []).append(row)
        epochs = sorted(by_epoch)
        delay_samples = [np.asarray([row["total_delay_ms"] for row in by_epoch[epoch]], dtype=float) for epoch in epochs]
        reward_samples = [np.asarray([row["reward"] for row in by_epoch[epoch]], dtype=float) for epoch in epochs]
        mean_delay = np.asarray([sample.mean() for sample in delay_samples])
        mean_reward = np.asarray([sample.mean() for sample in reward_samples])
        delay_ci = np.asarray([1.96 * sample.std(ddof=1) / np.sqrt(len(sample)) if len(sample) > 1 else 0.0 for sample in delay_samples])
        reward_ci = np.asarray([1.96 * sample.std(ddof=1) / np.sqrt(len(sample)) if len(sample) > 1 else 0.0 for sample in reward_samples])
        delay_rolling = self._rolling_mean(mean_delay, rolling_window)
        reward_rolling = self._rolling_mean(mean_reward, rolling_window)
        mean_loss = np.asarray([
            np.mean([row["train_loss"] for row in by_epoch[epoch] if row.get("train_loss") is not None])
            if any(row.get("train_loss") is not None for row in by_epoch[epoch]) else np.nan
            for epoch in epochs
        ])

        generated = []
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
        axes[0].plot(epochs, delay_rolling, color="#1f77b4", linewidth=2, label=f"{rolling_window}-round mean delay")
        axes[0].fill_between(epochs, mean_delay - delay_ci, mean_delay + delay_ci, color="#1f77b4", alpha=0.18, label="95% station CI")
        axes[0].axvline(100, color="#d62728", linestyle="--", label="Resource shock")
        axes[0].set_ylabel("Delay (ms)")
        axes[0].set_title("Scenario A: MAT-RL Latency and Reward")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        axes[1].plot(epochs, reward_rolling, color="#2ca02c", linewidth=2, label=f"{rolling_window}-round mean reward")
        axes[1].fill_between(epochs, mean_reward - reward_ci, mean_reward + reward_ci, color="#2ca02c", alpha=0.18, label="95% station CI")
        axes[1].set_xlabel("Communication round")
        axes[1].set_ylabel("Reward")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        path = os.path.join(self.log_dir, f"{prefix}_latency_reward.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        generated.append(path)

        fig, axis = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
        for station_id, color in ((1, "#9467bd"), (2, "#ff7f0e"), (3, "#17becf")):
            resources = [next(row for row in by_epoch[epoch] if row["station_id"] == station_id) for epoch in epochs]
            axis.step(epochs, [row["available_migs"] for row in resources], where="post", linewidth=2, color=color, label=f"Station {station_id}: available MIGs")
        axis.set_title("Scenario A: Available MIG Tide")
        axis.set_xlabel("Communication round")
        axis.set_ylabel("Available MIGs")
        axis.set_ylim(0, 8)
        axis.grid(alpha=0.3)
        axis.legend(ncol=3, fontsize=8)
        path = os.path.join(self.log_dir, f"{prefix}_mig_tide.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        generated.append(path)

        if np.isfinite(mean_loss).any():
            fig, axis = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
            axis.plot(epochs, self._rolling_mean(mean_loss, rolling_window), color="#8c564b", linewidth=2, label=f"{rolling_window}-round mean loss")
            axis.set_title("Scenario A: USFL Training Loss")
            axis.set_xlabel("Communication round")
            axis.set_ylabel("Cross-entropy loss")
            axis.grid(alpha=0.3)
            axis.legend()
            path = os.path.join(self.log_dir, f"{prefix}_training_loss.png")
            fig.savefig(path, dpi=200)
            plt.close(fig)
            generated.append(path)
        return generated
