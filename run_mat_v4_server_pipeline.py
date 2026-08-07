"""Run the complete MAT-v4 server experiment pipeline sequentially.

Stages are isolated in child Python processes so CUDA memory, dataloaders and
optimizers from one experiment cannot leak into the next one.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _latest_directory(root, pattern):
    matches = [path for path in root.glob(pattern) if path.is_dir()]
    if not matches:
        raise RuntimeError(f"stage completed without creating {pattern!r} under {root}")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _run_stage(name, command, root, pipeline_report):
    log_path = root / f"{name}.log"
    print(f"\n[PIPELINE] starting {name}", flush=True)
    print("[PIPELINE] " + " ".join(map(str, command)), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            command, cwd=Path(__file__).resolve().parent,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    pipeline_report["stages"][name] = {
        "command": list(map(str, command)), "log": str(log_path),
        "return_code": int(return_code),
    }
    _write_report(root, pipeline_report)
    if return_code != 0:
        raise RuntimeError(f"{name} failed with exit code {return_code}; see {log_path}")


def _read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_report(root, report):
    destination = root / "pipeline_report.json"
    temporary = root / "pipeline_report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(destination)


def _common_arguments(args, include_ppo_interval=False):
    values = [
        "--data-dir", str(args.data_dir), "--log-dir", str(args.pipeline_root),
        "--device", str(args.device), "--batch-size", str(args.batch_size),
        "--local-steps", str(args.local_steps),
        "--evaluation-batches", str(args.evaluation_batches),
    ]
    if include_ppo_interval:
        values.extend(["--ppo-update-interval", str(args.ppo_update_interval)])
    return values


def run_pipeline(args):
    code_root = Path(__file__).resolve().parent
    log_root = Path(args.log_dir).expanduser()
    if not log_root.is_absolute():
        log_root = code_root / log_root
    pipeline_root = log_root / f"mat_v4_pipeline_{datetime.now():%Y%m%d_%H%M%S}"
    pipeline_root.mkdir(parents=True, exist_ok=False)
    args.pipeline_root = pipeline_root
    report = {
        "schema_version": 1, "status": "running",
        "started_at": datetime.now().isoformat(),
        "python": sys.executable, "pipeline_root": str(pipeline_root), "stages": {},
    }
    _write_report(pipeline_root, report)
    try:
        development_command = [
            sys.executable, str(code_root / "mat_physics_guided_online_runner.py"),
            "--mode", "development", *_common_arguments(args, include_ppo_interval=True),
        ]
        _run_stage("01_development_seed41", development_command, pipeline_root, report)
        development_root = _latest_directory(
            pipeline_root, "mat_physics_guided_v4_development_*")
        development_report_path = development_root / "mat_v4_report.json"
        development_report = _read_json(development_report_path)
        report["stages"]["01_development_seed41"].update({
            "result_root": str(development_root), "report": str(development_report_path),
            "passed": bool(development_report.get("passed")),
        })
        _write_report(pipeline_root, report)
        if not development_report.get("passed"):
            raise RuntimeError(
                "seed-41 development gate failed; formal comparison was not started")

        baseline_command = [
            sys.executable, str(code_root / "scenario_a_baseline_cache_runner.py"),
            *_common_arguments(args),
        ]
        _run_stage("02_baseline_cache", baseline_command, pipeline_root, report)
        baseline_root = _latest_directory(pipeline_root, "scenario_a_baselines_v4_*")
        baseline_report_path = baseline_root / "baseline_contract_report.json"
        baseline_report = _read_json(baseline_report_path)
        report["stages"]["02_baseline_cache"].update({
            "result_root": str(baseline_root), "report": str(baseline_report_path),
            "completed": baseline_report.get("status") == "completed",
            "contract_hash": baseline_report.get("contract_hash"),
        })
        _write_report(pipeline_root, report)
        if baseline_report.get("status") != "completed":
            raise RuntimeError("baseline cache report is incomplete")

        formal_command = [
            sys.executable, str(code_root / "mat_physics_guided_online_runner.py"),
            "--mode", "formal", "--baseline-report", str(baseline_report_path),
            *_common_arguments(args, include_ppo_interval=True),
        ]
        _run_stage("03_formal_mat_and_physics_only", formal_command, pipeline_root, report)
        formal_root = _latest_directory(pipeline_root, "mat_physics_guided_v4_formal_*")
        formal_report_path = formal_root / "mat_v4_report.json"
        formal_report = _read_json(formal_report_path)
        report["stages"]["03_formal_mat_and_physics_only"].update({
            "result_root": str(formal_root), "report": str(formal_report_path),
            "passed": bool(formal_report.get("passed")),
            "gates": formal_report.get("gates", {}),
        })
        report["status"] = "completed"
        report["passed"] = bool(formal_report.get("passed"))
    except BaseException as error:
        report["status"] = "failed"
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_at"] = datetime.now().isoformat()
        _write_report(pipeline_root, report)
        print(f"\n[PIPELINE] report: {pipeline_root / 'pipeline_report.json'}", flush=True)
    return pipeline_root, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../Data")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-steps", type=int, default=4)
    parser.add_argument("--evaluation-batches", type=int, default=10)
    parser.add_argument("--ppo-update-interval", type=int, default=48)
    args = parser.parse_args()
    root, report = run_pipeline(args)
    print(json.dumps({"root": str(root), "status": report["status"],
                      "passed": report["passed"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
