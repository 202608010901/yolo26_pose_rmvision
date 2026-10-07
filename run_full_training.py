"""Run full GPU training and save completion status, curves and sample predictions."""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import yaml

BASE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=BASE / "train_config.yaml")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    name = config["output"]["name"]
    project = Path(config["output"]["project"])
    if not project.is_absolute():
        project = config_path.parent / project
    run_dir = project / name
    log_dir = BASE / "outputs" / "full_training" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.json"
    status = {"status": "running", "started_at": dt.datetime.now().astimezone().isoformat(),
              "supervisor_pid": os.getpid(), "run_dir": str(run_dir),
              "config": config}

    def save_status() -> None:
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(status_path)

    def run_step(script: str, arguments: list[str], log_name: str) -> None:
        status["step"] = script
        save_status()
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}
        with (log_dir / log_name).open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", str(BASE / script), *arguments],
                                       cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT)
            status["child_pid"] = process.pid
            save_status()
            code = process.wait()
        if code:
            raise RuntimeError(f"{script} exited with {code}; see {log_dir / log_name}")

    if os.name == "nt":
        # Keep the system awake while training; release the request on completion.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        save_status()
        run_step("check_dataset.py", [], "dataset_check.log")
        run_step("train.py", ["--config", str(config_path)], "train.log")
        run_step("plot_results.py", ["--run-dir", str(run_dir)], "plot.log")
        data_path = Path(config["data"])
        if not data_path.is_absolute():
            data_path = config_path.parent / data_path
        data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
        sample = Path(data["path"]) / data["val"] / "1004.jpg"
        run_step("predict.py", ["--weights", str(run_dir / "weights" / "best.pt"),
                                "--source", str(sample), "--device", "cuda",
                                "--name", name], "predict.log")
        status.update(status="completed", best_weights=str(run_dir / "weights" / "best.pt"),
                      last_weights=str(run_dir / "weights" / "last.pt"))
        return 0
    except Exception as exc:
        status.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        return 1
    finally:
        status["finished_at"] = dt.datetime.now().astimezone().isoformat()
        save_status()
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    raise SystemExit(main())
