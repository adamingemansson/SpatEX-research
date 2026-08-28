"""Launch self-contained SpatEX training suites."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from spatex.config import load_config


def _timestamp() -> str:
    """Return a UTC timestamp suitable for directory names."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _write_pointer(path: Path, target: Path) -> None:
    """Replace a latest-run pointer atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(str(target) + "\n")
    temporary.replace(path)


def prepare_suite(
    config_paths: list[str | Path],
    gpus: list[str],
    runs_root: str | Path,
    name: str,
    smoke_steps: int | None = None,
) -> tuple[Path, list[dict[str, object]]]:
    """Create one suite directory and resolved per-arm configurations."""
    if len(config_paths) != len(gpus):
        raise ValueError("one GPU is required for each configuration")
    if smoke_steps is not None and smoke_steps < 1:
        raise ValueError("smoke_steps must be positive")

    runs_root = Path(runs_root).expanduser().resolve()
    category = "smoke" if smoke_steps is not None else "main"
    suite_root = runs_root / category / f"{name}_{_timestamp()}"
    for directory in ("configs", "logs", "control", "runs"):
        (suite_root / directory).mkdir(parents=True, exist_ok=False)

    arms: list[dict[str, object]] = []
    for source, gpu in zip(config_paths, gpus, strict=True):
        source = Path(source).expanduser().resolve()
        config = load_config(source)
        arm = str(config.get("run_name") or source.stem)
        config["run_name"] = arm
        config.setdefault("training", {})["output_root"] = str(suite_root / "runs")
        if smoke_steps is not None:
            config["training"]["total_steps"] = smoke_steps
            config["training"]["max_hours"] = 1.0
            interval = max(1, min(smoke_steps, 5))
            config["training"]["validation_every"] = interval
            config["training"]["checkpoint_every"] = interval

        destination = suite_root / "configs" / f"{arm}.yaml"
        destination.write_text(yaml.safe_dump(config, sort_keys=False))
        arms.append(
            {
                "arm": arm,
                "gpu": str(gpu),
                "source_config": str(source),
                "config": str(destination),
                "log": str(suite_root / "logs" / f"{arm}.log"),
            }
        )

    manifest = {
        "name": name,
        "category": category,
        "created_utc": _timestamp(),
        "root": str(suite_root),
        "arms": arms,
    }
    (suite_root / "suite.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pointer = runs_root / ("LATEST_SMOKE_ROOT.txt" if smoke_steps is not None else "LATEST_MAIN_ROOT.txt")
    _write_pointer(pointer, suite_root)
    return suite_root, arms


def launch_suite(
    config_paths: list[str | Path],
    gpus: list[str],
    runs_root: str | Path,
    name: str,
    tensorboard_port: int | None = None,
    smoke_steps: int | None = None,
    dry_run: bool = False,
) -> Path:
    """Prepare a suite and optionally start its trainers and TensorBoard."""
    suite_root, arms = prepare_suite(
        config_paths=config_paths,
        gpus=gpus,
        runs_root=runs_root,
        name=name,
        smoke_steps=smoke_steps,
    )
    if dry_run:
        return suite_root

    status: dict[str, object] = {}
    for arm in arms:
        log_path = Path(str(arm["log"]))
        log_handle = log_path.open("w")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(arm["gpu"])
        threads = str(load_config(str(arm["config"]))["training"].get("cpu_threads", 6))
        for name in (
            "SCILIFESTDL_CPU_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[name] = threads
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "spatex.train", "--config", str(arm["config"])],
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_handle.close()
        Path(suite_root / "control" / f"{arm['arm']}.pid").write_text(f"{process.pid}\n")
        status[str(arm["arm"])] = {"gpu": arm["gpu"], "pid": process.pid, "state": "running"}

    if tensorboard_port is not None:
        log_path = suite_root / "logs" / "tensorboard.log"
        log_handle = log_path.open("w")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tensorboard.main",
                "--logdir",
                str(suite_root / "runs"),
                "--host",
                "127.0.0.1",
                "--port",
                str(tensorboard_port),
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_handle.close()
        (suite_root / "control" / "tensorboard.pid").write_text(f"{process.pid}\n")
        (suite_root / "control" / "tensorboard.port").write_text(f"{tensorboard_port}\n")

    (suite_root / "control" / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return suite_root


def main() -> None:
    """Launch a main run or an explicitly requested smoke suite."""
    parser = argparse.ArgumentParser(description="Launch a self-contained SpatEX suite")
    parser.add_argument("--config", action="append", required=True, dest="configs")
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU IDs")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--name", default="spatex")
    parser.add_argument("--tensorboard-port", type=int)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    root = launch_suite(
        config_paths=args.configs,
        gpus=gpus,
        runs_root=args.runs_root,
        name=args.name,
        tensorboard_port=args.tensorboard_port,
        smoke_steps=args.smoke_steps,
        dry_run=args.dry_run,
    )
    print(root)


if __name__ == "__main__":
    main()
