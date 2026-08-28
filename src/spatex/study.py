"""Launch, monitor, and evaluate the presentation ablation study."""

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
from spatex.launch import launch_suite
from spatex.panels import fit_panels


ARMS = (
    "local_seed10",
    "local_seed1",
    "spatial_base_seed10",
    "within_seed10",
    "between_seed10",
    "full_seed10",
    "full_seed1",
    "full_seed2",
)


def _alive(pid: int) -> bool:
    """Return whether a process still exists."""
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        try:
            if stat.read_text().split()[2] == "Z":
                return False
        except (IndexError, OSError):
            pass
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _last_step(log: Path) -> str:
    """Read the latest logged training step without loading the full log."""
    if not log.is_file():
        return "-"
    lines = log.read_text(errors="replace").splitlines()
    for line in reversed(lines):
        if line.startswith("[step "):
            return line.split("]", 1)[0].removeprefix("[step ")
    return "initializing"


def launch(
    repo: Path,
    data_root: Path,
    runs_root: Path,
    gpus: list[str],
    hours: float,
    port: int | None,
) -> Path:
    """Resolve server paths, fit panels, and launch the eight study arms."""
    if len(gpus) != len(ARMS):
        raise ValueError(f"the study requires {len(ARMS)} GPU assignments")
    manifest = data_root / "manifest.json"
    structure = data_root / "centered_gene_structure.pt"
    panels = data_root / "train_gene_panels.json"
    for path in (manifest, structure):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not panels.is_file():
        fit_panels(manifest, panels)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    request = runs_root / "requests" / f"presentation_study_{stamp}"
    request.mkdir(parents=True, exist_ok=False)
    configs: list[Path] = []
    for arm in ARMS:
        source = repo / "configs" / "study" / f"{arm}.yaml"
        config = load_config(source)
        config["data"]["manifest"] = str(manifest)
        config["data"]["structure"] = str(structure)
        config["evaluation"]["panel_file"] = str(panels)
        config["training"]["max_hours"] = float(hours)
        destination = request / f"{arm}.yaml"
        destination.write_text(yaml.safe_dump(config, sort_keys=False))
        configs.append(destination)

    return launch_suite(
        configs,
        gpus,
        runs_root,
        "presentation_study",
        tensorboard_port=port,
    )


def monitor(root: Path) -> None:
    """Print one compact line per arm and a GPU summary."""
    suite = json.loads((root / "suite.json").read_text())
    print(time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    for arm in suite["arms"]:
        name = str(arm["arm"])
        pid_path = root / "control" / f"{name}.pid"
        pid = int(pid_path.read_text()) if pid_path.is_file() else -1
        state = "running" if pid > 0 and _alive(pid) else "ended"
        print(
            f"{name:28s} GPU={arm['gpu']} {state:7s} "
            f"PID={pid if pid > 0 else '-'} step={_last_step(Path(arm['log']))}"
        )
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        print("\nGPUs (index, MiB used/total, utilization):")
        for line in output.splitlines():
            index = line.split(",", 1)[0].strip()
            if index in {str(value["gpu"]) for value in suite["arms"]}:
                print(line)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass


def evaluate_suite(root: Path, panel_file: Path, gpus: list[str]) -> Path:
    """Launch one whole-slide evaluation per completed training arm."""
    suite = json.loads((root / "suite.json").read_text())
    if len(gpus) != len(suite["arms"]):
        raise ValueError("one GPU assignment is required per evaluation")
    output_root = root / f"evaluation_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    (output_root / "logs").mkdir(parents=True)
    status: dict[str, dict[str, object]] = {}
    for arm, gpu in zip(suite["arms"], gpus, strict=True):
        name = str(arm["arm"])
        candidates = sorted((root / "runs").glob(f"{name}_*"))
        if len(candidates) != 1:
            raise ValueError(f"{name}: expected one run directory, found {len(candidates)}")
        checkpoint = candidates[0] / "checkpoints" / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output = output_root / f"{name}_validation.json"
        log = output_root / "logs" / f"{name}.log"
        handle = log.open("w")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[variable] = "6"
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "spatex.evaluate",
                "--config",
                str(arm["config"]),
                "--checkpoint",
                str(checkpoint),
                "--panel-file",
                str(panel_file),
                "--output",
                str(output),
            ],
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        handle.close()
        status[name] = {"gpu": gpu, "pid": process.pid, "output": str(output)}
    (output_root / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(output_root)
    return output_root


def _wait_for_jobs(
    jobs: dict[str, tuple[int, Path | None]], poll_seconds: int, label: str
) -> None:
    """Wait for detached jobs and fail if a required output is missing."""
    while True:
        running = [name for name, (pid, _) in jobs.items() if _alive(pid)]
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} "
            f"{label}: {len(running)}/{len(jobs)} running",
            flush=True,
        )
        if not running:
            break
        time.sleep(poll_seconds)
    missing = [
        name
        for name, (_, output) in jobs.items()
        if output is not None and not output.is_file()
    ]
    if missing:
        raise RuntimeError(f"{label} failed or incomplete: {', '.join(missing)}")


def finalize_suite(
    root: Path,
    panel_file: Path,
    gpus: list[str],
    poll_seconds: int = 60,
) -> Path:
    """Wait for training, evaluate every arm, and export presentation figures."""
    suite = json.loads((root / "suite.json").read_text())
    training_jobs: dict[str, tuple[int, Path | None]] = {}
    for arm in suite["arms"]:
        name = str(arm["arm"])
        pid = int((root / "control" / f"{name}.pid").read_text())
        training_jobs[name] = (pid, None)
    _wait_for_jobs(training_jobs, poll_seconds, "training")

    evaluation_root = evaluate_suite(root, panel_file, gpus)
    evaluation_status = json.loads((evaluation_root / "status.json").read_text())
    evaluation_jobs = {
        name: (int(values["pid"]), Path(str(values["output"])))
        for name, values in evaluation_status.items()
    }
    _wait_for_jobs(evaluation_jobs, poll_seconds, "evaluation")

    from spatex.presentation import _reports, export

    arguments = [
        f"{name}={values['output']}"
        for name, values in evaluation_status.items()
    ]
    presentation_root = root / f"presentation_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    export(_reports(arguments), presentation_root)
    result = {
        "evaluation_root": str(evaluation_root),
        "presentation_root": str(presentation_root),
    }
    (root / "finalization.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return presentation_root


def main() -> None:
    """Run one presentation-study operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--repo", default=".")
    launch_parser.add_argument("--data-root", required=True)
    launch_parser.add_argument("--runs-root", required=True)
    launch_parser.add_argument("--gpus", required=True)
    launch_parser.add_argument("--hours", type=float, default=8.0)
    launch_parser.add_argument("--tensorboard-port", type=int)
    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--root", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--root", required=True)
    evaluate_parser.add_argument("--panel-file", required=True)
    evaluate_parser.add_argument("--gpus", required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--root", required=True)
    finalize_parser.add_argument("--panel-file", required=True)
    finalize_parser.add_argument("--gpus", required=True)
    finalize_parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.command == "launch":
        root = launch(
            Path(args.repo).expanduser().resolve(),
            Path(args.data_root).expanduser().resolve(),
            Path(args.runs_root).expanduser().resolve(),
            [value.strip() for value in args.gpus.split(",")],
            args.hours,
            args.tensorboard_port,
        )
        print(root)
    elif args.command == "monitor":
        monitor(Path(args.root).expanduser().resolve())
    elif args.command == "evaluate":
        evaluate_suite(
            Path(args.root).expanduser().resolve(),
            Path(args.panel_file).expanduser().resolve(),
            [value.strip() for value in args.gpus.split(",")],
        )
    else:
        finalize_suite(
            Path(args.root).expanduser().resolve(),
            Path(args.panel_file).expanduser().resolve(),
            [value.strip() for value in args.gpus.split(",")],
            max(5, args.poll_seconds),
        )


if __name__ == "__main__":
    main()
