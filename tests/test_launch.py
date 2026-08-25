"""Tests for the self-contained suite launcher."""

from pathlib import Path

import yaml

from spatex.launch import prepare_suite


def _config(path: Path, name: str) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "run_name": name,
                "training": {
                    "output_root": "/old/location",
                    "total_steps": 100,
                    "max_hours": 24,
                    "validation_every": 20,
                    "checkpoint_every": 20,
                },
            }
        )
    )
    return path


def test_main_suite_is_self_contained(tmp_path: Path):
    """Main runs should keep every generated path below the run root."""
    runs_root = tmp_path / "SpatEX-runs"
    suite, arms = prepare_suite(
        [_config(tmp_path / "det.yaml", "deterministic")],
        ["0"],
        runs_root,
        "study",
    )
    assert suite.parent == (runs_root / "main").resolve()
    assert (runs_root / "LATEST_MAIN_ROOT.txt").read_text().strip() == str(suite)
    resolved = yaml.safe_load(Path(str(arms[0]["config"])).read_text())
    assert resolved["training"]["output_root"] == str(suite / "runs")
    assert not (tmp_path / "LATEST_SPATEX_MAIN_ROOT.txt").exists()


def test_smoke_suite_requires_explicit_steps(tmp_path: Path):
    """Smoke overrides should be isolated under the smoke directory."""
    runs_root = tmp_path / "SpatEX-runs"
    suite, arms = prepare_suite(
        [_config(tmp_path / "wae.yaml", "wae")],
        ["2"],
        runs_root,
        "check",
        smoke_steps=7,
    )
    assert suite.parent == (runs_root / "smoke").resolve()
    assert (runs_root / "LATEST_SMOKE_ROOT.txt").read_text().strip() == str(suite)
    resolved = yaml.safe_load(Path(str(arms[0]["config"])).read_text())
    assert resolved["training"]["total_steps"] == 7
    assert resolved["training"]["validation_every"] == 5
