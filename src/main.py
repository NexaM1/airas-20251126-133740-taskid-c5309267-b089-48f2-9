# src/main.py
"""Orchestrator that launches *one* experiment subprocess via Hydra."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import hydra


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):  # noqa: D401
    if not cfg.run:
        raise ValueError("run=RUN_ID must be supplied on the CLI.")

    root = Path(__file__).resolve().parents[1]

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"mode={cfg.mode}",
        f"results_dir={cfg.results_dir}",
    ]
    print("[main] launching:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
