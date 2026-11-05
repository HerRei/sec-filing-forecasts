#!/usr/bin/env python3
"""Run every pre-registered Gate 4 job on the CPU: baselines per fold, then the neural runs.

For example: PYTHONPATH=src python scripts/run_g4.py --workers 3 --threads 6
Completed outputs are skipped, so a restart continues where it stopped.
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ("final", "dev2022", "dev2021", "dev2020", "dev2019")  # the primary fold first
INITS = ("broad", "scratch", "varied")
LOG = ROOT / "runs/gate4/run_g4.jsonl"


def log(record):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    record["time"] = time.time()
    with LOG.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


THREADS = {"n": 6}


def run(command):
    import os

    started = time.time()
    # OpenMP/BLAS threads must match the CPU quota share, or sklearn's trees thrash.
    env = {
        **os.environ,
        "OMP_NUM_THREADS": str(THREADS["n"]),
        "MKL_NUM_THREADS": str(THREADS["n"]),
        "OPENBLAS_NUM_THREADS": str(THREADS["n"]),
    }
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    log(
        {
            "command": command[3:],
            "rc": result.returncode,
            "seconds": round(time.time() - started),
            "tail": (result.stdout + result.stderr)[-600:],
        }
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--skip-baselines", action="store_true")
    a = parser.parse_args()
    THREADS["n"] = a.threads
    py = [sys.executable, "-m", "disclosure_forecasting.real.g4"]
    for fold in FOLDS if not a.skip_baselines else ():
        if not (ROOT / f"runs/gate4/{fold}/baselines.json").exists():
            run(py + ["baselines", fold])
    jobs = []
    for fold in FOLDS:
        for seed in (1, 2, 3):
            for init in INITS:
                if not (ROOT / f"runs/gate4/{fold}/{init}-ft-s{seed}.npz").exists():
                    jobs.append(py + ["neural", fold, init, str(seed), "--threads", str(a.threads)])
    log({"event": "start", "jobs": len(jobs)})
    with ThreadPoolExecutor(a.workers) as pool:
        codes = list(pool.map(run, jobs))
    log({"event": "done", "failed": sum(c != 0 for c in codes)})


if __name__ == "__main__":
    main()
