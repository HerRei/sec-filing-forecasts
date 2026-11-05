#!/usr/bin/env python3
"""Write the pre-registered Stage S2 configs and plan (see docs/PREREGISTRATION_S2.md)."""

import json
import math
from pathlib import Path

root = Path(__file__).resolve().parents[1]
BATCH, FACTS, MARKET, HORIZONS = 64, 128, 96, 3
PER_STEP = BATCH * (FACTS + MARKET + HORIZONS)


def steps(tokens):
    return math.ceil(tokens / PER_STEP)


def config(name, family, memory, seed, budgets, fact_budget=None):
    milestones = [steps(b) for b in budgets]
    value = {
        "name": name,
        "family": family,
        "memory": memory,
        "seed": seed,
        "width": 256,
        "layers": 6,
        "heads": 8,
        "batch_size": BATCH,
        "steps": milestones[-1],
        "lr": 5e-4,
        "warmup_steps": 300,
        "weight_decay": 0.01,
        "fact_tokens": FACTS,
        "market_tokens": MARKET,
        "milestones": milestones,
        "monitor_every": 500,
        "workers": 8,
    }
    if fact_budget:
        value["fact_budget"] = fact_budget
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


Q1 = {  # memory under a fact-token budget, varied prior, Stage S budget
    "B-k16": ("varied", "sequence", 16),
    "D-k16": ("varied", "revision", 16),
    "B-k32": ("varied", "sequence", 32),
    "D-k32": ("varied", "revision", 32),
}
Q2 = {"B300": ("varied", "sequence"), "E300": ("broad", "sequence")}  # broader prior, 3x data
runs = []
for seed in (1, 2, 3):  # seed-major across both questions
    for name, (family, memory, budget) in Q1.items():
        run = f"{name}-s{seed}"
        write(
            root / f"configs/stage-s2/{run}.json",
            config(run, family, memory, seed, (10e6, 30e6, 100e6), budget),
        )
        runs.append(f"configs/stage-s2/{run}.json")
    for name, (family, memory) in Q2.items():
        run = f"{name}-s{seed}"
        write(
            root / f"configs/stage-s2/{run}.json",
            config(run, family, memory, seed, (10e6, 30e6, 100e6, 300e6)),
        )
        runs.append(f"configs/stage-s2/{run}.json")
write(
    root / "configs/stage-s2/plan.json",
    {
        "name": "stage-s2",
        "output": "runs/stage-s2",
        "runs": runs,
        "device": "cuda:0",
        "precision": "bf16",
        "eval_dir": "data/synthetic/stage-s2",
        "retries": 1,
        "report": True,
        "report_module": "disclosure_forecasting.report_s2",
        "references": ["gbdt-varied", "gbdt-broad"],
        "requires": [
            "docs/PREREGISTRATION_S2.md",
            "runs/stage-s2/baselines/gbdt-varied/predictions-final.npz",
            "runs/stage-s2/baselines/gbdt-broad/predictions-final.npz",
        ],
    },
)
total = sum(json.loads((root / r).read_text())["steps"] for r in runs)
print(json.dumps({"runs": len(runs), "total_steps": total, "q2_steps": steps(300e6)}))
