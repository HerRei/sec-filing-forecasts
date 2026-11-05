#!/usr/bin/env python3
"""Write the pre-registered Stage S configs (A/B/C/D x seeds 1-3) and the run plans."""

import json
import math
from pathlib import Path

root = Path(__file__).resolve().parents[1]
BATCH, FACTS, MARKET, HORIZONS = 64, 128, 96, 3
PER_STEP = BATCH * (FACTS + MARKET + HORIZONS)
BUDGETS = (10_000_000, 30_000_000, 100_000_000)  # dossier's first scaling points, padded tokens
VARIANTS = {
    "A": ("regular", "sequence"),
    "B": ("varied", "sequence"),
    "C": ("regular", "revision"),
    "D": ("varied", "revision"),
}


def config(name, family, memory, seed, budgets):
    milestones = [math.ceil(b / PER_STEP) for b in budgets]
    return {
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


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


runs = []
for seed in (1, 2, 3):  # seed-major: an interrupted study still has whole seeds
    for variant, (family, memory) in VARIANTS.items():
        name = f"{variant}-s{seed}"
        write(root / f"configs/stage-s/{name}.json", config(name, family, memory, seed, BUDGETS))
        runs.append(f"configs/stage-s/{name}.json")
common = {
    "device": "cuda:0",
    "precision": "bf16",
    "eval_dir": "data/synthetic/stage-s-v2",
    "retries": 1,
}
write(
    root / "configs/stage-s/plan.json",
    {
        "name": "stage-s",
        "output": "runs/stage-s",
        "runs": runs,
        "report": True,
        "requires": [
            "docs/PREREGISTRATION.md",
            "runs/stage-s/baselines/gbdt-regular/predictions-final.npz",
            "runs/stage-s/baselines/gbdt-varied/predictions-final.npz",
        ],
        **common,
    },
)
# Operational checks only (profile + pause/resume); seed 9 is never part of the study.
write(
    root / "configs/checks/profile-B.json",
    config("profile-B", "varied", "sequence", 9, (2_000_000, 4_000_000)),
)
write(
    root / "configs/checks/profile-D.json",
    config("profile-D", "varied", "revision", 9, (2_000_000, 4_000_000)),
)
write(
    root / "configs/checks/handoff-D.json",
    config("handoff-D", "varied", "revision", 9, (3_000_000, 30_000_000)),
)
write(
    root / "configs/checks/profile.json",
    {
        "name": "check-profile",
        "output": "runs/check-profile",
        "runs": ["configs/checks/profile-B.json", "configs/checks/profile-D.json"],
        **common,
    },
)
write(
    root / "configs/checks/handoff.json",
    {
        "name": "check-handoff",
        "output": "runs/check-handoff",
        "runs": ["configs/checks/handoff-D.json"],
        **common,
    },
)
print(
    json.dumps(
        {
            "tokens_per_step": PER_STEP,
            "steps": config("x", "varied", "sequence", 1, BUDGETS)["milestones"],
            "runs": len(runs),
        }
    )
)
