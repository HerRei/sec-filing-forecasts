"""Deterministic synthetic streams and frozen evaluation sets.

World seeds never overlap: training uses seed * 10**9 + index (seed >= 1), evaluation uses
10**8 + family * 10**6 + offset, with offsets below 500,000 for the monitor split and from
500,000 for the test split. Variants with the same training family and seed therefore see
exactly the same worlds in the same order.
"""

import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from .simulator import (
    ALL_FAMILIES,
    EVAL_FAMILIES,
    HELDOUT_FAMILIES,
    decision_for,
    encode,
    references,
    simulate,
    summary_features,
)

SPLITS = {"monitor": 0, "test": 500_000}


def train_seed(seed: int, index: int) -> int:
    if seed < 1:
        raise ValueError("training seeds start at 1")
    return seed * 10**9 + index


def eval_seed(family: str, split: str, index: int) -> int:
    return 10**8 + ALL_FAMILIES.index(family) * 10**6 + SPLITS[split] + index


class TrainStream(torch.utils.data.Dataset):
    """Sample i is always the same world; resuming at step k starts at sample k * batch."""

    def __init__(self, family, seed, length, fact_tokens=128, market_tokens=96):
        self.family, self.seed, self.length = family, seed, length
        self.fact_tokens, self.market_tokens = fact_tokens, market_tokens

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        world_seed = train_seed(self.seed, index)
        world = simulate(world_seed, self.family)
        return encode(
            world,
            decision_for(world_seed, self.market_tokens),
            self.fact_tokens,
            self.market_tokens,
        )


def _eval_row(job):
    family, split, index, fact_tokens, market_tokens = job
    world_seed = eval_seed(family, split, index)
    world = simulate(world_seed, family)
    decision = decision_for(world_seed, market_tokens)
    row = encode(world, decision, fact_tokens, market_tokens)
    row.update(references(world, decision))
    return row


def build_eval_sets(
    directory,
    sizes,
    fact_tokens=128,
    market_tokens=96,
    processes=8,
    families=EVAL_FAMILIES,
    heldout=HELDOUT_FAMILIES,
):
    """Write one .npz per (split, family) plus a manifest of SHA-256 hashes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "fact_tokens": fact_tokens,
        "market_tokens": market_tokens,
        "families": list(families),
        "heldout": list(heldout),
        "files": {},
    }
    with Pool(processes) as pool:
        for split, size in sizes.items():
            for family in families:
                jobs = [(family, split, i, fact_tokens, market_tokens) for i in range(size)]
                rows = pool.map(_eval_row, jobs, chunksize=64)
                arrays = {key: np.stack([np.asarray(r[key]) for r in rows]) for key in rows[0]}
                path = directory / f"{split}-{family}.npz"
                np.savez_compressed(path, **arrays)
                manifest["files"][path.name] = {
                    "rows": size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_eval(directory, split, family):
    with np.load(Path(directory) / f"{split}-{family}.npz") as data:
        return {key: data[key] for key in data.files}


def eval_families(directory):
    """(all families, held-out families) of a frozen evaluation set; Stage S predates the keys."""
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    families = manifest.get("families", list(EVAL_FAMILIES))
    return families, manifest.get("heldout", list(HELDOUT_FAMILIES))


def verify_eval_sets(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"evaluation set changed after it was frozen: {name}")
    return manifest


def _feature_row(job):
    kind, family, seed_or_split, index, market_tokens = job
    world_seed = (
        train_seed(seed_or_split, index)
        if kind == "train"
        else eval_seed(family, seed_or_split, index)
    )
    world = simulate(world_seed, family)
    decision = decision_for(world_seed, market_tokens)
    return summary_features(world, decision), encode(world, decision, 1, market_tokens)["target"]


def feature_table(kind, family, seed_or_split, size, market_tokens=96, processes=8):
    jobs = [(kind, family, seed_or_split, i, market_tokens) for i in range(size)]
    with Pool(processes) as pool:
        rows = pool.map(_feature_row, jobs, chunksize=256)
    return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])
