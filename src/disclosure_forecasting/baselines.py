"""CPU references for Stage S: boosted trees on hand-built event-study features.

The zero-mean/volatility and hidden-state references are stored with the evaluation sets.
Trees are trained per (horizon, quantile) on the same training worlds the neural seed-1
runs see first, and evaluated on the same frozen test worlds.
"""

import json
import time
from pathlib import Path

import numpy as np

from .data import eval_families, feature_table
from .simulator import HORIZONS, QUANTILES


def fit_trees(family, worlds, seed=1, processes=8):
    from sklearn.ensemble import HistGradientBoostingRegressor

    features, targets = feature_table("train", family, seed, worlds, processes=processes)
    models = {}
    for h, _ in enumerate(HORIZONS):
        for q, level in enumerate(QUANTILES):
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=level,
                max_iter=400,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=200,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=30,
                random_state=seed,
            )
            models[h, q] = model.fit(features, targets[:, h])
    return models


def predict_trees(models, features):
    out = np.zeros((len(features), len(HORIZONS), len(QUANTILES)), dtype=np.float32)
    for (h, q), model in models.items():
        out[:, h, q] = model.predict(features)
    return np.sort(out, axis=-1)  # repair quantile crossing


def run(output, eval_dir, train_families, worlds=200_000, processes=8):
    output = Path(output)
    families, _ = eval_families(eval_dir)
    test_size = len(np.load(Path(eval_dir) / f"test-{families[0]}.npz")["target"])
    test_features = {
        family: feature_table("eval", family, "test", test_size, processes=processes)[0]
        for family in families
    }
    for train_family in train_families:
        destination = output / f"gbdt-{train_family}"
        if (destination / "predictions-final.npz").exists():
            continue
        started = time.time()
        models = fit_trees(train_family, worlds, processes=processes)
        arrays = {f: predict_trees(models, x) for f, x in test_features.items()}
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / "predictions-final.npz", **arrays)
        (destination / "info.json").write_text(
            json.dumps(
                {
                    "train_family": train_family,
                    "train_worlds": worlds,
                    "iterations": {f"{h},{q}": int(m.n_iter_) for (h, q), m in models.items()},
                    "seconds": round(time.time() - started),
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps({"event": "baseline_done", "model": f"gbdt-{train_family}"}), flush=True)
