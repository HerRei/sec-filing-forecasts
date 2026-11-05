"""Gate 4 experiment: splits, labels, references and the neural runs (docs/PREREGISTRATION_G4.md).

  labels             outcome labels; refuses unless the pre-registration hash is recorded
  baselines <fold>   ref_zero, ref_volcal, linear quantile regression, boosted trees (3 seeds)
  neural <fold> <init> <seed> [--threads N] [--device cpu|cuda:0]
                     init = broad (E300), varied (B300), scratch; also writes zero-shot output
  folds              print the fold table

Predictions: runs/gate4/<fold>/<model>.npz with quantiles (N_test, 3, 7) in 1 % units, aligned
to the fold's test index (events.parquet row numbers).
"""

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch

from ..model import ForecastModel, pinball
from ..simulator import HORIZONS, QUANTILES, Z
from .common import PROCESSED, ROOT, sha256_file, write_json
from .sessions import session_index

G4 = PROCESSED / "g4"
RUNS = ROOT / ("runs/gate4-smoke" if os.environ.get("G4_SMOKE") == "1" else "runs/gate4")
PREREG = ROOT / "docs/PREREGISTRATION_G4.md"
PREREG_HASH = G4 / "prereg.sha256"
EMBARGO = 5
LABEL_SESSIONS = max(HORIZONS)
FOLDS = {  # name: (test start, test end exclusive); validation = the 12 months before test start
    "dev2019": ("2019-01-01", "2020-01-01"),
    "dev2020": ("2020-01-01", "2021-01-01"),
    "dev2021": ("2021-01-01", "2022-01-01"),
    "dev2022": ("2022-01-01", "2023-01-01"),
    "final": ("2023-01-01", "2027-01-01"),
}
METRIC_KEYS = (
    "has",
    "growth",
    "accel",
    "age",
    "lag",
    "is_revision",
    "revisions_63",
    "revision_size_63",
)
MARKET_FEATURES = [
    "ret_1",
    "ret_5",
    "ret_20",
    "ret_60",
    "vol_20",
    "vol_60",
    "arrivals_5",
    "has_8k",
    "sessions_since_8k",
    "reaction_8k",
    "drift_since_8k",
]
# Same information as the model tokens: revenue and inventory records + the market window.
PRIMARY_FEATURES = (
    [f"{m}_{k}" for m in ("revenue", "inventory") for k in METRIC_KEYS]
    + ["inventory_minus_revenue"]
    + MARKET_FEATURES
)
# Secondary inputs (trees and linear only): backlog (failed Gate 1 semantics) and ALFRED orders.
BACKLOG_FEATURES = [f"backlog_{k}" for k in METRIC_KEYS] + ["backlog_minus_revenue", "book_to_bill"]
MACRO_FEATURES = ["macro_orders_growth", "macro_orders_accel", "macro_orders_revision"]
CHECKPOINTS = {
    "broad": "runs/stage-s2/E300-s{seed}/checkpoint.pt",
    "varied": "runs/stage-s2/B300-s{seed}/checkpoint.pt",
}
NEURAL = {
    "lr": 2e-4,
    "weight_decay": 0.01,
    "batch_size": 64,
    "warmup_steps": 100,
    "max_epochs": 15,
    "patience": 3,
    "eval_horizons": (5, 20),
}


# Data and splits.


def load_events():
    events = pd.read_parquet(G4 / "events.parquet")
    if (G4 / "macro.parquet").exists():
        events = pd.concat([events, pd.read_parquet(G4 / "macro.parquet")], axis=1)
    with np.load(G4 / "tokens.npz") as data:
        tokens = {k: data[k] for k in data.files}
    return events, tokens


def last_label_session():
    panel_dates = pd.read_parquet(PROCESSED / "prices/panel.parquet", columns=["date", "ticker"])
    last = panel_dates[panel_dates["ticker"] == "XLI"]["date"].max()
    sessions = session_index()
    return int(np.searchsorted(sessions.values, np.datetime64(last), side="right") - 1)


def fold_masks(events, fold):
    """Boolean masks train / val / test with label-horizon purge, embargo and issuer holds."""
    sessions = session_index()
    start, end = (np.datetime64(d) for d in FOLDS[fold])
    test_start = int(np.searchsorted(sessions.values, start))
    val_start = int(np.searchsorted(sessions.values, start - np.timedelta64(365, "D")))
    test_end = int(np.searchsorted(sessions.values, end))
    s = events["s"].values
    mature = s + LABEL_SESSIONS <= last_label_session()
    eligible = ~events["held_out_issuer"].values
    gap = LABEL_SESSIONS + EMBARGO
    train = eligible & (s + gap <= val_start)
    val = eligible & (s >= val_start) & (s + gap <= test_start)
    test = (s >= test_start) & (s < test_end) & mature
    return train & mature, val & mature, test


# Labels (after the freeze).


def labels():
    if not PREREG.exists() or not PREREG_HASH.exists():
        raise SystemExit("pre-registration missing or its hash not recorded: no labels")
    if sha256_file(PREREG) != PREREG_HASH.read_text().split()[0]:
        raise SystemExit("pre-registration changed after its hash was recorded: no labels")
    events, _ = load_events()
    panel = pd.read_parquet(PROCESSED / "prices/panel.parquet")
    sessions = session_index()
    opens = panel.pivot(index="date", columns="ticker", values="tr_open").reindex(sessions)
    bench = opens["XLI"].values
    out = np.full((len(events), len(HORIZONS)), np.nan, dtype=np.float32)
    for i, (s, ticker) in enumerate(zip(events["s"].values, events["ticker"].values)):
        px = opens[ticker].values
        for j, h in enumerate(HORIZONS):
            if s + h < len(px) and np.isfinite(px[s]) and np.isfinite(px[s + h]):
                out[i, j] = 100 * ((px[s + h] / px[s] - 1) - (bench[s + h] / bench[s] - 1))
    np.save(G4 / "labels.npy", out)
    info = {
        "events": len(events),
        "missing_by_horizon": {
            str(h): int(np.isnan(out[:, j]).sum()) for j, h in enumerate(HORIZONS)
        },
        "prereg_sha256": PREREG_HASH.read_text().split()[0],
        "sha256": sha256_file(G4 / "labels.npy"),
    }
    write_json(G4 / "labels.json", info)
    return info


def load_labels():
    if os.environ.get("G4_SMOKE") == "1":  # software test only: random labels, never real outcomes
        n = len(pd.read_parquet(G4 / "events.parquet", columns=["s"]))
        return np.random.default_rng(0).normal(0, 5, (n, len(HORIZONS))).astype(np.float32)
    return np.load(G4 / "labels.npy")


def usable(mask, y):
    return mask & np.isfinite(y).all(axis=1)


def scale(events):
    """Per-event volatility scale (1 % units per sqrt(session)), floored."""
    return np.maximum(events["vol_20"].values.astype(np.float64), 0.1)


# References.


def ref_zero(events, idx):
    sd = scale(events)[idx][:, None, None] * np.sqrt(np.array(HORIZONS))[None, :, None]
    return (sd * Z[None, None, :]).astype(np.float32)


def ref_volcal(events, y, train, idx):
    sd = scale(events)[:, None] * np.sqrt(np.array(HORIZONS))[None, :]
    z = y[train] / sd[train]
    q = np.quantile(z, QUANTILES, axis=0).T  # (3, 7)
    return (sd[idx][:, :, None] * q[None]).astype(np.float32)


def _matrix(events, columns, train):
    x = events[columns].astype(np.float64).values
    return x


def linear_qr(events, y, train, idx, columns):
    from sklearn.linear_model import QuantileRegressor

    x = _matrix(events, columns, train)
    filled = np.where(np.isnan(x), 0.0, x)
    mean, std = filled[train].mean(0), filled[train].std(0) + 1e-9
    lo, hi = np.percentile(filled[train], 1, axis=0), np.percentile(filled[train], 99, axis=0)
    xs = (np.clip(filled, lo, hi) - mean) / std
    sd = scale(events)[:, None] * np.sqrt(np.array(HORIZONS))[None, :]
    z = y / sd
    out = np.zeros((len(idx), len(HORIZONS), len(QUANTILES)), dtype=np.float32)
    for h in range(len(HORIZONS)):
        for k, level in enumerate(QUANTILES):
            model = QuantileRegressor(quantile=level, alpha=1e-3, solver="highs")
            model.fit(xs[train], z[train, h])
            out[:, h, k] = model.predict(xs[idx]) * sd[idx, h]
    return np.sort(out, axis=-1)


def gbdt(events, y, train, idx, columns, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor

    x = _matrix(events, columns, train)
    sd = scale(events)[:, None] * np.sqrt(np.array(HORIZONS))[None, :]
    z = y / sd
    out = np.zeros((len(idx), len(HORIZONS), len(QUANTILES)), dtype=np.float32)
    iterations = {}
    for h in range(len(HORIZONS)):
        for k, level in enumerate(QUANTILES):
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=level,
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=100,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=seed,
            )
            model.fit(x[train], z[train, h])
            out[:, h, k] = model.predict(x[idx]) * sd[idx, h]
            iterations[f"{h},{k}"] = int(model.n_iter_)
    return np.sort(out, axis=-1), iterations


def save(fold, name, predictions, test_idx, info=None):
    path = RUNS / fold / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, q=predictions.astype(np.float32), idx=test_idx)
    if info is not None:
        write_json(RUNS / fold / f"{name}.json", info)


def baselines(fold):
    events, _ = load_events()
    y = load_labels()
    train, val, test = fold_masks(events, fold)
    fit = usable(train | val, y)  # references use train + validation (no early stopping needed)
    test_idx = np.where(usable(test, y))[0]
    started = time.time()
    save(fold, "ref_zero", ref_zero(events, test_idx), test_idx)
    save(fold, "ref_volcal", ref_volcal(events, y, fit, test_idx), test_idx)
    save(fold, "linear", linear_qr(events, y, fit, test_idx, PRIMARY_FEATURES), test_idx)
    save(fold, "linear_market_only", linear_qr(events, y, fit, test_idx, MARKET_FEATURES), test_idx)
    skipped = []
    have_macro = all(c in events.columns for c in MACRO_FEATURES)
    save(
        fold,
        "linear_backlog",
        linear_qr(events, y, fit, test_idx, PRIMARY_FEATURES + BACKLOG_FEATURES),
        test_idx,
    )
    if have_macro:
        save(
            fold,
            "linear_macro",
            linear_qr(events, y, fit, test_idx, PRIMARY_FEATURES + MACRO_FEATURES),
            test_idx,
        )
    else:
        skipped.append("linear_macro")
    for seed in (1, 2, 3):
        for name, columns in (
            ("gbdt", PRIMARY_FEATURES),
            ("gbdt_market_only", MARKET_FEATURES),
            ("gbdt_backlog", PRIMARY_FEATURES + BACKLOG_FEATURES),
            ("gbdt_macro", PRIMARY_FEATURES + MACRO_FEATURES),
        ):
            if name == "gbdt_macro" and not have_macro:
                skipped.append(f"{name}-s{seed}")
                continue
            pred, iters = gbdt(events, y, fit, test_idx, columns, seed)
            save(fold, f"{name}-s{seed}", pred, test_idx, {"iterations": iters})
    write_json(
        RUNS / fold / "baselines.json",
        {
            "train_events": int(fit.sum()),
            "test_events": len(test_idx),
            "skipped": skipped,
            "seconds": round(time.time() - started),
        },
    )


def macro_baselines(fold):
    """The ALFRED-orders secondary models alone (when the vintages arrived after the baselines)."""
    events, _ = load_events()
    if not all(c in events.columns for c in MACRO_FEATURES):
        raise SystemExit("macro.parquet missing: run alfred build + features first")
    y = load_labels()
    train, val, test = fold_masks(events, fold)
    fit = usable(train | val, y)
    test_idx = np.where(usable(test, y))[0]
    save(
        fold,
        "linear_macro",
        linear_qr(events, y, fit, test_idx, PRIMARY_FEATURES + MACRO_FEATURES),
        test_idx,
    )
    for seed in (1, 2, 3):
        pred, iters = gbdt(events, y, fit, test_idx, PRIMARY_FEATURES + MACRO_FEATURES, seed)
        save(fold, f"gbdt_macro-s{seed}", pred, test_idx, {"iterations": iters})


# Neural.


def batches(tokens, y, idx, batch_size, rng=None):
    order = rng.permutation(idx) if rng is not None else idx
    for start in range(0, len(order), batch_size):
        part = order[start : start + batch_size]
        batch = {
            k: torch.from_numpy(tokens[k][part]) for k in ("facts", "slots", "ranks", "market")
        }
        yield part, batch, (torch.from_numpy(y[part]) if y is not None else None)


@torch.no_grad()
def predict(model, tokens, idx, device, mask_facts=False):
    model.eval()
    out = []
    for _, batch, _ in batches(tokens, None, idx, 512):
        batch = {k: v.to(device) for k, v in batch.items()}
        if mask_facts:
            batch["slots"] = torch.full_like(batch["slots"], -1)  # every fact token padded
            batch["facts"] = torch.zeros_like(batch["facts"])
        out.append(model(**batch).float().cpu())
    model.train()
    return torch.cat(out).numpy()


def eval_loss(prediction, target):
    """Primary loss: pinball over 7 quantiles, horizons 5 and 20, each / sqrt(h); per event."""
    losses = pinball(torch.from_numpy(prediction), torch.from_numpy(target)).numpy()
    cols = [HORIZONS.index(h) for h in NEURAL["eval_horizons"]]
    return losses[:, cols].mean(axis=1)


def neural(fold, init, seed, threads=8, device="cpu"):
    torch.set_num_threads(threads)
    events, tokens = load_events()
    y = load_labels()
    train, val, test = fold_masks(events, fold)
    train_idx, val_idx = np.where(usable(train, y))[0], np.where(usable(val, y))[0]
    test_idx = np.where(usable(test, y))[0]
    name = f"{init}-ft-s{seed}"
    out_dir = RUNS / fold
    if (out_dir / f"{name}.npz").exists():
        return {"skipped": name}
    torch.manual_seed(seed)
    model = ForecastModel(width=256, layers=6, heads=8, memory="sequence")
    if init in CHECKPOINTS:
        saved = torch.load(
            ROOT / CHECKPOINTS[init].format(seed=seed), map_location="cpu", weights_only=True
        )
        model.load_state_dict(saved["model"])
        zero_shot = predict(model, tokens, test_idx, device)
        save(fold, f"{init}-zeroshot-s{seed}", zero_shot, test_idx)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=NEURAL["lr"], betas=(0.9, 0.95), weight_decay=NEURAL["weight_decay"]
    )
    steps_per_epoch = math.ceil(len(train_idx) / NEURAL["batch_size"])
    total = steps_per_epoch * NEURAL["max_epochs"]

    def lr_at(step):
        if step <= NEURAL["warmup_steps"]:
            return NEURAL["lr"] * step / NEURAL["warmup_steps"]
        progress = (step - NEURAL["warmup_steps"]) / max(1, total - NEURAL["warmup_steps"])
        return NEURAL["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    rng = np.random.default_rng(seed)
    best, best_state, best_epoch, waited, log = np.inf, None, -1, 0, []
    step, started = 0, time.time()
    for epoch in range(NEURAL["max_epochs"]):
        losses = []
        for _, batch, target in batches(tokens, y, train_idx, NEURAL["batch_size"], rng):
            step += 1
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = pinball(model(**batch).float(), target.to(device)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        val_loss = float(eval_loss(predict(model, tokens, val_idx, device), y[val_idx]).mean())
        log.append(
            {
                "epoch": epoch,
                "train": float(np.mean(losses)),
                "val": val_loss,
                "seconds": round(time.time() - started),
            }
        )
        print(json.dumps({"run": f"{fold}/{name}", **log[-1]}), flush=True)
        if val_loss < best - 1e-5:
            best, best_epoch, waited = val_loss, epoch, 0
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= NEURAL["patience"]:
                break
    model.load_state_dict(best_state)
    predictions = predict(model, tokens, test_idx, device)
    save(
        fold,
        name,
        predictions,
        test_idx,
        {
            "init": init,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val": best,
            "log": log,
            "train_events": len(train_idx),
            "val_events": len(val_idx),
            "test_events": len(test_idx),
            "settings": NEURAL,
            "threads": threads,
            "device": device,
        },
    )
    if init == "broad":
        save(
            fold,
            f"{init}-ft-nofacts-s{seed}",
            predict(model, tokens, test_idx, device, mask_facts=True),
            test_idx,
        )
    return {"done": name, "best_epoch": best_epoch, "seconds": round(time.time() - started)}


def folds_table():
    events, _ = load_events()
    y = load_labels() if (G4 / "labels.npy").exists() else None
    rows = {}
    for fold in FOLDS:
        train, val, test = fold_masks(events, fold)
        if y is not None:
            train, val, test = usable(train, y), usable(val, y), usable(test, y)
        rows[fold] = {
            "train": int(train.sum()),
            "val": int(val.sum()),
            "test": int(test.sum()),
            "test_held_out_issuers": int((test & events["held_out_issuer"].values).sum()),
        }
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("labels", "baselines", "macro", "neural", "folds"))
    parser.add_argument("args", nargs="*")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    a = parser.parse_args()
    if a.command == "labels":
        print(json.dumps(labels(), indent=2))
    elif a.command == "baselines":
        baselines(a.args[0])
    elif a.command == "macro":
        macro_baselines(a.args[0])
    elif a.command == "neural":
        print(json.dumps(neural(a.args[0], a.args[1], int(a.args[2]), a.threads, a.device)))
    else:
        print(json.dumps(folds_table(), indent=2))
