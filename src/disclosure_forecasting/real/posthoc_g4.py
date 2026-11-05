"""Post-hoc diagnostic, NOT pre-registered (written after the Gate 4 results were seen).

Question: does a model's gain over the volatility-only forecast come from where it puts the
middle of the range (direction) or from how wide and how skewed the range is (scale and shape)?
Each model is re-scored with its quantiles shifted so that the median is exactly zero. If the
loss barely changes, the median carried no useful directional information.

python -m disclosure_forecasting.real.posthoc_g4 --out docs/results/gate4
"""

import argparse

import numpy as np

from ..simulator import HORIZONS
from .common import ROOT, write_json
from .g4 import load_events, load_labels
from .report_g4 import Bootstrap, load_fold, per_event_loss

import pandas as pd

MODELS = ("broad-ft", "varied-ft", "scratch-ft", "gbdt", "linear", "ref_volcal")


def recentred(q):
    return q - q[..., 3:4]


def run(fold="final"):
    events, _ = load_events()
    y_all = load_labels()
    models, idx = load_fold(fold)
    y = y_all[idx]
    ev = events.iloc[idx].reset_index(drop=True)
    weeks = pd.to_datetime(ev["exec_session"]).dt.to_period("W").astype(str).values
    boot = Bootstrap(weeks, ev["family"].astype(str).values)
    ref = np.mean([per_event_loss(q, y) for q in models["ref_volcal"].values()], axis=0)
    out = {}
    for name in MODELS:
        qs = list(models[name].values())
        full = np.mean([per_event_loss(q, y) for q in qs], axis=0)
        zero = np.mean([per_event_loss(recentred(q), y) for q in qs], axis=0)
        h20 = HORIZONS.index(20)
        out[name] = {
            "vs_ref_volcal_full": boot.effect(full, ref)["effect"],
            "vs_ref_volcal_median_zero": boot.effect(zero, ref)["effect"],
            "median_zero_vs_full": boot.effect(zero, full),
            "mean_abs_median_h20_pct": float(np.mean([np.abs(q[:, h20, 3]).mean() for q in qs])),
        }
    return {"fold": fold, "note": "post-hoc, not pre-registered", "models": out}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/results/gate4")
    a = parser.parse_args()
    result = run()
    write_json(ROOT / a.out / "posthoc.json", result)
    for name, r in result["models"].items():
        m = r["median_zero_vs_full"]
        print(
            f"{name:12s} vs ref_volcal: full {r['vs_ref_volcal_full']:+.2f} %, median set to 0: "
            f"{r['vs_ref_volcal_median_zero']:+.2f} %; zeroing the median changes loss by "
            f"{m['effect']:+.2f} % [{m['ci'][0]:+.2f}, {m['ci'][1]:+.2f}]; mean |median h20| {r['mean_abs_median_h20_pct']:.2f} %"
        )
