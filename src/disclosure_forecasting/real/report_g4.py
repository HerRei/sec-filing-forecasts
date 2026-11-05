"""Gate 4 analysis, fixed in docs/PREREGISTRATION_G4.md before any label existed.

python -m disclosure_forecasting.real.report_g4 --out docs/results/gate4
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..model import pinball
from ..simulator import HORIZONS
from .common import ROOT, sha256_file, write_json
from .g4 import FOLDS, G4, RUNS, load_events, load_labels

FAMILIES = ("34", "35", "36", "37", "38")
EVAL_H = (5, 20)
SEEDS = (1, 2, 3)
RESAMPLES = 2000
RNG_SEED = 20261026
MIN_EFFECT = 0.5
PRIMARY = {
    "P1_broad_ft_vs_ref_volcal": ("broad-ft", "ref_volcal"),
    "P2_broad_ft_vs_linear": ("broad-ft", "linear"),
    "P3_broad_ft_vs_gbdt": ("broad-ft", "gbdt"),
    "P4_broad_ft_vs_scratch_ft": ("broad-ft", "scratch-ft"),
}
SECONDARY = {
    "broad_ft_vs_varied_ft": ("broad-ft", "varied-ft"),
    "broad_zeroshot_vs_ref_volcal": ("broad-zeroshot", "ref_volcal"),
    "varied_zeroshot_vs_ref_volcal": ("varied-zeroshot", "ref_volcal"),
    "scratch_ft_vs_ref_volcal": ("scratch-ft", "ref_volcal"),
    "varied_ft_vs_ref_volcal": ("varied-ft", "ref_volcal"),
    "linear_vs_ref_volcal": ("linear", "ref_volcal"),
    "gbdt_vs_ref_volcal": ("gbdt", "ref_volcal"),
    "ref_zero_vs_ref_volcal": ("ref_zero", "ref_volcal"),
    "gbdt_vs_gbdt_market_only": ("gbdt", "gbdt_market_only"),
    "linear_vs_linear_market_only": ("linear", "linear_market_only"),
    "broad_ft_vs_broad_ft_nofacts": ("broad-ft", "broad-ft-nofacts"),
    "gbdt_backlog_vs_gbdt": ("gbdt_backlog", "gbdt"),
    "gbdt_macro_vs_gbdt": ("gbdt_macro", "gbdt"),
    "linear_backlog_vs_linear": ("linear_backlog", "linear"),
    "linear_macro_vs_linear": ("linear_macro", "linear"),
}
SEEDED = {
    "broad-ft",
    "varied-ft",
    "scratch-ft",
    "broad-zeroshot",
    "varied-zeroshot",
    "broad-ft-nofacts",
    "gbdt",
    "gbdt_market_only",
    "gbdt_backlog",
    "gbdt_macro",
}


def per_event_loss(q, y, horizons=EVAL_H):
    losses = pinball(
        torch.from_numpy(q.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
    ).numpy()
    return losses[:, [HORIZONS.index(h) for h in horizons]].mean(axis=1)


def load_fold(fold):
    """{model: {"seeds": {k: q}, "q": seed-mean-free single q}} with a common test index."""
    out, idx = {}, None
    for path in sorted((RUNS / fold).glob("*.npz")):
        name = path.stem
        with np.load(path) as data:
            q, i = data["q"], data["idx"]
        if idx is None:
            idx = i
        if not np.array_equal(idx, i):
            raise ValueError(f"{fold}/{name}: test index differs")
        if name[-3:-1] == "-s" and name[-1].isdigit():
            base, seed = name[:-3], int(name[-1])
            out.setdefault(base, {})[seed] = q
        else:
            out.setdefault(name, {})[0] = q
    return out, idx


def losses_for(models, y, horizons=EVAL_H):
    """{model: {"mean": per-event loss averaged over seeds, "seeds": {k: per-event loss}}}"""
    result = {}
    for name, by_seed in models.items():
        seeds = {k: per_event_loss(q, y, horizons) for k, q in by_seed.items()}
        result[name] = {"mean": np.mean(list(seeds.values()), axis=0), "seeds": seeds}
    return result


class Bootstrap:
    """Week-cluster bootstrap of family-balanced relative effects."""

    def __init__(self, weeks, families):
        self.week_ids, self.week_index = np.unique(weeks, return_inverse=True)
        self.families = families
        rng = np.random.default_rng(RNG_SEED)
        n = len(self.week_ids)
        draws = rng.integers(0, n, size=(RESAMPLES, n))
        self.weights = np.stack([np.bincount(d, minlength=n) for d in draws]).astype(np.float64)

    def effect(self, x, y, mask=None):
        mask = np.ones(len(x), bool) if mask is None else mask
        point, per_family, boot_fam = [], {}, []
        fam_idx = []
        for f in FAMILIES:
            m = mask & (self.families == f)
            if m.sum() < 10:
                continue
            fam_idx.append(f)
            sx = np.bincount(self.week_index[m], weights=x[m], minlength=len(self.week_ids))
            sy = np.bincount(self.week_index[m], weights=y[m], minlength=len(self.week_ids))
            bx, by = self.weights @ sx, self.weights @ sy
            rel = 100 * (bx - by) / by
            boot_fam.append(rel)
            p = 100 * (x[m].mean() - y[m].mean()) / y[m].mean()
            point.append(p)
            per_family[f] = {
                "effect": p,
                "ci": [float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))],
                "n": int(m.sum()),
            }
        boot = np.mean(boot_fam, axis=0)
        return {
            "effect": float(np.mean(point)),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "per_family": per_family,
            "n": int(mask.sum()),
        }


def contrast(boot, losses, x, y, mask=None):
    if x not in losses or y not in losses:
        return None
    result = boot.effect(losses[x]["mean"], losses[y]["mean"], mask)
    seeds = []
    for k in SEEDS:
        lx = losses[x]["seeds"].get(k)
        if lx is None:
            continue
        ly = losses[y]["seeds"].get(k, losses[y]["mean"])
        seeds.append(boot.effect(lx, ly, mask)["effect"])
    result["seed_effects"] = seeds
    result["seeds_better"] = int(sum(s < 0 for s in seeds))
    n_seeds = len(seeds)
    lo, hi = result["ci"]
    seeds_ok = (result["seeds_better"] >= 2) if n_seeds >= 3 else True
    if hi < 0 and result["effect"] <= -MIN_EFFECT and seeds_ok:
        verdict = "supported"
    elif hi < 0:
        verdict = "negligible"
    elif lo > 0 and result["effect"] >= MIN_EFFECT:
        verdict = "harmful"
    elif lo > 0:
        verdict = "worse (below the 0.5 % harm threshold)"
    else:
        verdict = "not supported"
    result["verdict"] = verdict
    result["harmed_families"] = [
        f for f, v in result["per_family"].items() if v["ci"][0] > 0 and v["effect"] >= MIN_EFFECT
    ]
    return result


def coverage(q, y, horizons=EVAL_H):
    cols = [HORIZONS.index(h) for h in horizons]
    inside = (y[:, cols] >= q[:, cols, 1]) & (y[:, cols] <= q[:, cols, 5])
    return float(inside.mean())


def quintile_spread(q, y, years):
    """Top-minus-bottom quintile of predicted 20-session median: realised mean excess return."""
    h = HORIZONS.index(20)
    out = {}
    for year in sorted(set(years)):
        m = years == year
        pred, real = q[m, h, 3], y[m, h]
        lo, hi = np.quantile(pred, [0.2, 0.8])
        out[str(year)] = float(real[pred >= hi].mean() - real[pred <= lo].mean())
    return out


def analyse_fold(fold, events, y_all, details=True):
    models, idx = load_fold(fold)
    y = y_all[idx]
    ev = events.iloc[idx].reset_index(drop=True)
    weeks = pd.to_datetime(ev["exec_session"]).dt.to_period("W").astype(str).values
    families = ev["family"].astype(str).values
    boot = Bootstrap(weeks, families)
    losses = losses_for(models, y)
    table = {name: float(loss["mean"].mean()) for name, loss in losses.items()}
    family_table = {
        name: {
            f: float(loss["mean"][families == f].mean()) for f in FAMILIES if (families == f).any()
        }
        for name, loss in losses.items()
    }
    result = {
        "test_events": len(idx),
        "models": sorted(models),
        "loss": table,
        "loss_by_family": family_table,
        "primary": {k: contrast(boot, losses, *v) for k, v in PRIMARY.items()},
    }
    if not details:
        return result
    result["secondary"] = {k: contrast(boot, losses, *v) for k, v in SECONDARY.items()}
    held = ev["held_out_issuer"].values
    recent = ev["recent_revision"].values > 0
    years = pd.to_datetime(ev["exec_session"]).dt.year.values
    result["held_out_issuers"] = {
        k: contrast(boot, losses, *v, mask=held) for k, v in PRIMARY.items()
    }
    result["seen_issuers"] = {k: contrast(boot, losses, *v, mask=~held) for k, v in PRIMARY.items()}
    result["recent_revision"] = {
        k: contrast(boot, losses, *v, mask=recent) for k, v in PRIMARY.items()
    }
    result["recent_revision_events"] = int(recent.sum())
    result["by_year"] = {
        str(year): {k: contrast(boot, losses, *v, mask=years == year) for k, v in PRIMARY.items()}
        for year in sorted(set(years))
    }
    for h in EVAL_H:
        lh = losses_for(models, y, (h,))
        result[f"horizon_{h}"] = {
            "loss": {name: float(loss["mean"].mean()) for name, loss in lh.items()},
            "primary": {k: contrast(boot, lh, *v) for k, v in PRIMARY.items()},
        }
    result["coverage_10_90"] = {
        name: float(np.mean([coverage(q, y) for q in by_seed.values()]))
        for name, by_seed in models.items()
    }
    result["quintile_spread_20d"] = {
        name: quintile_spread(np.mean(list(by_seed.values()), axis=0), y, years)
        for name, by_seed in models.items()
        if name in ("broad-ft", "gbdt", "linear", "scratch-ft", "varied-ft")
    }
    result["realised"] = {
        f"h{h}": {
            "mean": float(y[:, HORIZONS.index(h)].mean()),
            "std": float(y[:, HORIZONS.index(h)].std()),
            "median": float(np.median(y[:, HORIZONS.index(h)])),
        }
        for h in EVAL_H
    }
    return result


def neural_runs(fold):
    rows = {}
    for path in sorted((RUNS / fold).glob("*-ft-s*.json")):
        info = json.loads(path.read_text())
        rows[path.stem] = {
            "best_epoch": info["best_epoch"],
            "best_val": info["best_val"],
            "epochs_run": len(info["log"]),
            "train_events": info["train_events"],
        }
    return rows


def build(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    events, _ = load_events()
    y = load_labels()
    if os.environ.get("G4_SMOKE") == "1":  # software test on random labels
        provenance = {"prereg_sha256": "smoke", "prereg_now": "smoke"}
    else:
        provenance = {
            "prereg_sha256": (G4 / "prereg.sha256").read_text().split()[0],
            "prereg_now": sha256_file(ROOT / "docs/PREREGISTRATION_G4.md"),
            "labels": json.loads((G4 / "labels.json").read_text()),
            "dataset": json.loads((G4 / "build.json").read_text())["files"],
        }
    result = {"provenance": provenance, "folds": {}}
    for fold in FOLDS:
        if not (RUNS / fold).exists():
            continue
        result["folds"][fold] = analyse_fold(fold, events, y, details=True)
        result["folds"][fold]["neural_runs"] = neural_runs(fold)
    write_json(out / "result.json", result)
    (out / "report.md").write_text(render(result))
    return result


def fmt(c):
    if c is None:
        return "not run"
    seeds = (
        f", better in {c['seeds_better']}/{len(c['seed_effects'])} seeds"
        if c["seed_effects"]
        else ""
    )
    harm = (
        f"; significantly worse in {', '.join(c['harmed_families'])}"
        if c["harmed_families"]
        else ""
    )
    return f"**{c['verdict']}**: {c['effect']:+.2f} % [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]{seeds}{harm} (n={c['n']})"


def render(r):
    final = r["folds"].get("final")
    lines = [
        "# Gate 4 result: the simulator-trained model on real filings",
        "",
        "Generated by `disclosure_forecasting.real.report_g4` from frozen predictions and labels. Real",
        "data: US industrial 10-Q/10-K filings, Yahoo total-return prices, XLI benchmark. Pre-registration:",
        f"`docs/PREREGISTRATION_G4.md`, sha256 `{r['provenance']['prereg_sha256']}` recorded before the labels"
        + (
            " (unchanged since)."
            if r["provenance"]["prereg_sha256"] == r["provenance"]["prereg_now"]
            else " (CHANGED since!)."
        ),
        "Effects are family-balanced relative pinball-loss differences over SIC groups 34-38 (negative =",
        "first model better), 95 % intervals from 2,000 week-cluster bootstrap resamples.",
        "",
    ]
    if final:
        lines += ["## Primary contrasts (final period 2023-2026)", ""]
        lines += [f"- `{k}`: {fmt(v)}" for k, v in final["primary"].items()]
        lines += [
            "",
            "## Loss by model (final period; 1 % units, mean over h = 5 and 20, each / sqrt(h))",
            "",
            "| model | loss | "
            + " | ".join(f"SIC {f}" for f in FAMILIES)
            + " | 10-90 % coverage |",
            "|---|---:|" + "---:|" * len(FAMILIES) + "---:|",
        ]
        for name, loss in sorted(final["loss"].items(), key=lambda kv: kv[1]):
            fam = final["loss_by_family"][name]
            lines.append(
                f"| {name} | {loss:.4f} | "
                + " | ".join(f"{fam.get(f, float('nan')):.4f}" for f in FAMILIES)
                + f" | {final['coverage_10_90'][name]:.3f} |"
            )
        lines += ["", "## Secondary (final period)", ""]
        lines += [f"- `{k}`: {fmt(v)}" for k, v in final["secondary"].items()]
        for title, key in (
            ("Held-out issuers only", "held_out_issuers"),
            ("Issuers seen in training", "seen_issuers"),
            (
                f"Events after a revenue/inventory revision ({final['recent_revision_events']})",
                "recent_revision",
            ),
        ):
            lines += ["", f"### {title}", ""] + [
                f"- `{k}`: {fmt(v)}" for k, v in final[key].items()
            ]
        for h in EVAL_H:
            lines += ["", f"### Horizon {h} only", ""] + [
                f"- `{k}`: {fmt(v)}" for k, v in final[f"horizon_{h}"]["primary"].items()
            ]
        lines += ["", "### By test year", ""]
        for year, contrasts in final["by_year"].items():
            lines.append(
                f"- {year}: "
                + "; ".join(
                    f"{k.split('_')[0]} {v['effect']:+.2f} % [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]"
                    for k, v in contrasts.items()
                    if v
                )
            )
        lines += ["", "### Realised outcomes and a gross quintile spread (not a strategy)", ""]
        lines += [
            f"- {h}: mean {v['mean']:+.3f} %, median {v['median']:+.3f} %, std {v['std']:.3f} %"
            for h, v in final["realised"].items()
        ]
        lines += [
            f"- top-minus-bottom quintile of predicted 20-session median, `{m}`: "
            + ", ".join(f"{y} {v:+.2f} %" for y, v in s.items())
            for m, s in final["quintile_spread_20d"].items()
        ]
    lines += ["", "## Development folds (secondary; same code, nothing tuned on them)", ""]
    for fold, res in r["folds"].items():
        if fold == "final":
            continue
        lines.append(f"### {fold} ({res['test_events']} test events)")
        lines += [f"- `{k}`: {fmt(v)}" for k, v in res["primary"].items()]
        lines.append("")
    lines += ["## Neural runs (early stopping on the validation year)", ""]
    for fold, res in r["folds"].items():
        runs = res.get("neural_runs", {})
        if runs:
            lines.append(
                f"- {fold}: "
                + ", ".join(
                    f"{k} best epoch {v['best_epoch']} of {v['epochs_run']}"
                    for k, v in runs.items()
                )
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/results/gate4")
    a = parser.parse_args()
    build(ROOT / a.out)
    print("ok")
