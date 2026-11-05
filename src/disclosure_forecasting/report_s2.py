"""Deterministic Stage S2 analysis (pre-registered in docs/PREREGISTRATION_S2.md).

Changes from Stage S, learned from it: effects are family-balanced relative differences (each
family weighs the same, whatever its loss scale), the bootstrap resamples worlds within each
family, "supported" needs at least MIN_EFFECT % improvement, and every contrast lists the
families where X is significantly worse (harm check).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .data import eval_families, load_eval
from .report import per_world  # same loss definition as Stage S
from .simulator import HELDOUT_FAMILIES, S2_NEW_HELDOUT

MIN_EFFECT = 0.5  # percent
RESAMPLES = 2000
NEW = list(S2_NEW_HELDOUT)
OLD = list(HELDOUT_FAMILIES)


class Results:
    def __init__(self, plan_path, root):
        self.root = Path(root)
        self.plan = json.loads((self.root / plan_path).read_text())
        self.eval_dir = self.root / self.plan["eval_dir"]
        self.output = self.root / self.plan["output"]
        self.families, _ = eval_families(self.eval_dir)
        self.data = {f: load_eval(self.eval_dir, "test", f) for f in self.families}
        self.loss = {}  # (model, seed, step) -> {family: (N,)}
        self.median = {}  # (model, seed, step) -> {family: (N,)} median at h=5
        self.milestones = {}
        self.missing = []
        for path in self.plan["runs"]:
            config = json.loads((self.root / path).read_text())
            model, seed = config["name"].rsplit("-s", 1)
            self.milestones[model] = config["milestones"]
            for step in config["milestones"]:
                file = self.output / config["name"] / f"predictions-step{step}.npz"
                if file.exists():
                    self._add(model, int(seed), step, np.load(file))
                elif step == config["milestones"][-1]:
                    self.missing.append(config["name"])
        for name in self.plan.get("references", []):
            file = self.output / "baselines" / name / "predictions-final.npz"
            if file.exists():
                self._add(name, 0, "final", np.load(file))
        self._add("ref_zero", 0, "final", {f: self.data[f]["ref_zero"] for f in self.families})

    def _add(self, model, seed, step, arrays):
        self.loss[model, seed, step] = {
            f: per_world(arrays[f], self.data[f]["target"]).mean(-1) for f in self.families
        }
        self.median[model, seed, step] = {f: np.asarray(arrays[f])[:, 1, 3] for f in self.families}

    def final(self, model):
        return self.milestones[model][-1] if model in self.milestones else "final"

    def seeds(self, model, step=None):
        step = self.final(model) if step is None else step
        return sorted(s for m, s, t in self.loss if m == model and t == step)

    def mean_loss(self, model, family, step=None):
        step = self.final(model) if step is None else step
        seeds = self.seeds(model, step)
        return (
            float(np.mean([self.loss[model, s, step][family].mean() for s in seeds]))
            if seeds
            else None
        )


def contrast(r, x, y, families, subset=None, rng=None):
    """Family-balanced relative difference of X vs Y in percent (negative = X better)."""
    sx, sy = r.seeds(x), r.seeds(y)
    if not sx or not sy:
        return None
    fx, fy = r.final(x), r.final(y)
    pairs = [(s, s if s in sy else sy[0]) for s in sx]
    per_family, diffs, bases = {}, {}, {}
    for f in families:
        mask = None if subset is None else r.data[f][subset].astype(bool)
        pick = (lambda a: a) if mask is None else (lambda a, m=mask: a[m])
        d = np.mean(
            [pick(r.loss[x, a, fx][f]) - pick(r.loss[y, b, fy][f]) for a, b in pairs], axis=0
        )
        base = np.mean([pick(r.loss[y, b, fy][f]).mean() for _, b in pairs])
        diffs[f], bases[f] = d, base
        seed_rel = [
            100 * (pick(r.loss[x, a, fx][f]).mean() - pick(r.loss[y, b, fy][f]).mean()) / base
            for a, b in pairs
        ]
        boot = np.empty(RESAMPLES)
        for i in range(0, RESAMPLES, 100):
            index = rng.integers(0, len(d), size=(min(100, RESAMPLES - i), len(d)))
            boot[i : i + len(index)] = 100 * d[index].mean(1) / base
        per_family[f] = {
            "rel_pct": float(100 * d.mean() / base),
            "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
            "seeds_better": int(sum(v < 0 for v in seed_rel)),
            "worlds": int(len(d)),
        }
    # Balanced effect: mean of per-family relative effects; stratified bootstrap.
    total = np.zeros(RESAMPLES)
    for f in families:
        d, base = diffs[f], bases[f]
        for i in range(0, RESAMPLES, 100):
            index = rng.integers(0, len(d), size=(min(100, RESAMPLES - i), len(d)))
            total[i : i + len(index)] += 100 * d[index].mean(1) / base / len(families)
    seed_balanced = []
    for a, b in pairs:
        values = []
        for f in families:
            mask = None if subset is None else r.data[f][subset].astype(bool)
            lx, ly = r.loss[x, a, fx][f], r.loss[y, b, fy][f]
            if mask is not None:
                lx, ly = lx[mask], ly[mask]
            values.append(100 * (lx.mean() - ly.mean()) / ly.mean())
        seed_balanced.append(float(np.mean(values)))
    effect = float(np.mean([v["rel_pct"] for v in per_family.values()]))
    ci = [float(v) for v in np.percentile(total, [2.5, 97.5])]
    better = int(sum(v < 0 for v in seed_balanced))
    harmed = [f for f, v in per_family.items() if v["rel_pct"] >= MIN_EFFECT and v["ci95"][0] > 0]
    return {
        "x": x,
        "y": y,
        "families": list(families),
        "subset": subset,
        "rel_pct": effect,
        "ci95": ci,
        "seeds_better": better,
        "seeds": len(pairs),
        "class": classify(effect, ci, better, len(pairs)),
        "harmed_families": harmed,
        "per_family": per_family,
    }


def classify(effect, ci, better, seeds):
    if ci[1] < 0 and effect <= -MIN_EFFECT and better >= max(1, math.ceil(2 * seeds / 3)):
        return "supported"
    if ci[0] > 0 and effect >= MIN_EFFECT:
        return "harmful"
    if ci[1] < 0 or ci[0] > 0:
        return "negligible"  # significant but below the minimum effect, or seeds disagree
    return "not supported"


def analyse(plan_path, root):
    rng = np.random.default_rng(20261021)
    r = Results(plan_path, root)
    c = {}

    def add(key, x, y, families, subset=None):
        c[key] = contrast(r, x, y, families, subset, rng)

    # Q1: revision memory under a fact-token budget (varied prior, 100M tokens).
    add("Q1_primary_D16_minus_B16_varied_late", "D-k16", "B-k16", ["varied", "h_late_revisions"])
    add("Q1_D32_minus_B32_varied_late", "D-k32", "B-k32", ["varied", "h_late_revisions"])
    add("Q1_D16_minus_B16_old_heldout", "D-k16", "B-k16", OLD)
    add("Q1_D32_minus_B32_old_heldout", "D-k32", "B-k32", OLD)
    add("Q1_D16_minus_B16_new_heldout", "D-k16", "B-k16", NEW)
    add("Q1_D16_minus_B16_varied_recent_revision", "D-k16", "B-k16", ["varied"], "recent_revision")
    # Q2: broader training prior, judged on families outside it (300M tokens).
    add("Q2_primary_E_minus_B_new_heldout", "E300", "B300", NEW)
    add("Q2_E_minus_B_old_heldout_partly_seen", "E300", "B300", OLD)
    add("Q2_E_minus_B_h_batched_seen", "E300", "B300", ["h_batched"])
    add("Q2_E_minus_B_varied_cost", "E300", "B300", ["varied"])
    add("Q2_E_minus_B_null", "E300", "B300", ["null"])
    # Q3: does 3x the data catch the tree reference with the same training prior?
    add("Q3_primary_E_minus_gbdt_broad_new_heldout", "E300", "gbdt-broad", NEW)
    add("Q3_B_minus_gbdt_varied_old_heldout", "B300", "gbdt-varied", OLD)
    for model in ("B300", "E300", "D-k16", "B-k16"):
        add(f"{model}_minus_ref_zero_new_heldout", model, "ref_zero", NEW)

    models = sorted({m for m, _, _ in r.loss})
    loss_table = {m: {f: r.mean_loss(m, f) for f in r.families} for m in models}
    null = {}
    for m in models:
        seeds = r.seeds(m)
        if not seeds:
            continue
        step = r.final(m)
        median = float(np.mean([r.median[m, s, step]["null"].mean() for s in seeds]))
        null[m] = {
            "median_h5_1pct": median,
            "loss_vs_ref_zero_pct": 100
            * (loss_table[m]["null"] / loss_table["ref_zero"]["null"] - 1),
            "ok": abs(median) < 0.1
            and loss_table[m]["null"] / loss_table["ref_zero"]["null"] < 1.01,
        }
    curves = {}
    for m, steps in r.milestones.items():
        curves[m] = []
        for step in steps:
            if not r.seeds(m, step):
                continue
            curves[m].append(
                {
                    "step": step,
                    "new_heldout": float(np.mean([r.mean_loss(m, f, step) for f in NEW])),
                    "old_heldout": float(np.mean([r.mean_loss(m, f, step) for f in OLD])),
                    "varied": r.mean_loss(m, "varied", step),
                }
            )

    def cls(key):
        return c[key]["class"] if c.get(key) else "missing"

    verdict = {
        "Q1_memory_under_budget": cls("Q1_primary_D16_minus_B16_varied_late"),
        "Q2_broad_prior_transfers": cls("Q2_primary_E_minus_B_new_heldout"),
        "Q2_harmed_new_families": c["Q2_primary_E_minus_B_new_heldout"]["harmed_families"]
        if c.get("Q2_primary_E_minus_B_new_heldout")
        else None,
        "Q3_beats_tree_reference": cls("Q3_primary_E_minus_gbdt_broad_new_heldout"),
        "null_worlds_ok": {m: v["ok"] for m, v in null.items()},
    }
    return {
        "generated_at": time.time(),
        "min_effect_pct": MIN_EFFECT,
        "missing_runs": r.missing,
        "seeds": {m: r.seeds(m) for m in models},
        "verdict": verdict,
        "contrasts": c,
        "loss": loss_table,
        "null": null,
        "learning_curves": curves,
    }


def fmt(c):
    if c is None:
        return "n/a (missing runs)"
    lo, hi = c["ci95"]
    harm = (
        f"; significantly worse in {', '.join(c['harmed_families'])}"
        if c["harmed_families"]
        else ""
    )
    return (
        f"**{c['class']}**: {c['rel_pct']:+.2f} % [{lo:+.2f}, {hi:+.2f}], better in "
        f"{c['seeds_better']}/{c['seeds']} seeds{harm}"
    )


def markdown(result):
    lines = [
        "# Stage S2 result",
        "",
        "Generated by `disclosure_forecasting.report_s2` from frozen test worlds; simulator results",
        "only. Pre-registration: `docs/PREREGISTRATION_S2.md`. Effects are family-balanced relative",
        f"loss differences (negative = first model better); supported needs <= -{MIN_EFFECT} %, a 95 %",
        "interval below 0 and 2/3 seeds.",
    ]
    if result["missing_runs"]:
        lines.append(f"\n**Partial:** missing {', '.join(result['missing_runs'])}.")
    lines += ["", "## Verdict", ""]
    lines += [f"- {k}: {v}" for k, v in result["verdict"].items()]
    lines += ["", "## Contrasts", ""]
    for key, c in result["contrasts"].items():
        lines.append(f"- `{key}`: {fmt(c)}")
        if c and len(c["families"]) > 1:
            cells = ", ".join(
                f"{f} {v['rel_pct']:+.2f} % [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"
                for f, v in c["per_family"].items()
            )
            lines.append(f"  - per family: {cells}")
    families = list(next(iter(result["loss"].values())).keys())
    lines += ["", "## Loss by family (lower is better)", ""]
    lines += ["| model | " + " | ".join(families) + " |", "|---" * (len(families) + 1) + "|"]
    for model, row in result["loss"].items():
        lines.append(
            f"| {model} | "
            + " | ".join("–" if row[f] is None else f"{row[f]:.4f}" for f in families)
            + " |"
        )
    lines += ["", "## Null worlds", ""]
    for m, v in result["null"].items():
        lines.append(
            f"- {m}: median h5 {v['median_h5_1pct']:+.3f} (1 % units), loss vs zero-mean "
            f"{v['loss_vs_ref_zero_pct']:+.2f} %, ok={v['ok']}"
        )
    lines += ["", "## Learning curves (new held-out / old held-out / varied, family means)", ""]
    for m, points in result["learning_curves"].items():
        cells = ", ".join(
            f"step {p['step']}: {p['new_heldout']:.4f} / {p['old_heldout']:.4f} / {p['varied']:.4f}"
            for p in points
        )
        lines.append(f"- {m}: {cells}")
    return "\n".join(lines) + "\n"


def compact(result):
    keep = [k for k in result["contrasts"] if "primary" in k]
    return {
        "missing_runs": result["missing_runs"],
        "verdict": result["verdict"],
        "primary": {
            k: {
                x: result["contrasts"][k][x]
                for x in ("rel_pct", "ci95", "seeds_better", "seeds", "class")
            }
            for k in keep
            if result["contrasts"].get(k)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="docs/results/stage-s2")
    args = parser.parse_args()
    result = analyse(args.plan, args.root)
    out = Path(args.root) / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "summary.json").write_text(json.dumps(compact(result), separators=(",", ":")) + "\n")
    (out / "report.md").write_text(markdown(result))
    print(json.dumps(compact(result)))


if __name__ == "__main__":
    main()
