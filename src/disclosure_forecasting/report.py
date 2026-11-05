"""Deterministic Stage S analysis: the pre-registered contrasts, nothing chosen afterwards.

Loss = mean pinball over QUANTILES and HORIZONS (each horizon divided by sqrt(h)), lower is
better. Contrasts are paired on identical test worlds; a difference X - Y < 0 means X is
better. Seeds are averaged per world before a 2,000-resample bootstrap over worlds; the
seed sign count shows training-seed consistency, which the bootstrap does not cover.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .data import load_eval
from .simulator import EVAL_FAMILIES, HELDOUT_FAMILIES, HORIZONS, QUANTILES

VARIANTS = {
    "A": "regular prior, sequence memory",
    "B": "varied prior, sequence memory",
    "C": "regular prior, revision memory",
    "D": "varied prior, revision memory",
}
REFERENCES = ("gbdt-regular", "gbdt-varied", "ref_zero", "ref_state")
POOLED = "heldout_pooled"


def per_world(predictions, target):
    levels = np.asarray(QUANTILES)
    residual = target[..., None] - predictions
    loss = np.maximum(levels * residual, (levels - 1) * residual).mean(-1)
    return loss / np.sqrt(HORIZONS)  # (N, H)


def bootstrap(diff, rng, resamples=2000):
    means = np.empty(resamples)
    for start in range(0, resamples, 100):
        index = rng.integers(0, len(diff), size=(min(100, resamples - start), len(diff)))
        means[start : start + len(index)] = diff[index].mean(1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


class Results:
    def __init__(self, plan_path, root):
        self.root = Path(root)
        self.plan = json.loads((self.root / plan_path).read_text())
        self.eval_dir = self.root / self.plan["eval_dir"]
        self.output = self.root / self.plan["output"]
        self.data = {f: load_eval(self.eval_dir, "test", f) for f in EVAL_FAMILIES}
        self.losses = {}  # (model, seed, step) -> {family: (N, H)}
        self.missing = []
        self.milestones = None
        for path in self.plan["runs"]:
            config = json.loads((self.root / path).read_text())
            variant, seed = config["name"].rsplit("-s", 1)
            self.milestones = config["milestones"]
            self.tokens_per_step = config["batch_size"] * (
                config["fact_tokens"] + config["market_tokens"] + len(HORIZONS)
            )
            for step in config["milestones"]:
                file = self.output / config["name"] / f"predictions-step{step}.npz"
                if file.exists():
                    self._add(variant, int(seed), step, np.load(file))
                elif step == config["milestones"][-1]:
                    self.missing.append(config["name"])
        self.final = self.milestones[-1]
        for name in ("gbdt-regular", "gbdt-varied"):
            file = self.output / "baselines" / name / "predictions-final.npz"
            if file.exists():
                self._add(name, 0, self.final, np.load(file))
        for name in ("ref_zero", "ref_state"):
            self._add(name, 0, self.final, {f: self.data[f][name] for f in EVAL_FAMILIES})

    def _add(self, model, seed, step, arrays):
        self.losses[model, seed, step] = {
            f: per_world(arrays[f], self.data[f]["target"]) for f in EVAL_FAMILIES
        }

    def seeds(self, model, step):
        return sorted(s for m, s, t in self.losses if m == model and t == step)

    def world_loss(self, model, seed, step, family, subset=None):
        if family == POOLED:
            parts = [self.world_loss(model, seed, step, f, subset) for f in HELDOUT_FAMILIES]
            return np.concatenate(parts)
        loss = self.losses[model, seed, step][family].mean(-1)
        return loss if subset is None else loss[self.data[family][subset].astype(bool)]

    def mean_loss(self, model, family, step=None, subset=None):
        step = self.final if step is None else step
        seeds = self.seeds(model, step)
        if not seeds:
            return None
        return float(
            np.mean([self.world_loss(model, s, step, family, subset).mean() for s in seeds])
        )

    def contrast(self, x, y, family, subset=None, step=None, rng=None):
        step = self.final if step is None else step
        sx, sy = self.seeds(x, step), self.seeds(y, step)
        if not sx or not sy:
            return None
        pairs = [(s, s if s in sy else sy[0]) for s in sx] if sy != [0] else [(s, 0) for s in sx]
        diffs = [
            self.world_loss(x, a, step, family, subset)
            - self.world_loss(y, b, step, family, subset)
            for a, b in pairs
        ]
        mean_diff = np.mean(diffs, axis=0)
        base = self.mean_loss(y, family, step, subset)
        return {
            "x": x,
            "y": y,
            "family": family,
            "subset": subset,
            "worlds": int(len(mean_diff)),
            "diff": float(mean_diff.mean()),
            "relative_pct": float(100 * mean_diff.mean() / base),
            "ci95": bootstrap(mean_diff, rng),
            "seeds_better": int(sum(d.mean() < 0 for d in diffs)),
            "seeds": len(diffs),
        }


def supported(c):
    if c is None:
        return False
    return c["ci95"][1] < 0 and c["seeds_better"] >= max(1, math.ceil(2 * c["seeds"] / 3))


def analyse(plan_path, root):
    rng = np.random.default_rng(20261020)
    r = Results(plan_path, root)
    variants = [v for v in VARIANTS if r.seeds(v, r.final)]
    models = variants + [m for m in REFERENCES if r.seeds(m, r.final)]
    families = list(EVAL_FAMILIES) + [POOLED]
    table = {m: {f: r.mean_loss(m, f) for f in families} for m in models}
    coverage, null = {}, {}
    for m in models:
        inside, bias = [], []
        for s in r.seeds(m, r.final):
            for f in HELDOUT_FAMILIES:
                p = _predictions(r, m, s, f)
                t = r.data[f]["target"]
                inside.append(((t >= p[..., 1]) & (t <= p[..., 5])).mean())
            bias.append(float(_predictions(r, m, s, "null")[:, 1, 3].mean()))
        coverage[m] = float(np.mean(inside))
        null[m] = {
            "median_h5_1pct": float(np.mean(bias)),
            "pinball_vs_ref_zero_pct": float(
                100 * (table[m]["null"] / table["ref_zero"]["null"] - 1)
            ),
        }
    c = {}

    def add(key, x, y, family, subset=None):
        c[key] = r.contrast(x, y, family, subset, rng=rng)

    add("prior_B_minus_A_heldout", "B", "A", POOLED)
    add("prior_D_minus_C_heldout", "D", "C", POOLED)
    add("memory_D_minus_B_varied", "D", "B", "varied")
    add("memory_D_minus_B_varied_recent_revision", "D", "B", "varied", "recent_revision")
    add("memory_D_minus_B_varied_stale_metric", "D", "B", "varied", "stale_metric")
    add("memory_D_minus_B_heldout", "D", "B", POOLED)
    add("memory_C_minus_A_heldout", "C", "A", POOLED)
    add("D_minus_A_heldout", "D", "A", POOLED)
    add("D_minus_gbdt_varied_heldout", "D", "gbdt-varied", POOLED)
    for v in variants:
        add(f"{v}_minus_ref_zero_heldout", v, "ref_zero", POOLED)
    per_family = {
        f: {
            "D_minus_B": r.contrast("D", "B", f, rng=rng),
            "B_minus_A": r.contrast("B", "A", f, rng=rng),
        }
        for f in HELDOUT_FAMILIES
    }
    curves = {
        v: [
            {
                "step": step,
                "tokens": step * r.tokens_per_step,
                POOLED: r.mean_loss(v, POOLED, step),
                "varied": r.mean_loss(v, "varied", step),
            }
            for step in r.milestones
            if r.seeds(v, step)
        ]
        for v in variants
    }
    learnable = any(supported(c.get(f"{v}_minus_ref_zero_heldout")) for v in variants)
    prior = [supported(c["prior_B_minus_A_heldout"]), supported(c["prior_D_minus_C_heldout"])]
    memory = supported(c["memory_D_minus_B_varied"])
    verdict = {
        "learnable_beyond_zero_mean": learnable,
        "prior_effect": "supported" if all(prior) else "mixed" if any(prior) else "not supported",
        "memory_effect_fixed_simulator": "supported" if memory else "not supported",
        "memory_effect_on_revisions": "supported"
        if supported(c["memory_D_minus_B_varied_recent_revision"])
        else "not supported",
        "memory_transfers_heldout": "supported"
        if supported(c["memory_D_minus_B_heldout"])
        else "not supported",
        "beats_tree_reference": "yes" if supported(c["D_minus_gbdt_varied_heldout"]) else "no",
        "null_worlds_ok": {
            v: abs(null[v]["median_h5_1pct"]) < 0.1 and null[v]["pinball_vs_ref_zero_pct"] < 1.0
            for v in variants
        },
    }
    if supported(c["D_minus_A_heldout"]) and not memory:
        verdict["dossier_rule"] = (
            "D beats A but not B: the data prior matters; the architectural contribution is unsupported."
        )
    if not learnable:
        nxt = "Inconclusive: no variant beats the zero-mean reference on held-out families. Fix the model or simulator before Gate 1."
    elif all(prior) and memory:
        nxt = "Both synthetic mechanisms supported. Next: Gate 1 real-data audit; these are simulator results, not market evidence."
    elif any(prior) and not memory:
        nxt = "Observation prior helps, revision memory does not. Next: Gate 1 with the generic model and varied prior; drop revision-slot claims unless real data says otherwise."
    elif memory and not any(prior):
        nxt = "Revision memory helps, the varied prior does not transfer. Next: widen the training families before Gate 1."
    else:
        nxt = "Neither mechanism holds in its own simulator: the dossier's stopping condition for Bet 1 as designed. Revise the hypothesis before spending on real data."
    verdict["next_step"] = nxt
    return {
        "generated_at": time.time(),
        "final_step": r.final,
        "final_tokens": r.final * r.tokens_per_step,
        "missing_runs": r.missing,
        "seeds": {m: r.seeds(m, r.final) for m in models},
        "loss": table,
        "coverage_80_heldout": coverage,
        "null": null,
        "contrasts": c,
        "per_family": per_family,
        "learning_curves": curves,
        "verdict": verdict,
    }


def _predictions(r, model, seed, family):
    if model in ("ref_zero", "ref_state"):
        return r.data[family][model]
    if model.startswith("gbdt"):
        return np.load(r.output / "baselines" / model / "predictions-final.npz")[family]
    return np.load(r.output / f"{model}-s{seed}" / f"predictions-step{r.final}.npz")[family]


def fmt(c):
    if c is None:
        return "n/a"
    lo, hi = c["ci95"]
    return (
        f"{c['diff']:+.4f} ({c['relative_pct']:+.2f} %), 95% CI [{lo:+.4f}, {hi:+.4f}], "
        f"better in {c['seeds_better']}/{c['seeds']} seeds, {c['worlds']} worlds"
    )


def markdown(result):
    lines = [
        "# Stage S result: synthetic Gates 2 and 3",
        "",
        "Generated by `disclosure_forecasting.report` from frozen test worlds. Simulator results only;",
        "nothing here is evidence about real markets. Pre-registration: `docs/PREREGISTRATION.md`.",
        "",
        f"Final milestone: step {result['final_step']} ({result['final_tokens'] / 1e6:.0f}M padded tokens per run).",
    ]
    if result["missing_runs"]:
        lines.append(
            f"**Partial:** missing final predictions for {', '.join(result['missing_runs'])}."
        )
    v = result["verdict"]
    lines += ["", "## Verdict", ""]
    lines += [f"- {k.replace('_', ' ')}: {val}" for k, val in v.items() if k != "next_step"]
    lines += ["", f"**Next step:** {v['next_step']}", "", "## Pre-registered contrasts", ""]
    lines += [f"- `{k}`: {fmt(val)}" for k, val in result["contrasts"].items()]
    families = list(EVAL_FAMILIES) + [POOLED]
    lines += ["", "## Loss by family (lower is better)", ""]
    lines += ["| model | " + " | ".join(families) + " |", "|---" * (len(families) + 1) + "|"]
    for model, row in result["loss"].items():
        label = f"{model} ({VARIANTS[model]})" if model in VARIANTS else model
        lines.append(f"| {label} | " + " | ".join(f"{row[f]:.4f}" for f in families) + " |")
    lines += ["", "## Held-out families: D - B and B - A", ""]
    for f, pair in result["per_family"].items():
        lines.append(f"- {f}: D-B {fmt(pair['D_minus_B'])}; B-A {fmt(pair['B_minus_A'])}")
    lines += ["", "## Calibration and null worlds", ""]
    for m, cov in result["coverage_80_heldout"].items():
        n = result["null"][m]
        lines.append(
            f"- {m}: 80% interval coverage on held-out {cov:.3f}; null median h5 "
            f"{n['median_h5_1pct']:+.3f} (1% units); null loss vs zero-mean {n['pinball_vs_ref_zero_pct']:+.2f} %"
        )
    lines += ["", "## Learning curves (pooled held-out / varied)", ""]
    for model, points in result["learning_curves"].items():
        cells = ", ".join(
            f"{p['tokens'] / 1e6:.0f}M: {p[POOLED]:.4f} / {p['varied']:.4f}" for p in points
        )
        lines.append(f"- {model}: {cells}")
    lines += [
        "",
        "`ref_state` uses hidden simulator state (pending drift only) and is a scale reference,",
        "not an upper bound or a competitor.",
    ]
    return "\n".join(lines) + "\n"


def compact(result):
    """A < 3 KB summary: verdicts and the headline contrasts only."""
    keep = (
        "prior_B_minus_A_heldout",
        "prior_D_minus_C_heldout",
        "memory_D_minus_B_varied",
        "memory_D_minus_B_varied_recent_revision",
        "memory_D_minus_B_heldout",
        "D_minus_gbdt_varied_heldout",
    )
    return {
        "final_tokens": result["final_tokens"],
        "missing_runs": result["missing_runs"],
        "verdict": result["verdict"],
        "contrasts": {
            k: {
                x: result["contrasts"][k][x]
                for x in ("relative_pct", "ci95", "seeds_better", "seeds")
            }
            for k in keep
            if result["contrasts"].get(k)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="docs/results/stage-s")
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
