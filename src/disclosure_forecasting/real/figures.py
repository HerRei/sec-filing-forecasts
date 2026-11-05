"""Figures for the reports and the project page, drawn only from result files.

python -m disclosure_forecasting.real.figures
Writes SVG + PNG to docs/results/gate1/figures and docs/results/gate4/figures. Colours are the
validated dark-mode categorical slots (blue #3987e5, orange #d95926, aqua #199e70) on the site's
paper (#080b12), ink #eceef2, muted #99a0ad, rules #262c38.
"""

import json

import matplotlib

from .common import ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (the backend has to be set first)

PAPER, INK, MUTED, RULE = "#080b12", "#eceef2", "#99a0ad", "#262c38"
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
G1 = ROOT / "docs/results/gate1"
G4 = ROOT / "docs/results/gate4"
S2 = ROOT / "docs/results/stage-s2/result.json"

plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 13,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": RULE,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "axes.facecolor": PAPER,
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
    }
)


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(folder / f"{name}.png", bbox_inches="tight", pad_inches=0.15, dpi=200)
    plt.close(fig)


def clean(ax, grid_axis="x"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def effects_chart(result):
    """Final period: every model against the zero-mean/volatility forecast."""
    final = result["folds"]["final"]
    sec = final["secondary"]
    rows = [
        ("Pretrained, broad prior", final["primary"]["P1_broad_ft_vs_ref_volcal"], BLUE),
        ("Pretrained, narrower prior", sec["varied_ft_vs_ref_volcal"], BLUE),
        ("Real data only", sec["scratch_ft_vs_ref_volcal"], ORANGE),
        ("Boosted trees", sec["gbdt_vs_ref_volcal"], AQUA),
        ("Linear model", sec["linear_vs_ref_volcal"], AQUA),
    ]
    rows = [r for r in rows if r[1]]
    fig, ax = plt.subplots(figsize=(5.0, 0.62 * len(rows) + 1.6))
    for i, (label, c, colour) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot(c["ci"], [y, y], color=colour, linewidth=2, solid_capstyle="round")
        ax.plot(
            c["effect"],
            y,
            "o",
            color=colour,
            markersize=8,
            markeredgecolor=PAPER,
            markeredgewidth=2,
        )
        ax.text(
            c["ci"][0] - 0.12,
            y,
            f"{c['effect']:+.1f} %",
            va="center",
            ha="right",
            color=INK,
            fontsize=12.5,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.axvspan(-0.5, 0, color=RULE, alpha=0.6, linewidth=0)
    lo = min(min(r[1]["ci"][0] for r in rows), -1.0)
    hi = max(max(r[1]["ci"][1] for r in rows), 0.5)
    ax.set_xlim(lo - 1.3, hi + 0.4)
    ax.set_xlabel("Loss vs. volatility-only forecast, % (left = better)")
    clean(ax)
    ax.set_title("Real filings, 2023 to 2026", color=INK, loc="left", fontsize=14, pad=10)
    save(fig, G4 / "figures", "effects-final")


def folds_chart(result):
    folds = [
        f for f in ("dev2019", "dev2020", "dev2021", "dev2022", "final") if f in result["folds"]
    ]
    names = {
        "dev2019": "2019",
        "dev2020": "2020",
        "dev2021": "2021",
        "dev2022": "2022",
        "final": "2023-26\n(final)",
    }
    series = [
        (
            "Pretrained, broad prior",
            lambda f: result["folds"][f]["primary"]["P1_broad_ft_vs_ref_volcal"],
            BLUE,
            -0.12,
        ),
        (
            "Boosted trees",
            lambda f: result["folds"][f].get("secondary", {}).get("gbdt_vs_ref_volcal"),
            AQUA,
            0.0,
        ),
        (
            "Real data only",
            lambda f: result["folds"][f].get("secondary", {}).get("scratch_ft_vs_ref_volcal"),
            ORANGE,
            0.12,
        ),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for label, get, colour, dx in series:
        xs, ys = [], []
        for i, f in enumerate(folds):
            c = get(f)
            if not c:
                continue
            ax.plot([i + dx, i + dx], c["ci"], color=colour, linewidth=2, solid_capstyle="round")
            ax.plot(
                i + dx,
                c["effect"],
                "o",
                color=colour,
                markersize=7,
                markeredgecolor=PAPER,
                markeredgewidth=2,
            )
            xs.append(i + dx)
            ys.append(c["effect"])
        ax.plot([], [], "o-", color=colour, label=label, markersize=7, linewidth=2)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([names[f] for f in folds])
    ax.tick_params(axis="x", colors=INK)
    ax.set_xlim(-0.5, len(folds) - 0.5)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=1,
        frameon=False,
        fontsize=11.5,
        labelcolor=INK,
        handlelength=1.6,
    )
    ax.set_ylabel("vs. volatility only, % (below 0 = better)")
    clean(ax, "y")
    ax.set_title("Each test period separately", color=INK, loc="left", fontsize=14, pad=74)
    save(fig, G4 / "figures", "effects-by-year")


def sim_vs_real(result):
    s2 = json.loads(S2.read_text())
    sim = s2["contrasts"]["Q2_primary_E_minus_B_new_heldout"] if "contrasts" in s2 else None
    real = result["folds"]["final"]["secondary"]["broad_ft_vs_varied_ft"]
    if not sim or not real:
        return
    labels = ["Simulation: held-out\ndisclosure families", "Real filings:\n2023 to 2026"]
    values = [sim["rel_pct"], real["effect"]]
    cis = [sim["ci95"], real["ci"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, (v, ci) in enumerate(zip(values, cis)):
        ax.barh(1 - i, v, height=0.5, color=BLUE if i == 0 else ORANGE)
        ax.plot(ci, [1 - i, 1 - i], color=INK, linewidth=1.4)
        ax.text(
            min(v, ci[0]) - 0.3,
            1 - i,
            f"{v:+.1f} %",
            va="center",
            ha="right",
            color=INK,
            fontsize=12.5,
        )
    ax.set_yticks([1, 0])
    ax.set_yticklabels(labels)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlim(min(min(c[0] for c in cis), -1) - 2.2, max(max(c[1] for c in cis), 1) + 0.8)
    ax.set_xlabel("Change in loss, % (left = broader prior better)")
    clean(ax)
    ax.set_title("Broader vs. narrower simulator prior", color=INK, loc="left", fontsize=14, pad=10)
    save(fig, G4 / "figures", "sim-vs-real")


def timestamp_chart(gate1):
    pattern = gate1["acceptance_times"]["json_pattern"]
    order = [
        ("true_utc", "Correct UTC"),
        ("eastern_labelled_utc", "Eastern time labelled UTC"),
        ("eastern_shifted_twice", "Eastern time shifted twice"),
    ]
    total = sum(pattern.get(k, 0) for k, _ in order) + pattern.get("other", 0)
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    for i, (key, label) in enumerate(order):
        share = 100 * pattern.get(key, 0) / total
        ax.barh(len(order) - 1 - i, share, height=0.55, color=AQUA if key == "true_utc" else ORANGE)
        ax.text(
            share + 1, len(order) - 1 - i, f"{share:.0f} %", va="center", color=INK, fontsize=12.5
        )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label for _, label in order][::-1])
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"Share of {total:,} filings")
    clean(ax)
    ax.set_title(
        "SEC bulk file vs. official acceptance time", color=INK, loc="left", fontsize=14, pad=10
    )
    save(fig, G1 / "figures", "acceptance-times")


def audit_chart(gate1):
    by = gate1["audit"]["value_verdict_by_stratum"]
    names = {
        "revenue_v1": "Revenue",
        "inventory_v1": "Inventory",
        "backlog_v1": "Backlog (RPO)",
        "revenue_q4_derived_v1": "Q4 revenue (derived)",
        "revision_amendment": "Revisions: amendments",
        "revision_later_filing": "Revisions: later filings",
    }
    keys = [
        ("match", "matches the filing", AQUA),
        ("not_total", "not the total", ORANGE),
        ("mismatch", "wrong", ORANGE),
        ("unverified", "no second source", MUTED),
    ]
    strata = [s for s in names if s in by]
    fig, ax = plt.subplots(figsize=(6.0, 0.6 * len(strata) + 1.8))
    for i, s in enumerate(strata):
        left, y = 0, len(strata) - 1 - i
        for key, label, colour in keys:
            n = by[s].get(key, 0)
            if n:
                ax.barh(
                    y,
                    n,
                    left=left,
                    height=0.55,
                    color=colour,
                    edgecolor=PAPER,
                    linewidth=2,
                    hatch="///" if key == "mismatch" else None,
                )
                left += n
        parts = [f"{by[s].get('match', 0)} of {sum(by[s].values())} match"]
        parts += [f"{by[s][k]} {lab}" for k, lab, _ in keys[1:] if by[s].get(k)]
        ax.text(left + 0.4, y, ", ".join(parts), va="center", color=INK, fontsize=11.5)
    ax.set_yticks(range(len(strata)))
    ax.set_yticklabels([names[s] for s in strata][::-1])
    ax.set_xlim(0, 52)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("Audited rows")
    clean(ax)
    ax.grid(False)
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=c, edgecolor=PAPER, hatch="///" if k == "mismatch" else None, label=lab)
        for k, lab, c in keys
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=4,
        frameon=False,
        fontsize=11.5,
        labelcolor=INK,
        handlelength=1.2,
        columnspacing=1.2,
    )
    ax.set_title(
        "120 random records vs. the original filings", color=INK, loc="left", fontsize=14, pad=30
    )
    save(fig, G1 / "figures", "audit")


def main():
    timeline_chart()
    gate1 = json.loads((G1 / "result.json").read_text()) if (G1 / "result.json").exists() else None
    if gate1:
        timestamp_chart(gate1)
        audit_chart(gate1)
    if (G4 / "result.json").exists():
        result = json.loads((G4 / "result.json").read_text())
        if "final" in result["folds"]:
            effects_chart(result)
            sim_vs_real(result)
        folds_chart(result)


def timeline_chart(cik=18230, period_end="2024-06-30"):
    """One real event (Caterpillar, Q2 2024) from the data files: what happens when."""
    import pandas as pd

    from .common import PROCESSED
    from .sessions import session_index

    acc = pd.read_parquet(PROCESSED / "sec/acceptance.parquet")
    rec = pd.read_parquet(PROCESSED / "sec/records.parquet")
    q = rec[
        (rec["cik"] == cik)
        & (rec["period_end"] == period_end)
        & (rec["version"] == 1)
        & (rec["metric"] == "inventory")
    ]
    if q.empty:
        return
    filing = q.iloc[0]

    def eastern(ts):
        ts = pd.Timestamp(ts)
        return (ts if ts.tzinfo else ts.tz_localize("UTC")).tz_convert("America/New_York")

    filed = eastern(filing["accepted_utc"])
    mine = acc[(acc["cik"] == cik) & (acc["form"] == "8-K")].copy()
    mine["t"] = pd.to_datetime(mine["accepted_utc_true"], utc=True)
    before = mine[
        (mine["t"] < pd.Timestamp(filing["accepted_utc"]))
        & (mine["t"] > pd.Timestamp(filing["accepted_utc"]) - pd.Timedelta(days=20))
    ]
    release = eastern(before.sort_values("t").iloc[-1]["t"]) if len(before) else None
    days = session_index()
    s = int(days.get_indexer([pd.Timestamp(filing["exec_session"])])[0])
    decision = pd.Timestamp(filing["executable_at"], tz="UTC").tz_convert("America/New_York")
    end5, end20 = days[s + 5], days[s + 20]
    points = [
        (pd.Timestamp(period_end), "Quarter ends", MUTED, "Jun 30"),
        (
            release.tz_localize(None) if release is not None else None,
            "Earnings press release (8-K)",
            ORANGE,
            release.strftime("%b %-d, %-I:%M %p") if release is not None else "",
        ),
        (
            filed.tz_localize(None),
            "10-Q accepted by the SEC",
            AQUA,
            filed.strftime("%b %-d, %-I:%M %p"),
        ),
        (
            decision.tz_localize(None),
            "Model decides at the next open",
            BLUE,
            decision.strftime("%b %-d, %-I:%M %p"),
        ),
        (end5 + pd.Timedelta(hours=9.5), "5 trading days later", BLUE, end5.strftime("%b %-d")),
        (end20 + pd.Timedelta(hours=9.5), "20 trading days later", BLUE, end20.strftime("%b %-d")),
    ]
    points = [p for p in points if p[0] is not None]
    fig, ax = plt.subplots(figsize=(5.0, 0.62 * len(points) + 1.0))
    n = len(points)
    ys = list(range(n - 1, -1, -1))
    ax.plot([0, 0], [ys[0], ys[-1]], color=RULE, linewidth=2, zorder=1)
    ax.plot([0, 0], [ys[3], ys[-1]], color=BLUE, linewidth=4, solid_capstyle="round", zorder=2)
    for (t, label, colour, when), y in zip(points, ys):
        ax.plot(
            0,
            y,
            "o",
            color=colour,
            markersize=10,
            markeredgecolor=PAPER,
            markeredgewidth=2,
            zorder=3,
        )
        ax.text(-0.35, y, when, ha="right", va="center", color=MUTED, fontsize=12.5)
        ax.text(0.35, y, label, ha="left", va="center", color=INK, fontsize=13)
    ax.text(0.35, (ys[3] + ys[-1]) / 2, "", fontsize=11.5)
    ax.annotate(
        "",
        xy=(5.7, ys[-1] - 0.1),
        xytext=(5.7, ys[3] - 0.25),
        arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=1.2),
    )
    ax.text(
        5.85,
        (ys[3] + ys[-1]) / 2,
        "what the model\nforecasts: the return\nagainst XLI in\nthis window",
        va="center",
        ha="left",
        color=MUTED,
        fontsize=11.5,
        linespacing=1.3,
    )
    ax.set_xlim(-2.6, 8.2)
    ax.set_ylim(-0.6, n - 0.4)
    ax.axis("off")
    ax.set_title("Caterpillar, Q2 2024 (New York time)", color=INK, loc="left", fontsize=14, pad=6)
    save(fig, G4 / "figures", "timeline-cat-q2-2024")


if __name__ == "__main__":
    main()
