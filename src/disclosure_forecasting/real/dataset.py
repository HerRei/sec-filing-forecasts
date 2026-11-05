"""Gate 4 events and point-in-time inputs (no outcome labels: those come after the freeze).

One decision per original 10-Q / 10-K filing with XBRL records, at its executable_at (a session
open). Everything an input uses is known at that open:
- disclosure records whose executable_at <= the decision (the filing's own records included);
- excess open-to-open returns that END at or before the decision open (the last one ends exactly
  at it), stock total return minus XLI total return;
- earnings 8-K (Item 2.02) and filing arrival markers at their executable sessions.

Outputs in data/processed/g4/: events.parquet (meta + tabular event-study features),
tokens.npz (facts/slots/ranks/market in the simulator's layout), build.json (counts, hashes).
"""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from .common import PROCESSED, sha256_file, write_json
from .sessions import executable_from_acceptance, session_index

OUT = PROCESSED / "g4"
METRICS = ("revenue", "inventory", "backlog")  # simulator metric slots 0, 1, 2
# Gate 1: backlog (RPO) failed the semantics audit, so the model tokens carry revenue and
# inventory only; backlog features stay in events.parquet for the secondary analysis.
TOKEN_METRICS = ("revenue", "inventory")
FACT_TOKENS = 48
MARKET_TOKENS = 96
HISTORY_SESSIONS = 120  # price history required before a decision
GROWTH_SCALE = 0.25  # log growth / 0.25: a 25 % YoY change is 1.0 (simulator latent scale)
LOOKBACK_YEARS = 3
FIRST_DECISION = "2010-06-01"
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 1e6
EIGHT_K_WINDOW = 75  # sessions
HELD_OUT_SHARE = 5  # sha256(cik) mod 5 == 0 -> issuer never used for training


def held_out(cik) -> bool:
    return int(hashlib.sha256(str(int(cik)).encode()).hexdigest(), 16) % HELD_OUT_SHARE == 0


def session_ordinals():
    sessions = session_index()
    return sessions, pd.Series(np.arange(len(sessions)), index=sessions)


def date_to_ordinal(dates, sessions):
    """Ordinal of the last session on or before each date (period ends fall on any day)."""
    values = pd.to_datetime(dates).values.astype("datetime64[ns]")
    return np.searchsorted(sessions.values.astype("datetime64[ns]"), values, side="right") - 1


def record_table(records, sessions):
    """Per-record point-in-time transforms, each using only what was known at its own time."""
    records = records[records["metric"].isin(METRICS)].copy()
    records["avail"] = date_to_ordinal(records["exec_session"], sessions)
    records["pe"] = date_to_ordinal(records["period_end"], sessions)
    records["pe_date"] = pd.to_datetime(records["period_end"])
    records = records.sort_values(["cik", "metric", "avail", "version"]).reset_index(drop=True)
    growth = np.full(len(records), np.nan)
    revision_change = np.full(len(records), np.nan)
    for (cik, metric), group in records.groupby(["cik", "metric"], sort=False):
        idx = group.index.values
        pe_date = group["pe_date"].values
        avail = group["avail"].values
        value = group["value"].values
        duration = group["duration"].values
        key = list(zip(group["period_start"], group["period_end"]))
        for j, i in enumerate(idx):
            # year-ago comparator: latest version known at this record's availability
            lag = (pe_date[j] - pe_date) / np.timedelta64(1, "D")
            ok = (lag >= 358) & (lag <= 372) & (avail <= avail[j])
            if metric == "revenue":
                ok &= np.abs(duration - duration[j]) <= 15
            if ok.any():
                cand = np.where(ok)[0]
                k = cand[np.lexsort((group["version"].values[cand], avail[cand]))[-1]]
                if value[j] > 0 and value[k] > 0:
                    growth[i] = np.log(value[j] / value[k])
            if group["version"].values[j] > 1:
                prev = [
                    p
                    for p in range(len(idx))
                    if key[p] == key[j]
                    and group["version"].values[p] == group["version"].values[j] - 1
                ]
                if prev and value[prev[0]] > 0 and value[j] > 0:
                    revision_change[i] = np.log(value[j] / value[prev[0]])
    records["growth"] = growth
    records["revision_change"] = revision_change
    return records


def build_events(sessions):
    from .records import true_acceptance

    universe = pd.read_parquet(PROCESSED / "prices/universe_tickers.parquet")
    filings = true_acceptance(pd.read_parquet(PROCESSED / "sec/filings.parquet"))
    records = pd.read_parquet(PROCESSED / "sec/records.parquet")
    events = filings[filings["form"].isin(["10-Q", "10-K"])].copy()
    events = events[events["accn"].isin(set(records["accn"]))]
    events = events.merge(universe[["cik", "ticker", "family", "sic"]], on="cik")
    events = events[events["filing_date"] >= FIRST_DECISION]
    events = events[events["accepted_utc"] != ""]  # every event has an authoritative time (checked)
    timing = executable_from_acceptance(events["accepted_utc"])
    events["exec_session"] = pd.to_datetime(timing["exec_session"])
    events["executable_at"] = pd.to_datetime(timing["executable_at"])
    events = events.dropna(subset=["exec_session"])
    events["s"] = date_to_ordinal(events["exec_session"], sessions)
    eight_k = filings[(filings["form"] == "8-K") & filings["items"].fillna("").str.contains("2.02")]
    eight_k = eight_k[eight_k["cik"].isin(universe["cik"])].copy()
    exact = eight_k["accepted_utc"] != ""
    when = pd.Series(pd.NaT, index=eight_k.index, dtype="datetime64[ns]")
    if exact.any():
        when[exact] = pd.to_datetime(
            executable_from_acceptance(eight_k.loc[exact, "accepted_utc"])["exec_session"]
        )
    if (
        ~exact
    ).any():  # no authoritative time: day-level fallback (first session after filing date)
        from .sessions import executable_from_filing_date

        when[~exact] = pd.to_datetime(
            executable_from_filing_date(eight_k.loc[~exact, "filing_date"])["exec_session"]
        )
    eight_k["s"] = date_to_ordinal(when, sessions)
    return universe, events.reset_index(drop=True), eight_k, records


def price_matrix(panel, tickers, sessions):
    """tr_open / quoted_close / dollar_volume as (sessions x tickers) arrays."""
    sub = panel[panel["ticker"].isin(tickers)]
    out = {}
    for column in ("tr_open", "quoted_close", "dollar_volume"):
        wide = sub.pivot(index="date", columns="ticker", values=column).reindex(sessions)
        out[column] = wide
    return out


def identity_check(universe, sessions, prices):
    """Implied market value (quoted close * shares) vs reported public float, per 10-K."""
    raw = pd.read_parquet(PROCESSED / "sec/raw_facts.parquet")
    floats = raw[(raw["metric"] == "public_float") & (raw["unit"] == "USD")]
    shares = raw[(raw["metric"] == "shares_outstanding") & (raw["unit"] == "shares")]
    rows = []
    ticker_of = dict(zip(universe["cik"], universe["ticker"]))
    close = prices["quoted_close"]
    for (cik, accn), f in floats.groupby(["cik", "accn"]):
        ticker = ticker_of.get(cik)
        if ticker is None or ticker not in close.columns:
            continue
        f = f.iloc[0]
        sh = shares[(shares["cik"] == cik) & (shares["accn"] == accn)]
        if sh.empty or f["val"] <= 0:
            continue
        ordinal = date_to_ordinal([f["end"]], sessions)[0]
        if ordinal < 0:
            continue
        price = close[ticker].iloc[max(0, ordinal - 5) : ordinal + 1].dropna()
        if price.empty:
            continue
        cap = price.iloc[-1] * sh.sort_values("end")["val"].iloc[-1]
        rows.append(
            {
                "cik": cik,
                "ticker": ticker,
                "float_date": f["end"],
                "public_float": f["val"],
                "implied_cap": cap,
                "ratio": f["val"] / cap if cap > 0 else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    table["ok"] = table["ratio"].between(0.05, 1.5)
    by_cik = table.groupby("cik")["ok"].mean()
    return table, set(by_cik[by_cik < 0.5].index)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    sessions, _ = session_ordinals()
    universe, events, eight_k, records = build_events(sessions)
    counts = {"events_candidate": len(events), "ciks_candidate": int(events["cik"].nunique())}
    panel = pd.read_parquet(PROCESSED / "prices/panel.parquet")
    tickers = sorted(set(events["ticker"].dropna()) | {"XLI"})
    prices = price_matrix(panel, tickers, sessions)
    identity, bad_ciks = identity_check(universe, sessions, prices)
    identity.to_parquet(OUT / "identity_check.parquet", index=False)
    counts["identity_checks"] = len(identity)
    counts["identity_checks_failed"] = int((~identity["ok"]).sum())
    counts["identity_ciks_excluded"] = sorted(int(c) for c in bad_ciks)
    tr_open = prices["tr_open"]
    bench = tr_open["XLI"].values
    rec = record_table(records, sessions)
    rec_by_cik = {cik: g for cik, g in rec.groupby("cik")}
    k_by_cik = {cik: g["s"].values for cik, g in eight_k.groupby("cik")}
    filing_s_by_cik = {cik: g["s"].values for cik, g in events.groupby("cik")}

    n_events = len(events)
    facts = np.zeros((n_events, FACT_TOKENS, 10), dtype=np.float32)
    slots = np.full((n_events, FACT_TOKENS), -1, dtype=np.int64)
    ranks = np.full((n_events, FACT_TOKENS), -1, dtype=np.int64)
    market = np.zeros((n_events, MARKET_TOKENS, 4), dtype=np.float32)
    tab_rows, keep, reasons = [], np.zeros(n_events, bool), {}

    def drop(reason):
        reasons[reason] = reasons.get(reason, 0) + 1

    for i, ev in events.iterrows():
        cik, s, ticker = ev["cik"], int(ev["s"]), ev["ticker"]
        if cik in bad_ciks:
            drop("identity_check_failed")
            continue
        if ticker not in tr_open.columns:
            drop("no_price_series")
            continue
        if s - HISTORY_SESSIONS - 1 < 0 or s >= len(sessions):
            drop("outside_calendar")
            continue
        px = tr_open[ticker].values
        window = px[s - MARKET_TOKENS - 1 : s + 1]  # opens s-97 .. s
        bwin = bench[s - MARKET_TOKENS - 1 : s + 1]
        if np.isnan(px[s]) or np.isnan(px[s - HISTORY_SESSIONS]):
            drop("insufficient_price_history")
            continue
        stock_r = window[1:] / window[:-1] - 1
        bench_r = bwin[1:] / bwin[:-1] - 1
        excess = (stock_r - bench_r)[-MARKET_TOKENS:]  # returns ending at opens s-95 .. s
        if np.isnan(excess).sum() > 5:
            drop("missing_returns_in_window")
            continue
        excess = np.nan_to_num(excess) / 0.01
        quoted = prices["quoted_close"][ticker].values[s - 1]
        dollar_volume = np.nanmedian(prices["dollar_volume"][ticker].values[s - 20 : s])
        if not quoted >= MIN_PRICE:
            drop("price_below_5")
            continue
        if not dollar_volume >= MIN_DOLLAR_VOLUME:
            drop("dollar_volume_below_1m")
            continue
        r = rec_by_cik.get(cik)
        visible_all = r[
            (r["avail"] <= s)
            & (r["pe_date"] >= ev["exec_session"] - pd.DateOffset(years=LOOKBACK_YEARS))
        ]
        visible = visible_all[visible_all["metric"].isin(TOKEN_METRICS)]
        visible = visible.dropna(subset=["growth"]).sort_values(["avail", "version", "metric"])
        if visible.empty:
            drop("no_transformable_records")
            continue
        tokens = visible.iloc[-FACT_TOKENS:]
        slot_of, count_of = {}, {}
        for j, t in enumerate(tokens.itertuples()):
            key = (t.metric, t.period_start, t.period_end)
            slots[i, j] = slot_of.setdefault(key, len(slot_of))
            ranks[i, j] = min(count_of.get(key, 0), 2)
            count_of[key] = count_of.get(key, 0) + 1
            m = METRICS.index(t.metric)
            facts[i, j] = (
                np.clip(t.growth / GROWTH_SCALE, -4, 4),
                m == 0,
                m == 1,
                m == 2,
                min(t.version - 1, 2) / 2,
                t.version > 1,
                (s - t.pe) / 64,
                (s - t.avail) / 64,
                (t.avail - t.pe) / 16,
                1.0,
            )
        start = s - MARKET_TOKENS
        arrivals = np.zeros(MARKET_TOKENS)
        for u in np.concatenate(
            [k_by_cik.get(cik, np.array([], int)), filing_s_by_cik.get(cik, np.array([], int))]
        ):
            if start < u <= s:
                arrivals[u - 1 - start] += 1
        market[i] = np.stack(
            (excess, np.abs(excess), np.arange(-MARKET_TOKENS, 0) / MARKET_TOKENS, arrivals),
            axis=-1,
        )
        row = tabular(
            visible_all.dropna(subset=["growth"]),
            s,
            excess,
            k_by_cik.get(cik, np.array([], int)),
            arrivals,
        )
        row["recent_revision"] = float(
            ((visible["version"] > 1) & (visible["avail"] > s - 63)).any()
        )
        # Level-based orders proxy: NOT in the primary feature set (the tokens carry growth
        # rates only); used in the pre-registered secondary orders-proxy check.
        row["book_to_bill"] = book_to_bill(visible_all)
        row["n_tokens"] = len(tokens)
        tab_rows.append({"i": i, **row})
        keep[i] = True

    tab = pd.DataFrame(tab_rows).set_index("i")
    events = events.join(tab, how="left")
    events["kept"] = keep
    events["held_out_issuer"] = events["cik"].map(held_out)
    events["vol20"] = events["vol_20"]
    counts["dropped"] = reasons
    counts["events_kept"] = int(keep.sum())
    counts["ciks_kept"] = int(events.loc[keep, "cik"].nunique())
    counts["kept_by_family"] = events.loc[keep, "family"].value_counts().to_dict()
    counts["kept_by_year"] = (
        events.loc[keep, "exec_session"].dt.year.value_counts().sort_index().to_dict()
    )
    counts["held_out_issuer_events"] = int(events.loc[keep, "held_out_issuer"].sum())
    ev = events[keep].reset_index(drop=True)
    idx = np.where(keep)[0]
    np.savez_compressed(
        OUT / "tokens.npz", facts=facts[idx], slots=slots[idx], ranks=ranks[idx], market=market[idx]
    )
    ev.to_parquet(OUT / "events.parquet", index=False)
    counts["files"] = {
        name: sha256_file(OUT / name)
        for name in ("events.parquet", "tokens.npz", "identity_check.parquet")
    }
    write_json(OUT / "build.json", counts)
    return counts


def _latest_two(group):
    """Latest-known growth of the newest period and of the period before it."""
    latest = group.sort_values(["pe", "avail", "version"]).groupby("pe").tail(1)
    g = latest["growth"].values
    return (
        g[-1] if len(g) else np.nan,
        g[-2] if len(g) > 1 else np.nan,
        latest.iloc[-1] if len(latest) else None,
    )


def tabular(visible, s, excess, eight_k_s, arrivals):
    out = {}
    current = {}
    for metric in METRICS:
        group = visible[visible["metric"] == metric]
        g1, g0, last = _latest_two(group)
        current[metric] = (g1, last)
        recent = group[(group["version"] > 1) & (group["avail"] > s - 63)]
        out.update(
            {
                f"{metric}_has": float(last is not None),
                f"{metric}_growth": g1 / GROWTH_SCALE if last is not None else np.nan,
                f"{metric}_accel": (g1 - g0) / GROWTH_SCALE if last is not None else np.nan,
                f"{metric}_age": (s - last["avail"]) / 64 if last is not None else np.nan,
                f"{metric}_lag": (last["avail"] - last["pe"]) / 16 if last is not None else np.nan,
                f"{metric}_is_revision": float(last["version"] > 1) if last is not None else np.nan,
                f"{metric}_revisions_63": float(len(recent)),
                f"{metric}_revision_size_63": float(np.nansum(np.abs(recent["revision_change"])))
                / GROWTH_SCALE,
            }
        )
    g_rev, g_inv, g_bl = (current[m][0] for m in METRICS)
    out["inventory_minus_revenue"] = (g_inv - g_rev) / GROWTH_SCALE
    out["backlog_minus_revenue"] = (g_bl - g_rev) / GROWTH_SCALE
    r = excess  # % units, last element ends at the decision open
    out.update(
        {
            "ret_1": r[-1],
            "ret_5": r[-5:].sum(),
            "ret_20": r[-20:].sum(),
            "ret_60": r[-60:].sum(),
            "vol_20": float(np.std(r[-20:])),
            "vol_60": float(np.std(r[-60:])),
            "arrivals_5": float(arrivals[-5:].sum()),
        }
    )
    past = eight_k_s[(eight_k_s <= s) & (eight_k_s > s - EIGHT_K_WINDOW)]
    if len(past):
        u = past.max()
        offset = len(r) - (s - u) - 1  # index of the return that ends at the 8-K's open
        out["has_8k"] = 1.0
        out["sessions_since_8k"] = (s - u) / 64
        out["reaction_8k"] = float(r[offset]) if offset >= 0 else np.nan
        out["drift_since_8k"] = float(r[offset + 1 :].sum())  # after the immediate reaction
    else:
        out.update(
            {
                "has_8k": 0.0,
                "sessions_since_8k": np.nan,
                "reaction_8k": np.nan,
                "drift_since_8k": np.nan,
            }
        )
    return out


def book_to_bill(visible):
    """(revenue_q + backlog_q - backlog_{q-1}) / revenue_q for the newest quarter with all three."""
    rev = (
        visible[visible["metric"] == "revenue"]
        .sort_values(["pe", "avail", "version"])
        .groupby("period_end")
        .tail(1)
    )
    bl = (
        visible[visible["metric"] == "backlog"]
        .sort_values(["pe", "avail", "version"])
        .groupby("period_end")
        .tail(1)
    )
    if len(bl) < 2 or rev.empty:
        return np.nan
    bl = bl.sort_values("pe_date")
    for _, row in rev.sort_values("pe_date", ascending=False).iterrows():
        now = bl[bl["period_end"] == row["period_end"]]
        before = bl[
            (bl["pe_date"] < row["pe_date"] - pd.Timedelta(days=80))
            & (bl["pe_date"] > row["pe_date"] - pd.Timedelta(days=100))
        ]
        if len(now) and len(before) and row["value"] > 0:
            ratio = (row["value"] + now["value"].iloc[0] - before["value"].iloc[-1]) / row["value"]
            return float(np.clip(ratio, -1, 3))
    return np.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build",))
    args = parser.parse_args()
    print(json.dumps(build(), indent=2, default=str))
