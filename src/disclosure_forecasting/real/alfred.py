"""Point-in-time macro orders from ALFRED vintages (keyless CSV downloads, no account).

Series: NEWORDER, Manufacturers' New Orders: Nondefense Capital Goods Excluding Aircraft
(Census M3, monthly, millions of USD, seasonally adjusted): the standard "core capital goods
orders" indicator for industrial demand. Every vintage is one CSV; consecutive vintages give the
first release and each later revision of every month.

Timing: ALFRED vintages are dated by day only. Census publishes at 08:30 or 10:00 ET, but the
time is not in the data, so a vintage is treated as executable at the first session open AFTER
its vintage date (never the same morning). Gate 1 rule: no invented intraday time.

Outputs: data/raw/alfred/NEWORDER/<vintage>.csv (cached), data/processed/alfred/neworder.parquet
(one row per month x vintage where the value changed: version lineage) and macro features per
G4 event in data/processed/g4/macro.parquet (secondary analysis only).
"""

import argparse
import io
import json
import re
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .common import PROCESSED, RAW, Throttle, write_json
from .sessions import executable_from_filing_date, session_index

SERIES = "NEWORDER"
CACHE = RAW / "alfred" / SERIES
OUT = PROCESSED / "alfred"
_throttle = Throttle(1.0)


def _get(url, attempts=5):
    import time
    from urllib.error import HTTPError, URLError

    for attempt in range(attempts):
        _throttle.wait()
        try:
            with urlopen(
                Request(url, headers={"User-Agent": "research script (no account)"}), timeout=90
            ) as r:
                return r.read()
        except (HTTPError, URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(10 * (attempt + 1))  # server errors (502) seen on 2026-10-26: back off


def vintages():
    CACHE.mkdir(parents=True, exist_ok=True)
    page = CACHE / "downloaddata.html"
    if not page.exists():
        page.write_bytes(_get(f"https://alfred.stlouisfed.org/series/downloaddata?seid={SERIES}"))
    dates = sorted(set(re.findall(r'<option value="(\d{4}-\d{2}-\d{2})">', page.read_text())))
    return [d for d in dates if d >= "2008-01-01"]


def fetch(workers=1):
    from concurrent.futures import ThreadPoolExecutor

    def one(d):
        path = CACHE / f"{d}.csv"
        if not path.exists():
            path.write_bytes(
                _get(
                    f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={SERIES}&vintage_date={d}"
                )
            )
        return d

    with ThreadPoolExecutor(workers) as pool:
        done = list(pool.map(one, vintages()))
    return {"vintages": len(done)}


def build():
    rows = []
    for d in vintages():
        frame = pd.read_csv(io.BytesIO((CACHE / f"{d}.csv").read_bytes()))
        frame.columns = ["month", "value"]
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna()
        frame["vintage"] = d
        rows.append(frame[frame["month"] >= "2005-01-01"])
    table = pd.concat(rows).sort_values(["month", "vintage"])
    # lineage: a new version only when the month's value changed
    table["changed"] = table.groupby("month")["value"].diff().fillna(1) != 0
    lineage = table[table["changed"]].copy()
    lineage["version"] = lineage.groupby("month").cumcount() + 1
    timing = executable_from_filing_date(lineage["vintage"])
    lineage["exec_session"] = pd.to_datetime(timing["exec_session"])
    OUT.mkdir(parents=True, exist_ok=True)
    lineage.drop(columns="changed").to_parquet(OUT / "neworder.parquet", index=False)
    info = {
        "vintages": int(table["vintage"].nunique()),
        "months": int(lineage["month"].nunique()),
        "records": len(lineage),
        "revisions": int((lineage["version"] > 1).sum()),
        "median_versions_per_month": float(lineage.groupby("month")["version"].max().median()),
    }
    write_json(OUT / "neworder.json", info)
    return info


def features():
    """Macro features per G4 event using only vintages executable at or before the decision."""
    lineage = pd.read_parquet(OUT / "neworder.parquet")
    sessions = session_index()
    lineage["avail"] = np.searchsorted(sessions.values, lineage["exec_session"].values)
    events = pd.read_parquet(PROCESSED / "g4/events.parquet", columns=["s"])
    out = []
    by_avail = lineage.sort_values("avail")
    for s in events["s"].values:
        known = by_avail[by_avail["avail"] <= s]
        latest = (
            known.sort_values(["month", "version"])
            .groupby("month")
            .tail(1)
            .set_index("month")["value"]
        )
        first = (
            known.sort_values(["month", "version"])
            .groupby("month")
            .head(1)
            .set_index("month")["value"]
        )
        if len(latest) < 14:
            out.append(
                {
                    "macro_orders_growth": np.nan,
                    "macro_orders_accel": np.nan,
                    "macro_orders_revision": np.nan,
                }
            )
            continue
        months = latest.index.sort_values()
        last, prev = months[-1], months[-2]

        def g(m):
            return np.log(latest[m] / latest[months[list(months).index(m) - 12]])

        out.append(
            {
                "macro_orders_growth": g(last) / 0.25,
                "macro_orders_accel": (g(last) - g(prev)) / 0.25,
                "macro_orders_revision": np.log(latest[prev] / first[prev]) / 0.25,
            }
        )
    frame = pd.DataFrame(out)
    frame.to_parquet(PROCESSED / "g4/macro.parquet", index=False)
    return frame.describe().to_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fetch", "build", "features"))
    a = parser.parse_args()
    result = {"fetch": fetch, "build": build, "features": features}[a.command]()
    print(
        json.dumps(result if not isinstance(result, dict) else result, indent=2, default=str)[:3000]
    )
