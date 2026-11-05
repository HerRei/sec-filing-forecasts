"""Daily prices from Yahoo Finance's public chart endpoint (no key, no login), cached forever.

Stooq, the first choice, answered with a JavaScript proof-of-work bot check on 2026-10-26, so it
cannot be used from a script without circumventing that check (not done).

Limits, all reported in Gate 1:
- Only tickers listed TODAY can be fetched: firms that were acquired, went private or failed are
  missing (survivorship). The universe is defined from SEC data, so the loss is measured.
- History sits under the current ticker. A ticker reused by an unrelated company would attach the
  wrong history: series that begin long before the CIK's first periodic filing are flagged.
- Yahoo's open/high/low/close are split-adjusted, adjclose is split- and dividend-adjusted.
  Total-return open = open * adjclose / close. Days with |total return| > 40 % are flagged and
  checked against split events.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .common import PROCESSED, RAW, Throttle, sha256_file, write_json
from .sessions import session_index

CACHE = RAW / "prices/yahoo"
OUT = PROCESSED / "prices"
URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start}&period2={end}"
    "&interval=1d&events=div%2Csplits&includeAdjustedClose=true"
)
BENCHMARKS = ("XLI", "SPY")
START = int(datetime(2005, 1, 1, tzinfo=timezone.utc).timestamp())
MAJOR = {"NYSE", "Nasdaq", "NYSE American", "NYSE MKT", "NYSE Arca", "CBOE"}


def primary_ticker(tickers, exchanges):
    pairs = [(t, e) for t, e in zip(tickers, exchanges) if e in MAJOR]
    if not pairs:
        return None, None
    common = [p for p in pairs if "-" not in p[0]] or pairs
    return common[0]


def fetch(tickers, end_date="2026-10-24", interval_s=0.8):
    CACHE.mkdir(parents=True, exist_ok=True)
    throttle = Throttle(interval_s)
    end = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp()) + 86400
    status = {}
    for ticker in tickers:
        path = CACHE / f"{ticker}.json"
        if path.exists():
            status[ticker] = "cached"
            continue
        url = URL.format(ticker=ticker, start=START, end=end)
        for attempt in range(3):
            throttle.wait()
            try:
                request = Request(url, headers={"User-Agent": "Mozilla/5.0 (research; no-login)"})
                with urlopen(request, timeout=60) as response:
                    body = response.read()
                status[ticker] = "ok"
                break
            except HTTPError as error:
                status[ticker] = f"http_{error.code}"
                if error.code == 404:
                    body = None
                    break
                if error.code == 429:
                    time.sleep(30 * (attempt + 1))
            except URLError as error:
                status[ticker] = f"url_error_{error.reason}"
                time.sleep(5)
        else:
            body = None
        if body is None and status[ticker] != "http_404":
            continue
        if body is None:
            body = json.dumps({"chart": {"result": None, "error": "404"}}).encode()
        path.write_bytes(body)
        write_json(
            CACHE / f"{ticker}.manifest.json",
            {
                "url": url,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "sha256": sha256_file(path),
            },
        )
    return status


def parse(ticker):
    path = CACHE / f"{ticker}.json"
    if not path.exists():
        return None, {}
    data = json.loads(path.read_text())
    result = (data.get("chart") or {}).get("result")
    if not result:
        return None, {}
    result = result[0]
    stamps = result.get("timestamp")
    if not stamps:
        return None, {}
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(stamps, unit="s", utc=True)
            .tz_convert("America/New_York")
            .tz_localize(None)
            .normalize(),
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
            "adjclose": adj if adj else quote["close"],
        }
    )
    frame = frame.dropna(subset=["close"]).drop_duplicates("date", keep="last")
    frame["ticker"] = ticker
    events = result.get("events") or {}
    splits = [
        {
            "date": pd.Timestamp(v["date"], unit="s", tz="UTC")
            .tz_convert("America/New_York")
            .tz_localize(None)
            .normalize(),
            "ratio": v.get("splitRatio"),
            "factor": float(v.get("numerator") or 1) / float(v.get("denominator") or 1),
        }
        for v in (events.get("splits") or {}).values()
    ]
    return frame, {"splits": splits, "first_trade": result["meta"].get("firstTradeDate")}


def build_panel(universe_path):
    """Total-return panel on XNYS sessions plus per-ticker data-quality flags."""
    universe = pd.read_parquet(universe_path)
    sessions = session_index()
    frames, quality = [], []
    tickers = list(universe["ticker"].dropna().unique()) + list(BENCHMARKS)
    for ticker in tickers:
        frame, meta = parse(ticker)
        if frame is None:
            quality.append({"ticker": ticker, "status": "no_data"})
            continue
        frame = frame[(frame["close"] > 0) & (frame["open"].fillna(0) >= 0)].copy()
        factor = frame["adjclose"] / frame["close"]
        frame["tr_open"] = frame["open"] * factor
        frame["tr_close"] = frame["adjclose"]
        frame["dollar_volume"] = frame["close"] * frame["volume"].fillna(0)  # split-invariant
        # Yahoo prices are split-adjusted backwards; the price actually quoted on a day is
        # close * product of the split factors of all later splits (for the $5 filter).
        after = np.ones(len(frame))
        for split in meta.get("splits", []):
            after *= np.where(
                frame["date"].values < np.datetime64(split["date"]), split["factor"], 1.0
            )
        frame["quoted_close"] = frame["close"] * after
        ret = frame["tr_close"].pct_change()
        jumps = frame.loc[ret.abs() > 0.4, "date"]
        split_days = {s["date"] for s in meta.get("splits", [])}
        near_split = sum(any(abs((d - s).days) <= 3 for s in split_days) for d in jumps)
        expected = set(
            sessions[(sessions >= frame["date"].min()) & (sessions <= frame["date"].max())]
        )
        have = set(frame["date"])
        quality.append(
            {
                "ticker": ticker,
                "status": "ok",
                "first_date": frame["date"].min(),
                "last_date": frame["date"].max(),
                "rows": len(frame),
                "missing_sessions": len(expected - have),
                "non_session_rows": len(have - expected),
                "jumps_over_40pct": int(len(jumps)),
                "jumps_near_split": int(near_split),
                "splits": len(split_days),
                "zero_open_rows": int((frame["open"].fillna(0) == 0).sum()),
            }
        )
        frame = frame[frame["date"].isin(sessions)]
        frames.append(
            frame[
                [
                    "date",
                    "ticker",
                    "open",
                    "close",
                    "quoted_close",
                    "tr_open",
                    "tr_close",
                    "volume",
                    "dollar_volume",
                ]
            ]
        )
    panel = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT / "panel.parquet", index=False)
    quality = pd.DataFrame(quality)
    quality.to_parquet(OUT / "quality.parquet", index=False)
    summary = {
        "tickers_requested": len(tickers),
        "tickers_with_data": int((quality["status"] == "ok").sum()),
        "rows": len(panel),
        "missing_sessions_total": int(
            quality.get("missing_sessions", pd.Series(dtype=float)).fillna(0).sum()
        ),
        "non_session_rows_total": int(
            quality.get("non_session_rows", pd.Series(dtype=float)).fillna(0).sum()
        ),
        "jumps_over_40pct": int(
            quality.get("jumps_over_40pct", pd.Series(dtype=float)).fillna(0).sum()
        ),
        "jumps_near_split": int(
            quality.get("jumps_near_split", pd.Series(dtype=float)).fillna(0).sum()
        ),
    }
    write_json(OUT / "panel.json", summary)
    return summary


def universe_tickers():
    universe = pd.read_parquet(PROCESSED / "sec/universe.parquet")
    picks = [
        primary_ticker(json.loads(t), json.loads(e))
        for t, e in zip(universe["tickers"], universe["exchanges"])
    ]
    universe["ticker"] = [p[0] for p in picks]
    universe["exchange"] = [p[1] for p in picks]
    universe.to_parquet(OUT / "universe_tickers.parquet", index=False)
    return universe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fetch", "panel"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "fetch":
        universe = universe_tickers()
        tickers = list(universe["ticker"].dropna().unique()) + list(BENCHMARKS)
        status = fetch(tickers)
        write_json(OUT / "fetch_status.json", status)
        print(json.dumps(pd.Series(status).value_counts().to_dict()))
    else:
        print(json.dumps(build_panel(OUT / "universe_tickers.parquet"), indent=2, default=str))
