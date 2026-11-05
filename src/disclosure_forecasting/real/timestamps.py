"""Authoritative EDGAR acceptance times (Eastern) replacing the submissions-JSON field.

Gate 1 found that `acceptanceDateTime` in the SEC's bulk submissions.zip is not reliably UTC: in
the audit sample it was true UTC for 40 of 120 filings, Eastern time labelled "Z" for 20 and
Eastern time shifted by twice the UTC offset for 60, while the EDGAR filing index page and the
XBRL RSS archive agreed with each other. The live data.sec.gov API returned the correct UTC value
in every spot check (`api-spotcheck`). Using the JSON value as UTC would make some filings
look executable hours before they were public (look-ahead). Sources used instead:

1. SEC monthly XBRL RSS archives (edgar/monthly/xbrlrss-YYYY-MM.xml): acceptanceDatetime in
   Eastern time for every XBRL filing (all 10-Q/10-K; 8-Ks once they carry cover-page XBRL).
2. The filing index page ("Accepted", Eastern) for every needed filing not in the RSS archive
   (mostly earnings 8-Ks before mid-2019).

Output: data/processed/sec/acceptance.parquet  accn -> accepted_et, accepted_utc (true), source.
"""

import argparse
import gzip
import json
import re
from datetime import date
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

from .audit import get as get_cached, parse_index, folder
from .common import (
    PROCESSED,
    RAW,
    SEC_MIN_INTERVAL_S,
    SEC_USER_AGENT,
    Throttle,
    sha256_file,
    write_json,
)

RSS = RAW / "sec/xbrlrss"
OUT = PROCESSED / "sec"
_throttle = Throttle(SEC_MIN_INTERVAL_S)


def fetch_rss(first=(2009, 4), last=(2026, 10)):
    RSS.mkdir(parents=True, exist_ok=True)
    year, month = first
    status = {}
    while (year, month) <= last:
        name = f"xbrlrss-{year}-{month:02d}.xml"
        path = RSS / (name + ".gz")
        if not path.exists():
            _throttle.wait()
            url = f"https://www.sec.gov/Archives/edgar/monthly/{name}"
            request = Request(
                url, headers={"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip"}
            )
            try:
                with urlopen(request, timeout=300) as response:
                    body = response.read()
                    if response.headers.get("Content-Encoding") != "gzip":
                        body = gzip.compress(body)
                path.write_bytes(body)
                write_json(
                    RSS / (name + ".manifest.json"),
                    {
                        "url": url,
                        "sha256_gz": sha256_file(path),
                        "retrieved": date.today().isoformat(),
                    },
                )
                status[name] = "ok"
            except HTTPError as error:
                status[name] = f"http_{error.code}"
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return status


ITEM = re.compile(r"<item>(.*?)</item>", re.S)


def parse_rss():
    rows = []
    for path in sorted(RSS.glob("xbrlrss-*.xml.gz")):
        text = gzip.decompress(path.read_bytes()).decode("utf-8", "replace")
        for item in ITEM.findall(text):
            accn = re.search(r"<edgar:accessionNumber>([^<]+)<", item)
            accepted = re.search(r"<edgar:acceptanceDatetime>(\d{14})<", item)
            form = re.search(r"<edgar:formType>([^<]+)<", item)
            if accn and accepted:
                rows.append(
                    {
                        "accn": accn.group(1).strip(),
                        "accepted_et": accepted.group(1),
                        "rss_form": form.group(1) if form else None,
                        "rss_file": path.name,
                    }
                )
    table = pd.DataFrame(rows).drop_duplicates("accn")
    table["accepted_et"] = pd.to_datetime(table["accepted_et"], format="%Y%m%d%H%M%S")
    return table


def needed_filings():
    """Filings whose time matters: every record's filing (events are a subset) and every
    earnings 8-K (Item 2.02) of universe CIKs that have a price series."""
    from .records import PERIODIC

    filings = pd.read_parquet(PROCESSED / "sec/filings.parquet")
    # Every periodic report that carries one of the raw facts (records are built from these, so
    # this runs before records.py); the first run in October 2026 used records.parquet's
    # filings instead, a subset of this list for periodic forms.
    facts = pd.read_parquet(PROCESSED / "sec/raw_facts.parquet", columns=["accn"])
    tickers = pd.read_parquet(PROCESSED / "prices/universe_tickers.parquet")
    priced = set(tickers.loc[tickers["ticker"].notna(), "cik"])
    need = filings["accn"].isin(set(facts["accn"])) & filings["form"].isin(PERIODIC)
    need |= (
        (filings["form"] == "8-K")
        & filings["items"].fillna("").str.contains("2.02")
        & filings["cik"].isin(priced)
    )
    need &= filings["filing_date"] >= "2009-01-01"
    return filings[need]


def fetch_index_pages(filings, workers=8):
    """Index pages in parallel; the shared throttle keeps the total at <= 4 requests/s."""
    from concurrent.futures import ThreadPoolExecutor

    def one(row):
        info = parse_index(get_cached(f"{folder(row.cik, row.accn)}/{row.accn}-index.htm"))
        return {
            "accn": row.accn,
            "accepted_et": info["accepted_et"],
            "index_filing_date": info["filing_date"],
        }

    with ThreadPoolExecutor(workers) as pool:
        out = list(pool.map(one, filings.itertuples()))
    table = pd.DataFrame(out)
    table["accepted_et"] = pd.to_datetime(table["accepted_et"], errors="coerce")
    return table


def to_utc(eastern):
    return (
        eastern.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
        .dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    )


def build():
    need = needed_filings()
    rss = parse_rss()
    have = need.merge(rss, on="accn", how="left")
    missing = have[have["accepted_et"].isna()][["cik", "accn"]]
    index = fetch_index_pages(missing)
    have = have.set_index("accn")
    index = index.set_index("accn")
    have["source"] = "xbrl_rss"
    have.loc[index.index, "accepted_et"] = index["accepted_et"]
    have.loc[index.index, "source"] = "index_page"
    have["accepted_utc_true"] = to_utc(have["accepted_et"])
    have = have.reset_index()
    # Gate 1 statistic: how the submissions-JSON value relates to the true time
    json_time = pd.to_datetime(have["accepted_utc"].str[:19])
    true_utc = pd.to_datetime(have["accepted_utc_true"].str[:19])
    et = have["accepted_et"]
    diff_utc = (json_time - true_utc).dt.total_seconds().abs()
    diff_et = (json_time - et).dt.total_seconds()
    offset = (true_utc - et).dt.total_seconds()
    have["json_pattern"] = "other"
    have.loc[diff_utc < 2, "json_pattern"] = "true_utc"
    have.loc[diff_et.abs() < 2, "json_pattern"] = "eastern_labelled_utc"
    have.loc[(diff_et - 2 * offset).abs() < 2, "json_pattern"] = "eastern_shifted_twice"
    have.loc[have["accepted_et"].isna(), "json_pattern"] = "no_true_time"
    keep = [
        "cik",
        "accn",
        "form",
        "filing_date",
        "accepted_utc",
        "accepted_et",
        "accepted_utc_true",
        "source",
        "json_pattern",
    ]
    have[keep].rename(columns={"accepted_utc": "json_accepted"}).to_parquet(
        OUT / "acceptance.parquet", index=False
    )
    stats = {
        "needed_filings": len(have),
        "source": have["source"].value_counts().to_dict(),
        "no_true_time": int(have["accepted_et"].isna().sum()),
        "json_pattern": have["json_pattern"].value_counts().to_dict(),
        "json_pattern_by_year": have.assign(y=have["filing_date"].str[:4])
        .groupby(["y", "json_pattern"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index"),
    }
    write_json(OUT / "acceptance.json", stats)
    return stats


def api_spotcheck(per_pattern=2, seed=7):
    """Compare bulk-zip values with the live data.sec.gov API for a few filings per pattern."""
    import time

    from .common import ROOT

    acceptance = pd.read_parquet(OUT / "acceptance.parquet")
    picks = pd.concat(
        [
            acceptance[acceptance["json_pattern"] == p].sample(per_pattern, random_state=seed)
            for p in ("true_utc", "eastern_labelled_utc", "eastern_shifted_twice")
        ]
    )
    rows = []
    for row in picks.itertuples():

        def api(path):
            _throttle.wait()
            return json.loads(
                urlopen(
                    Request(
                        "https://data.sec.gov/submissions/" + path,
                        headers={"User-Agent": SEC_USER_AGENT},
                    ),
                    timeout=60,
                ).read()
            )

        main = api(f"CIK{int(row.cik):010d}.json")
        live = None
        blocks = [main["filings"]["recent"]]
        for extra in main["filings"]["files"]:
            if extra["filingFrom"] <= row.filing_date <= extra["filingTo"]:
                blocks.append(api(extra["name"]))
        for block in blocks:
            if row.accn in block["accessionNumber"]:
                live = block["acceptanceDateTime"][block["accessionNumber"].index(row.accn)]
                break
        rows.append(
            {
                "accn": row.accn,
                "form": row.form,
                "pattern": row.json_pattern,
                "bulk_zip": row.json_accepted,
                "live_api": live,
                "true_utc": row.accepted_utc_true,
                "live_api_correct": (live or "")[:19] == row.accepted_utc_true[:19],
            }
        )
        time.sleep(0.3)
    result = {
        "checked": len(rows),
        "live_api_correct": sum(r["live_api_correct"] for r in rows),
        "rows": rows,
    }
    write_json(ROOT / "data/gate1/api_spotcheck.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fetch-rss", "build", "count", "api-spotcheck"))
    a = parser.parse_args()
    if a.command == "fetch-rss":
        status = fetch_rss()
        print(json.dumps(pd.Series(status).value_counts().to_dict() if status else {}))
    elif a.command == "api-spotcheck":
        print(json.dumps(api_spotcheck(), indent=2))
    elif a.command == "count":
        need = needed_filings()
        print(
            json.dumps(
                {"needed": len(need), "by_form": need["form"].value_counts().head(8).to_dict()}
            )
        )
    else:
        print(json.dumps(build(), indent=2, default=str))
