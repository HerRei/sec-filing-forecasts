"""Universe, filing index and raw XBRL facts from the SEC bulk files (no per-company requests).

Inputs (immutable, sha256 in their manifests): data/raw/sec/bulk/submissions.zip,
companyfacts.zip, company_tickers_exchange.json. Outputs in data/processed/sec/:

  universe.parquet  one row per CIK passing the SIC/US/operating filters, with current tickers
  filings.parquet   every filing of those CIKs: accession, form, dates, acceptance time (UTC), items
  raw_facts.parquet every companyfacts row for the CONCEPTS of those CIKs, as published
  build.json        counts and input hashes
"""

import argparse
import json
import zipfile
from collections import Counter

import pandas as pd

from .common import CONCEPTS, PROCESSED, RAW, US_STATES, sic_in_universe, write_json

BULK = RAW / "sec/bulk"
OUT = PROCESSED / "sec"
FILING_FIELDS = (
    "accessionNumber",
    "filingDate",
    "reportDate",
    "acceptanceDateTime",
    "form",
    "items",
    "primaryDocument",
    "isXBRL",
    "isInlineXBRL",
)
PERIODIC = ("10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT", "10-QT", "10-KT/A", "10-QT/A")


def _filings_block(block, cik):
    n = len(block.get("accessionNumber", []))
    rows = []
    for i in range(n):
        rows.append({"cik": cik, **{f: block.get(f, [None] * n)[i] for f in FILING_FIELDS}})
    return rows


def build(first_year=2009):
    OUT.mkdir(parents=True, exist_ok=True)
    manifests = {
        name: json.loads((BULK / f"{name}.manifest.json").read_text())
        for name in ("submissions.zip", "companyfacts.zip")
    }
    facts_zip = zipfile.ZipFile(BULK / "companyfacts.zip")
    subs_zip = zipfile.ZipFile(BULK / "submissions.zip")
    subs_names = set(subs_zip.namelist())
    fact_ciks = sorted(
        int(n[3:13]) for n in facts_zip.namelist() if n.startswith("CIK") and n.endswith(".json")
    )
    counts = Counter(companyfacts_ciks=len(fact_ciks))
    universe, filings = [], []
    for cik in fact_ciks:
        name = f"CIK{cik:010d}.json"
        if name not in subs_names:
            counts["no_submissions_file"] += 1
            continue
        sub = json.loads(subs_zip.read(name))
        if not sic_in_universe(sub.get("sic")):
            continue
        counts["sic_in_range"] += 1
        business = (sub.get("addresses") or {}).get("business") or {}
        state = (business.get("stateOrCountry") or "").upper()
        if state not in US_STATES:
            counts["dropped_non_us_address"] += 1
            continue
        if sub.get("entityType") not in ("operating", None, ""):
            counts["dropped_entity_type"] += 1
            continue
        rows = _filings_block(sub["filings"]["recent"], cik)
        for extra in sub["filings"].get("files", []):
            if extra["name"] in subs_names:
                rows += _filings_block(json.loads(subs_zip.read(extra["name"])), cik)
            else:
                counts["missing_extra_submission_page"] += 1
        periodic = [
            r
            for r in rows
            if r["form"] in PERIODIC and (r["filingDate"] or "0") >= f"{first_year}-01-01"
        ]
        if not periodic:
            counts["dropped_no_periodic_since_first_year"] += 1
            continue
        if not any(r["form"] in ("10-Q", "10-K") for r in periodic):
            counts["dropped_no_10q_10k"] += 1
            continue
        counts["universe"] += 1
        universe.append(
            {
                "cik": cik,
                "name": sub.get("name"),
                "sic": int(sub["sic"]),
                "family": str(sub["sic"])[:2],
                "sic_description": sub.get("sicDescription"),
                "state": state,
                "state_of_incorporation": sub.get("stateOfIncorporation"),
                "fiscal_year_end": sub.get("fiscalYearEnd"),
                "tickers": json.dumps(sub.get("tickers") or []),
                "exchanges": json.dumps(sub.get("exchanges") or []),
                "former_names": json.dumps([f.get("name") for f in (sub.get("formerNames") or [])]),
                "category": sub.get("category"),
            }
        )
        filings += rows
    universe = pd.DataFrame(universe)
    filings = pd.DataFrame(filings).drop_duplicates("accessionNumber")
    filings = filings.rename(
        columns={
            "accessionNumber": "accn",
            "filingDate": "filing_date",
            "reportDate": "report_date",
            "acceptanceDateTime": "accepted_utc",
            "primaryDocument": "primary_doc",
            "isXBRL": "is_xbrl",
            "isInlineXBRL": "is_ixbrl",
        }
    )
    wanted = {c: m for m, concepts in CONCEPTS.items() for c in concepts}
    wanted["EntityCommonStockSharesOutstanding"] = "shares_outstanding"
    wanted["EntityPublicFloat"] = "public_float"
    raw = []
    for cik in universe["cik"]:
        data = json.loads(facts_zip.read(f"CIK{cik:010d}.json"))
        for taxonomy in ("us-gaap", "dei"):
            for concept, body in (data.get("facts", {}).get(taxonomy) or {}).items():
                if concept not in wanted:
                    continue
                for unit, items in body.get("units", {}).items():
                    for item in items:
                        raw.append(
                            {
                                "cik": cik,
                                "metric": wanted[concept],
                                "concept": concept,
                                "unit": unit,
                                "start": item.get("start"),
                                "end": item["end"],
                                "val": float(item["val"]),
                                "accn": item["accn"],
                                "fy": item.get("fy"),
                                "fp": item.get("fp"),
                                "form": item.get("form"),
                                "filed": item.get("filed"),
                                "frame": item.get("frame"),
                            }
                        )
    raw = pd.DataFrame(raw)
    counts["raw_fact_rows"] = len(raw)
    counts["raw_rows_by_metric"] = raw["metric"].value_counts().to_dict()
    universe.to_parquet(OUT / "universe.parquet", index=False)
    filings.to_parquet(OUT / "filings.parquet", index=False)
    raw.to_parquet(OUT / "raw_facts.parquet", index=False)
    write_json(
        OUT / "build.json",
        {"inputs": manifests, "counts": dict(counts), "first_year": first_year},
    )
    return counts


def concept_scan(prefixes=("Backlog", "Order", "Booking", "RemainingPerformance")):
    """Which standard concepts mention orders/backlog in the universe (Gate 1 coverage check)."""
    universe = pd.read_parquet(OUT / "universe.parquet")
    facts_zip = zipfile.ZipFile(BULK / "companyfacts.zip")
    hits = Counter()
    for cik in universe["cik"]:
        data = json.loads(facts_zip.read(f"CIK{cik:010d}.json"))
        for taxonomy, concepts in data.get("facts", {}).items():
            for concept in concepts:
                if any(p.lower() in concept.lower() for p in prefixes):
                    hits[f"{taxonomy}:{concept}"] += 1
    write_json(OUT / "concept_scan.json", dict(hits.most_common()))
    return hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "scan"))
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build(), indent=2, default=str))
    else:
        print(json.dumps(dict(concept_scan().most_common(40)), indent=2))
