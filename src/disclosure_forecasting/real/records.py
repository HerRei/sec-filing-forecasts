"""Point-in-time, versioned disclosure records from raw companyfacts rows.

Rules (fixed before any outcome was computed):
- A key is (CIK, metric, period start, period end). Instants (inventory, backlog) have start == end.
- The concept of a key is fixed by the first filing that reports it (priority order in CONCEPTS
  within that filing). Later filings that report the key only under another concept are counted,
  not used: a concept switch is not evidence of a correction.
- Filings that report a key are ordered by acceptance time. The first is version 1. A later filing
  repeating the latest value (within 0.1 %) is a citation (counted, no new record). A different
  value is a new version that `replaces` the previous one; change_type is "amendment" for /A forms
  and "later_filing" otherwise (restatement, recast or reclassification: the reason is not in the
  data and is only established for audited rows).
- Two different values for one key inside one filing are a conflict: the key is dropped.
- Quarterly revenue for fiscal Q4 is rarely tagged directly; where a 10-K reports the fiscal year
  and the 9-month year-to-date value is known at that time, Q4 = FY - 9M is added as a derived
  record (derived = True). Direct three-month facts win inside the same filing.

Changes after the Gate 1 audit (data/audit/audit_rows.csv), before any outcome was computed:
- Only periodic reports (10-Q, 10-K, transition reports and their amendments) feed records. An
  8-K's XBRL can be a pro forma balance sheet (audit row 111: Tesco's pro forma inventory appeared
  as a new version of its reported inventory).
- No Q4 is derived from a 10-K that recasts the prior fiscal year (its FY may then be on a new
  basis while the stored 9M is not: audit row 90, Q4 off by 43 %).
- Acceptance times come from data/processed/sec/acceptance.parquet (XBRL RSS archive or index
  page, Eastern time converted to UTC); the submissions-JSON field is wrong for most filings.
"""

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

from .common import CONCEPTS, INSTANT, PROCESSED, ROUNDING_TOLERANCE, write_json
from .sessions import executable_from_acceptance, executable_from_filing_date

SEC = PROCESSED / "sec"
QUARTER = (80, 100)
NINE_MONTHS = (260, 290)
YEAR = (350, 380)


def _days(start, end):
    return (pd.to_datetime(end) - pd.to_datetime(start)).dt.days + 1


PERIODIC = ("10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT", "10-QT", "10-KT/A", "10-QT/A")


def true_acceptance(filings):
    """Replace the submissions-JSON acceptance time by the authoritative one where known."""
    path = SEC / "acceptance.parquet"
    filings = filings.copy()
    filings["json_accepted"] = filings["accepted_utc"]
    if not path.exists():
        raise SystemExit("acceptance.parquet missing: run timestamps build first")
    true = pd.read_parquet(path)[["accn", "accepted_utc_true", "source"]]
    filings = filings.merge(true, on="accn", how="left")
    filings["accepted_utc"] = filings["accepted_utc_true"].fillna("")  # unknown -> date fallback
    filings["time_source"] = filings["source"].fillna("filing_date_fallback")
    return filings.drop(columns=["accepted_utc_true", "source"])


def load_raw():
    raw = pd.read_parquet(SEC / "raw_facts.parquet")
    filings = true_acceptance(pd.read_parquet(SEC / "filings.parquet"))
    raw = raw[(raw["unit"] == "USD") & raw["metric"].isin(CONCEPTS)].copy()
    raw = raw.merge(
        filings[
            [
                "accn",
                "form",
                "accepted_utc",
                "filing_date",
                "report_date",
                "time_source",
                "json_accepted",
            ]
        ].rename(columns={"form": "filing_form"}),
        on="accn",
        how="left",
    )
    raw["start"] = np.where(raw["metric"].isin(INSTANT), raw["end"], raw["start"])
    raw["duration"] = _days(raw["start"], raw["end"])
    priority = {c: i for concepts in CONCEPTS.values() for i, c in enumerate(concepts)}
    raw["priority"] = raw["concept"].map(priority)
    raw["form"] = raw["filing_form"].fillna(raw["form"])
    raw = raw[raw["form"].isin(PERIODIC)]
    # Sort key for "known by then": acceptance time, else the end of the filing date (22:00 ET
    # is EDGAR's last acceptance; 23:59:59Z is later than that in both EST and EDT).
    raw["order_time"] = raw["accepted_utc"].where(
        raw["accepted_utc"].fillna("") != "",
        raw["filing_date"].fillna(raw["filed"]) + "T23:59:59.999Z",
    )
    raw["order_time"] = raw["order_time"].where(
        raw["order_time"].str.endswith("Z"), raw["order_time"]
    )
    return raw


def _timing(frame):
    has = frame["accepted_utc"].notna() & (frame["accepted_utc"] != "")
    out = pd.DataFrame(index=frame.index, columns=["exec_session", "executable_at"])
    if has.any():
        out.loc[has] = executable_from_acceptance(frame.loc[has, "accepted_utc"]).values
    if (~has).any():
        out.loc[~has] = executable_from_filing_date(
            frame.loc[~has, "filing_date"].fillna(frame.loc[~has, "filed"])
        ).values
    frame = frame.copy()
    frame["exec_session"] = pd.to_datetime(out["exec_session"])
    frame["executable_at"] = pd.to_datetime(out["executable_at"])
    frame["timestamp_basis"] = np.where(
        has, frame.get("time_source", "acceptance"), "filing_date_fallback"
    )
    frame["public_at"] = np.where(
        has,
        frame["accepted_utc"],
        pd.to_datetime(frame["filing_date"].fillna(frame["filed"])).dt.strftime(
            "%Y-%m-%dT00:00:00Z"
        ),
    )
    return frame


def derive_q4(raw, stats):
    """Q4 revenue = FY - 9M YTD, computed at every filing that reports the fiscal year."""
    revenue = raw[raw["metric"] == "revenue"]
    annual = revenue[revenue["duration"].between(*YEAR)]
    # Only the annual report for that fiscal year (and its amendments) derives Q4; later
    # comparatives may be recast while the stored 9M is not, which would fake a revision.
    annual = annual[annual["end"] == annual["report_date"]]
    nine = revenue[revenue["duration"].between(*NINE_MONTHS)]
    every_year = revenue[revenue["duration"].between(*YEAR)].sort_values(["order_time", "accn"])
    first_fy = every_year.groupby(["cik", "concept", "end"])["val"].first()
    recasting = set()
    for (cik, accn, concept), group in every_year.groupby(["cik", "accn", "concept"]):
        for row in group.itertuples():
            original = first_fy.get((cik, concept, row.end))
            if original is not None and abs(row.val - original) > ROUNDING_TOLERANCE * max(
                abs(original), 1.0
            ):
                recasting.add(
                    accn
                )  # this filing reports a fiscal year differently than first published
    rows = []
    nine_by_key = {
        k: g.sort_values(["order_time", "accn"])
        for k, g in nine.groupby(["cik", "start", "concept"])
    }
    for (cik, start, end, accn), group in annual.groupby(["cik", "start", "end", "accn"]):
        fy = group.sort_values("priority").iloc[0]
        if accn in recasting:
            stats["q4_skipped_recasting_10k"] += 1
            continue
        candidates = nine_by_key.get((cik, start, fy["concept"]))
        if candidates is None:
            stats["q4_no_9m"] += 1
            continue
        known = candidates[candidates["order_time"] <= fy["order_time"]]  # same filing allowed
        if known.empty:
            stats["q4_no_9m"] += 1
            continue
        latest = known.iloc[-1]
        q4_start = (pd.Timestamp(latest["end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        length = (pd.Timestamp(end) - pd.Timestamp(q4_start)).days + 1
        if not QUARTER[0] <= length <= QUARTER[1]:
            stats["q4_bad_length"] += 1
            continue
        row = fy.copy()
        row["start"], row["val"], row["duration"] = q4_start, fy["val"] - latest["val"], length
        row["derived"] = True
        row["priority"] = fy["priority"] + 0.5  # direct three-month facts win in one filing
        rows.append(row)
    stats["q4_derived_rows"] = len(rows)
    return pd.DataFrame(rows)


def _chains(rows, stats, conflicts):
    """Pure-Python pass over rows sorted by key and time: versions, citations, conflicts."""
    rows = rows.sort_values(["cik", "metric", "start", "end", "order_time", "accn", "priority"])
    columns = list(rows.columns)
    index = {c: i for i, c in enumerate(columns)}
    out = []
    current_key, chain, concept, conflict = None, [], None, False

    def flush():
        if current_key is None:
            return
        if conflict:
            conflicts[current_key[1]] += 1
            return
        out.extend(chain)

    by_key_filing = {}
    for values in rows.itertuples(index=False, name=None):
        key = tuple(values[index[c]] for c in ("cik", "metric", "start", "end"))
        if key != current_key:
            flush()
            current_key, chain, concept, conflict = key, [], values[index["concept"]], False
            by_key_filing = {}
            stats["keys"] += 1
            if values[index["derived"]]:
                stats["derived_keys"] += 1
        accn = values[index["accn"]]
        if values[index["concept"]] != concept:
            stats["rows_under_other_concepts_ignored"] += 1
            continue
        priority = values[index["priority"]]
        value = float(values[index["val"]])
        seen = by_key_filing.get(accn)
        if seen is not None:
            if priority > seen[0]:
                continue  # a lower-priority (derived) value inside the same filing
            if abs(value - seen[1]) > ROUNDING_TOLERANCE * max(abs(seen[1]), 1.0):
                conflict = True
            continue
        by_key_filing[accn] = (priority, value)
        latest = chain[-1]["value"] if chain else None
        if latest is not None and abs(value - latest) <= ROUNDING_TOLERANCE * max(abs(latest), 1.0):
            chain[-1]["citations"] += 1
            continue
        version = len(chain) + 1
        cik, metric, start, end = key
        record_id = f"{cik}-{metric}-{start}-{end}-v{version}-{accn}"
        form = values[index["form"]] or ""
        chain.append(
            {
                "record_id": record_id,
                "cik": cik,
                "metric": metric,
                "concept": concept,
                "value": value,
                "unit": "USD",
                "period_start": start,
                "period_end": end,
                "duration": int(values[index["duration"]]),
                "accn": accn,
                "form": form,
                "accepted_utc": values[index["accepted_utc"]] or "",
                "filing_date": values[index["filing_date"]] or values[index["filed"]],
                "filed": values[index["filed"]],
                "version": version,
                "replaces": chain[-1]["record_id"] if chain else "",
                "change_type": ""
                if version == 1
                else ("amendment" if form.endswith("/A") else "later_filing"),
                "derived": bool(values[index["derived"]]),
                "citations": 0,
                "time_source": values[index["time_source"]],
                "json_accepted": values[index["json_accepted"]] or "",
            }
        )
    flush()
    return pd.DataFrame(out)


def build():
    stats = Counter()
    raw = load_raw()
    raw["derived"] = False
    stats["usd_rows"] = len(raw)
    keep_flow = raw["metric"].isin(INSTANT) | raw["duration"].between(*QUARTER)
    q4 = derive_q4(raw, stats)
    rows = pd.concat([raw[keep_flow], q4], ignore_index=True)
    stats["missing_acceptance_rows"] = int((rows["accepted_utc"].fillna("") == "").sum())
    conflicts = Counter()
    records = _chains(rows, stats, conflicts)
    records = _timing(records)
    stats["conflict_keys"] = dict(conflicts)
    stats["records"] = len(records)
    stats["records_by_metric"] = records["metric"].value_counts().to_dict()
    stats["revisions_by_metric"] = (
        records[records["version"] > 1].groupby("metric").size().to_dict()
    )
    stats["revisions_by_type"] = (
        records[records["version"] > 1]["change_type"].value_counts().to_dict()
    )
    stats["timestamp_basis"] = records["timestamp_basis"].value_counts().to_dict()
    stats["citations"] = int(records["citations"].sum())
    records.to_parquet(SEC / "records.parquet", index=False)
    write_json(SEC / "records.json", dict(stats))
    return stats


def validate():
    """Run every record through the project's FactStore (immutable lineage checks)."""
    from ..facts import Fact, FactStore

    records = pd.read_parquet(SEC / "records.parquet")
    store = FactStore()
    records = records.sort_values(["executable_at", "version"])
    errors = Counter()
    for row in records.itertuples():
        fact = Fact(
            record_id=row.record_id,
            entity=str(row.cik),
            metric=row.metric,
            value=float(row.value),
            unit=row.unit,
            period_start=row.period_start,
            period_end=row.period_end,
            public_at=pd.Timestamp(row.public_at).strftime("%Y-%m-%dT%H:%M:%SZ"),
            executable_at=pd.Timestamp(row.executable_at).strftime("%Y-%m-%dT%H:%M:%SZ"),
            version=int(row.version),
            source=f"sec:{row.accn}",
            replaces=row.replaces,
        )
        try:
            store.ingest(fact)
        except ValueError as error:
            errors[str(error).split(" for ")[0]] += 1
    result = {"ingested": len(store.records), "errors": dict(errors)}
    write_json(SEC / "lineage_validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    args = parser.parse_args()
    out = build() if args.command == "build" else validate()
    print(json.dumps(out, indent=2, default=str))
