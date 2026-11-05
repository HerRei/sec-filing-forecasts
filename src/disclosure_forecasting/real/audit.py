"""Gate 1 source audit: a seeded random sample of normalized records checked against the filings.

  sample   draw the stratified sample (seed fixed) -> data/gate1/audit_sample.csv
  fetch    download (cached, throttled, SEC User-Agent) each record's filing index page, primary
           document, XBRL instance (pre-inline filings) and the earnings 8-K press release that
           preceded it -> data/raw/sec/audit/
  check    automated checks + source snippets per row -> data/gate1/audit_sheet.jsonl and .md
           (the sheet is what the row-by-row review reads; verdicts go to audit_review.csv)

Checks per row:
  A acceptance time on the EDGAR index page (Eastern) equals the submissions-JSON time (UTC)
  B the XBRL fact (same concept, same period, no dimensions) has the normalized value
  C the human-readable filing shows the number (formatted in units, thousands or millions)
  D the number already appeared in the press release of an earlier 8-K (Item 2.02): the
    filing time is then an upper bound and the true first release is earlier
  E for revisions: wording near restatement / recast / reclassification / discontinued
"""

import argparse
import html
import json
import re
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

from .common import PROCESSED, RAW, SEC_MIN_INTERVAL_S, SEC_USER_AGENT, ROOT, Throttle, write_json

CACHE = RAW / "sec/audit"
OUT = ROOT / "data/gate1"
SEED = 20261026
STRATA = {  # name -> (query, n)
    "revenue_v1": ("metric == 'revenue' and version == 1 and not derived", 30),
    "inventory_v1": ("metric == 'inventory' and version == 1", 30),
    "backlog_v1": ("metric == 'backlog' and version == 1", 20),
    "revenue_q4_derived_v1": ("metric == 'revenue' and version == 1 and derived", 15),
    "revision_amendment": ("version > 1 and change_type == 'amendment'", 8),
    "revision_later_filing": ("version > 1 and change_type == 'later_filing'", 17),
}
_throttle = Throttle(SEC_MIN_INTERVAL_S)
LABELS = {
    "revenue": r"revenue|net sales|sales",
    "inventory": r"inventor",
    "backlog": r"remaining performance|backlog|performance obligation",
}


def sample():
    records = pd.read_parquet(PROCESSED / "sec/records.parquet")
    records = records[records["accepted_utc"] >= "2010-01-01"]
    parts = []
    for name, (query, n) in STRATA.items():
        pool = records.query(query)
        part = pool.sample(n=min(n, len(pool)), random_state=SEED).copy()
        part["stratum"] = name
        parts.append(part)
    chosen = pd.concat(parts).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(OUT / "audit_sample.csv", index=False)
    return chosen


def get(url):
    """Cached GET with the SEC User-Agent; returns bytes or None (HTTP error recorded)."""
    path = CACHE / url.split("://", 1)[1]
    if path.exists():
        return path.read_bytes()
    _throttle.wait()
    try:
        with urlopen(Request(url, headers={"User-Agent": SEC_USER_AGENT}), timeout=120) as r:
            body = r.read()
    except HTTPError as error:
        body = f"HTTP_ERROR {error.code}".encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return body


def folder(cik, accn):
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn.replace('-', '')}"


def parse_index(body):
    text = body.decode("utf-8", "replace")

    def field(label):
        m = re.search(
            rf'<div class="infoHead">{label}</div>\s*<div class="info">([^<]+)</div>', text
        )
        return m.group(1).strip() if m else None

    docs = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        href = re.search(r'href="([^"]+)"', row)
        if len(cells) >= 4 and href:
            docs.append(
                {
                    "description": re.sub("<[^>]+>", "", cells[1]).strip(),
                    "href": href.group(1),
                    "type": re.sub("<[^>]+>", "", cells[3]).strip(),
                }
            )
    return {
        "accepted_et": field("Accepted"),
        "filing_date": field("Filing Date"),
        "period": field("Period of Report"),
        "docs": docs,
    }


def strip_html(body):
    text = body.decode("utf-8", "replace")
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<ix:header>.*?</ix:header>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text)


def contexts(text):
    out = {}
    for m in re.finditer(
        r'<(?:xbrli:)?context[^>]*\bid="([^"]+)"[^>]*>(.*?)</(?:xbrli:)?context>', text, re.S
    ):
        body = m.group(2)
        instant = re.search(r"<(?:xbrli:)?instant>\s*([\d-]+)", body)
        start = re.search(r"<(?:xbrli:)?startDate>\s*([\d-]+)", body)
        end = re.search(r"<(?:xbrli:)?endDate>\s*([\d-]+)", body)
        out[m.group(1)] = {
            "start": instant.group(1) if instant else (start.group(1) if start else None),
            "end": instant.group(1) if instant else (end.group(1) if end else None),
            "dimensional": bool(re.search(r"segment|scenario", body)),
        }
    return out


def xbrl_facts(body, concept):
    """All values of us-gaap:<concept> in an inline or classic instance: (value, context)."""
    text = body.decode("utf-8", "replace")
    ctx = contexts(text)
    found = []
    for attrs, inner in re.findall(r"<ix:nonFraction([^>]*)>(.*?)</ix:nonFraction>", text, re.S):
        if f'name="us-gaap:{concept}"' not in attrs:
            continue
        raw = re.sub("<[^>]+>", "", inner).strip().replace(",", "")
        if raw in ("", "-", "—"):
            value = 0.0
        else:
            try:
                value = float(raw)
            except ValueError:
                continue
        scale = re.search(r'scale="(-?\d+)"', attrs)
        value *= 10 ** int(scale.group(1)) if scale else 1
        if 'sign="-"' in attrs:
            value = -value
        ref = re.search(r'contextRef="([^"]+)"', attrs).group(1)
        found.append((value, ctx.get(ref, {})))
    for ref, raw in re.findall(
        rf'<us-gaap:{concept}[^>]*contextRef="([^"]+)"[^>]*>([^<]*)</us-gaap:{concept}>', text
    ):
        try:
            found.append((float(raw), ctx.get(ref, {})))
        except ValueError:
            pass
    return found


def formats(value):
    v = abs(value)
    out = {
        "units": f"{round(v):,}",
        "thousands": f"{round(v / 1e3):,}",
        "millions": f"{round(v / 1e6):,}",
        "millions_1dp": f"{v / 1e6:,.1f}",
        "millions_2dp": f"{v / 1e6:,.2f}",
        "billions_1dp": f"{v / 1e9:.1f} billion",
        "billions_2dp": f"{v / 1e9:.2f} billion",
    }
    # very short strings ("12") match everywhere: keep only distinctive ones
    return {k: s for k, s in out.items() if len(s.replace(",", "").replace(".", "")) >= 3}


def find_number(text, value, metric, width=260):
    """Best occurrence of the value: prefer one with the metric's label nearby."""
    best = None
    for name, s in formats(value).items():
        for m in re.finditer(rf"(?<![\d.,]){re.escape(s)}(?![\d]|,\d|\.\d)", text):
            window = text[max(0, m.start() - width) : m.end() + width]
            labelled = bool(re.search(LABELS[metric], window, re.I))
            candidate = (labelled, name, window)
            if best is None or (labelled and not best[0]):
                best = candidate
            if labelled:
                break
        if best and best[0]:
            break
    if best is None:
        return {"found": False}
    return {"found": True, "labelled": best[0], "format": best[1], "snippet": best[2]}


def revision_words(text, width=220, limit=4):
    hits = []
    pattern = r"restat|recast|reclassif|discontinued operation|retrospective|as adjusted|revis|correction of"
    for m in re.finditer(pattern, text, re.I):
        hits.append(text[max(0, m.start() - width) : m.end() + width])
        if len(hits) >= limit:
            break
    return hits


def earlier_press_release(row, filings):
    mine = filings[(filings["cik"] == row["cik"]) & (filings["form"] == "8-K")]
    mine = mine[mine["items"].fillna("").str.contains("2.02")]
    before = mine[
        (mine["accepted_utc"] < row["accepted_utc"])
        & (
            pd.to_datetime(mine["accepted_utc"], utc=True)
            >= pd.to_datetime(row["accepted_utc"], utc=True) - pd.Timedelta(days=75)
        )
    ]
    if before.empty:
        return None
    return before.sort_values("accepted_utc").iloc[-1]


def fetch_row(row, filings):
    base = folder(row["cik"], row["accn"])
    out = {}
    index = get(f"{base}/{row['accn']}-index.htm")
    out["index"] = parse_index(index)
    primary = filings.loc[filings["accn"] == row["accn"], "primary_doc"]
    primary = primary.iloc[0] if len(primary) else None
    out["primary_doc"] = primary
    if primary:
        get(f"{base}/{primary}")
    instance = [
        d
        for d in out["index"]["docs"]
        if d["type"] in ("EX-101.INS", "XML") and d["href"].endswith(".xml")
    ]
    out["instance"] = instance[0]["href"] if instance else None
    if out["instance"]:
        get("https://www.sec.gov" + out["instance"])
    release = earlier_press_release(row, filings)
    out["press"] = None
    if release is not None:
        rbase = folder(row["cik"], release["accn"])
        rindex = parse_index(get(f"{rbase}/{release['accn']}-index.htm"))
        exhibits = [d for d in rindex["docs"] if d["type"].upper().startswith("EX-99")]
        out["press"] = {
            "accn": release["accn"],
            "accepted_utc": release["accepted_utc"],
            "items": release["items"],
            "exhibits": [d["href"] for d in exhibits[:2]],
        }
        for href in out["press"]["exhibits"]:
            get("https://www.sec.gov" + href)
    return out


def fetch():
    chosen = pd.read_csv(OUT / "audit_sample.csv")
    filings = pd.read_parquet(PROCESSED / "sec/filings.parquet")
    meta = {}
    for _, row in chosen.iterrows():
        meta[row["record_id"]] = fetch_row(row, filings)
    write_json(OUT / "audit_fetch.json", meta)
    return len(meta)


def _utc_from_et(stamp):
    if not stamp:
        return None
    return (
        pd.Timestamp(stamp)
        .tz_localize("America/New_York")
        .tz_convert("UTC")
        .strftime("%Y-%m-%dT%H:%M:%S")
    )


def check():
    chosen = pd.read_csv(OUT / "audit_sample.csv")
    meta = json.loads((OUT / "audit_fetch.json").read_text())
    sheet = []
    for _, row in chosen.iterrows():
        m = meta[row["record_id"]]
        base = folder(row["cik"], row["accn"])
        entry = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in row.items()
            if k
            in (
                "record_id",
                "stratum",
                "cik",
                "metric",
                "concept",
                "value",
                "period_start",
                "period_end",
                "version",
                "form",
                "change_type",
                "accepted_utc",
                "executable_at",
                "derived",
                "accn",
            )
        }
        idx = m["index"]
        entry["index_accepted_et"] = idx["accepted_et"]
        entry["index_accepted_utc"] = _utc_from_et(idx["accepted_et"])
        entry["A_acceptance_match"] = entry["index_accepted_utc"] == row["accepted_utc"][:19]
        entry["index_period"] = idx["period"]
        entry["filing_url"] = f"{base}/{row['accn']}-index.htm"
        docs = []
        if m["primary_doc"]:
            docs.append(get(f"{base}/{m['primary_doc']}"))
        if m["instance"]:
            docs.append(get("https://www.sec.gov" + m["instance"]))
        facts = []
        for body in docs:
            facts += xbrl_facts(body, row["concept"])
        start = row["period_start"] if row["metric"] == "revenue" else row["period_end"]
        same_period = [
            v for v, c in facts if c.get("end") == row["period_end"] and c.get("start") == start
        ]
        plain = [
            v
            for v, c in facts
            if c.get("end") == row["period_end"]
            and c.get("start") == start
            and not c.get("dimensional")
        ]
        tolerance = max(abs(row["value"]) * 1e-3, 1.0)
        entry["B_xbrl_values_same_period_no_dims"] = sorted(set(plain))[:6]
        entry["B_xbrl_match"] = any(abs(v - row["value"]) <= tolerance for v in plain)
        entry["B_xbrl_any_concept_facts"] = len(facts)
        entry["B_dimensional_values_same_period"] = sorted(set(same_period) - set(plain))[:4]
        text = strip_html(docs[0]) if docs and m["primary_doc"] else ""
        entry["C_text"] = (
            find_number(text, row["value"], row["metric"]) if text else {"found": False}
        )
        if row["version"] > 1:
            entry["E_revision_words"] = revision_words(text)
        press = m.get("press")
        entry["D_press_release"] = None
        if press:
            ptexts = [strip_html(get("https://www.sec.gov" + h)) for h in press["exhibits"]]
            hit = {"found": False}
            for t in ptexts:
                hit = find_number(t, row["value"], row["metric"])
                if hit["found"]:
                    break
            lead = pd.to_datetime(row["accepted_utc"], utc=True) - pd.to_datetime(
                press["accepted_utc"], utc=True
            )
            entry["D_press_release"] = {
                "accn": press["accn"],
                "accepted_utc": press["accepted_utc"],
                "items": press["items"],
                "lead_days": round(lead.total_seconds() / 86400, 2),
                **hit,
            }
        sheet.append(entry)
    with (OUT / "audit_sheet.jsonl").open("w") as handle:
        for entry in sheet:
            handle.write(json.dumps(entry, default=str) + "\n")
    lines = ["# Gate 1 audit sheet (generated; verdicts in audit_review.csv)\n"]
    for i, e in enumerate(sheet, 1):
        c, d = e["C_text"], e["D_press_release"]
        lines.append(
            f"## {i}. {e['stratum']} | CIK {e['cik']} | {e['metric']} ({e['concept']}) | "
            f"{e['period_start']}..{e['period_end']} | v{e['version']} {e['form']} | value {e['value']:,.0f}\n"
        )
        lines.append(
            f"- A accepted index {e['index_accepted_et']} ET = {e['index_accepted_utc']}Z vs json "
            f"{e['accepted_utc']} -> {e['A_acceptance_match']}; period {e['index_period']}; {e['filing_url']}"
        )
        lines.append(
            f"- B xbrl match {e['B_xbrl_match']} (no-dim values {e['B_xbrl_values_same_period_no_dims']}, "
            f"dim values {e['B_dimensional_values_same_period']}, facts {e['B_xbrl_any_concept_facts']})"
        )
        lines.append(
            f"- C text found {c.get('found')} labelled {c.get('labelled')} as {c.get('format')}: "
            f"…{(c.get('snippet') or '')[:520]}…"
        )
        if d:
            lines.append(
                f"- D press {d['accn']} {d['accepted_utc']} items {d['items']} lead {d['lead_days']} d: "
                f"found {d.get('found')} labelled {d.get('labelled')} as {d.get('format')}: …{(d.get('snippet') or '')[:420]}…"
            )
        else:
            lines.append("- D no Item 2.02 8-K in the 75 days before")
        for w in e.get("E_revision_words", []) or []:
            lines.append(f"- E …{w}…")
        lines.append("")
    (OUT / "audit_sheet.md").write_text("\n".join(lines))
    return len(sheet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("sample", "fetch", "check"))
    args = parser.parse_args()
    result = {"sample": lambda: len(sample()), "fetch": fetch, "check": check}[args.command]()
    print(json.dumps({"rows": result}))
