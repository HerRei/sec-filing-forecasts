#!/usr/bin/env python3
"""Optional manual raw-source retrieval. No automatic feature or label construction."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("sec-submissions", "sec-facts", "alfred"))
    parser.add_argument("identifier", help="numeric CIK, or FRED series ID")
    parser.add_argument("--output", required=True, help="new raw JSON path; never overwrites")
    args = parser.parse_args()
    headers = {"Accept": "application/json"}
    if args.source.startswith("sec"):
        agent = os.environ.get("SEC_USER_AGENT", "")
        if "@" not in agent:
            parser.error("set SEC_USER_AGENT to your organization/name and contact email")
        headers["User-Agent"] = agent
        if not args.identifier.isdigit() or len(args.identifier) > 10:
            parser.error("CIK must contain at most 10 digits")
        cik = args.identifier.zfill(10)
        route = (
            f"submissions/CIK{cik}.json"
            if args.source == "sec-submissions"
            else f"api/xbrl/companyfacts/CIK{cik}.json"
        )
        base = "https://data.sec.gov/" + route
        url = base
    else:
        key = os.environ.get("FRED_API_KEY")
        if not key:
            parser.error("set FRED_API_KEY before retrieving ALFRED vintages")
        base = "https://api.stlouisfed.org/fred/series/observations"
        url = (
            base
            + "?"
            + urlencode(
                {
                    "series_id": args.identifier,
                    "api_key": key,
                    "file_type": "json",
                    "realtime_start": "1776-07-04",
                    "realtime_end": "9999-12-31",
                    "output_type": 2,
                    "limit": 100000,
                }
            )
        )
    destination = Path(args.output)
    if destination.exists() or destination.with_suffix(".manifest.json").exists():
        parser.error("output or manifest already exists; choose a new path")
    try:
        with urlopen(Request(url, headers=headers), timeout=60) as response:
            raw = response.read()
    except HTTPError as error:
        parser.exit(1, f"source returned HTTP {error.code}; no URL or credentials logged\n")
    except URLError:
        parser.exit(1, "source request failed; no URL or credentials logged\n")
    parsed = json.loads(raw)
    if args.source == "alfred" and ("error_code" in parsed or "observations" not in parsed):
        parser.exit(1, "FRED returned an API error; response not saved\n")
    if args.source == "alfred" and int(parsed.get("count", 0)) > len(parsed["observations"]):
        parser.exit(1, "response needs pagination; no partial dataset saved\n")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(raw)
    manifest = {
        "source": args.source,
        "identifier": args.identifier,
        "endpoint": base,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "status": "raw_source_only_not_audited_or_training_ready",
    }
    destination.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
