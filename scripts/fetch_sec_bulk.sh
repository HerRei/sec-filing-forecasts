#!/bin/sh
# Download the SEC bulk files the real-data build reads into data/raw/sec/bulk/, each with a
# manifest (URL, retrieval time, sha256). Files already there are kept. The SEC asks for a contact
# in the User-Agent. The study used the files retrieved on 2026-10-26; later downloads differ.
set -eu
: "${SEC_USER_AGENT:?set SEC_USER_AGENT, for example \"Your Name you@example.com\"}"
dir=data/raw/sec/bulk
mkdir -p "$dir"
for url in \
  https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip \
  https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip \
  https://www.sec.gov/files/company_tickers_exchange.json; do
  name=$(basename "$url")
  [ -f "$dir/$name" ] && continue
  curl --fail --location --silent --show-error -A "$SEC_USER_AGENT" -o "$dir/$name.part" "$url"
  mv "$dir/$name.part" "$dir/$name"
  printf '{"url": "%s", "retrieved": "%s", "sha256": "%s"}\n' "$url" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(shasum -a 256 "$dir/$name" | cut -d' ' -f1)" \
    > "$dir/$name.manifest.json"
done
