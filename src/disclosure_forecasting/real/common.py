"""Shared constants for the real-data stages. Every choice here is fixed in
docs/PREREGISTRATION_G4.md; changing one after the freeze makes a new, unregistered study."""

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"

# SEC fair access: a real contact in the User-Agent, well under 10 requests/s.
# Set it in the environment: a name and a contact email, as the SEC asks.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "")
SEC_MIN_INTERVAL_S = 0.25  # at most 4 requests per second

# Universe: US industrial manufacturers, SIC 3400-3899, keeping the groups where orders,
# backlog and inventory are core operating metrics and an industrial benchmark (XLI) fits.
SIC_KEEP = (3400, 3899)
SIC_DROP = (
    (3570, 3579),  # computer and office equipment (tech hardware, not industrial)
    (3660, 3679),  # communications equipment, semiconductors and components
    (3830, 3899),  # optical, medical, dental, ophthalmic, photographic, watches, misc.
)
FAMILIES = ("34", "35", "36", "37", "38")  # SIC major groups: the balance units of G4
US_STATES = set(
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ "
    "NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()
)

# Concepts per metric, in priority order. companyfacts holds only standard-taxonomy facts
# without dimensions; company extension elements (where most "orders" figures live) are absent.
CONCEPTS = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueGoodsNet",
    ),
    "inventory": (
        "InventoryNet",
        "InventoryNetOfAllowancesCustomerAdvancesAndProgressBillings",
        "InventoryGross",
    ),
    "backlog": (
        "RevenueRemainingPerformanceObligation",
    ),  # ASC 606 remaining performance obligations
}
INSTANT = {"inventory", "backlog"}
ROUNDING_TOLERANCE = (
    0.001  # relative change below 0.1 % = same value (rounding/scale), not a revision
)


def sic_in_universe(sic) -> bool:
    try:
        sic = int(sic)
    except (TypeError, ValueError):
        return False
    if not SIC_KEEP[0] <= sic <= SIC_KEEP[1]:
        return False
    return not any(lo <= sic <= hi for lo, hi in SIC_DROP)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


class Throttle:
    """At most one call per `interval` seconds across all threads (start times are spaced)."""

    def __init__(self, interval):
        import threading

        self.interval, self.next, self.lock = interval, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next)
            self.next = start + self.interval
        if start > now:
            time.sleep(start - now)
