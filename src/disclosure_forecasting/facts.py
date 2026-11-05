"""Immutable version lineage and conservative point-in-time fact access."""

import csv
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"timestamp must include timezone: {value!r}")
    return result


@dataclass(frozen=True)
class Fact:
    record_id: str
    entity: str
    metric: str
    value: float
    unit: str
    period_start: str
    period_end: str
    public_at: str
    executable_at: str
    version: int
    source: str
    replaces: str = ""
    received_at: str = ""

    def __post_init__(self):
        if not all((self.record_id, self.entity, self.metric, self.unit, self.source)):
            raise ValueError("record, entity, metric, unit and source are required")
        if not math.isfinite(self.value) or self.version < 1:
            raise ValueError("finite value and positive version required")
        if date.fromisoformat(self.period_start) > date.fromisoformat(self.period_end):
            raise ValueError("economic period is reversed")
        public = timestamp(self.public_at)
        executable = timestamp(self.executable_at)
        if executable < public:
            raise ValueError("execution time precedes public availability")
        if self.received_at and executable < timestamp(self.received_at):
            raise ValueError("execution time precedes feed receipt")

    @property
    def key(self):
        return self.entity, self.metric, self.period_start, self.period_end, self.unit

    @property
    def available_at(self):
        return timestamp(self.executable_at)


class FactStore:
    def __init__(self):
        self.records: dict[str, Fact] = {}
        self._latest: dict[tuple, Fact] = {}

    def ingest(self, fact: Fact) -> bool:
        """Return false on an exact replay; reject ID collisions and broken lineage."""
        if fact.record_id in self.records:
            if self.records[fact.record_id] != fact:
                raise ValueError(f"record ID reused with different content: {fact.record_id}")
            return False
        previous = self._latest.get(fact.key)
        if previous is None:
            if fact.version != 1 or fact.replaces:
                raise ValueError("lineage must begin with version 1")
        elif (
            fact.version != previous.version + 1
            or fact.replaces != previous.record_id
            or fact.available_at < previous.available_at
        ):
            raise ValueError(f"broken revision lineage for {fact.record_id}")
        self.records[fact.record_id] = fact
        self._latest[fact.key] = fact
        return True

    def as_of(self, decision: datetime) -> dict[tuple, Fact]:
        if decision.tzinfo is None or decision.utcoffset() is None:
            raise ValueError("decision time must include timezone")
        latest = {}
        for fact in sorted(self.records.values(), key=lambda f: (f.available_at, f.version)):
            if fact.available_at <= decision:
                latest[fact.key] = fact
        return latest


def load_audit(path: str | Path) -> tuple[FactStore, dict]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty audit dataset")
    facts = [
        Fact(**{**row, "value": float(row["value"]), "version": int(row["version"])})
        for row in rows
    ]
    store = FactStore()
    replays = 0
    for fact in sorted(facts, key=lambda f: (f.available_at, f.version)):
        replays += not store.ingest(fact)
    return store, {
        "rows": len(rows),
        "unique_records": len(store.records),
        "exact_replays": replays,
        "revisions": sum(f.version > 1 for f in store.records.values()),
        "entities": len({f.entity for f in store.records.values()}),
        "status": "structural_checks_passed_manual_source_audit_still_required",
    }
