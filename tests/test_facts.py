from dataclasses import replace

import pytest

from disclosure_forecasting.facts import Fact, FactStore, load_audit, timestamp
from disclosure_forecasting.splits import ExampleWindow, purged_split


def first():
    return Fact(
        "r1",
        "example",
        "inventory",
        100,
        "USD",
        "2026-01-01",
        "2026-03-31",
        "2026-04-10T20:00:00Z",
        "2026-04-13T13:30:00Z",
        1,
        "example://filing",
    )


def test_revision_does_not_rewrite_past_and_replay_is_invariant():
    store = FactStore()
    original = first()
    revised = replace(
        original,
        record_id="r2",
        value=90,
        version=2,
        replaces="r1",
        public_at="2026-05-01T20:00:00Z",
        executable_at="2026-05-04T13:30:00Z",
    )
    assert store.ingest(original)
    assert store.ingest(revised)
    before = store.as_of(timestamp("2026-04-20T00:00:00Z"))
    assert before[original.key].value == 100
    assert not store.ingest(original)
    assert before == store.as_of(timestamp("2026-04-20T00:00:00Z"))
    assert store.as_of(timestamp("2026-05-05T00:00:00Z"))[original.key].value == 90
    assert not store.as_of(timestamp("2026-04-11T00:00:00Z"))


def test_future_economic_guidance_is_usable_when_published():
    guidance = replace(
        first(), metric="guidance", period_start="2026-07-01", period_end="2026-12-31"
    )
    store = FactStore()
    store.ingest(guidance)
    assert guidance.key in store.as_of(timestamp("2026-04-20T00:00:00Z"))


def test_broken_lineage_conflicting_replay_and_naive_timestamps_fail():
    store = FactStore()
    store.ingest(first())
    with pytest.raises(ValueError, match="ID reused"):
        store.ingest(replace(first(), value=102))
    with pytest.raises(ValueError, match="lineage"):
        store.ingest(replace(first(), record_id="r2", version=2, replaces="missing"))
    with pytest.raises(ValueError, match="timezone"):
        replace(first(), public_at="2026-04-10T20:00:00")
    with pytest.raises(ValueError, match="feed receipt"):
        replace(first(), received_at="2026-05-01T12:00:00Z")


def test_entire_event_group_and_crossing_label_are_purged():
    windows = [
        ExampleWindow("a", 5, 10),
        ExampleWindow("b", 15, 22),
        ExampleWindow("c", 21, 25),
        ExampleWindow("d", 35, 40),
        ExampleWindow("b", 22, 24),
    ]
    assert purged_split(windows, 20, 30) == {
        "train": [0],
        "validation": [2],
        "test": [3],
        "purged": [1, 4],
    }


def test_example_audit_is_explicitly_not_a_real_data_gate():
    _, report = load_audit("data/audit/example_facts.csv")
    assert report["revisions"] == 1
    assert "manual_source_audit_still_required" in report["status"]
