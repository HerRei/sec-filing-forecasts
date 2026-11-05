"""Chronological splits use label endpoints and keep all snapshots of an event together."""

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ExampleWindow:
    event_id: str
    decision: int
    label_end: int


def purged_split(windows: list[ExampleWindow], train_end: int, validation_end: int):
    if train_end >= validation_end:
        raise ValueError("split boundaries must increase")
    groups = defaultdict(list)
    for index, window in enumerate(windows):
        if window.label_end <= window.decision:
            raise ValueError("label endpoint must follow the decision")
        groups[window.event_id].append((index, window))
    output = {"train": [], "validation": [], "test": [], "purged": []}
    for group in groups.values():
        first = min(w.decision for _, w in group)
        last = max(w.label_end for _, w in group)
        if last < train_end:
            destination = "train"
        elif first >= train_end and last < validation_end:
            destination = "validation"
        elif first >= validation_end:
            destination = "test"
        else:
            destination = "purged"
        output[destination].extend(i for i, _ in group)
    return output
