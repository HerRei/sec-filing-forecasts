from dataclasses import replace

import numpy as np
import pytest

import disclosure_forecasting.simulator as sim
from disclosure_forecasting.simulator import (
    EVAL_FAMILIES,
    World,
    decision_for,
    encode,
    references,
    simulate,
)


def copy(world, **changes):
    fields = dict(
        family=world.family,
        events=list(world.events),
        innovation=dict(world.innovation),
        returns=world.returns.copy(),
        sigma=world.sigma,
        pending=world.pending,
        params=world.params,
    )
    fields.update(changes)
    return World(**fields)


@pytest.mark.parametrize("family", EVAL_FAMILIES)
def test_inputs_ignore_the_future_and_labels_start_at_the_decision(family):
    world = simulate(7, family)
    decision = 150
    original = encode(world, decision)
    changed = copy(world)
    changed.returns[decision:] = 1000
    changed.events = [e for e in world.events if e.available <= decision] + [
        replace(e, value=99.0) for e in world.events if e.available > decision
    ]
    new = encode(changed, decision)
    for key in ("facts", "slots", "ranks", "market"):
        np.testing.assert_array_equal(original[key], new[key])
    assert original["target"][0] == pytest.approx(world.returns[decision] / 0.01)
    assert new["target"][0] == pytest.approx(100000)


def test_immediate_reaction_lands_before_the_record_is_executable(monkeypatch):
    original = sim.process_params

    def fixed(family, rng):
        p = original(family, rng)
        p.update(impact=0.05, immediate=1.0, speed=0.0, base_vol=1e-12)
        return p

    monkeypatch.setattr(sim, "process_params", fixed)
    world = simulate(3, "varied")
    shocks = np.zeros(len(world.returns))
    for event in world.events:
        shocks[event.available - 1] += (
            0.05 * sim.LOADINGS[event.metric] * world.innovation[event.record_id]
        )
    np.testing.assert_allclose(world.returns, shocks, atol=1e-6)


def test_pending_drift_is_all_that_remains_after_the_decision(monkeypatch):
    original = sim.process_params

    def fixed(family, rng):
        p = original(family, rng)
        p.update(impact=0.05, immediate=0.0, speed=0.5, base_vol=1e-12)
        return p

    monkeypatch.setattr(sim, "process_params", fixed)
    world = simulate(5, "regular")
    decision = 120
    future = [e for e in world.events if e.available > decision]
    horizon_end = min(e.available for e in future) - 1  # before the next record arrives
    realised = world.returns[decision:horizon_end].sum()
    expected = world.pending[decision] * (1 - 0.5 ** (horizon_end - decision))
    assert realised == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("family", EVAL_FAMILIES)
def test_version_lineage_is_contiguous_and_time_ordered(family):
    for seed in range(20):
        world = simulate(seed, family)
        by_key = {}
        for event in world.events:
            by_key.setdefault(event.key, []).append(event)
        for versions in by_key.values():
            assert [e.version for e in versions] == list(range(1, len(versions) + 1))
            times = [e.available for e in versions]
            assert times == sorted(times) and len(set(times)) == len(times)
            assert versions[0].available > versions[0].period_end


def test_records_do_not_carry_the_value_they_replace():
    world = simulate(11, "varied")
    revised = next(e for e in world.events if e.version == 2 and e.available < 190)
    decision = 200
    first = next(e for e in world.events if e.key == revised.key and e.version == 1)
    changed = copy(
        world, events=[replace(e, value=e.value + 5) if e == first else e for e in world.events]
    )
    a, b = encode(world, decision), encode(changed, decision)
    row = [
        i
        for i in range(len(a["slots"]))
        if a["facts"][i, 4] > 0 and a["facts"][i, 0] == pytest.approx(revised.value)
    ]
    assert row
    np.testing.assert_array_equal(a["facts"][row], b["facts"][row])


def test_family_definitions_separate_training_and_heldout_ranges():
    regular = [simulate(s, "regular") for s in range(40)]
    assert all(e.version == 1 for w in regular for e in w.events)
    slow = [simulate(s, "h_slow_response").params["speed"] for s in range(40)]
    trained = [simulate(s, "varied").params["speed"] for s in range(40)]
    assert max(slow) < min(trained)
    assert any(e.version == 3 for s in range(40) for e in simulate(s, "h_late_revisions").events)
    assert not any(e.version == 3 for s in range(40) for e in simulate(s, "varied").events)
    null = [simulate(s, "null") for s in range(20)]
    assert all(w.params["impact"] == 0 and not w.pending.any() for w in null)


def test_worlds_and_decisions_are_deterministic():
    a, b = simulate(123, "h_batched"), simulate(123, "h_batched")
    np.testing.assert_array_equal(a.returns, b.returns)
    assert a.events == b.events
    assert decision_for(123) == decision_for(123)
    extras = references(a, decision_for(123))
    assert extras["ref_zero"].shape == (3, 7) and np.all(np.diff(extras["ref_zero"], axis=-1) > 0)
