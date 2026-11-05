import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from disclosure_forecasting.data import build_eval_sets, eval_seed, load_eval
from disclosure_forecasting.model import ForecastModel, model_inputs
from disclosure_forecasting.simulator import (
    EVAL_FAMILIES,
    HELDOUT_FAMILIES,
    S2_EVAL_FAMILIES,
    S2_NEW_HELDOUT,
    decision_for,
    encode,
    references,
    simulate,
)
from disclosure_forecasting.train import read_config

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data/synthetic/stage-s-v2"
torch.set_num_threads(1)


def batch(seeds, family="h_late_revisions", decision=200):
    rows = [encode(simulate(s, family), decision) for s in seeds]
    return {k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in rows[0]}


@pytest.mark.skipif(
    not FROZEN.exists(), reason="frozen Stage S evaluation set not built (scripts/make_stage_s.py)"
)
@pytest.mark.parametrize("family", EVAL_FAMILIES)
def test_stage_s_worlds_are_unchanged(family):
    frozen = load_eval(FROZEN, "test", family)
    for index in range(12):
        seed = eval_seed(family, "test", index)
        world = simulate(seed, family)
        row = encode(world, decision_for(seed))
        row.update(references(world, decision_for(seed)))
        for key in ("facts", "slots", "ranks", "market", "target", "ref_zero"):
            np.testing.assert_array_equal(frozen[key][index], row[key])


@pytest.mark.parametrize("memory", ["sequence", "revision"])
def test_full_budget_is_stage_s_behaviour(memory):
    torch.manual_seed(1)
    plain = ForecastModel(width=32, layers=1, heads=4, memory=memory).eval()
    budgeted = ForecastModel(width=32, layers=1, heads=4, memory=memory, fact_budget=128).eval()
    budgeted.load_state_dict(plain.state_dict())
    data = batch([1, 2, 3])
    with torch.no_grad():
        torch.testing.assert_close(plain(**model_inputs(data)), budgeted(**model_inputs(data)))


def test_sequence_budget_keeps_the_newest_records():
    torch.manual_seed(2)
    model = ForecastModel(width=32, layers=1, heads=4, fact_budget=16).eval()
    reference = ForecastModel(width=32, layers=1, heads=4).eval()
    reference.load_state_dict(model.state_dict())
    seeds, decision = [4, 5, 6], 200
    full = batch(seeds, decision=decision)
    short = [encode(simulate(s, "h_late_revisions"), decision, fact_tokens=16) for s in seeds]
    short = {k: torch.from_numpy(np.stack([r[k] for r in short])) for k in short[0]}
    with torch.no_grad():
        # Same newest 16 records -> same prediction (slot ids differ but sequence ignores them).
        torch.testing.assert_close(model(**model_inputs(full)), reference(**model_inputs(short)))


def test_revision_budget_keeps_the_most_recently_updated_slots():
    torch.manual_seed(3)
    model = ForecastModel(width=32, layers=1, heads=4, memory="revision", fact_budget=8).eval()
    data = batch([7, 8])
    with torch.no_grad():
        embeddings = model.fact_encoder(data["facts"])
        states, padding = model.fact_memory(embeddings, data["slots"], data["ranks"])
        tokens, kept = model.budget(states, padding, data["slots"])
    for row in range(2):
        slots = data["slots"][row].tolist()
        last = {}
        for position, slot in enumerate(slots):
            if slot >= 0:
                last[slot] = position
        newest = sorted(last, key=last.get, reverse=True)[:8]
        expected = torch.stack([states[row, s] for s in newest])
        torch.testing.assert_close(tokens[row], expected)
        assert (~kept[row]).sum() == min(8, len(last))


def test_new_heldout_families_leave_the_broad_ranges():
    broad = [simulate(s, "broad").params for s in range(200)]
    assert max(p["immediate"] for p in broad) <= 1.3
    assert (
        max(p["batch_every"] for p in broad) <= 16
        and max(p["revision_lag"][1] for p in broad) <= 40
    )
    over = [simulate(s, "h2_overreaction").params["immediate"] for s in range(50)]
    assert min(over) >= 1.4
    assert all(e.metric != 0 for s in range(30) for e in simulate(s, "h2_metric_dropout").events)
    stale = simulate(1, "h2_stale")
    assert min(e.available - e.period_end for e in stale.events if e.version == 1) >= 15
    burst = simulate(2, "h2_burst")
    assert all(e.available % 24 == 0 for e in burst.events)
    restated = [
        e for s in range(20) for e in simulate(s, "h2_restatements").events if e.version == 2
    ]
    first = {
        (s, e.key): e.available
        for s in range(20)
        for e in simulate(s, "h2_restatements").events
        if e.version == 1
    }
    assert restated and all(
        e.available - first[(s, e.key)] >= 50
        for s in range(20)
        for e in simulate(s, "h2_restatements").events
        if e.version == 2
    )


def test_budget_config_is_optional():
    config = read_config(ROOT / "configs/stage-s2/D-k16-s1.json")
    assert config["fact_budget"] == 16
    assert "fact_budget" not in read_config(ROOT / "configs/stage-s/D-s1.json")


@pytest.mark.local_training
def test_stage_s2_pipeline_and_report(tmp_path):
    eval_dir = tmp_path / "eval"
    build_eval_sets(
        eval_dir,
        {"monitor": 2, "test": 6},
        fact_tokens=32,
        processes=2,
        families=S2_EVAL_FAMILIES,
        heldout=HELDOUT_FAMILIES + S2_NEW_HELDOUT,
    )
    base = read_config(ROOT / "configs/stage-s2/B-k16-s1.json")
    runs = []
    for name, family, memory, budget in (
        ("B-k16", "varied", "sequence", 4),
        ("D-k16", "varied", "revision", 4),
        ("B300", "varied", "sequence", None),
        ("E300", "broad", "sequence", None),
    ):
        config = {
            **base,
            "name": f"{name}-s1",
            "family": family,
            "memory": memory,
            "width": 16,
            "layers": 1,
            "heads": 4,
            "batch_size": 2,
            "steps": 3,
            "warmup_steps": 1,
            "fact_tokens": 32,
            "milestones": [3],
            "monitor_every": 3,
            "workers": 0,
        }
        config.pop("fact_budget", None)
        if budget:
            config["fact_budget"] = budget
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(config))
        runs.append(str(path))
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "name": "e2e-s2",
                "output": str(tmp_path / "runs"),
                "runs": runs,
                "device": "cpu",
                "precision": "fp32",
                "eval_dir": str(eval_dir),
                "references": [],
            }
        )
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(
        [sys.executable, "-m", "disclosure_forecasting.pipeline", str(plan)],
        check=True,
        env=env,
        timeout=600,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "disclosure_forecasting.report_s2",
            str(plan),
            "--root",
            "/",
            "--out",
            str(tmp_path / "out"),
        ],
        check=True,
        env=env,
        timeout=300,
    )
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["verdict"]["Q1_memory_under_budget"] in (
        "supported",
        "negligible",
        "not supported",
        "harmful",
    )
    assert "Stage S2 result" in (tmp_path / "out/report.md").read_text()
