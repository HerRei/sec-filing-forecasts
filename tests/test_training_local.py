import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import disclosure_forecasting.train as train
from disclosure_forecasting.data import build_eval_sets
from disclosure_forecasting.train import execute, plan, read_config

ROOT = Path(__file__).resolve().parents[1]


def tiny(name="tiny", memory="revision", family="varied", steps=4, milestones=(2, 4)):
    config = read_config(ROOT / "configs/stage-s/D-s1.json")
    config.update(
        name=name,
        memory=memory,
        family=family,
        width=16,
        layers=1,
        heads=4,
        batch_size=2,
        steps=steps,
        warmup_steps=1,
        fact_tokens=32,
        market_tokens=96,
        milestones=list(milestones),
        monitor_every=2,
        workers=0,
    )
    return config


@pytest.fixture(scope="module")
def eval_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("eval")
    build_eval_sets(directory, {"monitor": 3, "test": 6}, fact_tokens=32, processes=2)
    return directory


def test_plan_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert plan(tiny())["status"] == "planned_not_started"
    assert list(tmp_path.iterdir()) == []


def test_unknown_device_is_refused_before_anything_is_written(tmp_path):
    with pytest.raises(ValueError, match="use cpu"):
        execute(tiny(), tmp_path / "gpu", device="cuda:1", eval_dir=tmp_path)
    assert not (tmp_path / "gpu").exists()


@pytest.mark.local_training
def test_pause_during_a_milestone_is_finished_on_resume(tmp_path, monkeypatch, eval_dir):
    real = train.predict
    calls = {"n": 0}

    def predict_then_pause(model, data, *args, **kwargs):
        calls["n"] += 1
        if len(data["target"]) == 6 and calls.setdefault("paused", False) is False:
            calls["paused"] = True
            os.kill(os.getpid(), signal.SIGTERM)  # a chat request arrives mid-evaluation
        return real(model, data, *args, **kwargs)

    monkeypatch.setattr(train, "predict", predict_then_pause)
    config = tiny()
    execute(config, tmp_path, eval_dir=eval_dir)
    assert torch.load(tmp_path / "checkpoint.pt", weights_only=True)["step"] == 2
    assert not (tmp_path / "predictions-step2.npz").exists()
    execute(config, tmp_path, resume=True, eval_dir=eval_dir)
    records = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records if r.get("event") == "start"] == [0, 2]
    assert records[-1] == {**records[-1], "event": "done", "step": 4}
    assert (tmp_path / "predictions-step2.npz").exists() and (
        tmp_path / "predictions-step4.npz"
    ).exists()
    with pytest.raises(ValueError, match="checkpoint exists"):
        execute(config, tmp_path, eval_dir=eval_dir)


@pytest.mark.local_training
def test_pipeline_and_report_end_to_end(tmp_path, eval_dir):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    runs = []
    for variant, (family, memory) in {
        "A": ("regular", "sequence"),
        "B": ("varied", "sequence"),
        "C": ("regular", "revision"),
        "D": ("varied", "revision"),
    }.items():
        path = project / f"configs/{variant}-s1.json"
        path.write_text(json.dumps(tiny(f"{variant}-s1", memory, family, steps=3, milestones=(3,))))
        runs.append(str(path))
    plan_path = project / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "name": "e2e",
                "output": str(project / "runs/e2e"),
                "runs": runs,
                "device": "cpu",
                "precision": "fp32",
                "eval_dir": str(eval_dir),
            }
        )
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(
        [sys.executable, "-m", "disclosure_forecasting.pipeline", str(plan_path)],
        check=True,
        env=env,
        timeout=600,
    )
    status = json.loads((project / "runs/e2e/status.json").read_text())
    assert status["phase"] == "training_complete" and len(status["completed"]) == 4
    log = [json.loads(line) for line in (project / "runs/e2e/log.jsonl").read_text().splitlines()]
    assert log[0]["event"] == "start" and log[0]["total_steps"] == 12 and log[-1]["event"] == "done"
    steps = [r["step"] for r in log if "loss" in r]
    assert steps == sorted(steps) and steps[-1] == 12
    subprocess.run(
        [
            sys.executable,
            "-m",
            "disclosure_forecasting.report",
            str(plan_path),
            "--root",
            "/",
            "--out",
            str(project / "results"),
        ],
        check=True,
        env=env,
        timeout=300,
    )
    summary = json.loads((project / "results/summary.json").read_text())
    assert set(summary["verdict"]) >= {"prior_effect", "memory_effect_fixed_simulator", "next_step"}
    assert "Stage S result" in (project / "results/report.md").read_text()
