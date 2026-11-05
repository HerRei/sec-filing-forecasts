"""Resumable Stage S pipeline. A process supervisor owns start/stop; SIGTERM checkpoints.

Writes runs/<name>/log.jsonl (start header, then step/tokens/loss/tok_s records with GLOBAL
step numbers, then {"event": "done"}) and a compact runs/<name>/status.json for monitoring.
"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    plan = json.loads(Path(args.plan).read_text())
    output = root / plan["output"]
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_file = output / "status.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    # Crash-loop evidence for a supervisor that restarts a dead run.
    status["start_times"] = (status.get("start_times", []) + [time.time()])[-20:]
    done = set(status.get("completed", []))
    failures = dict(status.get("failures", {}))
    stopping = False
    child = None

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
        if child and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    configs = [(path, json.loads((root / path).read_text())) for path in plan["runs"]]
    per_step = {c["batch_size"] * (c["fact_tokens"] + c["market_tokens"] + 3) for _, c in configs}
    if len(per_step) != 1:
        raise SystemExit("all runs in one plan must have the same tokens per step")
    per_step = per_step.pop()
    total_steps = sum(c["steps"] for _, c in configs)

    def emit(record):
        record["time"] = time.time()
        with (output / "log.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def update(**values):
        status.update(
            values,
            updated_at=time.time(),
            completed=sorted(done),
            failures=failures,
            total_steps=total_steps,
            runs=len(configs),
        )
        atomic_json(status_file, status)

    emit(
        {"event": "start", "total_steps": total_steps, "tok_per_step": per_step, "plan": args.plan}
    )
    offset = 0
    for config_path, config in configs:
        name = config["name"]
        if name in done:
            offset += config["steps"]
            continue
        run = output / name
        while True:
            update(phase="training", run=name, global_step=offset, error=None)
            command = [
                sys.executable,
                "-m",
                "disclosure_forecasting.cli",
                "train",
                config_path,
                "--execute",
                "--output",
                str(run),
                "--device",
                plan["device"],
                "--precision",
                plan["precision"],
                "--eval-dir",
                plan["eval_dir"],
            ]
            if (run / "checkpoint.pt").exists():
                command.append("--resume")
            child = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            child_done = False
            last_step = 0
            with (output / f"{name}.console.log").open("a") as console:
                for line in child.stdout:
                    console.write(line)
                    console.flush()
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    event = record.get("event")
                    if event == "done":
                        child_done = True
                    if isinstance(record.get("step"), int):
                        last_step = record["step"]
                    if "loss" in record:
                        global_step = offset + last_step
                        emit(
                            {
                                "step": global_step,
                                "tokens": global_step * per_step,
                                "loss": record["loss"],
                                "tok_s": record["tok_s"],
                                "run": name,
                                "run_step": last_step,
                            }
                        )
                        update(
                            phase="training",
                            run=name,
                            global_step=global_step,
                            seconds_per_step=record["seconds_per_step"],
                            tok_s=record["tok_s"],
                        )
                    elif event in ("milestone", "monitor", "milestone_progress"):
                        emit({**record, "run": name, "run_step": record.get("step")})
                        if event == "milestone":
                            update(
                                last_milestone={k: record[k] for k in record if k != "time"}
                                | {"run": name}
                            )
                        else:
                            update(phase="evaluating", run=name)
            rc = child.wait()
            child = None
            if stopping:
                update(phase="paused", global_step=offset + last_step)
                emit({"event": "stopped", "step": offset + last_step})
                return
            if rc == 0 and child_done:
                break
            failures[name] = failures.get(name, 0) + 1
            if failures[name] > plan.get("retries", 1):
                update(
                    phase="failed",
                    error=f"{name}: exit {rc}, done={child_done}, attempts={failures[name]}",
                )
                emit({"event": "failed", "run": name, "exit_code": rc})
                raise SystemExit(rc or 1)
            emit({"event": "retry", "run": name, "exit_code": rc, "attempt": failures[name]})
            if not (run / "checkpoint.pt").exists() and run.exists():
                run.rename(run.with_name(f"{name}.failed-{failures[name]}"))
            time.sleep(30)
        done.add(name)
        offset += config["steps"]
        update(phase="run_complete", global_step=offset)
        emit({"event": "run_complete", "run": name, "step": offset})
    update(phase="training_complete", global_step=total_steps)
    emit({"event": "done", "step": total_steps})


if __name__ == "__main__":
    main()
