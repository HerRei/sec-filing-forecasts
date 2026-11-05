"""Synthetic Stage S runner. Planning is the default; training needs --execute."""

import contextlib
import hashlib
import json
import math
import signal
import time
from pathlib import Path

import numpy as np
import torch

from .data import TrainStream, eval_families, load_eval, verify_eval_sets
from .model import ForecastModel, model_inputs, pinball
from .simulator import HORIZONS, TRAINABLE

REQUIRED = {
    "name": str,
    "family": str,
    "memory": str,
    "seed": int,
    "width": int,
    "layers": int,
    "heads": int,
    "batch_size": int,
    "steps": int,
    "lr": float,
    "warmup_steps": int,
    "weight_decay": float,
    "fact_tokens": int,
    "market_tokens": int,
    "milestones": list,
    "monitor_every": int,
    "workers": int,
}
OPTIONAL = {"fact_budget": int}  # Stage S2; absent = all fact tokens (Stage S behaviour)


def read_config(path):
    config = json.loads(Path(path).read_text())
    unknown = set(config) - set(REQUIRED) - set(OPTIONAL)
    if set(REQUIRED) - set(config) or unknown:
        raise ValueError(f"config keys differ from required fields: {set(config) ^ set(REQUIRED)}")
    for key, kind in {**REQUIRED, **OPTIONAL}.items():
        if key not in config:
            continue
        if kind is float and isinstance(config[key], int):
            config[key] = float(config[key])
        if not isinstance(config[key], kind):
            raise ValueError(f"{key} must be {kind.__name__}")
    if config["memory"] not in ("sequence", "revision"):
        raise ValueError("invalid memory variant")
    if config["family"] not in TRAINABLE:
        raise ValueError("held-out and null families are evaluation-only")
    if config["width"] % config["heads"] or config["market_tokens"] >= 200 or config["lr"] <= 0:
        raise ValueError("invalid width/heads, history or learning rate")
    milestones = config["milestones"]
    if milestones != sorted(set(milestones)) or not milestones or milestones[-1] != config["steps"]:
        raise ValueError("milestones must increase and end at the final step")
    return config


def tokens_per_sample(config):
    return config["fact_tokens"] + config["market_tokens"] + len(HORIZONS)


def make_model(config):
    return ForecastModel(
        config["width"],
        config["layers"],
        config["heads"],
        config["memory"],
        config.get("fact_budget"),
    )


def plan(config):
    # CPU construction only: no optimizer, data generation, GPU discovery or output files.
    torch.manual_seed(config["seed"])
    model = make_model(config)
    per_step = config["batch_size"] * tokens_per_sample(config)
    return {
        "status": "planned_not_started",
        "config": config,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "worlds": config["steps"] * config["batch_size"],
        "padded_tokens": config["steps"] * per_step,
        "milestone_tokens": [step * per_step for step in config["milestones"]],
        "real_data_gate": "pending_manual_audit",
    }


def learning_rate(config, step):
    if step <= config["warmup_steps"]:
        return config["lr"] * step / max(1, config["warmup_steps"])
    progress = (step - config["warmup_steps"]) / max(1, config["steps"] - config["warmup_steps"])
    return config["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def autocast(device, precision):
    if device == "cpu" or precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[precision])


@torch.no_grad()
def predict(model, data, device, precision, should_stop, batch_size=1024):
    """Quantile predictions (N, H, Q) for a frozen evaluation set, or None if stopped."""
    model.eval()
    outputs = []
    for start in range(0, len(data["target"]), batch_size):
        if should_stop():
            model.train()
            return None
        part = {
            k: torch.from_numpy(data[k][start : start + batch_size]).to(device)
            for k in ("facts", "slots", "ranks", "market")
        }
        with autocast(device, precision):
            outputs.append(model(**model_inputs(part)).float().cpu())
    model.train()
    return torch.cat(outputs).numpy()


def summarize(predictions, data):
    target = torch.from_numpy(data["target"])
    loss = pinball(torch.from_numpy(predictions), target).mean().item()
    reference = pinball(torch.from_numpy(data["ref_zero"]), target).mean().item()
    inside = (data["target"] >= predictions[..., 1]) & (data["target"] <= predictions[..., 5])
    return {
        "pinball": loss,
        "ref_zero": reference,
        "coverage_80": float(inside.mean()),
        "median_bias": float(predictions[..., 3].mean()),
    }


def execute(config, output, device="cpu", precision="fp32", resume=False, eval_dir=None):
    if device not in ("cpu", "cuda:0"):
        raise ValueError("use cpu or one explicitly pinned GPU (cuda:0 on ROCm too)")
    if precision not in ("fp32", "fp16", "bf16") or (device == "cpu" and precision != "fp32"):
        raise ValueError("CPU uses fp32; GPU precision must be fp32/fp16/bf16")
    if eval_dir is None:
        raise ValueError("an evaluation-set directory is required")
    verify_eval_sets(eval_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if checkpoint.exists() and not resume:
        raise ValueError("checkpoint exists; use --resume or a new output directory")
    if (output / "log.jsonl").exists() and not checkpoint.exists():
        raise ValueError("log exists without checkpoint; use a new output directory")
    torch.manual_seed(config["seed"])
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], betas=(0.9, 0.95), weight_decay=config["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device != "cpu" and precision == "fp16")
    step = 0
    if resume:
        # Everything is restored from CPU copies; the optimizer state follows the parameters.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["config_hash"] != config_hash:
            raise ValueError("resume configuration differs from checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        step = saved["step"]
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    previous_handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def log(record):
        record["time"] = time.time()
        with (output / "log.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def save():
        temporary = output / "checkpoint.tmp"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config_hash": config_hash,
                "step": step,
            },
            temporary,
        )
        temporary.replace(checkpoint)

    per_step = config["batch_size"] * tokens_per_sample(config)
    families, heldout_families = eval_families(eval_dir)
    monitor = {f: load_eval(eval_dir, "monitor", f) for f in families}
    log({"event": "start", "step": step, "total_steps": config["steps"], "tok_per_step": per_step})
    stream = TrainStream(
        config["family"],
        config["seed"],
        config["steps"] * config["batch_size"],
        config["fact_tokens"],
        config["market_tokens"],
    )
    loader = torch.utils.data.DataLoader(
        stream,
        batch_size=config["batch_size"],
        sampler=range(step * config["batch_size"], len(stream)),
        num_workers=config["workers"] if device != "cpu" else 0,
        pin_memory=device != "cpu",
        prefetch_factor=4 if device != "cpu" and config["workers"] else None,
        # spawn: workers never inherit an initialised HIP runtime.
        multiprocessing_context="spawn" if device != "cpu" and config["workers"] else None,
        drop_last=True,
    )

    def milestone():
        """Test-set predictions at this step; False if a pause interrupted it."""
        arrays = {}
        for family in families:
            data = load_eval(eval_dir, "test", family)
            predictions = predict(model, data, device, precision, lambda: stopped)
            if predictions is None:
                return False
            arrays[family] = predictions.astype(np.float32)
            # Heartbeat for the controller's stall detector during long evaluations.
            log({"event": "milestone_progress", "step": step, "family": family})
        temporary = output / f"predictions-step{step}.tmp.npz"
        np.savez_compressed(temporary, **arrays)
        temporary.replace(output / f"predictions-step{step}.npz")
        heldout = [summarize(arrays[f], load_eval(eval_dir, "test", f)) for f in heldout_families]
        log(
            {
                "event": "milestone",
                "step": step,
                "tokens": step * per_step,
                "heldout_pinball": float(np.mean([s["pinball"] for s in heldout])),
                "heldout_ref_zero": float(np.mean([s["ref_zero"] for s in heldout])),
            }
        )
        return True

    def done(m):
        return (output / f"predictions-step{m}.npz").exists()

    last_save = time.monotonic()
    window_started, window_steps, losses = time.monotonic(), 0, []
    try:
        # A pause during a milestone evaluation checkpoints at that step: finish it first.
        if any(m < step and not done(m) for m in config["milestones"]):
            raise RuntimeError("a milestone before the checkpoint has no predictions")
        if step in config["milestones"] and not done(step) and not milestone():
            save()
            log({"event": "stopped", "step": step})
            return
        for batch in loader:
            step += 1
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(config, step)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, precision):
                prediction = model(**model_inputs(batch))
            loss = pinball(prediction.float(), batch["target"]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.detach())
            window_steps += 1
            if step % 10 == 0 or step in (1, config["steps"]):
                if device != "cpu":
                    torch.cuda.synchronize()
                elapsed = (time.monotonic() - window_started) / window_steps
                log(
                    {
                        "step": step,
                        "loss": torch.stack(losses).mean().item(),
                        "lr": learning_rate(config, step),
                        "seconds_per_step": elapsed,
                        "tokens": step * per_step,
                        "tok_s": per_step / elapsed,
                    }
                )
                window_started, window_steps, losses = time.monotonic(), 0, []
            if not stopped and step % config["monitor_every"] == 0:
                scores = {}
                for family, data in monitor.items():
                    predictions = predict(model, data, device, precision, lambda: stopped)
                    if predictions is None:
                        break
                    scores[family] = summarize(predictions, data)
                else:
                    log({"event": "monitor", "step": step, "scores": scores})
            if not stopped and step in config["milestones"] and milestone():
                save()
                last_save = time.monotonic()
            if stopped or time.monotonic() - last_save >= 600:
                save()
                last_save = time.monotonic()
            if stopped:
                log({"event": "stopped", "step": step})
                return
            if step >= config["steps"]:
                break
        missing = [m for m in config["milestones"] if not done(m)]
        if missing:
            raise RuntimeError(f"run ended without milestone predictions {missing}")
        save()
        log({"event": "done", "step": step})
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
