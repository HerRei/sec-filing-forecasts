"""Stage S simulator (v2): economic process -> disclosure process -> market response.

One world is one synthetic issuer over SESSIONS trading sessions. Timing convention:
returns[t] is the excess return from the open of session t to the open of t+1. A record
with available == t was released after the close of t-1 and is first executable at the
open of t, so its immediate price reaction lands in returns[t-1], before any decision can
use the record. Labels start at returns[decision]; only the remaining drift (under-
reaction) or reversal (over-reaction) of past releases is predictable from them.
Release times are drawn before the response is generated, so a different schedule always
regenerates prices. Units are arbitrary; this is not a market-realism claim.
"""

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

SESSIONS = 224
PERIOD = 8
METRICS = 3
LOADINGS = (1.0, 0.7, 0.4)
HORIZONS = (1, 5, 20)
QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
MAX_VERSIONS = 3
FACT_FEATURES = 10
MARKET_FEATURES = 4
RETURN_UNIT = 0.01
TRAIN_FAMILIES = ("regular", "varied")
HELDOUT_FAMILIES = ("h_batched", "h_late_revisions", "h_strategic", "h_noisy", "h_slow_response")
EVAL_FAMILIES = TRAIN_FAMILIES + HELDOUT_FAMILIES + ("null",)  # Stage S (frozen)
# Stage S2: a broader training prior and five held-out families outside even its ranges.
S2_NEW_HELDOUT = ("h2_overreaction", "h2_restatements", "h2_metric_dropout", "h2_stale", "h2_burst")
ALL_FAMILIES = EVAL_FAMILIES + ("broad",) + S2_NEW_HELDOUT  # order fixes evaluation seeds
TRAINABLE = ("regular", "varied", "broad")
S2_EVAL_FAMILIES = ("regular", "varied", "broad") + HELDOUT_FAMILIES + S2_NEW_HELDOUT + ("null",)
Z = np.array([NormalDist().inv_cdf(q) for q in QUANTILES])


@dataclass(frozen=True)
class Event:
    record_id: int
    metric: int
    period: int  # economic period: sessions [period * PERIOD, (period + 1) * PERIOD)
    available: int  # first session whose open may use this version
    version: int
    value: float

    @property
    def key(self):
        return self.metric, self.period

    @property
    def period_end(self):
        return (self.period + 1) * PERIOD - 1


@dataclass
class World:
    family: str
    events: list  # sorted by (available, record_id)
    innovation: dict  # record_id -> value minus what was known before it (observable)
    returns: np.ndarray
    sigma: np.ndarray  # hidden: volatility actually used
    pending: np.ndarray  # hidden: information response still to come at each open
    params: dict


def process_params(family: str, rng: np.random.Generator) -> dict:
    """Release and response parameters. Held-out families leave the training ranges."""
    if family not in ALL_FAMILIES:
        raise ValueError(f"unknown family {family}")
    # Training also contains worlds without conditional return predictability (dossier).
    informative = family != "null" and rng.random() >= (0.4 if family == "broad" else 0.2)
    p = {
        "rho": rng.uniform(0.8, 0.97),
        "base_vol": rng.uniform(0.008, 0.02),
        "impact": rng.uniform(0.04, 0.12) if informative else 0.0,
        "immediate": rng.uniform(0.0, 1.3),  # < 1 under-reaction (drift), > 1 over-reaction
        "speed": rng.uniform(0.08, 0.5),
        "delay": (2, 2),
        "bad_delay": (0, 0),
        "missing_bad": 0.0,
        "missing_other": 0.0,
        "prelim_sd": 0.0,
        "revision_prob": 0.0,
        "revision_lag": (3, 15),
        "second_revision_prob": 0.0,
        "second_lag": (10, 30),
        "batch_every": 0,
        "jump_prob": 0.0,
    }
    if family == "regular":
        return p
    p.update(
        delay=(1, 4),
        bad_delay=(0, 4),
        missing_bad=0.25,
        missing_other=0.05,
        prelim_sd=rng.uniform(0.1, 0.35),
        revision_prob=rng.uniform(0.3, 0.8),
        jump_prob=0.3,
    )
    if family == "h_batched":
        p["batch_every"] = 11  # calendar-batched releases and corrections
    elif family == "h_late_revisions":
        p.update(revision_prob=0.8, revision_lag=(20, 50), second_revision_prob=0.5)
    elif family == "h_strategic":
        p.update(delay=(1, 1), bad_delay=(6, 14), missing_bad=0.5)
    elif family == "h_noisy":
        p.update(prelim_sd=rng.uniform(0.5, 0.8), revision_prob=0.9)
    elif family == "h_slow_response":
        p.update(speed=rng.uniform(0.02, 0.05), immediate=rng.uniform(0.0, 0.3))
    elif family == "broad":  # Stage S2 training prior: wider schedules, speeds, noise, null share
        p.update(
            batch_every=int(rng.integers(6, 17)) if rng.random() < 0.4 else 0,
            speed=rng.uniform(0.03, 0.6),
            prelim_sd=rng.uniform(0.1, 0.6),
            revision_lag=(3, 40),
            second_revision_prob=rng.uniform(0.0, 0.4),
            bad_delay=(0, 10),
            missing_bad=rng.uniform(0.1, 0.5),
        )
    elif family == "h2_overreaction":  # broad: immediate <= 1.3
        p.update(immediate=rng.uniform(1.4, 2.0))
    elif family == "h2_restatements":  # biased preliminaries corrected 50-100 sessions later
        p.update(revision_prob=0.6, revision_lag=(50, 100), bias_share=0.4, prelim_bias=(0.5, 1.0))
    elif family == "h2_metric_dropout":  # the most informative metric is never reported
        p.update(never_report=(0,))
    elif family == "h2_stale":  # broad: at most 4 + 10 sessions late
        p.update(delay=(15, 24))
    elif family == "h2_burst":  # broad: batches every 6-16 sessions
        p.update(batch_every=24)
    return p


def _batched(session: int, every: int) -> int:
    return session if not every else -(-session // every) * every


def simulate(seed: int, family: str, sessions: int = SESSIONS) -> World:
    rng = np.random.default_rng(seed)
    p = process_params(family, rng)
    shocks = rng.normal(0, 0.25, sessions)
    jump_at = int(rng.integers(20, sessions - 20)) if rng.random() < p["jump_prob"] else -1
    jump = rng.normal(0, 1.0)
    latent = np.empty(sessions)
    level = 0.0
    for t in range(sessions):
        level = p["rho"] * level + shocks[t] + (jump if t == jump_at else 0.0)
        latent[t] = level

    events, record = [], 0
    for period in range(sessions // PERIOD):
        end = (period + 1) * PERIOD - 1
        state = latent[period * PERIOD : end + 1].mean()
        for metric in range(METRICS):
            if metric in p.get("never_report", ()):
                continue
            loading = LOADINGS[metric]
            truth = loading * state + rng.normal(0, 0.05)
            missing = p["missing_bad"] if truth < -0.5 * loading else p["missing_other"]
            if rng.random() < missing:
                continue  # selective reporting: this period is never disclosed
            delay = int(rng.integers(p["delay"][0], p["delay"][1] + 1))
            if truth < -0.3 * loading:
                delay += int(rng.integers(p["bad_delay"][0], p["bad_delay"][1] + 1))
            when = _batched(end + delay, p["batch_every"])
            prelim = truth + (rng.normal(0, p["prelim_sd"]) if p["prelim_sd"] else 0.0)
            if p.get("bias_share") and rng.random() < p["bias_share"]:
                prelim += rng.uniform(*p["prelim_bias"])  # S2 only: no draw for other families
            versions = [(when, prelim)]
            if rng.random() < p["revision_prob"]:
                twice = rng.random() < p["second_revision_prob"]
                lag = int(rng.integers(p["revision_lag"][0], p["revision_lag"][1] + 1))
                when = _batched(when + lag, p["batch_every"])
                partial = rng.normal(0, 0.5 * p["prelim_sd"]) if twice else rng.normal(0, 0.03)
                versions.append((when, truth + partial))
                if twice:
                    lag = int(rng.integers(p["second_lag"][0], p["second_lag"][1] + 1))
                    versions.append(
                        (_batched(when + lag, p["batch_every"]), truth + rng.normal(0, 0.03))
                    )
            for version, (when, value) in enumerate(versions, 1):
                if when >= sessions:
                    break  # later versions of this key are never observed in the window
                events.append(Event(record, metric, period, when, version, float(value)))
                record += 1
    events.sort(key=lambda e: (e.available, e.record_id))

    # Market response, generated after the release schedule and the measured values.
    arrivals: dict[int, list[Event]] = {}
    for event in events:
        arrivals.setdefault(event.available, []).append(event)
    known_value, known_period = [0.0] * METRICS, [-1] * METRICS
    key_value: dict[tuple, float] = {}
    innovation = {}
    sigma = p["base_vol"] * (1 + 0.5 * np.minimum(np.abs(latent), 2.0))
    noise = rng.standard_t(5, sessions) * np.sqrt(3 / 5)
    returns = np.zeros(sessions)
    pending = np.zeros(sessions)
    backlog = 0.0
    for t in range(sessions):
        shock = 0.0
        for event in arrivals.get(t, ()):
            prior = key_value[event.key] if event.version > 1 else known_value[event.metric]
            innovation[event.record_id] = event.value - prior
            key_value[event.key] = event.value
            if event.period >= known_period[event.metric]:
                known_value[event.metric] = event.value
                known_period[event.metric] = event.period
            shock += p["impact"] * LOADINGS[event.metric] * (event.value - prior)
        if t > 0:
            returns[t - 1] += p["immediate"] * shock  # reaction before the record is executable
        backlog += (1 - p["immediate"]) * shock
        pending[t] = backlog
        drift = p["speed"] * backlog
        backlog -= drift
        returns[t] += drift + sigma[t] * noise[t]
    return World(family, events, innovation, returns.astype(np.float32), sigma, pending, p)


def decision_for(seed: int, market_tokens: int = 96) -> int:
    return int(np.random.default_rng([seed, 913]).integers(market_tokens, SESSIONS - max(HORIZONS)))


def visible(world: World, decision: int) -> list:
    return [e for e in world.events if e.available <= decision]


def encode(world: World, decision: int, fact_tokens: int = 128, market_tokens: int = 96) -> dict:
    """Model inputs. Records carry only their own fields (no pre-joined previous value):
    relating a correction to the record it replaces is the models' job."""
    if decision < market_tokens or decision + max(HORIZONS) > len(world.returns):
        raise ValueError("insufficient market history or immature labels")
    seen = visible(world, decision)
    facts = np.zeros((fact_tokens, FACT_FEATURES), dtype=np.float32)
    slots = np.full(fact_tokens, -1, dtype=np.int64)
    ranks = np.full(fact_tokens, -1, dtype=np.int64)
    mapping: dict[tuple, int] = {}
    count: dict[tuple, int] = {}
    for index, event in enumerate(seen[-fact_tokens:]):
        slots[index] = mapping.setdefault(event.key, len(mapping))
        ranks[index] = count.get(event.key, 0)
        count[event.key] = ranks[index] + 1
        if ranks[index] >= MAX_VERSIONS:
            raise ValueError("more versions than MAX_VERSIONS")
        facts[index] = (
            event.value,
            event.metric == 0,
            event.metric == 1,
            event.metric == 2,
            (event.version - 1) / 2,
            event.version > 1,
            (decision - event.period_end) / 64,
            (decision - event.available) / 64,
            (event.available - event.period_end) / 16,
            1.0,
        )
    start = decision - market_tokens
    history = world.returns[start:decision]
    # Records that became executable at the next open: their reaction is in this return.
    arrived = np.bincount(
        [e.available - 1 - start for e in seen if start < e.available <= decision],
        minlength=market_tokens,
    )[:market_tokens]
    market = np.stack(
        (
            history / RETURN_UNIT,
            np.abs(history) / RETURN_UNIT,
            np.arange(-market_tokens, 0) / market_tokens,
            arrived,
        ),
        axis=-1,
    ).astype(np.float32)
    target = np.array(
        [world.returns[decision : decision + h].sum() / RETURN_UNIT for h in HORIZONS],
        dtype=np.float32,
    )
    return {"facts": facts, "slots": slots, "ranks": ranks, "market": market, "target": target}


def references(world: World, decision: int) -> dict:
    """Evaluation-only extras: subset flags and two Gaussian quantile references (1% units).
    ref_zero is a fair zero-mean / past-20-session volatility forecast. ref_state uses the
    HIDDEN simulator state (pending response, speed, current volatility): an upper-ish
    reference for how much is predictable, not a competitor."""
    seen = visible(world, decision)
    recent = [e for e in seen if e.available >= decision - 5]
    latest = [max((e.period for e in seen if e.metric == m), default=-1) for m in range(METRICS)]
    horizons = np.array(HORIZONS, dtype=np.float64)
    past = world.returns[decision - 20 : decision].astype(np.float64)
    vol = max(float(past.std()), 1e-3)
    zero = (vol * np.sqrt(horizons))[:, None] * Z[None, :]
    speed = world.params["speed"]
    mean = world.pending[decision] * (1 - (1 - speed) ** horizons)
    state = mean[:, None] + (world.sigma[decision] * np.sqrt(horizons))[:, None] * Z[None, :]
    return {
        "recent_revision": any(e.version > 1 for e in recent),
        "recent_release": any(e.version == 1 for e in recent),
        # A metric at least two periods behind the newest disclosed period: missing or late.
        "stale_metric": min(latest) <= max(latest) - 2,
        "informative": world.params["impact"] > 0,
        "ref_zero": (zero / RETURN_UNIT).astype(np.float32),
        "ref_state": (state / RETURN_UNIT).astype(np.float32),
    }


def summary_features(world: World, decision: int) -> np.ndarray:
    """Hand-built event-study features for the boosted-tree reference (observable only)."""
    seen = visible(world, decision)
    r = world.returns[:decision].astype(np.float64)
    features = []
    for metric in range(METRICS):
        mine = [e for e in seen if e.metric == metric]
        if mine:
            last_period = max(e.period for e in mine)
            current = [e for e in mine if e.period == last_period][-1]
            older = [e for e in mine if e.period < last_period]
            previous = (
                [e for e in older if e.period == max(o.period for o in older)][-1].value
                if older
                else 0.0
            )
            features += [
                current.value,
                current.value - previous,
                decision - current.available,
                current.version,
                decision - current.period_end,
            ]
        else:
            features += [0.0, 0.0, 99.0, 0.0, 99.0]
        weight = LOADINGS[metric]
        features += [
            weight
            * sum(world.innovation[e.record_id] for e in mine if e.available >= decision - 5),
            weight
            * sum(
                world.innovation[e.record_id]
                for e in mine
                if decision - 20 <= e.available < decision - 5
            ),
            weight
            * sum(
                world.innovation[e.record_id]
                for e in mine
                if e.version > 1 and e.available >= decision - 10
            ),
        ]
    reaction = sum(float(r[e.available - 1]) for e in seen if e.available >= decision - 5)
    features += [
        reaction / RETURN_UNIT,
        r[-1] / RETURN_UNIT,
        r[-5:].sum() / RETURN_UNIT,
        r[-20:].sum() / RETURN_UNIT,
        r[-20:].std() / RETURN_UNIT,
        r[-96:].std() / RETURN_UNIT,
        sum(e.available >= decision - 5 for e in seen),
        sum(e.version > 1 and e.available >= decision - 10 for e in seen),
    ]
    return np.asarray(features, dtype=np.float32)
