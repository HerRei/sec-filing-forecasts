# Stage S pre-registration: synthetic Gates 2 and 3

Written 20 October 2026, before any Stage S training run. The analysis is fixed in
`src/disclosure_forecasting/report.py`; configurations in `configs/stage-s/`; frozen test
worlds in `data/synthetic/stage-s-v2/` (SHA-256 per file in its `manifest.json`).
Changing any of these after the run starts makes this a different, unregistered study.

## Why this stage, and what it cannot show

The dossier (`docs/research_dossier.md`) orders the work as Gate 1 (real-data audit), then
Gates 2–5. Gate 1 needs a human source audit and real price data; it is not a GPU job. Stage S
runs the *simulator halves* of Gates 2 and 3 first because they are cheap (hours on one GPU)
and can falsify Bet 1 early: if the mechanism does not help even inside its own simulator,
the dossier's stopping condition applies before any money or weeks go into real data.
Stage S says nothing about real markets, alpha, or sim-to-real transfer.

## Question

With identical information, parameter count (5,140,487) and data budget, does

1. training over **varied disclosure processes** (delays, state-dependent lateness, selective
   non-reporting, preliminary values and corrections, regime jumps) improve forecasts when the
   disclosure process changes to **unseen families** (Gate 2, prior effect), and
2. an **explicit revision memory** (one GRU-updated slot per metric and economic period) beat a
   generic Transformer that sees the same records as independent tokens (Gate 3, memory effect)?

## Design

| Variant | Training family | Fact memory |
|---|---|---|
| A | regular (fixed delay, no corrections, no gaps) | sequence |
| B | varied | sequence |
| C | regular | revision slots |
| D | varied | revision slots |

- Both memories embed each record with the same encoder and GRU cell. Records carry only their
  own fields (value, metric, version, economic age, availability age, reporting lag). No record
  contains the value it replaces: relating a correction to its predecessor is the model's job.
- 3 seeds (1, 2, 3) per variant. Same-family variants see identical training worlds in the same
  order. 64 worlds per step, 6,884 steps, 440,576 worlds, about 100M padded tokens per run;
  AdamW 5e-4, 300 warm-up steps, cosine to 10 %, bf16 on the RX 9060 XT.
- Test predictions at 10M, 30M and 100M tokens (steps 689, 2,065, 6,884): the dossier's first
  scaling points, used as learning curves (same schedule for every variant).
- Simulator v2 timing: a record is executable at the open after its release; its immediate
  price reaction happens before that open, so only remaining drift (under-reaction) or reversal
  (over-reaction) is predictable. 20 % of training worlds have no return predictability at all.
  The information effect is deliberately stronger than in real markets (R² about 4–7 % at 1–5
  sessions in informative worlds) so mechanism differences are measurable.
- Evaluation: 4,096 frozen test worlds per family, one decision per world. Families: `regular`,
  `varied`, five held-out ones (`h_batched` calendar batching, `h_late_revisions` corrections
  20–50 sessions late plus third versions, `h_strategic` bad news 6–14 sessions late and
  frequently withheld, `h_noisy` preliminary noise far above training, `h_slow_response` response
  speed below the training range) and `null` (no return predictability).
- References on the same worlds: zero-mean Gaussian with past-20-session volatility
  (`ref_zero`); boosted trees on hand-built event-study features trained on 200,000 regular or
  varied worlds (`gbdt-regular`, `gbdt-varied`); `ref_state`, which uses hidden simulator state
  and is reported for scale only.

## Primary measure

Mean pinball loss over 7 quantiles (5, 10, 25, 50, 75, 90, 95 %) and horizons 1, 5 and 20
sessions, each horizon divided by √h, at the final milestone. Lower is better.

## Contrasts and decision rules

A contrast X − Y is computed per test world (seed-averaged, paired on identical worlds).
**Supported** means: the 95 % bootstrap interval over worlds (2,000 resamples, fixed RNG) lies
entirely below 0 **and** X is better in at least 2 of 3 seeds.

| Claim | Contrast | Data |
|---|---|---|
| Learnable at all | best variant − `ref_zero` | pooled held-out |
| Prior effect (Gate 2) | B − A **and** D − C | pooled held-out |
| Memory effect (Gate 3, simulator fixed) | D − B | `varied` |
| … on corrections | D − B | `varied`, decisions with a correction in the last 5 sessions |
| … transfers | D − B | pooled held-out |
| Beats the tree reference | D − `gbdt-varied` | pooled held-out |
| Null worlds | each variant: \|mean median\| at h = 5 < 0.1 (1 % units) and loss within +1 % of `ref_zero` | `null` |

Secondary, reported but not decisive: C − A, per-family contrasts, the stale-metric subset
(a metric ≥ 2 periods behind), 80 % interval coverage, learning curves (sample efficiency:
whether D reaches B's 100M loss with fewer tokens).

## What each outcome means (fixed in report.py)

- No variant beats `ref_zero` → inconclusive; fix the model or simulator before Gate 1.
- Prior and memory supported → proceed to Gate 1 (real-data audit).
- Prior only → continue with the generic model plus varied prior; drop revision-slot claims.
- Memory only → widen the training families before Gate 1.
- Neither → the dossier's stopping condition for Bet 1 as designed.
- D beats A but not B → "the data prior matters; the architectural contribution is unsupported"
  (dossier wording).
- D not better than `gbdt-varied` → the neural mechanism has not earned its complexity yet.

## Looked at before freezing (disclosed)

Only reference losses on the frozen test worlds, to confirm the simulator has learnable signal;
no neural variant had been trained. Nothing was tuned afterwards.

| family | ref_zero | ref_state | gbdt-regular | gbdt-varied |
|---|---|---|---|---|
| regular | 0.5440 | 0.5323 | 0.4925 | 0.5019 |
| varied | 0.5533 | 0.5492 | 0.5278 | 0.5127 |
| h_batched | 0.5484 | 0.5300 | 0.5187 | 0.5304 |
| h_late_revisions | 0.5663 | 0.5661 | 0.5369 | 0.5233 |
| h_strategic | 0.5281 | 0.5163 | 0.5031 | 0.4961 |
| h_noisy | 0.7651 | 0.8117 | 0.7241 | 0.6928 |
| h_slow_response | 0.4169 | 0.4055 | 0.4166 | 0.4177 |
| null | 0.4004 | 0.3905 | 0.4073 | 0.4085 |

The trees beat `ref_zero` by 3–9 % on most families, do not transfer to `h_slow_response`, and
invent signal in `null` worlds (+2 %): the comparisons are not decided in advance.

## Known limits

Single decision per world; one simulator author (the same person who designed the mechanism),
so held-out families are a weaker test than real data; padded-token budgets; no PFN or
state-space reference yet; seeds cover training noise only coarsely (3). None of the planned
real-data controls (survivorship, corporate actions, executable prices) are exercised here.
