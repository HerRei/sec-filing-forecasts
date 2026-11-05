# Stage S2 pre-registration: what Stage S taught, tested properly

Written 20 October 2026 after Stage S (`docs/results/stage-s/`, read its `INTERPRETATION.md`) and
before any S2 training. Analysis fixed in `src/disclosure_forecasting/report_s2.py`; configs in
`configs/stage-s2/`; frozen test worlds in `data/synthetic/stage-s2/` (hashes in its manifest);
everything hashed in `docs/stage-s2-frozen.sha256` at launch. The eight Stage S families produce
byte-identical worlds (tested against the frozen Stage S set), so Stage S stays reproducible.

## What Stage S left open

1. The revision memory made no practical difference when the Transformer could see every record.
   An explicit memory should only matter when it cannot: under a token budget.
2. The varied prior transferred to shifts along dimensions it had seen and failed on a new
   schedule type (calendar batching). Does a broader prior transfer to families outside it?
3. Trees on hand-built features stayed ahead, and B/D were still improving at 100M tokens.
4. Every model invented a negative drift in no-signal worlds ("silence = bad news").

## Questions and runs (all 5,140,487 parameters, bf16, RX 9060 XT, seeds 1–3)

| Q | Runs | Training family | Memory | Fact tokens reaching the Transformer | Tokens |
|---|---|---|---|---|---|
| Q1 | B-k16, D-k16, B-k32, D-k32 | varied | sequence / revision | 16 or 32 (of up to 128 records) | 100M |
| Q2/Q3 | B300, E300 | varied / **broad** | sequence | all | 300M |

- **Budget (Q1):** the sequence model passes on the newest K records; the revision model reads all
  records into its slots and passes on the K most recently updated slots. Same Transformer cost.
- **broad** = varied plus: calendar batching in 40 % of worlds (every 6–16 sessions), response speed
  0.03–0.6, preliminary noise 0.1–0.6, corrections 3–40 sessions late, second corrections up to
  40 %, bad news up to 10 sessions late, withholding 10–50 %, and 40 % no-signal worlds (was 20 %).
- **New held-out families, all outside broad:** `h2_overreaction` (immediate reaction 1.4–2.0×,
  broad ≤ 1.3), `h2_restatements` (40 % of preliminaries biased upward by 0.5–1.0, corrected 50–100
  sessions later), `h2_metric_dropout` (the most informative metric is never reported), `h2_stale`
  (every release 15–28 sessions late, broad ≤ 14), `h2_burst` (batches every 24 sessions, broad
  ≤ 16). The Stage S held-out families are partly inside broad and are reported as "partly seen".
- References: zero-mean/volatility, boosted trees trained on 200,000 varied or broad worlds.
- Milestones: 10M, 30M, 100M (+300M for Q2) tokens; learning curves from the same runs.

## Effect measure and rules (changed from Stage S, because of Stage S)

- **Family-balanced relative effect:** for each family, (loss X − loss Y) / loss Y in %, seed-
  averaged per world; the effect is the mean over the families, so no family dominates by scale.
- 95 % interval: 2,000 bootstrap resamples of worlds **within each family**.
- **supported:** effect ≤ −0.5 %, interval below 0, X better in ≥ 2 of 3 seeds.
  **negligible:** significant but smaller than 0.5 % (or seeds disagree). **not supported:**
  interval includes 0. **harmful:** effect ≥ +0.5 % with interval above 0.
- **Harm check:** every contrast lists the families where X is significantly worse by ≥ 0.5 %.

## Primary contrasts (one per question)

| Q | Contrast | Families |
|---|---|---|
| Q1 memory under a budget | D-k16 − B-k16 | varied + h_late_revisions |
| Q2 broader prior transfers | E300 − B300 | the five new held-out families |
| Q3 data catches the trees | E300 − gbdt-broad | the five new held-out families |

Secondary (reported, not decisive): the same at K = 32; Q1 on old and new held-out families and on
decisions right after a correction; E300 − B300 on the old held-out families (partly seen), on
h_batched (does broadening fix Stage S's failure), on varied (cost of broadening) and on null;
B300 − gbdt-varied on old held-out (Stage S's tree comparison at 3× data); every model vs
zero-mean; null check per model (|median h5| < 0.1 and loss within +1 % of zero-mean);
learning curves.

## Looked at before freezing (disclosed)

Reference losses only (no S2 neural run existed):

| family | ref_zero | gbdt-varied | gbdt-broad |
|---|---|---|---|
| varied | 0.5533 | 0.5127 | 0.5193 |
| broad | 0.5460 | 0.5183 | 0.5095 |
| h_batched | 0.5484 | 0.5304 | 0.5102 |
| h2_overreaction | 0.7472 | 0.6774 | 0.6807 |
| h2_restatements | 0.7056 | 0.6327 | 0.6367 |
| h2_metric_dropout | 0.4447 | 0.4376 | 0.4341 |
| h2_stale | 0.5353 | 0.5205 | 0.5056 |
| h2_burst | 0.5201 | 0.5122 | 0.4916 |
| null | 0.4004 | 0.4085 | 0.3995 |

For trees, the broad prior helps on burst/stale/batched releases and null worlds and costs a little
on overreaction, restatements and the training families: the neural Q2 outcome is open.

## Limits

Same simulator author as the mechanism; three seeds; one decision per world; Q1 tests memory as
compression of a fixed 224-session history, not truly long histories; broad changes several things
at once, so Q2 can say whether broadening helps, not which part does. Simulator results only.
