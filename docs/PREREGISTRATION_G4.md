# Gate 4 pre-registration: does the simulator-trained model help on real filings?

Written 26 October 2026, after the Gate 1 audit (`docs/results/gate1/report.md`) and before any
outcome label existed. Labels are computed by `disclosure_forecasting.real.g4 labels`, which refuses
to run unless the SHA-256 of this file is recorded in `data/processed/g4/prereg.sha256` and still
matches. The analysis is fixed in `src/disclosure_forecasting/real/report_g4.py`; the models in
`src/disclosure_forecasting/real/g4.py`. Changing any of them after the labels exist makes this a
different, unregistered study.

## Status of the evidence before this stage

Everything so far is simulation. Stage S2 found that a broader training prior (variant E, 300M
tokens) cut pinball loss by 5.6 % on five held-out simulated disclosure families (3/3 seeds), that an
explicit revision memory did not help, and that boosted trees on hand-built features were not beaten.
Gate 1 established which real disclosures can be used: revenue and inventory pass; backlog (ASC 606
remaining performance obligations) failed the semantics audit; orders and guidance have no
structured company-level source.

## Question

On real 10-Q and 10-K filings of US industrial companies, does the simulator-pretrained broad-prior
model (E300, fine-tuned on real events) forecast the distribution of the remaining 5- and 20-session
benchmark-adjusted return better than (1) a zero-mean volatility forecast, (2) a linear event-study
model, (3) boosted trees on the same information, and (4) the same network trained on real data
only? My prior expectation, stated before seeing anything: post-filing returns are close to
unpredictable, the filing usually comes days after the earnings release, and the likely outcome is
"not supported" for most contrasts.

## Data (frozen; hashes in `data/processed/g4/build.json` and below)

- Universe: SIC 3400-3899 without 357x, 366x-367x, 383x-389x; US business address; operating
  companies (`docs/results/gate1/report.md`).
- One decision per original 10-Q or 10-K with XBRL records, accepted from June 2010, at
  executable_at = the first NYSE open at or after the authoritative EDGAR acceptance time + 30 min.
- Filters, all known at the decision: price series exists (current listing; survivorship stated),
  identity check passed, 120 sessions of price history, quoted close at the previous session
  >= $5, median dollar volume over the previous 20 sessions >= $1M, at least one transformable
  revenue or inventory record.
- Events: **11,569** from **305** issuers (SIC families: 34: 1,562, 35: 3,624, 36: 1,254, 37: 3,064, 38: 2,065).
- Model inputs, identical in information for every model:
  - Disclosure records (revenue and inventory, every version, executable at or before the
    decision, economic period within 3 years): value = log year-over-year change / 0.25, using the
    year-ago value known when the record itself became available; version; economic age,
    availability age and reporting lag in sessions (simulator scaling). Up to 48 records.
  - Market: 96 daily excess returns (stock total return minus XLI total return, open to open, the
    last one ending at the decision open), their absolute values, position, and arrival markers
    (earnings 8-K Item 2.02 and periodic filings, placed on the return that contains their reaction).
  - Linear and tree models get the same content as hand-built features (28 columns,
    `PRIMARY_FEATURES` in `g4.py`).
- Issuer holds: issuers with sha256(CIK) mod 5 == 0 (3,167 events) are never used for training or
  validation; they appear only in test sets and are reported separately.

## Labels (computed only after this file is hashed)

y_h = 100 x [(P_stock(s+h) / P_stock(s) - 1) - (P_XLI(s+h) / P_XLI(s) - 1)], h in {1, 5, 20}, where
P is the total-return open price (open x adjclose / close, Yahoo), s the executable session, and
s+h counted on the XNYS calendar. Entry at the executable open, exit at the open h sessions later.
Events without an exit price are dropped and counted. h = 1 is an auxiliary training target only.

## Splits

| fold | test period (executable session) | validation | training |
|---|---|---|---|
| dev2019 | 2019 (719 events, 198 from held-out issuers) | the 12 months before (487) | everything earlier (3,130) |
| dev2020 | 2020 (734 events, 198 from held-out issuers) | the 12 months before (510) | everything earlier (3,626) |
| dev2021 | 2021 (844 events, 215 from held-out issuers) | the 12 months before (527) | everything earlier (4,145) |
| dev2022 | 2022 (817 events, 216 from held-out issuers) | the 12 months before (618) | everything earlier (4,683) |
| final | 2023-01-01 to the last mature decision (3,329 events, 867 from held-out issuers) | the 12 months before (589) | everything earlier (5,310) |

Training and validation events whose label window (20 sessions) plus a 5-session embargo reaches
the next block are purged. All feature scaling is fitted on training (plus validation for the
references) only. The final period from 1 January 2023 is the untouched test: evaluated once, after
every choice in this file was frozen. Development folds are evaluated with the same code and reported
as secondary evidence; nothing is tuned on them either.

## Models (all settings fixed here)

| name | what |
|---|---|
| ref_zero | Normal(0, sigma20 x sqrt(h)); sigma20 = std of the last 20 daily excess returns (floor 0.1 %) |
| ref_volcal | zero-mean/volatility, calibrated: sigma20 x sqrt(h) x empirical quantiles of y / (sigma20 sqrt(h)) on training + validation |
| linear | linear quantile regression (sklearn QuantileRegressor, alpha 1e-3, highs) of y / (sigma20 sqrt(h)) on the 28 standardised, 1/99 %-winsorised features, one model per horizon and quantile |
| gbdt-s{1,2,3} | HistGradientBoosting quantile regression on the same target and features: 300 iterations max, learning rate 0.05, 15 leaves, min 100 per leaf, L2 1.0, early stopping on a random 15 %, random_state = seed |
| broad-ft-s{k} | the 5,140,487-parameter Stage S2 network initialised from E300-s{k} (broad prior), fine-tuned on real training events |
| varied-ft-s{k} | the same from B300-s{k} (varied prior, Stage S2) |
| scratch-ft-s{k} | the same network, random initialisation (torch seed k) |
| broad-zeroshot-s{k}, varied-zeroshot-s{k} | the pretrained networks without any real training |

Fine-tuning (all three inits): AdamW (0.9, 0.95), learning rate 2e-4, weight decay 0.01, batch 64,
100 warm-up steps, cosine to 10 % over 15 epochs, gradient clip 1.0, pinball loss on h = 1, 5, 20
(each / sqrt(h)) as in the simulator stages; validation loss (h = 5 and 20) after every epoch,
early stopping with patience 3, the best epoch is kept. Seeds 1, 2, 3 set the initialisation
(scratch), the pretrained checkpoint (E300-s k, B300-s k) and the data order. CPU on ddp, fp32.
A software smoke test on random labels (never real outcomes) checked that the code path runs.

## Primary measure

Mean pinball loss over 7 quantiles (5, 10, 25, 50, 75, 90, 95 %) and horizons 5 and 20 sessions,
each divided by sqrt(h), in 1 % return units, on the final test period. Lower is better. Neural
models: per-event loss averaged over the three seeds; trees likewise.

## Effect measure and rules (as in Stage S2)

- Family-balanced relative effect: for each SIC major group (34, 35, 36, 37, 38),
  (loss X - loss Y) / loss Y in %; the effect is the mean over the five groups.
- 95 % interval: 2,000 bootstrap resamples of calendar weeks (all events executable in the same week
  stay together, across families), RNG seed 20261026.
- **supported**: effect <= -0.5 %, interval entirely below 0, and X better in >= 2 of 3 seeds (seed k
  of X against seed k of Y where Y has seeds, else against Y). **negligible**: interval below 0 but
  effect > -0.5 % or seeds disagree. **not supported**: interval includes 0. **harmful**:
  effect >= +0.5 % and interval above 0.
- Harm check: every contrast lists the families where X is significantly worse by >= 0.5 %.

## Primary contrasts (final period, all issuers)

| # | contrast | question |
|---|---|---|
| P1 | broad-ft - ref_volcal | does the model forecast real post-filing returns better than volatility alone? |
| P2 | broad-ft - linear | better than a linear event-study model on the same information? |
| P3 | broad-ft - gbdt | better than boosted trees on the same information? |
| P4 | broad-ft - scratch-ft | does simulator pretraining help on real data? |

Four primary contrasts, no multiplicity correction; a single "supported" among them is weak evidence.

## Secondary (reported, not decisive)

- broad-ft - varied-ft (does the broader prior also transfer better on real data, as in S2);
  broad-zeroshot - ref_volcal (pure transfer); scratch-ft, linear, gbdt - ref_volcal; ref_zero
  versus ref_volcal; every primary contrast on each development fold.
- Per horizon (5 and 20 separately), per test year, held-out issuers only, events with a revision of
  revenue or inventory in the last 63 sessions.
- Disclosure content: gbdt and linear with market features only versus the full features; broad-ft
  evaluated with all disclosure tokens masked (a distribution shift, so only indicative).
- Backlog and orders: gbdt / linear with backlog features (failed Gate 1, so secondary only) and
  with ALFRED core capital goods orders (NEWORDER vintages, executable the session after the
  vintage date: latest-vintage year-over-year growth, its change, and the revision of the previous
  month). The ALFRED server answered slowly (502 errors), so `data/processed/g4/macro.parquet` is
  built after this file is frozen, from vintage data only; its hash goes into the results.
- Calibration: share of outcomes inside the 10-90 % interval.
- Economic sanity check, not a strategy: mean realised 20-session excess return of the top minus the
  bottom quintile of predicted median, per test year, before costs.

## What each outcome means

- P1 not supported: no evidence that this model forecasts post-filing returns better than
  volatility alone; the simulator results do not transfer to this real task.
- P1 supported, P3 not: the network learned something real but has not earned its complexity
  over trees.
- P4 supported: simulator pretraining helps on real data (the claim the project was built to test).
  P4 not supported: the simulator prior adds nothing measurable here.
- Any result holds for currently listed US industrial companies, 2023-2026, after the filing time;
  it says nothing about trading profits (no costs, no capacity, survivorship-biased sample).

## Limits known before the labels

Survivorship (only companies listed today); the 10-Q/10-K comes after the earnings release (the
decision is late by design); a few hundred issuers and one test period with common shocks; backlog,
orders and guidance unavailable as clean structured data; Yahoo's prices are unofficial; one author
designed simulator, model and test.

## Frozen inputs

- `data/processed/g4/events.parquet`: sha256 `67405cb109a566da8a4e146cd06103009f9c1fa6c64cf1ceff29c262b02abf19`
- `data/processed/g4/tokens.npz`: sha256 `f7ea9abb318efc28ec506d7c222f6c51de391f6a5e8c6f40d2c0564d7ce3a8b1`
- `data/processed/g4/identity_check.parquet`: sha256 `ef1147780e7d810eab288ff23d40030a93493608ce5c5d72103fcd4cccf76e11`
- Counts in the fold table are before the label-availability filter (events without an exit price
  are dropped after labelling and counted in `labels.json`).
