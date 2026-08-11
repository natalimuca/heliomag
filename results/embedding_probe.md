# Surya embedding probe vs. classical baseline

Weekly Surya embeddings (`data/embeddings_merged.npz`, 754 samples, 1280-dim, 2010-05 to 2024-12,
98.7% coverage of the weekly grid — see gaps below), same chronological split as the classical
baseline: train <= 2021-12-31, test >= 2022-01-01 (598 train / 156 test).

Classical features recomputed at the same weekly sample points (not the daily grid used in
`baseline_probe.md`) so the comparison is apples-to-apples — absolute numbers differ from that
table because of the coarser, smaller sample, but the relative comparison (classical vs. embedding
on identical rows) is fair.

For embedding models, PCA dimensionality and Ridge alpha are selected per lead time via
`GridSearchCV` over `TimeSeriesSplit(5)` on the training set only (no test leakage): PCA in
{5,10,20,30,50,75,100}, alpha log-spaced 1e-2 to 1e4. Two follow-up variants added after the first
pass: a **residual-correction** model (embeddings predict what the classical linear fit gets
wrong, `classical_pred + embed_model(residual)`, rather than competing standalone), and the same
probe run against **Kp** as an alternate target (daily mean of the eight 3-hourly Kp values,
`scripts/fetch_data.py`) instead of Ap.

### Target: Ap

| lead (days) | persistence | classical (linear) | embed (PCA+Ridge) | embed+classical | embed (PCA+MLP) | residual-correction |
|---|---|---|---|---|---|---|
| 3  | 14.82 | **8.32** | 8.62 | 8.62 | 19.96 | 8.52 |
| 5  | 19.36 | **15.66** | 17.55 | 17.20 | 25.79 | 17.30 |
| 7  | 16.96 | **12.30** | 14.72 | 13.88 | 22.16 | 15.64 |
| 10 | 14.77 | 8.62 | **8.47** | **8.47** | 17.66 | 8.72 |
| 14 | 17.69 | **12.44** | 13.09 | 13.09 | 39.80 | 13.10 |

### Target: Kp

| lead (days) | persistence | classical (linear) | embed (PCA+Ridge) | embed+classical | embed (PCA+MLP) | residual-correction |
|---|---|---|---|---|---|---|
| 3  | 1.33 | **0.86** | 0.89 | 0.89 | 3.58 | 0.88 |
| 5  | 1.47 | **1.15** | 1.59 | 1.58 | 3.23 | 1.53 |
| 7  | 1.40 | **1.06** | 1.39 | 1.39 | 1.63 | 1.39 |
| 10 | 1.34 | 0.89 | **0.87** | **0.87** | 1.69 | 0.87 |
| 14 | 1.46 | **1.08** | 1.23 | 1.39 | 2.87 | 1.21 |

## Reading

**Still a negative result, and it holds up under both follow-ups.** The classical scalar-index
baseline beats the embedding probe at 4 of 5 lead times for both targets, tying only at lead 10
(where embeddings edge ahead by a hair for *both* Ap and Kp — the one place worth a second look,
though on a single 156-row test fold it's a thin basis for a claim). Switching the target from Ap
to Kp reproduces the exact same pattern almost line for line, unsurprising since Ap is essentially
a linear rescaling of averaged Kp — it does not open up any signal Ap was hiding.

**Residual correction narrows the gap but never closes it.** Letting the embeddings only explain
what the classical linear fit gets wrong — rather than compete with it head-on — helps at most
lead times (e.g. Ap lead 3: 8.62 -> 8.52; lead 5: 17.55 -> 17.30) but at lead 7 it's actually worse
than the standalone embedding probe (12.30 classical vs. 14.72 embed vs. 15.64 residual-corrected —
the residual model overfit noise in the classical fit's errors rather than a real pattern). No
lead time for either target sees the residual-corrected number beat classical outright. This is
useful negative evidence in itself: if embeddings held real complementary signal, framing the
problem as "explain the classical model's mistakes" (an easier, lower-variance target than the raw
series) should have surfaced it, and it didn't.

The nonlinear check (PCA + small MLP) remains substantially worse than every linear approach at
every lead time for both targets, including well below the naive persistence baseline — with 598
training rows against 1280 raw embedding dimensions, even after PCA compression the MLP does not
have enough data to generalize, and CV-selected PCA dims trend toward the high end of the grid
(n=100) where the linear Ridge probe stays low (n=5-50), consistent with the MLP chasing degrees
of freedom it can't afford.

Two candidate explanations, not mutually exclusive:
1. **Sample size.** 754 weekly points (598 train) against 1280-dim embeddings is a genuinely hard
   regime for any of these estimators — PCA+Ridge is the right conservative choice here, and even
   it is landing on low component counts (5-50) chosen by CV, suggesting most of the 1280 dims are
   uninformative or redundant for this specific downstream target.
2. **Signal isn't there at this resolution.** Surya's embedding may encode information at temporal
   or spatial resolution unhelpful for a single weekly scalar snapshot regressed onto Ap/Kp 3-14
   days out — the model's own evaluation (4-day-lead solar wind speed) is a much narrower, more
   dynamically-relevant target than geomagnetic activity indices at multi-day lead times.

**Remaining untried lever: denser-than-weekly embedding sampling** (more effective training rows —
the one variable neither the original probe nor these two follow-ups changed, and the sample-size
explanation above predicts it should matter most). That requires a new Kaggle extraction pass, not
just reanalysis, so it's the natural next step if this is worth pursuing further — Dst as a third
target was considered but dropped for now since it needs a new external data source not already
cached locally.

## Data coverage note

`embeddings_merged.npz` covers 754 of 764 expected weekly slots (2010-05-16 to 2024-12-29,
98.7%). Missing weeks (10, sporadic 2012-2023) are genuine SDO/Surya-benchmark source gaps
("No matching files found for requested date range"), not a pipeline bug — confirmed by a
dedicated re-run (`heliomag-embedding-extraction` v8, "phase 2b") that re-attempted exactly this
gap plus all of 2025: it re-fetched the 2023-2024 portion successfully but every value duplicated
data already in `embeddings_merged.npz`, and all 52 weekly attempts in 2025 failed identically
(`Failed listing s3://nasa-surya-bench/2025/...`) — the NASA Surya benchmark bucket does not yet
mirror 2025 data. 2010-2024 weekly coverage is as complete as the source allows; 2025 is blocked
upstream, not locally.
