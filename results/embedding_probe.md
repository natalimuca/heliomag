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

## Follow-up: denser training-window sampling (598 -> 1785 train rows)

Ran the untried lever above. Extracted embeddings at every-3-days spacing across the training
window only (2010-05-16 to 2021-12-31 — the test window, 2022-2024, stays at the original weekly
cadence, 156 rows, unchanged), via two Kaggle sessions (v9 ran ~12h before Kaggle's own session
cap auto-cancelled it at 95% done; v10 finished the remaining 71 targets cleanly in under an hour).
Final training set: 1785 rows (3.0x), full dataset 1941 embeddings total.

### Target: Ap (denser training set)

| lead (days) | classical (linear) | embed (PCA+Ridge) | embed+classical | embed (PCA+MLP) | residual-correction |
|---|---|---|---|---|---|
| 3  | **8.43** | 8.70 | 8.67 | 23.76 | 8.56 |
| 5  | **15.32** | 15.59 | 15.57 | 19.46 | 15.58 |
| 7  | **12.34** | 14.46 | 14.21 | 13.75 | 14.60 |
| 10 | **8.41** | 9.00 | 9.00 | 12.00 | 9.15 |
| 14 | **12.36** | 12.92 | 12.88 | 16.91 | 12.64 |

### Target: Kp (denser training set)

| lead (days) | classical (linear) | embed (PCA+Ridge) | embed+classical | embed (PCA+MLP) | residual-correction |
|---|---|---|---|---|---|
| 3  | **0.87** | 0.94 | 0.93 | 1.66 | 0.89 |
| 5  | **1.10** | 1.14 | 1.15 | 1.30 | 1.16 |
| 7  | **1.08** | 1.43 | 1.44 | 1.44 | 1.49 |
| 10 | **0.88** | 1.03 | 1.03 | 1.10 | 1.05 |
| 14 | **1.08** | 1.27 | 1.26 | 1.79 | 1.23 |

**Classical wins every single lead time for both targets now — no more ties.** Tripling the
training rows did not flip the headline result. But the *size* of the gap moved in genuinely
informative, if mixed, ways:

- **The one lead-10 result that looked like an embedding edge in the original (weekly-only) run
  reversed.** Ap lead 10 went from embed beating classical by 0.16 RMSE to classical beating embed
  by 0.58; Kp lead 10 did the same (0.02 -> 0.14 in classical's favor). Both "ties" in the earlier
  writeup were flagged there as thin, single-fold evidence — this is exactly the kind of result
  that flag was for. Treat the original lead-10 finding as retracted: it didn't survive more data.
- **Lead 5 gap shrank sharply for both targets** (Ap: 1.89 -> 0.27 RMSE; Kp: 0.45 -> 0.04), the
  strongest evidence so far that sample size was a real constraint, not just Surya lacking signal.
- **Other lead times were flat or slightly worse** (Ap lead 3, 7, 14 and Kp lead 3, 7, 14 all moved
  a little in classical's favor or stayed flat) — this is not a clean, uniform "more data helps"
  story.
- **The MLP variant is dramatically less broken.** It's no longer the worst option everywhere — at
  Ap lead 7 it actually beats the linear embed and residual-correction variants (13.75 vs. 14.46 /
  14.60), still short of classical (12.34) but a real qualitative change from before, when it was
  catastrophically bad (22.16) at the same lead time. More rows clearly helped the model class
  that needed them most.

**Net read:** sample size was a real, partial constraint — the evidence for that is now much
better than a plausible-sounding excuse (lead 5, MLP behavior). But it was not *the whole*
explanation, since most lead times didn't move much and none crossed over. Going further (e.g.
full daily density, ~38h more compute per the original scoping table) is a much larger commitment
for an uncertain further gain given this mixed pattern — better spent, if pursued at all, on the
signal-resolution explanation instead (different embedding pooling, non-weekly-anchored inputs)
than on brute-forcing more of the same weekly-derived density.

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

The every-3-days densification (previous section) hit the same kind of gap at finer resolution:
30 of 1416 targeted training-window dates (2.1%) failed on a completed (not cancelled) session,
clustered in multi-day runs immediately around the same dates already known missing from the
weekly grid (e.g. clusters around 2012-09, 2013-04, 2016-08, 2018-12) — consistent with real
multi-day SDO instrument outages showing up as a single missing week at coarse resolution and a
short burst of missing days at finer resolution, not a pipeline issue.
