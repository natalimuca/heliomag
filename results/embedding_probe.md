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
{5,10,20,30,50,75,100}, alpha log-spaced 1e-2 to 1e4.

| lead (days) | persistence | classical (linear) | embed (PCA+Ridge) | embed+classical | embed (PCA+MLP) |
|---|---|---|---|---|---|
| 3  | 14.82 | **8.32** | 8.62 | 8.62 | 19.96 |
| 5  | 19.36 | **15.66** | 17.55 | 17.20 | 25.79 |
| 7  | 16.96 | **12.30** | 14.72 | 13.88 | 22.16 |
| 10 | 14.77 | 8.62 | **8.47** | **8.47** | 17.66 |
| 14 | 17.69 | **12.44** | 13.09 | 13.09 | 39.80 |

## Reading

**Negative result.** The classical scalar-index baseline (SN + F10.7 + Ap persistence/rolling
means, plain linear regression) beats the Surya-embedding probe at 4 of 5 lead times, and ties at
one (lead 10, by 0.02 RMSE — noise, not signal). Combining embeddings with classical features
gives at best a marginal improvement over embeddings alone, and never clears the classical-alone
number by more than rounding error — the model isn't finding complementary information to fuse,
it's just partially falling back on the classical features already in the mix.

The nonlinear check (PCA + small MLP) is substantially worse than either linear approach at every
lead time, including well below the naive persistence baseline — with 598 training rows against
1280 raw embedding dimensions, even after PCA compression the MLP does not have enough data to
generalize, and CV-selected PCA dims trend toward the high end of the grid (n=100) where the linear
Ridge probe stays low (n=5-50), consistent with the MLP chasing degrees of freedom it can't afford.

Two candidate explanations, not mutually exclusive:
1. **Sample size.** 754 weekly points (598 train) against 1280-dim embeddings is a genuinely hard
   regime for any of these estimators — PCA+Ridge is the right conservative choice here, and even
   it is landing on low component counts (5-50) chosen by CV, suggesting most of the 1280 dims are
   uninformative or redundant for this specific downstream target.
2. **Signal isn't there at this resolution.** Surya's embedding may encode information at temporal
   or spatial resolution unhelpful for a single weekly scalar snapshot regressed onto Ap 3-14 days
   out — the model's own evaluation (4-day-lead solar wind speed) is a much narrower, more
   dynamically-relevant target than geomagnetic Ap at multi-day lead times.

**This does not close the question**, but it means the naive "embed once weekly, PCA+regress"
approach does not clear the classical bar. Follow-ups worth trying before concluding negatively:
denser-than-weekly embedding sampling (more effective training rows), predicting Kp/Dst instead of
Ap, or using embeddings as a residual correction on top of the classical linear fit rather than a
standalone feature set.

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
