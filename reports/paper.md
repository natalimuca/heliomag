# Reading the Sun for warnings Earth's own numbers miss

**A head-to-head test of Surya embeddings against classical solar indices for
geomagnetic-activity forecasting at 3 to 14 day lead times.**

Status: complete. Negative result. Work carried out 2026-07-30 to 2026-08-19.

Contents:
[1 Introduction](#1-introduction) ·
[2 Data](#2-data) ·
[3 Methods](#3-methods) ·
[4 Results](#4-results) ·
[5 Discussion](#5-discussion) ·
[6 Threats to validity](#6-threats-to-validity) ·
[7 Reproducibility](#7-reproducibility) ·
[Appendices](#appendix-a-exact-hyperparameter-grids)

---

## Abstract

Geomagnetic storms are forecast operationally from scalar summaries of solar
activity, principally the sunspot number (SN) and the F10.7 cm radio flux. Both
reduce the whole solar disk to one number per day and are therefore blind, by
construction, to where features sit on it. Coronal holes, which drive a large
share of geomagnetic activity at multi-day lead times through high-speed solar
wind streams, are a spatial phenomenon these indices cannot represent.

Surya, NASA and IBM's heliophysics foundation model, reads SDO extreme
ultraviolet imagery directly. Its published evaluation stops at a single
4-day-lead solar wind speed forecast benchmarked only against other neural
networks. It never compares against classical indices, never predicts a
geomagnetic index, and never tests multiple lead times.

This work builds that missing comparison. We extract 1,941 Surya embeddings
covering 2010-05 to 2024-12, probe them against the GFZ Potsdam Ap and Kp record
at lead times of 3, 5, 7, 10 and 14 days, and score them against a classical
linear baseline on identical rows under an identical chronological split. Five
model families are compared at each lead time: persistence, classical linear
regression, PCA and Ridge on embeddings, PCA and Ridge on embeddings combined
with classical features, PCA and a small MLP on embeddings, and a
residual-correction model in which embeddings predict only what the classical
fit gets wrong.

**The classical baseline wins at every lead time, for both targets, with no
exceptions.** Tripling the training set from 598 to 1,785 rows did not flip the
result. It did narrow the lead-5 gap by 86 to 91 percent and rescued the MLP
variant from catastrophic failure, which is real evidence that sample size was a
partial constraint. One earlier finding, an apparent embedding win at lead 10 in
the weekly-only pass, reversed under more data and is formally retracted here.

One caveat qualifies the scope of the conclusion, and it was identified after
the analysis was complete. The extraction step reduces Surya's
`[1, 65536, 1280]` token tensor to a single 1,280-dimensional vector by
`tokens.mean(dim=1)`, a global mean across all spatial tokens. That operation
discards exactly the spatial structure the hypothesis is about. The result is
therefore sound as a statement about this pipeline and overstated as a statement
about Surya. See [limitations.md](limitations.md) and
[future-work.md](future-work.md).

---

## 1. Introduction

### 1.1 The operational problem

Geomagnetic storms disturb Earth's magnetic field and degrade satellite
operations, GPS accuracy and power distribution. The disturbance has been
measured continuously since 1932 as the planetary Kp index, from which the
linearised Ap index is derived. That record is one of the longest uninterrupted
geophysical time series in existence, which is the reason this comparison is
possible at all: the ground-truth side is never sample-limited.

Forecasting that disturbance several days ahead is the hard part, and the
standard predictors are scalar solar indices.

### 1.2 Why scalar indices should be beatable

The limitation of SN and F10.7 is structural rather than empirical. Both track
active-region emission integrated over the visible disk. A large share of
geomagnetic activity in the multi-day window instead comes from high-speed solar
wind streams emitted by coronal holes, which are dark, low-density regions whose
geoeffectiveness depends on where they sit on the disk and when solar rotation
carries them into an Earth-facing position. No scalar summary of total emission
can express that geometry.

Two consequences follow, and both were measured rather than assumed:

1. Coronal-hole activity is **periodic**, because the holes are long-lived and
   the Sun rotates on a roughly 27-day Carrington period. Flare and CME driven
   storms are sporadic by comparison.
2. If classical indices are blind to that geometry, a classical model's
   **residual should retain the 27-day component essentially intact**.

Point 2 is falsifiable, cheap to test, and was tested before any GPU time was
committed. See section 4.4.

### 1.3 The gap in Surya's own evaluation

Surya reads SDO EUV imagery directly, at whatever spatial structure the imagery
contains. Its published evaluation is narrow in three specific ways, and each one
is a gap this work fills:

| Surya's published evaluation | This work |
|---|---|
| A single 4-day lead time | Five lead times, 3 to 14 days |
| Solar wind speed as target | Ap and Kp, the operational geomagnetic indices |
| Benchmarked against other neural networks | Benchmarked against classical scalar indices |

The third is the decisive one. A foundation model outperforming other neural
networks says nothing about whether it outperforms the cheap scalar that
operational forecasting actually uses. That comparison had not been made.

### 1.4 Choosing the lead-time window

The 3 to 14 day window was derived, not assumed. It follows from where
persistence stops dominating. Ap autocorrelation is r = 0.461 at 1 day and falls
to r = 0.100 by day 3, dropping below the classical-index correlation level
(r roughly 0.15 to 0.19) between day 3 and day 4. Before day 3 there is nothing
to add, because yesterday's value already carries the information. After day 3
the bar is low but non-trivial. Full tables in section 4.1 and 4.2.

### 1.5 Contributions

1. The first head-to-head comparison of Surya embeddings against classical solar
   indices for geomagnetic-activity prediction.
2. Five lead times and two targets, on identical rows and an identical split, so
   the comparison is internally fair.
3. A solid rather than strawman classical baseline, established independently
   and shown not to be improvable by a nonlinear model on the same features.
4. A pre-registered physical rationale (the 27-day recurrence check) gating the
   expensive stage.
5. A negative result reported as-is, including a formal retraction of an earlier
   sub-finding that did not survive more data.
6. Identification of a pooling confound that bounds how far the negative result
   generalises.

---

## 2. Data

### 2.1 Geomagnetic ground truth

Source: GFZ Potsdam combined Kp/Ap/SN/F10.7 file, 1932-01-01 to present,
CC BY 4.0.

```
https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt
```

`scripts/fetch_data.py` fetches it directly, with no manual download step, and
caches to `data/gfz_kp_ap_sn_f107.csv`. Handling detail that matters for
reproducibility:

- The file is whitespace-delimited with 28 columns and `#` comment lines, parsed
  with an explicit column list rather than inferred headers.
- Sentinel values are converted to NaN before any arithmetic. `Ap` and `SN` use
  `-1`, `F107obs` uses `-1.0`, and the eight 3-hourly `Kp` columns use `-1.0`.
  Skipping this step silently biases every downstream mean.
- **Kp is derived, not read.** The file stores eight 3-hourly values per day
  (`Kp1` to `Kp8`). The daily Kp used throughout is their arithmetic mean.
- Only four columns survive into the cached frame: `Ap`, `Kp`, `SN`, `F107obs`.
  Note `F107obs` (observed) is used, not `F107adj` (adjusted to 1 AU).

Row counts differ per analysis because of NaN handling and lag windows. The
pilot correlations use n = 34,544 for SN against the full record and n = 28,366
for F10.7, which is smaller because F10.7 measurements only begin in 1947.

### 2.2 Solar imagery

SDO EUV active-region imagery, 2010 to 2024, accessed through the NASA Surya
benchmark S3 bucket (`s3://nasa-surya-bench/`) during extraction. Images are
never stored locally. Only the resulting embedding vectors are kept.

### 2.3 Embeddings

`data/embeddings_merged.npz`, 1,941 vectors of 1,280 dimensions, keyed by date
string `YYYYMMDD`.

| Property | Value |
|---|---|
| Cadence, full span | Weekly, 2010-05-16 to 2024-12-29 |
| Cadence, training window | Additionally densified to every 3 days |
| Reference time of day | 12:00:00 UTC |
| Total embeddings | 1,941 |
| Training rows | 1,785 |
| Test rows | 156 |
| Weekly grid coverage | 754 of 764 slots, 98.7 percent |

Gap accounting is in [Appendix C](#appendix-c-coverage-and-gap-accounting). In
summary, the missing dates are genuine SDO source holes rather than pipeline
failures, and this was confirmed by a dedicated re-run rather than assumed.

---

## 3. Methods

### 3.1 Feature construction

The classical feature set is five columns, defined in
`scripts/baseline_probe.py::make_features`:

| Feature | Definition |
|---|---|
| `SN` | Sunspot number at time t |
| `F107` | F10.7 observed flux at time t |
| `Ap_t` | Ap at time t (the persistence term) |
| `Ap_3d_mean` | 3-day trailing rolling mean of Ap |
| `Ap_7d_mean` | 7-day trailing rolling mean of Ap |

The target is `Ap(t + lead)` for lead in {3, 5, 7, 10, 14}, produced by
`df["Ap"].shift(-lead)`. Rows with any NaN are dropped after joining features to
target, so rolling-window warm-up and record gaps are handled uniformly.

For the Kp runs, the same construction substitutes Kp for Ap in the three
persistence-derived columns and in the target, so `Kp_t`, `Kp_3d_mean`,
`Kp_7d_mean`.

### 3.2 Train and test split

A single chronological split, used identically in every stage:

```
train:  <= 2021-12-31
test:   >= 2022-01-01
```

Chronological rather than random, because the series is autocorrelated and a
random split would leak near-duplicate neighbouring days across the boundary.
The classical baseline additionally restricts to the SDO era, `.loc["2010-05-01":]`,
so it is scored on the same period the embeddings cover rather than on the full
1932 record. Sanity guards refuse to score a lead time with fewer than 100
training or 50 test rows.

### 3.3 Classical baseline models

Four models per lead time, from `scripts/baseline_probe.py`:

| Model | Features | Estimator |
|---|---|---|
| Persistence | none, predicts `Ap(t+lead) = Ap(t)` | identity |
| Classical scalars | `SN`, `F107` | `LinearRegression` |
| Classical + persistence | all five | `LinearRegression` |
| Nonlinear check | all five | `MLPRegressor(hidden_layer_sizes=(32,16), max_iter=2000, random_state=0)`, on standardised features |

The MLP exists to answer one question: is the classical relationship leaving
nonlinear structure unexploited? Only the MLP is standardised, because linear
regression is scale invariant for these purposes while the MLP is not.

### 3.4 Spectral gate

`scripts/regime_analysis.py`, run at lead 14, the longest tested lead and
therefore the one with most room for recurrent structure to separate from
persistence.

Method: fit the five-feature linear model on the training window, predict the
test window, take the residual `actual - predicted`, subtract its mean, and run
`scipy.signal.periodogram(signal, fs=1.0)` with fs in samples per day. Convert
frequencies to periods, mask to 25 to 29 days, and report banded power over
total power. The same measurement is applied to the raw test-period Ap for
comparison.

The decision rule was fixed in code before the result was known:

```python
if frac_resid > 0.5 * frac_raw:
    # classical indices are NOT explaining the recurrent component -> proceed
```

### 3.5 Surya embedding extraction

Validated first, then batched. Both stages are in the repository.

**Validation** (`notebooks/surya_pipeline_check.ipynb`, Colab T4, 2026-07-30).
Part A runs the official `easy_inference` quickstart on a deliberately tiny
window, 2014-10-23 10:00 to 13:00 with `rollout_steps: 1`, to confirm that
weight download, SDO data download and the GPU forward pass all work end to end.
It produced `prediction.nc`. Part B then sets `model.finetune = True`. In
`surya/models/helio_spectformer.py`, `HelioSpectFormer.forward()` takes an
`if self.finetune: return tokens` branch immediately after
`tokens = self.backbone(tokens)`, returning raw backbone tokens and skipping the
pixel-space decoder entirely. This produced a real tensor of shape
`[1, 65536, 1280]`, that is (batch, spatial tokens, embedding dimension).

**Batch extraction** (`kaggle/heliomag-embedding-extraction.ipynb`, Kaggle GPU).
Per reference timestamp:

1. Compute the input window from `advanced['time_delta_input_minutes']`.
2. `ezi.download_surya_bench_range(...)` pulls the required SDO frames from
   `s3://nasa-surya-bench/` at the configured cadence, with a match tolerance in
   minutes for imperfect timestamps.
3. `ezi.build_index_csv_for_range(...)` writes an index, which is then loaded
   and **indexed by `timestep`**. This detail cost real debugging time:
   `InputOnlyRolloutDataset` builds its `path_lookup` from
   `present_index["path"].to_dict()`, so a plain `RangeIndex` with timestep as a
   column produces silently wrong lookups.
4. Build `InputOnlyRolloutDataset` with `prediction_steps=1`, the config's
   `sdo_channels`, the config's scalers and its `pooling` setting.
5. Forward pass under `torch.no_grad()`, giving the token tensor.
6. **Reduce to one vector:** `pooled = tokens.mean(dim=1).squeeze(0).cpu().numpy()`.
7. Delete intermediates, `gc.collect()`, and remove the per-timestamp download
   directory.

Three calibration timestamps were run first, spread across the solar cycle:
2011-06-15, 2018-06-15 and 2024-06-15, all at 12:00. Batch runs checkpoint by
writing the full `.npz` every 10 successful extractions and clearing
`/kaggle/temp` every 20, and they resume by loading a previous merged `.npz` and
skipping keys already present. Disk free space on three mounts is logged at every
checkpoint, because disk exhaustion was a live risk.

Note that `pooling` passed to the dataset is Surya's own input-side setting from
its config. It is unrelated to the `tokens.mean(dim=1)` reduction in step 6,
which is this project's own choice and is the subject of the caveat in section
5.4.

### 3.6 Probe models

`scripts/embedding_probe.py`, five estimators per lead time per target.

| Name | Input | Pipeline |
|---|---|---|
| Persistence | `{target}_t` | identity |
| Classical | 5 classical columns | `LinearRegression` |
| Embed | 1,280 embedding columns | `StandardScaler` → `PCA` → `Ridge` |
| Embed + classical | both blocks | `ColumnTransformer`: embeddings through `StandardScaler` → `PCA`, classical through `StandardScaler`, then `Ridge` |
| Embed MLP | 1,280 embedding columns | `StandardScaler` → `PCA` → `MLPRegressor((32,16), max_iter=2000, random_state=0)` |
| Residual correction | 1,280 embedding columns | fit `StandardScaler` → `PCA` → `Ridge` on `y_train - classical_pred_train`, then predict `classical_pred_test + resid_pred_test` |

Note the asymmetry in the combined model: only the embedding block is
dimensionally reduced. The five classical columns pass through standardisation
alone, so PCA can never discard them.

### 3.7 Model selection protocol

Hyperparameters are chosen by `GridSearchCV` over `TimeSeriesSplit(n_splits=5)`,
scoring `neg_root_mean_squared_error`, fitted **on the training set only**. The
test window is never seen during selection. Grids are in
[Appendix A](#appendix-a-exact-hyperparameter-grids).

The residual-correction model runs its own independent CV against the residual
target rather than reusing the standalone embedding model's hyperparameters,
because the optimal regularisation for a residual is not the optimal
regularisation for the raw series.

### 3.8 Aligning classical features to embedding dates

The embedding grid is weekly or 3-daily, while the GFZ record is daily. Classical
features are recomputed at the embedding sample points using
`reindex(index, method="ffill")`, so both sides of the comparison see exactly the
same rows. Rolling means are computed on the **daily** series first and only then
reindexed, which preserves their intended 3-day and 7-day meaning rather than
turning them into 3-sample and 7-sample windows on a weekly grid.

Targets are built by shifting dates rather than rows:

```python
target_dates = emb.index + pd.Timedelta(days=lead)
target = gfz[target_col].reindex(target_dates)
```

This matters on an irregular index. A row shift would mean "lead sampling steps
ahead", which on a weekly grid is 3 to 14 weeks rather than days.

**Consequence for reading the tables.** Because the embedding probe scores on a
coarser and smaller sample than the daily baseline, its absolute RMSE values
differ from section 4.3. Only within-table comparisons are meaningful.

### 3.9 Metric

Root mean squared error on the held-out window,
`mean_squared_error(y_test, pred) ** 0.5`, in native units. Ap is dimensionless
and roughly 0 to 400. Kp is roughly 0 to 9. RMSE is not comparable across the
two targets.

---

## 4. Results

### 4.1 Pilot: do classical indices correlate with Ap at all?

Pearson r between each index and Ap, at lags in days.

| lag | SN, full record (n=34,544) | SN, SDO era (n=5,934) | F10.7, full record (n=28,366) | F10.7, SDO era (n=5,922) |
|---|---|---|---|---|
| 0 | 0.158 | 0.155 | 0.168 | 0.168 |
| 1 | 0.162 | 0.153 | 0.175 | 0.174 |
| 2 | 0.165 | 0.156 | 0.184 | 0.180 |
| 3 | 0.165 | 0.156 | 0.183 | 0.188 |
| 5 | 0.162 | 0.153 | 0.170 | 0.167 |
| 7 | 0.155 | 0.142 | 0.162 | 0.150 |
| 14 | 0.132 | 0.111 | 0.141 | 0.123 |

All p-values are effectively zero at these sample sizes. The effect is real and
robust but small: r of 0.15 to 0.19 corresponds to roughly 2 to 3.5 percent of
daily Ap variance explained. F10.7 is consistently slightly stronger than SN.
The full record and the SDO era agree closely, which rules out the correlation
being an artifact of one epoch.

This is consistent with the stated physics. F10.7 and SN track active-region
emission, not coronal-hole geometry.

### 4.2 Persistence decay

Ap autocorrelation, SDO era.

| lag (days) | r |
|---|---|
| 1 | 0.461 |
| 2 | 0.173 |
| 3 | 0.100 |
| 4 | 0.081 |
| 7 | 0.065 |

Persistence dominates at 1 day, collapses by a factor of 2.7 by day 2, and falls
below the classical-index level between day 3 and day 4. This defines the 3 to
14 day target window.

### 4.3 Classical baseline, daily grid

RMSE in Ap units. Lower is better. Bold marks the best model at each lead.

| lead (days) | persistence | SN+F107 | SN+F107+persistence | MLP (all features) |
|---|---|---|---|---|
| 3  | 19.60 | 14.44 | **14.34** | 14.58 |
| 5  | 19.77 | 14.52 | **14.47** | 14.80 |
| 7  | 20.18 | 14.61 | **14.56** | 14.81 |
| 10 | 20.13 | 14.68 | **14.59** | 15.28 |
| 14 | 20.25 | 14.71 | **14.64** | 15.02 |

Three readings:

- **Persistence is already beaten at lead 3**, by roughly 5.2 RMSE. This confirms
  the correlation-decay result in forecast-skill terms rather than in correlation
  terms.
- **Linear SN+F107+persistence is strongest at every lead**, and remarkably flat
  across the window, 14.34 to 14.64. Adding the persistence terms to the scalar
  indices buys only 0.05 to 0.10 RMSE, so nearly all the skill comes from SN and
  F10.7.
- **The MLP never wins**, and degrades as lead increases. The classical
  relationship is close to linear, with no obvious nonlinear structure being left
  on the table by a simple regression.

That last point is what makes this a fair bar rather than a strawman. The
classical side has been given a nonlinear model on the same features and did not
benefit.

### 4.4 Spectral gate

Lead 14, test period 2022 to 2025, n = 1,653, classical baseline RMSE 14.64.

| Signal | Power in the 25 to 29 day band |
|---|---|
| Raw Ap (test period) | 3.00% |
| Residual, actual minus classical prediction | 3.68% |

The residual retains **slightly more** 27-day power than the raw signal, not
less. The classical model removes essentially none of the recurrent component.
If it had removed any, the residual fraction would sit well below the raw
fraction.

This is the result that justified the GPU spend. Whatever a spatial view of
coronal holes could contribute would not be redundant with what F10.7 and SN
already capture, because they capture approximately none of it.

**Caveat carried from the source writeup.** 3 percent of total spectral power in
one band is a modest absolute signal. Most of Ap's variance is broadband,
because real storms are messier than a clean 27-day recurrence. This is a
periodogram ratio on a single window, with no Lomb-Scargle false-alarm
probability and no significance test. It is a cheap gate that the hypothesis
passed, not proof.

### 4.5 Embedding probe, weekly sampling (first pass)

754 samples, 598 train and 156 test, 1,280 dimensions.

**Target Ap**

| lead | persistence | classical | embed | embed+classical | embed MLP | residual-corr |
|---|---|---|---|---|---|---|
| 3  | 14.82 | **8.32**  | 8.62  | 8.62  | 19.96 | 8.52  |
| 5  | 19.36 | **15.66** | 17.55 | 17.20 | 25.79 | 17.30 |
| 7  | 16.96 | **12.30** | 14.72 | 13.88 | 22.16 | 15.64 |
| 10 | 14.77 | 8.62      | **8.47** | **8.47** | 17.66 | 8.72 |
| 14 | 17.69 | **12.44** | 13.09 | 13.09 | 39.80 | 13.10 |

**Target Kp**

| lead | persistence | classical | embed | embed+classical | embed MLP | residual-corr |
|---|---|---|---|---|---|---|
| 3  | 1.33 | **0.86** | 0.89 | 0.89 | 3.58 | 0.88 |
| 5  | 1.47 | **1.15** | 1.59 | 1.58 | 3.23 | 1.53 |
| 7  | 1.40 | **1.06** | 1.39 | 1.39 | 1.63 | 1.39 |
| 10 | 1.34 | 0.89     | **0.87** | **0.87** | 1.69 | 0.87 |
| 14 | 1.46 | **1.08** | 1.23 | 1.39 | 2.87 | 1.21 |

Classical wins 4 of 5 leads on both targets. Lead 10 showed embeddings ahead by
0.16 RMSE on Ap and 0.02 on Kp, and was written up at the time with an explicit
warning that a single 156-row test fold is thin evidence. Section 4.7 returns to
this.

The MLP is catastrophic in this pass, worse than naive persistence at every lead
on both targets, reaching 39.80 at Ap lead 14 against a classical 12.44.

### 4.6 Embedding probe, densified training set

1,785 train rows, a 3.0x increase. The test window is unchanged at 156 weekly
rows, so improvements cannot come from an easier test set.

**Target Ap**

| lead | classical | embed | embed+classical | embed MLP | residual-corr |
|---|---|---|---|---|---|
| 3  | **8.43**  | 8.70  | 8.67  | 23.76 | 8.56  |
| 5  | **15.32** | 15.59 | 15.57 | 19.46 | 15.58 |
| 7  | **12.34** | 14.46 | 14.21 | 13.75 | 14.60 |
| 10 | **8.41**  | 9.00  | 9.00  | 12.00 | 9.15  |
| 14 | **12.36** | 12.92 | 12.88 | 16.91 | 12.64 |

**Target Kp**

| lead | classical | embed | embed+classical | embed MLP | residual-corr |
|---|---|---|---|---|---|
| 3  | **0.87** | 0.94 | 0.93 | 1.66 | 0.89 |
| 5  | **1.10** | 1.14 | 1.15 | 1.30 | 1.16 |
| 7  | **1.08** | 1.43 | 1.44 | 1.44 | 1.49 |
| 10 | **0.88** | 1.03 | 1.03 | 1.10 | 1.05 |
| 14 | **1.08** | 1.27 | 1.26 | 1.79 | 1.23 |

**Classical wins all ten comparisons. No ties remain.**

### 4.7 What changed between the two passes

Differences quoted below come from unrounded values, so they can differ by 0.01
from the two-decimal tables above.

**The lead-10 finding is retracted.** Ap lead 10 moved from embeddings ahead by
0.16 RMSE to classical ahead by 0.58. Kp lead 10 moved from embeddings ahead by
0.02 to classical ahead by 0.14. Both reversals are larger than the original
margins. The flag placed on the original result was doing exactly its job, and
the retraction is recorded in `results/embedding_probe.md` rather than being
edited out.

**Lead 5 narrowed sharply.** The classical-to-embedding gap fell from 1.89 to
0.27 RMSE on Ap, an 86 percent reduction, and from roughly 0.45 to 0.04 on Kp,
about 91 percent. This is the single strongest piece of evidence that sample
size was a genuine constraint rather than a convenient excuse.

**Other leads were flat or slightly worse.** Ap leads 3, 7 and 14 and Kp leads
3, 7 and 14 all moved a little in classical's favour or stayed put. The pattern
is mixed, not a uniform "more data helps" story, which is why the honest
conclusion is "partial constraint" rather than "data-limited".

**The MLP stopped being broken.** At Ap lead 7 it went from 22.16 to 13.75,
beating both linear embedding variants for the first time while still short of
classical's 12.34. At Ap lead 14 it went from 39.80 to 16.91. The model class
that most needed rows benefited most from them, which is the expected signature
of a sample-size effect rather than a bug.

### 4.8 Hyperparameter selection behaviour

Cross-validation consistently selected **low PCA component counts for the linear
probe, in the 5 to 50 range**, from a grid extending to 100. The MLP variant
trended toward the high end, n = 100.

Two readings, both worth stating:

- Most of the 1,280 dimensions are uninformative or redundant **for this
  specific downstream target**. CV is discarding them, and it is doing so with
  the test set unseen, so this is not hindsight.
- The MLP reaching for maximum components while performing worst is the
  classic signature of a model chasing degrees of freedom it cannot afford at
  598 rows. Its recovery at 1,785 rows supports that reading directly.

---

## 5. Discussion

### 5.1 What the result establishes

Stated precisely: **a PCA and Ridge probe on globally mean-pooled Surya
embeddings does not beat a linear SN+F10.7+persistence baseline for Ap or Kp at
lead times of 3, 5, 7, 10 and 14 days on this data and this split.**

Four independent attempts to surface the signal all failed:

1. Embeddings standalone.
2. Embeddings concatenated with classical features, where PCA cannot discard the
   classical columns, so the combined model has strictly more information than
   the classical model alone and still does not beat it.
3. A nonlinear head on the embeddings.
4. Residual correction, which sets the easiest version of the task.

Point 2 deserves emphasis. The combined model contains the classical model as a
special case, up to regularisation. That it does not beat classical at any lead
means the embedding block contributes nothing the Ridge penalty considers worth
its variance cost.

### 5.2 Why the residual-correction null is informative

Residual correction asks the embeddings to explain only what the classical fit
gets wrong. That target has lower variance and no requirement to rediscover the
classical relationship. If the embeddings carried genuinely complementary
signal, this framing is where it should have appeared most easily.

It did not appear at any lead time on either target. On the weekly pass it was
actually **worse** than the standalone embedding probe at Ap lead 7, 15.64
against 14.72, which is the signature of a model fitting noise in the classical
model's errors rather than structure.

### 5.3 Sample size

1,785 rows against 1,280 raw dimensions is a hard regime for any estimator, and
the evidence that it mattered is concrete: the lead-5 collapse from 1.89 to 0.27
and the MLP's recovery from 22.16 to 13.75.

The evidence that it was not the whole story is equally concrete: most lead
times barely moved, and nothing crossed over.

The honest summary is that sample size was a real, partial constraint whose
remaining magnitude is unknown.

### 5.4 Representation: the pooling confound

This is the explanation the present work is least able to rule out, and the
reason lies in the extraction step rather than anywhere in the analysis.

```python
pooled = tokens.mean(dim=1).squeeze(0).cpu().numpy()
```

`tokens` has shape `[1, 65536, 1280]`, where dimension 1 is the spatial token
axis. 65,536 tokens is consistent with a 256 by 256 spatial grid. The operation
averages across every one of them, that is, across the entire solar disk.

The project exists because scalar indices compress the disk to a single number
and therefore cannot express coronal-hole geometry. A global mean over all
spatial tokens performs the same compression, just into 1,280 numbers rather
than one. What survives describes the Sun's average appearance, not where
anything is on it.

Put plainly: the comparison as built may have handed the embeddings the same
handicap it was designed to expose in the classical indices.

Three observations sharpen this rather than softening it:

- The motivating structure is **specifically spatial and specifically
  periodic**. A coronal hole matters because of where it is and when rotation
  carries it Earthward. Averaging over position is close to the worst possible
  reduction for that signal.
- The spectral gate in section 4.4 established that the 27-day component
  survives the classical fit. It never established that the component survives
  the pooling, and that check has not yet been run.
- CV selecting only 5 to 50 useful components out of 1,280 is consistent with a
  representation that has already had its informative structure averaged away.

**Scope of the conclusion.** The negative result stands for the pipeline as run.
It does not establish that Surya lacks the signal, and this report should not be
cited as if it did. The cheapest way to resolve this is a periodogram on the
existing embeddings, which needs no GPU and no new data. See
[future-work.md](future-work.md), step 1.

### 5.5 On reporting a negative result

The gating discipline described in section 1 and section 3.4 is what makes this
negative result worth reporting rather than a null pilot. The hypothesis had
already survived three independent cheap ways of being wrong before the
expensive test was run: classical indices do correlate with Ap, persistence does
decay fast enough to leave a window, and the 27-day structure does survive the
classical fit.

An earlier project iteration on NAO and tropospheric teleconnections was
abandoned when the usable sample turned out to be n = 16. The current design was
chosen for the opposite property.

---

## 6. Threats to validity

Summarised here, treated in full in [limitations.md](limitations.md).

| # | Threat | Severity |
|---|---|---|
| 1 | Global mean pooling discards the spatial structure under test | **High**, bounds the conclusion |
| 2 | Probe result, tests linear accessibility rather than presence | Medium |
| 3 | Sample size a partial, unresolved constraint | Medium |
| 4 | Single test window, 2022 onward, one phase of cycle 25 | Medium |
| 5 | Short embedding record, asymmetric train and test cadence | Low to medium |
| 6 | Spectral gate is a sanity check, not a significance test | Low, gate only |
| 7 | Ap and Kp are not independent evidence | Low, expected |

---

## 7. Reproducibility

Every number in this paper traces to a script and a saved writeup in `results/`,
not to a notebook cell run once and forgotten.

```bash
pip install -r requirements.txt

python scripts/fetch_data.py        # fetch and cache the GFZ record
python scripts/baseline_probe.py    # classical baseline, section 4.3
python scripts/regime_analysis.py   # 27-day spectral gate, section 4.4
python scripts/embedding_probe.py   # embeddings vs classical, sections 4.5 to 4.8
```

Dependencies are deliberately minimal: `pandas`, `numpy`, `scipy`, `requests`,
`scikit-learn`. Every random seed that matters is fixed, `random_state=0` on
both PCA and `MLPRegressor`.

The first three stages run anywhere on CPU. The fourth needs
`data/embeddings_merged.npz`, which is produced by the Kaggle notebook and
requires a GPU. Extraction is not reproducible locally.

---

## 8. Data and code availability

- **Geomagnetic record.** GFZ Potsdam combined Kp/Ap/SN/F10.7, 1932 to present,
  CC BY 4.0.
  https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt
- **Model.** Surya, NASA and IBM heliophysics foundation model,
  https://github.com/NASA-IMPACT/Surya, weights via `huggingface_hub`.
- **Imagery.** NASA Surya benchmark bucket, `s3://nasa-surya-bench/`.
- **Findings page.** https://natalimuca.github.io/heliomag/

---

## Appendix A: exact hyperparameter grids

```python
LEAD_TIMES = [3, 5, 7, 10, 14]
TRAIN_END  = "2021-12-31"
TEST_START = "2022-01-01"
TARGETS    = ["Ap", "Kp"]

PCA_GRID   = [5, 10, 20, 30, 50, 75, 100]
ALPHA_GRID = np.logspace(-2, 4, 13)     # 1e-2 ... 1e4, 13 points
N_SPLITS   = 5                          # TimeSeriesSplit
```

Search sizes per lead time per target: the embedding and combined and residual
models each search 7 x 13 = 91 configurations over 5 folds, so 455 fits each.
The MLP searches PCA only, 7 configurations over 5 folds, 35 fits. Scoring is
`neg_root_mean_squared_error`, parallelised with `n_jobs=-1`.

## Appendix B: extraction session log

| Session | Purpose | Outcome |
|---|---|---|
| Colab T4, 2026-07-30 | Pipeline validation, Parts A and B | `prediction.nc` written, embedding tensor `[1, 65536, 1280]` confirmed |
| Kaggle, weekly pass | 2010-05 to 2024-12 at weekly cadence | 754 of 764 slots |
| Kaggle v8 ("phase 2b") | Re-attempt the 10 weekly gaps plus all of 2025 | 2023-2024 portion re-fetched but every value duplicated existing data; all 52 attempts in 2025 failed identically |
| Kaggle v9 | Densify training window to every 3 days, 1,416 targets | Ran roughly 12 hours, auto-cancelled by Kaggle's session cap at about 95 percent |
| Kaggle v10 | Resume from v9 checkpoint | Remaining 71 targets completed in under an hour |

Final merged artifact: 1,941 embeddings, 1,785 train and 156 test.

## Appendix C: coverage and gap accounting

**Weekly grid.** 754 of 764 expected slots, 98.7 percent. The 10 missing weeks
are sporadic between 2012 and 2023.

**Are they real?** Session v8 re-attempted exactly those dates. The 2023 to 2024
portion re-fetched successfully but every returned value duplicated data already
held, and all 52 weekly attempts in 2025 failed identically with
`Failed listing s3://nasa-surya-bench/2025/...`. Conclusion: the gaps are genuine
SDO source holes, multi-day instrument outages, and 2025 is simply not mirrored
in the benchmark bucket yet.

**Densification.** 30 of 1,416 targeted training-window dates failed, 2.1
percent, on a session that completed rather than being cancelled. They cluster in
multi-day runs immediately around dates already known missing from the weekly
grid, for example around 2012-09, 2013-04, 2016-08 and 2018-12.

That clustering is the decisive evidence. A real multi-day outage appears as one
missing week at coarse resolution and a short burst of missing days at fine
resolution. An intermittent pipeline bug would scatter randomly instead.

## Appendix D: file map

| Path | Role |
|---|---|
| `scripts/fetch_data.py` | GFZ fetch, sentinel handling, Kp derivation, caching |
| `scripts/baseline_probe.py` | Feature construction, split constants, classical baseline |
| `scripts/regime_analysis.py` | 27-day periodogram gate |
| `scripts/embedding_probe.py` | All five probe models, CV protocol, both targets |
| `notebooks/surya_pipeline_check.ipynb` | Colab T4 end-to-end validation |
| `kaggle/heliomag-embedding-extraction.ipynb` | Batch GPU extraction, checkpointing, resume |
| `results/pilot_kp.md` | Section 4.1 and 4.2 numbers |
| `results/baseline_probe.md` | Section 4.3 numbers |
| `results/regime_analysis.md` | Section 4.4 numbers |
| `results/embedding_probe.md` | Sections 4.5 to 4.8 numbers |

## Supporting reports

- [background.md](background.md), the physics, the prior work, and what was
  rejected before this
- [engineering.md](engineering.md), the build log, every fix and what went wrong
- [limitations.md](limitations.md), what the result does not establish
- [future-work.md](future-work.md), four ranked next steps
