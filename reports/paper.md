# Reading the Sun for warnings Earth's own numbers miss

**A head-to-head test of Surya embeddings against classical solar indices for
geomagnetic-activity forecasting at 3 to 14 day lead times.**

Status: complete, negative result. Work carried out 2026-07-30 to 2026-08-19.

---

## Abstract

Geomagnetic storms are forecast operationally from scalar summaries of solar
activity, principally sunspot number (SN) and the F10.7 cm radio flux. Both
reduce the whole solar disk to one number per day and are therefore blind to
where features sit on it. Coronal holes, which drive a large share of
geomagnetic activity at multi-day lead times through high-speed solar wind
streams, are a spatial phenomenon these indices cannot represent by
construction.

Surya, NASA and IBM's heliophysics foundation model, reads SDO extreme
ultraviolet imagery directly. Its published evaluation stops at a single
4-day-lead solar wind speed forecast benchmarked only against other neural
networks. It never compares against classical indices, never predicts a
geomagnetic index, and never tests multiple lead times.

This work builds that missing comparison. We extract 1,941 Surya embeddings
covering 2010 to 2024, probe them against the GFZ Potsdam Ap and Kp record at
lead times of 3, 5, 7, 10 and 14 days, and score them against a classical linear
baseline on identical rows and an identical chronological split.

**The classical baseline wins at every lead time, for both targets, with no
exceptions.** Tripling the training set from 598 to 1,785 rows did not flip the
result, though it narrowed the lead-5 gap by 86 to 91 percent, which is real
evidence that sample size was a partial constraint.

One caveat qualifies the scope of this conclusion, and it was identified after
the analysis was complete. The extraction step reduces Surya's
`[1, 65536, 1280]` token tensor to a single 1280-dimensional vector by
`tokens.mean(dim=1)`, a global mean across the entire solar disk. That operation
discards exactly the spatial structure the hypothesis is about. The result is
therefore sound as a statement about this pipeline, and overstated as a
statement about Surya. See [limitations.md](limitations.md) and
[future-work.md](future-work.md).

---

## 1. Motivation

Two measured facts define the opportunity this project set out to test.

**Classical indices carry a real but small signal.** Across the full 1932 to
2026 GFZ record, SN and F10.7 correlate with Ap at r between 0.15 and 0.19.
That is statistically unambiguous at n in the tens of thousands, but it explains
only about 2 to 3.5 percent of daily Ap variance.

**Persistence decays quickly.** Ap autocorrelation is r = 0.461 at 1 day and
falls to r = 0.100 by day 3, dropping below the classical-index level between
day 3 and day 4.

That crossover defines the window this project targets. From 3 days out,
persistence is spent and classical indices are the bar to clear, yet the bar is
low. If a model with a spatial view of the Sun carries information those scalars
throw away, this is where it should show.

The physical reason to expect it is specific. Coronal holes are long-lived and
rotate with the Sun on a roughly 27-day Carrington period, so their geomagnetic
signature is periodic rather than sporadic. F10.7 and SN track active-region
emission, not coronal-hole geometry. If they are blind to it, the 27-day
component should survive essentially untouched in a classical model's residuals.
That is directly testable, and it was tested before any GPU time was committed.

## 2. Data

**Ground truth.** GFZ Potsdam combined Kp/Ap/SN/F10.7 record, 1932 to present,
CC BY 4.0, fetched directly by `scripts/fetch_data.py` and cached locally. Kp is
used as the daily mean of the eight 3-hourly values.

**Imagery.** SDO EUV active-region imagery, 2010 to 2024, accessed through the
NASA Surya benchmark S3 bucket during extraction. Images are not stored locally,
only the resulting embeddings.

**Embeddings.** `data/embeddings_merged.npz`, 1,941 vectors of 1,280 dimensions.
Weekly cadence from 2010-05 to 2024-12, with the training window additionally
densified to every 3 days, giving 1,785 training rows. Weekly coverage is 754 of
764 expected slots, 98.7 percent. The 10 missing weeks are genuine SDO source
gaps rather than pipeline failures, confirmed by a dedicated re-run that
reproduced them exactly. All 52 weekly attempts in 2025 failed identically
because the benchmark bucket does not yet mirror 2025 data.

## 3. Methods

### 3.1 Classical baseline

Features: SN, F10.7, Ap(t), and Ap 3-day and 7-day rolling means. Target is
Ap(t + lead). Chronological split, training on or before 2021-12-31 and testing
from 2022-01-01. Four models are compared: persistence, linear SN+F10.7, linear
SN+F10.7+persistence, and a small MLP on all features.

### 3.2 Spectral gate

A periodogram on the classical model's residual at lead 14, measuring the share
of spectral power falling in the 25 to 29 day band. Run before committing to
extraction, as a cheap test of whether the motivating structure survives the
classical fit.

### 3.3 Embedding extraction

Surya is run with `finetune=True` to expose token embeddings. The pipeline was
validated end to end on a Colab T4 before any batch work, with the real tensor
shape confirmed as `[1, 65536, 1280]`. Batch extraction ran as Kaggle GPU
sessions rather than one long job, because a single weekly pass exceeds Kaggle's
own session time cap. Each date's tensor is reduced to one 1,280-vector by
`tokens.mean(dim=1)`.

### 3.4 Probe design

PCA followed by Ridge regression, with PCA dimensionality in
{5, 10, 20, 30, 50, 75, 100} and Ridge alpha log-spaced from 1e-2 to 1e4,
selected per lead time by `GridSearchCV` over `TimeSeriesSplit(5)` on the
training set only, so no test data informs model selection. Three variants were
added: embeddings combined with classical features, PCA followed by a small MLP,
and a residual-correction model in which the embeddings predict what the
classical linear fit gets wrong rather than competing with it directly.

Classical features are recomputed at the embedding sample points so both sides
see identical rows. Absolute RMSE therefore differs from the daily-grid baseline
table, but the relative comparison is fair.

## 4. Results

### 4.1 Classical baseline, daily grid

RMSE in Ap units, lower is better.

| lead (days) | persistence | SN+F107 | SN+F107+persistence | MLP |
|---|---|---|---|---|
| 3  | 19.60 | 14.44 | **14.34** | 14.58 |
| 5  | 19.77 | 14.52 | **14.47** | 14.80 |
| 7  | 20.18 | 14.61 | **14.56** | 14.81 |
| 10 | 20.13 | 14.68 | **14.59** | 15.28 |
| 14 | 20.25 | 14.71 | **14.64** | 15.02 |

Linear SN+F107+persistence is strongest at every lead, holding 14.34 to 14.64
across the whole window. The MLP does not improve on it, which indicates the
classical relationship is close to linear with little nonlinear structure left
unexploited. This is the number the embeddings have to beat, and it is a solid
baseline rather than a strawman.

### 4.2 Spectral gate

| Signal | Power in the 25 to 29 day band |
|---|---|
| Raw Ap | 3.00% |
| Classical residual | 3.68% |

The residual retains slightly more of the 27-day band than the raw signal, not
less. The classical model removes essentially none of the recurrent component,
so whatever a spatial view of coronal holes could contribute would not be
redundant with F10.7 and SN. The motivating hypothesis survived its gate and
extraction proceeded.

### 4.3 Embedding probe, densified training set

Held-out RMSE, classical linear against PCA+Ridge on embeddings, 1,785 training
rows and 156 test rows.

**Target Ap**

| lead (days) | classical | embed | embed+classical | embed MLP | residual-corr |
|---|---|---|---|---|---|
| 3  | **8.43**  | 8.70  | 8.67  | 23.76 | 8.56  |
| 5  | **15.32** | 15.59 | 15.57 | 19.46 | 15.58 |
| 7  | **12.34** | 14.46 | 14.21 | 13.75 | 14.60 |
| 10 | **8.41**  | 9.00  | 9.00  | 12.00 | 9.15  |
| 14 | **12.36** | 12.92 | 12.88 | 16.91 | 12.64 |

**Target Kp**

| lead (days) | classical | embed | embed+classical | embed MLP | residual-corr |
|---|---|---|---|---|---|
| 3  | **0.87** | 0.94 | 0.93 | 1.66 | 0.89 |
| 5  | **1.10** | 1.14 | 1.15 | 1.30 | 1.16 |
| 7  | **1.08** | 1.43 | 1.44 | 1.44 | 1.49 |
| 10 | **0.88** | 1.03 | 1.03 | 1.10 | 1.05 |
| 14 | **1.08** | 1.27 | 1.26 | 1.79 | 1.23 |

Classical wins all ten comparisons.

### 4.4 What changed when the training set tripled

The first pass used weekly embeddings only, 598 training rows, and showed two
apparent ties at lead 10 where the embedding probe edged ahead for both Ap and
Kp. Both were flagged at the time as thin single-fold evidence. Densifying to
1,785 rows reversed both. Ap lead 10 moved from embeddings ahead by 0.16 RMSE to
classical ahead by 0.58, and Kp lead 10 did the same. **That lead-10 finding is
retracted.**

Three things moved in informative ways:

- **Lead 5 narrowed sharply.** Ap went from a 1.89 RMSE gap to 0.27, an 86
  percent reduction. Kp went from 0.45 to 0.04, 91 percent. This is the
  strongest evidence that sample size was a genuine constraint rather than a
  convenient excuse.
- **Other leads barely moved,** mostly drifting slightly in classical's favour.
  The pattern is mixed, not a uniform "more data helps" story.
- **The MLP stopped being catastrophic.** At Ap lead 7 it went from 22.16 to
  13.75, beating both linear embedding variants while still falling short of
  classical. More rows helped the model class that needed them most.

## 5. Discussion

The headline claim is narrow and it holds within its scope: **a PCA+Ridge probe
on globally mean-pooled Surya embeddings does not beat a linear
SN+F10.7+persistence baseline for Ap or Kp at 3 to 14 day lead times on this
data.** Four independent attempts to find the signal all failed, including the
residual-correction framing, which sets an easier and lower-variance target and
should have surfaced complementary signal if any existed.

Two explanations remain live and they are not mutually exclusive.

**Sample size.** 1,785 rows against 1,280 raw dimensions is a hard regime. The
cross-validation consistently selects low PCA component counts, between 5 and
50, which suggests most dimensions are uninformative or redundant for this
target. The lead-5 narrowing and the MLP's recovery both indicate this was a
real partial constraint.

**Representation.** This is the explanation the present work is least able to
rule out, and the reason lies in a design choice inside the extraction step
rather than anything in the analysis. `tokens.mean(dim=1)` averages over all
65,536 spatial tokens. The project's entire rationale is that Surya can see
coronal-hole geometry that scalar indices discard. A global mean over the disk
discards that geometry as well. What survives is a 1,280-dimensional description
of the Sun's average appearance, which is conceptually much closer to a scalar
index than to a spatial map.

Put plainly: the comparison as built may have handed the embeddings the same
handicap it was designed to expose in the classical indices. The negative result
stands for the pipeline as run. It does not establish that Surya lacks the
signal, and this report should not be read as claiming that it does.

## 6. Limitations and next steps

Full treatment in [limitations.md](limitations.md) and
[future-work.md](future-work.md). In short, the cheapest informative next step
costs no GPU time at all: run the same periodogram used in the spectral gate on
the existing 1,280-dimensional embeddings. If the mean-pooled embedding carries
little 27-day power while Ap carries 3.00 percent, that is direct evidence the
pooling removed the structure under test, and it justifies a re-extraction that
keeps a coarse spatial grid.

## 7. Reproducibility

Every number above traces to a script and a saved writeup in `results/`, not to
a notebook cell run once and forgotten. The first three stages run on CPU with
`requirements.txt`. The embedding probe requires `data/embeddings_merged.npz`,
produced by the Kaggle extraction notebook, which needs a GPU.

```bash
pip install -r requirements.txt
python scripts/fetch_data.py        # pilot correlations, caches the GFZ record
python scripts/baseline_probe.py    # classical baseline
python scripts/regime_analysis.py   # 27-day spectral check
python scripts/embedding_probe.py   # embeddings vs classical (needs the .npz)
```

## 8. Data and code availability

- GFZ Potsdam Kp/Ap/SN/F10.7, 1932 to present, CC BY 4.0:
  https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt
- Surya, NASA and IBM heliophysics foundation model, run via
  `kaggle/heliomag-embedding-extraction.ipynb` and validated in
  `notebooks/surya_pipeline_check.ipynb`.
- Findings page: https://natalimuca.github.io/heliomag/

## Supporting reports

- [background.md](background.md), what this work is based on and the gap it fills
- [engineering.md](engineering.md), the build log, the fixes and what went wrong
- [limitations.md](limitations.md)
- [future-work.md](future-work.md)
