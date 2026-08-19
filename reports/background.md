# Background: what this work is based on

Why this comparison was worth building, what it rests on, what the quantities
actually mean, and what was rejected along the way.

Contents:
[The problem](#1-the-operational-problem) ·
[The quantities](#2-what-the-quantities-actually-are) ·
[Why scalars should be beatable](#3-why-scalar-indices-should-be-beatable) ·
[Surya](#4-surya-and-what-its-own-evaluation-leaves-open) ·
[The window](#5-choosing-the-lead-time-window) ·
[What was rejected](#6-what-was-rejected-before-this) ·
[Gating](#7-the-gating-discipline)

---

## 1. The operational problem

Geomagnetic storms disturb Earth's magnetic field. The practical consequences are
degraded satellite operations and orbit prediction, GPS positioning error,
induced currents in long conductors including power grids and pipelines, and HF
radio disruption. Forecasting them several days ahead has direct operational
value.

The disturbance has been measured continuously since 1932. That length is the
reason this project is possible at all: it provides ground truth at n in the tens
of thousands, which removes small-sample risk from the classical side of every
comparison in this repository.

## 2. What the quantities actually are

Being precise about these matters, because two of them are derived rather than
measured and one is used in a specific variant.

**Kp**, the planetary K index. A quasi-logarithmic measure of geomagnetic
disturbance derived from a network of ground magnetometer stations, reported
every 3 hours, so eight values per day, on a scale running 0 to 9. In this
project the daily value is the arithmetic mean of the day's eight 3-hourly
values, computed in `scripts/fetch_data.py`. It is not read from the file as a
daily figure, because the file does not contain one.

**Ap**, the daily planetary amplitude. A linearised counterpart to Kp, so it can
be averaged and regressed without the compression a logarithmic scale imposes.
Because Ap is close to a linear rescaling of averaged Kp, the two targets in this
work are not independent evidence, which is stated explicitly as limitation 7.

**SN**, the sunspot number. A count-based index of visible active regions,
integrated over the whole visible disk.

**F10.7**, solar radio flux at 10.7 cm, in solar flux units. A proxy for
chromospheric and coronal activity, also disk-integrated. The **observed**
variant `F107obs` is used throughout, not `F107adj`, which is adjusted to a
standard 1 AU Sun-Earth distance. The choice matters only for consistency, and
consistency is what was chosen.

The essential shared property of SN and F10.7: both reduce the entire visible
disk to a single number per day. Position information is not degraded by them, it
is absent from them.

## 3. Why scalar indices should be beatable

The limitation is structural rather than empirical, which is what makes it worth
attacking.

SN and F10.7 track **active-region emission**. A large share of geomagnetic
activity in the multi-day window instead comes from **high-speed solar wind
streams** emitted by **coronal holes**, which are dark, low-density, open-field
regions. Their geoeffectiveness depends on where they sit on the disk and when
solar rotation carries them into an Earth-facing position. No scalar summary of
total emission can express that geometry.

Two consequences follow, and this project measured both rather than assuming
them.

### 3.1 The signature should be periodic

Coronal holes are long-lived, often persisting for several solar rotations, and
the Sun rotates on a roughly 27-day Carrington period as seen from Earth. Their
geomagnetic signature is therefore **recurrent**. Flare and CME driven storms are
sporadic by comparison.

### 3.2 It should survive a classical fit

If classical indices are blind to coronal-hole geometry, then fitting a classical
model and examining its residual should leave the 27-day component essentially
intact. If instead the classical model were quietly capturing the recurrence by
some indirect route, the residual would show markedly less 27-day power than the
raw signal.

This is a falsifiable prediction with a cheap test, and the decision rule was
written into `scripts/regime_analysis.py` before the answer was known:

```python
if frac_resid > 0.5 * frac_raw:
    # classical indices are NOT explaining the recurrent component -> proceed
```

The measurement: raw Ap carries 3.00 percent of its spectral power in the 25 to
29 day band, and the classical residual carries 3.68 percent. The residual
retains slightly more, not less. The classical fit removes essentially none of
it, and the prediction held.

## 4. Surya, and what its own evaluation leaves open

Surya is NASA and IBM's heliophysics foundation model, developed in the
NASA-IMPACT organisation. It reads SDO extreme ultraviolet imagery directly, at
whatever spatial structure the imagery contains, which is exactly the information
a disk-integrated scalar discards.

### 4.1 Architecture points relevant here

Three facts from working with it directly, all confirmed in
`notebooks/surya_pipeline_check.ipynb` rather than taken from documentation:

- The model class is `HelioSpectFormer`, in
  `surya/models/helio_spectformer.py`.
- `forward()` contains an `if self.finetune: return tokens` branch immediately
  after `tokens = self.backbone(tokens)`. Setting `model.finetune = True` after
  construction returns raw backbone tokens and skips the pixel-space decoder
  entirely. This is the entire mechanism by which embeddings are obtained.
- One forward pass on one SDO input produces a tensor of shape
  `[1, 65536, 1280]`, that is (batch, spatial tokens, embedding dimension).
  65,536 tokens is consistent with a 256 by 256 spatial grid.

That last shape is the crux of the caveat that ended up bounding the whole
result. The spatial axis exists, carries 65,536 positions, and this project's
extraction step averaged it away. See limitation 1 in
[limitations.md](limitations.md).

### 4.2 The gap in its published evaluation

Surya's published evaluation stops at a single 4-day-lead solar wind speed
forecast, benchmarked only against other neural networks. It is narrow in three
specific ways, and each is a gap this work fills:

| Surya's published evaluation | This work |
|---|---|
| A single 4-day lead time | Five lead times, 3 to 14 days |
| Solar wind speed as target | Ap and Kp, the operational geomagnetic indices |
| Benchmarked against other neural networks | Benchmarked against classical scalar indices |

The third is decisive. A foundation model outperforming other neural networks
says nothing about whether it outperforms the cheap scalar that operational
forecasting actually uses. Solar wind speed is also a more dynamically proximate
target than a geomagnetic index at multi-day lead, so strong performance there
does not transfer by assumption.

Making that comparison is the contribution, regardless of which way the result
falls. A negative result about a real foundation model against a real operational
baseline is worth more than a positive result against a strawman.

## 5. Choosing the lead-time window

The 3 to 14 day window was derived from the data, not chosen for convenience.

**Ap autocorrelation, SDO era:**

| lag (days) | r |
|---|---|
| 1 | 0.461 |
| 2 | 0.173 |
| 3 | 0.100 |
| 4 | 0.081 |
| 7 | 0.065 |

**Classical index correlation with Ap:** r roughly 0.15 to 0.19 across lags 0 to
14, stable between the full 1932 record and the SDO era.

Persistence is overwhelming at 1 day, collapses by a factor of 2.7 by day 2, and
crosses below the classical-index level between day 3 and day 4.

That crossover is the window boundary:

- **Before day 3**, there is nothing to add. Yesterday's value already carries
  the information, and any model has to beat r = 0.461 rather than r = 0.16.
- **After day 3**, persistence is spent and classical indices are the bar. The
  bar is low, explaining only 2 to 3.5 percent of daily Ap variance, but it is
  statistically unambiguous and, as the baseline table shows, not trivial to
  beat in RMSE terms.
- **14 days** is the upper bound tested, chosen as the point where classical
  correlation has decayed noticeably (r 0.111 to 0.141) while still remaining
  clearly non-zero.

## 6. What was rejected before this

An earlier pilot studied NAO and tropospheric teleconnections. It was carried far
enough to establish that the usable sample was n = 16, and then abandoned rather
than written up.

The replacement was chosen specifically for the opposite property. The GFZ record
gives tens of thousands of daily rows, so the classical side of the comparison is
never sample-limited, even though the embedding side turned out to be partially
so.

This matters for reading the final result. The negative finding is **not** a null
pilot that ran out of data on both sides. The baseline it failed to beat is
measured on a large sample, holds 14.34 to 14.64 RMSE across the entire window,
and does not improve when given a nonlinear model on the same features.

**Dst** was also scoped as a possible third target and dropped. It is a genuinely
different index, measuring ring-current depression rather than mid-latitude
disturbance, which is precisely why it needs an external data source not already
cached. Given that Kp had already reproduced the Ap pattern almost line for line,
the marginal value did not justify a new data dependency at that point.

## 7. The gating discipline

Three cheap checks ran before committing to GPU extraction, each capable of
killing the project on its own:

| Gate | Question | Answer | Cost |
|---|---|---|---|
| 1 | Do classical indices correlate with Ap at all? | Yes, r 0.15 to 0.19, consistent across epochs | CPU, minutes |
| 2 | Does persistence decay fast enough to leave a window? | Yes, below classical level by day 3 to 4 | CPU, minutes |
| 3 | Does the 27-day recurrent structure survive the classical fit? | Yes, essentially untouched, 3.00 vs 3.68 percent | CPU, minutes |

Only after all three passed was the Colab pipeline validation run. Only after
that succeeded end to end, producing a real `[1, 65536, 1280]` tensor, did batch
extraction begin. And batch extraction itself began with three calibration
timestamps spread across the solar cycle before the full 1,416-target run.

That ordering is why the eventual negative result is informative rather than
inconclusive. The hypothesis had already cleared every cheap way of being wrong
before the expensive test was run, so the failure is attributable to the
hypothesis or to the pipeline, and not to a broken premise.

## 8. Sources

- **GFZ Potsdam** combined Kp/Ap/SN/F10.7 record, 1932 to present, CC BY 4.0.
  https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt
- **SDO** EUV imagery via the NASA Surya benchmark bucket,
  `s3://nasa-surya-bench/`, 2010 to 2024.
- **Surya**, NASA and IBM heliophysics foundation model,
  https://github.com/NASA-IMPACT/Surya, weights distributed via
  `huggingface_hub`.
