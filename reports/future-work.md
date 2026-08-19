# Future work

Four candidate next steps, ordered by expected value per hour rather than by
ambition. Each entry states what it costs, what it would show, what result counts
as failure, and enough implementation detail to start without rediscovering the
pipeline.

The ordering rests on one judgement: **the weakest part of the current result is
not the amount of data, it is the representation.** See limitation 1 in
[limitations.md](limitations.md).

| Step | Cost | GPU | Resolves |
|---|---|---|---|
| 1. Periodogram the existing embeddings | An afternoon | No | Limitation 1, diagnostically |
| 2. Re-extract with spatial pooling retained | ~1 original extraction run | Yes | Limitation 1, substantively |
| 3. Rolling-origin cross-validation | Low | No | Limitation 4 |
| 4. Full daily-density extraction | ~38 hours | Yes | Limitation 3, partially |

---

## 1. Periodogram the existing embeddings

**Cost:** an afternoon. CPU only. No new data, no GPU, no re-extraction.

### Why this is first

`results/regime_analysis.md` established that the 27-day recurrent component
survives the **classical fit**. Nobody has checked whether it survives the
**pooling**. That check is free, and its outcome determines whether step 2 is
worth its GPU time or whether attention should move elsewhere entirely.

### Method

`scripts/regime_analysis.py` already contains the measurement:

```python
def band_power_fraction(signal, period_lo=25, period_hi=29):
    signal = signal - np.mean(signal)
    freqs, power = periodogram(signal, fs=1.0)
    periods = 1.0 / freqs
    mask = (periods >= period_lo) & (periods <= period_hi)
    return power[mask].sum() / power.sum()
```

Apply it to the embeddings instead of to Ap. Two practical notes:

- **Resample first.** `fs=1.0` means one sample per day. The embedding index is
  weekly outside the training window and 3-daily inside it, so the series must be
  put on a uniform grid before the periodogram is meaningful. Either restrict to
  the weekly grid and set `fs=1/7`, or reindex to daily and interpolate, and be
  explicit about which. The weekly-only route is cleaner: at weekly sampling the
  27-day period is under 4 samples per cycle, which is close to the Nyquist limit
  and should be stated as a caveat rather than glossed over. **This is a real
  design decision and should be made before looking at any output.**
- **Fix the components in advance.** Run it on the leading PCA components, for
  example the first 10, decided before looking. See the pre-commitment below.

### Interpretation

| Outcome | Reading | Next |
|---|---|---|
| Little or no 27-day power in embeddings, while Ap carries 3.00 percent | Direct evidence that `tokens.mean(dim=1)` removed the structure under test | Step 2, with justification |
| Clear 27-day power present | The pooling preserved the recurrence, so the representation explanation weakens considerably | Step 3, then reconsider step 4 |

Either outcome is informative, which is why this goes first. It is also the
cheapest item in this document by a wide margin.

### Failure condition

If the sampling caveat above turns out to make the measurement uninterpretable at
weekly cadence, say so and move to step 2 on the strength of the structural
argument alone. Do not manufacture a result by interpolating to daily and
treating the interpolation as data.

---

## 2. Re-extract with spatial pooling retained

**Cost:** roughly the compute of the original extraction run. Note carefully that
this is **not** the ~38 hour daily-density figure, because the cadence does not
change. Same 1,941 dates, different reduction.

### Method

The change is one line in `kaggle/heliomag-embedding-extraction.ipynb`. Replace:

```python
pooled = tokens.mean(dim=1).squeeze(0).cpu().numpy()          # [1280]
```

with a spatial average-pool. `tokens` is `[1, 65536, 1280]` and 65,536 is
consistent with a 256 by 256 grid:

```python
t = tokens.squeeze(0).reshape(256, 256, 1280)                  # H, W, C
g = 4                                                          # 4x4 or 8x8
t = t.reshape(g, 256 // g, g, 256 // g, 1280).mean(dim=(1, 3)) # g, g, C
pooled = t.reshape(g * g, 1280).cpu().numpy()                  # [16, 1280]
```

**Verify the 256 by 256 assumption before running the batch.** The validation
notebook confirms the token count but not the layout. Check the model config for
patch size and input resolution, or confirm empirically that a spatially
structured input produces a correspondingly structured token map. Getting this
wrong scrambles the grid and produces a result that looks like noise for the
wrong reason.

### Cheap additions worth storing in the same pass

They cost nothing extra once the tensor is in memory, and each one answers a
question the current data cannot:

- `tokens.std(dim=1)`, a scalar-per-channel proxy for how spatially structured
  the disk is, which distinguishes a uniform disk from a patchy one at the same
  mean.
- A few percentiles across tokens, for example the 10th and 90th, which capture
  the presence of extreme regions that a mean hides.
- The existing global mean, so the new dataset is a strict superset of the old
  one and results remain directly comparable.

### Storage

| Grid | Shape per date | Total for 1,941 dates, float32 |
|---|---|---|
| 1x1, current | 1,280 | 9.9 MB |
| 4x4 | 16 x 1,280 | ~159 MB |
| 8x8 | 64 x 1,280 | ~636 MB |

4x4 is committable if compressed and is the recommended starting point. 8x8 is
fine as a Kaggle output but should not go into git. Note that step 2 at 4x4
multiplies the probe's input dimensionality by 16, to 20,480, against 1,785 rows.
The PCA grid will need extending upward and the sample-size pressure in
limitation 3 gets worse, not better, so consider probing per-region first and
pooling predictions rather than concatenating all regions into one feature
vector.

### Reuse what already works

The extraction loop's checkpointing, resume-by-key, temp cleanup and disk logging
all carry over unchanged, and they are the reason a 12-hour platform timeout was
survivable last time. Keep the three calibration timestamps, 2011-06-15,
2018-06-15 and 2024-06-15, and check the new output shape on those before
launching the batch.

### What it tests

The actual stated hypothesis, which the current pipeline does not. Even a 4x4
grid preserves gross east-west and latitude structure, which is what matters for
a coronal hole rotating into a geoeffective position.

### Failure condition

If a spatially resolved probe still loses to classical at every lead time on the
same split, that is a **much stronger negative result** than the current one, and
worth publishing as such. The current result would then be correctly restated
rather than retracted.

---

## 3. Rolling-origin cross-validation

**Cost:** low. CPU only, existing data, no re-extraction.

### Method

Replace the single chronological split with several rolling test windows across
2010 to 2024, refitting on everything prior to each. `TimeSeriesSplit` is already
imported and used for hyperparameter selection, so the machinery is present. The
change is to the outer evaluation loop, not to the model code.

Report the distribution of the classical-minus-embedding RMSE gap per lead time
rather than a single number, so the spread is visible.

### The complication to handle honestly

Limitation 5 applies here. The training window is 3-daily and the test window is
weekly, so a naive rolling split will produce folds with different sampling
density on either side of the boundary, and the RMSE values will not be
comparable across folds. Either restrict every fold to the weekly grid, losing
the densification benefit, or state per-fold cadence explicitly. Choose before
running.

### What it tests

Whether the negative result is a property of the data or of the 2022-onward test
period specifically, which is the rising phase of solar cycle 25.

### Why it is worth doing regardless of steps 1 and 2

It upgrades the claim from "true in this one window" to "robust across windows"
at almost no cost, and it puts an uncertainty around differences that are
currently reported to two decimals off a 156-row fold. If classical stops winning
in some window, that is a genuinely interesting finding, and it is better to
discover it internally than to have a reader find it.

---

## 4. Full daily-density extraction

**Cost:** approximately 38 hours of GPU time per the original scoping.

**Recommendation: do not do this next.**

The densification evidence was mixed. Tripling the training set narrowed lead 5
sharply, by 86 to 91 percent, but left other lead times flat and moved nothing
across the line. Spending the largest compute budget in this document on more of
the same weekly-derived signal has the worst expected value per hour of the four
options here.

It becomes reasonable only if step 1 shows the pooling preserved the 27-day
structure, so step 2 is not indicated, leaving sample size as the main remaining
explanation. In that case it is the obvious move rather than a marginal one.

If it is run, note that the test window should be densified too. The current
asymmetry, 3-daily training against weekly testing, means added training density
cannot improve the resolution of the evaluation itself.

---

## Suggested sequence

```
1. Periodogram existing embeddings          (afternoon, CPU)
        |
        +-- little 27-day power  --->  2. Re-extract with spatial grid
        |                                 (GPU, ~one original run)
        |                                       |
        |                                       +-- still loses --> publish the
        |                                       |                   stronger null
        |                                       +-- wins --------> restate the
        |                                                           headline
        |
        +-- clear 27-day power   --->  4. Reconsider daily density (GPU, ~38h)

3. Rolling-origin CV                        (low cost, run alongside either branch)
```

## Pre-commitment

Worth writing down before running step 1, so the result cannot be reinterpreted
after the fact.

**The prediction.** If the pooling explanation is right, the mean-pooled
embeddings will carry substantially less 25 to 29 day spectral power than raw
Ap's 3.00 percent.

**What would change the conclusion.** If the embeddings carry comparable or
greater 27-day power, the representation explanation is weakened and the current
negative result stands closer to its face-value reading.

**Fixed in advance, before looking at output:**

- Which components are examined. The first 10 PCA components, not "whichever
  ones show something".
- Which sampling grid is used, weekly at `fs=1/7` or daily interpolated, and the
  Nyquist caveat stated either way.
- The comparison band, 25 to 29 days, unchanged from the existing gate.

**What does not count.** Finding 27-day power in *some* component after searching
all 1,280 is not the same as the leading components carrying it. With enough
components, something will always look periodic.

## Smaller items

- Two unused icon files remain in `public/`, `icon-dark-32x32.png` and
  `icon-light-32x32.png`, unreferenced since the icon fix in `962ce99`. They were
  presumably intended for light and dark favicon variants, which Next's file
  convention cannot express with media queries. Either wire them up through
  explicit `metadata.icons` entries with `media`, being careful about basePath,
  or delete them.
- `display-condensed` resolves to the plain system sans stack at
  `font-stretch: 100%`. Nothing is actually condensed and the site loads no
  webfont, which is why the hero headline needs so much width and why the fix in
  `d1ce419` had to shrink it at narrower viewports. A genuinely condensed face
  would let it run larger.
- `next.config.mjs` sets `typescript.ignoreBuildErrors: true`. Turning it off and
  fixing whatever surfaces would make CI green mean something.
- `data/*.npz` is about 14 MB tracked in git. Fine today. Worth revisiting if
  step 2 adds a 159 MB spatially resolved set, at which point Git LFS or an
  external artifact store becomes the right answer.
- The GFZ fetch has no pinned snapshot. `fetch_data.py` caches on first run, but
  a fresh clone re-downloads whatever the file says that day. Numbers in
  `results/` are therefore reproducible from the cache but not guaranteed
  bit-identical from a cold start, since GFZ revises recent values.
