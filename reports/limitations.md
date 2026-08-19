# Limitations

What the result does and does not establish, ordered by how much each one
constrains the headline claim.

**The claim under examination:** classical scalar indices beat a Surya embedding
probe at every lead time tested, for both Ap and Kp.

**Summary table**

| # | Limitation | Severity | Resolvable? |
|---|---|---|---|
| 1 | Global mean pooling discards the spatial structure under test | **High**, bounds the conclusion | Yes, step 1 then 2 of future work |
| 2 | Probe result, tests linear accessibility rather than presence | Medium | Partially, by richer heads |
| 3 | Sample size a partial, unresolved constraint | Medium | Yes, but expensive |
| 4 | Single test window, one phase of solar cycle 25 | Medium | Yes, cheaply, rolling-origin CV |
| 5 | Short embedding record, asymmetric train and test cadence | Low to medium | Partly, bounded upstream |
| 6 | Spectral gate is a sanity check, not a significance test | Low, affects the gate only | Yes, cheaply |
| 7 | Ap and Kp are not independent evidence | Low, expected | Only via Dst or similar |
| 8 | TypeScript errors ignored in the site build | None on results | Yes, trivially |

---

## 1. The pooling discards the signal under test

**This is the most serious limitation and it was found after the analysis was
complete.**

Extraction reduces Surya's `[1, 65536, 1280]` token tensor to one 1,280-vector:

```python
pooled = tokens.mean(dim=1).squeeze(0).cpu().numpy()
```

`dim=1` is the spatial token axis. The operation averages across all 65,536
tokens, that is, across the entire solar disk. 65,536 is consistent with a 256
by 256 grid, so what is being averaged away is a two-dimensional map.

### Why this specifically matters here

The project exists because scalar indices compress the disk to one number and
therefore cannot represent coronal-hole geometry. A global mean over all spatial
tokens performs the same compression, just into 1,280 numbers instead of one.
What survives describes the Sun's average appearance, not where anything is on
it.

The motivating structure is doubly ill-suited to this reduction:

- It is **positional**. A coronal hole matters because of where it sits.
  Averaging over position is close to the worst available reduction.
- It is **periodic through rotation**. The 27-day recurrence exists *because*
  position changes over time. Remove position and the mechanism generating the
  periodicity is no longer represented.

### What this does and does not do to the result

It does **not** invalidate any number in `results/`. Every RMSE reported is a
correct measurement of what was actually run.

It **does** bound what those numbers mean. The comparison as built may have
handed the embeddings the same handicap it was designed to expose in the
classical indices. The negative result is sound as a statement about this
pipeline and overstated as a statement about Surya.

### Corroborating detail

Cross-validation selects only 5 to 50 useful PCA components out of 1,280, which
is consistent with a representation whose informative structure has already been
averaged away. This is suggestive rather than conclusive, since the same pattern
would appear if Surya simply had no relevant signal.

### Status

Untested. The spectral gate in `results/regime_analysis.md` established that the
27-day component survives the **classical fit**. It never established that the
component survives the **pooling**, and that check has not been run. It is free.
See [future-work.md](future-work.md), step 1.

## 2. It is a probe result, not a model result

The method is PCA followed by Ridge regression. That tests whether the target is
**linearly accessible in a low-dimensional projection** of the embedding space.
It does not test whether the information is present in some form a different head
could reach.

### What partially mitigates this

Four model families were tried, not one:

- Embeddings standalone.
- Embeddings concatenated with classical features. Note the asymmetry here:
  only the embedding block passes through PCA, while the five classical columns
  pass through standardisation alone, so PCA can never discard them. This model
  contains the classical model as a special case up to regularisation, and still
  does not beat it at any lead.
- A nonlinear head, PCA followed by a small MLP.
- Residual correction, where embeddings predict only what the classical fit gets
  wrong. This is the easiest and lowest-variance version of the task, and it is
  where complementary signal should have appeared most readily.

### What does not

All four operate on the same mean-pooled input. If limitation 1 holds, then all
four inherited the same handicap, and the breadth of the model search does not
compensate for it. Model breadth and representation quality are independent axes,
and only one of them was explored.

## 3. Sample size is a partial, unresolved constraint

1,785 training rows against 1,280 raw dimensions is a difficult regime.

**Evidence that it mattered:**

- Tripling the training set narrowed the lead-5 gap by 86 percent for Ap,
  1.89 to 0.27 RMSE, and about 91 percent for Kp, roughly 0.45 to 0.04.
- The MLP variant went from catastrophic to competitive, 22.16 to 13.75 at Ap
  lead 7, and 39.80 to 16.91 at Ap lead 14.
- Cross-validation selects low PCA counts, 5 to 50 from a grid reaching 100,
  which is what a regulariser does when rows are scarce relative to dimensions.

**Evidence that it was not the whole story:**

- Other lead times were flat or drifted slightly in classical's favour.
- Nothing crossed over to beat classical anywhere, at any lead, on either target.
- The reversal at lead 10 moved in the *opposite* direction to what a pure
  sample-size story predicts.

**The unresolvable part.** Whether the residual gap is a sample-size effect or a
representation effect cannot be separated with the current data, because both
explanations predict the same observation. Resolving it requires changing one of
them, which is why the future-work ordering puts representation first: it is the
cheaper of the two to change per unit of information gained.

## 4. A single test window

One chronological split, train on or before 2021-12-31 and test from 2022-01-01,
giving 156 test rows on the embedding grid.

**What is properly protected.** Hyperparameters come from `GridSearchCV` over
`TimeSeriesSplit(5)` fitted on the training set only. The test window is never
seen during model selection, so there is no selection leakage. The split is
chronological rather than random, which is correct for an autocorrelated series,
since a random split would leak near-duplicate neighbouring days across the
boundary.

**What is not protected.** Every headline RMSE rests on one test period, and that
period is the rising phase of solar cycle 25, a specific and unusually active
regime. Nothing here establishes that the result is stable across other phases of
the cycle. A 156-row test fold is also small enough that differences of a few
hundredths of an RMSE unit, which is exactly the scale of several results in the
tables, carry meaningful uncertainty that is not quantified anywhere.

This is the cheapest serious limitation to fix. See
[future-work.md](future-work.md), step 3.

## 5. Short and unevenly sampled embedding record

**Length.** Classical indices span 1932 onward, tens of thousands of daily rows.
Embeddings cover 2010-05 to 2024-12, roughly one solar cycle. Any cycle-dependent
behaviour is therefore observed approximately once.

**Asymmetric cadence.** The training window was densified to every 3 days, but
the test window remained weekly at 156 rows. The two sides of the split have
different temporal resolution. This is fine for the comparison as run, since both
model families see identical rows, but it limits what can be said about behaviour
at finer test cadence, and it complicates any future rolling-origin design.

**Coverage.** 754 of 764 expected weekly slots, 98.7 percent, and 30 of 1,416
densification targets missing, 2.1 percent. Both were confirmed as genuine SDO
source gaps rather than pipeline defects, by a dedicated re-run and by the
clustering of the fine-resolution failures around the coarse-resolution ones.

**2025 is unavailable.** All 52 weekly attempts failed identically because the
NASA benchmark bucket does not yet mirror 2025. This is an upstream limit, not a
pipeline defect, but it still bounds the record and rules out testing on the most
recent data.

## 6. The spectral gate is a sanity check, not a significance test

The 27-day result, raw Ap at 3.00 percent of spectral power in the 25 to 29 day
band against a classical residual at 3.68 percent, is a periodogram ratio on a
single test window at a single lead time.

What is missing: a Lomb-Scargle false-alarm-probability estimate, any
significance test, and any check that the result is stable across windows or
across lead times.

Also worth stating plainly: **3 percent of total spectral power in one band is a
modest absolute signal.** Most of Ap's variance is broadband, because real storms
are messier than a clean 27-day recurrence. The gate establishes that the
classical fit removes essentially none of the recurrent component, which is what
it was for. It does not establish that the recurrent component is large enough to
be worth much even if fully captured.

This limitation affects the **motivation** for the work rather than the validity
of its measurements.

## 7. Ap and Kp are not independent evidence

Kp reproduces the Ap pattern almost line for line. That is expected rather than
confirmatory, because Ap is close to a linear rescaling of averaged Kp.

The two targets should be read as **one result checked for arithmetic
robustness**, not as two independent tests. Reporting both is still worthwhile,
since it rules out the possibility that the negative result was an artifact of
Ap's particular scaling, but it does not double the evidence.

Dst was considered as a genuinely different third target, measuring ring-current
depression rather than mid-latitude disturbance, and dropped because it requires
an external source not already cached.

## 8. Engineering caveat

`next.config.mjs` sets `typescript.ignoreBuildErrors: true`. The site build
passing is therefore not a type check. This affects the findings page only and
has no bearing on any reported number, but it means CI green is a weaker signal
than it looks.

---

## What survives all of this

The **classical baseline itself is not in question**. It is measured on the daily
grid at n in the thousands, holds 14.34 to 14.64 RMSE across the entire 3 to 14
day window, beats persistence by roughly 5.2 RMSE from lead 3 onward, and does
not improve when given a nonlinear model on the same five features. It is a solid
bar rather than a strawman, and any future embedding result has to clear the same
number on the same split.

The **gating evidence also survives**. Classical indices do correlate with Ap,
persistence does decay below them by day 3 to 4, and the 27-day recurrent
structure does survive the classical fit. Those three results are independent of
anything to do with Surya, and they stand whatever a better-pooled embedding
eventually shows.
