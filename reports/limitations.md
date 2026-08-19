# Limitations

What the result does and does not establish, ordered by how much each one
constrains the headline claim.

The claim under examination: *classical indices beat a Surya embedding probe at
every lead time tested, for both Ap and Kp.*

---

## 1. The pooling discards the signal under test

**This is the most serious limitation and it was found after the analysis was
complete.**

Extraction reduces Surya's `[1, 65536, 1280]` token tensor to one 1,280-vector:

```python
pooled = tokens.mean(dim=1).squeeze(0).cpu().numpy()
```

`dim=1` is the token axis. The operation averages across all 65,536 spatial
tokens, that is, the entire solar disk.

The project exists because scalar indices compress the disk to one number and
therefore cannot represent coronal-hole geometry. A global mean over all spatial
tokens performs the same compression, just into 1,280 numbers instead of one.
What survives describes the Sun's average appearance, not where anything is on
it.

**Consequence.** The comparison may have handed the embeddings the same handicap
it was designed to expose in the classical indices. The negative result is sound
as a statement about *this pipeline*. It does not establish that Surya lacks the
signal, and the finding should not be cited as if it did.

**Status.** Not yet tested. A free diagnostic exists and is the first item in
[future-work.md](future-work.md).

## 2. It is a probe result, not a model result

The method is PCA followed by Ridge regression. That tests whether the target is
*linearly accessible* in a low-dimensional projection of the embedding space. It
does not test whether the information is present in some form a different head
could reach.

The residual-correction variant partially mitigates this, since it sets an
easier and lower-variance target and should have surfaced complementary signal
if any existed. It did not, at any lead time, for either target. That is genuine
evidence, but it is evidence about linear accessibility rather than about
presence.

## 3. Sample size is a partial, unresolved constraint

1,785 training rows against 1,280 raw dimensions is a difficult regime.

Two observations show this mattered:

- Tripling the training set narrowed the lead-5 gap by 86 percent for Ap
  (1.89 to 0.27 RMSE) and 91 percent for Kp (0.45 to 0.04).
- The MLP variant went from catastrophic to competitive, 22.16 to 13.75 at
  Ap lead 7.

Two observations show it was not the whole story:

- Other lead times were flat or drifted slightly in classical's favour.
- Nothing crossed over to beat classical anywhere.

Cross-validation consistently selects low PCA component counts, 5 to 50 out of a
grid extending to 100, which suggests most of the 1,280 dimensions are
uninformative or redundant for this target. Whether that is a property of Surya
or of the pooling in limitation 1 cannot be separated with the current data.

## 4. A single test window

One chronological split, training on or before 2021-12-31 and testing from
2022-01-01, 156 test rows on the embedding grid.

Model selection is properly protected: hyperparameters come from
`GridSearchCV` over `TimeSeriesSplit(5)` on the training set only, so there is
no test leakage. The limitation is different. Every headline RMSE rests on one
test period, and that period is the rising phase of solar cycle 25, a specific
and unusually active regime. The result is not known to be stable across other
phases of the cycle.

## 5. The embedding record is short and unevenly sampled

Classical indices span 1932 onward, tens of thousands of daily rows. Embeddings
cover 2010 to 2024 only, roughly one solar cycle.

Sampling is also asymmetric. The training window was densified to every 3 days,
but the test window stayed weekly at 156 rows. The two sides of the split
therefore have different temporal resolution, which is fine for the comparison
as run but limits what can be said about behaviour at finer test cadence.

Coverage is 754 of 764 expected weekly slots, 98.7 percent, and 2025 is
unavailable because the NASA benchmark bucket does not yet mirror it. Both are
upstream data limits rather than pipeline defects, confirmed by a dedicated
re-run, but they still bound the record.

## 6. The spectral gate is a sanity check, not a significance test

The 27-day result (raw Ap 3.00 percent of spectral power in the 25 to 29 day
band, classical residual 3.68 percent) is a periodogram ratio on a single test
window. There is no Lomb-Scargle false-alarm-probability estimate and no
significance test.

Also worth stating plainly: 3 percent of total spectral power in one band is a
modest absolute signal. Most of Ap's variance is broadband, because real storms
are messier than a clean 27-day recurrence. The gate establishes that the
classical fit removes essentially none of the recurrent component, which is what
it was for. It does not establish that the recurrent component is large.

## 7. Ap and Kp are not independent evidence

Kp reproduces the Ap pattern almost line for line. That is expected rather than
confirmatory, since Ap is close to a linear rescaling of averaged Kp. The two
targets should be read as one result checked for arithmetic robustness, not as
two independent tests. Dst was considered as a genuinely different third target
and dropped because it requires an external source not already cached.

## 8. Engineering caveat

`next.config.mjs` sets `typescript.ignoreBuildErrors: true`. The site build
passing is therefore not a type check. This affects the findings page only, not
any reported number.

---

## What survives all of this

The classical baseline itself is not in question. It is measured on the daily
grid at n in the thousands, holds 14.34 to 14.64 RMSE across the entire 3 to 14
day window, and a small MLP on the same features does not improve on it. It is a
solid bar rather than a strawman, and any future embedding result has to clear
the same number on the same split.
