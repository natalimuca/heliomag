# Future work

Four candidate next steps, ordered by expected value per hour rather than by
ambition. Each entry states what it costs, what it would show, and what result
would count as a failure.

The ordering rests on one judgement: the weakest part of the current result is
not the amount of data, it is the representation. See limitation 1 in
[limitations.md](limitations.md).

---

## 1. Periodogram the existing embeddings

**Cost:** an afternoon. CPU only. No new data, no GPU, no re-extraction.

**Method.** `scripts/regime_analysis.py` already measures the share of spectral
power in the 25 to 29 day band for Ap and for the classical residual. Run the
same measurement on the existing 1,280-dimensional embeddings, per component or
on the leading PCA components, over the same window.

**What it tests.** Whether the mean-pooled embedding retains any trace of the
27-day Carrington recurrence. Coronal holes rotate with the Sun, so if the
embedding sees them at all, that periodicity should be present.

**Interpretation.**

- *Little or no 27-day power in the embeddings, while Ap carries 3.00 percent.*
  Direct evidence that `tokens.mean(dim=1)` removed the structure the project was
  built to test. This reframes the headline result from "Surya does not carry the
  signal" to "this pooling destroyed it", and justifies the GPU spend in step 2.
- *Clear 27-day power present.* The pooling preserved the recurrence, the
  representation explanation weakens considerably, and attention should move to
  sample size and to step 3.

Either outcome is informative, which is why this goes first. It is also the
cheapest thing in this document by a wide margin.

## 2. Re-extract with spatial pooling retained

**Cost:** roughly the compute of the original extraction run. Note carefully
that this is **not** the ~38 hour daily-density figure, because the cadence does
not change. Same 1,941 dates, different reduction.

**Method.** Replace the global mean with a coarse spatial grid. The 65,536
tokens correspond to a 256 by 256 layout, so average-pool to 4 by 4 or 8 by 8,
giving 16 or 64 regions of 1,280 dimensions per date.

Storage at 4 by 4: 1,941 x 16 x 1,280 float32, about 160MB. At 8 by 8 it is
roughly 640MB, which is fine as a Kaggle output but too large to commit.

Cheap additions worth storing in the same pass, since they cost nothing extra:
the standard deviation across tokens, as a scalar proxy for how structured the
disk is, and a few percentiles.

**What it tests.** The actual stated hypothesis, which the current pipeline does
not. Even a 4 by 4 grid preserves gross east-west and latitude structure, which
is what matters for a coronal hole rotating into a geoeffective position.

**Failure condition.** If a spatially resolved probe still loses to classical at
every lead time on the same split, that is a much stronger negative result than
the current one, and worth publishing as such.

## 3. Rolling-origin cross-validation on the headline comparison

**Cost:** low. CPU only, existing data, no re-extraction.

**Method.** Replace the single chronological split with several rolling test
windows across 2010 to 2024, refitting on everything prior to each.

**What it tests.** Whether the negative result is a property of the data or of
the 2022-onward test period specifically, which is the rising phase of solar
cycle 25.

**Why it is worth doing regardless.** It upgrades the claim from "true in this
one window" to "robust across windows", at almost no cost. If classical stops
winning in some window, that is a genuinely interesting finding and it would be
better to discover it internally than to have a reader find it.

Note the interaction with limitation 5: the test window is currently weekly at
156 rows while training is 3-daily. Rolling windows will need care so each fold
keeps a comparable test cadence.

## 4. Full daily-density extraction

**Cost:** approximately 38 hours of GPU time per the original scoping.

**Recommendation: do not do this next.** The densification evidence was mixed.
Tripling the training set narrowed lead 5 sharply, by 86 to 91 percent, but left
other lead times flat and moved nothing across the line. Spending the largest
compute budget in this document on more of the same weekly-derived signal has
the worst expected value per hour of the four options.

It becomes reasonable only if step 1 shows the pooling preserved the 27-day
structure and step 2 is therefore not indicated, leaving sample size as the main
remaining explanation.

---

## Suggested sequence

```
1. Periodogram existing embeddings        (afternoon, CPU)
        |
        +-- little 27-day power --> 2. Re-extract with spatial grid (GPU, ~original run cost)
        |
        +-- clear 27-day power  --> 4. Reconsider daily density (GPU, ~38h)

3. Rolling-origin CV                       (low cost, run in parallel with either branch)
```

## Pre-commitment

Worth writing down before running step 1, so the result cannot be reinterpreted
after the fact:

- **The prediction.** If the pooling explanation is right, the mean-pooled
  embeddings will carry substantially less 25 to 29 day spectral power than raw
  Ap's 3.00 percent.
- **What would change the conclusion.** If the embeddings carry comparable or
  greater 27-day power, the representation explanation is weakened and the
  current negative result stands closer to its face-value reading.
- **What does not count.** Finding 27-day power in *some* PCA component after
  searching all of them is not the same as the leading components carrying it.
  Fix the components to be examined before looking.

## Smaller items

- Two unused icon files remain in `public/`, `icon-dark-32x32.png` and
  `icon-light-32x32.png`, now unreferenced after the icon fix in `962ce99`.
- `display-condensed` resolves to the plain system sans stack at
  `font-stretch: 100%`. Nothing is actually condensed and the site loads no
  webfont, which is why the hero headline needs so much width. A genuinely
  condensed face would let it run larger at narrow viewports.
- `next.config.mjs` sets `typescript.ignoreBuildErrors: true`. Turning it off
  and fixing whatever surfaces would make the build a real check.
- `data/*.npz` is about 14MB tracked in git. Fine today, worth watching if step 2
  adds a spatially resolved set.
