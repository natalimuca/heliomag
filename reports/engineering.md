# Engineering log: what broke and how it was fixed

A record of the problems that actually cost time, on both the research pipeline
and the findings page. Several were diagnosed wrongly for a while, and the wrong
diagnoses are as instructive as the fixes, so they are kept here rather than
tidied away.

Timeline: 2026-07-30 to 2026-08-19, 52 commits.

Contents:
[Part 1, research pipeline](#part-1-the-research-pipeline) ·
[Part 2, findings page](#part-2-the-findings-page) ·
[Lessons](#lessons-worth-carrying-forward) ·
[Commit map](#appendix-commit-map)

---

## Part 1: the research pipeline

### A previous project was abandoned for sample size

Before this project existed, a pilot on NAO and tropospheric teleconnections was
carried far enough to discover that the usable sample was n = 16. It was dropped
rather than written up.

The replacement was chosen for the opposite property. The GFZ record gives tens
of thousands of daily rows, so the ground-truth side is never sample-limited.
This is why the first thing the current pipeline does is confirm n, not confirm
a correlation. `results/pilot_kp.md` reports n = 34,544 for SN against the full
record and n = 28,366 for F10.7, the difference being that F10.7 measurements
only begin in 1947.

### Sentinel values that would have silently poisoned every mean

The GFZ file encodes missing data as `-1` for `Ap` and `SN`, and `-1.0` for
`F107obs` and each of the eight 3-hourly `Kp` columns. These are ordinary
numbers to pandas.

Left unhandled, a missing day contributes `-1` to a rolling mean, to a
correlation, and to the derived daily Kp. Nothing crashes and nothing looks
wrong. `fetch_data.py` replaces all of them with NaN before any arithmetic, and
every downstream stage drops NaN rows after joining features to target, so
rolling-window warm-up and record gaps are handled by the same mechanism.

Also worth recording: daily Kp is **derived, not read**. The file stores eight
3-hourly values and the daily value used throughout is their arithmetic mean.
And `F107obs` is used rather than `F107adj`, the version adjusted to 1 AU.

### Finding where Surya exposes embeddings at all

Surya's `easy_inference` entry point produces a pixel-space forecast, not
embeddings. Getting tokens out required reading the model source rather than the
documentation.

The mechanism is in `surya/models/helio_spectformer.py`. `HelioSpectFormer.forward()`
takes an `if self.finetune: return tokens` branch immediately after
`tokens = self.backbone(tokens)`, returning raw backbone tokens and skipping the
pixel-space decoder entirely.

So the extraction path is: build the model exactly as the quickstart does, then
set `model.finetune = True` after construction, then run a normal batch through
it. No reimplementation of Surya's data loading was needed, which was the point.
`notebooks/surya_pipeline_check.ipynb` confirms this end to end on a Colab T4
and reports the real tensor shape, `[1, 65536, 1280]`, that is (batch, spatial
tokens, embedding dimension).

### The index bug that cost real debugging time

`InputOnlyRolloutDataset` builds its internal `path_lookup` from
`present_index["path"].to_dict()`. `Series.to_dict()` keys on the **DataFrame
index**, not on any column.

The index CSV is read with `pd.read_csv`, which produces a `RangeIndex` with
`timestep` as an ordinary column. Passing that straight in produces a lookup
keyed on 0, 1, 2 rather than on timestamps. It does not raise. It just fails to
match anything.

Fix, and the comment is preserved in the notebook because it is the kind of
thing that gets re-broken:

```python
idx_df['timestep'] = pd.to_datetime(idx_df['timestep'])
present_index = idx_df.set_index('timestep')
```

### Kaggle's session cap does not fit a full extraction pass

A single extraction pass exceeds Kaggle's own session time limit. The
densification attempt, session v9, ran roughly 12 hours and was auto-cancelled at
about 95 percent complete.

Losing 12 hours of GPU work to a platform timeout is only survivable if the job
is resumable, so the extraction loop is built around that:

- The full `.npz` is rewritten every 10 successful extractions, not at the end.
- A run starts by loading a previous merged `.npz` from a Kaggle dataset input
  and skipping every key already present.
- `/kaggle/temp` is deleted and recreated every 20 extractions, because each
  timestamp downloads its own SDO frames.
- Free space on `/kaggle/temp`, `/` and `/kaggle/working` is printed at every
  checkpoint, because disk exhaustion was a live risk rather than a theoretical
  one.

Session v10 resumed from the v9 checkpoint and finished the remaining 71 targets
in under an hour.

### Calibrating before committing

Three calibration timestamps were run before any batch job, deliberately spread
across the solar cycle: 2011-06-15, 2018-06-15 and 2024-06-15, each at 12:00
UTC, each timed and shape-checked. A pipeline that works on a 2018 date and
fails on a 2011 date is a data-availability problem, not a code problem, and it
is much cheaper to discover that on three dates than on 1,416.

### Missing dates that looked exactly like a pipeline bug

The weekly grid returned 754 of 764 expected slots, 98.7 percent. Ten missing
weeks scattered sporadically between 2012 and 2023 is precisely the pattern a
flaky fetch loop produces, so it could not be waved through.

**Diagnosis.** Session v8 re-attempted exactly those dates plus all of 2025. The
2023 to 2024 portion re-fetched successfully, but every returned value duplicated
data already held. All 52 weekly attempts in 2025 failed identically with
`Failed listing s3://nasa-surya-bench/2025/...`.

**Conclusion.** The gaps are genuine SDO source holes, multi-day instrument
outages, and 2025 is simply not mirrored in the benchmark bucket yet. Blocked
upstream, not locally.

**The confirming detail.** Densification hit the same gaps at finer resolution,
30 of 1,416 targeted dates, 2.1 percent, on a session that completed rather than
being cancelled. They cluster in multi-day runs immediately around the dates
already known missing from the weekly grid, for example around 2012-09, 2013-04,
2016-08 and 2018-12.

That clustering is what settled it. A real multi-day outage appears as one
missing week at coarse resolution and a short burst of missing days at fine
resolution. An intermittent bug would scatter randomly instead. The pattern is
diagnostic in a way a raw failure count never is.

### Aligning a daily record to an irregular embedding grid

Two traps here, both handled deliberately in `embedding_probe.py`.

**Rolling means must be computed before reindexing.** `Ap_3d_mean` computed on a
weekly grid would be a 3-week mean wearing a 3-day label. The code computes
rolling means on the daily series and only then reindexes onto the embedding
dates with `method="ffill"`.

**Targets must be shifted by date, not by row.**

```python
target_dates = emb.index + pd.Timedelta(days=lead)
target = gfz[target_col].reindex(target_dates)
```

A `shift(-lead)` on the embedding frame would mean "lead sampling steps ahead",
which on a weekly grid is 3 to 14 **weeks**. On the densified 3-day grid it would
be a different error again, and the two halves of the dataset have different
cadence, so the bug would not even be uniform.

### The MLP was catastrophically bad, and that turned out to be evidence

On the weekly pass with 598 training rows, PCA plus a small MLP scored worse than
naive persistence at every lead time on both targets, reaching 39.80 RMSE at Ap
lead 14 against a classical 12.44 and a persistence 17.69. The obvious reading
was a broken configuration.

It was not broken, it was starved. After densification to 1,785 rows the same
model went from 22.16 to 13.75 at Ap lead 7, beating both linear embedding
variants for the first time, and from 39.80 to 16.91 at Ap lead 14.

Cross-validation makes the mechanism visible. The MLP consistently selected the
top of the PCA grid, n = 100, while the linear Ridge probe selected 5 to 50.
That is a model reaching for degrees of freedom it cannot afford, and its
recovery under more rows is the expected signature.

Keeping the bad number on the record rather than deleting it as an obvious
outlier is what made the later improvement interpretable.

### A published finding was retracted

The weekly pass showed the embedding probe ahead of classical at lead 10 for both
targets, by 0.16 RMSE on Ap and 0.02 on Kp. It was written up at the time with
an explicit flag that a single 156-row test fold is thin evidence.

Densifying reversed both. Ap lead 10 went to classical ahead by 0.58, Kp to
classical ahead by 0.14. Both reversals exceed the original margins.

The finding is marked retracted in `results/embedding_probe.md` rather than
quietly edited out. The flag on the original was doing its job, and a retraction
that was pre-announced as a possibility is routine rather than embarrassing.

### Dst as a third target was scoped and dropped

Considered alongside Ap and Kp. Dropped for two reasons. It needs an external
data source not already cached locally, and Kp had already reproduced the Ap
pattern almost line for line. That reproduction is expected rather than
confirmatory, since Ap is close to a linear rescaling of averaged Kp, so a third
correlated target would not have added independent evidence. Dst is genuinely
different, being a ring-current index, which is exactly why it needs a new source.

### The pooling confound, found last

Found on 2026-08-19, after everything above was complete, while checking whether
a proposed next step was cheap or expensive.

Every stored embedding is a flat 1,280-vector. The extraction line is:

```python
pooled = tokens.mean(dim=1).squeeze(0).cpu().numpy()
```

Dimension 1 of `[1, 65536, 1280]` is the spatial token axis. The reduction
averages across the entire solar disk, which is the one operation most likely to
destroy the coronal-hole geometry the whole project was built to test.

This does not invalidate any number in `results/`. It bounds what those numbers
mean. Full treatment as limitation 1 in [limitations.md](limitations.md), and
the resolution path is step 1 of [future-work.md](future-work.md), which costs an
afternoon and no GPU time.

---

## Part 2: the findings page

### Every photo-based Earth failed

The hero shows Earth's limb rising as the Sun docks into a corner badge.
Photographic sources could not satisfy two requirements at once. A real
ISS-altitude horizon is nearly straight, so anything with visible curvature had
to be heavily upscaled and went soft, while anything sharp looked flat.

**Fix:** draw both Sun and Earth as pure CSS radial-gradient spheres
(`components/sun.tsx`, `components/earth.tsx`). Every band is a gradient stop
expressed as a fraction of the radius, so the result is resolution independent,
sharp at any device pixel ratio, and curvature collapses to one number,
`EARTH_DISC_VW`, which does not affect sharpness at all. The visible strip is
only the outermost few percent of the radius, which is why the interesting stops
all sit between 97 and 100 percent.

A second bug surfaced during that work: the atmosphere glow was an outward
`box-shadow` reaching about 36px above the limb, washing grey haze across the
lede text sitting above it. The planet cleared the text, its shadow did not.
Changed to an `inset` shadow, which cannot paint outside the circle.

### The hero seam: nine commits, eight of them wrong

A 1px horizontal line ran across the hero. It was visible at 100 percent browser
zoom and clean at 90 percent on a display at 125 percent OS scaling, giving
`devicePixelRatio: 1.25`. That signature points hard at a sub-pixel rendering
artifact, and every fix aimed at that theory failed.

| # | Attempt | Commit | Outcome |
|---|---|---|---|
| 1 | Draw Earth above the scrim so the scrim never dims the planet | `7692bbf` | No change |
| 2 | Fade the scrim to zero at the hero edge so stars do not step | `ac22ce4` | No change |
| 3 | Move the Earth rise inside the mask so the fade stays at the clip edge | `8b3a2c7` | No change |
| 4 | Lengthen the Earth fade so it dissolves rather than ending | `1977f07` | No change |
| 5 | Remove a full-width rule above the footer row | `a85d72c` | Real rule, correctly removed, line persisted |
| 6 | End both fades before the clip edge for fractional-DPR headroom | `88c7fa3` | No change |
| 7 | Drop `transform` and `will-change` to avoid GPU layer promotion | `478f557` | No change |
| 8 | Remove a second full-width rule at `top: 46%` | `78af5ce` | Real rule, correctly removed, line persisted |
| 9 | **Cap the headline so it cannot wrap** | `d1ce419` | **Fixed** |

**The actual cause.** The headline column was capped at `maxWidth: 720`, but the
longest line, `FOR WARNINGS EARTH'S`, needs about 1,020px at 82px in the
rendered face. All three headline spans wrapped to two lines each, so the `h1`
rendered 472px tall instead of 236px. The content stack overflowed the fixed
100vh stage by 57px, and `overflow: hidden` cut the footer row through the middle
of its glyphs. That cut, running the full viewport width, is the seam.

Measured at the moment of diagnosis, hero clipping at 624.88px while the footer
row ran from 592.05 to 642.15px, so 17.27px of it was being sliced off.

**Why the strongest clue misled.** DevTools' element picker kept selecting the
hero footer row, which was read as "the line is at that row's top edge". The row
genuinely straddles the clip edge, so the picker was correct and the
interpretation was wrong. The investigation then spent eight commits looking
above that row instead of at the boundary running through it.

**Fix.** `maxWidth` raised to 1100, font size changed to

```
clamp(30px, min(5.4vw, 12vh, calc((100vw - 460px) / 13)), 82px)
```

so the size is bounded by the width actually available and by viewport height
rather than by `vw` alone, plus `white-space: nowrap` on the three lines on
desktop so a wrap can never change the stage height again. Mobile keeps wrapping,
where the layout is auto height and wrapping is harmless.

The divisor 13 comes from measurement, not estimate. The rendered longest line is
about 12.4 times the font size, and 13 leaves margin for the transient wider
glyphs the scramble-text animation produces mid-reveal.

### Why it took so long to see

Three capture methods all reported the hero as clean while the bug was live.

- The **built-in preview pane** letterboxes the page into a corner of the frame,
  so a 1px line falls below the resolution of the capture entirely.
- **Chrome extension screenshots** downscale, 1568px wide from a 1920px physical
  frame, and the zoom action resizes the viewport, which silently broke the
  `vh`-based layout before capturing it. The captured page was no longer the page
  under test.
- **CDP screenshots** (`page.screenshot`) go through a different raster path than
  the compositor. Headless and headful captures came back byte-identical, which
  proves neither reflected the composited frame.

**What finally worked.** Drive Chrome with `puppeteer-core` at
`--force-device-scale-factor=1.25`, inject two magenta 1px markers at known
viewport y positions so the CSS-to-physical row mapping is exact and verifiable,
then capture the actual desktop with `PIL.ImageGrab` after
`ctypes.windll.shcore.SetProcessDpiAwareness(2)`. Analyse numerically instead of
by eye.

**The detector bug, which is the real lesson.** Even with true screen pixels, the
first detector searched for a row differing from **both** vertical neighbours in
the same direction. That finds a hairline. A clip edge is one-sided, content
above and nothing below, so it scored exactly zero every time and reported
"clean" across twelve viewport geometries and eight scroll positions.

Two detectors are needed:

- **two-sided**, for hairlines: `lum[y] - lum[y-1]` and `lum[y] - lum[y+1]` both
  exceed a threshold with the same sign
- **one-sided**, for cuts: `lum[y] - lum[y+1]` exceeds a threshold across a large
  fraction of columns

Better still, assert the geometry rather than inferring it from pixels. For
anything inside an `overflow: hidden` box, check
`child.bottom <= container.bottom - paddingBottom` across a sweep of viewport
sizes. That single assertion found in one run what pixel analysis had missed
repeatedly, and it is the verification now used for the fix across eight window
geometries from 880x620 to 1920x900.

### Two smaller real bugs found on the way

- `--surface` was `#06070a`, a dark blue-grey rather than black, so the hero read
  as slightly lifted against the true black of the sections below. Now `#000000`.
- Two full-width 1px `var(--line)` rules existed in the hero, one above the
  footer row and one at `top: 46%`. Both were genuine, both read as page seams,
  and both were correctly removed. Neither was the reported bug. The second was
  measured on screen at rgb(36,39,47) across 76 percent of the viewport width
  before removal, which is how it was distinguished from the gradient banding
  around it at under 1/255.

### Deploy 404s

The static Pages build logged two 404s on every visit.

- **`/_vercel/insights/script.js`.** The Vercel analytics beacon is served by
  Vercel's edge, so on a static export there is nothing to fetch. Now gated on
  `process.env.GITHUB_PAGES !== 'true'`, evaluated at build time in a server
  component, so a future Vercel deploy still gets analytics.
- **`/favicon.ico`.** The icons lived in `public/`, where Next emits no
  `<link rel="icon">` at all. With no icon declared, Chrome falls back to
  requesting `/favicon.ico` at the **domain root**, which a project Pages site
  can never serve because that path belongs to the user site. Moving them to
  `app/icon.svg` and `app/apple-icon.png` uses Next's file convention, which
  emits the link with the `/heliomag` basePath applied automatically.

Verified by serving the real static export under `/heliomag/` and recording every
network response. Zero failed or 4xx requests. Commit `962ce99`.

### Fixing the 404 surfaced a second, invisible bug

The icon file the `962ce99` fix made Next actually serve was never inspected
before that point, because it had been silently 404ing since the project began
and nobody had reason to open it. It turned out to be the placeholder v0.dev
logo mark left over from scaffolding the site with v0, a stylised "V0" glyph,
not heliomag branding. A separate `generator: 'v0.app'` meta tag had already
been removed in an earlier commit, but the logo file itself was never swapped.

So the 404 fix had a side effect: it took a wrong logo from invisible to
visible. Fixing a deploy bug uncovered a branding bug that the deploy bug had
been accidentally hiding.

**Fix.** Replaced `app/icon.svg` and regenerated `app/apple-icon.png` to match,
drawing from `components/sun.tsx`'s own gradient stops rather than inventing a
new mark: same colours, but opaque out to the edge, since the corona's
transparent falloff is designed for a 260px hero badge and disappears entirely
below 32px. A first attempt used a bold monospace "H" echoing the site's
uppercase mono labels, but at favicon size a thick blocky H reads as a hospital
or medical cross rather than a wordmark, so it was replaced with the sun orb.
Two further leftover files, `public/icon-dark-32x32.png` and
`icon-light-32x32.png`, turned out on inspection to be rasterised 32px copies
of the same v0 mark rather than real assets, and were deleted.

### Bumping the workflow actions, and the trap inside it

GitHub began force-running four actions on Node 24 with a deprecation warning.
Bumping them looked like a version-number tidy and was not.

`actions/upload-pages-artifact` v4 stopped bundling dotfiles by default.
`out/.nojekyll` is a dotfile. Losing it means GitHub Pages applies Jekyll
processing, and Jekyll ignores directories beginning with an underscore, which is
every Next.js asset in `_next/`. The HTML would still have loaded while all
JavaScript and CSS 404'd, so it would have presented as a styling bug rather than
a deploy bug.

Fix: bump to v5 and set `include-hidden-files: true`, restoring v3 behaviour.

Also checked and found not applicable: `actions/checkout` v5 changed
`pull_request_target` defaults, and this workflow only triggers on `push` and
`workflow_dispatch`. `actions/setup-node` v5 added automatic caching keyed on a
`packageManager` field in `package.json`, and there is none, so the explicit
`cache: npm` stays authoritative.

Final versions in `6b7c009`: checkout v7, setup-node v7, configure-pages v6,
upload-pages-artifact v5, deploy-pages v5.

---

## Lessons worth carrying forward

1. **Gate expensive work behind cheap falsifiable checks.** Three CPU checks ran
   before any GPU spend, each capable of killing the project. The negative result
   is informative precisely because the hypothesis had already survived every
   cheap way of being wrong.
2. **Flag thin evidence at the time, not in hindsight.** The lead-10 tie was
   labelled single-fold and thin when first written. When more data reversed it,
   the retraction was routine.
3. **Keep the bad numbers.** The MLP's catastrophic first pass only became
   evidence about sample size because it was still on the record to compare
   against.
4. **Design long jobs to be resumable before running them.** Checkpoint every N,
   resume by key, and log the resource that will actually run out. A 12-hour loss
   became a 1-hour recovery.
5. **A failure pattern is more diagnostic than a failure count.** Clustered gaps
   meant instrument outages. Scattered gaps would have meant a bug. The count
   alone, 30 of 1,416, said nothing either way.
6. **A confident diagnosis that survives eight failed fixes is probably wrong.**
   The sub-pixel-rendering theory explained the symptom well enough to keep
   generating plausible next attempts. What broke the loop was measuring the
   layout instead of the pixels.
7. **Match the detector to the artifact.** A hairline and a cut are different
   shapes in the data, and a test for one is blind to the other.
8. **Read the release notes on major version bumps.** One of five in the workflow
   bump would have broken the site in a way that looked like something else
   entirely.

---

## Appendix: commit map

Research pipeline:

| Date | Commit | What |
|---|---|---|
| 2026-07-30 | `a09218e` | Pilot Kp/Ap correlation and classical baseline |
| 2026-07-30 | `6eb7ec7` | 27-day spectral gate |
| 2026-07-30 | `d63ccb5` | Colab notebook validating Surya inference |
| 2026-07-30 | `1da3798` | Embedding extraction confirmed on Colab T4 |
| 2026-08-11 | `9ec428f` | Merged weekly embeddings, first probe, negative |
| 2026-08-11 | `678de2b` | Residual-correction and Kp follow-ups, still negative |
| 2026-08-12 | `7d2b83b` | Densify training window, 598 to 1,785 rows |
| 2026-08-12 | `2b6d197` | Densified results, lead-10 finding retracted |

Findings page, seam and deploy work:

| Commit | What |
|---|---|
| `f46d52a` | GitHub Pages static export and deploy workflow |
| `7692bbf` … `478f557` | Seven failed seam fixes, see table above |
| `78af5ce` | Removed the second full-width rule at `top: 46%` |
| `d1ce419` | Headline sizing fix, the actual seam |
| `962ce99` | Icons served, Vercel beacon dropped from the Pages build |
| `6b7c009` | Workflow actions bumped off the Node 20 runtime |
| `8fccc12` | This reports folder |
