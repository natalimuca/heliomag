# Engineering log: what broke and how it was fixed

A record of the problems that actually cost time, on both the research pipeline
and the findings page. Kept because several of them were diagnosed wrongly for a
while, and the wrong diagnoses are as instructive as the fixes.

Timeline: 2026-07-30 to 2026-08-19.

---

## Part 1: the research pipeline

### An earlier pilot was abandoned for sample size

Before this project existed, a pilot on NAO and tropospheric teleconnections was
carried far enough to discover that the usable sample was n = 16. It was
dropped rather than written up. The replacement was chosen for the opposite
property: the GFZ record gives tens of thousands of daily rows. This is why the
first thing the current pipeline does is confirm n, not confirm a correlation.

### Kaggle's session cap does not fit a full extraction pass

A single weekly extraction pass over 2010 to 2024 exceeds Kaggle's own session
time limit. The first densification attempt (session v9) ran roughly 12 hours
and was auto-cancelled by Kaggle at about 95 percent complete.

**Fix:** treat extraction as resumable batches rather than one long job. Session
v10 picked up the remaining 71 targets and finished in under an hour. The
merged output is `data/embeddings_merged.npz`.

### Missing dates that looked like a pipeline bug

The weekly grid came back with 754 of 764 expected slots, 98.7 percent. Ten
missing weeks scattered between 2012 and 2023 is exactly the pattern a flaky
fetch loop produces, so it could not be assumed benign.

**Diagnosis:** a dedicated re-run (session v8) re-attempted precisely those gaps
plus all of 2025. The 2023 to 2024 portion re-fetched successfully but every
value duplicated data already held, and all 52 weekly attempts in 2025 failed
identically with `Failed listing s3://nasa-surya-bench/2025/...`.

**Conclusion:** the gaps are genuine SDO source holes, multi-day instrument
outages, and 2025 is simply not mirrored in the benchmark bucket yet. Blocked
upstream, not locally. The 3-day densification hit the same gaps at finer
resolution, 30 of 1,416 targeted dates, clustered around the same outage dates.
That clustering is what confirmed the interpretation: a real multi-day outage
shows up as one missing week at coarse resolution and a short burst of missing
days at fine resolution, which is not what an intermittent bug looks like.

### The MLP was catastrophically bad, and that turned out to be informative

On the first pass with 598 training rows, PCA plus a small MLP scored worse than
naive persistence at every lead time, reaching 39.80 RMSE at Ap lead 14 against
a classical 12.44. The temptation was to treat it as a broken configuration.

It was not broken, it was starved. After densification to 1,785 rows the same
model went from 22.16 to 13.75 at Ap lead 7, beating both linear embedding
variants. Keeping the bad number in the record rather than deleting it is what
made the later improvement interpretable as evidence about sample size.

### A published finding was retracted

The weekly-only pass showed the embedding probe edging ahead of classical at
lead 10 for both Ap and Kp. It was written up at the time with an explicit flag
that a single 156-row test fold is thin evidence.

Densifying reversed both. Ap lead 10 went from embeddings ahead by 0.16 RMSE to
classical ahead by 0.58. The finding is marked retracted in
`results/embedding_probe.md` rather than quietly edited out. The flag on the
original was doing its job.

### Dst as a third target was scoped and dropped

Considered as an alternate target alongside Ap and Kp. Dropped because it needs
an external data source not already cached locally, and Kp had already
reproduced the Ap pattern almost line for line, which is unsurprising given Ap
is close to a linear rescaling of averaged Kp. A third correlated target would
not have added independent evidence.

---

## Part 2: the findings page

### Every photo-based Earth failed

The hero shows Earth's limb rising. Photographic sources could not satisfy two
requirements at once: a real ISS-altitude horizon is nearly straight, so
anything with visible curvature had to be heavily upscaled and went soft, while
anything sharp looked flat.

**Fix:** draw both Sun and Earth as pure CSS radial-gradient spheres
(`components/sun.tsx`, `components/earth.tsx`). Because every band is a gradient
stop expressed as a fraction of the radius, the result is resolution
independent, sharp at any device pixel ratio, and curvature becomes a single
number (`EARTH_DISC_VW`) that does not affect sharpness at all.

### The hero seam: nine commits, eight of them wrong

A 1px horizontal line ran across the hero. It was visible at 100 percent browser
zoom and clean at 90 percent on a 125-percent-scaled display, which strongly
suggested a sub-pixel rendering artifact. Every fix aimed at that theory failed:

| Attempt | Commit | Outcome |
|---|---|---|
| Draw Earth above the scrim | `7692bbf` | No change |
| Fade the scrim to zero at the hero edge | `ac22ce4` | No change |
| Move the Earth rise inside the mask | `8b3a2c7` | No change |
| Lengthen the Earth fade | `1977f07` | No change |
| Remove a full-width rule above the footer | `a85d72c` | Real rule, correctly removed, line persisted |
| End fades before the clip edge for fractional DPR | `88c7fa3` | No change |
| Drop GPU layer promotion on the clipped strip | `478f557` | No change |
| Remove a second full-width rule at `top: 46%` | `78af5ce` | Real rule, correctly removed, line persisted |
| **Cap the headline so it cannot wrap** | `d1ce419` | **Fixed** |

**The actual cause.** The headline column was capped at `maxWidth: 720`, but the
longest line, `FOR WARNINGS EARTH'S`, needs about 1,020px at 82px in the
rendered face. All three headline lines wrapped to two, so the `h1` rendered 472px
tall instead of 236px, the content stack overflowed the fixed 100vh stage by
57px, and `overflow: hidden` cut the footer row through the middle of its
glyphs. That cut, running the full width, is the seam.

This also explains the clue that misled the whole investigation. DevTools' element
picker kept selecting the footer row, which was read as "the line is at that
row's top edge". The row genuinely straddles the clip edge, so the picker was
correct and the interpretation was wrong.

**Fix:** `maxWidth` 1100, font size
`clamp(30px, min(5.4vw, 12vh, calc((100vw - 460px) / 13)), 82px)` so the size is
bounded by the width actually available and by viewport height rather than by
`vw` alone, and `white-space: nowrap` on the three lines on desktop so a wrap can
never change the stage height. Mobile keeps wrapping, where the layout is auto
height and wrapping is harmless.

### Why it took so long to see

Three capture methods all reported the hero as clean while the bug was live:

- The built-in browser preview letterboxes the page into a corner of the frame,
  so a 1px line is below the resolution of the capture.
- Chrome extension screenshots downscale, 1568px wide from a 1920px physical
  frame, and the zoom action resizes the viewport, which silently broke the
  `vh`-based layout before capturing it.
- CDP screenshots (`page.screenshot`) go through a different raster path than the
  compositor. Headless and headful captures came back byte-identical, so neither
  reflected the composited frame.

**What finally worked:** drive Chrome with `puppeteer-core` at
`--force-device-scale-factor=1.25`, inject two magenta 1px markers at known
viewport positions so the CSS-to-physical row mapping is exact, then capture the
actual desktop with `PIL.ImageGrab`. Analyse numerically instead of by eye.

**The detector bug.** Even with real pixels, the first detector looked for a row
differing from *both* vertical neighbours, which finds a hairline. A clip edge is
one-sided, content above and nothing below, so it scored exactly zero. Two
detectors are needed: a two-sided test for hairlines and a one-sided step test
for cuts. Better still, assert the geometry directly. For anything inside an
`overflow: hidden` box, check `child.bottom <= container.bottom - paddingBottom`
across a sweep of viewport sizes. That single assertion found in one run what
pixel analysis had missed repeatedly, and it is now the verification used for the
fix across eight window geometries.

### Two smaller real bugs found along the way

- `--surface` was `#06070a`, a dark blue-grey rather than black, which made the
  hero read as slightly lifted against the true black of the sections below. Now
  `#000000`.
- Two full-width 1px `var(--line)` rules existed in the hero, one above the
  footer row and one at `top: 46%`. Both were genuine and both read as page
  seams. Removing them did not fix the reported bug, but they were worth
  removing on their own terms.

### Deploy 404s

The static Pages build logged two 404s on every visit.

- `/_vercel/insights/script.js`. The Vercel analytics beacon is served by
  Vercel's edge, so on a static export there is nothing to fetch. Now gated on
  `GITHUB_PAGES !== 'true'` at build time, so a future Vercel deploy still gets
  analytics.
- `/favicon.ico`. The icons lived in `public/`, where Next emits no
  `<link rel="icon">` at all, so Chrome fell back to requesting `/favicon.ico` at
  the domain root, which a project Pages site can never serve. Moving them to
  `app/icon.svg` and `app/apple-icon.png` uses Next's file convention, which
  emits the link with the `/heliomag` basePath applied automatically.

Both verified by serving the real static export under `/heliomag/` and recording
every network response. Zero failed or 4xx requests. Commit `962ce99`.

---

## Lessons worth carrying forward

1. **Gate expensive work behind cheap falsifiable checks.** Three CPU checks ran
   before any GPU spend, each capable of killing the project. The negative result
   is informative precisely because the hypothesis had already survived every
   cheap way of being wrong.
2. **Flag thin evidence at the time, not in hindsight.** The lead-10 tie was
   labelled as single-fold and thin when first written. When more data reversed
   it, the retraction was routine rather than embarrassing.
3. **Keep the bad numbers.** The MLP's catastrophic first pass only became
   evidence about sample size because it was still on the record to compare
   against.
4. **A confident diagnosis that survives eight failed fixes is probably wrong.**
   The sub-pixel-rendering theory explained the symptom well enough to keep
   generating plausible next attempts. What broke the loop was measuring the
   layout rather than the pixels.
5. **Match the detector to the artifact.** A hairline and a cut are different
   shapes in the data, and a test for one is blind to the other.
