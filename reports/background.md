# Background: what this work is based on

Why this comparison was worth building, what it rests on, and what was rejected
along the way.

## The operational problem

Geomagnetic storms disturb Earth's magnetic field and degrade satellites, GPS
and power grids. The disturbance has been measured continuously since 1932 as
the Kp index, from which Ap is derived, giving one of the longest uninterrupted
geophysical records in existence. That length is the reason this project is
possible at all: it provides ground truth at n in the tens of thousands, which
removes small-sample risk from the classical side of every comparison.

Forecasting that disturbance days ahead is the hard part. The standard inputs
are scalar solar indices, principally sunspot number and the F10.7 cm radio
flux. Both compress the entire visible solar disk to a single daily number.

## Why scalar indices should be beatable

The limitation is structural rather than empirical. F10.7 and SN track
active-region emission. A large share of geomagnetic activity in the multi-day
window comes instead from high-speed solar wind streams emitted by coronal
holes, which are dark, low-density regions whose geoeffectiveness depends on
where they sit on the disk and when they rotate into an Earth-facing position.
No scalar summary of total emission can express that.

Two properties follow, and both were measured rather than assumed:

1. Coronal-hole activity is **periodic**, because the holes are long-lived and
   the Sun rotates on a roughly 27-day Carrington period. Flare and CME driven
   storms are sporadic by comparison.
2. If classical indices are blind to that geometry, a classical model's
   **residual should retain the 27-day component intact**.

Point 2 is a falsifiable prediction, and the spectral check in
`results/regime_analysis.md` tested it before any GPU spend. Raw Ap carries 3.00
percent of its spectral power in the 25 to 29 day band, and the classical
residual carries 3.68 percent. The classical fit removes essentially none of it.

## Why Surya, and what its own evaluation leaves open

Surya is NASA and IBM's heliophysics foundation model. It reads SDO extreme
ultraviolet imagery directly, at whatever spatial structure the imagery
contains, which is exactly the information a scalar index discards by
construction.

Its published evaluation is narrow in three specific ways, and each one is a gap
this project fills:

| Surya's own evaluation | What this project does |
|---|---|
| A single 4-day lead time | Five lead times, 3 to 14 days |
| Solar wind speed as target | Ap and Kp, the operational geomagnetic indices |
| Benchmarked against other neural networks | Benchmarked against classical indices |

The third is the important one. A foundation model beating other neural networks
says little about whether it beats the cheap scalar that operational forecasting
actually uses. That comparison had not been made, and making it is the
contribution here regardless of which way the result falls.

## Choosing the lead-time window

The 3 to 14 day window was not chosen for convenience. It follows from where
persistence stops dominating.

| lag (days) | Ap autocorrelation |
|---|---|
| 1 | 0.461 |
| 2 | 0.173 |
| 3 | 0.100 |
| 4 | 0.081 |
| 7 | 0.065 |

Persistence is overwhelming at 1 day and drops below the classical-index
correlation level (r roughly 0.15 to 0.19) between day 3 and day 4. Before day
3 there is nothing to add, because yesterday's value already carries the
information. After day 3 the bar is both low and non-trivial. That is the window
where a richer representation has room to matter.

## What was rejected before this

An earlier pilot studied NAO and tropospheric teleconnections. It was abandoned
because the usable sample was n = 16, which cannot support any claim. The
replacement was chosen specifically for the opposite property. The GFZ record
gives tens of thousands of daily rows, so the classical side of the comparison
is never sample-limited even though the embedding side turned out to be.

This matters for reading the final result. The negative finding is not a null
pilot that ran out of data on both sides. The baseline it failed to beat is
measured on a large, robust sample and holds 14.34 to 14.64 RMSE across the
whole window.

## The gating discipline

Three cheap checks ran before committing to GPU extraction, each one able to
kill the project:

1. **Do classical indices correlate with Ap at all?** Yes, r of 0.15 to 0.19,
   consistent between the full record and the SDO era.
2. **Does persistence decay fast enough to leave a window?** Yes, by day 3.
3. **Does the 27-day recurrent structure survive the classical fit?** Yes,
   essentially untouched.

Only after all three passed was the Colab pipeline validation run, and only
after that succeeded end to end (real tensor shape `[1, 65536, 1280]`) did batch
extraction begin. That ordering is why the eventual negative result is
informative: the hypothesis had already cleared every cheap way of being wrong
before the expensive test was run.

## Sources

- GFZ Potsdam combined Kp/Ap/SN/F10.7 record, 1932 to present, CC BY 4.0.
  https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt
- SDO EUV imagery via the NASA Surya benchmark S3 bucket, 2010 to 2024.
- Surya, NASA and IBM heliophysics foundation model.
