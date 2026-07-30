# heliomag

Does Surya (NASA/IBM's heliophysics foundation model) encode information about geomagnetic activity that classical scalar solar indices (F10.7, sunspot number) miss?

Surya's own evaluation covers solar wind speed at a single 4-day lead time, benchmarked only against other neural nets. It never compares against classical indices, never predicts geomagnetic indices (Kp/Ap/Dst) directly, and never tests multiple lead times. This project fills that gap, focused on the 3-14 day window where a 1-day persistence baseline has decayed but classical indices are still weak (~2-3.5% variance explained, confirmed against real data — see `results/pilot_kp.md`).

## Structure

- `scripts/` — data fetching, baseline probes, embedding extraction
- `data/` — cached raw indices (GFZ Kp/Ap/SN/F10.7, SDO imagery references)
- `notebooks/` — exploration
- `results/` — metrics, plots, pilot findings

## Status

Pilot correlation check done (classical indices vs. geomagnetic Ap, full 1932-2026 record and 2010+ SDO era). Signal confirmed real but weak — see `results/pilot_kp.md`. Next: classical-index baseline probes, then Surya embedding extraction (needs cloud GPU, none available locally).
