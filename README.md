# heliomag

Does Surya (NASA/IBM's heliophysics foundation model) encode information about geomagnetic activity that classical scalar solar indices (F10.7, sunspot number) miss?

Surya's own evaluation covers solar wind speed at a single 4-day lead time, benchmarked only against other neural nets. It never compares against classical indices, never predicts geomagnetic indices (Kp/Ap/Dst) directly, and never tests multiple lead times. This project fills that gap, focused on the 3-14 day window where a 1-day persistence baseline has decayed but classical indices are still weak (~2-3.5% variance explained, confirmed against real data — see `results/pilot_kp.md`).

## Structure

- `scripts/` — data fetching, baseline probes, embedding extraction
- `data/` — cached raw indices (GFZ Kp/Ap/SN/F10.7, SDO imagery references)
- `notebooks/` — exploration
- `results/` — metrics, plots, pilot findings

## Status

- Pilot correlation check done (classical indices vs. geomagnetic Ap, full 1932-2026 record and 2010+ SDO era). Signal confirmed real but weak — see `results/pilot_kp.md`.
- Classical-index baseline probe done (persistence vs. SN+F10.7 vs. combined vs. MLP, lead times 3-14 days) — see `results/baseline_probe.md`. Target for Surya to beat: ~14.3-14.6 RMSE.
- Spectral check done: classical baseline leaves the 27-day coronal-hole recurrence signal untouched — see `results/regime_analysis.md`.
- Surya embedding extraction pipeline validated end-to-end on Colab T4 (`notebooks/surya_pipeline_check.ipynb`): weights + SDO data download, GPU forward pass, and `finetune=True` embedding extraction all confirmed working, real tensor shape `[1, 65536, 1280]`.
- Next: batch embedding extraction across the 2010-2025 SDO record (Kaggle, for GPU-hour budget), then train the actual embedding-vs-classical-index probes.
