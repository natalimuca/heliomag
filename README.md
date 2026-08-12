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
- Batch embedding extraction done: weekly Surya embeddings 2010-05 to 2024-12, plus the training window (2010-05 to 2021-12) densified to every-3-days. 1941 embeddings total, 1785 training rows (3.0x the original 598). Remaining gaps (~2%, both weekly and dense) are genuine SDO source-data holes, not a pipeline issue. 2025 blocked upstream — the NASA Surya benchmark S3 bucket has no 2025 data yet. `data/embeddings_merged.npz`.
- Embedding-vs-classical probe done (PCA+Ridge, PCA+Ridge+classical, PCA+MLP, CV-selected hyperparameters, same grid for classical and embedding features), plus three follow-up checks: residual correction on the classical fit, Kp as an alternate target, and 3x-denser training data — see `results/embedding_probe.md`. **Negative result, holds under all three follow-ups:** classical baseline now beats the embedding probe at every lead time for both Ap and Kp (the one earlier "tie" at lead 10 reversed with more data — retracted, was thin single-fold noise). Denser training data measurably narrowed the gap at some lead times (Ap/Kp lead 5 especially) and fixed the previously-broken MLP variant, real evidence sample size was a partial constraint — but the pattern was mixed, not uniform, and nothing crossed over to beat classical.
- Next: no small next step left that's clearly worth it — full daily density is a much larger compute commitment (~38h) for an uncertain payoff given the mixed densification result. If pursued further, more promising than brute-forcing density: revisit embedding pooling/resolution rather than just adding more of the same weekly-derived signal.
