# heliomag

**Does Surya, NASA/IBM's heliophysics foundation model, encode information about
geomagnetic activity that classical scalar solar indices (sunspot number, F10.7)
miss?** 

Surya's own published evaluation stops at a single 4-day-lead solar-wind-speed
forecast, benchmarked only against other neural nets — it never compares against
classical indices, never predicts a geomagnetic index directly, and never tests
multiple lead times. This project builds that comparison from scratch: extract
Surya's embeddings from real SDO imagery, probe them against 90 years of ground-truth
geomagnetic activity (Ap/Kp, recorded continuously since 1932) at 3–14 day lead
times, and score them head-to-head against a classical linear baseline on identical
data. The result is negative and it holds up: **classical indices beat the embedding
probe at every lead time tested, for both Ap and Kp, even after tripling the training
data.** Reported as-is, not reframed into a win — see [Where this leaves it](#where-this-leaves-it)
for what the negative result does and doesn't rule out.

## Technology Stack

![Python](https://img.shields.io/badge/Python-3-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-Surya_inference-EE4C2C)
![scikit-learn](https://img.shields.io/badge/scikit--learn-PCA_%2F_Ridge_%2F_MLP-F7931E)
![NumPy](https://img.shields.io/badge/NumPy-latest-013243)
![pandas](https://img.shields.io/badge/pandas-latest-150458)
![SciPy](https://img.shields.io/badge/SciPy-periodogram-8CAAE6)
![Kaggle](https://img.shields.io/badge/Kaggle-GPU_extraction-20BEFF)
![Jupyter](https://img.shields.io/badge/Jupyter-pipeline_validation-F37626)
![Next.js](https://img.shields.io/badge/Next.js-16-000000)
![React](https://img.shields.io/badge/React-19-61DAFB)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-4-06B6D4)

| Area | Tools |
|---|---|
| Data & ML | Python 3, pandas, NumPy, scikit-learn (PCA, Ridge, MLPRegressor, `GridSearchCV`, `TimeSeriesSplit`), SciPy (`periodogram`) |
| Foundation model | Surya (NASA/IBM heliophysics model), PyTorch, run via Kaggle GPU notebooks and validated on Colab T4 |
| Ground truth | GFZ Potsdam combined Kp/Ap/SN/F10.7 record (1932–present, CC BY 4.0), fetched directly, no manual download |
| Findings page | Next.js 16, React 19, TypeScript, Tailwind CSS 4 (`app/`, `components/`) |

Analysis logic lives in standalone `scripts/`, one per pipeline stage — every number
in this README and on the findings page traces back to a script and a saved
`results/*.md` writeup, not a notebook cell run once and forgotten. `notebooks/` and
`kaggle/` hold the two GPU-dependent stages (pipeline validation, batch embedding
extraction) that don't run on a local machine.

## Core idea

Surya reads solar imagery (EUV active-region views from SDO) directly, at whatever
spatial structure the imagery contains — coronal holes, active-region geometry, the
things a single scalar index like sunspot number or F10.7 necessarily throws away by
construction. Classical indices are known to be weak predictors of geomagnetic
activity in the 3–14 day window specifically because they're blind to coronal-hole
geometry, which drives high-speed-stream-related activity independently of
active-region emission. If Surya's embeddings carry that spatial signal, they should
beat a classical baseline in exactly this window. If they don't, that's a real,
useful negative result about a real foundation model, not a strawman comparison.

## Research question

Does Surya's embedding space carry geomagnetic-activity signal that classical scalar
indices don't, at the 3–14 day lead times where persistence has decayed and classical
indices are known-weak?

Headline findings:

- **The classical-index signal is real but small**: sunspot number and F10.7 correlate
  with Ap at r ≈ 0.15–0.19 (full 1932–2026 record, n in the tens of thousands),
  explaining only ~2–3.5% of daily Ap variance — small, but statistically robust, not
  a small-sample artifact.
- **Persistence decays fast**: 1-day Ap autocorrelation is r = 0.461, but falls below
  the classical-index level by day 3–4 (r = 0.100 at 3 days). That crossover defines
  the project's 3–14 day window — where persistence has decayed and classical indices
  are the bar to clear, leaving room for a richer signal to matter.
- **The classical baseline is solid, not a strawman**: a linear SN+F10.7+persistence
  model holds ~14.3–14.6 RMSE across 3–14 days; a small MLP on the same features
  doesn't improve on it — the classical relationship is close to linear, with no
  obvious nonlinear structure left on the table.
- **Classical baseline leaves the 27-day coronal-hole recurrence signal essentially
  untouched**: raw Ap carries 3.00% of its spectral power in the 25–29 day band (the
  Sun's rotation period); the classical model's residual retains *slightly more*,
  3.68% — meaning it removes almost none of that periodic structure. This is the
  hypothesis that motivated extracting embeddings at all.
- **The embedding probe loses anyway.** 1,941 Surya embeddings extracted (weekly,
  2010–2024, plus the training window densified to every 3 days, 1,785 training rows —
  3× the original 598), PCA+Ridge probed against Ap and Kp at 3–14 day leads.
  **Classical beats the embedding probe at every lead time, for both targets, with no
  exceptions** — the one apparent tie in the earlier weekly-only pass reversed after
  densifying and is retracted.
- **Tripling the data measurably helped, but not enough**: the classical-vs-embedding
  gap at lead 5 narrowed sharply (Ap: 1.89 → 0.27 RMSE, 86% narrower; Kp: 0.45 → 0.04,
  91% narrower) — real evidence sample size was a genuine constraint. Other lead times
  barely moved. Not a clean story either way; see
  [Where this leaves it](#where-this-leaves-it).

## How It Works

1. **Pilot correlation** — confirm classical indices (SN, F10.7) correlate with Ap at
   all, and that persistence decays fast enough to leave a useful 3–14 day window.
   Cheap, no small-sample risk (n in the thousands+), gates everything after it.
2. **Classical baseline** — fit the actual number Surya has to beat: persistence vs.
   linear SN+F10.7 vs. linear+persistence vs. a small MLP, same chronological
   train/test split used everywhere downstream.
3. **Spectral check** — a periodogram sanity check on whether the classical baseline's
   *residual* still carries the 27-day coronal-hole recurrence signal. If it does,
   that's specifically the kind of structure a spatial model like Surya could plausibly
   explain, and the reason to bother extracting embeddings at all.
4. **Pipeline validation** — confirm Surya's weights, SDO data download, GPU forward
   pass, and `finetune=True` embedding extraction all actually work end-to-end
   (`notebooks/surya_pipeline_check.ipynb`, Colab T4) before committing to a full batch
   extraction run.
5. **Batch embedding extraction** — weekly Surya embeddings over the full 2010–2024
   SDO era, training window densified to every 3 days, run as Kaggle GPU notebook
   sessions (`kaggle/heliomag-embedding-extraction.ipynb`).
6. **Embedding-vs-classical probe** — PCA+Ridge (and PCA+MLP, and a residual-correction
   variant) on the embeddings, CV-selected hyperparameters via `GridSearchCV` over
   `TimeSeriesSplit`, scored against the identical classical baseline and split.
7. **Follow-ups** — the same probe re-run with Kp as an alternate target, and with 3×
   the training data, specifically to stress-test whether the negative result was a
   sample-size artifact.

## Key Results

### Pilot correlation & persistence decay

See [`results/pilot_kp.md`](results/pilot_kp.md). SN/F10.7 vs. Ap: r ≈ 0.15–0.19
across 0–14 day lags, full 1932–2026 record and 2010+ SDO era agree closely. Ap
1-day autocorrelation r = 0.461, decaying to r = 0.100 by day 3. No small-sample
problem — this pilot explicitly replaced an earlier NAO/tropospheric-teleconnection
pilot that was abandoned for exactly that reason (n=16).

### Classical baseline

See [`results/baseline_probe.md`](results/baseline_probe.md). SDO-era chronological
split (train ≤ 2021-12-31, test ≥ 2022-01-01):

| lead (days) | persistence | SN+F107 | SN+F107+persistence | MLP (all features) |
|---|---|---|---|---|
| 3  | 19.60 | 14.44 | **14.34** | 14.58 |
| 5  | 19.77 | 14.52 | **14.47** | 14.80 |
| 7  | 20.18 | 14.61 | **14.56** | 14.81 |
| 10 | 20.13 | 14.68 | **14.59** | 15.28 |
| 14 | 20.25 | 14.71 | **14.64** | 15.02 |

Linear SN+F107+persistence is the strongest classical baseline at every lead; the MLP
doesn't improve on it.

### Spectral check

See [`results/regime_analysis.md`](results/regime_analysis.md). 25–29 day band share
of spectral power: raw Ap 3.00%, classical model's residual 3.68% — the classical fit
removes essentially none of the 27-day recurrent component.

### Embedding-vs-classical probe

See [`results/embedding_probe.md`](results/embedding_probe.md). Densified training set
(1,785 rows), held-out RMSE, classical vs. Surya embedding probe (PCA+Ridge):

**Target Ap** — classical / embedding: 3d 8.43/8.70 · 5d 15.32/15.59 · 7d 12.34/14.46
· 10d 8.41/9.00 · 14d 12.36/12.92
**Target Kp** — classical / embedding: 3d 0.87/0.94 · 5d 1.10/1.14 · 7d 1.08/1.43 ·
10d 0.88/1.03 · 14d 1.08/1.27

Classical wins every lead time, both targets, no exceptions. A residual-correction
variant (embeddings explain only what the classical fit gets wrong) narrows the gap
at some leads but never beats classical outright.

## Where this leaves it

Tripling the training set (598 → 1,785 rows) didn't flip the headline result, but it
wasn't a wash either — the lead-5 gap between classical and the embedding probe
shrank sharply (Ap: 1.89 → 0.27 RMSE; Kp: 0.45 → 0.04), real evidence that sample size
was a genuine, partial constraint. Other lead times barely moved, so the pattern is
mixed, not uniform, and nothing crossed over to beat classical.

**No small next step left that's clearly worth it.** Full daily-density extraction is
a much larger compute commitment (~38h) for an uncertain, likely-partial payoff given
the mixed densification result. If this is picked up again, the more promising lever
is revisiting embedding pooling/resolution — not just adding more of the same
weekly-derived signal.

## Data

- [GFZ Potsdam combined Kp/Ap/SN/F10.7 record](https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt) —
  1932–present, CC BY 4.0, fetched directly by `scripts/fetch_data.py`, cached to
  `data/gfz_kp_ap_sn_f107.csv`.
- SDO EUV active-region imagery, 2010–2024, accessed through the NASA Surya benchmark
  S3 bucket during embedding extraction (not stored locally — only the resulting
  embeddings are). 2025 has no data yet in that bucket; blocked upstream, not a
  pipeline issue.
- Surya embeddings: `data/embeddings_merged.npz` — 1,941 weekly embeddings
  (2010-05–2024-12), training window (2010-05–2021-12) additionally densified to
  every 3 days, 1,785 training rows. 754 of 764 expected weekly slots present (98.7%);
  the remaining gaps are confirmed genuine SDO source-data holes (multi-day instrument
  outages), not a pipeline bug.

## Model

**Surya** — NASA/IBM's heliophysics foundation model, reading SDO EUV imagery
directly. Embedding extraction validated end-to-end on Colab T4
(`notebooks/surya_pipeline_check.ipynb`): weights + SDO data download, GPU forward
pass, and `finetune=True` embedding extraction all confirmed working, real tensor
shape `[1, 65536, 1280]`. Batch extraction runs as Kaggle GPU notebook sessions
(`kaggle/heliomag-embedding-extraction.ipynb`) rather than one long job, since a single
weekly-extraction pass exceeds Kaggle's own session time cap.

## Repo Structure

```
scripts/         standalone runnable pipeline stages
  fetch_data.py        GFZ Kp/Ap/SN/F10.7 fetch + cache
  baseline_probe.py    classical persistence / linear / MLP baseline
  regime_analysis.py   27-day coronal-hole spectral check
  embedding_probe.py   PCA+Ridge/MLP probe, embeddings vs. classical, both targets
data/            cached GFZ record + Surya embedding .npz files (gitignored where large)
notebooks/       surya_pipeline_check.ipynb — Colab T4 end-to-end pipeline validation
kaggle/          heliomag-embedding-extraction.ipynb — batch GPU embedding extraction
results/         pilot_kp.md, baseline_probe.md, regime_analysis.md, embedding_probe.md
app/, components/  Next.js findings page (this project's landing page / research log)
```

## Reproduce

```bash
pip install -r requirements.txt

# 1. Pilot correlation + persistence decay (fast, fetches and caches the GFZ record)
python scripts/fetch_data.py

# 2. Classical baseline — the number Surya has to beat
python scripts/baseline_probe.py

# 3. Spectral check — does the classical residual still carry the 27-day signal?
python scripts/regime_analysis.py

# 4. Embedding-vs-classical probe (needs data/embeddings_merged.npz — see Data above;
#    produced by the Kaggle notebook, not reproducible locally without a GPU)
python scripts/embedding_probe.py
```

Steps 1–3 run anywhere with the packages in `requirements.txt`, no GPU needed. Step 4
needs the Surya embeddings already extracted — that part runs on Kaggle
(`kaggle/heliomag-embedding-extraction.ipynb`, GPU required), not locally.

Findings page:

```bash
npm install
npm run dev   # http://localhost:3000
```
