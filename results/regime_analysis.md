# Spectral check: does the classical baseline explain the 27-day recurrent (coronal-hole) signal?

Lead time 14 days, test period 2022-2025 (n=1653), same classical baseline as `baseline_probe.md` (SN+F10.7+persistence, RMSE 14.64).

Coronal holes are long-lived and rotate with the Sun (~27-day Carrington period), so their geomagnetic signature is periodic — unlike flare/CME-driven storms, which are sporadic. If classical scalar indices (F10.7, sunspot number) are blind to coronal-hole geometry, the 27-day periodic component should survive essentially untouched in the model's residuals.

| Signal | Fraction of spectral power in 25-29 day band |
|---|---|
| Raw Ap (actual) | 3.00% |
| Residual (actual - classical prediction) | 3.68% |

The residual retains slightly *more* of the 27-day band than the raw signal, not less. The classical model removes essentially none of the recurrent component — if it removed any, the residual fraction would drop well below the raw fraction. This directly supports the project's central rationale: whatever structure Surya's spatial view of coronal holes could add, it wouldn't be redundant with what F10.7/SN already capture, because they capture ~none of it.

**Caveat:** 3% of total spectral power in a single band is a modest absolute signal (most of Ap's variance is broadband, not periodic — real storms are messier than a clean 27-day recurrence). This is a coarse periodogram-ratio sanity check, not a rigorous significance test (no Lomb-Scargle false-alarm-probability estimate, single test-period window). Treat as: hypothesis survives a cheap gate, not proof. Worth revisiting with a proper significance test once real embeddings exist to compare against directly.

**Decision:** proceeds to Surya embedding extraction. Three independent cheap checks now support the project (weak-but-real classical signal, persistence decay defining a 3-14 day window, and this untouched recurrent-structure result) — worth the GPU/cloud investment.
