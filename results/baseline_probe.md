# Classical-index baseline (the number Surya has to beat)

SDO era (2010-05 to 2026-07), chronological split: train <= 2021-12-31, test >= 2022-01-01.
Features: SN, F10.7, Ap(t), Ap 3d/7d rolling mean. Target: Ap(t+lead). RMSE in Ap units.

| lead (days) | persistence | SN+F107 | SN+F107+persistence | MLP (all features) |
|---|---|---|---|---|
| 3  | 19.60 | 14.44 | 14.34 | 14.58 |
| 5  | 19.77 | 14.52 | 14.47 | 14.80 |
| 7  | 20.18 | 14.61 | 14.56 | 14.81 |
| 10 | 20.13 | 14.68 | 14.59 | 15.28 |
| 14 | 20.25 | 14.71 | 14.64 | 15.02 |

## Reading

Confirms the pilot correlation finding directly in forecast-skill terms: persistence is worse than classical scalar indices at every lead time from 3 days out (matches the earlier correlation-decay result — persistence's r fell below F10.7/SN's by day 3-4). The linear "SN+F107+persistence" combination is the strongest classical baseline (~14.3-14.6 RMSE across the whole 3-14 day window) and a small MLP does not improve on it with this feature set — the relationship classical indices capture appears close to linear, no obvious nonlinear structure being left on the table by a simple regression.

**Target for Surya embeddings:** beat ~14.3-14.6 RMSE at 3-14 day lead times. If embeddings can't clear this bar, that's still a real (negative) result, not a null pilot — the baseline itself is solid, reproducible, and non-trivial to beat.
