# Reports

Write-ups covering the project end to end: the paper, what it was based on, what
broke along the way, what the result does not establish, and what to do next.

These are the narrative layer. The numbers themselves live in `results/`, one
file per pipeline stage, each traceable to a script in `scripts/`.

| File | What it covers |
|---|---|
| [paper.md](paper.md) | The paper. Abstract, motivation, data, methods, results, discussion. Start here. |
| [background.md](background.md) | What the work rests on: the physics, the gap in Surya's own evaluation, why the 3 to 14 day window, what was rejected before this. |
| [engineering.md](engineering.md) | Build log. The Kaggle session cap, the SDO gaps, the retracted finding, the hero seam that took nine commits, the deploy 404s. |
| [limitations.md](limitations.md) | What the result does and does not establish, ordered by severity. The pooling issue is limitation 1. |
| [future-work.md](future-work.md) | Four candidate next steps ranked by expected value per hour, with a suggested sequence and a pre-commitment. |

## The result in one line

Classical scalar indices beat a Surya embedding probe at every lead time tested,
for both Ap and Kp, and the honest caveat is that the extraction step
mean-pooled away the spatial structure the whole hypothesis was about.

## Reading order

- **Just want the finding:** `paper.md`, sections 4 and 5.
- **Deciding whether to trust it:** `limitations.md` first, then `paper.md`.
- **Picking the work back up:** `future-work.md`, then step 1 costs an afternoon
  and no GPU time.
- **Curious what went wrong:** `engineering.md`.

## Related

- `results/` for the raw stage-by-stage numbers
- `scripts/` for the runnable pipeline
- Findings page: https://natalimuca.github.io/heliomag/
