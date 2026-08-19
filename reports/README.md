# Reports

Write-ups covering the project end to end: the paper, what it rests on, what
broke along the way, what the result does not establish, and what to do next.

These are the narrative layer. The numbers themselves live in `results/`, one
file per pipeline stage, each traceable to a runnable script in `scripts/`.

| File | Words | What it covers |
|---|---|---|
| [paper.md](paper.md) | ~5,700 | The paper. Abstract, motivation, data provenance, full methods including exact estimator configuration, all result tables for both passes and both targets, discussion, threats to validity, four appendices. |
| [background.md](background.md) | ~1,700 | What the work rests on: what Kp, Ap, SN and F10.7 actually are, the coronal-hole physics, Surya's architecture as it pertains to embeddings, the gap in its published evaluation, how the 3 to 14 day window was derived, what was rejected first. |
| [engineering.md](engineering.md) | ~3,500 | Build log. Sentinel handling, the `finetune` flag, the index bug, Kaggle's session cap, the gap forensics, the retraction, the nine-commit hero seam, the deploy 404s, the workflow bump trap. Eight lessons and a commit map. |
| [limitations.md](limitations.md) | ~1,800 | Eight limitations ordered by severity, each with what it does and does not affect. The pooling confound is limitation 1. |
| [future-work.md](future-work.md) | ~1,900 | Four next steps ranked by expected value per hour, with code sketches, storage arithmetic, failure conditions, a sequence diagram and a pre-commitment. |

## The result in one line

Classical scalar indices beat a Surya embedding probe at every lead time tested,
for both Ap and Kp, and the honest caveat is that the extraction step
mean-pooled away the spatial structure the whole hypothesis was about.

## The numbers, condensed

| | |
|---|---|
| Lead times tested | 3, 5, 7, 10, 14 days |
| Targets | Ap and Kp |
| Model families compared | 6, including persistence |
| Comparisons where classical wins | 10 of 10 |
| Classical baseline RMSE, daily grid | 14.34 to 14.64 across the whole window |
| Embeddings extracted | 1,941, 1,280-dim, 2010-05 to 2024-12 |
| Training rows / test rows | 1,785 / 156 |
| Weekly coverage | 754 of 764 slots, 98.7 percent |
| Findings retracted | 1, the lead-10 apparent win |

## Reading order

- **Just want the finding.** `paper.md` sections 4.6 and 5.
- **Deciding whether to trust it.** `limitations.md` first, then `paper.md`
  section 3 for how it was actually measured.
- **Picking the work back up.** `future-work.md`. Step 1 costs an afternoon and
  no GPU time, and its outcome decides everything after it.
- **Reproducing it.** `paper.md` section 7, plus Appendix A for the exact grids
  and Appendix D for which file does what.
- **Curious what went wrong.** `engineering.md`.

## A note on the caveat

The paper does not present the negative result at face value. Section 5.4 and
limitation 1 both state that `tokens.mean(dim=1)` averages away the spatial
structure the hypothesis was about, so the finding is sound about *this pipeline*
and overstated as a claim about Surya.

That is deliberate. It is the objection a reviewer raises first, and it was found
by reading the extraction code rather than by anyone pointing it out. Raising it
here is stronger than having it raised elsewhere, and it converts a flat null
into a specific, testable next experiment.

## Related

- `results/` for the raw stage-by-stage numbers and their original readings
- `scripts/` for the runnable pipeline
- `notebooks/`, `kaggle/` for the two GPU-dependent stages
- Findings page: https://natalimuca.github.io/heliomag/
