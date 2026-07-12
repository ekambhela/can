# Karkive accuracy study — baseline → ensemble

Goal: maximize predictive accuracy of the GDSC drug-response model **without
cheating on the split**.

## Rules held for every experiment

- **Split by cell line** — a line is never in both train and test.
- **Fit all target scaling / feature stats on TRAIN only** (no leakage). The
  per-drug `-z(logIC50)` scaler uses train-line mean/std; the <1%-prevalence
  feature filter uses train prevalence.
- **Tune on validation, never on test.** A 64/16/20 train/val/test split (by
  cell line, seed 0); model and feature choices are made on VAL, and TEST is the
  honest final estimate. (The shipped production model uses 80/20 train/test and
  refits on all lines — numbers below are the harness's 632-train VAL/TEST unless
  marked "production".)
- Metrics are computed in the same train-z-scored sensitivity space so every
  architecture is directly comparable.

Reproduce: `python -m experiments.run_baseline` · `run_tier1` · `run_tier2_mt` ·
`run_tier2b` · `run_mf` · `run_tier4` · `run_ensemble`.

## Results (harness, decisions on VAL, confirmed on TEST)

| # | Change | split | Spearman | R² | top1 | top3 | top10 | pct |
|---|---|---|---|---|---|---|---|---|
| — | **Baseline** per-drug, curated 15 feats | VAL | 0.310 | 0.121 | 0.044 | 0.082 | 0.183 | 0.691 |
| | | TEST | 0.316 | 0.147 | 0.091 | 0.131 | 0.212 | 0.699 |
| T1 | full 525 genomic feats (train-filtered) | VAL | 0.287 | 0.097 | 0.013 | 0.044 | 0.120 | 0.635 |
| T1 | curated + pathway-active flags | VAL | 0.300 | 0.111 | 0.025 | 0.089 | 0.165 | 0.686 |
| T2 | multi-task, curated, drug=pathway+targets | VAL | 0.309 | 0.074 | 0.025 | 0.063 | 0.171 | 0.673 |
| T2 | multi-task, full feats | VAL | 0.264 | 0.057 | 0.019 | 0.063 | 0.183 | 0.624 |
| T2 | **multi-task + capped drug identity** | VAL | 0.342 | 0.099 | 0.019 | 0.070 | 0.183 | 0.708 |
| T2.5 | matrix factorization (K=32, side-info) | VAL | 0.334 | 0.108 | 0.032 | 0.063 | 0.165 | 0.670 |
| T4 | multi-task + drugID, tuned | VAL | 0.348 | 0.099 | 0.019 | 0.076 | 0.196 | 0.712 |
| | | TEST | 0.368 | 0.162 | 0.091 | 0.136 | 0.212 | 0.715 |
| T4 | **ENSEMBLE 0.35·per-drug + 0.65·mtl** | VAL | 0.355 | 0.131 | 0.032 | 0.089 | 0.209 | 0.724 |
| | **(shipped)** | TEST | **0.378** | **0.179** | 0.091 | **0.151** | **0.217** | **0.724** |

### Production (80/20 by cell line, refit on all lines — apples-to-apples)

| production model | Spearman | R² | top1 | top3 | top10 | pct |
|---|---|---|---|---|---|---|
| old per-drug | 0.336 | 0.157 | 0.101 | 0.162 | **0.247** | 0.730 |
| **shipped ensemble** | **0.382** | **0.179** | **0.111** | **0.177** | 0.232 | **0.750** |

### Honesty check — tissue-blocked CV (hold out whole tissues)

| model | Spearman | R² | top10 | pct |
|---|---|---|---|---|
| per-drug | 0.048 | −0.08 | 0.131 | 0.668 |
| ensemble | 0.074 | −0.05 | 0.117 | 0.663 |

## What worked, what didn't (one-liners)

- **Full feature matrix (T1): did NOT help** — 525 features overfit the
  data-starved per-drug models (~600 lines each); the curated 15 is a better
  inductive bias. *More features only pay off with more samples.*
- **Pathway-active flags (T1): no gain** — redundant with the curated drivers.
- **Multi-task alone (T2): a wash** — GDSC is 82% dense, so per-drug models
  aren't starved; representing drugs only by properties *lost* per-drug fidelity.
- **Multi-task + capped drug identity (T2): the real jump** — identity restores
  per-drug fidelity while the shared mapping (learned over ~200k pairs)
  generalizes: Spearman 0.316 → 0.368 on TEST.
- **Matrix factorization (T2.5): strong second** — beat the baseline (0.334 vs
  0.310), confirming the matrix-completion framing helps, but lost to the GBM.
- **Ensemble (T4): the winner** — blending per-drug (calibration) with
  multi-task (ranking) dominated both on TEST across every metric.
- **Tuning (T4): marginal** — a small bump; architecture and the ensemble did
  the heavy lifting.

## Not pursued (honest gaps)

- **AUC target (T3.6):** the repo ships only logIC50; the GDSC fitted-dose-
  response tables that carry AUC are behind hosts blocked in this environment, so
  AUC was not tested. Plausible future win.
- **Cross-screen denoise (T3.7):** after de-duplication GDSC2 only *adds*
  compounds absent from GDSC1, so the two screens share no drugs to correlate.
  The per-drug predictability breakdown below is the practical substitute.

## Per-drug predictability (ensemble, TEST)

270 / 369 drugs are well-predicted (Spearman > 0.3); only 15 are near-random.
The best-predicted are exactly the biomarker-driven classes —
**MEK/ERK inhibitors** (Refametinib, Trametinib, CI-1040), **p53/MDM2**
(Nutlin-3a), **WEE1** (MK-1775) — while broad cytotoxics with no clean biomarker
(Doxorubicin, Erlotinib) are near-random. The model recovers real biology, and
the ranked list is most trustworthy for targeted agents.

## Bottom line

Rank correlation improved **~14%** (0.34 → 0.38) with a clean, leakage-free,
split-by-cell-line evaluation, driven by (1) a multi-task model that borrows
strength across drugs and (2) ensembling it with the per-drug baseline. The big
caveat, stated plainly: hold out **whole tissues** and performance collapses —
the model leans on tissue context and does not yet extrapolate to unseen tissues.
