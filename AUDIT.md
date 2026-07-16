# Karkive — engineering & modeling audit

**Scope:** end-to-end review of the FastAPI service and the GDSC drug-response
model behind it, against a list of five suspected prior issues plus a modeling
assessment (feature representation, architecture, reproducibility).

**Method:** static read of every module in `model/`, `experiments/`, `app.py`,
and the test suite; plus empirical verification — the pinned stack was installed,
the suite was run (`28 passed`), the by-cell-line/tissue-blocked splits were
checked for overlap programmatically, and a full `python -m model.train` was
re-run to reproduce the shipped metrics.

**Headline:** the codebase has **already been remediated for most of the listed
issues.** Leakage is ruled out; the reported accuracy is *not* leakage-inflated.
The one clearly-open infrastructure item is the model binary committed to git
history. Details, with evidence, below. Severities reflect *current* state.

Legend: ✅ resolved · 🟡 partial / minor gap · 🔴 open.

---

## 1. Target leakage — ✅ RESOLVED (severity: none; was critical)

**Verdict: no leakage.** Verified statically and empirically.

- **Split is by cell line.** `model/train.py:185-188` permutes cell-line indices
  (`rng.permutation(n)` over `n = len(feats)`, one row per line) and slices
  train/test from that. Empirical check: train∩test cell lines = **0**; at the
  (cell line, drug) pair level, train∩eval lines = **0**.
- **Split is by cell line, *not* also by drug — and that is correct here.** The
  same drug appears in train and test on purpose: the modeling goal is to
  generalize to a *new tumor/cell line*, not to a *never-before-seen drug*. Drug
  identity is a deliberate shared feature (`model/mtl.py:31,66`). A by-drug
  (cold-start-drug) split answers a different question and would make the drug
  embedding meaningless; it is out of scope for this product. This is documented
  so the choice is not mistaken for an oversight.
- **No IC50-derived / post-hoc column leaks into X.** `model/gdsc.py:164-182`
  builds features from genomic flags + tissue only; the raw log-IC50 is returned
  *separately* as the target dict, never merged into `feats`. The multi-task
  design matrix (`model/mtl.py:83-108`) is exactly
  `[genomic flags, drug target-token multi-hot (dt_*), tissue, drug_pathway,
  drug_id]` — confirmed by dumping `X.columns`. `drug_pathway` is the drug's
  *target pathway* (a drug property, e.g. "ERK MAPK signaling"), **not** the IC50
  response.
- **Target scaling is fit on TRAIN lines only.** `fit_target_scaler`
  (`model/train.py:46-57`) indexes `y[train_idx]`; `_build_truth`
  (`model/train.py:65-70`) is called *separately* per split (`tr`, `trb`,
  `all_idx`), so held-out statistics never influence the `-z(logIC50)` target.
  This is the exact prior bug, and it is pinned by a regression test
  (`tests/test_scaler_leakage.py`).
- **Honest generalization gap is already reported.** Random-line held-out:
  mean per-drug Spearman ≈ **0.38**. Tissue-blocked (hold out whole tissues):
  mean Spearman ≈ **0.07**, R² slightly negative
  (`experiments/RESULTS.md`, and `metrics["tissue_blocked"]` in the bundle). The
  model leans heavily on tissue context and does not extrapolate to unseen
  tissues — stated plainly in README and UI.

**Minor note (not leakage):** HistGBM early stopping carves an internal
validation slice from the *training* pairs (`model/mtl.py:32-35`,
`validation_fraction=0.1`). That split is by pair, not by line, but it only
touches training rows and never the held-out test lines, so reported metrics are
unaffected. Left as-is.

**Fix:** none required. Added a pipeline-level regression test that fails if a
cell line ever appears in both splits or if an IC50 column reaches X (see
`tests/test_split_integrity.py`).

## 2. Test coverage — 🟡 PARTIAL (severity: medium; was "zero")

**Not zero.** The suite (`28 passed`) already covers parsing/coercion
(`tests/test_parsing.py`), the leakage-scaler contract
(`tests/test_scaler_leakage.py`), and an end-to-end train→predict smoke test
(`tests/test_train_smoke.py`).

**Genuine gaps, now filled:**
- **API endpoint / schema validation** — there was no test exercising the actual
  FastAPI routes. Added `tests/test_api.py` (FastAPI `TestClient`): health,
  schema shape, `/api/predict`, `/api/predict_form`, `/api/predict_batch`,
  and error paths (empty upload, oversized, malformed).
- **Split-integrity leakage guard** — added `tests/test_split_integrity.py`
  (item 1).
- **Golden / determinism test** — added a seeded determinism test
  (`tests/test_determinism.py`): a fixed tiny model + fixed input yields a stable
  ranking and stable scores within tolerance, so a silent change to the scoring
  path is caught. (A byte-exact golden file on the *shipped* 11 MB model is
  intentionally avoided — it would couple the test to library-version pickle
  details; a seeded retrain is the robust equivalent.)

## 3. Large model artifact committed to git — 🔴 OPEN (severity: high)

**Confirmed.** `artifacts/model.joblib` is tracked
(`git ls-files` matches it) and `.gitignore:11` documents the deliberate choice
to commit it. Current blob ≈ **10.8 MB**; **git history holds several versions**
(blobs of ~29, ~21, ~20, ~15, ~11, ~7.8 MB) — every retrain added a fresh
multi-MB blob, so the packed history carries ~100 MB of dead binaries that every
clone pays for.

**Fix (this PR):**
- Stop tracking the binary: `.gitignore` now excludes `artifacts/*.joblib`.
- It becomes a build/runtime artifact instead:
  - `load_bundle()` already trains on demand if the file is absent
    (`model/predict.py:52-61`), so local `uvicorn` still works out of the box.
  - The `Dockerfile` builds the model at image-build time
    (`RUN python -m model.train`) so the container still starts with no boot-time
    training.
  - `make model` / `scripts/get_model.py` gives a one-command local build, with a
    documented hook for pulling a released asset instead.
- `.gitattributes` ships a ready **Git LFS** rule (commented) for teams who
  prefer to *version* the binary rather than rebuild it — pick one story, not
  both.
- **History purge is documented, not executed.** Rewriting published history is
  destructive (force-push, breaks every existing clone/PR), so it is left as an
  explicit, opt-in step for the maintainer (`git filter-repo` command in the
  README / this file), not something this PR force-pushes.

Purge command (run intentionally, then force-push):

```bash
# removes artifacts/model.joblib from ALL history; coordinate with collaborators
pip install git-filter-repo
git filter-repo --path artifacts/model.joblib --invert-paths
git push --force-with-lease origin <branch>
```

## 4. Bloated API payloads — 🟡 MOSTLY FINE (severity: low)

Already careful:
- `/api/health` returns **summary metrics only** — `summary_metrics`
  (`model/train.py:252-254`) strips the two 369-entry per-drug dicts; the comment
  at `app.py:136-137` shows this was a deliberate fix.
- The large per-drug metrics live behind their own explicit route `/api/metrics`
  (`app.py:146-152`).
- `predict` returns `top_k=8` ranked items, not all 369 (`model/predict.py:257`).
- Upload size is capped at 2 MB and cohorts at 500 rows
  (`app.py:39-40,186-190`).

**Minor, optional:** every `/api/predict*` response also embeds
`model_metrics` (summary-sized, ~a dozen numbers). The frontend does not read it
(`static/app.js` pulls stats from `/api/health`), so it is redundant per-call
weight. Left in place to avoid churn/contract changes; flagged here as the only
remaining trim. No large payloads exist.

## 5. Missing license — ✅ RESOLVED (severity: none)

- Root `LICENSE` is **MIT** (© 2026 Ekam Bhela).
- GDSC data licensing is documented: `data/gdsc/LICENSE.md` plus the README "The
  data" section (GDSC1 via Sanger `gdsctools`, BSD-3; GDSC2 via the public
  DeepTTC redistribution) with citations (Iorio et al., *Cell* 2016; Yang et
  al., *NAR* 2013).

**Fix:** none required. (README already states terms; no change needed.)

---

## Modeling assessment

### Feature representation (currently ~15 curated binary features)

The request to add expression / CNV / richer drug descriptors is reasonable, but
the repo already contains **evidence that the obvious version of this hurts**:

- **CNV / full genomic matrix already tested and rejected.**
  `experiments/RESULTS.md` (tier T1) shows the full 525–677-feature genomic
  matrix (which *includes* copy-number features) **lowers** held-out Spearman
  (0.316 → 0.287 VAL) — it overfits the data-starved per-drug models (~600 lines
  each). The curated 15-feature set is a deliberately stronger inductive bias.
  *More features only pay off with more samples/regularization.*
- **Expression data is not in the repo.** `data/gdsc/` ships genomic features
  (mutations, CNV factors, tissue, MSI) and IC50 matrices only — no expression
  matrix. Adding expression means an external download (GDSC/CCLE), which this
  sandbox's network policy does not guarantee, and a retrain. Documented as a
  concrete follow-up, not silently skipped.
- **Drug chemical descriptors (RDKit / Morgan):** the current drug
  representation is target-pathway + target multi-hot + capped identity
  (`model/mtl.py:44-68`). Morgan fingerprints are a sensible *addition*. This PR
  adds the scaffolding — `model/drug_features.py`, a pure function that turns a
  SMILES string into a Morgan-fingerprint vector (RDKit), with a unit test — but
  **does not claim a benchmark gain**: GDSC compound→SMILES mapping is not in the
  repo (needs a PubChem/ChEMBL lookup + network), so an honest leakage-free
  benchmark of fingerprints cannot be produced in this environment. Wiring +
  evaluation is scoped as the next step. Per the brief, no accuracy improvement
  is reported for a change that has not been measured.

### Architecture: per-drug → multi-task — ✅ ALREADY DONE & BENCHMARKED

The requested move from independent per-drug models to a shared-encoder /
drug-representation multi-task model **already exists and is benchmarked on a
leakage-free split.** `model/mtl.py` is a single `HistGradientBoostingRegressor`
over ~200k (cell line, drug) pairs with a shared genomic→response mapping and a
drug representation (pathway + target multi-hot + capped drug-identity category).
The shipped model **ensembles** it with the per-drug baseline
(`0.35·per-drug + 0.65·multi-task`, `model/train.py:36`). The full baseline →
multi-task → ensemble study is in `experiments/RESULTS.md` with per-drug
Spearman/R² (not global R²) on an 80/20 by-cell-line split.

### Reproducibility — 🟡 MOSTLY DONE

- **Pinned deps:** `requirements.txt` / `requirements-dev.txt` are fully pinned,
  with a comment explaining the pin also protects pickle compatibility.
- **Seeded:** `run_training(seed=0)`; splits use `np.random.default_rng(seed)`.
- **Single command:** `python -m model.train` already goes committed raw data →
  trained model → `metrics.json`.
- **Added:** a `Makefile` (`make setup|model|test|benchmark|run`) to make the
  one-command paths discoverable, and `scripts/get_model.py`.

---

## Benchmark (leakage-free, by-cell-line split)

The metric that matters for drug response is **per-drug rank correlation**
(Spearman) and top-k best-drug recovery — **not** global R². From
`experiments/RESULTS.md` (production 80/20 by cell line, refit on all lines):

| Model | Mean per-drug Spearman | Mean per-drug R² | top-1 | top-3 | top-10 | best-drug pct |
|---|---|---|---|---|---|---|
| Per-drug baseline (old) | 0.336 | 0.157 | 0.101 | 0.162 | **0.247** | 0.730 |
| **Ensemble (shipped)** | **0.382** | **0.179** | **0.111** | **0.177** | 0.232 | **0.750** |

Rank correlation improves ~14% (0.34 → 0.38) on a clean split. **Honesty check
(tissue-blocked):** Spearman collapses to ~0.07 and R² goes slightly negative —
the model does not extrapolate to unseen tissues. This audit reproduced a full
`python -m model.train`; see the run log / regenerated `artifacts/metrics.json`
for the confirmed numbers.

**Bottom line on the brief's warning:** leakage is ruled out (verified), so the
gains are reported as real — but they are *modest*, and the tissue-blocked
collapse is the honest ceiling. No number here depends on a change that wasn't
measured on a held-out split.
