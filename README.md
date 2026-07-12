# Karkive — Tumor → Drug Matcher (real GDSC data)

A web application that matches a tumor's genomic profile to the drug it is
predicted to be most sensitive to, trained on **real GDSC (Genomics of Drug
Sensitivity in Cancer) cell-line data**. Enter a profile (tissue, driver
mutations, HER2 amplification, MSI) and the model ranks a panel of drugs by
predicted sensitivity and shows the biomarkers behind each match.

The single-page site has tabs for a content-rich **Home** (mission, vision,
animated stats), **The problem** (why one-size-fits-all chemo falls short),
**The science** (therapy panel, data-derived biomarker associations, honest
performance), **How it works** (methodology), the **Matcher** tool, and an
**About** page — in a light theme with custom graphics.

> ⚠️ **Research / educational use only.** GDSC measures drug response in cultured
> cancer **cell lines**, not patients. These signals are useful for research but
> do **not** translate directly to patient outcomes. Not a clinical tool.

---

## The data

- **Source:** two independent GDSC screens — **GDSC1** (release 17, redistributed
  in the official Sanger `gdsctools` package, BSD-3) and **GDSC2** (25 Feb 2020
  fitted-dose-response, redistributed via the public DeepTTC repo). Files live in
  [`data/gdsc/`](data/gdsc/).
- **IC50 matrix:** 988 human cancer cell lines × 265 screened GDSC1 drug columns
  (264 named), plus the GDSC2 assay for the 806 of those lines it also tested
  (natural-log IC50). Both screens key on the same `COSMIC_ID`, so GDSC2 joins
  onto the same features.
- **Genomic features:** per cell line — tissue of origin, MSI status, driver-gene
  mutation flags, and copy-number alterations (incl. ERBB2/HER2 amplification).
- **Drug panel:** **369** distinct compounds — **248** from GDSC1 (264 named,
  16 replicate screens merged into one denoised target each) plus **121** new
  compounds from GDSC2 (namespaced ids so they never collide), spanning **24**
  target pathways (BRAF/MEK inhibitors, EGFR/HER2 inhibitors incl. Osimertinib,
  PI3K/AKT incl. Alpelisib, PARP incl. Olaparib/Niraparib, DNA-damaging
  cytotoxics, cell-cycle/mitosis, and more). GDSC2 compounds already present in
  GDSC1 are skipped so no drug appears twice.
- **Cite:** Iorio et al., *Cell* 2016; Yang et al., *Nucleic Acids Research* 2013.

## How the model works

- Framed as **drug-response prediction**: for each drug, a
  `HistGradientBoostingRegressor` is trained on the real cell lines screened
  against it, predicting sensitivity `-z(logIC50)` (higher = more sensitive).
  Drugs are modeled independently because the screen is sparse (not every drug
  was tested on every line).
- **Features (curated):** tissue (one-hot), MSI, ERBB2/HER2 amplification, and
  13 driver-gene mutation flags (TP53, KRAS, EGFR, BRAF, ALK, ERBB2, BRCA1/2,
  PIK3CA, PTEN, NRAS, APC, CDKN2A). A curated set beats the full 680-feature
  matrix here (which overfits ~800 training lines).
- **Ranking:** drugs are ordered by predicted sensitivity, shown as a percentile
  ("more sensitive than X% of cell lines"), with an uncertainty band from
  held-out residuals and a decision margin vs the runner-up.
- **Explanations:** the recommended drug's supporting/caution factors are
  computed by toggling each present feature and measuring the change in predicted
  sensitivity — a data-driven attribution, not hand-written rules.

### Performance (held-out 20% of cell lines)

| Metric | Value |
| --- | --- |
| Top-10 accuracy (true best drug in top 10 of 369) | **~25%** |
| Mean percentile rank of the true best drug | **~0.73** |
| Mean per-drug Spearman (predicted vs real IC50) | **~0.34** |
| Mean per-drug R² | **~0.16** |

The per-drug **target scaling (`-z(logIC50)`) is fit on the training split
only** — computing the mean/std over all cell lines would leak held-out
statistics into the target. In practice, with ~790 training lines the train-only
statistics are almost identical to the all-lines statistics, so this only nudged
R² (0.158 → 0.157); the rank metrics are affine-invariant and unaffected. The
point is a clean, defensible evaluation, not a bigger number.

These are honest, modest numbers — predicting drug response from a small
biomarker panel is genuinely hard. What matters is that the model **recovers real
biology**: BRAF mutation → strong sensitivity to BRAF inhibitors (Dabrafenib,
PLX-4720, SB590885), measured directly in the data at **p < 10⁻⁶**; NRAS mutation
likewise via the RAS/RAF pathway; ERBB2 amplification → HER2 inhibitors (Afatinib,
CP724714); EGFR mutation → EGFR inhibitors (Gefitinib, Afatinib).

## Input format

One profile per file (or per row for a cohort). Recognized fields:

- `tissue`: a GDSC tissue (e.g. `skin`, `lung_NSCLC`, `breast`, `large_intestine`,
  `ovary`, `pancreas`, …). Friendly labels are also accepted.
- `MSI`, `ERBB2_amp`: 0/1 (or yes/no).
- `<GENE>_mut`: 0/1 for TP53, KRAS, EGFR, BRAF, ALK, ERBB2, BRCA1, BRCA2, PIK3CA,
  PTEN, NRAS, APC, CDKN2A.

```csv
tissue,BRAF_mut,TP53_mut,MSI
skin,1,1,0
```

Or JSON: `{ "tissue": "breast", "ERBB2_amp": 1, "TP53_mut": 1 }`. Examples live in
[`static/samples/`](static/samples/) and download from the app.

## Run it

```bash
pip install -r requirements.txt

# start the web app (a pre-trained model is committed, so it loads instantly)
uvicorn app:app --reload
# open http://localhost:8000

# (optional) retrain on the GDSC data, then commit the refreshed artifacts/
python -m model.train
```

### API

- `GET  /` — the web UI
- `GET  /api/health` — model status + **summary** metrics (small; safe to poll)
- `GET  /api/metrics` — full metrics incl. per-drug R²/Spearman (large)
- `GET  /api/schema` — input-field schema (drives the manual-entry form)
- `POST /api/predict` — single sample file → ranked JSON with attribution
- `POST /api/predict_form` — JSON body `{tissue, <feature>: value, …}` → ranked JSON
- `POST /api/predict_batch` — cohort file (one tumor per row) → ranked table

## Deploy (public URL)

A portable `Dockerfile` installs dependencies and copies the committed model
(no training at build or boot), serving on `$PORT`. A `render.yaml` blueprint
provisions a free Render web service. See the Deploy section notes.

The trained model (`artifacts/model.joblib`) is committed, compressed with
joblib (`compress=3`, ~7.4 MB) so clones stay light while the container still
starts instantly. If you'd rather keep the binary out of git history entirely,
track it with **Git LFS** (`git lfs track "artifacts/*.joblib"`) or attach it as
a GitHub Release asset the Dockerfile pulls at build time. `model.joblib` is a
Python pickle — only load artifacts you trained yourself (see the security note
in `model/predict.py`).

## Project layout

```
app.py                 FastAPI server (UI + prediction endpoints)
data/gdsc/             real GDSC1 + GDSC2 matrices + provenance/licensing
model/
  gdsc.py              load GDSC data, curated feature + drug schema
  train.py             trains per-drug models on real data, evaluates
  predict.py           parse → rank → data-driven explanation (single + batch)
artifacts/model.joblib trained per-drug models (committed, compressed; no boot training)
templates/index.html   single-page UI (Home / Problem / Science / How / Matcher / About)
static/                style.css, app.js, sample files
tests/                 pytest: parsing, coercion, leakage regression, train smoke
```
