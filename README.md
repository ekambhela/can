# Karkive — Tumor → Drug Matcher (real GDSC data)

A web application that matches a tumor's genomic profile to the drug it is
predicted to be most sensitive to, trained on **real GDSC (Genomics of Drug
Sensitivity in Cancer) cell-line data**. Enter a profile (tissue, driver
mutations, HER2 amplification, MSI) and the model ranks a panel of drugs by
predicted sensitivity and shows the biomarkers behind each match.

The single-page site has four tabs — a content-rich **Home** (mission, vision,
animated stats), **How it works** (methodology), **The science** (therapy panel,
data-derived biomarker associations, honest performance), and the **Matcher**
tool — in a light theme with custom graphics.

> ⚠️ **Research / educational use only.** GDSC measures drug response in cultured
> cancer **cell lines**, not patients. These signals are useful for research but
> do **not** translate directly to patient outcomes. Not a clinical tool.

---

## The data

- **Source:** GDSC release 17, redistributed in the official Sanger `gdsctools`
  package (BSD-3). Files live in [`data/gdsc/`](data/gdsc/).
- **IC50 matrix:** 988 human cancer cell lines × 265 screened compounds
  (natural-log IC50).
- **Genomic features:** per cell line — tissue of origin, MSI status, driver-gene
  mutation flags, and copy-number alterations (incl. ERBB2/HER2 amplification).
- **Drug panel:** all **248** distinct GDSC compounds we can name authoritatively
  (264 named annotations, with the 16 compounds screened twice merged into one
  denoised target each), spanning
  **24** target pathways (BRAF/MEK inhibitors, EGFR/HER2 inhibitors, PI3K/MTOR,
  DNA-damaging cytotoxics, cell-cycle, and more). Names and target pathways come
  from the public GDSC screened-compounds annotation (redistributed via the
  DeepCDR repo), validated against the authoritative 9-drug GDSC database export
  bundled in `gdsctools` (all anchor IDs match). 264 of the 265 v17 drug columns
  are named; the one unnameable column is dropped rather than mislabeled.
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
| Top-10 accuracy (true best drug in top 10 of 248) | **~30%** |
| Mean percentile rank of the true best drug | **~0.72** |
| Mean per-drug Spearman (predicted vs real IC50) | **~0.31** |
| Mean per-drug R² | **~0.14** |

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
- `GET  /api/health` — model status + metrics
- `GET  /api/schema` — input-field schema (drives the manual-entry form)
- `POST /api/predict` — single sample file → ranked JSON with attribution
- `POST /api/predict_form` — JSON body `{tissue, <feature>: value, …}` → ranked JSON
- `POST /api/predict_batch` — cohort file (one tumor per row) → ranked table

## Deploy (public URL)

A portable `Dockerfile` installs dependencies and copies the committed model
(no training at build or boot), serving on `$PORT`. A `render.yaml` blueprint
provisions a free Render web service. See the Deploy section notes.

## Project layout

```
app.py                 FastAPI server (UI + prediction endpoints)
data/gdsc/             real GDSC release-17 matrices + provenance
model/
  gdsc.py              load GDSC data, curated feature + drug schema
  train.py             trains per-drug models on real data, evaluates
  predict.py           parse → rank → data-driven explanation (single + batch)
artifacts/model.joblib trained per-drug models (committed; no training at boot)
templates/index.html   single-page UI (Home / How / Science / Matcher)
static/                style.css, app.js, sample files
```
