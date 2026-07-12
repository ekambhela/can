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

Framed as **drug-response prediction**, predicting sensitivity `-z(logIC50)`
(higher = more sensitive). The shipped model is an **ensemble of two
gradient-boosted models** (`HistGradientBoostingRegressor`):

- a **per-drug** model — one regressor per drug on its own screened lines, which
  calibrates each drug well; and
- a **multi-task** model — a single regressor over every observed
  `(cell line, drug)` pair, with each drug represented by its target pathway,
  target multi-hot, and a capped drug-identity category. It learns one shared
  genomic→response mapping across all ~200k pairs, so a drug tested on few lines
  borrows the pattern from similar drugs.

Their predictions are blended (`0.35·per-drug + 0.65·multi-task`, weight chosen
on a validation split). The blend beat either model alone on a held-out-by-cell-
line test set — the two have complementary strengths (per-drug calibration vs
cross-drug ranking). See [`experiments/RESULTS.md`](experiments/RESULTS.md) for
the full baseline→ensemble study.

- **Features (curated):** tissue (one-hot), MSI, ERBB2/HER2 amplification, and
  13 driver-gene mutation flags (TP53, KRAS, EGFR, BRAF, ALK, ERBB2, BRCA1/2,
  PIK3CA, PTEN, NRAS, APC, CDKN2A). A curated set beats the full 677-feature
  genomic matrix here (which overfits the per-drug models) — see the experiments.
- **No leakage:** the split is **by cell line** (a line is never in both train
  and test) and the per-drug `-z(logIC50)` scaling is fit on **train lines only**.
- **Ranking:** drugs are ordered by predicted sensitivity, shown as a percentile,
  with an uncertainty band from held-out residuals and a decision margin.
- **Explanations:** supporting/caution factors are computed by toggling each
  present feature and measuring the change in predicted sensitivity.

### Performance (held out on 20% of cell lines, split by cell line)

| Metric | Per-drug baseline | **Ensemble (shipped)** |
| --- | --- | --- |
| Top-10 accuracy (true best drug in top 10 of 369) | ~25% | **~23%** |
| Mean percentile rank of the true best drug | 0.73 | **0.75** |
| Mean per-drug Spearman (predicted vs real IC50) | 0.34 | **0.38** |
| Mean per-drug R² | 0.16 | **0.18** |

The ensemble improves rank correlation (~+0.05), R², percentile, and top-1/3;
top-10 is within noise of the baseline. **Honesty check — tissue-blocked CV:**
when whole tissue types are held out, mean Spearman collapses to **~0.07** (R²
goes slightly negative). The model relies heavily on tissue context and does
**not** extrapolate to unseen tissues — a real limitation we state plainly.

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
joblib (`compress=3`, ~11 MB — it bundles both ensemble components) so clones
stay reasonable while the container still starts instantly. If you'd rather keep the binary out of git history entirely,
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
  perdrug.py           per-drug HistGBM component of the ensemble
  mtl.py               multi-task (cell line, drug) component of the ensemble
  train.py             trains + blends both components, evaluates, saves bundle
  predict.py           parse → blend → rank → data-driven explanation
artifacts/model.joblib trained ensemble (committed, compressed; no boot training)
templates/index.html   single-page UI (Home / Problem / Science / How / Matcher / About)
static/                style.css, app.js, sample files
tests/                 pytest: parsing, coercion, leakage regression, train smoke
experiments/           accuracy study harness + tier runners + RESULTS.md
```
