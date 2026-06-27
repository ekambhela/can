# OncoMatch — Tumor → Chemotherapy Matcher

A web application that matches a tumor sample to its **most effective therapy**
with as much precision as the model can muster. Upload a tumor profile and the
model ranks a panel of 15 chemotherapies and targeted agents by predicted drug
sensitivity, and explains the biomarkers behind each match.

> ⚠️ **Research / educational use only.** OncoMatch is a demonstration of
> pharmacogenomic drug-response modeling. It is **not** a validated clinical
> decision-support tool and must not be used to guide patient care.

---

## What it does

1. You upload a **single tumor sample** (CSV / TSV / JSON) describing its
   genomic alterations and expression/clinical features.
2. A trained **multi-output gradient-boosted model** predicts a sensitivity
   score for every therapy in the panel.
3. The UI shows the **top match + confidence**, the **full ranking**, and the
   **biomarkers** that drove the recommendation.

### Model performance (held-out test set, 1,800 samples)

| Metric | Value |
| --- | --- |
| Top-1 accuracy (recovers the single best therapy) | **~82%** |
| Top-3 accuracy (best therapy in top 3) | **~98%** |
| Mean per-drug R² | **~0.82** |
| Mean per-sample Spearman rank correlation | **~0.89** |

These numbers print when you train, and are surfaced live in the app header and
at `GET /api/health`.

---

## How the model works

The matcher is framed as a **drug-response prediction** problem — the standard
approach in pharmacogenomics. Rather than picking a single label, it predicts a
continuous sensitivity score for *every* therapy and then ranks them, so you
always get a full ordered recommendation list with margins.

- **Algorithm:** `MultiOutputRegressor` of `HistGradientBoostingRegressor`s over
  a preprocessing pipeline (one-hot cancer type + passthrough genomics/expression).
  Gradient-boosted trees capture the non-linear biomarker interactions and
  resistance gradients well.
- **Training data:** a biologically grounded synthetic cohort
  (`model/generate_data.py`). Tumor samples are drawn with literature-inspired
  per-cancer-type mutation prevalences; drug-sensitivity labels are produced
  from well-established biomarker–drug rules in `model/biomarkers.py`
  (e.g. *BRCA loss → PARP/platinum sensitivity*, *HER2 amplification →
  trastuzumab*, *EGFR-activating mutation → EGFR-TKI*, *MSI-high/high TMB →
  checkpoint inhibitor*, *ERCC1-high → platinum resistance*), plus realistic
  assay noise.
- **Explanations:** for the recommended therapy the app reports which of the
  sample's biomarkers moved its score, by perturbing the rule model against a
  wild-type baseline.

> Because the labels come from a rules-based simulator, the model is a faithful
> *demonstration* of the matching workflow, not a clinically validated predictor.
> To make it clinical-grade, swap `generate_data.py` for a real labeled cohort
> (e.g. GDSC / DepMap / NCI-60 drug-response data) and re-run training — the rest
> of the pipeline and UI stay the same.

---

## Tumor sample format

One sample per file. Provide whatever you have — missing features are imputed to
population defaults and flagged as warnings.

**Wide CSV** (header row + one data row):

```csv
cancer_type,EGFR_mut,KRAS_mut,TP53_mut,TMB,proliferation
lung_nsclc,1,0,1,5.0,0.45
```

**Key/value CSV:**

```csv
feature,value
cancer_type,ovarian
BRCA1_mut,yes
```

**JSON:**

```json
{ "cancer_type": "breast", "HER2_amp": 1, "ER_expr": 0.2 }
```

### Recognized fields

- `cancer_type`: one of `breast, lung_nsclc, ovarian, colorectal, melanoma,
  pancreatic, gastric, glioma`
- Genomic flags (0/1 or yes/no): `TP53_mut, KRAS_mut, EGFR_mut, BRAF_V600,
  ALK_fusion, HER2_amp, BRCA1_mut, BRCA2_mut, PIK3CA_mut, PTEN_loss, MSI_high`
- Continuous (0–1 unless noted): `TMB` (mut/Mb, 0–40), `proliferation`,
  `ER_expr`, `ABCB1_expr`, `ERCC1_expr`, `TUBB3_expr`

Ready-made examples live in [`static/samples/`](static/samples/) and are
downloadable from the app.

---

## Run it

```bash
pip install -r requirements.txt

# start the web app — the model trains itself (~7 s) on first request
uvicorn app:app --reload
# open http://localhost:8000

# (optional) train ahead of time / re-evaluate
python -m model.train
```

### API

- `GET  /` — the web UI
- `GET  /api/health` — model status + metrics
- `POST /api/predict` — multipart upload (`file=<sample>`) → ranked JSON

```bash
curl -F "file=@static/samples/sample_egfr_lung.csv" http://localhost:8000/api/predict
```

---

## Project layout

```
app.py                 FastAPI server (UI + /api/predict)
model/
  biomarkers.py        feature schema, therapy panel, pharmacogenomic rules
  generate_data.py     biologically grounded synthetic training cohort
  train.py             trains + evaluates the multi-output model
  predict.py           parse upload → rank therapies → explain
artifacts/
  model.joblib         trained model bundle (auto-generated on first run)
  metrics.json         held-out evaluation metrics
templates/index.html   single-page UI
static/                style.css, app.js, sample files
```
