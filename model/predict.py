"""
Inference layer: parse an uploaded tumor sample, rank therapies, explain why.

Public surface
--------------
  load_bundle()                     -> cached model bundle
  parse_sample(raw_bytes, filename) -> (sample_dict, warnings)
  predict(sample_dict)              -> ranked recommendation payload

A "sample" is one tumor. We accept three on-disk encodings, all describing a
single sample:
  * CSV  with a header row and one data row (column = feature)
  * CSV  in two-column key,value form
  * JSON object {feature: value, ...}

Missing features are imputed to population defaults and reported as warnings so
the clinician/researcher knows the prediction leaned on priors.
"""

from __future__ import annotations

import io
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from .biomarkers import (
    CANCER_TYPES,
    CONTINUOUS_FEATURES,
    CONTINUOUS_PRIORS,
    MUTATION_FEATURES,
    THERAPIES,
    sensitivity_scores,
)

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS, "model.joblib")

# Human-readable explanations for the biomarkers that most drive a match.
BIOMARKER_RATIONALE = {
    "EGFR_mut": "EGFR-activating mutation predicts response to EGFR tyrosine-kinase inhibitors.",
    "HER2_amp": "HER2 (ERBB2) amplification predicts benefit from anti-HER2 therapy.",
    "BRAF_V600": "BRAF V600 mutation predicts response to BRAF-targeted inhibitors.",
    "ALK_fusion": "ALK fusion predicts response to ALK inhibitors.",
    "BRCA1_mut": "BRCA1 loss confers synthetic lethality with PARP inhibition and platinum sensitivity.",
    "BRCA2_mut": "BRCA2 loss confers synthetic lethality with PARP inhibition and platinum sensitivity.",
    "MSI_high": "MSI-high / dMMR status predicts benefit from checkpoint inhibition.",
    "KRAS_mut": "KRAS mutation predicts resistance to EGFR-targeted therapy.",
    "TP53_mut": "TP53 mutation modestly shifts cytotoxic sensitivity.",
    "PTEN_loss": "PTEN loss alters PI3K-pathway-dependent drug sensitivity.",
    "PIK3CA_mut": "PIK3CA mutation activates PI3K signalling.",
}


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    """Load the trained model bundle, training it once if not yet present.

    The model is small (~7 s to fit) and fully reproducible from the committed
    code, so on a fresh checkout we transparently train it on first use rather
    than requiring a binary artifact to be committed.
    """
    if not os.path.exists(MODEL_PATH):
        from .train import main as train_main
        train_main()
    import joblib
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _coerce(name: str, value) -> float | str | None:
    """Coerce a raw cell to the right type for feature `name`."""
    if name == "cancer_type":
        v = str(value).strip().lower().replace(" ", "_").replace("-", "_")
        return v if v in CANCER_TYPES else None
    try:
        f = float(value)
    except (TypeError, ValueError):
        # accept yes/no/true/false/positive/negative for genomic flags
        s = str(value).strip().lower()
        if s in {"yes", "true", "positive", "pos", "mut", "amplified", "y"}:
            return 1.0
        if s in {"no", "false", "negative", "neg", "wt", "wildtype", "n", ""}:
            return 0.0
        return None
    if name in MUTATION_FEATURES:
        return 1.0 if f >= 0.5 else 0.0
    return f


def _raw_to_dict(raw: bytes, filename: str) -> dict:
    """Turn an uploaded file's bytes into a flat {feature: raw_value} dict."""
    text = raw.decode("utf-8-sig", errors="replace").strip()
    name = (filename or "").lower()

    # JSON
    if name.endswith(".json") or text[:1] in "{[":
        obj = json.loads(text)
        if isinstance(obj, list):
            obj = obj[0]
        return {str(k): v for k, v in obj.items()}

    # CSV / TSV
    sep = "\t" if (name.endswith(".tsv") or "\t" in text.splitlines()[0]) else ","
    df = pd.read_csv(io.StringIO(text), sep=sep)

    # Two-column key,value layout?
    cols = [c.strip().lower() for c in df.columns]
    if df.shape[1] == 2 and cols[0] in {"feature", "key", "name", "marker"}:
        return {str(k): v for k, v in zip(df.iloc[:, 0], df.iloc[:, 1])}

    # Otherwise: header + (at least one) data row, wide format.
    if len(df) == 0:
        raise ValueError("File has a header but no data row.")
    return {str(c): df.iloc[0][c] for c in df.columns}


ALL_FEATURES = MUTATION_FEATURES + CONTINUOUS_FEATURES + ["cancer_type"]


def _normalize(flat: dict, prefix: str = "") -> tuple[dict, list[str]]:
    """Coerce + impute a flat {feature: raw} dict into a full sample.

    `prefix` (e.g. "Row 3: ") is prepended to warnings for batch context.
    """
    lookup = {str(k).strip().lower(): k for k in flat}
    sample: dict = {}
    warnings: list[str] = []

    for feat in ALL_FEATURES:
        src = lookup.get(feat.lower())
        if src is None:
            if feat == "cancer_type":
                sample[feat] = "breast"
                warnings.append(f"{prefix}cancer_type missing — defaulted to 'breast'.")
            elif feat in MUTATION_FEATURES:
                sample[feat] = 0.0
            else:
                sample[feat] = CONTINUOUS_PRIORS[feat][0]
            continue
        coerced = _coerce(feat, flat[src])
        if coerced is None:
            if feat == "cancer_type":
                sample[feat] = "breast"
                warnings.append(
                    f"{prefix}unrecognized cancer_type '{flat[src]}' — defaulted to 'breast'. "
                    f"Known: {', '.join(CANCER_TYPES)}."
                )
            else:
                sample[feat] = CONTINUOUS_PRIORS[feat][0] if feat in CONTINUOUS_FEATURES else 0.0
                warnings.append(f"{prefix}could not parse '{feat}'='{flat[src]}' — imputed default.")
        else:
            sample[feat] = coerced

    provided = sum(1 for f in ALL_FEATURES if f.lower() in lookup)
    if provided < 3:
        warnings.append(
            f"{prefix}very few recognized features were provided; prediction relies heavily on priors."
        )
    return sample, warnings


def parse_sample(raw: bytes, filename: str = "") -> tuple[dict, list[str]]:
    """Parse + validate + impute a single tumor sample.

    Returns (sample, warnings). `sample` always has every model feature.
    """
    return _normalize(_raw_to_dict(raw, filename))


def parse_cohort(raw: bytes, filename: str = "") -> tuple[list[dict], list[str]]:
    """Parse a multi-row file (one tumor per row) into a list of samples.

    Wide CSV/TSV with a header is treated row-per-sample. JSON may be a list of
    objects. A single-sample file is accepted as a cohort of one.
    """
    text = raw.decode("utf-8-sig", errors="replace").strip()
    name = (filename or "").lower()
    warnings: list[str] = []

    if name.endswith(".json") or text[:1] in "{[":
        obj = json.loads(text)
        records = obj if isinstance(obj, list) else [obj]
    else:
        sep = "\t" if (name.endswith(".tsv") or "\t" in text.splitlines()[0]) else ","
        df = pd.read_csv(io.StringIO(text), sep=sep)
        cols = [c.strip().lower() for c in df.columns]
        # key,value layout => a single sample
        if df.shape[1] == 2 and cols[0] in {"feature", "key", "name", "marker"}:
            records = [{str(k): v for k, v in zip(df.iloc[:, 0], df.iloc[:, 1])}]
        else:
            records = [{str(c): row[c] for c in df.columns} for _, row in df.iterrows()]

    if not records:
        raise ValueError("No tumor samples found in file.")

    samples = []
    for i, rec in enumerate(records, start=1):
        prefix = f"Row {i}: " if len(records) > 1 else ""
        s, w = _normalize(rec, prefix=prefix)
        samples.append(s)
        warnings.extend(w)
    return samples, warnings


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def _confidence(scores: np.ndarray) -> float:
    """Confidence from the separation between the top therapy and the field.

    Uses a softmax over predicted sensitivities; the top probability is a
    well-behaved 0–1 confidence that rises when one therapy clearly wins.
    """
    z = (scores - scores.mean()) / (scores.std() + 1e-9)
    ex = np.exp(z - z.max())
    p = ex / ex.sum()
    return float(p.max())


# Display names + caution texts for the contribution analysis.
FEATURE_LABEL = {
    "TP53_mut": "TP53 mutation", "KRAS_mut": "KRAS mutation",
    "EGFR_mut": "EGFR-activating mutation", "BRAF_V600": "BRAF V600 mutation",
    "ALK_fusion": "ALK fusion", "HER2_amp": "HER2 amplification",
    "BRCA1_mut": "BRCA1 loss", "BRCA2_mut": "BRCA2 loss",
    "PIK3CA_mut": "PIK3CA mutation", "PTEN_loss": "PTEN loss",
    "MSI_high": "MSI-high / dMMR", "TMB": "Tumor mutational burden",
    "proliferation": "High proliferation index", "ER_expr": "ER expression",
    "ABCB1_expr": "ABCB1 / MDR1 efflux", "ERCC1_expr": "ERCC1 (NER capacity)",
    "TUBB3_expr": "Class-III β-tubulin (TUBB3)",
}

CAUTION_RATIONALE = {
    "ERCC1_expr": "Elevated ERCC1 (nucleotide-excision repair) predicts reduced platinum sensitivity.",
    "TUBB3_expr": "Elevated class-III β-tubulin predicts taxane resistance.",
    "ABCB1_expr": "High ABCB1/MDR1 drug efflux predicts multidrug resistance.",
    "KRAS_mut": "KRAS mutation predicts resistance to EGFR-targeted therapy.",
    "PTEN_loss": "PTEN loss can blunt response to this agent.",
    "ER_expr": "ER-positive biology is typically less chemo-driven.",
}

# Effect (in sensitivity units) below which a feature is treated as irrelevant.
_EPS = 0.03


def _wild_type(sample: dict) -> dict:
    """A same-cancer-type baseline tumor with no alterations, mean expression."""
    wt = {k: 0.0 for k in MUTATION_FEATURES}
    wt.update({k: CONTINUOUS_PRIORS[k][0] for k in CONTINUOUS_FEATURES})
    wt["cancer_type"] = sample["cancer_type"]
    return wt


def explain(sample: dict, therapy: str) -> dict:
    """Quantified, signed per-feature attribution for one therapy.

    The label-generating rule model is (near-)additive, so each feature's
    marginal effect vs a wild-type baseline is a faithful attribution. Returns
    supporting factors and cautions, each with an effect in percentage points.
    """
    wt = _wild_type(sample)
    base_score = sensitivity_scores(wt)[therapy]
    supporting, cautions = [], []

    for f in MUTATION_FEATURES + CONTINUOUS_FEATURES:
        val = sample.get(f, 0.0)
        # marginal effect of moving feature f from baseline to its sample value
        probe = dict(wt)
        probe[f] = val
        effect = sensitivity_scores(probe)[therapy] - base_score
        if abs(effect) < _EPS:
            continue
        item = {
            "feature": f,
            "label": FEATURE_LABEL.get(f, f),
            "effect_pct": round(effect * 100, 1),
        }
        if effect > 0:
            item["text"] = BIOMARKER_RATIONALE.get(
                f, f"{item['label']} favors this therapy.")
            supporting.append(item)
        else:
            item["text"] = CAUTION_RATIONALE.get(
                f, f"{item['label']} reduces predicted benefit.")
            cautions.append(item)

    supporting.sort(key=lambda d: -d["effect_pct"])
    cautions.sort(key=lambda d: d["effect_pct"])
    if not supporting:
        supporting.append({
            "feature": None, "label": "Overall profile",
            "effect_pct": None,
            "text": "Selected as the strongest option given the overall tumor profile.",
        })
    return {"supporting": supporting, "cautions": cautions}


def _decision_margin(scores: np.ndarray) -> float:
    """Gap between the best and second-best therapy (0–1 sensitivity units)."""
    s = np.sort(scores)[::-1]
    return float(s[0] - s[1]) if len(s) > 1 else float(s[0])


def predict(sample: dict, top_k: int = 5) -> dict:
    """Rank therapies for one parsed sample and return a UI-ready payload."""
    bundle = load_bundle()
    model = bundle["model"]
    therapies = bundle["therapies"]

    X = pd.DataFrame([sample])
    raw_scores = np.clip(np.asarray(model.predict(X)).ravel(), 0.0, 1.0)

    # Per-therapy uncertainty interval from the quantile models (v2 bundles).
    lo = hi = None
    if "model_lo" in bundle and "model_hi" in bundle:
        lo = np.clip(np.asarray(bundle["model_lo"].predict(X)).ravel(), 0.0, 1.0)
        hi = np.clip(np.asarray(bundle["model_hi"].predict(X)).ravel(), 0.0, 1.0)
        lo = np.minimum(lo, raw_scores)
        hi = np.maximum(hi, raw_scores)

    order = raw_scores.argsort()[::-1]
    confidence = _confidence(raw_scores)
    margin = _decision_margin(raw_scores)

    ranked = []
    for rank, j in enumerate(order[:top_k], start=1):
        name = therapies[j]
        exp = explain(sample, name)
        item = {
            "rank": rank,
            "therapy": name,
            "drug_class": THERAPIES.get(name, ""),
            "sensitivity": round(float(raw_scores[j]), 4),
            "match_percent": round(float(raw_scores[j]) * 100, 1),
            "rationale": [s["text"] for s in exp["supporting"][:2]],
            "supporting": exp["supporting"],
            "cautions": exp["cautions"],
        }
        if lo is not None:
            item["ci_low"] = round(float(lo[j]) * 100, 1)
            item["ci_high"] = round(float(hi[j]) * 100, 1)
        ranked.append(item)

    return {
        "recommendation": ranked[0]["therapy"],
        "confidence": round(confidence, 4),
        "decision_margin": round(margin, 4),
        "ranked": ranked,
        "all_scores": {therapies[j]: round(float(raw_scores[j]), 4)
                       for j in order},
        "model_metrics": bundle.get("metrics", {}),
    }


def predict_batch(samples: list[dict]) -> dict:
    """Rank therapies for a cohort of samples; returns a compact table payload.

    Each row carries the top recommendation, confidence, decision margin, and
    the runner-up so a clinician/researcher can triage a whole cohort at once.
    """
    bundle = load_bundle()
    model = bundle["model"]
    therapies = bundle["therapies"]

    X = pd.DataFrame(samples)
    scores = np.clip(np.asarray(model.predict(X)), 0.0, 1.0)

    rows = []
    for i, sc in enumerate(scores):
        order = sc.argsort()[::-1]
        top, second = therapies[order[0]], therapies[order[1]]
        rows.append({
            "index": i + 1,
            "cancer_type": samples[i].get("cancer_type", ""),
            "recommendation": top,
            "drug_class": THERAPIES.get(top, ""),
            "match_percent": round(float(sc[order[0]]) * 100, 1),
            "confidence": round(_confidence(sc), 4),
            "decision_margin": round(_decision_margin(sc), 4),
            "runner_up": second,
            "runner_up_percent": round(float(sc[order[1]]) * 100, 1),
        })
    return {"n": len(rows), "therapies": therapies, "rows": rows,
            "model_metrics": bundle.get("metrics", {})}


# ---------------------------------------------------------------------------
# Manual-entry support (schema + dict → sample)
# ---------------------------------------------------------------------------
CANCER_LABEL = {
    "breast": "Breast", "lung_nsclc": "Lung (NSCLC)", "ovarian": "Ovarian",
    "colorectal": "Colorectal", "melanoma": "Melanoma", "pancreatic": "Pancreatic",
    "gastric": "Gastric", "glioma": "Glioma",
}


def feature_schema() -> dict:
    """Describe the input fields so the UI can build a manual-entry form."""
    cancers = [{"key": c, "label": CANCER_LABEL.get(c, c)} for c in CANCER_TYPES]
    muts = [{"key": k, "label": FEATURE_LABEL.get(k, k)} for k in MUTATION_FEATURES]
    cont = []
    for k in CONTINUOUS_FEATURES:
        mean, _sd, lo, hi = CONTINUOUS_PRIORS[k]
        cont.append({
            "key": k, "label": FEATURE_LABEL.get(k, k),
            "min": lo, "max": hi,
            "step": 1 if (hi - lo) > 5 else 0.05,
            "default": round(mean, 2),
        })
    return {"cancer_types": cancers, "mutations": muts, "continuous": cont}


def sample_from_dict(d: dict) -> tuple[dict, list[str]]:
    """Coerce + impute a raw feature dict (e.g. from the manual form)."""
    return _normalize({str(k): v for k, v in d.items()})
