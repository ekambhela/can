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


def parse_sample(raw: bytes, filename: str = "") -> tuple[dict, list[str]]:
    """Parse + validate + impute a single tumor sample.

    Returns (sample, warnings). `sample` always has every model feature.
    """
    flat = _raw_to_dict(raw, filename)
    # normalize keys (strip, exact match against known features, case-insensitive)
    lookup = {k.strip().lower(): k for k in flat}
    sample: dict = {}
    warnings: list[str] = []

    all_features = MUTATION_FEATURES + CONTINUOUS_FEATURES + ["cancer_type"]
    for feat in all_features:
        src = lookup.get(feat.lower())
        if src is None:
            # impute
            if feat == "cancer_type":
                sample[feat] = "breast"  # neutral, common default
                warnings.append("cancer_type missing — defaulted to 'breast'.")
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
                    f"Unrecognized cancer_type '{flat[src]}' — defaulted to 'breast'. "
                    f"Known: {', '.join(CANCER_TYPES)}."
                )
            else:
                sample[feat] = CONTINUOUS_PRIORS[feat][0] if feat in CONTINUOUS_FEATURES else 0.0
                warnings.append(f"Could not parse '{feat}'='{flat[src]}' — imputed default.")
        else:
            sample[feat] = coerced

    provided = sum(1 for f in all_features if f.lower() in lookup)
    if provided < 3:
        warnings.append(
            "Very few recognized features were provided; prediction relies heavily on priors."
        )
    return sample, warnings


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


def _drivers(sample: dict, therapy: str) -> list[str]:
    """Explain which biomarkers pushed this therapy up for this sample."""
    # Compare this sample's latent score to a 'wild-type' baseline sample.
    wt = {k: 0.0 for k in MUTATION_FEATURES}
    wt.update({k: CONTINUOUS_PRIORS[k][0] for k in CONTINUOUS_FEATURES})
    wt["cancer_type"] = sample["cancer_type"]
    delta = sensitivity_scores(sample)[therapy] - sensitivity_scores(wt)[therapy]

    reasons = []
    for g in MUTATION_FEATURES:
        if sample.get(g, 0) >= 0.5 and g in BIOMARKER_RATIONALE:
            # only surface genomic drivers that plausibly affect this drug
            test = dict(wt)
            test[g] = 1.0
            if abs(sensitivity_scores(test)[therapy] - sensitivity_scores(wt)[therapy]) > 0.05:
                reasons.append(BIOMARKER_RATIONALE[g])
    if not reasons:
        if delta > 0.05:
            reasons.append("Expression / cancer-type profile favors this therapy.")
        else:
            reasons.append("Selected as the strongest option given the overall profile.")
    return reasons[:3]


def predict(sample: dict, top_k: int = 5) -> dict:
    """Rank therapies for one parsed sample and return a UI-ready payload."""
    bundle = load_bundle()
    model = bundle["model"]
    therapies = bundle["therapies"]

    X = pd.DataFrame([sample])
    raw_scores = np.asarray(model.predict(X)).ravel()
    raw_scores = np.clip(raw_scores, 0.0, 1.0)

    order = raw_scores.argsort()[::-1]
    confidence = _confidence(raw_scores)

    ranked = []
    for rank, j in enumerate(order[:top_k], start=1):
        name = therapies[j]
        ranked.append({
            "rank": rank,
            "therapy": name,
            "drug_class": THERAPIES.get(name, ""),
            "sensitivity": round(float(raw_scores[j]), 4),
            "match_percent": round(float(raw_scores[j]) * 100, 1),
            "rationale": _drivers(sample, name),
        })

    return {
        "recommendation": ranked[0]["therapy"],
        "confidence": round(confidence, 4),
        "ranked": ranked,
        "all_scores": {therapies[j]: round(float(raw_scores[j]), 4)
                       for j in order},
        "model_metrics": bundle.get("metrics", {}),
    }
