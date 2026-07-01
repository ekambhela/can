"""
Inference on the real-GDSC model: parse a tumor/cell-line profile, rank drugs
by predicted sensitivity, and explain the call.

A "sample" is one profile described by:
  * tissue        (a GDSC tissue type)
  * MSI           (0/1)
  * ERBB2_amp     (0/1)
  * <GENE>_mut    (0/1) for the curated driver genes

Predictions are the model's per-drug sensitivity (z of -logIC50), shown as a
sensitivity percentile (norm CDF): "more sensitive than X% of cell lines".
"""

from __future__ import annotations

import io
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import norm

from .gdsc import (
    BINARY_FEATURES,
    ERBB2_AMP,
    FEATURE_LABEL,
    MSI,
    MUTATION_FEATURES,
    TISSUE_LABELS,
    feature_schema as _schema,
)

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS, "model.joblib")

# Short, data-flavored notes for the biomarkers we surface.
BIOMARKER_NOTE = {
    "BRAF_mut": "BRAF mutation is associated in GDSC with strong sensitivity to BRAF inhibitors.",
    "NRAS_mut": "NRAS mutation (RAS/RAF pathway) tracks with BRAF-inhibitor sensitivity in GDSC.",
    "EGFR_mut": "EGFR alteration is linked to EGFR-inhibitor response.",
    "ERBB2_amp": "ERBB2/HER2 amplification marks HER2-driven biology.",
    "TP53_mut": "TP53 status shifts cytotoxic sensitivity.",
    "MSI": "MSI-high reflects mismatch-repair deficiency.",
}


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    if not os.path.exists(MODEL_PATH):
        from .train import main as train_main
        train_main()
    import joblib
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Parsing / normalization
# ---------------------------------------------------------------------------
def _known_tissues() -> list[str]:
    try:
        return load_bundle()["tissues"]
    except Exception:  # noqa: BLE001
        return ["lung_NSCLC"]


def _coerce_binary(v) -> float | None:
    try:
        return 1.0 if float(v) >= 0.5 else 0.0
    except (TypeError, ValueError):
        s = str(v).strip().lower()
        if s in {"yes", "true", "positive", "pos", "mut", "amplified", "y", "high", "1"}:
            return 1.0
        if s in {"no", "false", "negative", "neg", "wt", "n", "0", ""}:
            return 0.0
        return None


def _coerce_tissue(v, tissues) -> str | None:
    s = str(v).strip()
    lut = {t.lower(): t for t in tissues}
    if s.lower() in lut:
        return lut[s.lower()]
    # accept friendly labels
    for key, lab in TISSUE_LABELS.items():
        if s.lower() == lab.lower() and key in tissues:
            return key
    # loose match on normalized text
    norm_s = s.lower().replace(" ", "_").replace("-", "_")
    return lut.get(norm_s)


DEFAULT_TISSUE = "lung_NSCLC"


def _normalize(flat: dict, prefix: str = "") -> tuple[dict, list[str]]:
    lookup = {str(k).strip().lower(): k for k in flat}
    tissues = _known_tissues()
    sample: dict = {}
    warnings: list[str] = []

    # tissue
    src = lookup.get("tissue") or lookup.get("tissue_factor") or lookup.get("cancer_type")
    if src is None:
        sample["tissue"] = DEFAULT_TISSUE
        warnings.append(f"{prefix}tissue missing — defaulted to '{DEFAULT_TISSUE}'.")
    else:
        t = _coerce_tissue(flat[src], tissues)
        if t is None:
            sample["tissue"] = DEFAULT_TISSUE
            warnings.append(f"{prefix}unrecognized tissue '{flat[src]}' — defaulted to '{DEFAULT_TISSUE}'.")
        else:
            sample["tissue"] = t

    # binary features
    for f in BINARY_FEATURES:
        s = lookup.get(f.lower())
        if s is None:
            sample[f] = 0.0
            continue
        c = _coerce_binary(flat[s])
        if c is None:
            sample[f] = 0.0
            warnings.append(f"{prefix}could not parse '{f}'='{flat[s]}' — set to 0.")
        else:
            sample[f] = c
    return sample, warnings


def _raw_to_records(raw: bytes, filename: str) -> list[dict]:
    text = raw.decode("utf-8-sig", errors="replace").strip()
    name = (filename or "").lower()
    if name.endswith(".json") or text[:1] in "{[":
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    sep = "\t" if (name.endswith(".tsv") or "\t" in text.splitlines()[0]) else ","
    df = pd.read_csv(io.StringIO(text), sep=sep)
    cols = [c.strip().lower() for c in df.columns]
    if df.shape[1] == 2 and cols[0] in {"feature", "key", "name", "marker"}:
        return [{str(k): v for k, v in zip(df.iloc[:, 0], df.iloc[:, 1])}]
    return [{str(c): row[c] for c in df.columns} for _, row in df.iterrows()]


def parse_sample(raw: bytes, filename: str = "") -> tuple[dict, list[str]]:
    recs = _raw_to_records(raw, filename)
    if not recs:
        raise ValueError("No sample found in file.")
    return _normalize(recs[0])


def parse_cohort(raw: bytes, filename: str = "") -> tuple[list[dict], list[str]]:
    recs = _raw_to_records(raw, filename)
    if not recs:
        raise ValueError("No samples found in file.")
    samples, warnings = [], []
    for i, r in enumerate(recs, start=1):
        prefix = f"Row {i}: " if len(recs) > 1 else ""
        s, w = _normalize(r, prefix=prefix)
        samples.append(s)
        warnings.extend(w)
    return samples, warnings


def sample_from_dict(d: dict) -> tuple[dict, list[str]]:
    return _normalize({str(k): v for k, v in d.items()})


def feature_schema() -> dict:
    return _schema(_known_tissues())


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def _row(sample: dict) -> pd.DataFrame:
    row = {f: float(sample.get(f, 0.0)) for f in BINARY_FEATURES}
    row["tissue"] = sample.get("tissue", DEFAULT_TISSUE)
    return pd.DataFrame([row])


def _pct(z: float) -> float:
    return float(np.clip(norm.cdf(z) * 100.0, 0.0, 100.0))


def _confidence(zs: np.ndarray) -> float:
    z = (zs - zs.mean()) / (zs.std() + 1e-9)
    ex = np.exp(z - z.max())
    p = ex / ex.sum()
    return float(p.max())


def _explain(sample: dict, therapy: str, models: dict) -> dict:
    """Data-driven attribution: effect of each present feature on this drug's
    predicted sensitivity, by toggling it off and measuring the change."""
    model = models[therapy]
    base = float(model.predict(_row(sample))[0])
    supporting, cautions = [], []
    for f in MUTATION_FEATURES + [ERBB2_AMP, MSI]:
        if float(sample.get(f, 0)) < 0.5:
            continue
        off = dict(sample)
        off[f] = 0.0
        eff_pct = _pct(base) - _pct(float(model.predict(_row(off))[0]))
        if abs(eff_pct) < 1.5:
            continue
        item = {"feature": f, "label": FEATURE_LABEL.get(f, f),
                "effect_pct": round(eff_pct, 1),
                "text": BIOMARKER_NOTE.get(f, f"{FEATURE_LABEL.get(f, f)} shifts predicted response.")}
        (supporting if eff_pct > 0 else cautions).append(item)
    supporting.sort(key=lambda d: -d["effect_pct"])
    cautions.sort(key=lambda d: d["effect_pct"])
    if not supporting:
        supporting.append({"feature": None, "label": "Tissue & overall profile", "effect_pct": None,
                           "text": "Selected mainly from the tissue type and overall genomic profile."})
    return {"supporting": supporting, "cautions": cautions}


def predict(sample: dict, top_k: int | None = None) -> dict:
    bundle = load_bundle()
    models = bundle["models"]
    meta = bundle["drug_meta"]
    resid = bundle.get("resid_std", {})
    names = list(models.keys())

    X = _row(sample)
    zs = {n: float(models[n].predict(X)[0]) for n in names}
    order = sorted(names, key=lambda n: zs[n], reverse=True)
    if top_k:
        order = order[:top_k]
    zarr = np.array([zs[n] for n in names])
    confidence = _confidence(zarr)
    pcts = {n: _pct(zs[n]) for n in names}
    margin = round((pcts[order[0]] - pcts[order[1]]) / 100.0, 4) if len(order) > 1 else 0.0

    ranked = []
    for rank, n in enumerate(order, start=1):
        rs = resid.get(n, 0.6)
        exp = _explain(sample, n, models)
        ranked.append({
            "rank": rank, "therapy": n, "drug_class": meta.get(n, {}).get("target", ""),
            "sensitivity": round(zs[n], 4),
            "match_percent": round(pcts[n], 1),
            "ci_low": round(_pct(zs[n] - rs), 1),
            "ci_high": round(_pct(zs[n] + rs), 1),
            "rationale": [s["text"] for s in exp["supporting"][:2]],
            "supporting": exp["supporting"], "cautions": exp["cautions"],
        })
    return {
        "recommendation": ranked[0]["therapy"],
        "confidence": round(confidence, 4),
        "decision_margin": margin,
        "ranked": ranked,
        "model_metrics": bundle.get("metrics", {}),
    }


def predict_batch(samples: list[dict]) -> dict:
    bundle = load_bundle()
    models = bundle["models"]
    meta = bundle["drug_meta"]
    names = list(models.keys())
    X = pd.concat([_row(s) for s in samples], ignore_index=True)
    Z = {n: models[n].predict(X) for n in names}

    rows = []
    for i, s in enumerate(samples):
        zs = {n: float(Z[n][i]) for n in names}
        order = sorted(names, key=lambda n: zs[n], reverse=True)
        top, second = order[0], order[1]
        pcts = {n: _pct(zs[n]) for n in names}
        rows.append({
            "index": i + 1,
            "cancer_type": TISSUE_LABELS.get(s.get("tissue", ""), s.get("tissue", "")),
            "recommendation": top, "drug_class": meta.get(top, {}).get("target", ""),
            "match_percent": round(pcts[top], 1),
            "confidence": round(_confidence(np.array(list(zs.values()))), 4),
            "decision_margin": round((pcts[top] - pcts[second]) / 100.0, 4),
            "runner_up": second, "runner_up_percent": round(pcts[second], 1),
        })
    return {"n": len(rows), "therapies": names, "rows": rows,
            "model_metrics": bundle.get("metrics", {})}
