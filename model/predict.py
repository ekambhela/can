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

from . import mtl, perdrug
from .schema import (
    BINARY_FEATURES,
    ERBB2_AMP,
    FEATURE_LABEL,
    MSI,
    MUTATION_FEATURES,
    TISSUE_LABELS,
    summary_metrics,
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
    """Load the trained bundle, or raise if it isn't there.

    Deliberately does NOT train on demand. Training reads the full GDSC
    matrices and fits ~370 boosters — minutes of CPU and hundreds of MB — so
    triggering it from a web request turns one cold start into a request that
    never returns, on a box that may not even have data/ (see model/schema.py).
    Building the artifact is an operator step; serving just loads it.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Build it with: python -m model.train"
        )
    import joblib
    # SECURITY: joblib.load unpickles arbitrary Python objects, so only ever load
    # model.joblib artifacts we produced ourselves (see model/train.py). Never
    # point MODEL_PATH at an untrusted file — a malicious pickle can run code.
    return joblib.load(MODEL_PATH)


def api_metrics() -> dict:
    """Summary metrics only (no 369-entry per-drug dicts) — for API payloads."""
    return summary_metrics(load_bundle().get("metrics", {}))


def full_metrics() -> dict:
    """Complete metrics incl. per-drug R^2 / Spearman — for the /api/metrics route."""
    return load_bundle().get("metrics", {})


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
def _score(bundle: dict, sample: dict, drug_ids=None) -> dict:
    """Blended sensitivity per drug name: w*per-drug + (1-w)*multi-task."""
    ids = drug_ids if drug_ids is not None else bundle["drug_ids"]
    w = bundle.get("blend_w_perdrug", 0.35)
    mt = mtl.score_sample(bundle, sample, drug_ids=ids)
    pd_scores = perdrug.score_sample(bundle["per_drug_models"], sample,
                                     bundle["cell_cols"], ids, bundle["id_to_name"])
    out = {}
    for n in set(mt) | set(pd_scores):
        a, b = pd_scores.get(n), mt.get(n)
        out[n] = a if b is None else (b if a is None else w * a + (1 - w) * b)
    return out


def _pct(z: float) -> float:
    return float(np.clip(norm.cdf(z) * 100.0, 0.0, 100.0))


def _confidence(zs: np.ndarray, resid_top: float = 0.6) -> float:
    """How clearly the top pick separates from its *closest rivals* — the next
    few best drugs — relative to the model's own prediction noise.

    Measuring against the whole drug panel is useless: the best drug is
    always many std above the panel mean, so that saturates at ~1.0 for every
    sample. What actually matters is whether #1 stands apart from the handful of
    near-ties just behind it. We take the gap between the top drug and the mean
    of ranks 2-6, in units of the top drug's held-out residual std, then map it
    through the normal CDF. A clear standout -> high; a cluster of near-ties ->
    ~0.5 (honestly uncertain which single drug is best)."""
    zs = np.asarray(zs, dtype=float)
    if zs.size < 2:
        return 0.5
    order = np.sort(zs)[::-1]
    top = float(order[0])
    rivals = float(order[1:6].mean()) if order.size >= 6 else float(order[1:].mean())
    z = (top - rivals) / (resid_top + 1e-9)
    return float(np.clip(norm.cdf(z), 0.01, 0.99))


def _explain(sample: dict, therapy: str, bundle: dict) -> dict:
    """Data-driven attribution: effect of each present feature on this drug's
    predicted sensitivity, by toggling it off and measuring the change."""
    did = bundle["name_to_id"][therapy]
    base = _score(bundle, sample, drug_ids=[did])[therapy]
    supporting, cautions = [], []
    for f in MUTATION_FEATURES + [ERBB2_AMP, MSI]:
        if float(sample.get(f, 0)) < 0.5:
            continue
        off = dict(sample)
        off[f] = 0.0
        eff_pct = _pct(base) - _pct(_score(bundle, off, drug_ids=[did])[therapy])
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


def predict(sample: dict, top_k: int | None = 8) -> dict:
    """Rank drugs for a tumor profile. Results are cached per (bundle, sample) so
    repeated/identical inputs — e.g. the built-in example files — are instant."""
    items = tuple(sorted(sample.items()))
    # id(bundle) keys the cache to the loaded model, so a reload invalidates it.
    return dict(_predict_cached(id(load_bundle()), items, top_k))


@lru_cache(maxsize=2048)
def _predict_cached(_bundle_id: int, items: tuple, top_k: int | None) -> dict:
    return _predict_impl(dict(items), top_k)


def _predict_impl(sample: dict, top_k: int | None = 8) -> dict:
    bundle = load_bundle()
    meta = bundle["drug_meta"]
    resid = bundle.get("resid_std", {})

    zs = _score(bundle, sample)   # {drug_name: blended sensitivity}
    names = list(zs.keys())
    order = sorted(names, key=lambda n: zs[n], reverse=True)
    if top_k:
        order = order[:top_k]
    zarr = np.array([zs[n] for n in names])
    confidence = _confidence(zarr, resid.get(order[0], 0.6))
    pcts = {n: _pct(zs[n]) for n in names}
    margin = round((pcts[order[0]] - pcts[order[1]]) / 100.0, 4) if len(order) > 1 else 0.0

    ranked = []
    for rank, n in enumerate(order, start=1):
        rs = resid.get(n, 0.6)
        exp = _explain(sample, n, bundle)
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
        "model_metrics": summary_metrics(bundle.get("metrics", {})),
    }


def predict_batch(samples: list[dict]) -> dict:
    bundle = load_bundle()
    meta = bundle["drug_meta"]
    resid = bundle.get("resid_std", {})
    names = [bundle["id_to_name"][d] for d in bundle["drug_ids"]]

    rows = []
    for i, s in enumerate(samples):
        zs = _score(bundle, s)
        order = sorted(zs, key=lambda n: zs[n], reverse=True)
        top, second = order[0], order[1]
        pcts = {n: _pct(zs[n]) for n in names}
        rows.append({
            "index": i + 1,
            "cancer_type": TISSUE_LABELS.get(s.get("tissue", ""), s.get("tissue", "")),
            "recommendation": top, "drug_class": meta.get(top, {}).get("target", ""),
            "match_percent": round(pcts[top], 1),
            "confidence": round(_confidence(np.array(list(zs.values())), resid.get(top, 0.6)), 4),
            "decision_margin": round((pcts[top] - pcts[second]) / 100.0, 4),
            "runner_up": second, "runner_up_percent": round(pcts[second], 1),
        })
    return {"n": len(rows), "therapies": names, "rows": rows,
            "model_metrics": summary_metrics(bundle.get("metrics", {}))}
