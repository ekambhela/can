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
import logging
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
    build_reliability,
    summary_metrics,
    feature_schema as _schema,
)

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS, "model.joblib")

log = logging.getLogger("karkive.predict")


class ModelUnavailable(RuntimeError):
    """The trained model could not be loaded — for any reason.

    A missing artifact is the obvious case, but the one that actually bites in
    production is a version skew: model.joblib is a pickle of scikit-learn
    estimators, so bumping scikit-learn/numpy without retraining can make
    joblib.load raise AttributeError, ModuleNotFoundError, or fail some internal
    reconstruct. Those aren't FileNotFoundError, and left untyped they surface
    as unhandled 500s — including on /api/health, the route the host polls to
    decide whether the deploy is alive.

    Everything that loads the model raises this instead, and the app maps it to
    a structured 503.
    """


class InvalidSample(ValueError):
    """Input the model cannot honestly score, carrying what the client must fix.

    Used where guessing would silently change the question being answered rather
    than merely lose a little precision — see the tissue handling in _normalize.
    The app maps this to a 422 that names the field and lists valid values.
    """

    def __init__(self, message: str, field: str | None = None, valid_values=None):
        super().__init__(message)
        self.field = field
        self.valid_values = list(valid_values) if valid_values is not None else None


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


def get_bundle() -> dict:
    """Load the model, converting ANY failure into ModelUnavailable.

    The single load entry point for the serving path: everything below calls
    this rather than load_bundle(), so no load failure can escape as an
    unhandled 500. The original exception is logged with its traceback (the
    operator needs it) and chained, but not exposed to the client.
    """
    try:
        return load_bundle()
    except Exception as exc:  # noqa: BLE001 — deliberately broad; see ModelUnavailable
        log.exception("model load failed (%s)", type(exc).__name__)
        raise ModelUnavailable(
            f"Model could not be loaded ({type(exc).__name__}: {exc})"
        ) from exc


# Drugs the model predicts poorly are flagged, not hidden, by default: a
# clinician-facing shortlist that silently drops candidates is its own kind of
# dishonesty. Set KARKIVE_EXCLUDE_LOW_RELIABILITY=1 to drop them from rankings
# instead.
EXCLUDE_LOW_RELIABILITY = os.environ.get("KARKIVE_EXCLUDE_LOW_RELIABILITY", "") \
    .strip().lower() in {"1", "true", "yes", "on"}


def reliability_map() -> dict:
    """{drug_name: {tier, spearman, r2}} for the loaded model.

    Prefers the bundle's stored table, falling back to deriving it from the
    bundle's own held-out metrics — bundles trained before this existed (the
    shipped v7 artifact) still get tiers without a retrain.
    """
    bundle = get_bundle()
    stored = bundle.get("reliability")
    if stored:
        return stored
    # Memoized onto the bundle itself, so the cache is scoped to the loaded
    # object and a reload invalidates it for free.
    derived = bundle.get("_derived_reliability")
    if derived is None:
        derived = build_reliability(bundle.get("metrics", {}))
        bundle["_derived_reliability"] = derived
    return derived


def api_metrics() -> dict:
    """Summary metrics only (no 369-entry per-drug dicts) — for API payloads."""
    return summary_metrics(get_bundle().get("metrics", {}))


def full_metrics() -> dict:
    """Complete metrics incl. per-drug R^2 / Spearman — for the /api/metrics route."""
    return get_bundle().get("metrics", {})


# ---------------------------------------------------------------------------
# Parsing / normalization
# ---------------------------------------------------------------------------
def _known_tissues() -> list[str]:
    """The tissue vocabulary the model was trained on, from the bundle.

    Propagates ModelUnavailable rather than falling back to a stub list: with no
    model we cannot validate a tissue against anything real, and quietly
    accepting input against a fabricated vocabulary is worse than a 503.
    """
    return get_bundle()["tissues"]


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


def _normalize(flat: dict, prefix: str = "") -> tuple[dict, list[str], list[str]]:
    """Coerce a raw record into a model sample.

    Returns (sample, warnings, specified_features) where `specified_features`
    lists the biomarkers the input actually stated — everything else is an
    assumption, and callers surface that rather than hiding it.
    """
    lookup = {str(k).strip().lower(): k for k in flat}
    tissues = _known_tissues()
    sample: dict = {}
    warnings: list[str] = []
    specified: list[str] = []

    # --- tissue: required, never guessed ------------------------------------
    # Tissue is not one feature among many. Tissue-blocked CV puts mean Spearman
    # at 0.10 vs 0.38 within-tissue, i.e. the model leans hard on tissue
    # identity. Defaulting a missing tissue to lung_NSCLC therefore doesn't
    # answer the question imprecisely — it confidently answers a different
    # question, with nothing in the response indicating that happened.
    src = lookup.get("tissue") or lookup.get("tissue_factor") or lookup.get("cancer_type")
    if src is None:
        raise InvalidSample(
            f"{prefix}'tissue' is required — the model's predictions depend "
            "heavily on it, so it cannot be inferred.",
            field="tissue", valid_values=tissues,
        )
    t = _coerce_tissue(flat[src], tissues)
    if t is None:
        raise InvalidSample(
            f"{prefix}unrecognized tissue '{flat[src]}'.",
            field="tissue", valid_values=tissues,
        )
    sample["tissue"] = t

    # --- binary biomarkers: absent means unknown, and we say so --------------
    # Unspecified markers are still scored as 0 (negative/wild-type): that is
    # what the model was trained on — the GDSC feature matrix has no missing
    # values, so the boosters never learned a missing-direction, and encoding
    # unspecified as NaN measured no better on held-out lines while changing
    # ~half of all top-1 picks (experiments/run_missing_features.py). So the
    # encoding stays; what changes is that the assumption is now reported.
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
            specified.append(f)

    assumed = [f for f in BINARY_FEATURES if f not in specified]
    if assumed:
        warnings.append(
            f"{prefix}{len(assumed)} of {len(BINARY_FEATURES)} biomarkers were not "
            f"specified and were assumed negative / wild-type: "
            f"{', '.join(FEATURE_LABEL.get(f, f) for f in assumed)}."
        )
    return sample, warnings, specified


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


def parse_sample(raw: bytes, filename: str = "") -> tuple[dict, list[str], list[str]]:
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
        s, w, _specified = _normalize(r, prefix=prefix)
        samples.append(s)
        warnings.extend(w)
    return samples, warnings


def sample_from_dict(d: dict) -> tuple[dict, list[str], list[str]]:
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
    """Sensitivity z -> percentile against the GDSC cell-line panel.

    Surfaced as `match_percent`. Reads as: "predicted more sensitive to this
    drug than X% of the cell lines the model was trained on". It is NOT a
    probability of clinical response, and not a probability of anything about
    this tumor — the panel is the reference, and the panel is cell lines.
    """
    return float(np.clip(norm.cdf(z) * 100.0, 0.0, 100.0))


def _separation_score(zs: np.ndarray, resid_top: float = 0.6) -> float:
    """How clearly the top pick separates from its *closest rivals* — the next
    few best drugs — relative to the model's own prediction noise.

    WHAT THIS IS NOT: it is not P(the recommendation is correct). Nothing here
    is calibrated against whether the top pick is actually the best drug for the
    line — held-out top-1 accuracy is 11.1%, so a reading of 0.84 emphatically
    does not mean "84% chance this is right". It is a *geometry* statistic about
    one prediction vector: it can be high for a confidently wrong answer whenever
    the model cleanly separates the wrong drug.

    WHAT IT IS: the gap between the top drug and the mean of ranks 2-6, in units
    of the top drug's held-out residual std, mapped through the normal CDF.
    Measuring against the whole panel would be useless — the best drug is always
    many std above the panel mean, so that saturates at ~1.0 for every sample.
    What varies is whether #1 stands apart from the handful of near-ties just
    behind it. A clear standout -> high; a cluster of near-ties -> ~0.5
    (honestly uncertain which single drug is best).

    Exposed as `separation_score`; also emitted as `confidence` for one release
    for backwards compatibility. See _predict_impl.
    """
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


def predict(sample: dict, top_k: int | None = 8, exclude_low_reliability: bool | None = None) -> dict:
    """Rank drugs for a tumor profile. Results are cached per (bundle, sample) so
    repeated/identical inputs — e.g. the built-in example files — are instant."""
    items = tuple(sorted(sample.items()))
    excl = EXCLUDE_LOW_RELIABILITY if exclude_low_reliability is None else exclude_low_reliability
    # id(bundle) keys the cache to the loaded model, so a reload invalidates it.
    return dict(_predict_cached(id(get_bundle()), items, top_k, excl))


@lru_cache(maxsize=2048)
def _predict_cached(_bundle_id: int, items: tuple, top_k: int | None, excl: bool) -> dict:
    return _predict_impl(dict(items), top_k, excl)


def _predict_impl(sample: dict, top_k: int | None = 8, excl: bool = False) -> dict:
    bundle = get_bundle()
    meta = bundle["drug_meta"]
    resid = bundle.get("resid_std", {})
    rel = reliability_map()

    zs = _score(bundle, sample)   # {drug_name: blended sensitivity}
    names = list(zs.keys())
    ranked_names = names
    if excl:
        kept = [n for n in names if rel.get(n, {}).get("tier") != "low"]
        ranked_names = kept or names   # never return an empty ranking
    order = sorted(ranked_names, key=lambda n: zs[n], reverse=True)
    if top_k:
        order = order[:top_k]
    zarr = np.array([zs[n] for n in ranked_names])
    separation = _separation_score(zarr, resid.get(order[0], 0.6))
    pcts = {n: _pct(zs[n]) for n in names}
    margin = round((pcts[order[0]] - pcts[order[1]]) / 100.0, 4) if len(order) > 1 else 0.0

    ranked = []
    for rank, n in enumerate(order, start=1):
        rs = resid.get(n, 0.6)
        exp = _explain(sample, n, bundle)
        r = rel.get(n, {})
        tier = r.get("tier", "unknown")
        ranked.append({
            "rank": rank, "therapy": n, "drug_class": meta.get(n, {}).get("target", ""),
            "sensitivity": round(zs[n], 4),
            "match_percent": round(pcts[n], 1),
            "ci_low": round(_pct(zs[n] - rs), 1),
            "ci_high": round(_pct(zs[n] + rs), 1),
            # How well the model predicts THIS drug on held-out lines. 34 of 369
            # drugs are "low" (R^2 below zero or no ranking signal) and could
            # otherwise reach rank 1 looking exactly like a well-predicted one.
            "reliability": tier,
            "low_reliability": tier == "low",
            "reliability_r2": r.get("r2"),
            "reliability_spearman": r.get("spearman"),
            "rationale": [s["text"] for s in exp["supporting"][:2]],
            "supporting": exp["supporting"], "cautions": exp["cautions"],
        })
    return {
        "recommendation": ranked[0]["therapy"],
        # How far #1 stands from ranks 2-6 in residual-std units — NOT a
        # probability that the pick is right. See _separation_score.
        "separation_score": round(separation, 4),
        # DEPRECATED alias for separation_score, kept one release for clients
        # written against the old name. Read separation_score instead.
        "confidence": round(separation, 4),
        "decision_margin": margin,
        "ranked": ranked,
        "low_reliability_count": sum(1 for r in ranked if r["low_reliability"]),
        "excluded_low_reliability": bool(excl),
        "model_metrics": summary_metrics(bundle.get("metrics", {})),
    }


def predict_batch(samples: list[dict], exclude_low_reliability: bool | None = None) -> dict:
    bundle = get_bundle()
    meta = bundle["drug_meta"]
    resid = bundle.get("resid_std", {})
    rel = reliability_map()
    names = [bundle["id_to_name"][d] for d in bundle["drug_ids"]]
    excl = EXCLUDE_LOW_RELIABILITY if exclude_low_reliability is None else exclude_low_reliability

    rows = []
    for i, s in enumerate(samples):
        zs = _score(bundle, s)
        candidates = list(zs)
        if excl:
            candidates = [n for n in candidates
                          if rel.get(n, {}).get("tier") != "low"] or list(zs)
        order = sorted(candidates, key=lambda n: zs[n], reverse=True)
        top, second = order[0], order[1]
        pcts = {n: _pct(zs[n]) for n in names}
        separation = round(_separation_score(np.array(list(zs.values())),
                                             resid.get(top, 0.6)), 4)
        rows.append({
            "index": i + 1,
            "cancer_type": TISSUE_LABELS.get(s.get("tissue", ""), s.get("tissue", "")),
            "recommendation": top, "drug_class": meta.get(top, {}).get("target", ""),
            "match_percent": round(pcts[top], 1),
            "separation_score": separation,
            "confidence": separation,   # DEPRECATED alias — see _predict_impl
            "reliability": rel.get(top, {}).get("tier", "unknown"),
            "low_reliability": rel.get(top, {}).get("tier") == "low",
            "decision_margin": round((pcts[top] - pcts[second]) / 100.0, 4),
            "runner_up": second, "runner_up_percent": round(pcts[second], 1),
        })
    return {"n": len(rows), "therapies": names, "rows": rows,
            "low_reliability_count": sum(1 for r in rows if r["low_reliability"]),
            "excluded_low_reliability": bool(excl),
            "model_metrics": summary_metrics(bundle.get("metrics", {}))}
