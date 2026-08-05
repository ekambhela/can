"""
Multi-task ("borrow strength across drugs") model for Karkive.

Instead of 369 independent per-drug regressors, a SINGLE
HistGradientBoostingRegressor is trained on one row per observed
(cell_line, drug) pair:

    features = [curated cell genomics + tissue] + [drug pathway + target multi-hot
                + a capped drug-identity category], target = -z(logIC50)

The shared genomic->response mapping is estimated across ~200k pairs (far more
than any single drug's ~600 lines), so it generalizes better per drug, while the
drug-identity category preserves per-drug fidelity. This beat the per-drug
baseline on a held-out (by cell line) test set — see experiments/ and the README.

Methodology note: the per-drug target z-score is fit on TRAIN lines only
(model/train.py), so held-out statistics never leak into the target.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

CAT_COLS = ["tissue", "drug_pathway", "drug_id"]
GBM_PARAMS = dict(max_leaf_nodes=127, learning_rate=0.05, max_iter=600,
                  l2_regularization=1.0, early_stopping=True,
                  validation_fraction=0.1, random_state=0,
                  categorical_features="from_dtype")


def _target_tokens(targets: str):
    if not targets or str(targets).lower() == "nan":
        return []
    return [t.strip().upper() for t in re.split(r"[,;/]| and ", str(targets)) if t.strip()]


def build_drug_features(drug_ids, obs_counts=None, min_drugs=3, id_cap=254) -> dict:
    """Drug representation: target-pathway + multi-hot targets + capped identity.

    `obs_counts` {drug_id: n train observations} decides which drugs keep their
    own identity category (the `id_cap` most-screened) vs fall into 'OTHER',
    keeping cardinality under HistGBM's 255-category limit.

    Training-time only: the resulting dict is stored in the bundle, so serving
    never calls this. The data-layer import is therefore deferred into the body —
    importing model.mtl must not pull in `data/` (see model/schema.py).
    """
    from .gdsc import DRUGS

    tok_counts = Counter()
    per_drug = {}
    for d in drug_ids:
        toks = set(_target_tokens(DRUGS[d][2]))
        per_drug[d] = toks
        tok_counts.update(toks)
    vocab = [t for t, c in tok_counts.items() if c >= min_drugs]
    tcol = {t: f"dt_{re.sub(r'[^A-Z0-9]', '_', t)}" for t in vocab}
    multihot = {d: {tcol[t]: 1.0 for t in per_drug[d] if t in tcol} for d in drug_ids}
    pathway = {d: (DRUGS[d][1] or "Other") for d in drug_ids}

    if obs_counts:
        top = set(sorted(drug_ids, key=lambda d: obs_counts.get(d, 0), reverse=True)[:id_cap])
    else:
        top = set(drug_ids)
    id_bucket = {d: (str(d) if d in top else "OTHER") for d in drug_ids}
    return {"pathway": pathway, "target_cols": list(tcol.values()),
            "multihot": multihot, "id_bucket": id_bucket}


def _as_cat(values, cats):
    """Categorical with `cats` levels; values outside the vocabulary become NaN
    (HistGBM handles that as missing) — avoids a pandas out-of-category error."""
    catset = set(cats)
    return pd.Categorical([v if v in catset else None for v in values], categories=cats)


def design_columns(cell_cols, drug_feat):
    """Full ordered feature-column list for the design matrix."""
    return list(cell_cols) + list(drug_feat["target_cols"]) + CAT_COLS


def _build_long(feats, tissue_arr, truth, line_idx, cell_cols, drug_feat, drug_ids):
    pathway = drug_feat["pathway"]
    target_cols = drug_feat["target_cols"]
    multihot = drug_feat["multihot"]
    id_bucket = drug_feat["id_bucket"]
    cell = feats[cell_cols].to_numpy(dtype=np.float32)
    line_set = list(line_idx)
    rows, tiss, dpath, dbk, y, ln, did = [], [], [], [], [], [], []
    for d in drug_ids:
        t = truth[d]
        mh = [multihot[d].get(c, 0.0) for c in target_cols]
        for i in line_set:
            if not np.isnan(t[i]):
                rows.append(np.concatenate([cell[i], mh]))
                tiss.append(tissue_arr[i])
                dpath.append(pathway[d])
                dbk.append(id_bucket[d])
                y.append(t[i])
                ln.append(i)
                did.append(d)
    cols = list(cell_cols) + list(target_cols)
    X = pd.DataFrame(np.asarray(rows, dtype=np.float32), columns=cols)
    X["tissue"] = pd.Series(tiss, dtype="category")
    X["drug_pathway"] = pd.Series(dpath, dtype="category")
    X["drug_id"] = pd.Series(dbk, dtype="category")
    return X, np.asarray(y, dtype=np.float32), np.asarray(ln), np.asarray(did)


def fit(feats, tissue_arr, truth, train_idx, cell_cols, drug_feat, drug_ids, **params):
    """Train one multi-task model on all observed pairs with line in train_idx."""
    p = dict(GBM_PARAMS)
    p.update(params)
    X, y, _, _ = _build_long(feats, tissue_arr, truth, train_idx, cell_cols, drug_feat, drug_ids)
    model = HistGradientBoostingRegressor(**p)
    model.fit(X, y)
    cat_levels = {c: list(X[c].cat.categories) for c in CAT_COLS}
    return model, cat_levels


def predict_pairs(model, cat_levels, feats, tissue_arr, truth, eval_idx,
                  cell_cols, drug_feat, drug_ids) -> dict:
    """Predict every observed (eval line, drug) pair. Returns {drug_id: {line: sens}}."""
    X, _, ln, did = _build_long(feats, tissue_arr, truth, eval_idx, cell_cols, drug_feat, drug_ids)
    for c in CAT_COLS:
        X[c] = _as_cat(X[c], cat_levels[c])
    p = model.predict(X)
    out = {}
    for d, i, pp in zip(did, ln, p):
        out.setdefault(int(d), {})[int(i)] = float(pp)
    return out


# ---------------------------------------------------------------------------
# Inference for a single sample (used by model/predict.py)
# ---------------------------------------------------------------------------
def score_pairs(bundle: dict, samples: list[dict], pairs) -> np.ndarray:
    """Score arbitrary (sample, drug) pairs in ONE predict call.

    `pairs` is [(sample_index, drug_id), ...] indexing into `samples`. Returns
    predictions aligned to `pairs`.

    Boosted-tree prediction is row-independent, so batching is exact — the point
    is purely to stop paying per-call overhead 240 times per request (the
    explanation pass) or 184,500 times per cohort.
    """
    model = bundle["model"]
    cell_cols = bundle["cell_cols"]
    drug_feat = bundle["drug_feat"]
    cat_levels = bundle["cat_levels"]
    target_cols = drug_feat["target_cols"]
    multihot, pathway, id_bucket = (drug_feat["multihot"], drug_feat["pathway"],
                                    drug_feat["id_bucket"])

    cell_vecs = [[float(s.get(c, 0.0)) for c in cell_cols] for s in samples]
    mh_cache: dict = {}

    rows, tiss, dpath, dbk = [], [], [], []
    for si, d in pairs:
        mh = mh_cache.get(d)
        if mh is None:
            mh = mh_cache[d] = [multihot[d].get(c, 0.0) for c in target_cols]
        rows.append(cell_vecs[si] + mh)
        tiss.append(samples[si].get("tissue"))
        dpath.append(pathway[d])
        dbk.append(id_bucket[d])

    cols = list(cell_cols) + list(target_cols)
    X = pd.DataFrame(np.asarray(rows, dtype=np.float32), columns=cols)
    X["tissue"] = _as_cat(tiss, cat_levels["tissue"])
    X["drug_pathway"] = _as_cat(dpath, cat_levels["drug_pathway"])
    X["drug_id"] = _as_cat(dbk, cat_levels["drug_id"])
    return model.predict(X)


def score_sample(bundle: dict, sample: dict, drug_ids=None) -> dict:
    """Score one tumor profile against drugs. Returns {drug_name: sensitivity}."""
    ids = drug_ids if drug_ids is not None else bundle["drug_ids"]
    preds = score_pairs(bundle, [sample], [(0, d) for d in ids])
    id_to_name = bundle["id_to_name"]
    return {id_to_name[d]: float(p) for d, p in zip(ids, preds)}
