"""
Per-drug baseline component of the shipped ensemble.

One HistGradientBoostingRegressor per drug on the curated cell features. On its
own this is the original Karkive model; in production its predictions are blended
with the multi-task model (model/mtl.py) — the two have complementary strengths
(per-drug calibration vs cross-drug ranking), and the blend beats either alone on
a held-out-by-cell-line test set. See experiments/ and the README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(tissues, cell_cols):
    pre = ColumnTransformer([
        ("tissue", OneHotEncoder(categories=[tissues], handle_unknown="ignore"), ["tissue"]),
        ("bin", "passthrough", cell_cols),
    ])
    gbm = HistGradientBoostingRegressor(
        max_iter=170, learning_rate=0.07, l2_regularization=1.5,
        max_leaf_nodes=13, early_stopping=True, validation_fraction=0.12,
        random_state=0,
    )
    return Pipeline([("pre", pre), ("gbm", gbm)])


def fit_all(feats, truth, train_idx, cell_cols, tissues, drug_ids, id_to_name):
    X = feats[cell_cols + ["tissue"]]
    models = {}
    for d in drug_ids:
        y = truth[d]
        obs = ~np.isnan(y)
        tr = np.array([i for i in train_idx if obs[i]])
        if tr.size < 10:
            continue
        pipe = build_pipeline(tissues, cell_cols)
        pipe.fit(X.iloc[tr], y[tr])
        models[id_to_name[d]] = pipe
    return models


def predict_pairs(models, feats, truth, eval_idx, cell_cols, drug_ids, id_to_name) -> dict:
    X = feats[cell_cols + ["tissue"]]
    out = {}
    for d in drug_ids:
        name = id_to_name[d]
        if name not in models:
            continue
        y = truth[d]
        obs = ~np.isnan(y)
        ev = np.array([i for i in eval_idx if obs[i]])
        if ev.size == 0:
            continue
        p = models[name].predict(X.iloc[ev])
        out[int(d)] = {int(i): float(pp) for i, pp in zip(ev, p)}
    return out


def score_pairs(models, samples, cell_cols, pairs, id_to_name) -> np.ndarray:
    """Score arbitrary (sample, drug) pairs, one predict call per DRUG.

    `pairs` is [(sample_index, drug_id), ...]. Returns an array aligned to
    `pairs`, with NaN where that drug has no fitted model.

    Two batching wins over calling this per (sample, drug):
      * every per-drug pipeline shares an identical preprocessor (same tissue
        categories + passthrough), so the ColumnTransformer runs ONCE for all
        sample rows rather than once per drug;
      * each booster is then called once on all the rows it needs, instead of
        once per row — which is what made a 500-row cohort 369 x 500 calls.
    """
    rows = []
    for s in samples:
        row = {c: float(s.get(c, 0.0)) for c in cell_cols}
        row["tissue"] = s.get("tissue")
        rows.append(row)
    X = pd.DataFrame(rows)

    by_drug: dict = {}
    for k, (si, d) in enumerate(pairs):
        by_drug.setdefault(id_to_name[d], []).append((k, si))

    out = np.full(len(pairs), np.nan, dtype=float)
    Xt = None
    for name, items in by_drug.items():
        m = models.get(name)
        if m is None:
            continue                      # leaves NaN -> caller falls back to mtl
        if Xt is None:
            Xt = m.named_steps["pre"].transform(X)
        idx = [si for _k, si in items]
        preds = m.named_steps["gbm"].predict(Xt[idx])
        for (k, _si), p in zip(items, preds):
            out[k] = float(p)
    return out


def score_sample(models, sample, cell_cols, drug_ids, id_to_name) -> dict:
    """Score one tumor profile against drugs. Returns {drug_name: sensitivity}."""
    preds = score_pairs(models, [sample], cell_cols,
                        [(0, d) for d in drug_ids], id_to_name)
    return {id_to_name[d]: float(p) for d, p in zip(drug_ids, preds)
            if not np.isnan(p)}
