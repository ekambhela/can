"""
Train the Karkive drug-response model on real GDSC data.

For each drug we fit a gradient-boosted regressor that predicts a cell line's
sensitivity (-z(logIC50)) from its curated genomic profile (driver mutations,
ERBB2 amplification, MSI, tissue). Drugs are modeled independently because each
was screened on a different (overlapping) subset of cell lines, so the target
matrix is sparse; per-drug fitting uses exactly the lines with a measurement.

Reported on a held-out 20% of cell lines:
  * per-drug R^2 and Spearman rank correlation (does predicted sensitivity track
    the real IC50 ranking?)
  * top-1 / top-3 accuracy of recommending the drug a line is truly most
    sensitive to.

Artifacts -> artifacts/: model.joblib (per-drug models + schema + metrics),
metrics.json.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import spearmanr

from .gdsc import (
    BINARY_FEATURES,
    DRUG_COL,
    DRUGS,
    THERAPY_CLASS,
    load_frame,
)

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


def build_pipeline(tissues: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("tissue", OneHotEncoder(categories=[tissues], handle_unknown="ignore"), ["tissue"]),
        ("bin", "passthrough", BINARY_FEATURES),
    ])
    gbm = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, l2_regularization=1.0,
        max_leaf_nodes=15, early_stopping=True, validation_fraction=0.1,
        random_state=0,
    )
    return Pipeline([("pre", pre), ("gbm", gbm)])


def main(seed: int = 0) -> dict:
    os.makedirs(ARTIFACTS, exist_ok=True)
    print("Loading real GDSC release-17 data ...")
    feats, targets, tissues = load_frame()
    n = len(feats)
    print(f"{n} cell lines, {len(DRUGS)} drugs, {len(BINARY_FEATURES)} binary features, "
          f"{len(tissues)} tissues")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr, te = set(idx[:cut].tolist()), idx[cut:]

    models, per_r2, per_rho, resid_std = {}, {}, {}, {}
    test_pred = {}   # drug_id -> {line_index: predicted sensitivity}
    for did, col in DRUG_COL.items():
        name = DRUGS[did][0]
        y = targets[col].to_numpy()
        obs = ~np.isnan(y)
        tr_i = np.array([i for i in range(n) if i in tr and obs[i]])
        te_i = np.array([i for i in idx[cut:] if obs[i]])

        pipe = build_pipeline(tissues)
        pipe.fit(feats.iloc[tr_i], y[tr_i])
        p = pipe.predict(feats.iloc[te_i])

        per_r2[name] = round(float(r2_score(y[te_i], p)), 3)
        rho = spearmanr(y[te_i], p).correlation
        per_rho[name] = round(float(0 if np.isnan(rho) else rho), 3)
        resid_std[name] = round(float(np.std(y[te_i] - p)), 3)
        test_pred[did] = dict(zip(te_i.tolist(), p.tolist()))
        models[name] = pipe

    # ranking metrics across drugs, per held-out line
    hits = hits3 = tot = 0
    truth = {did: targets[col].to_numpy() for did, col in DRUG_COL.items()}
    for i in idx[cut:]:
        avail = [d for d in DRUG_COL if not np.isnan(truth[d][i]) and i in test_pred[d]]
        if len(avail) < 3:
            continue
        best = max(avail, key=lambda d: truth[d][i])
        order = sorted(avail, key=lambda d: test_pred[d][i], reverse=True)
        tot += 1
        hits += order[0] == best
        hits3 += best in order[:3]

    metrics = {
        "n_cell_lines": n,
        "n_drugs": len(DRUGS),
        "n_features": len(BINARY_FEATURES) + len(tissues),
        "mean_r2": round(float(np.mean(list(per_r2.values()))), 3),
        "mean_spearman": round(float(np.mean(list(per_rho.values()))), 3),
        "top1_accuracy": round(hits / tot, 3),
        "top3_accuracy": round(hits3 / tot, 3),
        "n_test_lines": tot,
        "per_drug_spearman": per_rho,
        "per_drug_r2": per_r2,
    }
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("per_drug_spearman", "per_drug_r2")}, indent=2))

    bundle = {
        "models": models,
        "tissues": tissues,
        "binary_features": BINARY_FEATURES,
        "drug_meta": {DRUGS[i][0]: {"id": i, "target": DRUGS[i][1]} for i in DRUGS},
        "therapy_class": THERAPY_CLASS,
        "resid_std": resid_std,
        "metrics": metrics,
        "version": 3,
        "data": "GDSC release 17 (real cell-line drug response)",
    }
    joblib.dump(bundle, os.path.join(ARTIFACTS, "model.joblib"))
    with open(os.path.join(ARTIFACTS, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Saved model + metrics to {ARTIFACTS}")
    return metrics


if __name__ == "__main__":
    main()
