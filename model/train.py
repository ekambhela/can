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

Target normalization is fit on the **train split only** (see
`fit_target_scaler`) to avoid leaking test-line statistics into the target — a
subtle but real form of data leakage.

Artifacts -> artifacts/: model.joblib (per-drug models + schema + metrics),
metrics.json.
"""

from __future__ import annotations

import json
import logging
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

log = logging.getLogger("karkive.train")


# ---------------------------------------------------------------------------
# Target scaling (fit on TRAIN lines only — no test-set leakage)
# ---------------------------------------------------------------------------
def fit_target_scaler(y: np.ndarray, train_idx) -> tuple[float, float]:
    """Fit the per-drug target scaler (mean, std) using ONLY `train_idx` rows.

    The model's target is the sensitivity `-z(logIC50)`. The mean/std that define
    that z-score must be estimated from training cell lines alone; estimating
    them over all lines (including the held-out test lines) would let test-set
    statistics influence the target the model is trained and scored against.
    """
    yt = np.asarray(y, dtype=float)[np.asarray(list(train_idx), dtype=int)]
    yt = yt[~np.isnan(yt)]
    mu = float(np.mean(yt))
    sd = float(np.std(yt))
    return mu, (sd if sd > 1e-9 else 1.0)


def apply_target_scaler(y: np.ndarray, mu: float, sd: float) -> np.ndarray:
    """Map raw log-IC50 to sensitivity: higher = more sensitive (lower IC50)."""
    return -(np.asarray(y, dtype=float) - mu) / sd


def build_pipeline(tissues: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("tissue", OneHotEncoder(categories=[tissues], handle_unknown="ignore"), ["tissue"]),
        ("bin", "passthrough", BINARY_FEATURES),
    ])
    gbm = HistGradientBoostingRegressor(
        max_iter=170, learning_rate=0.07, l2_regularization=1.5,
        max_leaf_nodes=13, early_stopping=True, validation_fraction=0.12,
        random_state=0,
    )
    return Pipeline([("pre", pre), ("gbm", gbm)])


def run_training(seed: int = 0, max_lines: int | None = None,
                 max_drugs: int | None = None) -> dict:
    """Train per-drug models and return the full bundle dict (unsaved).

    `max_lines` / `max_drugs` subsample the data for fast smoke tests; leave both
    None for the real run.
    """
    feats, targets, tissues = load_frame()
    if max_lines is not None:
        feats = feats.iloc[:max_lines].reset_index(drop=True)
        targets = {c: s.iloc[:max_lines].reset_index(drop=True) for c, s in targets.items()}
    n = len(feats)

    drug_col = dict(list(DRUG_COL.items())[:max_drugs]) if max_drugs else DRUG_COL
    log.info("%d cell lines, %d drugs, %d binary features, %d tissues",
             n, len(drug_col), len(BINARY_FEATURES), len(tissues))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr = set(idx[:cut].tolist())
    test_idx = idx[cut:]

    models, per_r2, per_rho, resid_std = {}, {}, {}, {}
    truth_sens = {}   # drug_id -> full sensitivity array (test lines scaled by TRAIN stats)
    test_pred = {}    # drug_id -> {line_index: predicted sensitivity}
    for did, col in drug_col.items():
        name = DRUGS[did][0]
        y_raw = targets[col].to_numpy()
        obs = ~np.isnan(y_raw)
        tr_i = np.array([i for i in range(n) if i in tr and obs[i]])
        te_i = np.array([i for i in test_idx if obs[i]])
        if tr_i.size < 3 or te_i.size < 1:
            continue

        # Scale the target using TRAIN-line statistics only, then apply to all.
        mu, sd = fit_target_scaler(y_raw, tr_i)
        y = apply_target_scaler(y_raw, mu, sd)
        truth_sens[did] = y

        pipe = build_pipeline(tissues)
        pipe.fit(feats.iloc[tr_i], y[tr_i])
        p = pipe.predict(feats.iloc[te_i])

        per_r2[name] = round(float(r2_score(y[te_i], p)), 3)
        rho = spearmanr(y[te_i], p).correlation
        per_rho[name] = round(float(0 if np.isnan(rho) else rho), 3)
        resid_std[name] = round(float(np.std(y[te_i] - p)), 3)
        test_pred[did] = dict(zip(te_i.tolist(), p.tolist()))
        models[name] = pipe

    # ranking metrics across drugs, per held-out line. With a large panel, top-1
    # is near-random, so we also report top-10 and the mean percentile rank of
    # the truly-best drug within the model's ranking.
    hits = hits3 = hits10 = tot = 0
    pct_ranks = []
    for i in test_idx:
        avail = [d for d in test_pred
                 if not np.isnan(truth_sens[d][i]) and i in test_pred[d]]
        if len(avail) < 5:
            continue
        best = max(avail, key=lambda d: truth_sens[d][i])
        order = sorted(avail, key=lambda d: test_pred[d][i], reverse=True)
        tot += 1
        hits += order[0] == best
        hits3 += best in order[:3]
        hits10 += best in order[:10]
        pct_ranks.append(1 - order.index(best) / (len(order) - 1))  # 1 = best ranked first

    metrics = {
        "n_cell_lines": n,
        "n_drugs": len(models),
        "n_features": len(BINARY_FEATURES) + len(tissues),
        "mean_r2": round(float(np.mean(list(per_r2.values()))), 3) if per_r2 else 0.0,
        "mean_spearman": round(float(np.mean(list(per_rho.values()))), 3) if per_rho else 0.0,
        "top1_accuracy": round(hits / tot, 3) if tot else 0.0,
        "top3_accuracy": round(hits3 / tot, 3) if tot else 0.0,
        "top10_accuracy": round(hits10 / tot, 3) if tot else 0.0,
        "best_drug_percentile": round(float(np.mean(pct_ranks)), 3) if pct_ranks else 0.0,
        "n_test_lines": tot,
        "per_drug_spearman": per_rho,
        "per_drug_r2": per_r2,
    }

    return {
        "models": models,
        "tissues": tissues,
        "binary_features": BINARY_FEATURES,
        "drug_meta": {DRUGS[i][0]: {"id": i, "target": THERAPY_CLASS[DRUGS[i][0]]}
                      for i in DRUGS if DRUGS[i][0] in models},
        "therapy_class": THERAPY_CLASS,
        "resid_std": resid_std,
        "metrics": metrics,
        "version": 6,
        "data": "GDSC release 17 (GDSC1) + GDSC2 (25Feb20) real cell-line drug response",
    }


# Keys that are large per-drug diagnostic dicts, not for API payloads.
PER_DRUG_METRIC_KEYS = ("per_drug_spearman", "per_drug_r2")


def summary_metrics(metrics: dict) -> dict:
    """Metrics minus the big per-drug dicts — safe for API responses."""
    return {k: v for k, v in metrics.items() if k not in PER_DRUG_METRIC_KEYS}


def main(seed: int = 0) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    os.makedirs(ARTIFACTS, exist_ok=True)
    log.info("Loading real GDSC data and training ...")
    bundle = run_training(seed=seed)
    metrics = bundle["metrics"]
    log.info("metrics:\n%s", json.dumps(summary_metrics(metrics), indent=2))

    # compress=3 shrinks the joblib ~4-6x; joblib.load auto-detects it.
    joblib.dump(bundle, os.path.join(ARTIFACTS, "model.joblib"), compress=3)
    with open(os.path.join(ARTIFACTS, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("Saved model + metrics to %s", ARTIFACTS)
    return metrics


if __name__ == "__main__":
    main()
