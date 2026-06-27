"""
Train and evaluate the tumor → chemotherapy matcher.

Model design
------------
We treat this as a *drug-response prediction* problem (the standard framing in
pharmacogenomics): for a given tumor sample we predict a continuous
sensitivity score for every therapy, then rank therapies by predicted score.

Concretely we fit one HistGradientBoostingRegressor per therapy wrapped in a
MultiOutputRegressor over a shared preprocessing pipeline (one-hot cancer
type, passthrough genomics/expression). Gradient-boosted trees handle the
mixed feature types, non-linear biomarker interactions, and resistance
gradients well, and give strong accuracy without hand-tuning.

Evaluation reports, on a held-out test split:
  * Top-1 accuracy  : how often the model's #1 therapy is the true best one
  * Top-3 accuracy  : how often the true best therapy is in the model's top 3
  * Mean per-drug R^2 and Spearman rank correlation of the sensitivity vector

Artifacts written to artifacts/:
  * model.joblib   : the fitted pipeline + therapy order + feature schema
  * metrics.json   : evaluation metrics surfaced in the web UI
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import spearmanr

from .biomarkers import (
    CANCER_TYPES,
    CONTINUOUS_FEATURES,
    MUTATION_FEATURES,
    THERAPY_NAMES,
)
from .generate_data import generate

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")

NUMERIC_FEATURES = MUTATION_FEATURES + CONTINUOUS_FEATURES


def build_pipeline(quantile: float | None = None) -> Pipeline:
    """Preprocessing + multi-output gradient-boosted regressor.

    When `quantile` is given, the trees optimize the pinball loss for that
    quantile instead of the mean, so a pair of pipelines (e.g. 0.1 / 0.9)
    yields a calibrated prediction interval per therapy.
    """
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(categories=[CANCER_TYPES], handle_unknown="ignore"),
             ["cancer_type"]),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    if quantile is None:
        base = HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=None, max_iter=400, learning_rate=0.06,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, random_state=0,
        )
    else:
        base = HistGradientBoostingRegressor(
            loss="quantile", quantile=quantile,
            max_depth=None, max_iter=200, learning_rate=0.07,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, random_state=0,
        )
    return Pipeline([("pre", pre), ("model", MultiOutputRegressor(base))])


def evaluate(model: Pipeline, X_test: pd.DataFrame, Y_test: pd.DataFrame) -> dict:
    """Compute ranking + regression metrics on the held-out test set."""
    pred = model.predict(X_test)
    pred = np.asarray(pred)
    true = Y_test.to_numpy()

    true_best = true.argmax(axis=1)
    pred_order = pred.argsort(axis=1)[:, ::-1]      # high score first
    top1 = float(np.mean(pred_order[:, 0] == true_best))
    top3 = float(np.mean([tb in row[:3] for tb, row in zip(true_best, pred_order)]))

    per_drug_r2 = [float(r2_score(true[:, j], pred[:, j])) for j in range(true.shape[1])]
    # Spearman of the per-sample ranking across drugs, averaged over samples.
    rhos = []
    for i in range(true.shape[0]):
        rho = spearmanr(true[i], pred[i]).correlation
        if not np.isnan(rho):
            rhos.append(rho)

    return {
        "n_test": int(true.shape[0]),
        "n_therapies": int(true.shape[1]),
        "top1_accuracy": round(top1, 4),
        "top3_accuracy": round(top3, 4),
        "mean_r2": round(float(np.mean(per_drug_r2)), 4),
        "mean_spearman": round(float(np.mean(rhos)), 4),
        "per_drug_r2": {THERAPY_NAMES[j]: round(per_drug_r2[j], 3)
                        for j in range(len(THERAPY_NAMES))},
    }


def main(n_samples: int = 9000, seed: int = 7) -> dict:
    os.makedirs(ARTIFACTS, exist_ok=True)
    print("Generating synthetic pharmacogenomic dataset ...")
    X, Y = generate(n_samples=n_samples, seed=seed)
    Y = Y[THERAPY_NAMES]  # lock column order

    # Deterministic 80/20 split.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    tr, te = idx[:cut], idx[cut:]
    X_tr, X_te = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
    Y_tr, Y_te = Y.iloc[tr].reset_index(drop=True), Y.iloc[te].reset_index(drop=True)

    print(f"Training on {len(X_tr)} samples, {len(THERAPY_NAMES)} therapies ...")
    model = build_pipeline()
    model.fit(X_tr, Y_tr)

    # Quantile models give a per-therapy prediction interval (uncertainty band).
    print("Training quantile interval models (q10 / q90) ...")
    model_lo = build_pipeline(quantile=0.10)
    model_hi = build_pipeline(quantile=0.90)
    model_lo.fit(X_tr, Y_tr)
    model_hi.fit(X_tr, Y_tr)

    print("Evaluating ...")
    metrics = evaluate(model, X_te, Y_te)
    # Empirical interval coverage: fraction of held-out truths inside [q10, q90].
    lo = np.asarray(model_lo.predict(X_te))
    hi = np.asarray(model_hi.predict(X_te))
    inside = (Y_te.to_numpy() >= lo) & (Y_te.to_numpy() <= hi)
    metrics["interval_coverage"] = round(float(inside.mean()), 4)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_drug_r2"}, indent=2))

    bundle = {
        "model": model,
        "model_lo": model_lo,
        "model_hi": model_hi,
        "therapies": THERAPY_NAMES,
        "mutation_features": MUTATION_FEATURES,
        "continuous_features": CONTINUOUS_FEATURES,
        "cancer_types": CANCER_TYPES,
        "metrics": metrics,
        "version": 2,
    }
    joblib.dump(bundle, os.path.join(ARTIFACTS, "model.joblib"))
    with open(os.path.join(ARTIFACTS, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Saved model + metrics to {ARTIFACTS}")
    return metrics


if __name__ == "__main__":
    main()
