"""
Train the Karkive drug-response model on real GDSC data.

The model is a single MULTI-TASK gradient-boosted regressor over
(cell_line, drug) pairs (see model/mtl.py) — it beat 369 independent per-drug
regressors on a held-out-by-cell-line test set (see experiments/ and README).

Methodology enforced here:
  * Split is BY CELL LINE (a line is never in both train and test).
  * The per-drug target z-score is fit on TRAIN lines only (fit_target_scaler),
    so held-out statistics never leak into the target.
  * Reported metrics come from that held-out test split; we ALSO report
    tissue-blocked CV (holding out whole tissues) as an honest harder metric.
  * The shipped model is then refit on ALL cell lines for the best predictions.

Artifacts -> artifacts/: model.joblib (multi-task model + schema + metrics),
metrics.json.
"""

from __future__ import annotations

import json
import logging
import os

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from . import mtl, perdrug
from .gdsc import DRUG_COL, DRUGS, THERAPY_CLASS, load_frame

# ensemble weight on the per-drug component (rest on multi-task); chosen on the
# validation split — see experiments/run_ensemble.py.
BLEND_W_PERDRUG = 0.35

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")

log = logging.getLogger("karkive.train")


# ---------------------------------------------------------------------------
# Target scaling (fit on TRAIN lines only — no test-set leakage)
# ---------------------------------------------------------------------------
def fit_target_scaler(y: np.ndarray, train_idx) -> tuple[float, float]:
    """Fit the per-drug target scaler (mean, std) using ONLY `train_idx` rows.

    The model's target is the sensitivity `-z(logIC50)`. Estimating that z over
    all lines (including held-out test lines) would let test statistics influence
    the target the model is trained and scored against — a subtle leak.
    """
    yt = np.asarray(y, dtype=float)[np.asarray(list(train_idx), dtype=int)]
    yt = yt[~np.isnan(yt)]
    mu = float(np.mean(yt)) if yt.size else 0.0
    sd = float(np.std(yt))
    return mu, (sd if sd > 1e-9 else 1.0)


def apply_target_scaler(y: np.ndarray, mu: float, sd: float) -> np.ndarray:
    """Map raw log-IC50 to sensitivity: higher = more sensitive (lower IC50)."""
    return -(np.asarray(y, dtype=float) - mu) / sd


def _build_truth(targets_raw, drug_ids, train_idx):
    truth = {}
    for d in drug_ids:
        mu, sd = fit_target_scaler(targets_raw[d], train_idx)
        truth[d] = apply_target_scaler(targets_raw[d], mu, sd)
    return truth


def _obs_counts(targets_raw, drug_ids, idx):
    return {d: int(np.sum(~np.isnan(targets_raw[d][idx]))) for d in drug_ids}


def _blend(pd_pred, mt_pred, w=BLEND_W_PERDRUG) -> dict:
    """Blend per-drug and multi-task pair-predictions: w*perdrug + (1-w)*mtl."""
    out = {}
    for d in set(pd_pred) | set(mt_pred):
        a, b = pd_pred.get(d, {}), mt_pred.get(d, {})
        keys = set(a) & set(b)
        if keys:
            out[d] = {i: w * a[i] + (1 - w) * b[i] for i in keys}
        else:
            out[d] = a or b
    return out


# ---------------------------------------------------------------------------
# Metrics (identical space to the experiments harness)
# ---------------------------------------------------------------------------
def _metrics(pred, truth, drug_ids, eval_idx) -> dict:
    per_rho, per_r2 = {}, {}
    for d in drug_ids:
        pr = pred.get(d)
        if not pr:
            continue
        xs, ys = [], []
        for i in eval_idx:
            p = pr.get(int(i))
            if p is not None and not np.isnan(truth[d][i]):
                xs.append(p)
                ys.append(truth[d][i])
        if len(xs) >= 5 and np.std(xs) > 1e-9 and np.std(ys) > 1e-9:
            r = spearmanr(ys, xs).correlation
            per_rho[DRUGS[d][0]] = round(float(0 if np.isnan(r) else r), 3)
            per_r2[DRUGS[d][0]] = round(float(r2_score(ys, xs)), 3)

    hits = hits3 = hits10 = tot = 0
    pcts = []
    for i in eval_idx:
        avail = [d for d in drug_ids
                 if d in pred and pred[d].get(int(i)) is not None and not np.isnan(truth[d][i])]
        if len(avail) < 5:
            continue
        best = max(avail, key=lambda d: truth[d][i])
        order = sorted(avail, key=lambda d: pred[d][int(i)], reverse=True)
        tot += 1
        hits += order[0] == best
        hits3 += best in order[:3]
        hits10 += best in order[:10]
        pcts.append(1 - order.index(best) / (len(order) - 1))

    return {
        "mean_spearman": round(float(np.mean(list(per_rho.values()))), 3) if per_rho else 0.0,
        "mean_r2": round(float(np.mean(list(per_r2.values()))), 3) if per_r2 else 0.0,
        "top1_accuracy": round(hits / tot, 3) if tot else 0.0,
        "top3_accuracy": round(hits3 / tot, 3) if tot else 0.0,
        "top10_accuracy": round(hits10 / tot, 3) if tot else 0.0,
        "best_drug_percentile": round(float(np.mean(pcts)), 3) if pcts else 0.0,
        "n_test_lines": tot,
        "per_drug_spearman": per_rho,
        "per_drug_r2": per_r2,
    }


def _resid_std(pred, truth, drug_ids, eval_idx) -> dict:
    out = {}
    for d in drug_ids:
        pr = pred.get(d)
        if not pr:
            continue
        res = [truth[d][i] - pr[int(i)] for i in eval_idx
               if pr.get(int(i)) is not None and not np.isnan(truth[d][i])]
        if len(res) >= 3:
            out[DRUGS[d][0]] = round(float(np.std(res)), 3)
    return out


def _tissue_blocked_split(tissue_arr, seed=0, test_frac=0.25):
    rng = np.random.default_rng(seed)
    tissues = sorted(set(tissue_arr.tolist()))
    rng.shuffle(tissues)
    n = len(tissue_arr)
    test_t, cum = set(), 0
    for t in tissues:
        if cum < test_frac * n:
            test_t.add(t)
            cum += int((tissue_arr == t).sum())
    test_idx = np.array([i for i in range(n) if tissue_arr[i] in test_t])
    train_idx = np.array([i for i in range(n) if tissue_arr[i] not in test_t])
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_training(seed: int = 0, max_lines: int | None = None,
                 max_drugs: int | None = None) -> dict:
    """Train the multi-task model and return the full bundle dict (unsaved)."""
    feats, targets_col, tissues = load_frame()
    # rekey targets by canonical drug id (load_frame keys by column name)
    targets_raw = {d: targets_col[DRUG_COL[d]].to_numpy() for d in DRUGS}
    if max_lines is not None:
        feats = feats.iloc[:max_lines].reset_index(drop=True)
        targets_raw = {d: v[:max_lines] for d, v in targets_raw.items()}
    n = len(feats)
    tissue_arr = feats["tissue"].to_numpy()
    from .gdsc import BINARY_FEATURES
    cell_cols = list(BINARY_FEATURES)
    drug_ids = list(DRUGS.keys())[:max_drugs] if max_drugs else list(DRUGS.keys())

    id_to_name = {d: DRUGS[d][0] for d in drug_ids}
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]
    log.info("%d cell lines, %d drugs, %d cell features; train=%d test=%d",
             n, len(drug_ids), len(cell_cols), len(tr), len(te))

    def ensemble_pred(train_idx, eval_idx, truth):
        """Train per-drug + multi-task on train_idx, return blended eval predictions."""
        pd_models = perdrug.fit_all(feats, truth, train_idx, cell_cols, tissues, drug_ids, id_to_name)
        pd_pred = perdrug.predict_pairs(pd_models, feats, truth, eval_idx, cell_cols, drug_ids, id_to_name)
        dfeat = mtl.build_drug_features(drug_ids, _obs_counts(targets_raw, drug_ids, train_idx))
        mt_model, cat = mtl.fit(feats, tissue_arr, truth, train_idx, cell_cols, dfeat, drug_ids)
        mt_pred = mtl.predict_pairs(mt_model, cat, feats, tissue_arr, truth, eval_idx,
                                    cell_cols, dfeat, drug_ids)
        return _blend(pd_pred, mt_pred)

    # honest held-out (random-line) evaluation of the ENSEMBLE
    truth_tr = _build_truth(targets_raw, drug_ids, tr)
    blended_te = ensemble_pred(tr, te, truth_tr)
    metrics = _metrics(blended_te, truth_tr, drug_ids, te)
    resid = _resid_std(blended_te, truth_tr, drug_ids, te)
    metrics["n_cell_lines"] = n
    metrics["n_drugs"] = len(drug_ids)
    metrics["n_features"] = len(cell_cols) + len(tissues)

    # tissue-blocked CV (harder honesty metric) — skip on tiny smoke runs
    if max_lines is None:
        trb, teb = _tissue_blocked_split(tissue_arr, seed=seed)
        truth_b = _build_truth(targets_raw, drug_ids, trb)
        blended_b = ensemble_pred(trb, teb, truth_b)
        m_blk = _metrics(blended_b, truth_b, drug_ids, teb)
        metrics["tissue_blocked"] = {k: m_blk[k] for k in
                                     ("mean_spearman", "mean_r2", "top10_accuracy",
                                      "best_drug_percentile", "n_test_lines")}

    # shipped ensemble: refit BOTH components on ALL lines for the best predictions
    all_idx = np.arange(n)
    truth_all = _build_truth(targets_raw, drug_ids, all_idx)
    pd_models = perdrug.fit_all(feats, truth_all, all_idx, cell_cols, tissues, drug_ids, id_to_name)
    drug_feat = mtl.build_drug_features(drug_ids, _obs_counts(targets_raw, drug_ids, all_idx))
    model, cat_levels = mtl.fit(feats, tissue_arr, truth_all, all_idx, cell_cols, drug_feat, drug_ids)

    return {
        "kind": "ensemble",
        "model": model,                       # multi-task component
        "per_drug_models": pd_models,         # per-drug component
        "blend_w_perdrug": BLEND_W_PERDRUG,
        "cell_cols": cell_cols,
        "drug_feat": drug_feat,
        "cat_levels": cat_levels,
        "drug_ids": drug_ids,
        "id_to_name": id_to_name,
        "name_to_id": {v: k for k, v in id_to_name.items()},
        "drug_meta": {DRUGS[d][0]: {"id": d, "target": THERAPY_CLASS[DRUGS[d][0]]} for d in drug_ids},
        "tissues": tissues,
        "resid_std": resid,
        "metrics": metrics,
        "version": 7,
        "data": "GDSC1 (release 17) + GDSC2 (25Feb20); ensemble of a per-drug model "
                "and a multi-task (cell line, drug) model, split and scaled by cell line.",
    }


PER_DRUG_METRIC_KEYS = ("per_drug_spearman", "per_drug_r2")


def summary_metrics(metrics: dict) -> dict:
    """Metrics minus the big per-drug dicts — safe for API responses."""
    return {k: v for k, v in metrics.items() if k not in PER_DRUG_METRIC_KEYS}


def main(seed: int = 0) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    os.makedirs(ARTIFACTS, exist_ok=True)
    log.info("Loading real GDSC data and training multi-task model ...")
    bundle = run_training(seed=seed)
    metrics = bundle["metrics"]
    log.info("metrics:\n%s", json.dumps(summary_metrics(metrics), indent=2))

    joblib.dump(bundle, os.path.join(ARTIFACTS, "model.joblib"), compress=3)
    with open(os.path.join(ARTIFACTS, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("Saved model + metrics to %s", ARTIFACTS)
    return metrics


if __name__ == "__main__":
    main()
