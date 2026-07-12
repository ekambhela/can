"""Tier 2: single multi-task model over (cell_line, drug) pairs.

Long format: one row per observed pair, features = [cell genomics] + [drug
identity/pathway], target = train-z-scored sensitivity. A single
HistGradientBoostingRegressor learns cross-drug structure, so a drug screened on
few lines borrows the pattern from similar drugs. Split is still by cell line.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from model.gdsc import DRUGS
from experiments.harness import build_truth


def _target_tokens(targets: str):
    """Split a drug 'Targets' string into normalized tokens (e.g. 'EGFR, HER2')."""
    if not targets or str(targets).lower() == "nan":
        return []
    return [t.strip().upper() for t in re.split(r"[,;/]| and ", str(targets)) if t.strip()]


def build_drug_features(drug_ids, truth=None, train_idx=None, min_drugs=3, use_id=False):
    """Drug representation: target-pathway (categorical) + multi-hot targets, and
    optionally a capped drug-IDENTITY categorical (254 most-screened drugs keep
    their id, the rest fall into 'OTHER') so common drugs get per-drug fidelity
    while staying under HistGBM's 255-category cap.
    """
    from collections import Counter
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

    id_bucket = None
    if use_id and truth is not None and train_idx is not None:
        counts = {d: int(np.sum(~np.isnan(truth[d][train_idx]))) for d in drug_ids}
        top = sorted(drug_ids, key=lambda d: counts[d], reverse=True)[:254]
        keep = set(top)
        id_bucket = {d: (str(d) if d in keep else "OTHER") for d in drug_ids}
    return {"pathway": pathway, "target_cols": list(tcol.values()),
            "multihot": multihot, "id_bucket": id_bucket}


def build_long(U, truth, line_idx, cell_cols, drug_feat):
    """Rows for all observed (line, drug) pairs with line in `line_idx`."""
    pathway = drug_feat["pathway"]
    target_cols = drug_feat["target_cols"]
    multihot = drug_feat["multihot"]
    id_bucket = drug_feat["id_bucket"]
    line_set = list(line_idx)
    cell = U["feats"][cell_cols].to_numpy(dtype=np.float32)
    tissue = U["tissue_arr"]
    rows_cell, tiss, dpath, dtgt, y, ln, did, dbk = [], [], [], [], [], [], [], []
    for d in U["drug_ids"]:
        t = truth[d]
        mh = [multihot[d].get(c, 0.0) for c in target_cols]
        for i in line_set:
            if not np.isnan(t[i]):
                rows_cell.append(cell[i]); tiss.append(tissue[i])
                dpath.append(pathway[d]); dtgt.append(mh)
                y.append(t[i]); ln.append(i); did.append(d)
                if id_bucket is not None:
                    dbk.append(id_bucket[d])
    X = pd.DataFrame(np.asarray(rows_cell, dtype=np.float32), columns=cell_cols)
    if target_cols:
        X = pd.concat([X, pd.DataFrame(np.asarray(dtgt, dtype=np.float32), columns=target_cols)], axis=1)
    X["tissue"] = pd.Series(tiss, dtype="category")
    X["drug_pathway"] = pd.Series(dpath, dtype="category")
    if id_bucket is not None:
        X["drug_id"] = pd.Series(dbk, dtype="category")
    return X, np.asarray(y, dtype=np.float32), np.asarray(ln), np.asarray(did)


def run_multitask(U, train_idx, eval_idx, cell_cols, truth=None, drug_feat=None, **gbm):
    if truth is None:
        truth = build_truth(U["targets_raw"], U["drug_ids"], train_idx)
    if drug_feat is None:
        drug_feat = build_drug_features(U["drug_ids"])
    Xtr, ytr, _, _ = build_long(U, truth, train_idx, cell_cols, drug_feat)
    Xev, yev, ln_ev, did_ev = build_long(U, truth, eval_idx, cell_cols, drug_feat)

    # align eval categorical levels to train's (unseen -> NaN, handled natively)
    for c in ["tissue", "drug_pathway", "drug_id"]:
        if c in Xtr.columns:
            Xev[c] = pd.Categorical(Xev[c], categories=Xtr[c].cat.categories)

    params = dict(max_iter=400, learning_rate=0.06, l2_regularization=1.0,
                  max_leaf_nodes=63, early_stopping=True, validation_fraction=0.1,
                  random_state=0, categorical_features="from_dtype")
    params.update(gbm)
    model = HistGradientBoostingRegressor(**params)
    model.fit(Xtr, ytr)
    p = model.predict(Xev)

    pred = {}
    for d, i, pp in zip(did_ev, ln_ev, p):
        pred.setdefault(int(d), {})[int(i)] = float(pp)
    return pred, model
