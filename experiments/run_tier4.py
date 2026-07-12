"""Tier 4 + honest evaluation: tune the multi-task winner on VAL, confirm on
TEST, then stress-test with tissue-blocked CV and report per-drug predictability.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr

from model.gdsc import DRUGS
from experiments.harness import (
    build_truth, compute_metrics, fmt, load_universe, random_splits, run_perdrug,
    tissue_blocked_split,
)
from experiments.multitask import build_drug_features, run_multitask

warnings.simplefilter("ignore", ConstantInputWarning)

GRID = [
    dict(max_leaf_nodes=63, learning_rate=0.06, max_iter=400, l2_regularization=1.0),
    dict(max_leaf_nodes=127, learning_rate=0.05, max_iter=600, l2_regularization=1.0),
    dict(max_leaf_nodes=127, learning_rate=0.03, max_iter=900, l2_regularization=1.0),
    dict(max_leaf_nodes=255, learning_rate=0.05, max_iter=600, l2_regularization=1.0),
    dict(max_leaf_nodes=127, learning_rate=0.05, max_iter=700, l2_regularization=2.0,
         min_samples_leaf=40),
]


def per_drug_spearman(pred, truth, ids, eval_idx):
    out = {}
    for d in ids:
        pr = pred.get(d)
        if not pr:
            continue
        xs, ys = [], []
        for i in eval_idx:
            if i in pr and not np.isnan(truth[d][i]):
                xs.append(pr[i]); ys.append(truth[d][i])
        if len(xs) >= 8 and np.std(xs) > 1e-9 and np.std(ys) > 1e-9:
            r = spearmanr(ys, xs).correlation
            out[d] = 0.0 if np.isnan(r) else float(r)
    return out


def main():
    U = load_universe()
    tr, va, te = random_splits(U["n"], seed=0)
    truth = build_truth(U["targets_raw"], U["drug_ids"], tr)
    ids = U["drug_ids"]
    df_id = build_drug_features(ids, truth=truth, train_idx=tr, use_id=True)

    print(f"train={len(tr)} val={len(va)} test={len(te)}\n--- Tier 4 grid (VAL) ---")
    best, best_rho = None, -1
    for g in GRID:
        pred, _ = run_multitask(U, tr, va, U["curated"], truth=truth, drug_feat=df_id, **g)
        m = compute_metrics(pred, truth, ids, va)
        tag = f"leaf{g['max_leaf_nodes']} lr{g['learning_rate']} it{g['max_iter']} l2{g['l2_regularization']}"
        print(fmt(tag, m))
        if m["mean_spearman"] > best_rho:
            best, best_rho = g, m["mean_spearman"]
    print(f"\nbest on VAL: {best}\n")

    print("--- winner vs baseline (TEST, random-line split) ---")
    print(fmt("baseline per-drug [TEST]",
              compute_metrics(run_perdrug(U, tr, te, U["curated"], truth=truth), truth, ids, te)))
    pred_te, _ = run_multitask(U, tr, te, U["curated"], truth=truth, drug_feat=df_id, **best)
    print(fmt("WINNER mt+drugID [TEST]", compute_metrics(pred_te, truth, ids, te)))

    print("\n--- tissue-blocked CV (hold out WHOLE tissues) ---")
    trb, teb = tissue_blocked_split(U["tissue_arr"], seed=0, test_frac=0.25)
    truth_b = build_truth(U["targets_raw"], ids, trb)
    df_id_b = build_drug_features(ids, truth=truth_b, train_idx=trb, use_id=True)
    print(f"tissue-blocked: train={len(trb)} test={len(teb)} lines")
    print(fmt("baseline per-drug [TISSUE-BLK]",
              compute_metrics(run_perdrug(U, trb, teb, U["curated"], truth=truth_b), truth_b, ids, teb)))
    predb, _ = run_multitask(U, trb, teb, U["curated"], truth=truth_b, drug_feat=df_id_b, **best)
    print(fmt("WINNER mt+drugID [TISSUE-BLK]", compute_metrics(predb, truth_b, ids, teb)))

    print("\n--- per-drug predictability (winner, TEST) ---")
    pds = per_drug_spearman(pred_te, truth, ids, te)
    ordered = sorted(pds.items(), key=lambda kv: kv[1], reverse=True)
    print(f"drugs scored: {len(pds)};  well-predicted (rho>0.3): "
          f"{sum(v>0.3 for v in pds.values())};  near-random (rho<0.1): "
          f"{sum(v<0.1 for v in pds.values())}")
    print("TOP 12 best-predicted drugs:")
    for d, r in ordered[:12]:
        print(f"   {r:+.3f}  {DRUGS[d][0]:<22} [{DRUGS[d][1]}]")
    print("BOTTOM 6 (near-random):")
    for d, r in ordered[-6:]:
        print(f"   {r:+.3f}  {DRUGS[d][0]:<22} [{DRUGS[d][1]}]")


if __name__ == "__main__":
    main()
