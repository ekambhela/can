"""Expression-feature experiment.

Question: does adding gene-expression features (706 census genes, PCA-reduced)
raise accuracy — and does it help the hard tissue-blocked case where the current
model collapses?

Fair design: restrict to the 577 cell lines that HAVE expression, split those by
cell line, and compare curated-only vs curated+expression on the SAME lines.
Everything (target z-scoring, PCA) is fit on TRAIN only.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import ConstantInputWarning

from experiments.harness import (
    build_truth, compute_metrics, expression_pca, fmt, load_expression,
    load_universe, run_perdrug,
)
from experiments.multitask import build_drug_features, run_multitask

warnings.simplefilter("ignore", ConstantInputWarning)
MT = dict(max_leaf_nodes=127, learning_rate=0.05, max_iter=600, l2_regularization=1.0)


def split_covered(covered, seed=0, fracs=(0.64, 0.16, 0.20)):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(covered)
    a = int(fracs[0] * len(idx))
    b = int((fracs[0] + fracs[1]) * len(idx))
    return idx[:a], idx[a:b], idx[b:]


def blend(a, b, w=0.35):
    out = {}
    for d in set(a) | set(b):
        da, db = a.get(d, {}), b.get(d, {})
        keys = set(da) & set(db)
        out[d] = {i: w * da[i] + (1 - w) * db[i] for i in keys} if keys else (da or db)
    return out


def main():
    U = load_universe()
    expr, mask = load_expression(U["cosmic_arr"])
    covered = np.where(mask)[0]
    print(f"expression covers {len(covered)}/{U['n']} lines; {expr.shape[1]} genes\n")

    tr, va, te = split_covered(covered, seed=0)
    ids = U["drug_ids"]
    truth = build_truth(U["targets_raw"], ids, tr)

    # PCA(50) fit on train only, attached to the feature frame
    import pandas as pd
    pcs = expression_pca(expr, tr, k=50)
    U["feats"] = pd.concat([U["feats"], pcs, expr], axis=1)
    pc_cols = list(pcs.columns)
    gene_cols = list(expr.columns)
    cur = U["curated"]

    dfid = build_drug_features(ids, truth=truth, train_idx=tr, use_id=True)

    def pd_eval(cols, split):
        return compute_metrics(run_perdrug(U, tr, split, cols, truth=truth), truth, ids, split)

    def mt_eval(cols, split):
        pred, _ = run_multitask(U, tr, split, cols, truth=truth, drug_feat=dfid, **MT)
        return pred, compute_metrics(pred, truth, ids, split)

    print("--- per-drug (VAL) ---")
    print(fmt("A curated", pd_eval(cur, va)))
    print(fmt("B curated+exprPCA50", pd_eval(cur + pc_cols, va)))

    print("\n--- multi-task+drugID (VAL) ---")
    _, mc = mt_eval(cur, va); print(fmt("C curated", mc))
    _, md = mt_eval(cur + pc_cols, va); print(fmt("D curated+exprPCA50", md))
    _, me = mt_eval(cur + gene_cols, va); print(fmt("E curated+expr(raw706)", me))

    # choose expression variant by VAL spearman; ensemble on TEST
    use_raw = me["mean_spearman"] >= md["mean_spearman"]
    best_cols = cur + (gene_cols if use_raw else pc_cols)
    print(f"\nchosen MT expr variant: {'raw706' if use_raw else 'PCA50'}")

    print("\n--- TEST: baseline (curated) vs expression, per-drug / MT / ensemble ---")
    base_pd = run_perdrug(U, tr, te, cur, truth=truth)
    base_mt, _ = run_multitask(U, tr, te, cur, truth=truth, drug_feat=dfid, **MT)
    print(fmt("baseline ensemble", compute_metrics(blend(base_pd, base_mt), truth, ids, te)))

    exp_pd = run_perdrug(U, tr, te, cur + pc_cols, truth=truth)
    exp_mt, _ = run_multitask(U, tr, te, best_cols, truth=truth, drug_feat=dfid, **MT)
    print(fmt("expression ensemble", compute_metrics(blend(exp_pd, exp_mt), truth, ids, te)))

    # ---- the key question: tissue-blocked generalization ----
    print("\n--- TISSUE-BLOCKED (whole tissues held out, covered lines) ---")
    from experiments.harness import tissue_blocked_split
    tb_tr, tb_te = tissue_blocked_split(U["tissue_arr"], seed=0)
    tb_tr = np.array([i for i in tb_tr if mask[i]]); tb_te = np.array([i for i in tb_te if mask[i]])
    truth_b = build_truth(U["targets_raw"], ids, tb_tr)
    pcs_b = expression_pca(expr, tb_tr, k=50)
    U["feats"][pc_cols] = pcs_b.to_numpy()
    dfid_b = build_drug_features(ids, truth=truth_b, train_idx=tb_tr, use_id=True)

    bpd = run_perdrug(U, tb_tr, tb_te, cur, truth=truth_b)
    bmt, _ = run_multitask(U, tb_tr, tb_te, cur, truth=truth_b, drug_feat=dfid_b, **MT)
    print(fmt("baseline ensemble [TB]", compute_metrics(blend(bpd, bmt), truth_b, ids, tb_te)))
    epd = run_perdrug(U, tb_tr, tb_te, cur + pc_cols, truth=truth_b)
    emt, _ = run_multitask(U, tb_tr, tb_te, cur + gene_cols, truth=truth_b, drug_feat=dfid_b, **MT)
    print(fmt("expression ensemble [TB]", compute_metrics(blend(epd, emt), truth_b, ids, tb_te)))


if __name__ == "__main__":
    main()
