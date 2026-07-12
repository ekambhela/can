"""Tier 2b: lock in the multi-task + drug-identity design; confirm on TEST,
try full cell features, and a small capacity bump. Decisions on VAL."""

from __future__ import annotations

import time
import warnings

from scipy.stats import ConstantInputWarning

from experiments.harness import (
    build_truth, compute_metrics, fmt, load_universe, prevalence_filter,
    random_splits, run_perdrug,
)
from experiments.multitask import build_drug_features, run_multitask

warnings.simplefilter("ignore", ConstantInputWarning)


def main():
    U = load_universe()
    tr, va, te = random_splits(U["n"], seed=0)
    truth = build_truth(U["targets_raw"], U["drug_ids"], tr)
    ids = U["drug_ids"]
    full_keep = prevalence_filter(U["feats"], U["full_bin"], tr, min_frac=0.01)
    df_id = build_drug_features(ids, truth=truth, train_idx=tr, use_id=True)

    def mt(cols, split, tag, **gbm):
        t0 = time.time()
        pred, _ = run_multitask(U, tr, split, cols, truth=truth, drug_feat=df_id, **gbm)
        print(fmt(tag, compute_metrics(pred, truth, ids, split)), f"  ({time.time()-t0:.0f}s)")

    print(f"train={len(tr)} val={len(va)} test={len(te)}\n")
    # reference baseline on TEST
    print(fmt("A per-drug curated [TEST]",
              compute_metrics(run_perdrug(U, tr, te, U["curated"], truth=truth), truth, ids, te)))
    print()
    mt(U["curated"], va, "G curated+drugID [VAL]")
    mt(full_keep, va, "H full+drugID [VAL]")
    mt(U["curated"], va, "I curated+drugID +cap [VAL]", max_leaf_nodes=127, max_iter=600, learning_rate=0.05)
    print()
    mt(U["curated"], te, "G curated+drugID [TEST]")


if __name__ == "__main__":
    main()
