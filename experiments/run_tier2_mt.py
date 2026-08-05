"""Tier 2 head-to-head: multi-task vs per-drug baseline (decisions on VAL)."""

from __future__ import annotations

import time
import warnings

from scipy.stats import ConstantInputWarning

from experiments.harness import (
    build_truth,
    compute_metrics,
    fmt,
    load_universe,
    prevalence_filter,
    random_splits,
    run_perdrug,
)
from experiments.multitask import build_drug_features, run_multitask

warnings.simplefilter("ignore", ConstantInputWarning)


def main():
    U = load_universe()
    tr, va, te = random_splits(U["n"], seed=0)
    truth = build_truth(U["targets_raw"], U["drug_ids"], tr)
    ids = U["drug_ids"]
    print(f"train={len(tr)} val={len(va)} test={len(te)}\n")

    # baseline (per-drug curated) for reference
    t0 = time.time()
    m = compute_metrics(run_perdrug(U, tr, va, U["curated"], truth=truth), truth, ids, va)
    print(fmt("A per-drug curated [VAL]", m), f"  ({time.time()-t0:.0f}s)")

    # multi-task, curated cell features
    t0 = time.time()
    pred, _ = run_multitask(U, tr, va, U["curated"], truth=truth)
    print(fmt("E multitask curated [VAL]", compute_metrics(pred, truth, ids, va)),
          f"  ({time.time()-t0:.0f}s)")

    # multi-task, full (train-filtered) cell features
    full_keep = prevalence_filter(U["feats"], U["full_bin"], tr, min_frac=0.01)
    t0 = time.time()
    pred, _ = run_multitask(U, tr, va, full_keep, truth=truth)
    print(fmt("F multitask full [VAL]", compute_metrics(pred, truth, ids, va)),
          f"  ({time.time()-t0:.0f}s)  ({len(full_keep)} cell feats)")

    # multi-task + capped drug identity (fidelity for common drugs + sharing)
    df_id = build_drug_features(ids, truth=truth, train_idx=tr, use_id=True)
    t0 = time.time()
    pred, _ = run_multitask(U, tr, va, U["curated"], truth=truth, drug_feat=df_id)
    print(fmt("G multitask curated+drugID [VAL]", compute_metrics(pred, truth, ids, va)),
          f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
