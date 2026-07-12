"""Baseline: current production architecture (curated 15 binary features,
one HistGBM per drug) under the harness. Locks the numbers everything else
is compared against."""

from __future__ import annotations

import warnings

from scipy.stats import ConstantInputWarning

from experiments.harness import (
    build_truth, compute_metrics, fmt, load_universe, random_splits, run_perdrug,
)

warnings.simplefilter("ignore", ConstantInputWarning)


def main():
    U = load_universe()
    print(f"universe: {U['n']} cell lines, {len(U['drug_ids'])} drugs, "
          f"{len(U['full_bin'])} binary genomic features (curated uses {len(U['curated'])})")
    tr, va, te = random_splits(U["n"], seed=0)
    print(f"split by cell line: train={len(tr)} val={len(va)} test={len(te)}\n")

    truth = build_truth(U["targets_raw"], U["drug_ids"], tr)

    pred_va = run_perdrug(U, tr, va, U["curated"], truth=truth)
    print(fmt("BASELINE curated [VAL]", compute_metrics(pred_va, truth, U["drug_ids"], va)))

    pred_te = run_perdrug(U, tr, te, U["curated"], truth=truth)
    print(fmt("BASELINE curated [TEST]", compute_metrics(pred_te, truth, U["drug_ids"], te)))


if __name__ == "__main__":
    main()
