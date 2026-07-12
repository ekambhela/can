"""Tier 1: feature experiments (per-drug architecture, decisions on VAL).

  A curated (15)            — baseline
  B full (train-filtered)   — all 677 genomic features, drop <1% prevalence
  C curated + pathway flags
  D full + pathway flags
"""

from __future__ import annotations

import warnings

from scipy.stats import ConstantInputWarning

from experiments.harness import (
    build_truth, compute_metrics, fmt, load_universe, pathway_features,
    prevalence_filter, random_splits, run_perdrug,
)

warnings.simplefilter("ignore", ConstantInputWarning)


def main():
    U = load_universe()
    tr, va, te = random_splits(U["n"], seed=0)
    truth = build_truth(U["targets_raw"], U["drug_ids"], tr)
    va_ids = U["drug_ids"]

    def ev(cols, tag):
        m = compute_metrics(run_perdrug(U, tr, va, cols, truth=truth), truth, va_ids, va)
        print(fmt(tag, m))
        return m

    print(f"train={len(tr)} val={len(va)} test={len(te)}\n")

    # A curated
    ev(U["curated"], "A curated(15) [VAL]")

    # B full, prevalence-filtered on TRAIN only
    full_keep = prevalence_filter(U["feats"], U["full_bin"], tr, min_frac=0.01)
    print(f"   full features kept (>=1% train prevalence): {len(full_keep)} of {len(U['full_bin'])}")
    ev(full_keep, "B full(train>=1%) [VAL]")

    # add pathway flags to the frame
    U["feats"], pw_cols = pathway_features(U["feats"])
    print(f"   pathway flags added: {pw_cols}")

    # C curated + pathways
    ev(U["curated"] + pw_cols, "C curated+pathway [VAL]")

    # D full + pathways
    ev(full_keep + pw_cols, "D full+pathway [VAL]")


if __name__ == "__main__":
    main()
