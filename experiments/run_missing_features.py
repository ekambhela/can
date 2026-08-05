"""Does passing NaN for UNSPECIFIED biomarkers beat passing 0.0?

The app fills every biomarker the user didn't mention with 0.0 — i.e. it tells
the model "wild-type", not "unknown". HistGradientBoosting handles missing
values natively, so NaN is available as an alternative encoding.

Two questions, in order:
  1. Does it change the ranking?  (part A — scores the shipped bundle both ways)
  2. Is it more ACCURATE?         (part B — held-out lines with biomarkers
                                   genuinely masked, scored against real IC50s)

Part B is the one that decides it. Part A only establishes that the choice
matters.

Run:  python -m experiments.run_missing_features
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from experiments.harness import (
    build_truth,
    load_universe,
    make_pipeline,
    random_splits,
)
from model.predict import _score, get_bundle
from model.schema import BINARY_FEATURES

# Realistic partial inputs: what a visitor actually types in. Each is the set of
# biomarkers the user DID specify; everything else is unspecified.
SCENARIOS = [
    ("tissue only", {}),
    ("BRAF+", {"BRAF_mut": 1.0}),
    ("EGFR+", {"EGFR_mut": 1.0}),
    ("ERBB2 amp", {"ERBB2_amp": 1.0}),
    ("KRAS+ TP53+", {"KRAS_mut": 1.0, "TP53_mut": 1.0}),
    ("MSI-high", {"MSI": 1.0}),
]


def _fill(tissue, specified, value):
    """A sample with `specified` set and every other biomarker set to `value`."""
    s = {"tissue": tissue}
    for f in BINARY_FEATURES:
        s[f] = float(specified.get(f, value))
    return s


def accuracy_under_masking(mask_frac=0.6, seed=0):
    """Part B: which encoding survives genuinely-unknown biomarkers better?

    Train exactly as production does (0-filled, no missing anywhere), then hide
    a random `mask_frac` of each held-out line's biomarkers and predict them
    both ways. Truth is the real held-out sensitivity, so this measures which
    encoding actually degrades less — not merely which one differs.
    """
    U = load_universe()
    feats, curated = U["feats"], U["curated"]
    tr, _val, te = random_splits(U["n"], seed=seed)
    tr = np.concatenate([tr, _val])
    truth = build_truth(U["targets_raw"], U["drug_ids"], tr)

    X = feats[curated + ["tissue"]]
    rng = np.random.default_rng(seed)
    # One mask per held-out line: which biomarkers the "user" didn't specify.
    hidden = rng.random((len(feats), len(curated))) < mask_frac

    X_zero, X_nan = X.copy(), X.copy()
    for j, c in enumerate(curated):
        col = X[c].to_numpy(dtype=float).copy()
        z, nn = col.copy(), col.copy()
        z[hidden[:, j]] = 0.0
        nn[hidden[:, j]] = np.nan
        X_zero[c], X_nan[c] = z, nn

    rho_zero, rho_nan, rho_full = [], [], []
    for d in U["drug_ids"]:
        y = truth[d]
        obs = ~np.isnan(y)
        tr_d = np.array([i for i in tr if obs[i]])
        te_d = np.array([i for i in te if obs[i]])
        if tr_d.size < 30 or te_d.size < 10:
            continue
        pipe = make_pipeline(U["tissue_vals"], curated)
        pipe.fit(X.iloc[tr_d], y[tr_d])          # train as production does

        for arm, Xe in ((rho_full, X), (rho_zero, X_zero), (rho_nan, X_nan)):
            p = pipe.predict(Xe.iloc[te_d])
            if np.std(p) > 1e-9:
                r = spearmanr(y[te_d], p).correlation
                arm.append(0.0 if np.isnan(r) else float(r))

    return (float(np.mean(rho_full)), float(np.mean(rho_zero)),
            float(np.mean(rho_nan)), len(rho_zero))


def main() -> None:
    bundle = get_bundle()
    tissues = bundle["tissues"]

    top1_same = top1_tot = 0
    overlaps, rhos, shifts = [], [], []
    flips = []

    for tissue in tissues:
        for label, spec in SCENARIOS:
            zeros = _score(bundle, _fill(tissue, spec, 0.0))
            nans = _score(bundle, _fill(tissue, spec, np.nan))

            names = sorted(zeros)
            zv = np.array([zeros[n] for n in names])
            nv = np.array([nans[n] for n in names])

            oz = sorted(names, key=lambda n: zeros[n], reverse=True)
            on = sorted(names, key=lambda n: nans[n], reverse=True)

            top1_tot += 1
            if oz[0] == on[0]:
                top1_same += 1
            else:
                flips.append((tissue, label, oz[0], on[0]))

            overlaps.append(len(set(oz[:10]) & set(on[:10])) / 10.0)
            rhos.append(spearmanr(zv, nv).correlation)
            shifts.append(float(np.mean(np.abs(zv - nv))))

    print(f"scenarios evaluated: {top1_tot}  ({len(tissues)} tissues x {len(SCENARIOS)} inputs)")
    print(f"top-1 unchanged    : {top1_same}/{top1_tot} = {top1_same / top1_tot:.1%}")
    print(f"top-10 overlap     : {np.mean(overlaps):.3f} (mean fraction shared)")
    print(f"rank correlation   : {np.mean(rhos):.4f} (mean Spearman, 0.0-fill vs NaN-fill)")
    print(f"score shift        : {np.mean(shifts):.4f} mean |delta| in sensitivity z units")

    if flips:
        print(f"\ntop-1 flips ({len(flips)}):")
        for t, lab, a, b in flips[:25]:
            print(f"  {t:26s} {lab:12s}  {a}  ->  {b}")
    else:
        print("\nno top-1 flips")

    print("\n--- part B: accuracy on held-out lines with 60% of biomarkers masked ---")
    full, zero, nan, nd = accuracy_under_masking()
    print(f"drugs evaluated       : {nd}")
    print(f"mean Spearman, nothing masked (ceiling) : {full:.4f}")
    print(f"mean Spearman, masked -> 0.0            : {zero:.4f}")
    print(f"mean Spearman, masked -> NaN            : {nan:.4f}")
    print(f"NaN - 0.0                               : {nan - zero:+.4f}")


if __name__ == "__main__":
    main()
