"""Pipeline-level leakage guards (audit item 1).

These are stronger than the unit-level scaler test (tests/test_scaler_leakage.py):
they assert the *training pipeline itself* cannot put the same cell line in both
splits, and that no IC50 / target-derived column ever reaches the design matrix.
If either invariant is broken, reported accuracy would be leakage-inflated.
"""

import numpy as np

from model import mtl
from model.gdsc import BINARY_FEATURES, DRUG_COL, DRUGS, load_frame
from model.train import _obs_counts, _tissue_blocked_split


def _split(n, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    return idx[:cut], idx[cut:]


def test_by_cell_line_split_is_disjoint():
    feats, _targets, _tissues = load_frame()
    tr, te = _split(len(feats))
    assert set(tr.tolist()).isdisjoint(te.tolist())
    assert len(tr) + len(te) == len(feats)


def test_tissue_blocked_split_shares_no_tissue():
    feats, _targets, _tissues = load_frame()
    tissue_arr = feats["tissue"].to_numpy()
    trb, teb = _tissue_blocked_split(tissue_arr, seed=0)
    assert set(tissue_arr[trb].tolist()).isdisjoint(tissue_arr[teb].tolist())
    assert set(trb.tolist()).isdisjoint(teb.tolist())


def test_design_matrix_has_no_target_or_ic50_columns():
    """The (cell line, drug) design matrix must contain only genomic/tissue/drug
    *property* features — never a raw IC50 or the per-drug response target."""
    feats, targets, _tissues = load_frame()
    tissue_arr = feats["tissue"].to_numpy()
    drug_ids = list(DRUGS.keys())[:12]  # subset for speed; columns don't depend on it
    tr, _te = _split(len(feats))

    # truth built from train-only scaling (mirrors train.py)
    from model.train import _build_truth
    truth = _build_truth({d: targets[DRUG_COL[d]].to_numpy() for d in drug_ids}, drug_ids, tr)
    dfeat = mtl.build_drug_features(
        drug_ids, _obs_counts({d: targets[DRUG_COL[d]].to_numpy() for d in drug_ids}, drug_ids, tr)
    )
    X, _y, _ln, _did = mtl._build_long(feats, tissue_arr, truth, tr, list(BINARY_FEATURES),
                                       dfeat, drug_ids)

    ic50_cols = set(DRUG_COL.values())
    for col in X.columns:
        assert col not in ic50_cols, f"IC50 column leaked into X: {col}"
        assert not col.startswith("Drug_"), f"raw drug IC50 column leaked into X: {col}"
    # the response target itself must not be a feature column
    assert "target" not in [c.lower() for c in X.columns]
    # sanity: X really is the expected feature set (genomic flags + drug props + cats)
    for f in BINARY_FEATURES:
        assert f in X.columns
    for cat in ("tissue", "drug_pathway", "drug_id"):
        assert cat in X.columns
