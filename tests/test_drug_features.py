"""Unit tests for the Morgan-fingerprint scaffolding (model/drug_features.py).

Skipped automatically when RDKit is not installed (it is an optional dep).
"""

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit", reason="RDKit is an optional dependency")

from model.drug_features import morgan_fingerprint, morgan_matrix  # noqa: E402

# A couple of real, small drug-like SMILES.
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


def test_fingerprint_shape_and_binary():
    fp = morgan_fingerprint(ASPIRIN, n_bits=1024)
    assert fp.shape == (1024,)
    assert fp.dtype == np.float32
    assert set(np.unique(fp)).issubset({0.0, 1.0})
    assert fp.sum() > 0  # a real molecule sets some bits


def test_fingerprint_is_deterministic():
    assert np.array_equal(morgan_fingerprint(ASPIRIN), morgan_fingerprint(ASPIRIN))


def test_different_molecules_differ():
    assert not np.array_equal(morgan_fingerprint(ASPIRIN), morgan_fingerprint(CAFFEINE))


def test_invalid_smiles_raises():
    with pytest.raises(ValueError):
        morgan_fingerprint("not_a_molecule(((")


def test_matrix_skips_bad_and_keeps_good():
    ids, X, skipped = morgan_matrix({1: ASPIRIN, 2: "???", 3: CAFFEINE}, n_bits=512)
    assert ids == [1, 3]
    assert X.shape == (2, 512)
    assert skipped == [2]
