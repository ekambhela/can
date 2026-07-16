"""Chemical drug descriptors (Morgan / ECFP fingerprints) — scaffolding.

The shipped model represents a drug by its target pathway + target multi-hot + a
capped identity category (see model/mtl.py). A natural richer representation is a
**structural fingerprint** of the compound, which lets the model relate drugs by
chemistry rather than only by curated target annotations.

This module is the pure, testable building block for that: SMILES -> Morgan
fingerprint bit vector. It is deliberately **not wired into training yet**:

  * GDSC compound -> SMILES is not shipped in data/gdsc/ (it needs a PubChem or
    ChEMBL lookup), and
  * per the audit, no accuracy gain is claimed for a feature that has not been
    benchmarked on a leakage-free split.

To integrate later: build {drug_id: SMILES}, precompute `morgan_fingerprint` per
drug, concatenate the bits onto the drug block in mtl._build_long / score_sample,
and re-run experiments/ on the by-cell-line split before reporting any change.

RDKit is an optional dependency (not in requirements.txt); import is lazy so the
app and the rest of the model package do not depend on it.
"""

from __future__ import annotations

import numpy as np


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:  # pragma: no cover - exercised only without rdkit
        raise ImportError(
            "RDKit is required for chemical fingerprints. Install with "
            "`pip install rdkit` (it is intentionally not in requirements.txt "
            "because the shipped model does not use it)."
        ) from exc
    return Chem, rdFingerprintGenerator


def morgan_fingerprint(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Return the Morgan (ECFP-like) fingerprint of a SMILES string as a float32
    bit vector of length `n_bits`. Raises ValueError on an unparseable SMILES.

    radius=2 corresponds to ECFP4. The result is deterministic and depends only on
    the molecule + parameters, so it is safe to precompute and cache per drug.
    """
    Chem, rdFingerprintGenerator = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles!r}")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(n_bits, dtype=np.float32)
    for bit in fp.GetOnBits():
        arr[bit] = 1.0
    return arr


def morgan_matrix(smiles_by_id: dict, n_bits: int = 2048, radius: int = 2):
    """Fingerprint several compounds at once.

    Returns (ids, X) where X has one row per id (in the returned order), suitable
    for concatenating onto the drug feature block. Unparseable SMILES are skipped
    and reported in the returned `skipped` list.
    """
    ids, rows, skipped = [], [], []
    for did, smiles in smiles_by_id.items():
        try:
            rows.append(morgan_fingerprint(smiles, n_bits=n_bits, radius=radius))
            ids.append(did)
        except ValueError:
            skipped.append(did)
    X = np.vstack(rows) if rows else np.empty((0, n_bits), dtype=np.float32)
    return ids, X, skipped
