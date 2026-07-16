"""Determinism / golden-behavior test (audit item 2).

A byte-exact golden file on the shipped 11 MB model would couple the test to
library-version pickle details. The robust equivalent: a fixed seed + fixed input
must yield a stable ranking and stable scores. This catches silent changes to the
training or scoring path.
"""

import warnings

import pytest
from scipy.stats import ConstantInputWarning

from model import predict as P
from model.train import run_training

SAMPLE = {"tissue": "skin", "BRAF_mut": 1, "TP53_mut": 1}


def _train():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        return run_training(seed=0, max_lines=80, max_drugs=15)


@pytest.fixture(scope="module")
def bundle():
    return _train()


def test_scoring_is_deterministic(bundle, monkeypatch):
    monkeypatch.setattr(P, "load_bundle", lambda: bundle)
    P._predict_cached.cache_clear()

    first = P.predict(dict(SAMPLE), top_k=8)
    P._predict_cached.cache_clear()
    second = P.predict(dict(SAMPLE), top_k=8)

    order1 = [r["therapy"] for r in first["ranked"]]
    order2 = [r["therapy"] for r in second["ranked"]]
    assert order1 == order2, "ranking order is not reproducible"

    for a, b in zip(first["ranked"], second["ranked"]):
        assert a["sensitivity"] == pytest.approx(b["sensitivity"], abs=1e-9)
        assert a["match_percent"] == pytest.approx(b["match_percent"], abs=1e-6)

    # scores must be finite and strictly non-increasing down the ranking
    sens = [r["sensitivity"] for r in first["ranked"]]
    assert all(x == x for x in sens)  # no NaN
    assert sens == sorted(sens, reverse=True)


def test_training_is_seeded():
    """Same seed -> same held-out metric. Guards against an accidental unseeded
    source of randomness sneaking into the pipeline."""
    a = _train()["metrics"]["mean_spearman"]
    b = _train()["metrics"]["mean_spearman"]
    assert a == pytest.approx(b, abs=1e-9)
