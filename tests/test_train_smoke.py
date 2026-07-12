"""End-to-end smoke test: train on a tiny subset, then rank drugs through it.

Keeps the run small (a handful of drugs on ~80 lines) so CI stays fast while
still exercising load_frame -> train -> predict wired together.
"""

import warnings

import pytest
from scipy.stats import ConstantInputWarning

from model import predict as P
from model.train import run_training


@pytest.fixture(scope="module")
def bundle():
    # constant-target drugs on a tiny subset are expected; don't fail on them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        return run_training(seed=0, max_lines=80, max_drugs=15)


def test_bundle_has_expected_keys(bundle):
    for key in ["models", "tissues", "binary_features", "drug_meta",
                "therapy_class", "resid_std", "metrics", "version", "data"]:
        assert key in bundle, key
    assert len(bundle["models"]) >= 3
    for key in ["mean_r2", "mean_spearman", "top10_accuracy",
                "best_drug_percentile", "n_drugs", "n_test_lines"]:
        assert key in bundle["metrics"], key


def test_predict_returns_ranked_results(bundle, monkeypatch):
    P.load_bundle.cache_clear()
    monkeypatch.setattr(P, "load_bundle", lambda: bundle)

    sample, _warnings = P.sample_from_dict({"tissue": bundle["tissues"][0], "TP53_mut": 1})
    result = P.predict(sample, top_k=3)

    assert result["recommendation"] in bundle["models"]
    assert len(result["ranked"]) == 3
    assert result["ranked"][0]["rank"] == 1
    # sensitivities are sorted descending
    sens = [r["sensitivity"] for r in result["ranked"]]
    assert sens == sorted(sens, reverse=True)
    # API payload is slim — no 369-entry per-drug dicts embedded
    assert "per_drug_r2" not in result["model_metrics"]
    assert "mean_r2" in result["model_metrics"]
    # monkeypatch restores the real (lru_cached) load_bundle after the test.
