"""The separation/match numbers must not read as probabilities of being right.

_separation_score measures how far the top drug sits above ranks 2-6 in
residual-std units. Held-out top-1 accuracy is ~11%, so surfacing that number as
"confidence" next to a drug name invited reading 0.84 as "84% chance this is
right". These pin the renamed contract; the computation is unchanged.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
from model import predict as P


@pytest.fixture(scope="module")
def client():
    return TestClient(app_module.app)


@pytest.fixture(scope="module")
def result(client):
    resp = client.post("/api/predict_form", json={"tissue": "skin", "BRAF_mut": 1})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_separation_score_is_exposed(result):
    assert "separation_score" in result
    assert 0.0 <= result["separation_score"] <= 1.0


def test_confidence_kept_as_deprecated_alias(result):
    """Old clients keep working for one release — same value, new name added."""
    assert "confidence" in result
    assert result["confidence"] == result["separation_score"]


def test_batch_rows_carry_both_names(client):
    resp = client.post(
        "/api/predict_batch",
        files={"file": ("c.csv", b"tissue,BRAF_mut\nskin,1\nbreast,0\n", "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    for row in resp.json()["rows"]:
        assert row["separation_score"] == row["confidence"]


def test_accuracy_is_available_next_to_the_callout(result):
    """The UI shows top-1/top-10 beside the single-drug recommendation, so the
    payload has to carry them."""
    m = result["model_metrics"]
    assert "top1_accuracy" in m
    assert "top10_accuracy" in m
    # sanity: this really is the low number that motivates the relabel
    assert 0.0 <= m["top1_accuracy"] < 0.5


# --- the computation itself is unchanged -------------------------------------
def test_separation_is_monotonic_in_the_gap():
    """Bigger gap between #1 and its closest rivals -> higher score."""
    rivals = [1.0, 0.9, 0.8, 0.7, 0.6]
    scores = [P._separation_score(np.array([top] + rivals)) for top in (1.05, 1.5, 3.0)]
    assert scores == sorted(scores), scores
    assert scores[0] < scores[-1]


def test_separation_is_bounded():
    assert P._separation_score(np.array([50.0, 0.0, 0.0, 0.0])) <= 0.99
    assert P._separation_score(np.array([0.0, 50.0, 50.0, 50.0])) >= 0.01


def test_near_ties_land_near_one_half():
    """A cluster of indistinguishable drugs should read as 'we can't tell'."""
    s = P._separation_score(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    assert 0.45 <= s <= 0.55, s


def test_single_drug_is_undefined_not_certain():
    assert P._separation_score(np.array([2.0])) == 0.5


def test_pct_is_bounded_percentile():
    assert P._pct(-50) == 0.0
    assert P._pct(50) == 100.0
    assert 49.0 < P._pct(0.0) < 51.0
    for z in (-3.0, -0.5, 0.0, 0.5, 3.0):
        assert 0.0 <= P._pct(z) <= 100.0
