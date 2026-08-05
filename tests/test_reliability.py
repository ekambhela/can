"""Drugs the model predicts badly must be marked, not presented as equals.

On the held-out split, per-drug R^2 spans -0.274 to 0.495 (median 0.188) and 29
of 369 drugs are BELOW ZERO — the model predicts them worse than always guessing
that drug's mean. They could reach rank 1 with exactly the same visual treatment
as a well-predicted drug.

Default is to flag, not exclude: a shortlist that silently drops candidates is
its own kind of dishonesty. Exclusion is opt-in.
"""

import json
import os

import pytest
from fastapi.testclient import TestClient

import app as app_module
from model import predict as P
from model.schema import build_reliability, reliability_tier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Worst held-out R^2 in artifacts/metrics.json (-0.274), i.e. the drug this test
# most needs to see flagged.
KNOWN_BAD_DRUG = "Ribociclib"


@pytest.fixture(scope="module")
def client():
    return TestClient(app_module.app)


# --- tiering ------------------------------------------------------------------
@pytest.mark.parametrize("rho,r2,expected", [
    (0.50, 0.30, "high"),
    (0.40, 0.20, "high"),
    (0.35, 0.10, "moderate"),    # weak R^2
    (0.25, 0.30, "moderate"),    # weak ranking
    (0.40, -0.01, "low"),        # R^2 below zero is the cliff
    (0.03, 0.20, "low"),         # no usable ranking signal
    (None, 0.30, "unknown"),
    (0.40, None, "unknown"),
])
def test_reliability_tiers(rho, r2, expected):
    assert reliability_tier(rho, r2) == expected


def test_known_negative_r2_drug_is_tiered_low():
    metrics = json.load(open(os.path.join(ROOT, "artifacts", "metrics.json")))
    rel = build_reliability(metrics)

    assert rel[KNOWN_BAD_DRUG]["r2"] < 0, "fixture drug should have negative R^2"
    assert rel[KNOWN_BAD_DRUG]["tier"] == "low"

    # every negative-R^2 drug must be caught, not just this one
    negatives = [n for n, v in rel.items() if (v["r2"] or 0) < 0]
    assert len(negatives) >= 25, len(negatives)
    assert all(rel[n]["tier"] == "low" for n in negatives)


def test_reliability_available_for_the_shipped_bundle():
    """The shipped artifact predates the stored table; it must still get tiers."""
    rel = P.reliability_map()
    assert rel[KNOWN_BAD_DRUG]["tier"] == "low"
    assert {v["tier"] for v in rel.values()} <= {"high", "moderate", "low", "unknown"}


# --- surfaced through the API -------------------------------------------------
def _rank_a_bad_drug():
    """Rank with the low-reliability drug guaranteed present, by asking for all."""
    sample, _w, _s = P.sample_from_dict({"tissue": "breast", "ERBB2_amp": 1})
    return P.predict(sample, top_k=None)


def test_ranked_entries_carry_the_flag():
    result = _rank_a_bad_drug()
    by_name = {r["therapy"]: r for r in result["ranked"]}

    bad = by_name[KNOWN_BAD_DRUG]
    assert bad["low_reliability"] is True
    assert bad["reliability"] == "low"
    assert bad["reliability_r2"] < 0

    assert all("low_reliability" in r for r in result["ranked"])
    assert result["low_reliability_count"] >= 1


def test_default_flags_but_does_not_exclude():
    result = _rank_a_bad_drug()
    assert result["excluded_low_reliability"] is False
    assert KNOWN_BAD_DRUG in {r["therapy"] for r in result["ranked"]}


def test_exclusion_is_opt_in_and_works():
    sample, _w, _s = P.sample_from_dict({"tissue": "breast", "ERBB2_amp": 1})
    result = P.predict(sample, top_k=None, exclude_low_reliability=True)

    assert result["excluded_low_reliability"] is True
    assert result["low_reliability_count"] == 0
    assert KNOWN_BAD_DRUG not in {r["therapy"] for r in result["ranked"]}
    assert not any(r["low_reliability"] for r in result["ranked"])


def test_exclusion_never_empties_the_ranking():
    """If every candidate were low-reliability we must still answer."""
    sample, _w, _s = P.sample_from_dict({"tissue": "skin", "BRAF_mut": 1})
    result = P.predict(sample, top_k=8, exclude_low_reliability=True)
    assert len(result["ranked"]) == 8


def test_api_response_exposes_reliability(client):
    resp = client.post("/api/predict_form", json={"tissue": "skin", "BRAF_mut": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert "low_reliability_count" in body
    for row in body["ranked"]:
        assert row["reliability"] in {"high", "moderate", "low", "unknown"}


def test_batch_rows_carry_reliability(client):
    resp = client.post(
        "/api/predict_batch",
        files={"file": ("c.csv", b"tissue,BRAF_mut\nskin,1\nbreast,0\n", "text/csv")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "low_reliability_count" in body
    for row in body["rows"]:
        assert "low_reliability" in row
        assert row["reliability"] in {"high", "moderate", "low", "unknown"}
