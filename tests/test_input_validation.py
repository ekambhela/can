"""HTTP contract for input the model refuses to guess at.

Tissue drives the prediction — tissue-blocked CV puts mean Spearman at 0.10
versus 0.38 within-tissue — so silently defaulting a missing tissue to
lung_NSCLC answered a different question than the one asked, with nothing in
the response saying so. It is now a 422 that names the field and lists what is
accepted.

Unspecified biomarkers are still scored as negative/wild-type (see
experiments/run_missing_features.py for why), but the response now says which
ones the model was told about and which it assumed.
"""

import pytest
from fastapi.testclient import TestClient

import app as app_module
from model.schema import BINARY_FEATURES


@pytest.fixture(scope="module")
def client():
    return TestClient(app_module.app)


def _upload(client, raw, name="s.csv", route="/api/predict"):
    return client.post(route, files={"file": (name, raw, "text/csv")})


# --- tissue is required ------------------------------------------------------
@pytest.mark.parametrize("payload,why", [
    ({"TP53_mut": 1}, "missing tissue"),
    ({"tissue": "Klingon homeworld", "TP53_mut": 1}, "unknown tissue"),
    ({"tissue": "", "TP53_mut": 1}, "blank tissue"),
])
def test_form_rejects_unusable_tissue(client, payload, why):
    resp = client.post("/api/predict_form", json=payload)
    assert resp.status_code == 422, why

    body = resp.json()
    assert body["field"] == "tissue"
    assert body["valid_values"], "must list acceptable values"
    assert "skin" in body["valid_values"]


def test_upload_rejects_missing_tissue(client):
    resp = _upload(client, b"TP53_mut,KRAS_mut\n1,0\n")
    assert resp.status_code == 422
    assert resp.json()["field"] == "tissue"


def test_cohort_rejects_bad_row_and_says_which(client):
    raw = b"tissue,TP53_mut\nskin,1\nnot_a_tissue,1\n"
    resp = _upload(client, raw, route="/api/predict_batch")
    assert resp.status_code == 422
    body = resp.json()
    assert body["field"] == "tissue"
    assert "Row 2" in body["detail"], body["detail"]


def test_valid_tissue_still_works(client):
    resp = _upload(client, b"tissue,BRAF_mut\nskin,1\n")
    assert resp.status_code == 200, resp.text


# --- specified vs assumed ----------------------------------------------------
def test_response_reports_what_was_measured_and_what_was_assumed(client):
    resp = _upload(client, b"tissue,BRAF_mut\nskin,1\n")
    assert resp.status_code == 200
    body = resp.json()

    assert body["specified_features"] == ["BRAF_mut"]
    assumed = body["assumed_features"]
    assert "TP53_mut" in assumed
    assert "BRAF_mut" not in assumed
    assert len(assumed) + len(body["specified_features"]) == len(BINARY_FEATURES)

    assert any("not specified" in w for w in body["warnings"]), body["warnings"]


def test_form_response_reports_assumptions(client):
    resp = client.post("/api/predict_form", json={"tissue": "skin", "BRAF_mut": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert "specified_features" in body
    assert "assumed_features" in body


def test_fully_specified_upload_has_no_assumptions(client):
    header = "tissue," + ",".join(BINARY_FEATURES)
    row = "skin," + ",".join("1" if f == "BRAF_mut" else "0" for f in BINARY_FEATURES)
    resp = _upload(client, f"{header}\n{row}\n".encode())

    assert resp.status_code == 200
    body = resp.json()
    assert body["assumed_features"] == []
    assert not any("not specified" in w for w in body["warnings"])
