"""HTTP-layer coverage: every endpoint, every error branch in app.py.

The 503-on-broken-model branch lives in test_model_unavailable.py and the
413-on-oversized-cohort branch in test_cohort_cap.py, since each needed its own
fixture; everything else in app.py is here.
"""

import pytest
from fastapi.testclient import TestClient

import app as app_module

CSV = b"tissue,BRAF_mut\nskin,1\n"
COHORT = b"tissue,BRAF_mut\nskin,1\nbreast,0\nlung_NSCLC,0\n"


@pytest.fixture(scope="module")
def client():
    return TestClient(app_module.app)


def _upload(client, raw, route="/api/predict", name="s.csv", mime="text/csv"):
    return client.post(route, files={"file": (name, raw, mime)})


# --- pages and static ---------------------------------------------------------
def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<html" in resp.text.lower()


def test_static_is_mounted(client):
    assert client.get("/static/app.js").status_code == 200


# --- /api/schema --------------------------------------------------------------
def test_schema_shape(client):
    resp = client.get("/api/schema")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("tissues", "mutations", "extras"):
        assert body[key], key
        assert all({"key", "label"} <= set(item) for item in body[key])
    assert any(t["key"] == "skin" for t in body["tissues"])


# --- /api/health and /api/metrics --------------------------------------------
def test_health_ok(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"]
    assert "top1_accuracy" in body["metrics"]
    # health is polled constantly — it must stay small
    assert "per_drug_r2" not in body["metrics"]


def test_metrics_includes_per_drug(client):
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["per_drug_r2"]) > 300
    assert len(body["per_drug_spearman"]) > 300


# --- /api/predict: success ----------------------------------------------------
def test_predict_success_shape(client):
    resp = _upload(client, CSV)
    assert resp.status_code == 200
    body = resp.json()

    for key in ("recommendation", "separation_score", "confidence", "decision_margin",
                "ranked", "model_metrics", "parsed_sample", "warnings", "filename",
                "specified_features", "assumed_features", "low_reliability_count"):
        assert key in body, key

    assert body["filename"] == "s.csv"
    assert len(body["ranked"]) == 8
    assert body["ranked"][0]["therapy"] == body["recommendation"]

    for i, row in enumerate(body["ranked"], start=1):
        assert row["rank"] == i
        for key in ("therapy", "drug_class", "sensitivity", "match_percent",
                    "ci_low", "ci_high", "rationale", "supporting", "cautions",
                    "reliability", "low_reliability"):
            assert key in row, key
        assert 0.0 <= row["match_percent"] <= 100.0
        assert row["ci_low"] <= row["match_percent"] <= row["ci_high"]

    sens = [r["sensitivity"] for r in body["ranked"]]
    assert sens == sorted(sens, reverse=True)


@pytest.mark.parametrize("raw,name,mime", [
    (CSV, "s.csv", "text/csv"),
    (b"tissue\tBRAF_mut\nskin\t1\n", "s.tsv", "text/tab-separated-values"),
    (b'{"tissue":"skin","BRAF_mut":1}', "s.json", "application/json"),
    (b'[{"tissue":"skin","BRAF_mut":1}]', "s.json", "application/json"),
    (b"feature,value\ntissue,skin\nBRAF_mut,1\n", "s.csv", "text/csv"),
])
def test_predict_accepts_every_supported_shape(client, raw, name, mime):
    resp = _upload(client, raw, name=name, mime=mime)
    assert resp.status_code == 200, resp.text
    assert resp.json()["parsed_sample"]["tissue"] == "skin"


# --- /api/predict: error branches ---------------------------------------------
def test_empty_file_is_400(client):
    resp = _upload(client, b"")
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_oversized_file_is_413(client):
    resp = _upload(client, b"x" * (app_module.MAX_BYTES + 1))
    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"].lower()


def test_at_the_byte_cap_is_not_rejected_for_size(client):
    """Exactly MAX_BYTES must pass the size gate (it fails later, on content)."""
    resp = _upload(client, b"x" * app_module.MAX_BYTES)
    assert resp.status_code != 413


@pytest.mark.parametrize("raw,name", [
    (b"\x00\x01\x02 not a table", "s.csv"),
    (b"{not valid json", "s.json"),
    (b"   \n  \t \n  ", "s.csv"),          # whitespace only
    (b"just one column\n", "s.csv"),       # no tissue
])
def test_malformed_input_is_422(client, raw, name):
    resp = _upload(client, raw, name=name)
    assert resp.status_code == 422, f"{name}: {resp.status_code} {resp.text[:200]}"
    assert resp.json()["detail"]


def test_missing_file_field_is_422(client):
    assert client.post("/api/predict").status_code == 422


# --- /api/predict_form --------------------------------------------------------
def test_predict_form_success(client):
    resp = client.post("/api/predict_form", json={"tissue": "skin", "BRAF_mut": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ranked"] and body["recommendation"]
    assert body["parsed_sample"]["BRAF_mut"] == 1.0


def test_predict_form_empty_payload_is_400(client):
    assert client.post("/api/predict_form", json={}).status_code == 400


def test_predict_form_non_object_payload_is_422(client):
    """A list never reaches the handler — FastAPI rejects the body type first."""
    assert client.post("/api/predict_form", json=[]).status_code == 422


def test_predict_form_unusable_tissue_is_422(client):
    resp = client.post("/api/predict_form", json={"tissue": "nowhere", "BRAF_mut": 1})
    assert resp.status_code == 422
    assert resp.json()["field"] == "tissue"


# --- /api/predict_batch -------------------------------------------------------
def test_predict_batch_success_shape(client):
    resp = _upload(client, COHORT, route="/api/predict_batch", name="c.csv")
    assert resp.status_code == 200
    body = resp.json()

    assert body["n"] == 3
    assert len(body["rows"]) == 3
    assert body["therapies"]
    assert body["filename"] == "c.csv"

    for i, row in enumerate(body["rows"], start=1):
        assert row["index"] == i
        for key in ("cancer_type", "recommendation", "drug_class", "match_percent",
                    "separation_score", "confidence", "decision_margin",
                    "runner_up", "runner_up_percent", "reliability"):
            assert key in row, key
        assert row["match_percent"] >= row["runner_up_percent"]


def test_predict_batch_empty_file_is_400(client):
    assert _upload(client, b"", route="/api/predict_batch").status_code == 400


def test_predict_batch_malformed_is_422(client):
    resp = _upload(client, b"\x00\x01 nonsense", route="/api/predict_batch")
    assert resp.status_code == 422


def test_batch_warnings_are_capped(client):
    """Cohort warnings are truncated so a bad upload can't return a novel."""
    rows = b"".join(b"skin,1\n" for _ in range(60))
    resp = _upload(client, b"tissue,BRAF_mut\n" + rows, route="/api/predict_batch")
    assert resp.status_code == 200
    assert len(resp.json()["warnings"]) <= 20


# --- single/batch parity ------------------------------------------------------
@pytest.mark.parametrize("row", [
    b"skin,1,0,0,0",
    b"breast,0,1,0,0",
    b"lung_NSCLC,0,0,1,0",
    b"large_intestine,0,0,0,1",
])
def test_batch_and_single_agree_on_top_1(client, row):
    header = b"tissue,BRAF_mut,ERBB2_amp,EGFR_mut,KRAS_mut\n"

    single = _upload(client, header + row + b"\n")
    batch = _upload(client, header + row + b"\n", route="/api/predict_batch", name="c.csv")

    assert single.status_code == batch.status_code == 200
    assert batch.json()["rows"][0]["recommendation"] == single.json()["recommendation"]
    assert batch.json()["rows"][0]["match_percent"] == \
           single.json()["ranked"][0]["match_percent"]


def test_multi_row_batch_matches_each_single(client):
    header = b"tissue,BRAF_mut,ERBB2_amp\n"
    rows = [b"skin,1,0", b"breast,0,1", b"lung_NSCLC,0,0"]

    batch = _upload(client, header + b"\n".join(rows) + b"\n",
                    route="/api/predict_batch", name="c.csv").json()

    for i, row in enumerate(rows):
        single = _upload(client, header + row + b"\n").json()
        assert batch["rows"][i]["recommendation"] == single["recommendation"], row
