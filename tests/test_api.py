"""API endpoint / schema tests (audit item 2).

Exercises the real FastAPI routes end-to-end against a tiny, seeded model so the
request→parse→predict→response wiring and error handling are covered without the
11 MB shipped artifact. The tiny model is monkeypatched in for `load_bundle`, and
the app is used without triggering its (heavy) lifespan startup.
"""

import warnings

import pytest
from scipy.stats import ConstantInputWarning


@pytest.fixture(scope="module")
def tiny_bundle():
    from model.train import run_training
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        return run_training(seed=0, max_lines=80, max_drugs=15)


@pytest.fixture()
def client(tiny_bundle, monkeypatch):
    import app as appmod
    from model import predict as P

    # both the app-level name (used by /api/health) and the model-level name
    # (used internally by predict/parse) must point at the tiny bundle.
    monkeypatch.setattr(P, "load_bundle", lambda: tiny_bundle)
    monkeypatch.setattr(appmod, "load_bundle", lambda: tiny_bundle)
    P._predict_cached.cache_clear()

    from fastapi.testclient import TestClient
    # no `with` -> app lifespan (which would train/warm) is not run.
    return TestClient(appmod.app)


def test_health_is_slim(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    m = body["metrics"]
    assert "mean_spearman" in m
    # the big per-drug dicts must NOT be in the summary payload
    assert "per_drug_spearman" not in m
    assert "per_drug_r2" not in m


def test_schema_shape(client):
    r = client.get("/api/schema")
    assert r.status_code == 200
    s = r.json()
    for key in ("tissues", "mutations", "extras"):
        assert isinstance(s[key], list) and s[key]
    for item in s["mutations"]:
        assert "key" in item and "label" in item


def test_predict_csv_file(client):
    csv = b"tissue,BRAF_mut,TP53_mut\nskin,1,1\n"
    r = client.post("/api/predict", files={"file": ("s.csv", csv, "text/csv")})
    assert r.status_code == 200
    d = r.json()
    assert d["ranked"], "expected a non-empty ranking"
    assert d["recommendation"] == d["ranked"][0]["therapy"]
    assert 0.0 <= d["ranked"][0]["match_percent"] <= 100.0
    assert d["parsed_sample"]["tissue"] == "skin"
    # slim: no per-drug metric dicts embedded
    assert "per_drug_r2" not in d["model_metrics"]


def test_predict_form_json(client):
    r = client.post("/api/predict_form", json={"tissue": "skin", "BRAF_mut": 1})
    assert r.status_code == 200
    assert r.json()["ranked"]


def test_predict_form_rejects_empty(client):
    r = client.post("/api/predict_form", json={})
    assert r.status_code == 400


def test_predict_rejects_empty_file(client):
    r = client.post("/api/predict", files={"file": ("empty.csv", b"", "text/csv")})
    assert r.status_code == 400


def test_predict_rejects_oversized_file(client):
    big = b"x" * (2 * 1024 * 1024 + 1)
    r = client.post("/api/predict", files={"file": ("big.csv", big, "text/csv")})
    assert r.status_code == 413


def test_predict_batch_ranks_every_row(client):
    cohort = b"tissue,BRAF_mut\nskin,1\nbreast,0\n"
    r = client.post("/api/predict_batch", files={"file": ("c.csv", cohort, "text/csv")})
    assert r.status_code == 200
    d = r.json()
    assert d["n"] == 2
    assert len(d["rows"]) == 2
    assert d["rows"][0]["recommendation"]


def test_predict_rejects_malformed_json(client):
    bad = b'{"tissue": "skin", '  # truncated JSON
    r = client.post("/api/predict", files={"file": ("s.json", bad, "application/json")})
    assert r.status_code == 422
