"""A broken model must 503 on every route — never 500.

The realistic production failure isn't a missing file, it's version skew: a
scikit-learn or numpy bump makes the committed pickle unloadable, and
joblib.load raises AttributeError (or ModuleNotFoundError, or something else
entirely). Those used to escape unhandled, so /api/health — the route Render
polls to decide whether the deploy is alive — returned 500.

AttributeError is used here as the stand-in for "not FileNotFoundError".
"""

import pytest
from fastapi.testclient import TestClient

import app as app_module
from model import predict as P

# The real lru_cache wrapper, kept so teardown can clear the cache even while
# P.load_bundle is monkeypatched to a plain function (which has no cache_clear).
_REAL_LOAD_BUNDLE = P.load_bundle


@pytest.fixture
def client():
    # No lifespan: it calls get_bundle() at startup, which we are breaking.
    return TestClient(app_module.app)


@pytest.fixture
def broken_model(monkeypatch):
    """Make every model load raise something that is NOT FileNotFoundError."""
    def boom():
        raise AttributeError("Can't get attribute '_loss' on <module 'sklearn...'>")

    _REAL_LOAD_BUNDLE.cache_clear()
    monkeypatch.setattr(P, "load_bundle", boom)
    yield
    _REAL_LOAD_BUNDLE.cache_clear()


CSV = b"tissue,BRAF_mut\nskin,1\n"


def _upload(client, route, raw=CSV, name="s.csv"):
    return client.post(route, files={"file": (name, raw, "text/csv")})


@pytest.mark.parametrize("call", [
    pytest.param(lambda c: c.get("/api/health"), id="health"),
    pytest.param(lambda c: c.get("/api/metrics"), id="metrics"),
    pytest.param(lambda c: c.get("/api/schema"), id="schema"),
    pytest.param(lambda c: _upload(c, "/api/predict"), id="predict"),
    pytest.param(lambda c: _upload(c, "/api/predict_batch"), id="predict_batch"),
    pytest.param(lambda c: c.post("/api/predict_form", json={"tissue": "skin"}),
                 id="predict_form"),
])
def test_broken_model_yields_503_not_500(client, broken_model, call):
    resp = call(client)
    assert resp.status_code == 503, f"got {resp.status_code}: {resp.text[:300]}"


def test_health_503_body_is_structured(client, broken_model):
    """Health is the contract the host reads; its shape is part of the fix."""
    resp = client.get("/api/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "no_model"
    assert body["detail"], "detail should say what went wrong"
    assert "AttributeError" in body["detail"]


def test_get_bundle_wraps_arbitrary_failures(monkeypatch):
    def boom():
        raise RuntimeError("pickle blew up")

    _REAL_LOAD_BUNDLE.cache_clear()
    monkeypatch.setattr(P, "load_bundle", boom)
    try:
        with pytest.raises(P.ModelUnavailable) as exc:
            P.get_bundle()
        assert isinstance(exc.value.__cause__, RuntimeError), "original error should be chained"
    finally:
        _REAL_LOAD_BUNDLE.cache_clear()


def test_get_bundle_wraps_missing_file(monkeypatch, tmp_path):
    _REAL_LOAD_BUNDLE.cache_clear()
    monkeypatch.setattr(P, "MODEL_PATH", str(tmp_path / "absent.joblib"))
    try:
        with pytest.raises(P.ModelUnavailable):
            P.get_bundle()
    finally:
        _REAL_LOAD_BUNDLE.cache_clear()
