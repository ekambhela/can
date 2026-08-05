"""The prediction routes are rate limited.

/api/predict* are public, unauthenticated and genuinely expensive — ~0.4 s of
CPU for one prediction, ~20 s for a full 500-row cohort — on a free-tier
single-instance deploy behind a custom domain with autoDeploy on. Without a
limit, one loop saturates the box.

conftest.py disables the limiter for the rest of the suite (which hammers these
routes on purpose); this module builds its own app with it enabled.
"""

import importlib
import os

import pytest
from fastapi.testclient import TestClient

CSV = b"tissue,BRAF_mut\nskin,1\n"


@pytest.fixture
def limited_app(monkeypatch):
    """A fresh app instance with rate limiting on and a tiny limit."""
    monkeypatch.setenv("KARKIVE_RATE_LIMIT", "1")
    monkeypatch.setenv("KARKIVE_PREDICT_RATE_LIMIT", "3/minute")
    monkeypatch.setenv("KARKIVE_BATCH_RATE_LIMIT", "2/minute")

    import app as app_module
    reloaded = importlib.reload(app_module)
    try:
        yield reloaded
    finally:
        # Restore the module other test files imported, with limiting off again.
        monkeypatch.setenv("KARKIVE_RATE_LIMIT", "0")
        importlib.reload(app_module)


def _client(mod):
    return TestClient(mod.app)


def test_limits_are_configurable_from_the_environment(limited_app):
    assert limited_app.PREDICT_RATE_LIMIT == "3/minute"
    assert limited_app.BATCH_RATE_LIMIT == "2/minute"
    assert limited_app.limiter.enabled


def test_predict_form_is_limited(limited_app):
    client = _client(limited_app)
    codes = [client.post("/api/predict_form",
                         json={"tissue": "skin", "BRAF_mut": 1}).status_code
             for _ in range(6)]

    assert codes[:3] == [200, 200, 200], codes
    assert 429 in codes, codes


def test_predict_upload_is_limited(limited_app):
    client = _client(limited_app)
    codes = [client.post("/api/predict",
                         files={"file": ("s.csv", CSV, "text/csv")}).status_code
             for _ in range(6)]
    assert 429 in codes, codes


def test_batch_has_its_own_tighter_limit(limited_app):
    client = _client(limited_app)
    codes = [client.post("/api/predict_batch",
                         files={"file": ("c.csv", CSV, "text/csv")}).status_code
             for _ in range(5)]
    assert codes[:2] == [200, 200], codes
    assert 429 in codes, codes


def test_health_is_not_limited(limited_app):
    """The host polls /api/health constantly; limiting it would fail the deploy."""
    client = _client(limited_app)
    codes = [client.get("/api/health").status_code for _ in range(25)]
    assert set(codes) == {200}, sorted(set(codes))


def test_index_and_schema_are_not_limited(limited_app):
    client = _client(limited_app)
    assert {client.get("/").status_code for _ in range(15)} == {200}
    assert {client.get("/api/schema").status_code for _ in range(15)} == {200}


def test_limiter_is_disabled_for_the_rest_of_the_suite():
    """Guards the conftest switch — otherwise unrelated tests fail mysteriously."""
    assert os.environ.get("KARKIVE_RATE_LIMIT") == "0"
