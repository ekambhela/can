"""The prediction cache must never outlive the model it was computed from.

_predict_cached was keyed on id(load_bundle()). CPython reuses address values
after an object is freed, so a reload — load_bundle.cache_clear() plus a fresh
load, which tests/test_train_smoke.py does and any redeploy-in-place would —
can produce a bundle sitting at the old bundle's address and silently inherit
its cached predictions.
"""

import pytest

from model import predict as P


@pytest.fixture
def sample():
    s, _w, _spec = P.sample_from_dict({"tissue": "skin", "BRAF_mut": 1})
    return s


# --- the token itself ---------------------------------------------------------
def test_token_is_stable_for_one_bundle():
    b = {"kind": "fake"}
    assert P._bundle_token(b) == P._bundle_token(b)


def test_distinct_bundles_get_distinct_tokens():
    a, b = {"kind": "a"}, {"kind": "b"}
    assert P._bundle_token(a) != P._bundle_token(b)


def test_tokens_are_not_recycled_when_ids_are():
    """The actual bug: two objects can share an id() but must not share a token."""
    victim = {"kind": "first"}
    token = P._bundle_token(victim)
    address = id(victim)
    del victim

    collided = None
    for _ in range(200_000):
        candidate = {"kind": "second"}
        if id(candidate) == address:
            collided = candidate
            break
        del candidate

    if collided is None:
        pytest.skip("could not provoke an id() collision on this interpreter")

    assert P._bundle_token(collided) != token, (
        "a recycled id() handed the new bundle the old bundle's cache key"
    )


# --- end to end ---------------------------------------------------------------
def test_reload_invalidates_cached_predictions(monkeypatch, sample):
    """Swapping the bundle must change the answer, with no explicit cache clear."""
    real = P.get_bundle()
    first = P.predict(sample, top_k=5)

    # A genuinely different model: per-drug component only, so scores differ.
    swapped = dict(real)
    swapped.pop("_cache_token", None)
    swapped["blend_w_perdrug"] = 1.0
    monkeypatch.setattr(P, "load_bundle", lambda: swapped)

    second = P.predict(sample, top_k=5)

    assert [r["sensitivity"] for r in second["ranked"]] != \
           [r["sensitivity"] for r in first["ranked"]], \
           "second bundle served the first bundle's cached predictions"


def test_same_bundle_still_caches(sample):
    """The cache must still work — this is what keeps repeat clicks instant."""
    P.predict(sample, top_k=5)
    before = P._predict_cached.cache_info()
    P.predict(sample, top_k=5)
    after = P._predict_cached.cache_info()
    assert after.hits == before.hits + 1


def test_cache_key_separates_exclusion_setting(sample):
    """Flagged-vs-excluded are different questions and must not share an entry."""
    flagged = P.predict(sample, top_k=8, exclude_low_reliability=False)
    excluded = P.predict(sample, top_k=8, exclude_low_reliability=True)
    assert flagged["excluded_low_reliability"] is False
    assert excluded["excluded_low_reliability"] is True
