"""An oversized cohort must be rejected during parsing, not after it.

/api/predict_batch used to parse and normalize the entire upload, then compare
len(samples) to the cap. A 2 MB CSV is tens of thousands of rows, so the work an
oversized upload bought was exactly the work the cap exists to prevent. The cap
is now applied to the read itself.
"""

import time

import pytest
from fastapi.testclient import TestClient

import app as app_module
from model import predict as P

CAP = app_module.MAX_COHORT_ROWS

# Rejection should be near-instant. Generous vs. the ~0.1 s it actually takes,
# but far below the seconds a full parse-then-reject costs.
BUDGET_S = 3.0


@pytest.fixture(scope="module")
def client():
    return TestClient(app_module.app)


def _cohort_csv(rows: int) -> bytes:
    head = "tissue,TP53_mut,KRAS_mut,BRAF_mut,EGFR_mut,MSI\n"
    body = "".join(
        f"{'skin' if i % 2 else 'breast'},{i % 2},{(i + 1) % 2},0,0,0\n"
        for i in range(rows)
    )
    return (head + body).encode()


def _cohort_csv_of_size(target_bytes: int) -> bytes:
    """A whole-line cohort CSV just under `target_bytes`."""
    raw = _cohort_csv(target_bytes // 16 + 1000)
    raw = raw[:target_bytes]
    return raw[:raw.rfind(b"\n") + 1]


def _post(client, raw, name="cohort.csv"):
    return client.post("/api/predict_batch", files={"file": (name, raw, "text/csv")})


def test_two_megabyte_cohort_is_rejected_fast(client):
    """The stated case: a ~2 MB upload, just under the byte cap so the ROW cap
    is what has to reject it."""
    raw = _cohort_csv_of_size(app_module.MAX_BYTES)
    assert 1_900_000 < len(raw) <= app_module.MAX_BYTES, len(raw)
    assert raw.count(b"\n") > 100_000, "should be a very high row count"

    start = time.perf_counter()
    resp = _post(client, raw)
    elapsed = time.perf_counter() - start

    assert resp.status_code == 413, resp.text
    assert str(CAP) in resp.json()["detail"]
    assert elapsed < BUDGET_S, f"rejection took {elapsed:.2f}s — is it parsing everything?"


def test_cohort_at_the_cap_is_accepted(client):
    resp = _post(client, _cohort_csv(CAP))
    assert resp.status_code == 200, resp.text
    assert resp.json()["n"] == CAP


def test_one_row_over_the_cap_is_rejected(client):
    resp = _post(client, _cohort_csv(CAP + 1))
    assert resp.status_code == 413, resp.text


def test_oversized_json_cohort_is_rejected(client):
    body = "[" + ",".join('{"tissue":"skin","TP53_mut":1}' for _ in range(CAP + 50)) + "]"
    resp = client.post("/api/predict_batch",
                       files={"file": ("c.json", body.encode(), "application/json")})
    assert resp.status_code == 413, resp.text


# --- the parse layer ----------------------------------------------------------
def test_parse_cohort_raises_before_normalizing(monkeypatch):
    """Nothing should be normalized if the row count is already over."""
    calls = []
    real = P._normalize
    monkeypatch.setattr(P, "_normalize", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    with pytest.raises(P.CohortTooLarge):
        P.parse_cohort(_cohort_csv(CAP + 10), "c.csv", max_rows=CAP)

    assert not calls, "rows were normalized before the cap was enforced"


def test_key_value_file_is_not_truncated_by_the_cap():
    """A feature/value file is ONE sample over many rows — the cap must not
    silently drop its later features."""
    rows = "\n".join(f"{f},1" for f in P.BINARY_FEATURES)
    raw = f"feature,value\ntissue,skin\n{rows}\n".encode()

    samples, _warnings = P.parse_cohort(raw, "s.csv", max_rows=2)

    assert len(samples) == 1
    assert samples[0]["tissue"] == "skin"
    assert all(samples[0][f] == 1.0 for f in P.BINARY_FEATURES)


def test_no_cap_means_no_limit():
    samples, _w = P.parse_cohort(_cohort_csv(CAP + 25), "c.csv")
    assert len(samples) == CAP + 25
