"""
Karkive — Tumor → Chemotherapy matching web service.

FastAPI backend that serves the single-page UI and exposes a prediction
endpoint. A tumor sample file is uploaded, parsed, and ranked against the
therapy panel by the trained drug-response model.

Run:
    uvicorn app:app --reload
or:
    python app.py
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from model.predict import (
    BINARY_FEATURES,
    CohortTooLarge,
    InvalidSample,
    ModelUnavailable,
    api_metrics,
    feature_schema,
    full_metrics,
    get_bundle,
    parse_cohort,
    parse_sample,
    predict,
    predict_batch,
    sample_from_dict,
)

BASE = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger("karkive.app")

MAX_BYTES = 2 * 1024 * 1024  # 2 MB upload cap
MAX_COHORT_ROWS = 500        # cap batch size to keep responses snappy

# Rate limits for the prediction routes. These are public, unauthenticated and
# genuinely expensive — a single prediction is ~0.4 s of CPU and a full 500-row
# cohort ~20 s — on a free-tier single-instance deploy with autoDeploy on a
# custom domain. The limits are set well above interactive use (a visitor
# clicking through every bundled example stays inside them) and well below what
# it takes to saturate the box.
PREDICT_RATE_LIMIT = os.environ.get("KARKIVE_PREDICT_RATE_LIMIT", "30/minute")
BATCH_RATE_LIMIT = os.environ.get("KARKIVE_BATCH_RATE_LIMIT", "6/minute")

# Keyed on client IP. Render terminates TLS in front of the app, so uvicorn is
# started with --proxy-headers (see Dockerfile) — without that every request
# would carry the proxy's address and the whole internet would share one bucket.
#
# Off by default under pytest (conftest.py), since the suite deliberately hammers
# these routes; tests/test_rate_limit.py turns it back on for itself.
limiter = Limiter(
    key_func=get_remote_address,
    headers_enabled=True,
    enabled=os.environ.get("KARKIVE_RATE_LIMIT", "1").strip().lower()
    not in {"0", "false", "off", "no"},
)


SAMPLES_DIR = os.path.join(BASE, "static", "samples")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the model at boot so the first request doesn't pay for it.

    There used to be a background thread pre-scoring every bundled example,
    because a cold example click took a few seconds. Batching the scoring passes
    brought a single prediction to ~0.4 s (see model/predict.py), so the warming
    thread — and the startup CPU spike it caused on a free-tier box — is gone.

    Replaces the deprecated @app.on_event("startup") hook with FastAPI's
    lifespan context manager (the supported API since Starlette 0.26).
    """
    try:
        get_bundle()
    except Exception as exc:  # noqa: BLE001 — log and continue; /api/health reports it
        log.warning("model not ready at startup: %s", exc)
    yield


app = FastAPI(title="Karkive", version="1.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # -> 429


@app.exception_handler(ModelUnavailable)
async def model_unavailable_handler(_request: Request, exc: ModelUnavailable) -> JSONResponse:
    """Any failure to load the model becomes a structured 503, on every route.

    Registered once rather than caught per-route: the failure mode that matters
    is a dependency bump making the pickle unloadable, and that can surface from
    any handler that touches the model — including /api/health, which the host
    polls to decide whether the deploy is alive. A 500 there reads as "app
    crashed"; a 503 with status=no_model says what is actually wrong.
    """
    return JSONResponse(status_code=503, content={"status": "no_model", "detail": str(exc)})


@app.exception_handler(InvalidSample)
async def invalid_sample_handler(_request: Request, exc: InvalidSample) -> JSONResponse:
    """Input we refuse to guess at -> 422 naming the field and its valid values."""
    content: dict = {"status": "invalid_sample", "detail": str(exc)}
    if exc.field:
        content["field"] = exc.field
    if exc.valid_values:
        content["valid_values"] = exc.valid_values
    return JSONResponse(status_code=422, content=content)


@app.exception_handler(CohortTooLarge)
async def cohort_too_large_handler(_request: Request, exc: CohortTooLarge) -> JSONResponse:
    return JSONResponse(status_code=413, content={"status": "too_large", "detail": str(exc)})


def _assumption_report(specified: list[str]) -> dict:
    """What the model was told vs what it assumed — surfaced on every response."""
    return {
        "specified_features": specified,
        "assumed_features": [f for f in BINARY_FEATURES if f not in specified],
    }


async def read_upload(file: UploadFile) -> bytes:
    """Read an uploaded file, rejecting empty or oversized payloads.

    Shared by /api/predict and /api/predict_batch so the size/emptiness policy
    lives in exactly one place.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 2 MB).")
    return raw


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    with open(os.path.join(BASE, "templates", "index.html"), encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


@app.get("/api/schema")
def schema() -> dict:
    """Input-field schema for the manual-entry form."""
    return feature_schema()


@app.post("/api/predict_form")
@limiter.limit(PREDICT_RATE_LIMIT)
async def api_predict_form(request: Request, payload: dict = Body(...)) -> JSONResponse:
    """Rank therapies for a single tumor described by a JSON field dict."""
    if not isinstance(payload, dict) or not payload:
        raise HTTPException(status_code=400, detail="Empty or invalid payload.")
    try:
        sample, warnings, specified = sample_from_dict(payload)
        result = predict(sample)
    except (ModelUnavailable, InvalidSample, HTTPException):
        raise  # structured 503/422 — must not be recast as a generic 422 below
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422,
                            detail=f"Could not score sample: {exc}") from exc
    result["parsed_sample"] = sample
    result["warnings"] = warnings
    result.update(_assumption_report(specified))
    return JSONResponse(result)


@app.get("/api/health")
def health() -> dict:
    """Liveness + summary metrics. Render polls this frequently, so it returns
    only the small summary — the 369-entry per-drug dicts live at /api/metrics."""
    bundle = get_bundle()   # ModelUnavailable -> 503 via the exception handler
    return {"status": "ok", "model_version": bundle.get("version"),
            "metrics": api_metrics()}


@app.get("/api/metrics")
def metrics() -> dict:
    """Full evaluation metrics, including per-drug R^2 / Spearman (large)."""
    return full_metrics()


@app.post("/api/predict")
@limiter.limit(PREDICT_RATE_LIMIT)
async def api_predict(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    raw = await read_upload(file)

    try:
        sample, warnings, specified = parse_sample(raw, file.filename or "")
    except (ModelUnavailable, InvalidSample, HTTPException):
        raise  # structured 503/422 — must not be recast as a generic 422 below
    except Exception as exc:  # noqa: BLE001 — surface a clean parse error to the UI
        raise HTTPException(status_code=422,
                            detail=f"Could not parse sample: {exc}") from exc

    result = predict(sample)   # ModelUnavailable -> 503 via the exception handler

    # Echo back the parsed sample so the UI can show what the model "saw".
    result["parsed_sample"] = sample
    result["warnings"] = warnings
    result["filename"] = file.filename
    result.update(_assumption_report(specified))
    return JSONResponse(result)


@app.post("/api/predict_batch")
@limiter.limit(BATCH_RATE_LIMIT)
async def api_predict_batch(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    """Rank therapies for a whole cohort (one tumor per row)."""
    raw = await read_upload(file)

    try:
        # The cap is applied during the read, so an oversized upload is rejected
        # without parsing and normalizing every row.
        samples, warnings = parse_cohort(raw, file.filename or "",
                                         max_rows=MAX_COHORT_ROWS)
    except (ModelUnavailable, InvalidSample, CohortTooLarge, HTTPException):
        raise  # structured 503/422/413 — must not be recast as a generic 422 below
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422,
                            detail=f"Could not parse cohort: {exc}") from exc

    result = predict_batch(samples)   # ModelUnavailable -> 503 via the exception handler

    result["warnings"] = warnings[:20]  # cap noise
    result["filename"] = file.filename
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
