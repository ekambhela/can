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

from model.predict import (
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


SAMPLES_DIR = os.path.join(BASE, "static", "samples")


def _warm_examples() -> None:
    """Pre-compute (and cache) predictions for the built-in example files so the
    very first click on an example returns instantly, not in a few seconds."""
    import glob

    for path in sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.csv"))
                       + glob.glob(os.path.join(SAMPLES_DIR, "*.json"))):
        name = os.path.basename(path)
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
            if "cohort" in name.lower():
                samples, _ = parse_cohort(raw, name)
                predict_batch(samples)
            else:
                sample, _ = parse_sample(raw, name)
                predict(sample)
        except Exception as exc:  # noqa: BLE001 — warming is best-effort
            log.warning("could not warm example %s: %s", name, exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the model at boot (so the first request is fast) and pre-warm the
    example files in the background (so their first click is instant, without
    delaying startup or the health check).

    Replaces the deprecated @app.on_event("startup") hook with FastAPI's
    lifespan context manager (the supported API since Starlette 0.26).
    """
    import threading

    try:
        get_bundle()
        # warm examples off the startup path; lru_cache is thread-safe.
        threading.Thread(target=_warm_examples, name="warm-examples", daemon=True).start()
    except Exception as exc:  # noqa: BLE001 — log and continue; /api/health reports it
        log.warning("model not ready at startup: %s", exc)
    yield


app = FastAPI(title="Karkive", version="1.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")


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
async def api_predict_form(payload: dict = Body(...)) -> JSONResponse:
    """Rank therapies for a single tumor described by a JSON field dict."""
    if not isinstance(payload, dict) or not payload:
        raise HTTPException(status_code=400, detail="Empty or invalid payload.")
    try:
        sample, warnings = sample_from_dict(payload)
        result = predict(sample)
    except (ModelUnavailable, HTTPException):
        raise  # 503 / explicit status — must not be recast as a 422 below
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not score sample: {exc}")
    result["parsed_sample"] = sample
    result["warnings"] = warnings
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
async def api_predict(file: UploadFile = File(...)) -> JSONResponse:
    raw = await read_upload(file)

    try:
        sample, warnings = parse_sample(raw, file.filename or "")
    except (ModelUnavailable, HTTPException):
        raise  # 503 / explicit status — must not be recast as a 422 below
    except Exception as exc:  # noqa: BLE001 — surface a clean parse error to the UI
        raise HTTPException(status_code=422, detail=f"Could not parse sample: {exc}")

    result = predict(sample)   # ModelUnavailable -> 503 via the exception handler

    # Echo back the parsed sample so the UI can show what the model "saw".
    result["parsed_sample"] = sample
    result["warnings"] = warnings
    result["filename"] = file.filename
    return JSONResponse(result)


@app.post("/api/predict_batch")
async def api_predict_batch(file: UploadFile = File(...)) -> JSONResponse:
    """Rank therapies for a whole cohort (one tumor per row)."""
    raw = await read_upload(file)

    try:
        samples, warnings = parse_cohort(raw, file.filename or "")
    except (ModelUnavailable, HTTPException):
        raise  # 503 / explicit status — must not be recast as a 422 below
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not parse cohort: {exc}")

    if len(samples) > MAX_COHORT_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"Cohort has {len(samples)} rows; max {MAX_COHORT_ROWS}.",
        )

    result = predict_batch(samples)   # ModelUnavailable -> 503 via the exception handler

    result["warnings"] = warnings[:20]  # cap noise
    result["filename"] = file.filename
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
