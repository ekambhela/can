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

import os

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from model.predict import (
    feature_schema,
    load_bundle,
    parse_cohort,
    parse_sample,
    predict,
    predict_batch,
    sample_from_dict,
)

BASE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Karkive", version="1.1.0")
app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")

MAX_BYTES = 2 * 1024 * 1024  # 2 MB upload cap
MAX_COHORT_ROWS = 500        # cap batch size to keep responses snappy


@app.on_event("startup")
def _prewarm() -> None:
    """Train/load the model at boot so the first user request is fast."""
    try:
        load_bundle()
    except Exception as exc:  # noqa: BLE001 — log and continue; /api/health reports it
        print(f"[startup] model not ready: {exc}")


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
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not score sample: {exc}")
    result["parsed_sample"] = sample
    result["warnings"] = warnings
    return JSONResponse(result)


@app.get("/api/health")
def health() -> dict:
    try:
        bundle = load_bundle()
        return {"status": "ok", "model_version": bundle.get("version"),
                "metrics": bundle.get("metrics", {})}
    except FileNotFoundError as exc:
        return JSONResponse(status_code=503, content={"status": "no_model", "detail": str(exc)})


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 2 MB).")

    try:
        sample, warnings = parse_sample(raw, file.filename or "")
    except Exception as exc:  # noqa: BLE001 — surface a clean parse error to the UI
        raise HTTPException(status_code=422, detail=f"Could not parse sample: {exc}")

    try:
        result = predict(sample)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Echo back the parsed sample so the UI can show what the model "saw".
    result["parsed_sample"] = sample
    result["warnings"] = warnings
    result["filename"] = file.filename
    return JSONResponse(result)


@app.post("/api/predict_batch")
async def api_predict_batch(file: UploadFile = File(...)) -> JSONResponse:
    """Rank therapies for a whole cohort (one tumor per row)."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 2 MB).")

    try:
        samples, warnings = parse_cohort(raw, file.filename or "")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not parse cohort: {exc}")

    if len(samples) > MAX_COHORT_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"Cohort has {len(samples)} rows; max {MAX_COHORT_ROWS}.",
        )

    try:
        result = predict_batch(samples)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    result["warnings"] = warnings[:20]  # cap noise
    result["filename"] = file.filename
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
