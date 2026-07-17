"""
app.py
AI-Powered Financial Fraud Detection — FastAPI + Uvicorn Backend

Routes:
  GET  /                          → Serve the SPA (templates/index.html)
  GET  /api/status                → Model readiness flags
  POST /api/analyze/transaction   → ML + optional NLP fusion
  POST /api/analyze/text          → NLP-only analysis
  POST /api/analyze/signature     → CV signature verification

Run:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
import tempfile
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
SRC_DIR  = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# ── Thread pool for CPU-bound ML inference ────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=4)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Fraud Detection System",
    description="Multi-modal fraud detection: ML + NLP + CV",
    version="2.0.0",
)

# ── Static files & templates ──────────────────────────────────────────────────
STATIC_DIR    = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Model readiness helpers ───────────────────────────────────────────────────

def _model_ready(path: str) -> bool:
    p = Path(path)
    if p.is_file():
        return p.stat().st_size > 0
    if p.is_dir():
        return any(p.iterdir())
    return False

ML_READY  = _model_ready("models/xgboost_fraud.ubj") or _model_ready("models/xgboost_fraud.pkl")
NLP_READY = _model_ready("models/distilbert_fraud")
CV_READY  = _model_ready("models/siamese_resnet18.pt")

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    time_val:  float
    amount:    float
    pca:       List[float]          # 28 PCA features V1-V28
    text:      Optional[str] = None # optional NLP input
    threshold: float = 0.30


class TextRequest(BaseModel):
    text: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page application."""
    html_path = TEMPLATES_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def get_status():
    """Return which models are currently trained and ready."""
    return JSONResponse({
        "ml":  ML_READY,
        "nlp": NLP_READY,
        "cv":  CV_READY,
    })


@app.post("/api/analyze/transaction")
async def analyze_transaction(req: TransactionRequest):
    """Run ML (XGBoost) + optional NLP fusion on transaction features."""
    if not ML_READY:
        raise HTTPException(
            status_code=503,
            detail="XGBoost model not trained. Run `python train_all.py` first."
        )
    if len(req.pca) != 28:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 28 PCA features, got {len(req.pca)}"
        )

    # Build 30-feature vector: Time, V1..V28, Amount
    features = [req.time_val] + req.pca + [req.amount]

    def _run():
        from fusion import predict_fraud
        # Apply custom threshold by temporarily patching metrics file
        metrics_path = "models/ml_metrics.json"
        original_threshold = 0.5
        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            original_threshold = metrics.get("threshold", 0.5)
        metrics["threshold"] = req.threshold
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        try:
            result = predict_fraud(
                transaction_features=features,
                transaction_text=req.text.strip() if req.text and req.text.strip() else None,
            )
        finally:
            # Restore original threshold
            metrics["threshold"] = original_threshold
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
        return result

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _run)
    return JSONResponse(result)


@app.post("/api/analyze/text")
async def analyze_text(req: TextRequest):
    """Run NLP (DistilBERT) analysis on text input."""
    if not NLP_READY:
        raise HTTPException(
            status_code=503,
            detail="NLP model not trained. Run `python train_all.py` first."
        )
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Text input is empty.")

    def _run():
        from fusion import predict_fraud
        return predict_fraud(transaction_text=req.text.strip())

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _run)
    return JSONResponse(result)


@app.post("/api/analyze/signature")
async def analyze_signature(
    ref_image:  UploadFile = File(...),
    test_image: UploadFile = File(...),
):
    """Run CV (Siamese ResNet18) signature verification."""
    if not CV_READY:
        raise HTTPException(
            status_code=503,
            detail="CV model not trained. Run `python train_all.py` first."
        )

    allowed = {"image/png", "image/jpeg", "image/jpg"}
    for f in (ref_image, test_image):
        if f.content_type not in allowed:
            raise HTTPException(
                status_code=422,
                detail=f"File '{f.filename}' must be PNG or JPEG."
            )

    # Save uploaded files to a temp directory for the CV model
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path  = os.path.join(tmpdir, "ref.png")
        test_path = os.path.join(tmpdir, "test.png")

        ref_bytes  = await ref_image.read()
        test_bytes = await test_image.read()

        with open(ref_path,  "wb") as f: f.write(ref_bytes)
        with open(test_path, "wb") as f: f.write(test_bytes)

        def _run(rp, tp):
            from fusion import predict_fraud
            return predict_fraud(signature_paths=(rp, tp))

        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, _run, ref_path, test_path)

    return JSONResponse(result)
