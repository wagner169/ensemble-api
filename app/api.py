from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
from .inference import ModelBundle, CLASS_NAMES
from .ensemble import average_probs
from .explainability import gradcam_base64
from PIL import Image

app = FastAPI(title="Ensemble API")
models = None

@app.on_event("startup")
async def _load():
    global models
    models = ModelBundle(models_dir="models")

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...), gradcam: Optional[bool] = Form(False)):
    img_bytes = await file.read()
    logits, per_model_probs = models.predict_single(img_bytes)
    ens = average_probs(per_model_probs)
    top_class = max(ens, key=ens.get)

    payload = {
        "ensemble": ens,
        "per_model": per_model_probs,
        "top_class": top_class,
        "classes": CLASS_NAMES,
    }

    if gradcam:
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        b64 = gradcam_base64(pil, models.resnet)
        payload["gradcam_jpg_base64"] = b64

    return JSONResponse(payload)