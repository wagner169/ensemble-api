from __future__ import annotations

import io
import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image

from .inference import ModelBundle, CLASS_NAMES
from .ensemble import average_probs

# Grad-CAM es opcional y está apagado por defecto para ahorrar memoria
ALLOW_GRADCAM = os.getenv("ALLOW_GRADCAM", "0").lower() in ("1", "true", "yes")
try:
    from .explainability import gradcam_base64  # si no existe, no rompemos
except Exception:  # noqa: BLE001
    gradcam_base64 = None

app = FastAPI(title="Ensemble API")
models: ModelBundle | None = None


@app.on_event("startup")
async def _load():
    global models
    models = ModelBundle(models_dir="models")


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    gradcam: Optional[bool] = Form(False),
):
    """
    Predice la clase con ensamble y devuelve métricas de confianza.
    Grad-CAM sólo se genera si:
      - el cliente pide gradcam=true
      - ALLOW_GRADCAM=1|true en variables de entorno
      - existe gradcam_base64
    """
    assert models is not None, "Models not loaded"

    img_bytes = await file.read()
    _, per_model_probs = models.predict_single(img_bytes)
    ens = average_probs(per_model_probs)

    # Top-1 + métricas de confianza/discordancia
    sorted_classes = sorted(ens.items(), key=lambda kv: kv[1], reverse=True)
    top_class, top_prob = sorted_classes[0]
    second_prob = sorted_classes[1][1] if len(sorted_classes) > 1 else 0.0
    margin = float(top_prob - second_prob)

    votes = [max(p, key=p.get) for p in per_model_probs.values()]
    disagreement = 1.0 - (votes.count(top_class) / len(votes)) if votes else 0.0

    payload = {
        "ensemble": ens,
        "per_model": per_model_probs,
        "top_class": top_class,
        "confidence": float(top_prob),
        "margin_top2": margin,
        "uncertain": bool(margin < 0.05),
        "model_disagreement": float(disagreement),
        "classes": CLASS_NAMES,
    }

    # Grad-CAM opcional (apagado por defecto)
    if gradcam and ALLOW_GRADCAM and gradcam_base64 is not None:
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        payload["gradcam_jpg_base64"] = gradcam_base64(pil, models.resnet)

    return JSONResponse(payload)


