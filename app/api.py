from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional

# --- NUEVOS IMPORTS ---
import io
import base64
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
# -----------------------

from .inference import ModelBundle, CLASS_NAMES, IMG_TFMS
from .ensemble import average_probs
from .explainability import gradcam_base64

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

    # Métricas de confianza/discordancia
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

    if gradcam:
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        b64 = gradcam_base64(pil, models.resnet)
        payload["gradcam_jpg_base64"] = b64

    return JSONResponse(payload)

# ---------- LIME ENDPOINT ----------
@app.post("/explain/lime")
async def explain_lime(file: UploadFile = File(...)):
    img_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Explicamos con el ResNet (puedes cambiar a otro)
    model = models.resnet
    model.eval()
    device = next(model.parameters()).device

    def batch_predict(imgs: list[np.ndarray]) -> np.ndarray:
        xs = []
        for arr in imgs:
            xs.append(IMG_TFMS(Image.fromarray(arr)).unsqueeze(0))
        x = torch.cat(xs, dim=0).to(device)
        with torch.inference_mode():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.array(pil_img),
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    img_boundaries = mark_boundaries(temp / 255.0, mask)
    img_boundaries = (img_boundaries * 255).astype(np.uint8)

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_boundaries, cv2.COLOR_RGB2BGR))
    if not ok:
        return JSONResponse({"error": "No se pudo codificar imagen LIME"}, status_code=500)
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return {"lime_jpg_base64": b64}
# -----------------------------------
