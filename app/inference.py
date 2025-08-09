from __future__ import annotations
import io
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
CLASS_NAMES = ["Normal", "Opacidad Pulmonar", "Neumonía", "COVID-19"]  

IMG_TFMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _strip_module(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Limpia prefijo 'module.' si el modelo se guardó con DataParallel
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _build_resnet50() -> torch.nn.Module:
    m = models.resnet50(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def _build_densenet121() -> torch.nn.Module:
    m = models.densenet121(weights=None)
    m.classifier = torch.nn.Linear(m.classifier.in_features, NUM_CLASSES)
    return m


def _build_efficientnet_b0() -> torch.nn.Module:
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    return m


class ModelBundle:
    def __init__(self, models_dir: str | Path = "models") -> None:
        models_dir = Path(models_dir)
        self.resnet = _build_resnet50()
        self.densenet = _build_densenet121()
        self.efficient = _build_efficientnet_b0()

        # Carga de pesos
        for name, model in [
            ("resnet50", self.resnet),
            ("densenet121", self.densenet),
            ("efficientnet_b0", self.efficient),
        ]:
            weight_path = models_dir / f"{name}.pth"
            if not weight_path.exists():
                raise FileNotFoundError(f"No existe {weight_path}. Corre download_models.py")
            state = torch.load(weight_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = _strip_module(state)
            model.load_state_dict(state, strict=False)
            model.eval().to(DEVICE)

    @torch.inference_mode()
    def _predict_logits(self, image_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = {
            "resnet50": self.resnet(image_tensor),
            "densenet121": self.densenet(image_tensor),
            "efficientnet_b0": self.efficient(image_tensor),
        }
        return logits

    @torch.inference_mode()
    def predict_single(self, image_bytes: bytes) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, float]]]:
        """Devuelve (logits_por_modelo, probs_por_modelo)"""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = IMG_TFMS(img).unsqueeze(0).to(DEVICE)
        logits = self._predict_logits(x)
        probs = {}
        for k, v in logits.items():
            p = F.softmax(v, dim=1).squeeze(0).detach().cpu().tolist()
            probs[k] = {cls: float(p[i]) for i, cls in enumerate(CLASS_NAMES)}
        return logits, probs