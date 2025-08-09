from __future__ import annotations
import cv2
import io
import base64
import numpy as np
import torch
from PIL import Image
from .inference import IMG_TFMS

# Grad-CAM sobre la última capa conv del bloque final (resnet50)

class GradCAMResNet:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        # layer4[-1].conv2
        target_layer = self.model.layer4[-1].conv2
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    @torch.inference_mode(False)
    def run(self, pil_img: Image.Image) -> np.ndarray:
        x = IMG_TFMS(pil_img).unsqueeze(0)
        x = x.to(next(self.model.parameters()).device)

        # Forward + elegir clase top
        scores = self.model(x)  # [1, C]
        top_idx = scores.argmax(dim=1).item()

        # Backward de la clase top
        self.model.zero_grad()
        scores[0, top_idx].backward(retain_graph=True)

        # Grad-CAM
        grads = self.gradients  # [1, K, H, W]
        acts = self.activations # [1, K, H, W]
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, K, 1, 1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [1,1,H,W]
        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        # Redimensionar CAM a tamaño original
        img_np = np.array(pil_img)
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = (0.4 * heatmap + 0.6 * img_np).astype(np.uint8)
        return overlay


def gradcam_base64(pil_img: Image.Image, resnet_model) -> str:
    cam = GradCAMResNet(resnet_model).run(pil_img)
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(cam, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("No se pudo codificar Grad-CAM a JPG")
    b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
    return b64