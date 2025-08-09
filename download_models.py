import gdown
from pathlib import Path

FILES = {
    "resnet50": "1yhdFMGyaw-gW4yU5pPl6CFJRZVvS44np",
    "densenet121": "1KpLhlrcAgSaIUIIABNpu6LzFGGG-a2TS",
    "efficientnet_b0": "1FuliZMAQdaiOWYlsahiYJqzc8fy_2oUQ",
}

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

for name, file_id in FILES.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    out = MODELS_DIR / f"{name}.pth"
    print(f"Descargando {name} → {out} ...")
    gdown.download(url, str(out), quiet=False)

print("Listo: pesos descargados en ./models/")