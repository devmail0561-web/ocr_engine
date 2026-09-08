from PIL import Image, UnidentifiedImageError
import numpy as np

_MODE_DEPTH = {
    "1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32,
    "CMYK": 32, "YCbCr": 24, "LAB": 24, "HSV": 24,
    "I": 32, "F": 32, "LA": 16, "PA": 16,
}


def load_image(path: str):
    """
    Charge une image et retourne :
      - matrice numpy RGB (H, W, 3) en uint8
      - meta: dictionnaire avec encodage détecté
    """
    try:
        img = Image.open(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image introuvable: {path}")
    except UnidentifiedImageError:
        raise UnidentifiedImageError(f"Impossible d'identifier le fichier image: {path}")

    mode = img.mode
    dpi = img.info.get("dpi", (72, 72))
    bit_depth = _MODE_DEPTH.get(mode, 8 * len(img.getbands()))

    img_rgb = img.convert("RGB")
    bands = img_rgb.getbands()
    mat = np.array(img_rgb)

    meta = {
        "mode_original": mode,
        "bands": bands,
        "bit_depth": bit_depth,
        "dpi": dpi,
        "shape": mat.shape,
    }
    return mat, meta
