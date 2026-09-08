"""
Wrapper optimisé pour image-processor-lite.
Utilise Tesseract directement (sans EasyOCR/PyTorch) via le pipeline preprocessing.py.
"""

import cv2
import time
import numpy as np
from typing import Dict

from .preprocessing import (
    to_grayscale, binarize,
    apply_gaussian_blur, invert_if_dark,
    upscale_if_small, apply_clahe,
)
from .ocr_module import run_ocr


class FastImageProcessor:
    """
    Wrapper optimisé utilisant Tesseract directement.
    - Pas de chargement d'EasyOCR ou PyTorch
    - Pipeline complet : upscale → grayscale → CLAHE → blur → inversion → binarisation

    Note : quand enable_preprocessing=False, l'image doit déjà être en RGB (pas BGR).
    """

    def __init__(self, enable_preprocessing: bool = True):
        self.enable_preprocessing = enable_preprocessing

    def extract_text(self, image: np.ndarray) -> str:
        """
        Extrait le texte d'une image numpy RGB.
        """
        if self.enable_preprocessing:
            image = self._preprocess(image)
        return run_ocr(image)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Pipeline de prétraitement : upscale → grayscale → CLAHE → flou → inversion → binarisation.
        """
        image = upscale_if_small(image)
        gray = to_grayscale(image)
        gray = apply_clahe(gray)
        gray = apply_gaussian_blur(gray)
        gray = invert_if_dark(gray)
        return binarize(gray, method="otsu")

    def process_image(self, image_path: str) -> Dict:
        """
        Traite une image depuis un chemin (interface compatible avec ImageProcessor).
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"cv2.imread: impossible d'ouvrir {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        text = self.extract_text(image)

        return {
            'text': text,
            'shapes': [],
            'visual_analysis': {},
        }


if __name__ == "__main__":
    processor = FastImageProcessor(enable_preprocessing=True)

    image_path = '/home/virus-one/Documents/projet_orc/amanda1/img/Capture.png'
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"cv2.imread: impossible d'ouvrir {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    start_time = time.time()
    text = processor.extract_text(image)
    elapsed_time = time.time() - start_time

    print(text)
    print(f"\nTemps d'exécution: {elapsed_time:.3f}s")
