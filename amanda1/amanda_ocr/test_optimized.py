"""
Version optimisée utilisant Tesseract directement (comme amanda1).
"""
import cv2
import time
from preprocessing import (
    to_grayscale, binarize,
    apply_clahe, apply_gaussian_blur,
    invert_if_dark, upscale_if_small,
)
from ocr_module import run_ocr


def optimize_and_extract_text(image_path):
    """Extraction de texte optimisée avec pipeline complet."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"cv2.imread: impossible d'ouvrir {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = upscale_if_small(image)
    gray = to_grayscale(image)
    gray = apply_clahe(gray)
    gray = apply_gaussian_blur(gray)
    gray = invert_if_dark(gray)
    binary = binarize(gray, method="otsu")

    return run_ocr(binary, lang="fra+eng")


if __name__ == "__main__":
    start_time = time.time()
    text = optimize_and_extract_text('/home/virus-one/Documents/projet_orc/amanda1/img/image.png')
    print("Texte extrait:")
    print(text)
    elapsed_time = time.time() - start_time
    print(f"\nTemps d'exécution (version optimisée): {elapsed_time:.3f}s")
