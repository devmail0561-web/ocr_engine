import time
from .image_loader import load_image
from .preprocessing import (
    to_grayscale, binarize,
    apply_clahe, apply_gaussian_blur,
    invert_if_dark, upscale_if_small,
)
from .ocr_module import run_ocr


def main():
    start_time = time.time()

    path = "/home/virus-one/Documents/projet_orc/amanda1/img/image.png"
    mat, meta = load_image(path)
    print("Métadonnées détectées:", meta)

    # Pipeline de prétraitement
    mat = upscale_if_small(mat)
    gray = to_grayscale(mat)
    gray = apply_clahe(gray)
    gray = apply_gaussian_blur(gray)
    gray = invert_if_dark(gray)
    binary = binarize(gray, method="otsu")

    texte = run_ocr(binary, lang="fra+eng")
    print("Texte extrait:\n", texte)

    elapsed_time = time.time() - start_time
    print(f"\nTemps d'exécution (amanda1): {elapsed_time:.3f}s")


if __name__ == "__main__":
    main()
