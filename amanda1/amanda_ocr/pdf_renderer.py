# Copyright (c) 2026 M. Tendeng — MIT License (see LICENSE)
import pymupdf as fitz
import numpy as np


def page_count(pdf_path: str) -> int:
    with fitz.open(pdf_path) as doc:
        return len(doc)


def _adaptive_dpi(page: fitz.Page) -> int:
    """DPI adaptatif basé sur la taille physique de la page (points PDF → pouces)."""
    width_in  = page.rect.width  / 72  # 1 point = 1/72 pouce
    height_in = page.rect.height / 72
    min_in    = min(width_in, height_in)

    if   min_in < 3:   return 600   # petite page / carte de visite
    elif min_in < 12:  return 300   # A4 (8.27 in), lettre (8.5 in), A3 (11.69 in)
    else:              return 200   # grand format A2+


def render_page(pdf_path: str, page_index: int) -> tuple[np.ndarray, dict]:
    """
    Rend une page PDF en tableau numpy RGB uint8 (H, W, 3).
    DPI adaptatif selon la taille physique. Thread-safe (pas de shared state).
    """
    with fitz.open(pdf_path) as doc:
        page = doc[page_index]
        dpi  = _adaptive_dpi(page)
        mat  = fitz.Matrix(dpi / 72, dpi / 72)
        pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        # .copy() nécessaire : les bytes pix.samples sont libérés à la fermeture du doc
        arr  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        ).copy()
        total = len(doc)

    meta = {
        "dpi":             (dpi, dpi),
        "mode_original":   "RGB",
        "bands":           ("R", "G", "B"),
        "bit_depth":       24,
        "shape":           arr.shape,
        "pdf_page":        page_index,
        "pdf_total_pages": total,
    }
    return arr, meta


def render_thumb(pdf_path: str, page_index: int, size: int = 56):
    """
    Retourne une PIL.Image miniature de la page (appelé depuis le thread UI).
    """
    from PIL import Image

    with fitz.open(pdf_path) as doc:
        page = doc[page_index]
        pix  = page.get_pixmap(dpi=72, colorspace=fitz.csRGB)
        # .copy() via frombuffer avant que doc se ferme
        samples = bytes(pix.samples)

    img = Image.frombytes("RGB", (pix.width, pix.height), samples)
    img.thumbnail((size, size), Image.LANCZOS)
    return img
