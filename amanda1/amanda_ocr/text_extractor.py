# Copyright (c) 2026 M. Tendeng — MIT License (see LICENSE)
import pymupdf as fitz


def classify_page(pdf_path: str, page_index: int) -> tuple[bool, str]:
    """
    Analyse les blocs d'une page PDF pour déterminer si elle contient du texte
    vectoriel exploitable (True) ou si elle nécessite un OCR (False).

    Retourne (is_native, text).
    """
    with fitz.open(pdf_path) as doc:
        page      = doc[page_index]
        page_area = page.rect.width * page.rect.height
        if page_area == 0:
            return False, ""

        # get_text("blocks") → list de tuples (x0, y0, x1, y2, text, block_no, block_type)
        # block_type : 0 = texte, 1 = image
        blocks = page.get_text("blocks")

    text_blocks  = [b for b in blocks if b[6] == 0]
    image_blocks = [b for b in blocks if b[6] == 1]

    full_text = " ".join(b[4].strip() for b in text_blocks).strip()
    if len(full_text) < 30:
        return False, ""

    text_area  = sum((b[2] - b[0]) * (b[3] - b[1]) for b in text_blocks)
    image_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in image_blocks)

    # Page scannée : image dominante (> 60%) ET texte minoritaire (< 20%)
    if image_area > 0.6 * page_area and text_area < 0.2 * page_area:
        return False, ""

    return True, full_text
