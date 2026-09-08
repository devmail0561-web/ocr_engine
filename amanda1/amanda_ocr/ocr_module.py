# Copyright (c) 2026 M. Tendeng — MIT License (see LICENSE)
import numpy as np
import pytesseract
from PIL import Image


def _assemble_tess_data(data: dict, min_conf: int) -> tuple[str, float]:
    """
    Reconstruit le texte et la confiance à partir d'un dict pytesseract.image_to_data.
    Fonction partagée entre run_ocr et DynamicOCR._run_tesseract_array.
    """
    confs = []
    prev_block, block_order, lines = None, [], {}
    for i in range(len(data["text"])):
        conf = int(data["conf"][i])
        if conf < min_conf:
            continue
        word = data["text"][i].strip()
        if not word:
            continue
        confs.append(conf)
        block = data["block_num"][i]
        key   = (block, data["par_num"][i], data["line_num"][i])
        if block != prev_block:
            if block not in block_order:
                block_order.append(block)
            prev_block = block
        lines.setdefault(key, []).append(word)

    paragraphs = []
    for block in block_order:
        blines = [" ".join(lines[k]) for k in sorted(lines) if k[0] == block]
        if blines:
            paragraphs.append("\n".join(blines))
    text      = "\n\n".join(paragraphs).strip()
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return text, mean_conf


def run_ocr(image, lang="fra+eng", config=r"--oem 1 --psm 6 --dpi 300", min_conf=60):
    """
    Lance l'OCR sur une matrice numpy (RGB ou grayscale) ou une PIL.Image.
    Retourne le texte reconstruit ligne par ligne, filtré par score de confiance.

    lang     : packs de langue Tesseract (ex: "fra+eng", "fra", "eng")
    config   : options Tesseract (--oem 1 = LSTM, --psm 6 = bloc uniforme, --dpi 300)
    min_conf : seuil de confiance minimum (0-100), les mots en dessous sont ignorés
    """
    try:
        pil_img = image if isinstance(image, Image.Image) else Image.fromarray(image)
        data = pytesseract.image_to_data(
            pil_img, lang=lang, config=config,
            output_type=pytesseract.Output.DICT
        )
    except pytesseract.pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract introuvable. Installez-le avec : "
            "sudo apt install tesseract-ocr tesseract-ocr-fra"
        )

    text, _ = _assemble_tess_data(data, min_conf)
    return text
