# Copyright (c) 2026 M. Tendeng — MIT License (see LICENSE)
"""
Moteur OCR adaptatif — sélectionne automatiquement Tesseract ou EasyOCR
selon la taille de l'image et le script détecté (OSD Tesseract).
"""

import time
import sys
import os
import numpy as np
import pytesseract
from PIL import Image

from .image_loader import load_image
from .preprocessing import (
    upscale_if_small, to_grayscale,
    apply_clahe, apply_gaussian_blur,
    invert_if_dark, binarize,
    deskew, binarize_sauvola,
)
from .ocr_module import run_ocr, _assemble_tess_data

# ── Mappings script OSD → langue ──────────────────────────────────────────

SCRIPT_TO_TESSERACT: dict[str, str] = {
    "Latin": "fra+eng",
}

SCRIPT_TO_EASYOCR: dict[str, list[str]] = {
    "Latin":      ["fr", "en"],
    "Arabic":     ["ar"],
    "Han":        ["ch_sim", "en"],
    "Cyrillic":   ["ru", "en"],
    "Devanagari": ["hi"],
    "Bengali":    ["bn"],
    "Thai":       ["th"],
}

# Codes Tesseract (ISO 639-2) → codes EasyOCR (ISO 639-1)
TESS_TO_EASYOCR: dict[str, str] = {
    "fra": "fr",  "eng": "en",  "ara": "ar",
    "chi_sim": "ch_sim",        "rus": "ru",
    "hin": "hi",  "ben": "bn",  "tha": "th",
}

_NON_LATIN = {"Arabic", "Han", "Devanagari", "Bengali", "Thai", "Cyrillic"}

# ── Cache EasyOCR Reader (instanciation PyTorch coûteuse) ─────────────────

_easyocr_cache: dict = {}


def _get_easyocr_reader(lang_list: list[str]):
    key = tuple(sorted(lang_list))
    if key not in _easyocr_cache:
        import easyocr  # import paresseux — PyTorch seulement si nécessaire
        print(f"[EasyOCR] Chargement du modèle pour {lang_list}…", flush=True)
        _easyocr_cache[key] = easyocr.Reader(lang_list, gpu=False, verbose=False)
    return _easyocr_cache[key]


# ── Classe principale ──────────────────────────────────────────────────────

class DynamicOCR:
    """
    Sélectionne et exécute automatiquement le meilleur moteur OCR pour chaque image.

    Règles de sélection :
      - Script Latin + conf ≥ 2.0 + min_dim ≥ 150px → Tesseract (rapide)
      - Toute autre condition (image trop petite, script non-Latin, image trop uniforme,
        OSD échoue) → EasyOCR (robuste, multilingue)
    """

    def __init__(self, default_lang: str = "fra+eng", min_conf: int = 60):
        self.default_lang = default_lang
        self.min_conf = min_conf

    def analyze_image(self, mat: np.ndarray) -> dict:
        """
        Analyse les caractéristiques d'une image RGB numpy.
        Retourne un dict avec min_dim, mean_brightness, std_dev,
        script_detected, script_conf, orientation.
        """
        h, w = mat.shape[:2]
        gray = mat.mean(axis=2) if mat.ndim == 3 else mat
        mean_brightness = float(gray.mean())
        std_dev = float(gray.std())

        # Détection script/orientation via Tesseract OSD
        script_detected = "unknown"
        script_conf = 0.0
        orientation = 0
        try:
            pil = Image.fromarray(mat)
            osd = pytesseract.image_to_osd(
                pil, lang="osd",
                output_type=pytesseract.Output.DICT
            )
            script_detected = osd.get("script", "unknown")
            script_conf = float(osd.get("script_conf", 0.0))
            orientation = int(osd.get("orientation", 0))
        except Exception:
            pass  # OSD échoue sur petites images ou images sans texte → fallback EasyOCR

        return {
            "min_dim": min(h, w),
            "width": w,
            "height": h,
            "mean_brightness": round(mean_brightness, 1),
            "std_dev": round(std_dev, 1),
            "script_detected": script_detected,
            "script_conf": round(script_conf, 2),
            "orientation": orientation,
        }

    def select_engine(self, analysis: dict) -> str:
        """
        Retourne 'tesseract' ou 'easyocr' selon l'analyse de l'image.

        Stratégie : Tesseract par défaut (preprocessing gère dark/noisy via
        invert_if_dark + CLAHE). EasyOCR uniquement pour scripts non-Latin
        confirmés ou images trop petites pour Tesseract.
        """
        if analysis["min_dim"] < 150:
            return "easyocr"
        if analysis["script_detected"] in _NON_LATIN and analysis["script_conf"] >= 3.0:
            return "easyocr"
        return "tesseract"

    def process(
        self,
        image_path: str,
        force_engine: str | None = None,
        force_lang: str | None = None,
    ) -> dict:
        """
        Traite une image depuis son chemin.

        Args:
            image_path  : chemin vers l'image
            force_engine: 'tesseract' | 'easyocr' | None (auto)
            force_lang  : override langue (ex: 'fra', 'fra+eng')

        Returns:
            dict unifié avec path, engine, lang, text, confidence,
            word_count, processing_time_s, image_size, script_detected, error
        """
        t0 = time.time()
        result: dict = {
            "path": image_path,
            "engine": None,
            "lang": None,
            "text": "",
            "confidence": 0.0,
            "word_count": 0,
            "processing_time_s": 0.0,
            "image_size": [0, 0],
            "script_detected": "unknown",
            "error": None,
        }

        try:
            mat, _ = load_image(image_path)
            result["image_size"] = [mat.shape[1], mat.shape[0]]

            analysis = self.analyze_image(mat)
            result["script_detected"] = analysis["script_detected"]

            engine = force_engine if force_engine else self.select_engine(analysis)
            result["engine"] = engine

            if engine == "tesseract":
                lang = force_lang or SCRIPT_TO_TESSERACT.get(
                    analysis["script_detected"], self.default_lang
                )
                result["lang"] = lang
                text, confidence = self._run_tesseract(mat, lang)
            else:
                raw_lang = SCRIPT_TO_EASYOCR.get(
                    analysis["script_detected"], ["fr", "en"]
                )
                if force_lang:
                    # Traduire codes Tesseract (fra, eng…) → codes EasyOCR (fr, en…)
                    raw_lang = [
                        TESS_TO_EASYOCR.get(c, c)
                        for c in force_lang.replace("+", " ").split()
                    ]
                result["lang"] = "+".join(raw_lang)
                text, confidence = self._run_easyocr(mat, raw_lang)

            result["text"] = text
            result["confidence"] = round(confidence, 1)
            result["word_count"] = len(text.split()) if text else 0

        except Exception as exc:
            result["error"] = str(exc)

        result["processing_time_s"] = round(time.time() - t0, 3)
        return result

    # ── Moteurs privés ────────────────────────────────────────────────────

    # ── Helpers langue ────────────────────────────────────────────────────

    def _tess_lang(self, analysis: dict, force_lang: str | None) -> str:
        return force_lang or SCRIPT_TO_TESSERACT.get(
            analysis["script_detected"], self.default_lang
        )

    def _easy_lang(self, analysis: dict, force_lang: str | None) -> list[str]:
        raw = SCRIPT_TO_EASYOCR.get(analysis["script_detected"], ["fr", "en"])
        if force_lang:
            raw = [TESS_TO_EASYOCR.get(c, c)
                   for c in force_lang.replace("+", " ").split()]
        return raw

    # ── Méthode acceptant un array pré-chargé ────────────────────────────

    def process_array(
        self,
        mat:          np.ndarray,
        meta:         dict,
        force_engine: str | None = None,
        force_lang:   str | None = None,
    ) -> dict:
        """
        Identique à process() mais accepte un tableau numpy pré-chargé
        (évite load_image() redondant pour les pages PDF).
        """
        t0 = time.time()
        result: dict = {
            "engine": None, "lang": None,
            "text": "", "confidence": 0.0, "word_count": 0,
            "processing_time_s": 0.0,
            "image_size": (mat.shape[1], mat.shape[0]),
            "script_detected": "unknown",
            "error": None,
        }
        try:
            dpi    = meta.get("dpi", (300, 300))[0]
            is_pdf = "pdf_page" in meta
            # PSM 3 (auto layout) pour PDFs multi-colonnes, PSM 6 pour images simples
            psm    = 3 if is_pdf else 6

            analysis = self.analyze_image(mat)
            result["script_detected"] = analysis["script_detected"]

            engine = force_engine or self.select_engine(analysis)
            result["engine"] = engine

            if engine == "tesseract":
                lang = self._tess_lang(analysis, force_lang)
                result["lang"] = lang
                text, confidence = self._run_tesseract_array(mat, lang,
                                                             dpi=dpi, psm=psm)
            else:
                lang_list = self._easy_lang(analysis, force_lang)
                result["lang"] = "+".join(lang_list)
                text, confidence = self._run_easyocr(mat, lang_list)

            result["text"]       = text
            result["confidence"] = round(confidence, 1)
            result["word_count"] = len(text.split()) if text else 0

        except Exception as exc:
            result["error"] = str(exc)

        result["processing_time_s"] = round(time.time() - t0, 3)
        return result

    # ── Moteurs privés ────────────────────────────────────────────────────

    def _run_tesseract(self, mat: np.ndarray, lang: str) -> tuple[str, float]:
        """Pipeline Tesseract pour images simples (PSM 6, DPI 300, Sauvola)."""
        return self._run_tesseract_array(mat, lang, dpi=300, psm=6)

    def _run_tesseract_array(
        self, mat: np.ndarray, lang: str, dpi: int = 300, psm: int = 6
    ) -> tuple[str, float]:
        """
        Pipeline complet : upscale → gray → CLAHE → blur → inversion → deskew → Sauvola.
        Remplace Otsu par Sauvola (meilleur sur documents scannés à éclairage inégal).
        """
        mat    = upscale_if_small(mat)
        gray   = to_grayscale(mat)
        gray   = apply_clahe(gray)
        gray   = apply_gaussian_blur(gray)
        gray   = invert_if_dark(gray)
        gray   = deskew(gray)
        binary = binarize_sauvola(gray, dpi=dpi)

        data = pytesseract.image_to_data(
            Image.fromarray(binary),
            lang=lang,
            config=f"--oem 1 --psm {psm} --dpi {dpi}",
            output_type=pytesseract.Output.DICT,
        )
        return _assemble_tess_data(data, self.min_conf)

    def _run_easyocr(self, mat: np.ndarray, lang_list: list[str]) -> tuple[str, float]:
        """Pipeline EasyOCR (image RGB numpy → texte + confiance)."""
        reader = _get_easyocr_reader(lang_list)
        results = reader.readtext(mat, detail=1)
        lines, confs = [], []
        for _, text, conf in results:
            if conf >= self.min_conf / 100:
                lines.append(text)
                confs.append(conf * 100)
        text = "\n".join(lines).strip()
        mean_conf = float(np.mean(confs)) if confs else 0.0
        return text, mean_conf


# ── Expansion des inputs (niveau module → testable) ───────────────────────

from pathlib import Path

SUPPORTED_CLI = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp", ".pdf"}


def expand_inputs(paths: list[str], recursive: bool = False) -> list[str]:
    """
    Résout une liste de chemins (fichiers et/ou dossiers) en liste plate de
    fichiers compatibles. Avertit sur stderr pour les chemins invalides.
    """
    result = []
    for p in paths:
        entry = Path(p)
        if entry.is_dir():
            iterator = entry.rglob("*") if recursive else entry.iterdir()
            found_subdir = False
            for f in sorted(iterator):
                if f.is_dir() and not recursive:
                    found_subdir = True
                    continue
                if f.is_file() and f.suffix.lower() in SUPPORTED_CLI:
                    result.append(str(f))
            if found_subdir and not recursive:
                print(
                    f"[INFO] Sous-dossiers ignorés dans {p} (utilisez --recursive)",
                    file=sys.stderr,
                )
        elif entry.is_file():
            if entry.suffix.lower() in SUPPORTED_CLI:
                result.append(str(entry))
            else:
                print(f"[IGNORÉ] Extension non supportée : {p}", file=sys.stderr)
        else:
            print(f"[IGNORÉ] Introuvable : {p}", file=sys.stderr)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────


def cli_main() -> None:
    """Point d'entrée de la commande `amanda-cli`."""
    import argparse
    import csv
    import json

    from amanda_ocr.ocr_pipeline import process_pdf
    from amanda_ocr.pdf_renderer import page_count
    from amanda_ocr.models import ProcessingItem, PageSource

    # Garantit l'affichage Unicode (arabe, chinois, accents) quel que soit le terminal
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="OCR — images, PDFs, dossiers")
    parser.add_argument("inputs", nargs="+", help="Fichiers image/PDF ou dossiers")
    parser.add_argument("--engine", choices=["auto", "tesseract", "easyocr"], default="auto")
    parser.add_argument("--lang",       default=None, help="Override langue (ex: fra+eng)")
    parser.add_argument("--min-conf",   type=int, default=60)
    parser.add_argument("--workers",    type=int, default=4, help="Workers parallèles PDF")
    parser.add_argument("--output-dir", default=None, help="Dossier de sortie des résultats")
    parser.add_argument("--format",     choices=["txt", "json", "csv"], default="txt")
    parser.add_argument("--recursive",  action="store_true", help="Parcourir les sous-dossiers")
    parser.add_argument("--quiet",      action="store_true", help="Métadonnées seulement, pas le texte")
    parser.add_argument("--verbose",    action="store_true", help="JSON complet des métadonnées")
    args = parser.parse_args()

    expanded = expand_inputs(args.inputs, recursive=args.recursive)
    if not expanded:
        print("[ERREUR] Aucun fichier valide trouvé.", file=sys.stderr)
        sys.exit(1)

    engine       = DynamicOCR(min_conf=args.min_conf)
    force_engine = None if args.engine == "auto" else args.engine
    all_items: list[ProcessingItem] = []
    has_error = False

    def _dict_to_item(result: dict, path: str) -> ProcessingItem:
        item = ProcessingItem(source_path=path, page_index=None,
                              source_type=PageSource.IMAGE)
        item.update_from_dict(result)
        return item

    def _print_item(item: ProcessingItem) -> None:
        if item.error:
            print(f"\n── {item.display_name} ──", file=sys.stderr)
            print(f"[ERREUR] {item.error}", file=sys.stderr)
            return
        print(f"\n── {item.display_name} ──")
        if args.verbose:
            print(json.dumps({
                "engine":            item.engine,
                "lang":              item.lang,
                "confidence":        item.confidence,
                "word_count":        item.word_count,
                "processing_time_s": item.processing_time_s,
                "script_detected":   item.script_detected,
                "image_size":        list(item.image_size) if item.image_size else [],
            }, indent=2, ensure_ascii=False))
        if not args.quiet:
            print(item.text or "(aucun texte extrait)")
        print(f"[{item.engine} · {item.lang or '—'} · "
              f"{item.confidence:.0f}% · {item.processing_time_s:.2f}s]")

    def _save_output(items: list[ProcessingItem], outdir: str, fmt: str) -> None:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        if fmt == "txt":
            for item in items:
                (Path(outdir) / f"{item.export_stem}.txt").write_text(
                    item.text or "", encoding="utf-8"
                )
        elif fmt == "json":
            records = [
                {
                    "display_name": item.display_name, "engine": item.engine,
                    "lang": item.lang, "confidence": item.confidence,
                    "word_count": item.word_count, "text": item.text,
                    "error": item.error,
                }
                for item in items
            ]
            (Path(outdir) / "ocr_results.json").write_text(
                json.dumps({"total": len(records), "results": records},
                           indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        elif fmt == "csv":
            with open(Path(outdir) / "ocr_results.csv", "w",
                      newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "display_name", "engine", "lang",
                    "confidence", "word_count", "error", "text",
                ])
                w.writeheader()
                for item in items:
                    w.writerow({
                        "display_name": item.display_name,
                        "engine":       item.engine,
                        "lang":         item.lang,
                        "confidence":   item.confidence,
                        "word_count":   item.word_count,
                        "error":        item.error or "",
                        "text":         (item.text or "")[:500],
                    })
        print(f"\n[Export {fmt.upper()}] → {outdir}", file=sys.stderr)

    # ── Traitement ────────────────────────────────────────────────────────
    for path in expanded:
        if path.lower().endswith(".pdf"):
            print(f"\n── PDF : {os.path.basename(path)} ──", file=sys.stderr)
            n_pages = page_count(path)

            def _progress(item: ProcessingItem, _n=n_pages) -> None:
                print(f"  p.{item.page_index + 1}/{_n} …",
                      file=sys.stderr, end="\r", flush=True)

            items = process_pdf(
                path, engine,
                max_workers=args.workers,
                force_engine=force_engine,
                force_lang=args.lang,
                progress_cb=_progress,
            )
            print("", file=sys.stderr)  # efface la ligne de progression
        else:
            result = engine.process(path, force_engine=force_engine,
                                    force_lang=args.lang)
            items = [_dict_to_item(result, path)]

        for item in items:
            if item.error:
                has_error = True
            _print_item(item)
            all_items.append(item)

    # ── Export fichier ────────────────────────────────────────────────────
    if args.output_dir and all_items:
        _save_output(all_items, args.output_dir, args.format)

    sys.exit(2 if has_error else 0)


if __name__ == "__main__":
    import sys as _sys, os as _os
    # Support exécution directe : ajoute la racine du projet au path
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    cli_main()
