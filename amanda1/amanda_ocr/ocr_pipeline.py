"""
Orchestrateur de traitement PDF haute performance.

Stratégie de parallélisme :
  - Pages Tesseract : ThreadPoolExecutor (le subprocess libère le GIL → vrai parallélisme)
  - Pages EasyOCR   : traitement sériel (PyTorch tient le GIL sur les tenseurs)
  - Pages natives   : extraction directe PyMuPDF (pas d'OCR)
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .models import PageSource, ProcessingItem
from .pdf_renderer import page_count, render_page
from .text_extractor import classify_page

try:
    import amanda_native as _native  # noqa: F401 — module Rust Phase 2
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False


def enumerate_pdf_items(pdf_path: str) -> list[ProcessingItem]:
    """Retourne la liste des ProcessingItem (un par page) avant traitement."""
    n = page_count(pdf_path)
    return [
        ProcessingItem(
            source_path=pdf_path,
            page_index=i,
            source_type=PageSource.PDF_SCAN,
        )
        for i in range(n)
    ]


def process_pdf(
    pdf_path:     str,
    engine,                        # DynamicOCR instance
    max_workers:  int = 4,
    force_engine: str | None = None,
    force_lang:   str | None = None,
    progress_cb=None,              # callable(item: ProcessingItem) | None
) -> list[ProcessingItem]:
    """
    Traite toutes les pages d'un PDF.

    Phase A (parallèle) : pages natives (embedded text) + pages Tesseract.
                          Retourne aussi mat/meta pour les pages EasyOCR différées.
    Phase B (sérielle)  : pages EasyOCR — réutilise le mat de Phase A (pas de double rendu).

    Retourne la liste triée par page_index.
    """
    n = page_count(pdf_path)
    results:       list[ProcessingItem | None] = [None] * n
    # (idx → (mat, meta)) pour les pages différées EasyOCR
    deferred: dict[int, tuple[np.ndarray, dict]] = {}

    def _process_one(
        idx: int,
    ) -> tuple[int, ProcessingItem | None, np.ndarray | None, dict | None]:
        """
        Retourne :
          (idx, item, None, None)        si la page est traitée (native ou Tesseract)
          (idx, None, mat, meta)         si la page doit aller en EasyOCR (Phase B)
        """
        # 1. Texte natif ?
        is_native, text = classify_page(pdf_path, idx)
        if is_native:
            item = ProcessingItem(
                source_path=pdf_path,
                page_index=idx,
                source_type=PageSource.PDF_NATIVE,
                text=text,
                confidence=100.0,
                engine="embedded",
                lang="",
                word_count=len(text.split()),
            )
            return idx, item, None, None

        # 2. Rendu + analyse
        mat, meta = render_page(pdf_path, idx)
        analysis  = engine.analyze_image(mat)
        selected  = force_engine or engine.select_engine(analysis)

        if selected == "easyocr":
            # Conserver mat pour Phase B — évite un double rendu
            return idx, None, mat, meta

        # 3. Tesseract (subprocess libère le GIL → parallélisme réel)
        result_dict = engine.process_array(
            mat, meta,
            force_engine="tesseract",
            force_lang=force_lang,
        )
        item = ProcessingItem(
            source_path=pdf_path,
            page_index=idx,
            source_type=PageSource.PDF_SCAN,
        )
        item.update_from_dict(result_dict)
        return idx, item, None, None

    # ── Phase A : Tesseract + embedded en parallèle ───────────────────────
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one, i): i for i in range(n)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                idx, item, mat, meta = future.result()
            except Exception as exc:
                item = ProcessingItem(
                    source_path=pdf_path,
                    page_index=idx,
                    source_type=PageSource.PDF_SCAN,
                    error=f"Erreur page {idx}: {exc}",
                )
                results[idx] = item
                if progress_cb:
                    progress_cb(item)
                continue

            if item is None:
                # Différé EasyOCR — mat conservé, pas de double rendu en Phase B
                deferred[idx] = (mat, meta)
            else:
                results[idx] = item
                if progress_cb:
                    progress_cb(item)

    # ── Phase B : EasyOCR en série (PyTorch tient le GIL) ────────────────
    for idx in sorted(deferred):
        mat, meta = deferred[idx]
        try:
            result_dict = engine.process_array(
                mat, meta,
                force_engine="easyocr",
                force_lang=force_lang,
            )
        except Exception as exc:
            result_dict = {"error": f"EasyOCR page {idx}: {exc}"}
        item = ProcessingItem(
            source_path=pdf_path,
            page_index=idx,
            source_type=PageSource.PDF_SCAN,
        )
        item.update_from_dict(result_dict)
        results[idx] = item
        if progress_cb:
            progress_cb(item)

    # Remplacer les éventuels None résiduels (page sans traitement)
    for i in range(n):
        if results[i] is None:
            results[i] = ProcessingItem(
                source_path=pdf_path,
                page_index=i,
                source_type=PageSource.PDF_SCAN,
                error="Page non traitée",
            )

    return results  # type: ignore[return-value]
