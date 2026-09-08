# Copyright (c) 2026 M. Tendeng — MIT License (see LICENSE)
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class PageSource(Enum):
    IMAGE       = "image"
    PDF_SCAN    = "pdf_scan"    # page image-only → OCR requis
    PDF_NATIVE  = "pdf_native"  # page avec texte vectoriel → extraction directe


@dataclass
class ProcessingItem:
    # Identité (immuable après création)
    source_path: str
    page_index:  int | None  # None pour images simples
    source_type: PageSource

    # Résultat (rempli après traitement)
    text:              str        = ""
    confidence:        float      = 0.0
    processing_time_s: float      = 0.0
    engine:            str        = ""
    lang:              str        = ""
    error:             str | None = None
    word_count:        int        = 0
    image_size:        tuple      = field(default_factory=tuple)
    script_detected:   str        = "unknown"
    dpi_used:          int        = 0

    @property
    def display_name(self) -> str:
        if self.page_index is not None:
            return f"{Path(self.source_path).name} — p.{self.page_index + 1}"
        return Path(self.source_path).name

    @property
    def export_stem(self) -> str:
        stem = Path(self.source_path).stem
        if self.page_index is not None:
            return f"{stem}_p{self.page_index + 1:03d}"
        return stem

    @property
    def uid(self) -> str:
        return f"{self.source_path}|{self.page_index}"

    def update_from_dict(self, d: dict) -> None:
        for key in ("text", "confidence", "processing_time_s", "engine", "lang",
                    "error", "word_count", "image_size", "script_detected"):
            if key in d:
                setattr(self, key, d[key])
