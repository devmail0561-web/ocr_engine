# Copyright (c) 2026 M. Tendeng — MIT License (see LICENSE)
"""amanda_ocr — moteur OCR adaptatif pour images et PDFs."""
from .dynamic_ocr import DynamicOCR
from .models import ProcessingItem, PageSource

__version__ = "0.1.0"
__all__ = ["DynamicOCR", "ProcessingItem", "PageSource"]
