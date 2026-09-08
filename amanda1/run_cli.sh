#!/usr/bin/env bash
# Copyright (c) 2026 M. Tendeng — MIT License (see LICENSE)
# Lance le moteur OCR en ligne de commande.
# Usage : ./run_cli.sh image.png [--engine auto|tesseract|easyocr] [--lang fra+eng]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/../.venv/bin/python3"
exec "$PYTHON" -m amanda_ocr.dynamic_ocr "$@"
