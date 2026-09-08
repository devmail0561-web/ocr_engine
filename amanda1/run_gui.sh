#!/usr/bin/env bash
# Copyright (c) 2026 M. Tendeng — MIT License (see LICENSE)
# Lance l'interface graphique OCR Engine.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/../.venv/bin/python3"
exec "$PYTHON" -m amanda_ocr.gui "$@"
