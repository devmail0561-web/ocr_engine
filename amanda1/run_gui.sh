#!/usr/bin/env bash
# Lance l'interface graphique OCR Engine.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/../.venv/bin/python3"
exec "$PYTHON" -m amanda_ocr.gui "$@"
