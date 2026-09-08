# amanda1 — OCR Engine

Moteur OCR adaptatif pour images et PDFs. Sélectionne automatiquement Tesseract ou EasyOCR
selon le contenu de chaque page (script détecté, taille, contraste).

---

## Prérequis

- Python 3.12 dans le venv `/home/virus-one/Documents/projet_orc/.venv`
- Tesseract 5.x avec les packs de langue installés :
  ```bash
  sudo apt install tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng
  ```
- PyMuPDF installé dans le venv :
  ```bash
  /home/virus-one/Documents/projet_orc/.venv/bin/python3 -m pip install PyMuPDF
  ```

---

## Démarrage rapide

```bash
cd ~/Documents/projet_orc/amanda1

# Interface graphique
./run_gui.sh

# Ligne de commande
./run_cli.sh --help
```

---

## Interface graphique

```bash
./run_gui.sh
```

- **Ajouter** des images (PNG, JPG, TIFF, BMP, WebP) ou des **PDFs** via les boutons de la barre latérale, ou en glissant un dossier entier.
- Pour les PDFs, chaque page apparaît comme une entrée distincte dans la liste.
- **Moteurs** : Auto (défaut), Tesseract, EasyOCR
- **Langues** : Auto, Français + Anglais, Français, Anglais, Arabe, Chinois, Russe, Hindi
- **Export** : TXT (un fichier par page/image), JSON (batch complet), CSV
- L'indicateur en haut à droite affiche `PYTHON` (mode standard) ou `* NATIF` (module Rust actif)

---

## Ligne de commande

```
./run_cli.sh [options] inputs [inputs ...]
```

### Arguments positionnels

| Argument | Description |
|---|---|
| `inputs` | Un ou plusieurs fichiers (image ou PDF) et/ou dossiers |

### Options

| Option | Défaut | Description |
|---|---|---|
| `--engine` | `auto` | `auto` \| `tesseract` \| `easyocr` |
| `--lang` | auto | `fra+eng` \| `fra` \| `eng` \| `ara` \| `chi_sim` \| `rus` \| `hin` |
| `--min-conf` | `60` | Seuil de confiance minimum (0–100) |
| `--workers` | `4` | Workers parallèles pour le traitement des PDFs |
| `--output-dir` | — | Dossier où sauvegarder les résultats |
| `--format` | `txt` | Format d'export : `txt` \| `json` \| `csv` (avec `--output-dir`) |
| `--recursive` | off | Parcourir les sous-dossiers |
| `--quiet` | off | Affiche uniquement les métadonnées, pas le texte |
| `--verbose` | off | Affiche les métadonnées complètes en JSON par item |

### Codes de sortie

| Code | Signification |
|---|---|
| `0` | Succès, tous les items traités sans erreur |
| `1` | Aucun fichier valide trouvé |
| `2` | Au moins un item en erreur (les autres ont été traités) |

### Exemples

```bash
# Image seule
./run_cli.sh img/Capture.png

# PDF — traitement parallèle, texte natif extrait directement si disponible
./run_cli.sh rapport.pdf --lang fra

# Dossier entier
./run_cli.sh ~/Documents/scans/ --engine tesseract

# Mix fichiers + dossier
./run_cli.sh img/Capture.png rapport.pdf ~/Documents/scans/ --verbose

# Grand PDF — feedback page par page, sans afficher le texte
./run_cli.sh gros_rapport.pdf --quiet --workers 8

# Exporter en TXT (un fichier par page)
./run_cli.sh docs/ --output-dir ./resultats --format txt

# Exporter tout en JSON
./run_cli.sh img/ rapport.pdf --output-dir ./out --format json

# Sous-dossiers inclus
./run_cli.sh ~/Documents/ --recursive --output-dir ./out --format csv
```

---

## Installation

```bash
cd ~/Documents/projet_orc/amanda1
/home/virus-one/Documents/projet_orc/.venv/bin/python3 -m pip install -e ".[dev]"
```

Cela installe le package `amanda-ocr` en mode éditable et rend disponibles les commandes
`amanda-cli` et `amanda-gui` dans le venv.

---

## Architecture

```
amanda1/
├── pyproject.toml          Configuration du package Python
├── run_gui.sh              Lanceur GUI
├── run_cli.sh              Lanceur CLI
└── amanda_ocr/             Package Python
    ├── __init__.py         Exports publics : DynamicOCR, ProcessingItem, PageSource
    ├── gui.py              Interface graphique (customtkinter)
    ├── dynamic_ocr.py      Moteur OCR adaptatif + CLI (amanda-cli)
    ├── ocr_pipeline.py     Orchestrateur PDF parallèle
    ├── pdf_renderer.py     Rendu PDF → numpy (PyMuPDF, DPI adaptatif)
    ├── text_extractor.py   Détection texte natif PDF (sans OCR)
    ├── models.py           ProcessingItem, PageSource
    ├── preprocessing.py    Pipeline image : deskew, Sauvola, CLAHE…
    ├── ocr_module.py       Wrapper Tesseract bas niveau
    ├── image_loader.py     Chargement images (PIL → numpy)
    ├── fast_processor.py   Wrapper Tesseract simplifié
    └── tests.py            Suite de tests (59 tests)
```

### Pipeline PDF

```
PDF
 ├── Page avec texte vectoriel → text_extractor.classify_page()
 │       → texte extrait directement (< 1 s / document, engine="embedded")
 │
 └── Page scannée → pdf_renderer.render_page() [DPI adaptatif]
         → preprocessing : upscale → gray → CLAHE → blur → deskew → Sauvola
         → Tesseract (parallèle, ThreadPoolExecutor)
         → EasyOCR   (sériel, GIL PyTorch)
```

### DPI adaptatif

| Taille physique | DPI |
|---|---|
| < 3 pouces (cartes) | 600 |
| 3–12 pouces (A4, A3, lettre) | 300 |
| > 12 pouces (grand format) | 200 |

---

## Tests

```bash
cd ~/Documents/projet_orc/amanda1
/home/virus-one/Documents/projet_orc/.venv/bin/python3 -m pytest amanda_ocr/tests.py -v
# ou avec unittest :
/home/virus-one/Documents/projet_orc/.venv/bin/python3 -m unittest amanda_ocr.tests -v
```

59 tests couvrant : preprocessing, OCR, image loader, moteur adaptatif,
rendu PDF, extraction texte natif, pipeline, modèles de données, expansion CLI.

---

## Phase 2 — Module natif Rust (optionnel)

Remplace le preprocessing Python et les appels subprocess Tesseract par du code Rust
compilé (Sauvola natif, Tesseract C API directe, parallélisme Rayon sans GIL).

```bash
# Prérequis
sudo apt install libtesseract-dev libleptonica-dev
/home/virus-one/Documents/projet_orc/.venv/bin/python3 -m pip install maturin

# Build
cd amanda_ocr/native
maturin develop --release
```

Une fois le module compilé, l'indicateur `* NATIF` apparaît dans la GUI
et le pipeline utilise automatiquement le moteur Rust.
