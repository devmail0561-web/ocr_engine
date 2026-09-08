"""
Interface graphique OCR Engine — customtkinter dark theme.
Lancer : /home/virus-one/Documents/projet_orc/.venv/bin/python core/gui.py
"""

import os
import csv
import json
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import customtkinter as ctk
from PIL import Image, ImageDraw

from .dynamic_ocr import DynamicOCR
from .models import PageSource, ProcessingItem
from .pdf_renderer import render_page, render_thumb
from .ocr_pipeline import enumerate_pdf_items, HAS_NATIVE
from .text_extractor import classify_page

# ── Thème ─────────────────────────────────────────────────────────────────

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

C_BG      = "#0d1117"
C_SIDEBAR = "#161b22"
C_SURFACE = "#1c2128"
C_CARD    = "#21262d"
C_BORDER  = "#30363d"
C_ACCENT  = "#388bfd"
C_ACCENT2 = "#1f6feb"
C_TEXT    = "#e6edf3"
C_MUTED   = "#8b949e"
C_SUCCESS = "#3fb950"
C_WARN    = "#d29922"
C_ERROR   = "#f85149"

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp", ".pdf"}

LANG_OPTIONS = [
    "Auto",
    "Français + Anglais",
    "Français",
    "Anglais",
    "Arabe",
    "Chinois simplifié",
    "Russe",
    "Hindi",
]

LANG_VALUES: dict[str, str | None] = {
    "Auto":               None,
    "Français + Anglais": "fra+eng",
    "Français":           "fra",
    "Anglais":            "eng",
    "Arabe":              "ara",
    "Chinois simplifié":  "chi_sim",
    "Russe":              "rus",
    "Hindi":              "hin",
}

# ── Utilitaires ───────────────────────────────────────────────────────────

def _make_thumb(item: ProcessingItem, size: int = 56) -> ctk.CTkImage | None:
    """Génère un thumbnail carré arrondi pour une image ou une page PDF."""
    try:
        if item.source_type in (PageSource.PDF_SCAN, PageSource.PDF_NATIVE):
            pil_img = render_thumb(item.source_path, item.page_index, size=size)
        else:
            pil_img = Image.open(item.source_path).convert("RGB")
            pil_img.thumbnail((size, size), Image.LANCZOS)
        bg = Image.new("RGB", (size, size), (33, 38, 45))
        offset = ((size - pil_img.width) // 2, (size - pil_img.height) // 2)
        bg.paste(pil_img, offset)
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=8, fill=255)
        result = Image.new("RGBA", (size, size))
        result.paste(bg, mask=mask)
        return ctk.CTkImage(light_image=result, dark_image=result, size=(size, size))
    except Exception:
        return None


def _truncate(text: str, n: int = 40) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


# ── Widget carte image ────────────────────────────────────────────────────

class ImageCard(ctk.CTkFrame):
    """Une ligne dans la liste d'images : thumbnail + nom + badge statut."""

    STATUS_STYLE = {
        "waiting":    ("○ En attente",  C_MUTED),
        "processing": ("◌ Traitement…", C_WARN),
        "done":       ("● Terminé",     C_SUCCESS),
        "error":      ("✕ Erreur",      C_ERROR),
    }

    def __init__(self, parent, item: ProcessingItem, on_click, **kwargs):
        super().__init__(
            parent,
            fg_color=C_CARD,
            corner_radius=10,
            border_width=1,
            border_color=C_BORDER,
            **kwargs,
        )
        self.uid = item.uid
        self._selected = False

        thumb = _make_thumb(item)
        thumb_label = ctk.CTkLabel(self, image=thumb, text="", width=56)
        thumb_label.grid(row=0, column=0, rowspan=2, padx=(10, 8), pady=8)

        name_label = ctk.CTkLabel(
            self,
            text=_truncate(item.display_name, 30),
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=C_TEXT,
            anchor="w",
        )
        name_label.grid(row=0, column=1, sticky="w", pady=(8, 0))

        self._badge = ctk.CTkLabel(
            self,
            text="○ En attente",
            font=ctk.CTkFont(size=11),
            text_color=C_MUTED,
            anchor="w",
        )
        self._badge.grid(row=1, column=1, sticky="w", pady=(0, 8))
        self.columnconfigure(1, weight=1)

        # Clic sur toute la carte
        for widget in (self, thumb_label, name_label, self._badge):
            widget.bind("<Button-1>", lambda e, uid=item.uid: on_click(uid))
            widget.bind("<Enter>", self._on_hover)
            widget.bind("<Leave>", self._on_leave)

    def set_status(self, status: str):
        text, color = self.STATUS_STYLE.get(status, ("", C_MUTED))
        self._badge.configure(text=text, text_color=color)

    def set_selected(self, selected: bool):
        self._selected = selected
        color = C_ACCENT2 if selected else C_CARD
        border = C_ACCENT if selected else C_BORDER
        self.configure(fg_color=color, border_color=border)

    def _on_hover(self, _):
        if not self._selected:
            self.configure(fg_color="#2d333b")

    def _on_leave(self, _):
        if not self._selected:
            self.configure(fg_color=C_CARD)


# ── Application principale ────────────────────────────────────────────────

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("OCR Engine")
        self.geometry("1140x720")
        self.minsize(920, 600)
        self.configure(fg_color=C_BG)

        self._engine = DynamicOCR()
        self._items:        list[ProcessingItem]  = []
        self._cards:        dict[str, ImageCard]  = {}  # uid → card widget
        self._selected_uid: str | None            = None
        self._queue:        queue.Queue           = queue.Queue()
        self._processing = False

        self._build_layout()
        self.after(150, self._poll_queue)

    # ── Construction de l'interface ───────────────────────────────────────

    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main()

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color=C_SIDEBAR)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(2, weight=1)

        # ── Logo
        logo_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        logo_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(22, 6))
        ctk.CTkLabel(
            logo_frame,
            text="◈",
            font=ctk.CTkFont(size=26),
            text_color=C_ACCENT,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(
            logo_frame,
            text="OCR Engine",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=C_TEXT,
        ).pack(side="left")

        # ── Boutons Ajouter
        btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_frame.grid(row=1, column=0, padx=14, pady=(4, 8), sticky="ew")
        btn_frame.columnconfigure((0, 1), weight=1)

        ctk.CTkButton(
            btn_frame,
            text="＋  Images",
            height=38,
            corner_radius=8,
            fg_color=C_ACCENT,
            hover_color=C_ACCENT2,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._add_images,
        ).grid(row=0, column=0, padx=(0, 4), sticky="ew")

        ctk.CTkButton(
            btn_frame,
            text="📁  Dossier",
            height=38,
            corner_radius=8,
            fg_color=C_SURFACE,
            hover_color=C_CARD,
            border_width=1,
            border_color=C_BORDER,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._add_folder,
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

        # ── Liste scrollable
        list_label = ctk.CTkLabel(
            sidebar, text="FICHIERS",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=C_MUTED,
        )
        list_label.grid(row=2, column=0, sticky="nw", padx=18, pady=(2, 0))

        self._image_list_frame = ctk.CTkScrollableFrame(
            sidebar,
            fg_color="transparent",
            scrollbar_fg_color=C_SIDEBAR,
            scrollbar_button_color=C_BORDER,
            scrollbar_button_hover_color=C_MUTED,
        )
        self._image_list_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(18, 4))

        # ── Séparateur + Export
        ctk.CTkFrame(sidebar, height=1, fg_color=C_BORDER).grid(
            row=3, column=0, sticky="ew", padx=14, pady=6
        )
        export_label = ctk.CTkLabel(
            sidebar, text="EXPORT",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=C_MUTED,
        )
        export_label.grid(row=4, column=0, sticky="w", padx=18, pady=(2, 6))

        export_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        export_frame.grid(row=5, column=0, padx=14, pady=(0, 16), sticky="ew")
        export_frame.columnconfigure((0, 1, 2), weight=1)

        for col, (label, cmd) in enumerate([
            ("TXT",  self._export_txt),
            ("JSON", self._export_json),
            ("CSV",  self._export_csv),
        ]):
            ctk.CTkButton(
                export_frame, text=label, height=32,
                corner_radius=6, fg_color=C_SURFACE,
                border_width=1, border_color=C_BORDER,
                hover_color=C_CARD,
                font=ctk.CTkFont(size=12),
                command=cmd,
            ).grid(row=0, column=col, padx=2, sticky="ew")

        sidebar.columnconfigure(0, weight=1)

    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color=C_BG, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew", padx=0)
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # ── Header chips
        self._header = ctk.CTkFrame(main, fg_color=C_SURFACE, corner_radius=12, height=60)
        self._header.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 10))
        self._header.grid_propagate(False)
        self._header.grid_columnconfigure(0, weight=1)

        self._filename_label = ctk.CTkLabel(
            self._header,
            text="Sélectionnez un fichier dans la liste",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=C_TEXT,
            anchor="w",
        )
        self._filename_label.grid(row=0, column=0, padx=16, pady=(10, 2), sticky="w")

        # Indicateur mode natif (Rust) ou Python
        _mode_text  = "* NATIF"  if HAS_NATIVE else "PYTHON"
        _mode_color = C_SUCCESS  if HAS_NATIVE else C_WARN
        ctk.CTkLabel(
            self._header,
            text=_mode_text,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=_mode_color,
        ).grid(row=0, column=1, padx=(0, 16), pady=(10, 2), sticky="e")

        self._chips_frame = ctk.CTkFrame(self._header, fg_color="transparent")
        self._chips_frame.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="w")

        # ── Textbox résultat
        self._textbox = ctk.CTkTextbox(
            main,
            font=ctk.CTkFont(family="Courier New", size=13),
            fg_color=C_SURFACE,
            text_color=C_TEXT,
            corner_radius=12,
            border_width=1,
            border_color=C_BORDER,
            wrap="word",
            state="disabled",
        )
        self._textbox.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 10))

        # ── Réglages + Bouton OCR
        settings = ctk.CTkFrame(main, fg_color="transparent")
        settings.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 6))
        settings.columnconfigure(4, weight=1)

        ctk.CTkLabel(
            settings, text="Moteur",
            font=ctk.CTkFont(size=12), text_color=C_MUTED,
        ).grid(row=0, column=0, padx=(0, 6))

        self._engine_var = ctk.StringVar(value="Auto")
        ctk.CTkOptionMenu(
            settings,
            values=["Auto", "Tesseract", "EasyOCR"],
            variable=self._engine_var,
            fg_color=C_SURFACE,
            button_color=C_ACCENT,
            button_hover_color=C_ACCENT2,
            dropdown_fg_color=C_CARD,
            corner_radius=8,
            width=140,
            font=ctk.CTkFont(size=13),
        ).grid(row=0, column=1, padx=(0, 16))

        ctk.CTkLabel(
            settings, text="Langue",
            font=ctk.CTkFont(size=12), text_color=C_MUTED,
        ).grid(row=0, column=2, padx=(0, 6))

        self._lang_var = ctk.StringVar(value="Auto")
        ctk.CTkOptionMenu(
            settings,
            values=LANG_OPTIONS,
            variable=self._lang_var,
            fg_color=C_SURFACE,
            button_color=C_ACCENT,
            button_hover_color=C_ACCENT2,
            dropdown_fg_color=C_CARD,
            corner_radius=8,
            width=180,
            font=ctk.CTkFont(size=13),
        ).grid(row=0, column=3, padx=(0, 16))

        self._btn_run = ctk.CTkButton(
            settings,
            text="▶   OCR tout",
            height=42,
            corner_radius=10,
            fg_color=C_ACCENT,
            hover_color=C_ACCENT2,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._start_batch,
        )
        self._btn_run.grid(row=0, column=4, sticky="e")

        # ── Barre de progression
        progress_frame = ctk.CTkFrame(main, fg_color="transparent")
        progress_frame.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 14))
        progress_frame.columnconfigure(0, weight=1)

        self._progress = ctk.CTkProgressBar(
            progress_frame,
            progress_color=C_ACCENT,
            fg_color=C_SURFACE,
            corner_radius=6,
            height=8,
        )
        self._progress.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self._progress.set(0)

        self._progress_label = ctk.CTkLabel(
            progress_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=C_MUTED,
            width=70,
            anchor="e",
        )
        self._progress_label.grid(row=0, column=1)

    # ── Gestion des images ────────────────────────────────────────────────

    def _expand_path(self, path: str) -> list[ProcessingItem]:
        """Convertit un chemin en liste de ProcessingItem (1 pour image, N pour PDF)."""
        if path.lower().endswith(".pdf"):
            return enumerate_pdf_items(path)
        return [ProcessingItem(
            source_path=path,
            page_index=None,
            source_type=PageSource.IMAGE,
        )]

    def _add_items(self, items: list[ProcessingItem]) -> int:
        """Ajoute des items s'ils ne sont pas déjà présents. Retourne le nombre ajouté."""
        existing_uids = {i.uid for i in self._items}
        added = 0
        for item in items:
            if item.uid not in existing_uids:
                self._items.append(item)
                self._add_card(item)
                existing_uids.add(item.uid)
                added += 1
        return added

    def _add_images(self):
        paths = filedialog.askopenfilenames(
            title="Sélectionner des images ou PDF",
            filetypes=[
                ("Images & PDF", "*.png *.jpg *.jpeg *.tiff *.bmp *.webp *.pdf"),
                ("Tous les fichiers", "*.*"),
            ],
        )
        for path in paths:
            self._add_items(self._expand_path(path))
        self._update_run_button()

    def _add_folder(self):
        folder = filedialog.askdirectory(title="Sélectionner un dossier")
        if not folder:
            return
        added = 0
        for p in sorted(Path(folder).iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                added += self._add_items(self._expand_path(str(p)))
        self._update_run_button()
        if added == 0:
            messagebox.showinfo(
                "Dossier vide",
                "Aucune image ou PDF compatible trouvé dans ce dossier.",
            )

    def _add_card(self, item: ProcessingItem):
        card = ImageCard(
            self._image_list_frame,
            item=item,
            on_click=self._show_result,
        )
        card.pack(fill="x", padx=4, pady=3)
        self._cards[item.uid] = card

    def _show_result(self, uid: str):
        # Désélectionner l'ancien
        if self._selected_uid and self._selected_uid in self._cards:
            self._cards[self._selected_uid].set_selected(False)
        self._selected_uid = uid
        self._cards[uid].set_selected(True)

        item = next((i for i in self._items if i.uid == uid), None)
        if item is None:
            return

        self._filename_label.configure(text=item.display_name)

        # Chips
        for widget in self._chips_frame.winfo_children():
            widget.destroy()

        if item.engine or item.error:
            chips = []
            if item.engine:
                chips.append((item.engine.upper(), C_ACCENT))
            if item.lang:
                chips.append((item.lang, C_MUTED))
            if item.confidence:
                chips.append((
                    f"{item.confidence:.0f}%",
                    C_SUCCESS if item.confidence > 70 else C_WARN,
                ))
            if item.processing_time_s:
                chips.append((f"{item.processing_time_s:.2f}s", C_MUTED))
            if item.word_count:
                chips.append((f"{item.word_count} mots", C_MUTED))
            if item.script_detected and item.script_detected != "unknown":
                chips.append((item.script_detected, C_MUTED))

            for text, color in chips:
                chip = ctk.CTkFrame(self._chips_frame, fg_color=C_CARD, corner_radius=20)
                chip.pack(side="left", padx=(0, 6))
                ctk.CTkLabel(
                    chip, text=text,
                    font=ctk.CTkFont(size=11), text_color=color,
                ).pack(padx=10, pady=2)

        # Textbox
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        if item.error:
            self._textbox.insert("1.0", f"[Erreur] {item.error}")
            self._textbox.configure(text_color=C_ERROR)
        elif item.engine or item.text:
            self._textbox.insert("1.0", item.text or "(aucun texte extrait)")
            self._textbox.configure(text_color=C_TEXT)
        self._textbox.configure(state="disabled")

    # ── Batch OCR ─────────────────────────────────────────────────────────

    def _start_batch(self):
        if not self._items:
            messagebox.showinfo("Aucun fichier", "Ajoutez des images ou PDFs avant de lancer l'OCR.")
            return
        if self._processing:
            return

        self._processing = True
        self._btn_run.configure(state="disabled", text="⏳  OCR en cours…")
        self._progress.set(0)
        self._progress_label.configure(text=f"0 / {len(self._items)}")

        engine_choice = self._engine_var.get().lower()
        force_engine  = None if engine_choice == "auto" else engine_choice
        lang_raw      = LANG_VALUES.get(self._lang_var.get())

        threading.Thread(
            target=self._run_batch_thread,
            args=(force_engine, lang_raw),
            daemon=True,
        ).start()

    def _run_batch_thread(self, force_engine, force_lang):
        total = len(self._items)
        for i, item in enumerate(self._items):
            self._queue.put(("progress", i, item.uid))

            if item.source_type in (PageSource.PDF_SCAN, PageSource.PDF_NATIVE):
                # Vérifier d'abord le texte natif
                is_native, text = classify_page(item.source_path, item.page_index)
                if is_native:
                    item.source_type  = PageSource.PDF_NATIVE
                    item.text         = text
                    item.confidence   = 100.0
                    item.engine       = "embedded"
                    item.lang         = ""
                    item.word_count   = len(text.split())
                else:
                    mat, meta = render_page(item.source_path, item.page_index)
                    result    = self._engine.process_array(
                        mat, meta,
                        force_engine=force_engine,
                        force_lang=force_lang,
                    )
                    item.update_from_dict(result)
            else:
                result = self._engine.process(
                    item.source_path,
                    force_engine=force_engine,
                    force_lang=force_lang,
                )
                item.update_from_dict(result)

            self._queue.put(("result", i, item.uid))

        self._queue.put(("done", total, None))

    def _poll_queue(self):
        total = len(self._items) or 1
        while not self._queue.empty():
            msg = self._queue.get()
            kind = msg[0]
            if kind == "progress":
                _, _i, uid = msg
                if uid in self._cards:
                    self._cards[uid].set_status("processing")
            elif kind == "result":
                _, idx, uid = msg
                item = next((i for i in self._items if i.uid == uid), None)
                if item and uid in self._cards:
                    status = "error" if item.error else "done"
                    self._cards[uid].set_status(status)
                    done = idx + 1
                    self._progress.set(done / total)
                    self._progress_label.configure(text=f"{done} / {total}")
                    if self._selected_uid == uid:
                        self._show_result(uid)
            elif kind == "done":
                self._on_batch_done()
        self.after(150, self._poll_queue)

    def _on_batch_done(self):
        self._processing = False
        n = len(self._items)
        self._btn_run.configure(state="normal", text=f"▶   OCR tout — {n} fichier{'s' if n > 1 else ''}")
        self._progress.set(1)
        self._progress_label.configure(text=f"{n} / {n}  ✓")

    def _update_run_button(self):
        n = len(self._items)
        if n == 0:
            self._btn_run.configure(text="▶   OCR tout")
        else:
            self._btn_run.configure(text=f"▶   OCR tout — {n} fichier{'s' if n > 1 else ''}")

    # ── Export ────────────────────────────────────────────────────────────

    def _processed_items(self) -> list[ProcessingItem]:
        return [i for i in self._items if i.engine or i.error]

    def _check_results(self) -> bool:
        if not self._processed_items():
            messagebox.showinfo("Aucun résultat", "Lancez l'OCR avant d'exporter.")
            return False
        return True

    def _export_txt(self):
        if not self._check_results():
            return
        outdir = filedialog.askdirectory(title="Choisir le dossier de sortie TXT")
        if not outdir:
            return
        count = 0
        for item in self._processed_items():
            dest = Path(outdir) / f"{item.export_stem}.txt"
            dest.write_text(item.text or "", encoding="utf-8")
            count += 1
        messagebox.showinfo("Export TXT", f"{count} fichier(s) sauvegardé(s) dans\n{outdir}")

    def _export_json(self):
        if not self._check_results():
            return
        dest = filedialog.asksaveasfilename(
            title="Enregistrer en JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="ocr_results.json",
        )
        if not dest:
            return
        done = self._processed_items()
        records = [
            {
                "path":              item.source_path,
                "page":              item.page_index,
                "display_name":      item.display_name,
                "engine":            item.engine,
                "lang":              item.lang,
                "text":              item.text,
                "confidence":        item.confidence,
                "word_count":        item.word_count,
                "processing_time_s": item.processing_time_s,
                "image_size":        list(item.image_size) if item.image_size else [],
                "script_detected":   item.script_detected,
                "error":             item.error,
            }
            for item in done
        ]
        summary = {
            "total":    len(records),
            "engines": {
                "tesseract": sum(1 for r in records if r["engine"] == "tesseract"),
                "easyocr":   sum(1 for r in records if r["engine"] == "easyocr"),
                "embedded":  sum(1 for r in records if r["engine"] == "embedded"),
            },
            "mean_confidence": round(
                sum(r["confidence"] for r in records) / len(records), 1
            ) if records else 0,
        }
        payload = {"batch_summary": summary, "results": records}
        Path(dest).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        messagebox.showinfo("Export JSON", f"Sauvegardé : {dest}")

    def _export_csv(self):
        if not self._check_results():
            return
        dest = filedialog.asksaveasfilename(
            title="Enregistrer en CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="ocr_results.csv",
        )
        if not dest:
            return
        fieldnames = [
            "display_name", "engine", "lang", "script_detected",
            "word_count", "confidence", "processing_time_s", "error", "text",
        ]
        with open(dest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in self._processed_items():
                writer.writerow({
                    "display_name":      item.display_name,
                    "engine":            item.engine,
                    "lang":              item.lang,
                    "script_detected":   item.script_detected,
                    "word_count":        item.word_count,
                    "confidence":        item.confidence,
                    "processing_time_s": item.processing_time_s,
                    "error":             item.error or "",
                    "text":              (item.text or "")[:500],
                })
        messagebox.showinfo("Export CSV", f"Sauvegardé : {dest}")


# ── Entrée ────────────────────────────────────────────────────────────────

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
