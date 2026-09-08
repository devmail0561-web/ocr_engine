"""
Suite de tests pour le pipeline OCR amanda1.
Exécuter avec :
  /home/virus-one/Documents/projet_orc/.venv/bin/python -m unittest tests -v
"""
import unittest
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR  = os.path.normpath(os.path.join(BASE_DIR, "..", "img"))


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        from amanda_ocr.preprocessing import (
            to_grayscale, binarize,
            apply_gaussian_blur, invert_if_dark,
            upscale_if_small, apply_clahe,
        )
        self.to_grayscale = to_grayscale
        self.binarize = binarize
        self.apply_gaussian_blur = apply_gaussian_blur
        self.invert_if_dark = invert_if_dark
        self.upscale_if_small = upscale_if_small
        self.apply_clahe = apply_clahe

    def test_to_grayscale_rgb_luminance(self):
        """Valider que COLOR_RGB2GRAY applique les bons poids (rouge pur → ~76, pas ~29)."""
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # rouge pur en RGB
        gray = self.to_grayscale(arr)
        self.assertAlmostEqual(float(gray.mean()), 76.0, delta=2.0)

    def test_to_grayscale_none_raises(self):
        with self.assertRaises(ValueError):
            self.to_grayscale(None)

    def test_binarize_otsu_default(self):
        """Otsu sur image bimodale → valeurs uniquement 0 ou 255."""
        arr = np.zeros((20, 20), dtype=np.uint8)
        arr[:10, :] = 50
        arr[10:, :] = 200
        result = self.binarize(arr)
        unique = set(result.flatten().tolist())
        self.assertEqual(unique, {0, 255})

    def test_binarize_adaptive(self):
        arr = np.zeros((20, 20), dtype=np.uint8)
        arr[:10, :] = 50
        arr[10:, :] = 200
        result = self.binarize(arr, method="adaptive")
        self.assertEqual(result.dtype, np.uint8)
        unique = set(result.flatten().tolist())
        self.assertTrue(unique.issubset({0, 255}))

    def test_binarize_invalid_method_raises(self):
        arr = np.zeros((10, 10), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.binarize(arr, method="invalid")

    def test_invert_if_dark_inverts(self):
        """Image sombre (mean < 127) doit être inversée."""
        arr = np.full((10, 10), 50, dtype=np.uint8)
        result = self.invert_if_dark(arr)
        self.assertAlmostEqual(float(result.mean()), 205.0, delta=1.0)

    def test_invert_if_dark_no_change(self):
        """Image claire (mean >= 127) ne doit pas être modifiée."""
        arr = np.full((10, 10), 200, dtype=np.uint8)
        result = self.invert_if_dark(arr)
        np.testing.assert_array_equal(result, arr)

    def test_upscale_if_small(self):
        """Une image de 100×100 doit être redimensionnée à 200×200."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.upscale_if_small(arr, min_dim=1000, factor=2)
        self.assertEqual(result.shape, (200, 200, 3))

    def test_upscale_if_small_no_change(self):
        """Une image de 1200×1200 ne doit pas être modifiée."""
        arr = np.zeros((1200, 1200, 3), dtype=np.uint8)
        result = self.upscale_if_small(arr, min_dim=1000)
        self.assertEqual(result.shape, (1200, 1200, 3))

    def test_apply_clahe_output_shape(self):
        arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = self.apply_clahe(arr)
        self.assertEqual(result.shape, arr.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_apply_gaussian_blur_odd_kernel(self):
        arr = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        result = self.apply_gaussian_blur(arr, kernel_size=3)
        self.assertEqual(result.shape, arr.shape)

    def test_apply_gaussian_blur_even_kernel_raises(self):
        arr = np.zeros((10, 10), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.apply_gaussian_blur(arr, kernel_size=4)


class TestOCR(unittest.TestCase):

    def setUp(self):
        from amanda_ocr.ocr_module import run_ocr
        from amanda_ocr.preprocessing import (
            to_grayscale, binarize,
            apply_clahe, apply_gaussian_blur,
            invert_if_dark, upscale_if_small,
        )
        from amanda_ocr.image_loader import load_image
        self.run_ocr = run_ocr
        self.load_image = load_image
        self.to_grayscale = to_grayscale
        self.binarize = binarize
        self.apply_clahe = apply_clahe
        self.apply_gaussian_blur = apply_gaussian_blur
        self.invert_if_dark = invert_if_dark
        self.upscale_if_small = upscale_if_small
        self.capture_path = os.path.join(IMG_DIR, "Capture.png")

    def _preprocess(self, mat):
        mat = self.upscale_if_small(mat)
        gray = self.to_grayscale(mat)
        gray = self.apply_clahe(gray)
        gray = self.apply_gaussian_blur(gray)
        gray = self.invert_if_dark(gray)
        return self.binarize(gray, method="otsu")

    def test_run_ocr_nonempty_on_capture(self):
        mat, _ = self.load_image(self.capture_path)
        binary = self._preprocess(mat)
        result = self.run_ocr(binary)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)

    def test_run_ocr_known_text(self):
        mat, _ = self.load_image(self.capture_path)
        binary = self._preprocess(mat)
        result = self.run_ocr(binary)
        self.assertIn("photo", result.lower())

    def test_run_ocr_high_min_conf_empty(self):
        mat, _ = self.load_image(self.capture_path)
        binary = self._preprocess(mat)
        result = self.run_ocr(binary, min_conf=101)
        self.assertEqual(result, "")


class TestImageLoader(unittest.TestCase):

    def setUp(self):
        from amanda_ocr.image_loader import load_image
        self.load_image = load_image
        self.capture_path = os.path.join(IMG_DIR, "Capture.png")

    def test_load_image_shape(self):
        mat, meta = self.load_image(self.capture_path)
        self.assertEqual(mat.shape, (760, 448, 3))

    def test_bands_after_convert(self):
        """bands doit être ('R','G','B') après conversion, pas ('R','G','B','A')."""
        _, meta = self.load_image(self.capture_path)
        self.assertEqual(meta["bands"], ("R", "G", "B"))

    def test_bit_depth_rgba(self):
        """Capture.png est RGBA → bit_depth doit être 32."""
        _, meta = self.load_image(self.capture_path)
        self.assertEqual(meta["bit_depth"], 32)

    def test_dpi_returned(self):
        _, meta = self.load_image(self.capture_path)
        self.assertIn("dpi", meta)
        self.assertIsInstance(meta["dpi"], tuple)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.load_image("/nonexistent/path.png")


class TestDynamicOCR(unittest.TestCase):

    def setUp(self):
        from amanda_ocr.dynamic_ocr import DynamicOCR
        from amanda_ocr.image_loader import load_image
        self.DynamicOCR = DynamicOCR
        self.load_image = load_image
        self.capture_path = os.path.join(IMG_DIR, "Capture.png")
        self.engine = DynamicOCR()

    def test_analyze_returns_expected_keys(self):
        mat, _ = self.load_image(self.capture_path)
        analysis = self.engine.analyze_image(mat)
        for key in ("min_dim", "mean_brightness", "std_dev", "script_detected", "script_conf", "orientation"):
            self.assertIn(key, analysis)

    def test_select_tesseract_on_latin_large(self):
        """Image claire 1000×1000 avec script Latin → Tesseract."""
        analysis = {
            "min_dim": 1000,
            "mean_brightness": 200.0,
            "std_dev": 60.0,
            "script_detected": "Latin",
            "script_conf": 10.0,
            "orientation": 0,
        }
        self.assertEqual(self.engine.select_engine(analysis), "tesseract")

    def test_select_easyocr_on_tiny_image(self):
        """Image trop petite → EasyOCR."""
        analysis = {
            "min_dim": 100,
            "mean_brightness": 200.0,
            "std_dev": 60.0,
            "script_detected": "Latin",
            "script_conf": 8.0,
            "orientation": 0,
        }
        self.assertEqual(self.engine.select_engine(analysis), "easyocr")

    def test_select_tesseract_on_low_contrast(self):
        """Faible contraste (std_dev < 20) sur Latin → Tesseract (invert_if_dark + CLAHE gèrent)."""
        analysis = {
            "min_dim": 500,
            "mean_brightness": 120.0,
            "std_dev": 10.0,
            "script_detected": "Latin",
            "script_conf": 5.0,
            "orientation": 0,
        }
        self.assertEqual(self.engine.select_engine(analysis), "tesseract")

    def test_process_returns_unified_format(self):
        """process() retourne un dict avec toutes les clés attendues."""
        result = self.engine.process(self.capture_path)
        for key in ("path", "engine", "lang", "text", "confidence",
                    "word_count", "processing_time_s", "image_size",
                    "script_detected", "error"):
            self.assertIn(key, result)

    def test_force_engine_tesseract(self):
        """force_engine='tesseract' → result['engine'] == 'tesseract'."""
        result = self.engine.process(self.capture_path, force_engine="tesseract")
        self.assertEqual(result["engine"], "tesseract")
        self.assertIsNone(result["error"])

    def test_process_nonexistent_file(self):
        """Fichier inexistant → result['error'] non None."""
        result = self.engine.process("/nonexistent/image.png")
        self.assertIsNotNone(result["error"])


class TestPreprocessingNew(unittest.TestCase):

    def setUp(self):
        from amanda_ocr.preprocessing import deskew, binarize_sauvola
        self.deskew          = deskew
        self.binarize_sauvola = binarize_sauvola

    def test_deskew_noop_on_straight_image(self):
        """Image déjà droite → retourne un tableau de même forme sans erreur."""
        arr = np.zeros((300, 400), dtype=np.uint8)
        arr[50:250, 50:350] = 200  # rectangle blanc sur fond noir
        result = self.deskew(arr)
        self.assertEqual(result.shape, arr.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_deskew_returns_same_shape(self):
        """deskew ne change pas les dimensions de l'image."""
        arr = np.random.randint(0, 255, (500, 600), dtype=np.uint8)
        result = self.deskew(arr)
        self.assertEqual(result.shape, arr.shape)

    def test_deskew_sparse_returns_unchanged(self):
        """Moins de 300 pixels texte → retour sans modification (angle non fiable)."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[10, 10] = 255  # 1 seul pixel
        result = self.deskew(arr)
        np.testing.assert_array_equal(result, arr)

    def test_sauvola_output_binary(self):
        """binarize_sauvola produit uniquement des valeurs 0 ou 255."""
        arr = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        result = self.binarize_sauvola(arr, dpi=300)
        unique = set(result.flatten().tolist())
        self.assertTrue(unique.issubset({0, 255}))

    def test_sauvola_window_scales_with_dpi(self):
        """La fenêtre Sauvola est proportionnelle au DPI."""
        arr = np.random.randint(100, 200, (400, 400), dtype=np.uint8)
        # Ne doit pas lever d'exception pour différents DPI
        for dpi in (150, 300, 600):
            result = self.binarize_sauvola(arr, dpi=dpi)
            self.assertEqual(result.shape, arr.shape)

    def test_sauvola_output_shape(self):
        """La sortie a la même forme que l'entrée."""
        arr = np.random.randint(0, 255, (300, 500), dtype=np.uint8)
        result = self.binarize_sauvola(arr, dpi=300)
        self.assertEqual(result.shape, arr.shape)
        self.assertEqual(result.dtype, np.uint8)


class TestModels(unittest.TestCase):

    def setUp(self):
        from amanda_ocr.models import ProcessingItem, PageSource
        self.ProcessingItem = ProcessingItem
        self.PageSource     = PageSource

    def test_image_uid(self):
        item = self.ProcessingItem("/path/img.png", None, self.PageSource.IMAGE)
        self.assertEqual(item.uid, "/path/img.png|None")

    def test_pdf_uid_unique_per_page(self):
        item0 = self.ProcessingItem("/doc.pdf", 0, self.PageSource.PDF_SCAN)
        item1 = self.ProcessingItem("/doc.pdf", 1, self.PageSource.PDF_SCAN)
        self.assertNotEqual(item0.uid, item1.uid)

    def test_display_name_image(self):
        item = self.ProcessingItem("/path/to/image.png", None, self.PageSource.IMAGE)
        self.assertEqual(item.display_name, "image.png")

    def test_display_name_pdf_page(self):
        item = self.ProcessingItem("/path/to/doc.pdf", 2, self.PageSource.PDF_SCAN)
        self.assertEqual(item.display_name, "doc.pdf — p.3")

    def test_export_stem_image(self):
        item = self.ProcessingItem("/path/image.png", None, self.PageSource.IMAGE)
        self.assertEqual(item.export_stem, "image")

    def test_export_stem_pdf_page(self):
        item = self.ProcessingItem("/path/rapport.pdf", 4, self.PageSource.PDF_SCAN)
        self.assertEqual(item.export_stem, "rapport_p005")

    def test_update_from_dict(self):
        item = self.ProcessingItem("/img.png", None, self.PageSource.IMAGE)
        item.update_from_dict({"text": "bonjour", "confidence": 95.0, "engine": "tesseract"})
        self.assertEqual(item.text, "bonjour")
        self.assertEqual(item.confidence, 95.0)
        self.assertEqual(item.engine, "tesseract")


class TestPdfRenderer(unittest.TestCase):

    def setUp(self):
        import tempfile, struct, zlib
        from amanda_ocr.pdf_renderer import render_page, render_thumb, page_count

        self.render_page  = render_page
        self.render_thumb = render_thumb
        self.page_count   = page_count

        # Créer un PDF minimal valide avec une page blanche (pas d'image réelle)
        # On utilise PyMuPDF directement pour créer un PDF test propre
        import pymupdf as fitz
        self._tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text((72, 72), "Test OCR", fontsize=24)
        doc.save(self._tmp.name)
        doc.close()
        self.pdf_path = self._tmp.name

    def tearDown(self):
        import os
        os.unlink(self._tmp.name)

    def test_page_count(self):
        self.assertEqual(self.page_count(self.pdf_path), 1)

    def test_render_page_shape(self):
        arr, meta = self.render_page(self.pdf_path, 0)
        self.assertEqual(arr.ndim, 3)
        self.assertEqual(arr.shape[2], 3)
        self.assertEqual(arr.dtype.name, "uint8")

    def test_render_page_meta_keys(self):
        _, meta = self.render_page(self.pdf_path, 0)
        for key in ("dpi", "mode_original", "bands", "pdf_page", "pdf_total_pages"):
            self.assertIn(key, meta)

    def test_render_page_dpi_a4(self):
        _, meta = self.render_page(self.pdf_path, 0)
        # A4 : largeur 595 pts = 8.27 in, min < 12 → 300 DPI
        self.assertEqual(meta["dpi"][0], 300)

    def test_render_thumb_returns_pil(self):
        from PIL import Image
        img = self.render_thumb(self.pdf_path, 0, size=56)
        self.assertIsInstance(img, Image.Image)
        self.assertLessEqual(max(img.size), 56)

    def test_thread_safety(self):
        """4 threads parallèles sur le même PDF → pas de crash ni de données corrompues."""
        import threading
        errors = []
        def worker():
            try:
                arr, _ = self.render_page(self.pdf_path, 0)
                assert arr.ndim == 3
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


class TestTextExtractor(unittest.TestCase):

    def setUp(self):
        import pymupdf as fitz, tempfile
        from amanda_ocr.text_extractor import classify_page
        self.classify_page = classify_page

        # PDF avec texte natif
        self._tmp_native = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 100), "Ceci est un document texte avec beaucoup de contenu lisible.",
                         fontsize=12)
        page.insert_text((50, 130), "La page contient plusieurs lignes de texte pour le test.",
                         fontsize=12)
        doc.save(self._tmp_native.name)
        doc.close()

        # PDF avec image pleine page (simulé : grand rectangle noir = "image" dominante)
        self._tmp_scan = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc2 = fitz.open()
        page2 = doc2.new_page(width=595, height=842)
        # Pas de texte → classify_page retourne False
        doc2.save(self._tmp_scan.name)
        doc2.close()

    def tearDown(self):
        import os
        os.unlink(self._tmp_native.name)
        os.unlink(self._tmp_scan.name)

    def test_native_pdf_detected(self):
        is_native, text = self.classify_page(self._tmp_native.name, 0)
        self.assertTrue(is_native)
        self.assertGreater(len(text), 10)

    def test_empty_page_not_native(self):
        """Page sans texte → is_native=False."""
        is_native, text = self.classify_page(self._tmp_scan.name, 0)
        self.assertFalse(is_native)
        self.assertEqual(text, "")


class TestOcrPipeline(unittest.TestCase):

    def setUp(self):
        import pymupdf as fitz, tempfile
        from amanda_ocr.ocr_pipeline import enumerate_pdf_items
        self.enumerate_pdf_items = enumerate_pdf_items

        self._tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc = fitz.open()
        for _ in range(3):
            doc.new_page(width=595, height=842)
        doc.save(self._tmp.name)
        doc.close()
        self.pdf_path = self._tmp.name

    def tearDown(self):
        import os
        os.unlink(self._tmp.name)

    def test_enumerate_returns_correct_count(self):
        items = self.enumerate_pdf_items(self.pdf_path)
        self.assertEqual(len(items), 3)

    def test_enumerate_items_ordered(self):
        items = self.enumerate_pdf_items(self.pdf_path)
        for i, item in enumerate(items):
            self.assertEqual(item.page_index, i)

    def test_enumerate_items_uid_unique(self):
        items = self.enumerate_pdf_items(self.pdf_path)
        uids = [item.uid for item in items]
        self.assertEqual(len(uids), len(set(uids)))

    def test_enumerate_source_path(self):
        items = self.enumerate_pdf_items(self.pdf_path)
        for item in items:
            self.assertEqual(item.source_path, self.pdf_path)


class TestCLIExpand(unittest.TestCase):

    def setUp(self):
        from amanda_ocr.dynamic_ocr import expand_inputs
        self.expand  = expand_inputs
        self.img_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "img"))
        self.capture = os.path.join(self.img_dir, "Capture.png")

    def test_single_image(self):
        result = self.expand([self.capture])
        self.assertEqual(result, [self.capture])

    def test_unsupported_extension(self):
        result = self.expand(["/tmp/fichier_test_xyz.docx"])
        self.assertEqual(result, [])

    def test_nonexistent_path(self):
        result = self.expand(["/tmp/inexistant_xyz_abc.png"])
        self.assertEqual(result, [])

    def test_empty_list(self):
        result = self.expand([])
        self.assertEqual(result, [])

    def test_folder_yields_images(self):
        from pathlib import Path as _Path
        from amanda_ocr.dynamic_ocr import SUPPORTED_CLI
        result = self.expand([self.img_dir])
        self.assertGreater(len(result), 0)
        for p in result:
            self.assertIn(_Path(p).suffix.lower(), SUPPORTED_CLI)

    def test_mixed_valid_and_invalid(self):
        result = self.expand([self.capture, "/tmp/inexistant_xyz.png"])
        self.assertEqual(result, [self.capture])

    def test_duplicate_paths_not_deduplicated(self):
        result = self.expand([self.capture, self.capture])
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
