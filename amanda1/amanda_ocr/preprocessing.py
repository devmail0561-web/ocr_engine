import cv2
import numpy as np


def deskew(gray: np.ndarray) -> np.ndarray:
    """
    Corrige l'inclinaison d'un document scanné.
    Binarise d'abord (Otsu) pour isoler les pixels de texte avant de calculer l'angle,
    évitant que les ombres et dégradés faussent la détection.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # np.where retourne (row_indices, col_indices) = (y, x).
    # cv2.minAreaRect attend des points (x, y) → on passe (cols, rows).
    ys, xs = np.where(binary > 0)
    coords  = np.column_stack((xs, ys))
    if len(coords) < 300:
        return gray  # pas assez de pixels texte pour estimer un angle fiable

    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    # minAreaRect retourne un angle dans [-90, 0) ; on recentre autour de 0
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.3:
        return gray  # correction inférieure à 0.3° → ignorée

    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def binarize_sauvola(gray: np.ndarray, dpi: int = 300, k: float = 0.2) -> np.ndarray:
    """
    Binarisation adaptative Sauvola — supérieure à Otsu pour les documents scannés
    avec éclairage inégal, ombres ou pliures.

    La fenêtre locale est proportionnelle au DPI : window = max(15, dpi // 12).
    T(x,y) = mean * (1 + k * (std / 128 - 1))
    """
    window = max(15, dpi // 12)
    if window % 2 == 0:
        window += 1  # doit être impair pour cv2.boxFilter symétrique

    gray_f  = gray.astype(np.float32)
    mean    = cv2.boxFilter(gray_f, -1, (window, window))
    sq_mean = cv2.boxFilter(gray_f ** 2, -1, (window, window))
    std     = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))
    threshold = mean * (1.0 + k * (std / 128.0 - 1.0))
    return (gray_f > threshold).astype(np.uint8) * 255


def to_grayscale(rgb_matrix):
    """Convertit une image RGB en niveaux de gris."""
    if rgb_matrix is None:
        raise ValueError("to_grayscale: l'image d'entrée est None")
    if not isinstance(rgb_matrix, np.ndarray) or rgb_matrix.ndim < 2:
        raise ValueError(f"to_grayscale: tableau numpy 2D ou 3D attendu, reçu {type(rgb_matrix)}")
    return cv2.cvtColor(rgb_matrix, cv2.COLOR_RGB2GRAY)


def binarize(gray_matrix, threshold=128, method="otsu"):
    """
    Binarise une image en niveaux de gris.

    method="otsu"     → seuil calculé automatiquement (recommandé)
    method="adaptive" → seuillage adaptatif par bloc (éclairage inégal)
    method="fixed"    → seuil fixe (valeur threshold)
    """
    if gray_matrix is None:
        raise ValueError("binarize: l'image d'entrée est None")
    if method == "otsu":
        _, binary = cv2.threshold(gray_matrix, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray_matrix, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    elif method == "fixed":
        _, binary = cv2.threshold(gray_matrix, threshold, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"binarize: méthode inconnue '{method}'. Valeurs: otsu, adaptive, fixed")
    return binary


def apply_gaussian_blur(gray_matrix, kernel_size=3):
    """Flou gaussien pour réduire le bruit et l'anti-aliasing avant binarisation."""
    if kernel_size % 2 == 0:
        raise ValueError("apply_gaussian_blur: kernel_size doit être impair")
    return cv2.GaussianBlur(gray_matrix, (kernel_size, kernel_size), 0)


def invert_if_dark(gray_matrix, dark_threshold=127):
    """
    Inverse l'image si le fond est sombre (ex: captures dark-mode, texte clair sur fond noir).
    Doit être appelée AVANT binarize().
    """
    if gray_matrix.mean() < dark_threshold:
        return cv2.bitwise_not(gray_matrix)
    return gray_matrix


def upscale_if_small(image, min_dim=1000, factor=2):
    """
    Redimensionne l'image ×factor si la plus petite dimension est < min_dim.
    Améliore la reconnaissance Tesseract sur petites images (screenshots < 1000px).
    """
    h, w = image.shape[:2]
    if min(h, w) < min_dim:
        return cv2.resize(image, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)
    return image


def apply_clahe(gray_matrix, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Normalisation locale du contraste (CLAHE) — améliore les images sous-exposées."""
    if gray_matrix.dtype != np.uint8:
        raise ValueError("apply_clahe: entrée doit être uint8 (image en niveaux de gris)")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_matrix)
