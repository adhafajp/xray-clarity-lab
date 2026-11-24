import time
import cv2
import numpy as np


def _validate_image(image):
    if not isinstance(image, np.ndarray):
        raise TypeError("Citra harus berupa numpy.ndarray.")
    if image.dtype != np.uint8 or image.ndim != 2 or min(image.shape, default=0) < 1:
        raise ValueError("Citra harus berupa grayscale 8-bit dengan dimensi tidak kosong.")


def _make_odd(n, min_val=3):
    n = max(int(n), min_val)
    return n if n % 2 == 1 else n + 1


def apply_median_filter(image, ksize=5):
    _validate_image(image)
    k = _make_odd(ksize, 3)
    t0 = time.perf_counter()
    out = cv2.medianBlur(image, k)
    ms = (time.perf_counter() - t0) * 1000.0
    return out, ms


def apply_bilateral_filter(image, diameter=9, sigma_color=75.0, sigma_space=75.0):
    _validate_image(image)
    d = max(int(diameter), 1)
    t0 = time.perf_counter()
    out = cv2.bilateralFilter(image, d, float(sigma_color), float(sigma_space))
    ms = (time.perf_counter() - t0) * 1000.0
    return out, ms


def apply_clahe(image, clip_limit=2.0, tile_grid_size=8):
    _validate_image(image)
    grid = max(int(tile_grid_size), 2)
    t0 = time.perf_counter()
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(grid, grid))
    out = clahe.apply(image)
    ms = (time.perf_counter() - t0) * 1000.0
    return out, ms


def run_all_methods(
    noisy_image,
    median_ksize=5,
    bilateral_d=9,
    bilateral_sigma_color=75.0,
    bilateral_sigma_space=75.0,
    clahe_clip_limit=2.0,
    clahe_grid_size=8,
):
    _validate_image(noisy_image)
    med_img, med_ms = apply_median_filter(noisy_image, median_ksize)
    bil_img, bil_ms = apply_bilateral_filter(
        noisy_image, bilateral_d, bilateral_sigma_color, bilateral_sigma_space
    )
    clahe_img, clahe_ms = apply_clahe(noisy_image, clahe_clip_limit, clahe_grid_size)

    return {
        "Median Filter": {"image": med_img, "elapsed_ms": med_ms},
        "Bilateral Filter": {"image": bil_img, "elapsed_ms": bil_ms},
        "CLAHE": {"image": clahe_img, "elapsed_ms": clahe_ms},
    }
