import cv2
import numpy as np


def _validate_pair(ref, target):
    if not isinstance(ref, np.ndarray) or not isinstance(target, np.ndarray):
        raise TypeError("Citra referensi dan hasil harus berupa numpy.ndarray.")
    if ref.dtype != np.uint8 or target.dtype != np.uint8 or ref.ndim != 2 or target.ndim != 2:
        raise ValueError("Citra harus berupa grayscale 8-bit.")
    if ref.shape != target.shape or min(ref.shape, default=0) < 1:
        raise ValueError("Citra referensi dan hasil harus memiliki dimensi sama dan tidak kosong.")


def compute_mse(ref, target):
    _validate_pair(ref, target)
    diff = cv2.absdiff(ref, target).astype(np.float32)
    return float(np.mean(diff * diff))


def compute_psnr(ref, target, max_val=255.0):
    mse = compute_mse(ref, target)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((max_val ** 2) / mse))


def compute_ssim(ref, target, data_range=255.0):
    _validate_pair(ref, target)
    # Perhitungan SSIM dengan window Gaussian 11x11 (sigma=1.5)
    img1 = ref.astype(np.float64)
    img2 = target.astype(np.float64)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(np.mean(num / den))


def evaluate_enhancement(ref, processed, elapsed_ms=0.0):
    return {
        "PSNR (dB)": round(compute_psnr(ref, processed), 4),
        "SSIM": round(compute_ssim(ref, processed), 4),
        "MSE": round(compute_mse(ref, processed), 4),
        "Waktu (ms)": round(float(elapsed_ms), 3),
    }
