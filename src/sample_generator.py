"""Pembuat citra phantom X-Ray toraks sintetis untuk pengujian awal & demo cepat.

Citra sintetis ini hanya digunakan sebagai data contoh siap pakai saat repositori
baru diunduh. Untuk eksperimen laporan resmi, tambahkan citra X-Ray publik berlisensi
(misalnya dari Kaggle Chest X-Ray) ke folder `data/sample_xray/`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def generate_thorax_phantom(size: int = 512, variant: int = 0) -> np.ndarray:
    """Menghasilkan citra grayscale 8-bit menyerupai radiografi toraks (phantom)."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size / 2.0, size / 2.0

    # Latar belakang radiografi gelap
    img = np.full((size, size), 18.0, dtype=np.float32)

    # Siluet rongga dada (torso)
    torso = ((x - cx) / (size * 0.40)) ** 2 + ((y - cy * 1.04) / (size * 0.44)) ** 2 <= 1.0
    img[torso] = 155.0 - 25.0 * (((x[torso] - cx) / (size * 0.40)) ** 2)

    # Bidang paru kiri & kanan (lebih gelap karena berisi udara)
    lung_offset = size * (0.145 + 0.008 * variant)
    left_lung = ((x - (cx - lung_offset)) / (size * 0.14)) ** 2 + (
        (y - cy * 0.96) / (size * 0.28)
    ) ** 2 <= 1.0
    right_lung = ((x - (cx + lung_offset)) / (size * 0.145)) ** 2 + (
        (y - cy * 0.96) / (size * 0.28)
    ) ** 2 <= 1.0
    img[left_lung | right_lung] = 52.0 + 6.0 * variant

    # Struktur tulang belakang (kolom vertikal terang di tengah)
    spine = np.abs(x - cx) < size * 0.042
    img[spine & torso] = 195.0

    # Siluet jantung sedikit ke kanan (anatomi kiri pasien)
    heart = ((x - (cx + size * 0.055)) / (size * 0.10)) ** 2 + (
        (y - (cy + size * 0.075)) / (size * 0.115)
    ) ** 2 <= 1.0
    img[heart] = 178.0

    # Lengkung tulang rusuk (ribs) tipis melintang pada area paru
    for rib_idx in range(6):
        rib_y = size * (0.28 + rib_idx * 0.075)
        curve = rib_y + 0.00055 * ((x - cx) ** 2)
        rib_mask = (np.abs(y - curve) < size * 0.011) & (left_lung | right_lung)
        img[rib_mask] = np.minimum(img[rib_mask] + 38.0, 220.0)

    # Variasi kontras ringan antar sampel untuk skenario uji berbeda
    if variant == 1:
        # Kontras lebih rendah (mensimulasikan citra under-exposed)
        img = img * 0.72 + 22.0
    elif variant == 2:
        # Kontras lebih tinggi dengan sedikit opasitas lokal pada paru kanan
        opacity = ((x - (cx + size * 0.15)) / (size * 0.06)) ** 2 + (
            (y - (cy + size * 0.04)) / (size * 0.05)
        ) ** 2 <= 1.0
        img[opacity] = np.minimum(img[opacity] + 45.0, 210.0)

    # Haluskan tepi anatomi agar realistis
    img = cv2.GaussianBlur(img, (7, 7), 1.6)
    return np.clip(img, 0, 255).astype(np.uint8)


def create_default_samples(output_dir: Path) -> list[Path]:
    """Menyimpan 3 citra phantom X-Ray ke folder tujuan."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "sample_xray_normal_contrast.png",
        "sample_xray_low_contrast.png",
        "sample_xray_local_opacity.png",
    ]
    paths: list[Path] = []
    for idx, filename in enumerate(names):
        out_path = output_dir / filename
        phantom = generate_thorax_phantom(size=512, variant=idx)
        cv2.imwrite(str(out_path), phantom)
        paths.append(out_path)
    return paths


if __name__ == "__main__":
    default_dir = Path(__file__).resolve().parent.parent / "data" / "sample_xray"
    saved = create_default_samples(default_dir)
    for p in saved:
        print(f"Tersimpan: {p}")
