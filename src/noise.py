import numpy as np


def add_gaussian_noise(image, sigma=20.0, seed=42):
    # Tambahkan derau Gaussian sesuai nilai sigma
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, float(sigma), image.shape)
    result = np.clip(image.astype(np.float64) + noise, 0, 255)
    return result.astype(np.uint8)


def add_salt_and_pepper_noise(image, amount=0.05, salt_ratio=0.5, seed=42):
    # Tambahkan derau impuls (titik putih = 255, titik hitam = 0)
    rng = np.random.default_rng(seed)
    result = image.copy()
    amount = float(np.clip(amount, 0.0, 0.5))
    rand_map = rng.random(image.shape)

    result[rand_map < (amount * salt_ratio)] = 255
    result[(rand_map >= (amount * salt_ratio)) & (rand_map < amount)] = 0
    return result


def apply_noise(image, noise_type="gaussian", intensity=20.0, seed=42):
    mode = noise_type.strip().lower()
    if mode == "gaussian":
        return add_gaussian_noise(image, sigma=intensity, seed=seed)
    if mode in ("salt_pepper", "salt_and_pepper", "sp"):
        return add_salt_and_pepper_noise(image, amount=intensity, seed=seed)
    if mode == "none":
        return image.copy()
    raise ValueError(f"Jenis derau tidak dikenal: {noise_type}")
