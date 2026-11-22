import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filters import run_all_methods
from src.metrics import evaluate_enhancement
from src.noise import apply_noise
from src.sample_generator import create_default_samples


def save_grid(ref, noisy, outputs, metrics_map, title, out_path):
    items = [
        ("Citra Asli", ref, None),
        ("Berderau", noisy, metrics_map["Citra Berderau"]),
        ("Median Filter", outputs["Median Filter"]["image"], metrics_map["Median Filter"]),
        ("Bilateral Filter", outputs["Bilateral Filter"]["image"], metrics_map["Bilateral Filter"]),
        ("CLAHE", outputs["CLAHE"]["image"], metrics_map["CLAHE"]),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(17, 4), dpi=150)
    for ax, (label, img, m) in zip(axes, items):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        if m is None:
            ax.set_title(label, fontsize=11, fontweight="bold")
        else:
            ax.set_title(f"{label}\nPSNR: {m['PSNR (dB)']:.2f} dB | SSIM: {m['SSIM']:.4f}", fontsize=9.5)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_chart(df_sum, out_path):
    scenarios = df_sum["Skenario"].unique().tolist()
    methods = ["Citra Berderau", "Median Filter", "Bilateral Filter", "CLAHE"]
    colors = ["#6c757d", "#1f77b4", "#2ca02c", "#ff7f0e"]

    fig, axes = plt.subplots(len(scenarios), 2, figsize=(13, 4.2 * len(scenarios)), dpi=150)
    if len(scenarios) == 1:
        axes = [axes]

    for i, scen in enumerate(scenarios):
        sub = df_sum[df_sum["Skenario"] == scen].set_index("Metode").reindex(methods)
        for j, (col, ylabel) in enumerate([("PSNR_mean", "PSNR (dB)"), ("SSIM_mean", "SSIM")]):
            ax = axes[i][j]
            bars = ax.bar(methods, sub[col], color=colors, width=0.55)
            ax.set_title(f"{scen} - {ylabel}", fontsize=11)
            ax.tick_params(axis="x", rotation=15, labelsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            for bar in bars:
                val = bar.get_height()
                txt = f"{val:.4f}" if col == "SSIM_mean" else f"{val:.2f}"
                ax.text(bar.get_x() + bar.get_width() / 2, val, txt, ha="center", va="bottom", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    data_dir = ROOT / "data" / "sample_xray"
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(data_dir.glob("*.png")) + sorted(data_dir.glob("*.jpg"))
    if not images:
        images = create_default_samples(data_dir)

    scenarios = [
        ("Gaussian (sigma=20)", "gaussian", 20.0),
        ("Salt-and-Pepper (5%)", "salt_pepper", 0.05),
    ]

    rows = []
    for scen_name, n_type, intensity in scenarios:
        for idx, path in enumerate(images):
            ref = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if ref is None:
                continue
            noisy = apply_noise(ref, noise_type=n_type, intensity=intensity, seed=42)
            outputs = run_all_methods(noisy)

            metrics_map = {"Citra Berderau": evaluate_enhancement(ref, noisy, 0.0)}
            for m_name, m_res in outputs.items():
                metrics_map[m_name] = evaluate_enhancement(ref, m_res["image"], m_res["elapsed_ms"])

            for m_name, m_vals in metrics_map.items():
                rows.append({"Skenario": scen_name, "Citra": path.name, "Metode": m_name, **m_vals})

            if idx == 0:
                save_grid(
                    ref,
                    noisy,
                    outputs,
                    metrics_map,
                    f"Perbandingan Visual - {scen_name} ({path.name})",
                    out_dir / f"comparison_grid_{n_type}.png",
                )

    df_detail = pd.DataFrame(rows)
    df_detail.to_csv(out_dir / "benchmark_detail.csv", index=False)

    df_sum = (
        df_detail.groupby(["Skenario", "Metode"], sort=False)[["PSNR (dB)", "SSIM", "MSE", "Waktu (ms)"]]
        .agg(["mean", "std"])
        .round(4)
    )
    df_sum.columns = [f"{a.split(' ')[0]}_{b}" for a, b in df_sum.columns]
    df_sum = df_sum.reset_index()
    df_sum.to_csv(out_dir / "benchmark_summary.csv", index=False)

    save_chart(df_sum, out_dir / "metrics_chart.png")
    print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
