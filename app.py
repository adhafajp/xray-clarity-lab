import io
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.filters import run_all_methods
from src.metrics import evaluate_enhancement
from src.noise import apply_noise
from src.sample_generator import create_default_samples

SAMPLE_DIR = Path(__file__).resolve().parent / "data" / "sample_xray"


def load_gray_bytes(raw_bytes):
    img = Image.open(io.BytesIO(raw_bytes)).convert("L")
    return np.array(img, dtype=np.uint8)


def load_gray_file(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Citra tidak ditemukan: {path}")
    return img


def get_samples():
    files = sorted(SAMPLE_DIR.glob("*.png")) + sorted(SAMPLE_DIR.glob("*.jpg"))
    if not files:
        files = create_default_samples(SAMPLE_DIR)
    return files


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def make_comparison_figure(ref, noisy, outputs, metrics_map):
    items = [
        ("Citra Asli", ref, None),
        ("Berderau", noisy, metrics_map["Citra Berderau"]),
        ("Median Filter", outputs["Median Filter"]["image"], metrics_map["Median Filter"]),
        ("Bilateral Filter", outputs["Bilateral Filter"]["image"], metrics_map["Bilateral Filter"]),
        ("CLAHE", outputs["CLAHE"]["image"], metrics_map["CLAHE"]),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(17, 4), dpi=130)
    for ax, (label, img, m) in zip(axes, items):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        if m is None:
            ax.set_title(label, fontsize=11, fontweight="bold")
        else:
            ax.set_title(
                f"{label}\nPSNR: {m['PSNR (dB)']:.2f} dB | SSIM: {m['SSIM']:.4f}",
                fontsize=9.5,
            )
    fig.tight_layout()
    return fig


def make_metrics_chart(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=130)
    colors = ["#6c757d", "#1f77b4", "#2ca02c", "#ff7f0e"]
    specs = [
        ("PSNR (dB)", "Perbandingan PSNR (dB)"),
        ("SSIM", "Perbandingan SSIM"),
        ("Waktu (ms)", "Waktu Pemrosesan (ms)"),
    ]
    for ax, (col, title) in zip(axes, specs):
        bars = ax.bar(df["Metode"], df[col], color=colors[: len(df)], width=0.55)
        ax.set_title(title, fontsize=10.5)
        ax.tick_params(axis="x", rotation=18, labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for bar in bars:
            val = bar.get_height()
            txt = f"{val:.4f}" if col == "SSIM" else f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, val, txt, ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout()
    return fig


def make_histogram_chart(ref, noisy, outputs):
    fig, ax = plt.subplots(figsize=(10, 4), dpi=130)
    series = [
        ("Citra Asli", ref, "#222222", "-"),
        ("Berderau", noisy, "#888888", "--"),
        ("Median Filter", outputs["Median Filter"]["image"], "#1f77b4", "-"),
        ("Bilateral Filter", outputs["Bilateral Filter"]["image"], "#2ca02c", "-"),
        ("CLAHE", outputs["CLAHE"]["image"], "#ff7f0e", "-"),
    ]
    for name, img, color, style in series:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
        ax.plot(hist, label=name, color=color, linestyle=style, linewidth=1.6)
    ax.set_title("Perbandingan Histogram Intensitas (0 - 255)")
    ax.set_xlabel("Intensitas Piksel")
    ax.set_ylabel("Frekuensi")
    ax.set_xlim(0, 255)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="Komparasi Filter Citra X-Ray",
        layout="wide",
    )

    st.title("Analisis Komparatif Filter Median, Bilateral, dan CLAHE pada Citra X-Ray Berderau")
    st.caption("Proyek Akhir Mata Kuliah Pengolahan Citra Digital - Evaluasi teknis kualitas citra (non-klinis).")

    samples = get_samples()

    with st.sidebar:
        st.subheader("1. Input Citra")
        mode_input = st.radio("Sumber citra:", ["Citra Contoh", "Upload Citra Sendiri"])

        uploaded = None
        chosen_sample = samples[0]
        if mode_input == "Upload Citra Sendiri":
            uploaded = st.file_uploader("Pilih file X-Ray (PNG/JPG)", type=["png", "jpg", "jpeg"])
        else:
            chosen_sample = st.selectbox("Pilih citra contoh:", samples, format_func=lambda p: p.name)

        st.subheader("2. Pengaturan Derau")
        noise_opt = st.selectbox("Jenis derau:", ["Gaussian", "Salt-and-Pepper", "Tanpa Derau"])
        seed = st.number_input("Seed acak", min_value=0, max_value=9999, value=42, step=1)

        if noise_opt == "Gaussian":
            noise_type = "gaussian"
            noise_level = st.slider("Sigma Gaussian", 5.0, 60.0, 20.0, 1.0)
        elif noise_opt == "Salt-and-Pepper":
            noise_type = "salt_pepper"
            noise_level = st.slider("Rasio piksel derau", 0.01, 0.25, 0.05, 0.01)
        else:
            noise_type = "none"
            noise_level = 0.0

        st.subheader("3. Parameter Metode")
        median_k = st.select_slider("Median - Ukuran Kernel", options=[3, 5, 7, 9, 11], value=5)
        bilat_d = st.select_slider("Bilateral - Diameter (d)", options=[5, 7, 9, 11, 15], value=9)
        bilat_sc = st.slider("Bilateral - Sigma Color", 10.0, 150.0, 75.0, 5.0)
        bilat_ss = st.slider("Bilateral - Sigma Space", 10.0, 150.0, 75.0, 5.0)
        clahe_clip = st.slider("CLAHE - Clip Limit", 1.0, 8.0, 2.0, 0.5)
        clahe_grid = st.select_slider("CLAHE - Tile Grid", options=[4, 8, 12, 16], value=8)

    if mode_input == "Upload Citra Sendiri" and uploaded is not None:
        ref_img = load_gray_bytes(uploaded.getvalue())
        img_name = uploaded.name
    else:
        ref_img = load_gray_file(chosen_sample)
        img_name = chosen_sample.name

    noisy_img = apply_noise(ref_img, noise_type=noise_type, intensity=noise_level, seed=int(seed))
    outputs = run_all_methods(
        noisy_img,
        median_ksize=median_k,
        bilateral_d=bilat_d,
        bilateral_sigma_color=bilat_sc,
        bilateral_sigma_space=bilat_ss,
        clahe_clip_limit=clahe_clip,
        clahe_grid_size=clahe_grid,
    )

    metrics_map = {"Citra Berderau": evaluate_enhancement(ref_img, noisy_img, 0.0)}
    for name, res in outputs.items():
        metrics_map[name] = evaluate_enhancement(ref_img, res["image"], res["elapsed_ms"])

    df = pd.DataFrame([{"Metode": k, **v} for k, v in metrics_map.items()])
    only_methods = df[df["Metode"] != "Citra Berderau"]
    best_psnr = only_methods.loc[only_methods["PSNR (dB)"].idxmax()]
    best_ssim = only_methods.loc[only_methods["SSIM"].idxmax()]
    fastest = only_methods.loc[only_methods["Waktu (ms)"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Citra Uji", img_name, f"{ref_img.shape[1]}x{ref_img.shape[0]}")
    c2.metric("PSNR Terbaik", f"{best_psnr['PSNR (dB)']:.2f} dB", best_psnr["Metode"])
    c3.metric("SSIM Terbaik", f"{best_ssim['SSIM']:.4f}", best_ssim["Metode"])
    c4.metric("Tercepat", f"{fastest['Waktu (ms)']:.2f} ms", fastest["Metode"])

    t1, t2, t3, t4 = st.tabs(["Hasil Visual", "Metrik Evaluasi", "Histogram", "Uji Banyak Citra"])

    with t1:
        fig_grid = make_comparison_figure(ref_img, noisy_img, outputs, metrics_map)
        st.pyplot(fig_grid, use_container_width=True)
        st.download_button(
            "Download Gambar Perbandingan (PNG)",
            data=fig_to_bytes(fig_grid),
            file_name=f"hasil_{Path(img_name).stem}_{noise_type}.png",
            mime="image/png",
        )
        plt.close(fig_grid)

        st.write("Bandingkan dua gambar secara langsung:")
        opts = ["Citra Asli", "Citra Berderau", "Median Filter", "Bilateral Filter", "CLAHE"]
        col_a, col_b = st.columns(2)
        pick_a = col_a.selectbox("Kiri", opts, index=1)
        pick_b = col_b.selectbox("Kanan", opts, index=3)

        img_dict = {
            "Citra Asli": ref_img,
            "Citra Berderau": noisy_img,
            "Median Filter": outputs["Median Filter"]["image"],
            "Bilateral Filter": outputs["Bilateral Filter"]["image"],
            "CLAHE": outputs["CLAHE"]["image"],
        }
        v1, v2 = st.columns(2)
        v1.image(img_dict[pick_a], caption=pick_a, use_container_width=True, clamp=True)
        v2.image(img_dict[pick_b], caption=pick_b, use_container_width=True, clamp=True)

    with t2:
        st.dataframe(df, use_container_width=True, hide_index=True)
        fig_bar = make_metrics_chart(df)
        st.pyplot(fig_bar, use_container_width=True)
        b1, b2 = st.columns(2)
        b1.download_button(
            "Download Tabel (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"metrik_{Path(img_name).stem}_{noise_type}.csv",
            mime="text/csv",
        )
        b2.download_button(
            "Download Grafik (PNG)",
            data=fig_to_bytes(fig_bar),
            file_name=f"grafik_{Path(img_name).stem}_{noise_type}.png",
            mime="image/png",
        )
        plt.close(fig_bar)

    with t3:
        fig_hist = make_histogram_chart(ref_img, noisy_img, outputs)
        st.pyplot(fig_hist, use_container_width=True)
        st.download_button(
            "Download Histogram (PNG)",
            data=fig_to_bytes(fig_hist),
            file_name=f"histogram_{Path(img_name).stem}_{noise_type}.png",
            mime="image/png",
        )
        plt.close(fig_hist)

    with t4:
        st.write("Upload beberapa citra X-Ray untuk menghitung rata-rata PSNR, SSIM, MSE, dan waktu proses.")
        batch_files = st.file_uploader(
            "Pilih beberapa file citra",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        if batch_files:
            rows = []
            for f in batch_files:
                r_img = load_gray_bytes(f.getvalue())
                n_img = apply_noise(r_img, noise_type=noise_type, intensity=noise_level, seed=int(seed))
                outs = run_all_methods(
                    n_img,
                    median_ksize=median_k,
                    bilateral_d=bilat_d,
                    bilateral_sigma_color=bilat_sc,
                    bilateral_sigma_space=bilat_ss,
                    clahe_clip_limit=clahe_clip,
                    clahe_grid_size=clahe_grid,
                )
                rows.append({"Citra": f.name, "Metode": "Citra Berderau", **evaluate_enhancement(r_img, n_img, 0.0)})
                for m_name, m_res in outs.items():
                    rows.append({"Citra": f.name, "Metode": m_name, **evaluate_enhancement(r_img, m_res["image"], m_res["elapsed_ms"])})

            df_all = pd.DataFrame(rows)
            df_avg = (
                df_all.groupby("Metode", sort=False)[["PSNR (dB)", "SSIM", "MSE", "Waktu (ms)"]]
                .agg(["mean", "std"])
                .round(4)
            )
            df_avg.columns = [f"{a}_{b}" for a, b in df_avg.columns]
            df_avg = df_avg.reset_index()

            st.dataframe(df_avg, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Ringkasan Batch (CSV)",
                data=df_avg.to_csv(index=False).encode("utf-8"),
                file_name=f"batch_{noise_type}.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
