# xray-clarity-lab

Proyek Akhir Mata Kuliah Pengolahan Citra Digital (Program Studi Informatika, 2026/2027).

**Judul:** Analisis Komparatif Filter Median, Bilateral, dan CLAHE untuk Peningkatan Kualitas Citra X-Ray Berderau Berbasis Web.

Aplikasi ini membandingkan kinerja Median Filter, Bilateral Filter, dan CLAHE pada citra X-Ray yang diberi derau Gaussian maupun Salt-and-Pepper. Pengujian berfokus pada metrik kualitas citra (PSNR, SSIM, MSE, dan waktu pemrosesan) serta perbandingan visual, tanpa mengklaim manfaat diagnosis klinis.

## Struktur Folder

- `app.py` : antarmuka web berbasis Streamlit
- `src/noise.py` : penambahan derau Gaussian dan Salt-and-Pepper
- `src/filters.py` : implementasi Median Filter, Bilateral Filter, dan CLAHE
- `src/metrics.py` : perhitungan MSE, PSNR, dan SSIM
- `src/sample_generator.py` : pembuat citra contoh X-Ray untuk uji cepat
- `scripts/run_benchmark.py` : skrip pengujian otomatis (output CSV & grafik ke folder `results/`)
- `data/sample_xray/` : folder citra uji

## Instalasi dan Menjalankan Aplikasi

1. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan web Streamlit:
   ```bash
   streamlit run app.py
   ```
3. Jalankan skrip evaluasi batch (opsional):
   ```bash
   python scripts/run_benchmark.py
   ```

## Pustaka yang Digunakan

- `opencv-python-headless` (Median Filter, Bilateral Filter, CLAHE, histogram)
- `numpy` (operasi matriks, derau, dan perhitungan metrik)
- `streamlit` (tampilan web interaktif)
- `matplotlib` dan `pandas` (grafik dan ekspor tabel CSV)
