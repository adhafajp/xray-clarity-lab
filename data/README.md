# Dokumentasi Data Uji — XRay Clarity Lab

**Proyek:** Analisis Komparatif Filter Median, Bilateral, dan CLAHE untuk Peningkatan Kualitas Citra X-Ray Berderau Berbasis Web  
**Topik resmi:** Penyaringan Citra Medis  
**Tujuan:** Evaluasi teknis algoritma pengolahan citra, **bukan diagnosis atau penilaian klinis**.

## 1. Inventaris data yang disertakan

Folder `sample_xray/` pada repositori saat ini berisi **tiga phantom X-Ray toraks sintetis**, **bukan radiografi pasien**. Ketiganya dihasilkan secara terprogram oleh `src/sample_generator.py` untuk demonstrasi aplikasi dan pengujian awal.

| Nama berkas | Skenario sintetis | Resolusi | Format |
|---|---|---|---|
| `sample_xray_normal_contrast.png` | Phantom dengan kontras dasar (variant 0) | 512 × 512 | PNG, grayscale 8-bit |
| `sample_xray_low_contrast.png` | Phantom dengan kontras lebih rendah (variant 1) | 512 × 512 | PNG, grayscale 8-bit |
| `sample_xray_local_opacity.png` | Phantom dengan opasitas lokal buatan (variant 2) | 512 × 512 | PNG, grayscale 8-bit |

Citra dibuat menggunakan bentuk geometris dan penghalusan Gaussian untuk menyimulasikan pola radiografi toraks. **Phantom ini tidak boleh diperlakukan sebagai citra klinis atau sebagai data validasi medis.**

Bila file contoh tidak ditemukan, `src/sample_generator.py` dapat membuatnya kembali otomatis. Untuk reproduksi manual dari direktori akar proyek:

```bash
python -m src.sample_generator
```

### Integritas berkas contoh

Hash SHA-256 di bawah berlaku untuk tiga berkas yang disertakan pada revisi ini. Hash bermanfaat untuk memeriksa apakah data berubah sebelum eksperimen.

| Berkas | SHA-256 |
|---|---|
| `sample_xray_normal_contrast.png` | `c942ca82d2bc7dc2f28320c35e3367caf7998d62214e17d791c6bce988f38dee` |
| `sample_xray_low_contrast.png` | `f409c97e788f95bfbc622911ad57beef5ffcf84c8f4dbba1762c7666f26ae535` |
| `sample_xray_local_opacity.png` | `4e24d0e9f937287404362cdcc56224b477a15afb34fc500039bc82c6344198d4` |

## 2. Penambahan dataset X-Ray nyata untuk evaluasi penelitian

Tiga phantom sudah memadai untuk **demo UI**, tetapi belum cukup untuk menarik simpulan komparatif yang mewakili citra X-Ray nyata. Untuk laporan akhir, tim sebaiknya menggunakan **setidaknya 15–30 radiografi X-Ray nyata** yang bisa dipertanggungjawabkan asal-usulnya; jumlah tersebut merupakan target praktis awal, **bukan jaminan validitas statistik**.

Langkah pengumpulan:

1. Pilih dataset radiografi publik yang benar-benar menampilkan X-Ray (misalnya *chest radiograph*), bukan foto peralatan atau gambar ilustrasi.
2. **Buka dan verifikasi halaman resmi dataset** untuk memastikan sumber, perizinan, persyaratan atribusi, serta batasan redistribusi. Contoh kandidat yang perlu diverifikasi: *Chest X-Ray Images (Pneumonia)* dan NIH *ChestX-ray14*. Nama kandidat bukan pernyataan bahwa lisensinya otomatis cocok.
3. Ambil sampel secara terdokumentasi (misalnya 30 gambar dari sumber yang sama), dan hindari citra yang identik/duplikat. Bila informasi tersedia, hindari melibatkan pasien yang sama berulang kali dalam penarikan kesimpulan seolah-olah independen.
4. Jangan menyimpan nama pasien, tanggal lahir, ID rekam medis, atau informasi pengenal lain. Hilangkan tulisan/metadata pengenal dan ikuti ketentuan dataset.
5. Catat nama dataset, URL resmi, versi/tanggal akses, penyedia, lisensi, jumlah data, kriteria pemilihan, karakteristik sumber, dan prapemrosesan yang diterapkan.
6. Pastikan hak untuk mengunggah atau mendistribusikan citra jika hendak menyertakannya di repositori publik. Jika tidak ada izin redistribusi, **jangan commit citra tersebut ke GitHub**; dokumentasikan cara mengaksesnya secara resmi.

**Template metadata pencatatan yang disarankan** (boleh dibuat sebagai CSV di penyimpanan internal tim):

```csv
filename,dataset,source_url,license_or_terms,downloaded_on,original_resolution,preprocessing,notes
example_001.png,NAMA_DATASET,URL_RESMI,SYARAT_LISENSI,YYYY-MM-DD,1024x1024,grayscale_8bit,contoh_tanpa_data_pribadi
```

Template di atas hanya contoh format; baris `example_001.png` **bukan data yang telah tersedia**.

## 3. Standar format dan prapemrosesan

- **Input web:** `.png`, `.jpg`, `.jpeg` melalui uploader. Citra X-Ray DICOM (`.dcm`) **belum didukung langsung**.
- **Pembacaan:** file gambar dikonversi ke **grayscale 8-bit** (`uint8`, intensitas 0–255). Citra berwarna, jika diunggah, juga dikonversi ke grayscale.
- **Ukuran:** resolusi asli gambar input dipertahankan oleh pipeline web. Saat ini tidak ada normalisasi resolusi semua citra ke satu ukuran. Dokumentasikan perbedaan ukuran karena waktu pemrosesan dipengaruhi dimensi citra.
- **Batas unggahan di `app.py`:** maksimum **20 MB per file** dan sisi minimal **16 piksel**. Batas server Streamlit dapat berbeda sesuai konfigurasi deployment.
- **Peringatan radiologi:** konversi JPEG, konversi bit-depth, dan intensitas gambar yang sudah diproses oleh sumber dapat memengaruhi hasil. Jangan menyimpulkan perubahan intensitas sebagai perbaikan akurasi diagnosis.

## 4. Skenario eksperimen yang dilayani aplikasi

1. Baca citra input sebagai `I_ref` (diasumsikan referensi awal sebelum penambahan derau sintetis).
2. Buat `I_noisy` dengan salah satu jenis derau dari `src/noise.py`:
   - **Gaussian:** sigma pada antarmuka 5–60; nilai awal **20**.
   - **Salt-and-Pepper:** rasio 0,01–0,25; nilai awal **0,05** (5% piksel).
   - **Tanpa Derau:** untuk inspeksi visual atau pengujian kontrol.
3. Gunakan random seed tetap (**42** secara default) agar derau simulasi dapat direproduksi.
4. Proses **citra berderau yang sama** secara terpisah melalui Median, Bilateral, dan CLAHE menggunakan parameter sidebar.
5. Bandingkan masing-masing hasil terhadap `I_ref` untuk menghitung **MSE**, **PSNR**, dan **SSIM**; catat juga **waktu pemrosesan (ms)** setiap metode.
6. Gunakan tab **Visual Studio** untuk Before–After interaktif, **Metrik & Grafik** untuk hasil numerik, **Histogram** untuk distribusi intensitas, dan **Batch Uji** untuk evaluasi banyak file.
7. Unduh hasil gambar atau CSV. Sebelum pelaporan, periksa konfigurasi percobaan dan simpan catatan seed, parameter, dan identitas setiap data.

> **Batas interpretasi:** metrik referensi ini valid terutama untuk eksperimen **derau buatan** karena `I_ref` tersedia. Jika radiografi sumber sebenarnya sudah berderau, aplikasi *tidak dapat* menyatakan bahwa `I_ref` adalah citra klinis bebas derau (*ground truth*). Hindari klaim bahwa PSNR/SSIM tinggi berarti diagnosis lebih akurat.

## 5. Parameter yang perlu dicatat dalam laporan

| Parameter | Nilai awal | Rentang UI |
|---|---:|---|
| Jenis derau | Gaussian | Gaussian, Salt-and-Pepper, Tanpa Derau |
| Sigma Gaussian | 20 | 5–60 |
| Rasio Salt-and-Pepper | 0,05 | 0,01–0,25 |
| Seed | 42 | 0–9999 |
| Median kernel | 5 | 3, 5, 7, 9, 11 |
| Bilateral diameter | 9 | 5, 7, 9, 11, 15 |
| Bilateral sigma color | 75 | 10–150 |
| Bilateral sigma space | 75 | 10–150 |
| CLAHE clip limit | 2,0 | 1–8 |
| CLAHE tile grid | 8×8 | 4, 8, 12, 16 |

**Catatan tentang keadilan komparasi:** Median dan Bilateral merupakan filter peredam derau, sedangkan CLAHE berfokus pada peningkatan kontras lokal; metrik kesamaan piksel saja tidak menggambarkan seluruh kegunaan CLAHE. Pengujian tambahan terhadap kontras boleh dipertimbangkan dan perlu diberi label terpisah.

## 6. Pengujian batch dan pencatatan hasil

Tab **Batch Uji** menerima beberapa berkas sekaligus. Setelah menekan **Jalankan evaluasi batch**, aplikasi menyediakan:

- Tabel ringkasan **rata-rata** dan **standar deviasi** PSNR, SSIM, MSE, serta waktu proses per metode.
- Tabel rinci metrik tiap citra (di dalam expander).
- Ekspor `batch_summary_<noise>.csv` dan `batch_detail_<noise>.csv`.

Standar deviasi pada batch **satu citra** ditampilkan `0` agar tidak muncul `NaN`, tetapi **bukan estimasi ketidakpastian statistik**. Gunakan beberapa citra untuk analisis bermakna. Waktu proses diukur di lingkungan perangkat saat eksekusi dan dapat berfluktuasi; untuk laporan yang serius, ulangi percobaan berkali-kali pada perangkat yang sama.

Skrip benchmark terpisah tetap tersedia melalui:

```bash
python scripts/run_benchmark.py
```

Output default disimpan ke `results/` sesuai implementasi skrip tersebut. Benchmark bawaan saat ini menggunakan phantom sintetis dan parameter yang ditentukan dalam skrip, sehingga nilainya tidak harus sama dengan eksperimen manual di web bila parameter berbeda.

## 7. Privasi, sumber data, dan batasan penggunaan

- Aplikasi ditujukan untuk **pembelajaran dan penelitian teknis non-klinis**. Tidak memprediksi penyakit dan tidak memvalidasi kegunaan diagnostik.
- Jangan mengunggah citra yang mengandung informasi kesehatan pribadi tanpa izin dan perlindungan yang memadai, khususnya jika aplikasi dideploy pada server publik.
- Cantumkan atribusi dan batasan lisensi dari **sumber dataset asli** dalam laporan/README proyek. Jangan menyebut tiga phantom sintetis ini sebagai dataset X-Ray pasien asli.
- Pisahkan bukti **hasil gambar contoh** dari **hasil pengujian X-Ray nyata** dalam presentasi agar evaluasi tidak menyesatkan.

## 8. Status data saat ini

| Komponen data | Status |
|---|---|
| 3 contoh phantom X-Ray sintetis | Tersedia dalam repo |
| Generator gambar contoh | Tersedia (`src/sample_generator.py`) |
| Penambahan derau terkontrol | Tersedia (`src/noise.py`) |
| Dataset radiografi nyata yang terkurasi dan terdokumentasi | **Belum disertakan** |
| Data provenance, lisensi dan daftar citra riil terpilih | **Perlu disiapkan kelompok** |
| Ground truth klinis / analisis diagnosis | **Di luar lingkup proyek** |

**Penanggung jawab dokumentasi:** anggota tim pengembangan UI/UX dan dokumentasi data uji. Identitas personal dan pembagian kontribusi dapat dilengkapi oleh kelompok sesuai ketentuan tugas.
