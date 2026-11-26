import os
import tempfile
import streamlit as st
import whitebox
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Deteksi Kerucut Vulkanik Otomatis (Multi-Skala & Adaptif)",
    page_icon="🌋",
    layout="wide"
)

# --- CUSTOM CSS & BANNER HEADER ---
st.markdown("""
<style>
    .hero-banner {
        background: linear-gradient(135deg, #4A151B 0%, #8B2635 100%);
        padding: 1.8rem 2rem;
        border-radius: 14px;
        color: #FAF7F2;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .hero-banner h1 {
        color: #FAF7F2 !important;
        margin: 0;
        font-size: 1.9rem;
    }
    .hero-banner p {
        color: #F3E5E7;
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
    }
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E5DDD5;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .stDownloadButton button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
</style>

<div class="hero-banner">
    <h1>🌋 Sistem Deteksi Otomatis & Morfometri Kerucut Vulkanik</h1>
    <p>Deteksi adaptif berbasis Multi-Scale Morphometry & rasio tak berdimensi (Scale-Independent)</p>
</div>
""", unsafe_allow_html=True)

# --- FUNGSI EKSTRAKSI MORFOMETRI MULTI-SKALA (DI-CACHE & HEMAT RAM) ---
@st.cache_data(show_spinner="Sedang mengekstrak parameter morfometri multi-skala (Slope, TPI, RTP, Curvature)...")
def ekstrak_morfometri(_uploaded_file, file_id):
    tmpdir = tempfile.mkdtemp()
    dem_path = os.path.join(tmpdir, "dem_input.tif")
    slope_path = os.path.join(tmpdir, "slope.tif")
    prof_path = os.path.join(tmpdir, "prof_curv.tif")
    plan_path = os.path.join(tmpdir, "plan_curv.tif")

    _uploaded_file.seek(0)
    with open(dem_path, "wb") as f:
        f.write(_uploaded_file.read())

    # 1. Baca dimensi DEM lebih dulu untuk menentukan skala jendela proporsional
    with rasterio.open(dem_path) as src:
        dem_arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            dem_arr[dem_arr == src.nodata] = np.nan
        transform = src.transform
        pixel_size = abs(transform.a)
        is_geographic = src.crs.is_geographic if src.crs is not None else False
        crs_text = str(src.crs) if src.crs is not None else "Tidak diketahui"
        min_dim = min(dem_arr.shape)  # Sisi terpendek peta DEM (dalam piksel)

    # Buat 4 tingkat skala jendela proporsional (1%, 4%, 12%, 25% dari sisi terpendek DEM)
    proporsi = [0.01, 0.04, 0.12, 0.25]
    skala_jendela = []
    for p in proporsi:
        w = max(5, int(min_dim * p))
        if w % 2 == 0:
            w += 1  # WhiteboxTools mewajibkan ukuran filter ganjil
        if w not in skala_jendela:
            skala_jendela.append(w)

    # 2. Ekstraksi Parameter Dasar
    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)

    wbt.slope(dem_path, slope_path, units="degrees")
    wbt.profile_curvature(dem_path, prof_path)
    wbt.plan_curvature(dem_path, plan_path)

    # 3. Ekstraksi TPI & RTP Bertingkat (Multi-Scale) & Penggabungan
    tpi_gabungan = np.full(dem_arr.shape, -np.inf, dtype=np.float32)
    rtp_gabungan = np.full(dem_arr.shape, -np.inf, dtype=np.float32)

    for w in skala_jendela:
        tpi_w_path = os.path.join(tmpdir, f"tpi_{w}.tif")
        rtp_w_path = os.path.join(tmpdir, f"rtp_{w}.tif")

        wbt.diff_from_mean_elev(dem_path, tpi_w_path, filterx=w, filtery=w)
        wbt.relative_topographic_position(dem_path, rtp_w_path, filterx=w, filtery=w)

        with rasterio.open(tpi_w_path) as s_tpi:
            tpi_arr = s_tpi.read(1).astype(np.float32)
            if s_tpi.nodata is not None:
                tpi_arr[tpi_arr == s_tpi.nodata] = np.nan
            # Normalisasi ke Z-score per skala agar skala kecil & besar setara
            tpi_std = np.nanstd(tpi_arr)
            tpi_z_w = (tpi_arr - np.nanmean(tpi_arr)) / tpi_std if tpi_std > 0 else np.zeros_like(tpi_arr)
            tpi_gabungan = np.fmax(tpi_gabungan, tpi_z_w)

        with rasterio.open(rtp_w_path) as s_rtp:
            rtp_arr = s_rtp.read(1).astype(np.float32)
            if s_rtp.nodata is not None:
                rtp_arr[rtp_arr == s_rtp.nodata] = np.nan
            rtp_gabungan = np.fmax(rtp_gabungan, rtp_arr)

    # Kembalikan nilai -inf (area nodata) menjadi NaN
    tpi_gabungan[np.isinf(tpi_gabungan)] = np.nan
    rtp_gabungan[np.isinf(rtp_gabungan)] = np.nan

    rasters = {
        "dem": dem_arr,
        "tpi": tpi_gabungan,        # Sudah berupa Multi-Scale TPI Z-Score
        "rel_height": rtp_gabungan  # Sudah berupa Multi-Scale RTP
    }

    for key, path in [("slope", slope_path), ("prof_curv", prof_path), ("plan_curv", plan_path)]:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            rasters[key] = arr

    return rasters, transform, pixel_size, is_geographic, crs_text, skala_jendela


# --- FUNGSI PANDUAN PARAMETER OTOMATIS ---
def tampilkan_panduan_parameter():
    st.subheader("📘 Metodologi Deteksi Otomatis (Multi-Skala & Tanpa Threshold Manual)")
    st.markdown(
        "Sistem ini tidak lagi menggunakan batas angka kaku (seperti tinggi dalam meter, luas maksimal piksel, atau jendela TPI tunggal). "
        "Sebagai gantinya, seluruh penyaringan menggunakan **Multi-Scale TPI/RTP proporsional**, **statistik adaptif DEM lokal**, "
        "dan **rasio geometri tak berdimensi** sehingga dapat diterapkan lintas skala (Bumi maupun Mars):"
    )

    data_panduan = [
        {
            "Parameter Otomatis": "Multi-Scale TPI & RTP (1%, 4%, 12%, 25% DEM)",
            "Sifat": "Proporsional & Tak Berdimensi",
            "Makna Geologis": "Mengukur tonjolan topografi pada 4 tingkat skala secara otomatis agar kerucut parasit kecil maupun tubuh gunung api utama ikut terbaca."
        },
        {
            "Parameter Otomatis": "Slope Adaptif (Persentil ke-75)",
            "Sifat": "Data-Driven (Otomatis)",
            "Makna Geologis": "Batas kemiringan lereng dihitung otomatis dari distribusi lereng DEM yang diunggah, bukan angka derajat kaku."
        },
        {
            "Parameter Otomatis": "Standardized TPI (Z-Score > 0.3) & RTP (> 0.05)",
            "Sifat": "Tak Berdimensi",
            "Makna Geologis": "Menormalisasi tonjolan topografi terhadap simpangan baku lokal agar bebas dari perbedaan skala elevasi."
        },
        {
            "Parameter Otomatis": "Rasio Tinggi-Diameter / H/D (0.03 – 0.40)",
            "Sifat": "Tak Berdimensi",
            "Makna Geologis": "Menggantikan batas tinggi (m) dan luas maksimal. Proporsi H/D kerucut vulkanik konsisten baik pada skala kecil maupun raksasa."
        },
        {
            "Parameter Otomatis": "Rasio Kontras Lereng (≥ 1.25)",
            "Sifat": "Tak Berdimensi",
            "Makna Geologis": "Perbandingan rata-rata lereng objek terhadap cincin sekitarnya. Memastikan kerucut lebih curam dari lingkungan."
        },
        {
            "Parameter Otomatis": "Circularity (≥ 0.45) & Aspect Ratio (≤ 2.5)",
            "Sifat": "Tak Berdimensi",
            "Makna Geologis": "Menyaring punggungan erosi yang memanjang dan tidak beraturan tanpa bergantung pada ukuran objek."
        }
    ]
    st.dataframe(pd.DataFrame(data_panduan), use_container_width=True, hide_index=True)


# --- SIDEBAR: UPLOAD DATA ---
st.sidebar.header("1. Unggah Data DEM")
uploaded_file = st.sidebar.file_uploader(
    "Pilih file DEM (.tif / .tiff)",
    type=["tif", "tiff"],
    help="Wajib menggunakan sistem koordinat proyeksi metrik (misal UTM atau Equidistant Cylindrical Mars) agar hitungan luas dan volume valid."
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Mode Otomatis & Multi-Skala Aktif**\n\n"
    "Ukuran jendela TPI/RTP dihitung proporsional terhadap dimensi DEM (4 tingkat skala), dan seluruh ambang batas menggunakan statistik lokal serta rasio tak berdimensi."
)


# --- TAMPILAN UTAMA ---
if uploaded_file is None:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🚀 Langkah-Langkah Penggunaan")
        st.markdown(
            """
1. **Siapkan File DEM:** Pastikan file DEM sudah diproyeksikan ke sistem metrik.
2. **Unggah di Panel Kiri:** Klik tombol *Browse files* pada bagian **1. Unggah Data DEM**.
3. **Deteksi Multi-Skala Otomatis:** Sistem otomatis mengekstrak morfometri 4 skala, menghitung batas adaptif, dan menyaring kerucut berdasarkan rasio tak berdimensi.
4. **Unduh Hasil:** Tinjau peta deteksi dan unduh tabel morfometri (sudah dilengkapi fitur rasio untuk *Machine Learning*) dalam format `.csv`.
            """
        )

    with col_b:
        st.subheader("📋 Syarat File DEM yang Diunggah")
        st.markdown(
            """
- **Format File:** GeoTIFF (`.tif` atau `.tiff`).
- **Sistem Koordinat (CRS):** Wajib **proyeksi metrik** (misal `EPSG:32749` untuk Lamongan atau `IAU_2015:49910` untuk Mars).
- **Praproses (Disarankan):** DEM sebaiknya sudah melalui tahap *Sink Fill* agar batas kerucut lebih presisi.
            """
        )

    st.markdown("---")
    tampilkan_panduan_parameter()

else:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    rasters, transform, pixel_size, is_geographic, crs_text, skala_jendela = ekstrak_morfometri(uploaded_file, file_id)

    if is_geographic:
        st.error(
            f"⚠️ **Peringatan Sistem Koordinat ({crs_text}):** File DEM menggunakan proyeksi geografis (derajat), bukan metrik (meter). "
            "Harap lakukan *Reproject* ke proyeksi metrik terlebih dahulu."
        )

    pixel_area = pixel_size * pixel_size

    # 1. Thresholding Adaptif Berbasis Statistik DEM & Multi-Scale TPI
    slope_adaptif = float(np.nanpercentile(rasters["slope"], 75))

    mask_awal = (
        (rasters["slope"] > slope_adaptif) &
        (rasters["tpi"] > 0.3) &          # rasters["tpi"] sudah berupa Multi-Scale Z-Score
        (rasters["rel_height"] > 0.05)    # rasters["rel_height"] sudah berupa Multi-Scale RTP
    )

    mask_bersih = ndimage.binary_opening(mask_awal, structure=np.ones((3, 3)))
    mask_bersih = ndimage.binary_closing(mask_bersih, structure=np.ones((5, 5)))
    mask_bersih = ndimage.binary_fill_holes(mask_bersih)

    # 2. Pelabelan & Filter Berbasis Rasio Tak Berdimensi
    labeled_array, num_features = ndimage.label(mask_bersih)
    hasil = []
    mask_final = np.zeros_like(labeled_array)
    cone_id = 1
    gugur = {"noise": 0, "aspect": 0, "circ": 0, "hd": 0, "contrast": 0}

    for i in range(1, num_features + 1):
        blob = (labeled_array == i)
        n_pixels = np.sum(blob)

        # Hanya membuang noise mikro (< 10 piksel), tanpa batas maksimal piksel
        if n_pixels < 10:
            gugur["noise"] += 1
            continue

        # Rasio Kelonjongan (Aspect Ratio) - Tak Berdimensi
        ys, xs = np.where(blob)
        h_box = ys.max() - ys.min() + 1
        w_box = xs.max() - xs.min() + 1
        aspect_ratio = max(h_box, w_box) / min(h_box, w_box)
        if aspect_ratio > 2.5:
            gugur["aspect"] += 1
            continue

        # Kebulatan (Circularity) - Tak Berdimensi
        eroded = ndimage.binary_erosion(blob)
        boundary_pixels = np.sum(blob ^ eroded)
        area = n_pixels * pixel_area
        perimeter = boundary_pixels * pixel_size
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        if circularity < 0.45:
            gugur["circ"] += 1
            continue

        # Rasio Tinggi terhadap Diameter Ekuivalen (H/D) - Tak Berdimensi
        dem_vals = rasters["dem"][blob]
        z_min = np.nanmin(dem_vals)
        z_max = np.nanmax(dem_vals)
        height = z_max - z_min
        diameter_ekivalen = 2 * np.sqrt(area / np.pi)
        hd_ratio = height / diameter_ekivalen if diameter_ekivalen > 0 else 0
        if not (0.03 <= hd_ratio <= 0.40):
            gugur["hd"] += 1
            continue

        # Rasio Kontras Lereng (Slope Dalam / Slope Cincin Luar) - Tak Berdimensi
        mean_slope_blob = float(np.nanmean(rasters["slope"][blob]))
        inner = ndimage.binary_dilation(blob, iterations=3)
        outer = ndimage.binary_dilation(blob, iterations=8)
        ring = outer ^ inner
        mean_slope_ring = float(np.nanmean(rasters["slope"][ring])) if np.any(ring) else 0.0
        if np.isnan(mean_slope_ring):
            mean_slope_ring = 0.0

        slope_contrast = mean_slope_blob / (mean_slope_ring + 0.1)
        if slope_contrast < 1.25:
            gugur["contrast"] += 1
            continue

        # Perhitungan Volume & Koordinat Pusat Massa
        volume_m3 = float(np.nansum(dem_vals - z_min) * pixel_area)
        cy, cx = ndimage.center_of_mass(blob)
        utm_x, utm_y = transform * (cx, cy)

        mask_final[blob] = cone_id
        hasil.append({
            "ID_Kerucut": cone_id,
            "UTM_X": round(utm_x, 2),
            "UTM_Y": round(utm_y, 2),
            "Luas_m2": round(area, 2),
            "Tinggi_m": round(height, 2),
            "Volume_m3": round(volume_m3, 2),
            "Rasio_H_D": round(hd_ratio, 4),
            "Kontras_Slope": round(slope_contrast, 2),
            "Circularity": round(circularity, 3),
            "Aspect_Ratio": round(aspect_ratio, 2),
            "Mean_Slope_deg": round(mean_slope_blob, 2),
            "Surrounding_Slope": round(mean_slope_ring, 2),
            "Mean_TPI_Z": round(float(np.nanmean(rasters["tpi"][blob])), 2),
            "Mean_ProfCurv": round(float(np.nanmean(rasters["prof_curv"][blob])), 5),
            "Mean_PlanCurv": round(float(np.nanmean(rasters["plan_curv"][blob])), 5)
        })
        cone_id += 1

    df_hasil = pd.DataFrame(hasil)

    # Info Diagnostik Singkat
    st.caption(f"🔎 Skala jendela TPI/RTP otomatis (piksel): {skala_jendela} | Total kandidat awal: {num_features} | Gugur di filter: {gugur}")

    # --- RINGKASAN METRIK ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Kerucut Terdeteksi", f"{len(df_hasil)} objek")
    col2.metric("Slope Adaptif (P75)", f"{slope_adaptif:.1f}°")
    if len(df_hasil) > 0:
        col3.metric("Rata-rata Rasio H/D", f"{df_hasil['Rasio_H_D'].mean():.3f}")
        col4.metric("Total Volume", f"{df_hasil['Volume_m3'].sum() / 1e6:.2f} juta m³")
    else:
        col3.metric("Rata-rata Rasio H/D", "-")
        col4.metric("Total Volume", "-")

    # --- TAB HASIL & PANDUAN ---
    tab1, tab2, tab3 = st.tabs([
        "🗺️ Peta Hasil Deteksi",
        "📊 Tabel Morfometri & Unduh CSV",
        "📘 Metodologi Otomatis"
    ])

    with tab1:
        fig, ax = plt.subplots(figsize=(12, 8))
        # Downsampling otomatis untuk tampilan agar hemat RAM pada DEM besar
        step = max(1, rasters["slope"].shape[0] // 2000)
        ax.imshow(rasters["slope"][::step, ::step], cmap="gray", vmin=0, vmax=35)
        overlay = np.ma.masked_where(mask_final[::step, ::step] == 0, mask_final[::step, ::step])
        ax.imshow(overlay, cmap="autumn", alpha=0.75)
        ax.set_title(f"Hasil Deteksi Otomatis ({len(df_hasil)} Kerucut) | CRS: {crs_text}", fontsize=13)
        ax.axis("off")
        st.pyplot(fig)

    with tab2:
        if len(df_hasil) > 0:
            st.dataframe(df_hasil, use_container_width=True, hide_index=True)
            csv_data = df_hasil.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Unduh Hasil Inventarisasi (.CSV)",
                data=csv_data,
                file_name="inventarisasi_kerucut_multiskala.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.warning("Tidak ada objek yang memenuhi kriteria rasio kerucut vulkanik pada DEM ini.")

    with tab3:
        tampilkan_panduan_parameter()
