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
    page_title="Deteksi Kerucut Vulkanik Otomatis",
    page_icon="🌋",
    layout="wide"
)

# --- CUSTOM CSS & BANNER HEADER ---
st.markdown("""
<style>
    /* Banner Header Gradien */
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

    /* Kartu Metrik */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E5DDD5;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }

    /* Tombol Download lebih modern */
    .stDownloadButton button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
</style>

<div class="hero-banner">
    <h1>🌋 Sistem Deteksi Otomatis & Morfometri Kerucut Vulkanik</h1>
    <p>Ekstraksi otomatis kandidat kerucut, tinggi relatif, luas tapak, dan volume berbasis DEM & WhiteboxTools</p>
</div>
""", unsafe_allow_html=True)

# --- FUNGSI EKSTRAKSI MORFOMETRI (DI-CACHE) ---
@st.cache_data(show_spinner="Sedang mengekstrak parameter morfometri (Slope, TPI, Curvature, RTP)...")
def ekstrak_morfometri(file_bytes):
    tmpdir = tempfile.mkdtemp()
    dem_path = os.path.join(tmpdir, "dem_input.tif")
    slope_path = os.path.join(tmpdir, "slope.tif")
    tpi_path = os.path.join(tmpdir, "tpi.tif")
    rtp_path = os.path.join(tmpdir, "rtp.tif")
    prof_path = os.path.join(tmpdir, "prof_curv.tif")
    plan_path = os.path.join(tmpdir, "plan_curv.tif")

    with open(dem_path, "wb") as f:
        f.write(file_bytes)

    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)

    wbt.slope(dem_path, slope_path, units="degrees")
    wbt.diff_from_mean_elev(dem_path, tpi_path, filterx=11, filtery=11)
    wbt.relative_topographic_position(dem_path, rtp_path, filterx=11, filtery=11)
    wbt.profile_curvature(dem_path, prof_path)
    wbt.plan_curvature(dem_path, plan_path)

    paths = {
        "dem": dem_path,
        "slope": slope_path,
        "tpi": tpi_path,
        "rel_height": rtp_path,
        "prof_curv": prof_path,
        "plan_curv": plan_path
    }

    rasters = {}
    transform = None
    pixel_size = 30.0
    is_geographic = False
    crs_text = "Tidak diketahui"

    for key, path in paths.items():
        with rasterio.open(path) as src:
            arr = src.read(1).astype(float)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            rasters[key] = arr
            if transform is None:
                transform = src.transform
                pixel_size = abs(transform.a)
                if src.crs is not None:
                    is_geographic = src.crs.is_geographic
                    crs_text = str(src.crs)

    return rasters, transform, pixel_size, is_geographic, crs_text


# --- FUNGSI PANDUAN PARAMETER (BISA DIPANGGIL DI LANDING PAGE & TAB) ---
def tampilkan_panduan_parameter():
    st.subheader("📘 Makna Parameter & Saran Kalibrasi")
    st.markdown(
        "Secara *default*, aplikasi sudah berjalan otomatis dengan parameter optimal untuk dataran vulkanik monogenetik. "
        "Jika wilayah studi memiliki karakteristik topografi berbeda, gunakan panduan berikut untuk mengatur *slider* di panel kiri:"
    )

    data_panduan = [
        {
            "Parameter": "Slope Minimum (°)",
            "Makna Geologis": "Kemiringan lereng minimal yang dianggap sebagai badan kerucut vulkanik.",
            "Saran Pengaturan": "Turunkan (misal 6–8°) jika kerucut di area studi sudah tua/landai. Naikkan (10–12°) jika banyak bukit landai non-vulkanik ikut terdeteksi."
        },
        {
            "Parameter": "TPI Minimum",
            "Makna Geologis": "Topographic Position Index; membedakan tonjolan positif (bukit/kerucut) dari cekungan (lembah/kawah/maar).",
            "Saran Pengaturan": "Pertahankan > 0 agar kawah/lembah tidak masuk. Naikkan sedikit (0.1–0.3) jika area datar di kaki kerucut ikut terseret."
        },
        {
            "Parameter": "Relative Height (RTP)",
            "Makna Geologis": "Posisi elevasi relatif suatu piksel dibanding lingkungan sekitarnya (skala -1 sampai 1).",
            "Saran Pengaturan": "Nilai 0.05–0.15 efektif menangkap kerucut menonjol. Turunkan mendekati 0 jika bentuk kerucut terdeteksi terlalu tipis."
        },
        {
            "Parameter": "Luas Minimal & Maksimal (piksel)",
            "Makna Geologis": "Membatasi ukuran tapak kerucut agar noise mikro dan badan gunung api utama (stratovolcano) tidak ikut terhitung.",
            "Saran Pengaturan": "Sesuaikan dengan resolusi DEM. Pada DEM 30 m, 25 piksel ≈ 2,25 hektar. Turunkan jika ingin mendeteksi kerucut sangat kecil."
        },
        {
            "Parameter": "Circularity Minimal (0 - 1)",
            "Makna Geologis": "Indeks kebulatan bentuk tapak (1 = lingkaran sempurna). Kerucut vulkanik cenderung membulat dibanding punggungan erosi.",
            "Saran Pengaturan": "Gunakan 0.45–0.55. Naikkan jika banyak punggungan bukit tidak beraturan lolos; turunkan jika kerucut tererosi/bercelah banyak terbuang."
        },
        {
            "Parameter": "Aspect Ratio Maksimal",
            "Makna Geologis": "Rasio panjang terhadap lebar objek. Mencegah igir/punggungan bukit yang memanjang terdeteksi sebagai kerucut.",
            "Saran Pengaturan": "Nilai ideal 1.8–2.2. Semakin kecil angkanya, semakin ketat penyaringan terhadap objek lonjong."
        },
        {
            "Parameter": "Tinggi Minimal (m)",
            "Makna Geologis": "Selisih elevasi tertinggi (puncak) dan terendah (dasar) pada satu objek kandidat.",
            "Saran Pengaturan": "Gunakan 15–25 m untuk menyaring gundukan kecil. Naikkan untuk area dengan skala kerucut raksasa (misal Mars)."
        },
        {
            "Parameter": "Maks. Slope Sekitar / Isolasi (°)",
            "Makna Geologis": "Rata-rata kemiringan cincin di sekeliling objek. Membedakan kerucut terisolasi di dataran dari punggungan gunung curam.",
            "Saran Pengaturan": "Kunci utama membuang false positive di pegunungan tua! Turunkan (6.5–7.5°) jika lereng erosi masih ramai; naikkan (9–11°) jika kerucut di lereng gunung utama ikut hilang."
        }
    ]
    st.dataframe(pd.DataFrame(data_panduan), use_container_width=True, hide_index=True)

    st.info(
        "**💡 Tips Cepat Kalibrasi:**\n"
        "- **Terlalu banyak salah deteksi (*False Positives*) di area pegunungan:** Turunkan *Maks. Slope Sekitar* atau naikkan *Circularity* & *TPI*.\n"
        "- **Kerucut asli banyak yang hilang (*False Negatives*):** Turunkan *Slope Minimum*, *Luas Minimal*, atau *Tinggi Minimal*."
    )


# --- SIDEBAR: UPLOAD & PENGATURAN PARAMETER ---
st.sidebar.header("1. Unggah Data DEM")
uploaded_file = st.sidebar.file_uploader(
    "Pilih file DEM (.tif / .tiff)",
    type=["tif", "tiff"],
    help="Wajib menggunakan sistem koordinat proyeksi metrik (misal UTM) agar perhitungan luas dan volume akurat."
)

st.sidebar.markdown("---")
st.sidebar.header("2. Kalibrasi Parameter")
st.sidebar.caption("Aplikasi berjalan otomatis dengan nilai default. Buka menu di bawah jika ingin melakukan penyesuaian manual.")

with st.sidebar.expander("⚙️ Pengaturan Lanjutan (Opsional)", expanded=False):
    st.markdown("**A. Parameter Morfometri Dasar**")
    slope_min = st.slider(
        "Slope Minimum (°)", 2.0, 25.0, 8.5, 0.5,
        help="Batas kemiringan lereng minimal untuk badan kerucut."
    )
    tpi_min = st.slider(
        "TPI Minimum", -1.0, 5.0, 0.1, 0.1,
        help="Topographic Position Index; nilai > 0 menandakan bentuk menonjol ke atas."
    )
    rtp_min = st.slider(
        "Relative Height (RTP) Minimum", -0.5, 0.5, 0.08, 0.01,
        help="Posisi ketinggian relatif terhadap lingkungan sekitar."
    )

    st.markdown("**B. Filter Bentuk & Ukuran**")
    min_pixels = st.slider(
        "Luas Minimal (piksel)", 5, 150, 25, 5,
        help="Menyaring noise berukuran terlalu kecil."
    )
    max_pixels = st.slider(
        "Luas Maksimal (piksel)", 500, 10000, 4000, 100,
        help="Mencegah badan gunung api utama (stratovolcano) ikut terdeteksi."
    )
    min_circ = st.slider(
        "Circularity Minimal (0 - 1)", 0.1, 0.9, 0.50, 0.05,
        help="Tingkat kebulatan objek (1 = lingkaran sempurna)."
    )
    max_aspect = st.slider(
        "Aspect Ratio Maksimal", 1.2, 4.0, 2.0, 0.1,
        help="Rasio panjang vs lebar; membuang punggungan bukit yang lonjong."
    )
    min_height = st.slider(
        "Tinggi Kerucut Minimal (m)", 5.0, 100.0, 18.0, 1.0,
        help="Selisih elevasi puncak dan dasar kerucut minimal."
    )

    st.markdown("**C. Filter Isolasi Lingkungan**")
    max_ring_slope = st.slider(
        "Maksimal Slope Sekitar (°)", 3.0, 20.0, 8.0, 0.5,
        help="Rata-rata kemiringan area cincin di luar objek untuk membuang punggungan pegunungan curam."
    )


# --- TAMPILAN UTAMA ---
if uploaded_file is None:
    # HALAMAN AWAL (SEBELUM UPLOAD)
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🚀 Langkah-Langkah Penggunaan")
        st.markdown(
            """
            1. **Siapkan File DEM:** Pastikan file DEM sudah memenuhi syarat di sebelah kanan.
            2. **Unggah di Panel Kiri:** Klik tombol *Browse files* pada bagian **1. Unggah Data DEM**.
            3. **Ekstraksi Otomatis:** Sistem akan otomatis menghitung *Slope*, *TPI*, *Curvature*, dan *Relative Height*, lalu mendeteksi kandidat kerucut vulkanik.
            4. **Kalibrasi (Opsional):** Jika hasil belum sesuai karakteristik topografi lokal, buka menu **⚙️ Pengaturan Lanjutan** di panel kiri.
            5. **Unduh Hasil:** Tinjau peta deteksi dan unduh tabel morfometri lengkap dalam format `.csv`.
            """
        )

    with col_b:
        st.subheader("📋 Syarat File DEM yang Diunggah")
        st.markdown(
            """
            - **Format File:** GeoTIFF (`.tif` atau `.tiff`).
            - **Sistem Koordinat (CRS):** Wajib menggunakan **proyeksi metrik** (seperti UTM, misal `EPSG:32749`) agar satuan piksel dalam meter, sehingga hitungan luas ($\text{m}^2$) dan volume ($\text{m}^3$) valid.
            - **Praproses (Disarankan):** DEM sebaiknya sudah melalui tahap *Sink Fill* (pengisian cekungan semu) dan *Filtering* (penghalusan *noise*) agar batas kerucut lebih presisi.
            - **Resolusi Spasial:** Optimal untuk resolusi menengah (misal 10–30 meter/piksel).
            """
        )

    st.markdown("---")
    tampilkan_panduan_parameter()

else:
    # PROSES DETEKSI SETELAH FILE DIUNGGAH
    rasters, transform, pixel_size, is_geographic, crs_text = ekstrak_morfometri(uploaded_file.getvalue())

    if is_geographic:
        st.error(
            f"⚠️ **Peringatan Sistem Koordinat ({crs_text}):** File DEM yang diunggah menggunakan proyeksi geografis (derajat), bukan metrik (meter). "
            "Hasil perhitungan Luas dan Volume tidak akan akurat. Harap lakukan *Reproject* ke proyeksi UTM terlebih dahulu."
        )

    pixel_area = pixel_size * pixel_size

    # 1. Thresholding & Pembersihan Morfologi
    mask_awal = (
        (rasters["slope"] > slope_min) &
        (rasters["tpi"] > tpi_min) &
        (rasters["rel_height"] > rtp_min)
    )
    mask_bersih = ndimage.binary_opening(mask_awal, structure=np.ones((3, 3)))
    mask_bersih = ndimage.binary_closing(mask_bersih, structure=np.ones((5, 5)))
    mask_bersih = ndimage.binary_fill_holes(mask_bersih)

    # 2. Pelabelan & Filter Objek
    labeled_array, num_features = ndimage.label(mask_bersih)
    hasil = []
    mask_final = np.zeros_like(labeled_array)
    cone_id = 1

    for i in range(1, num_features + 1):
        blob = (labeled_array == i)
        n_pixels = np.sum(blob)

        if n_pixels < min_pixels or n_pixels > max_pixels:
            continue

        ys, xs = np.where(blob)
        h_box = ys.max() - ys.min() + 1
        w_box = xs.max() - xs.min() + 1
        aspect_ratio = max(h_box, w_box) / min(h_box, w_box)
        if aspect_ratio > max_aspect:
            continue

        eroded = ndimage.binary_erosion(blob)
        boundary_pixels = np.sum(blob ^ eroded)
        area = n_pixels * pixel_area
        perimeter = boundary_pixels * pixel_size
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        if circularity < min_circ:
            continue

        dem_vals = rasters["dem"][blob]
        z_min = np.nanmin(dem_vals)
        z_max = np.nanmax(dem_vals)
        height = z_max - z_min
        if height < min_height:
            continue

        inner = ndimage.binary_dilation(blob, iterations=3)
        outer = ndimage.binary_dilation(blob, iterations=8)
        ring = outer ^ inner
        mean_slope_ring = np.nanmean(rasters["slope"][ring])
        if mean_slope_ring > max_ring_slope:
            continue

        volume_m3 = np.nansum(dem_vals - z_min) * pixel_area
        cy, cx = ndimage.center_of_mass(blob)
        utm_x, utm_y = transform * (cx, cy)

        mask_final[blob] = cone_id
        hasil.append({
            "ID_Kerucut": cone_id,
            "UTM_X": round(utm_x, 2),
            "UTM_Y": round(utm_y, 2),
            "Luas_m2": round(area, 2),
            "Circularity": round(circularity, 3),
            "Tinggi_m": round(height, 2),
            "Volume_m3": round(volume_m3, 2),
            "Mean_Slope_deg": round(np.nanmean(rasters["slope"][blob]), 2),
            "Surrounding_Slope": round(mean_slope_ring, 2),
            "Mean_TPI": round(np.nanmean(rasters["tpi"][blob]), 2),
            "Mean_ProfCurv": round(np.nanmean(rasters["prof_curv"][blob]), 5),
            "Mean_PlanCurv": round(np.nanmean(rasters["plan_curv"][blob]), 5)
        })
        cone_id += 1

    df_hasil = pd.DataFrame(hasil)

    # --- RINGKASAN METRIK ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Kerucut Terdeteksi", f"{len(df_hasil)} objek")
    col2.metric("Resolusi DEM", f"{pixel_size:.1f} m")
    if len(df_hasil) > 0:
        col3.metric("Rata-rata Tinggi", f"{df_hasil['Tinggi_m'].mean():.1f} m")
        col4.metric("Total Volume", f"{df_hasil['Volume_m3'].sum() / 1e6:.2f} juta m³")
    else:
        col3.metric("Rata-rata Tinggi", "-")
        col4.metric("Total Volume", "-")

    # --- TAB HASIL & PANDUAN ---
    tab1, tab2, tab3 = st.tabs([
        "🗺️ Peta Hasil Deteksi",
        "📊 Tabel Morfometri & Unduh CSV",
        "📘 Panduan Parameter & Kalibrasi"
    ])

    with tab1:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(rasters["slope"], cmap="gray", vmin=0, vmax=35)
        overlay = np.ma.masked_where(mask_final == 0, mask_final)
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
                file_name="inventarisasi_kerucut_vulkanik.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.warning("Tidak ada kerucut yang lolos filter. Buka menu **⚙️ Pengaturan Lanjutan** di panel kiri untuk melonggarkan parameter.")

    with tab3:
        tampilkan_panduan_parameter()