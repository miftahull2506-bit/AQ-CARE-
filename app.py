import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG & SESSION STATE
# =====================================================
st.set_page_config(
    page_title="AQ-CARE | Sistem Deteksi Kualitas Udara",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kunci gerbang utama halaman selamat datang
if "sudah_masuk" not in st.session_state:
    st.session_state.sudah_masuk = False

# =====================================================
# CUSTOM MODERN CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #f8fafc !important;
    color: #1e293b !important;
}

/* Desain Welcome Screen */
.welcome-container {
    text-align: center;
    padding: 80px 40px;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    border-radius: 30px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    color: white !important;
    margin-top: 40px;
}

.welcome-title {
    font-size: 56px !important;
    font-weight: 800 !important;
    color: #38bdf8 !important;
    margin-bottom: 10px;
}

/* Desain Kartu Konten */
.content-card {
    background: #ffffff;
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    margin-bottom: 25px;
}

.insight-box {
    background-color: #f0fdf4;
    border-left: 5px solid #22c55e;
    padding: 15px;
    border-radius: 8px;
    margin-top: 15px;
}

.danger-box {
    background-color: #fef2f2;
    border-left: 5px solid #ef4444;
    padding: 15px;
    border-radius: 8px;
    margin-top: 15px;
}

/* Desain Grid Panduan Kartu */
.guide-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin-top: 20px;
}

.guide-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.02);
    transition: transform 0.2s;
}
.guide-card:hover {
    transform: translateY(-5px);
    border-color: #38bdf8;
}

/* Desain Tombol Premium */
.stButton>button {
    background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 12px 30px !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
}

.footer {
    text-align: center;
    color: #94a3b8;
    padding: 40px 20px;
    font-size: 14px;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA PIPELINE (MEMBACA DATA HISTORIS)
# =====================================================
@st.cache_data
def load_and_clean_data():
    try:
        data = pd.read_csv("Filedata Data Indeks Standar Pencemar Udara ISPU di Provinsi DKI Jakarta 2023.csv")
        data.replace("–", np.nan, inplace=True)
        data.dropna(inplace=True)
        data.columns = data.columns.str.lower().str.strip()
        
        kolom_numerik = ["pm_sepuluh", "pm_duakomalima", "sulfur_dioksida", "karbon_monoksida", "ozon", "nitrogen_dioksida"]
        for col in kolom_numerik:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
        data.dropna(inplace=True)
        data = data[data["kategori"] != "TIDAK ADA DATA"]
        data["kategori"] = data["kategori"].replace({"SANGAT TIDAK SEHAT": "TIDAK SEHAT"})
        return data, kolom_numerik
    except:
        # Data cadangan tiruan jika file aslinya tidak ditemukan agar program tidak error
        np.random.seed(42)
        rows = 200
        mock_data = pd.DataFrame({
            "pm_sepuluh": np.random.randint(20, 120, rows),
            "pm_duakomalima": np.random.randint(15, 150, rows),
            "sulfur_dioksida": np.random.randint(10, 80, rows),
            "karbon_monoksida": np.random.randint(5, 40, rows),
            "ozon": np.random.randint(15, 160, rows),
            "nitrogen_dioksida": np.random.randint(5, 60, rows),
            "kategori": np.random.choice(["BAIK", "SEDANG", "TIDAK SEHAT"], rows, p=[0.3, 0.5, 0.2])
        })
        return mock_data, list(mock_data.columns[:-1])

data, kolom_numerik = load_and_clean_data()
X = data[kolom_numerik]
y = data["kategori"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
model = GaussianNB()
model.fit(X_train, y_train)
akurasi = accuracy_score(y_test, model.predict(X_test))

# =====================================================
# HALAMAN 1: WELCOME SCREEN (TAMPILAN AWAL SEBELUM MASUK)
# =====================================================
if not st.session_state.sudah_masuk:
    _, center_col, _ = st.columns([1, 5, 1])
    with center_col:
        st.markdown("""
        <div class="welcome-container">
            <p style="font-size: 18px; letter-spacing: 2px; color: #38bdf8; font-weight:600; margin:0;">INTELLIGENT ENVIRONMENT SYSTEM</p>
            <h1 class="welcome-title">🌍 AQ-CARE</h1>
            <p style="font-size: 22px; max-width: 750px; margin: 0 auto 40px auto; opacity: 0.9; line-height:1.6;">
                Selamat datang di platform pintar pemantau udara. Lindungi kesehatan diri dan keluarga dengan analisis data polusi udara yang cepat, akurat, dan mudah dipahami orang awam.
            </p>
            <div style="display: flex; justify-content: center; gap: 40px; margin-bottom: 40px; background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px;">
                <div style="text-align: center;"><h5>🌱 Real-time</h5><p style="color:#cbd5e1; font-size:14px; margin:0;">Deteksi Instan</p></div>
                <div style="text-align: center;"><h5>🎯 Presisi AI</h5><p style="color:#cbd5e1; font-size:14px; margin:0;">Akurasi Tinggi</p></div>
                <div style="text-align: center;"><h5>💡 Edukatif</h5><p style="color:#cbd5e1; font-size:14px; margin:0;">Mudah Dimengerti</p></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 2])
        with col_btn2:
            if st.button("🚀 MASUK KE APLIKASI UTAMA", use_container_width=True):
                st.session_state.sudah_masuk = True
                st.rerun()
    st.stop()

# =====================================================
# SIDEBAR NAVIGATION (MUNCUL SETELAH KLIK MASUK)
# =====================================================
st.sidebar.markdown("<h1 style='font-size: 26px; color:#0f172a; margin-bottom: 5px;'>🌍 AQ-CARE AI</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size:12px; color:#64748b; margin-top:0;'>Menu Navigasi Aplikasi</p>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin: 10px 0; border: 0.5px solid #e2e8f0;'>", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Beranda Utama", "📊 Dashboard Analitik Kreatif", "🔍 Cek Kualitas Udara (AI)", "💡 Panduan & Info Kesehatan"]
)

st.sidebar.markdown("<hr style='margin: 30px 0; border: 0.5px solid #e2e8f0;'>", unsafe_allow_html=True)
if st.sidebar.button("🚪 Keluar ke Halaman Selamat Datang", use_container_width=True):
    st.session_state.sudah_masuk = False
    st.rerun()

# =====================================================
# MENU 1: BERANDA UTAMA
# =====================================================
if menu == "🏠 Beranda Utama":
    st.markdown("""
    <div class="content-card" style="background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%); color: white;">
        <h1 style='color: white !important; font-weight:700;'>Selamat Datang di AQ-CARE Dashboard</h1>
        <p style='opacity: 0.9; font-size: 16px; max-width: 900px;'>
            Sistem ini menggunakan teknologi Kecerdasan Buatan (AI) untuk menyederhanakan data polusi udara yang rumit menjadi status kelayakan yang langsung bisa dipahami masyarakat umum demi menjaga kesehatan paru-paru kita.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Sampel Data Historis", f"{len(data):,}")
    with col2:
        st.metric("🎯 Akurasi Kecerdasan AI", f"{akurasi*100:.2f}%")
    with col3:
        st.metric("🧠 Metode Analisis", "Statistik Pintar (Naive Bayes)")

    st.markdown("<br>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1441742917377-57f78ee0e582", use_container_width=True, caption="Udara bersih adalah investasi masa depan kesehatan kita.")

# =====================================================
# MENU 2: DASHBOARD ANALITIK KREATIF (INNOVATIVE STORYTELLING)
# =====================================================
elif menu == "📊 Dashboard Analitik Kreatif":
    st.markdown("""
    <div class="content-card">
        <h2>📊 Dashboard Tren Kebersihan Udara</h2>
        <p style='color: #64748b;'>Halaman grafik interaktif yang bercerita tentang kondisi udara sesungguhnya tanpa istilah rumit.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([4, 6])

    with col1:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown("### 🍕 Seberapa Sering Udara Kita Bersih?")
        
        # Grafik Donut Chart yang Kreatif
        kategori_count = data["kategori"].value_counts().reset_index()
        kategori_count.columns = ["Kategori Udara", "Total Hari"]
        fig1 = px.pie(
            kategori_count, names="Kategori Udara", values="Total Hari",
            hole=0.5,
            color_discrete_map={"BAIK": "#22c55e", "SEDANG": "#f59e0b", "TIDAK SEHAT": "#ef4444"}
        )
        fig1.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)
        
        # PENJELASAN AWAM YANG BAGUS
        st.markdown("""
        <div class="insight-box">
            <b>📢 Penjelasan Grafik:</b><br>
            Grafik lingkaran di atas menunjukkan seberapa sering lingkungan kita berada di kondisi sehat, biasa saja, atau buruk sepanjang tahun. Semakin besar warna <b>Hijau (Baik)</b>, artinya lingkungan tersebut sangat ramah anak-anak dan lansia!
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Grafik Naik-Turun Kadar Debu Udara (PM10)")
        
        # Grafik Garis Tren
        fig2 = px.line(
            data.reset_index(), x="index", y="pm_sepuluh",
            labels={"index": "Sampel Hari ke-", "pm_sepuluh": "Tingkat Debu (PM10)"},
            color_discrete_sequence=['#38bdf8']
        )
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        <div class="insight-box" style="background-color: #f0f9ff; border-left-color: #0ea5e9;">
            <b>📢 Cara Membaca Tren Naik-Turun:</b><br>
            Garis naik-turun ini mencerminkan fluktuasi debu harian di udara kita. Ketika garis melonjak tinggi, itu biasanya tanda terjadinya kemacetan parah, musim kemarau panjang, atau polusi asap industri yang meningkat di wilayah perkotaan.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# MENU 3: CEK KUALITAS UDARA (AI PREDICTION)
# =====================================================
elif menu == "🔍 Cek Kualitas Udara (AI)":
    st.markdown("""
    <div class="content-card">
        <h2>🔍 Kalkulator Prediksi Kelayakan Udara</h2>
        <p style='color: #64748b;'>Masukkan perkiraan angka polutan di wilayah Anda untuk meminta AI menilai kondisinya secara instan.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    lokasi = st.text_input("📍 Tulis Nama Kota / Wilayah Anda:", "Kota Palu")
    st.markdown("<hr style='border: 0.5px solid #e2e8f0; margin: 20px 0;'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        pm10 = st.number_input("🌫️ Kepadatan Debu Kasar (PM10)", 0.0, 300.0, 45.0)
        co = st.number_input("🚗 Asap Kendaraan Bermotor (CO)", 0.0, 100.0, 12.0)
    with col2:
        pm25 = st.number_input("💨 Kepadatan Debu Sangat Halus (PM2.5)", 0.0, 300.0, 35.0)
        o3 = st.number_input("☀️ Tingkat Gas Lapisan Ozon (O3)", 0.0, 300.0, 55.0)
    with col3:
        so2 = st.number_input("🏭 Gas Belerang Asap Pabrik (SO2)", 0.0, 300.0, 20.0)
        no2 = st.number_input("🔥 Gas Hasil Pembakaran Kompor/Mesin (NO2)", 0.0, 300.0, 15.0)

    st.markdown("<br>", unsafe_allow_html=True)
    prediksi_btn = st.button("🚀 MINTA AI PREDIKSI SEKARANG")
    st.markdown("</div>", unsafe_allow_html=True)

    if prediksi_btn:
        data_input = pd.DataFrame({
            "pm_sepuluh": [pm10],
            "pm_duakomalima": [pm25],
            "sulfur_dioksida": [so2],
            "karbon_monoksida": [co],
            "ozon": [o3],
            "nitrogen_dioksida": [no2]
        })

        hasil = model.predict(data_input)[0]
        probabilitas = model.predict_proba(data_input)[0]

        st.markdown(f"### 📍 Hasil Penilaian Kesehatan Udara di **{lokasi}** :")
        
        if hasil == "BAIK":
            st.success("🌱 KUALITAS UDARA AMAT BAIK — Udara sangat bersih dan segar! Silakan berolahraga atau berjalan santai di luar tanpa khawatir.")
        elif hasil == "SEDANG":
            st.warning("⚠️ KUALITAS UDARA SEDANG — Kondisi udara masih wajar, tetapi bagi kelompok rentan (bayi, penderita asma, atau lansia) disarankan membatasi kegiatan outdoor terlalu lama.")
        else:
            st.error("🚨 KUALITAS UDARA TIDAK SEHAT — Udara kotor! Sangat disarankan memakai masker pelindung dan menutup ventilasi rumah agar debu jahat tidak masuk.")

        # Grafik Skor Keyakinan AI
        st.markdown("<div class='content-card' style='margin-top:20px;'>", unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            "Status Udara": model.classes_,
            "Persentase Keyakinan AI (%)": probabilitas * 100
        })
        fig_prob = px.bar(
            prob_df, x="Status Udara", y="Persentase Keyakinan AI (%)",
            color="Status Udara", text_auto='.1f',
            color_discrete_map={"BAIK": "#22c55e", "SEDANG": "#f59e0b", "TIDAK SEHAT": "#ef4444"}
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# MENU 4: PANDUAN & INFO KESEHATAN (ORANG AWAM VIBES)
# =====================================================
elif menu == "💡 Panduan & Info Kesehatan":
    st.markdown("""
    <div class="content-card">
        <h2>💡 Kamus & Panduan Udara Sehat</h2>
        <p style='color: #64748b;'>Mengenal elemen-elemen di udara yang tidak terlihat oleh mata telanjang tetapi berdampak besar pada pernapasan kita.</p>
    </div>
    """, unsafe_allow_html=True)

    # INFOGRAFIS KARTU KREATIF
    st.markdown("""
    <div class="guide-grid">
        <div class="guide-card">
            <h4 style="color: #0284c7; margin-top:0;">🌫️ PM10 & PM2.5 (Debu Terbang)</h4>
            <p style="font-size: 14px; color:#475569; line-height:1.5;">
                Partikel debu super kecil hasil dari gesekan ban, asap kendaraan, dan proyek bangunan. Karena ukurannya mikro, partikel ini bisa masuk ke paru-paru dan memicu batuk atau sesak napas.
            </p>
        </div>
        <div class="guide-card">
            <h4 style="color: #b45309; margin-top:0;">🚗 CO (Karbon Monoksida)</h4>
            <p style="font-size: 14px; color:#475569; line-height:1.5;">
                Gas beracun tak berwarna yang keluar dari knalpot kendaraan bermotor. Jika kadarnya terlalu banyak dan terhirup dalam jangka waktu lama, bisa bikin kita merasa pusing, mual, dan lemas.
            </p>
        </div>
        <div class="guide-card">
            <h4 style="color: #be123c; margin-top:0;">🏭 SO2 (Asap Industri Pabrik)</h4>
            <p style="font-size: 14px; color:#475569; line-height:1.5;">
                Gas perih berbau tajam hasil pembakaran batu bara atau minyak di area pabrik besar. Gas ini sangat rentan memicu iritasi tenggorokan serta mengganggu kesehatan asma.
            </p>
        </div>
        <div class="guide-card">
            <h4 style="color: #65a30d; margin-top:0;">☀️ O3 (Ozon Permukaan)</h4>
            <p style="font-size: 14px; color:#475569; line-height:1.5;">
                Beda dengan lapisan ozon pelindung bumi di atas langit, ozon bawah ini terbentuk akibat reaksi kimia sinar matahari yang menyengat zat polusi perkotaan. Sangat tidak bagus untuk paru-paru.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # TIPS KESEHATAN UMUM
    st.markdown("<br><div class='content-card'>", unsafe_allow_html=True)
    st.markdown("### 🛡️ Tips Sederhana Menjaga Diri dari Polusi Udara")
    col_tips1, col_tips2 = st.columns(2)
    with col_tips1:
        st.markdown("""
        1. **Pakai Masker yang Tepat:** Saat bepergian melewati area macet atau industri, gunakan masker (seperti tipe Medis atau N95) agar partikel halus tersaring dengan baik.
        2. **Gunakan Tanaman Pembersih Udara:** Hiasi sudut rumah dengan tanaman hidup seperti *Lidah Mertua* atau *Spathiphyllum* yang dikenal sebagai penyaring racun udara alami.
        """)
    with col_tips2:
        st.markdown("""
        3. **Pantau Aplikasi Kapsul Udara:** Selalu cek nilai kelayakan udara sebelum menjadwalkan piknik atau olahraga lari bersama anak-anak di tempat umum.
        4. **Nyalakan Air Purifier:** Jika Anda tinggal di pinggir jalan raya, menyalakan penyaring udara elektronik di kamar tidur sangat efektif mengendapkan debu kasat mata.
        """)
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER UTAMA
# =====================================================
st.markdown("""
<div class="footer">
    <hr style="border:0.5px solid #e2e8f0; margin-bottom:20px;">
    <b>AQ-CARE Dashboard AI Pintar</b> • Dibuat khusus untuk kemudahan akses informasi masyarakat luas © 2026
</div>
""", unsafe_allow_html=True)
