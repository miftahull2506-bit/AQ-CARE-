import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AQ-CARE | Smart Air Quality",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM MODERN CSS (Google & Premium Tech Vibe)
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* GLOBAL APP STYLES */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #f8fafc !important;
    color: #1e293b !important;
}

/* SIDEBAR RE-DESIGN */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0;
    box-shadow: 4px 0 10px rgba(0,0,0,0.02);
}
section[data-testid="stSidebar"] .stMarkdown h1 {
    color: #0f172a !important;
    font-weight: 700;
}

/* PREMIUM HERO CARD */
.hero-card {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    padding: 45px;
    border-radius: 24px;
    box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.2);
    margin-bottom: 30px;
    color: white !important;
}
.hero-card h1, .hero-card h2, .hero-card h3, .hero-card p {
    color: white !important;
}

/* CONTENT CARD */
.content-card {
    background: #ffffff;
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    margin-bottom: 25px;
}
.content-card h1, .content-card h2, .content-card h3 {
    color: #0f172a !important;
    font-weight: 600;
}

/* METRIC CARDS */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 700 !important;
}

/* MODERN BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    height: 52px !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
    transition: all 0.3s ease !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.3) !important;
}

/* INPUT COMPONENT FIXES */
.stTextInput input, .stNumberInput input {
    border-radius: 10px !important;
    border: 1px solid #cbd5e1 !important;
    background-color: #ffffff !important;
    color: #0f172a !important;
}
label {
    color: #475569 !important;
    font-weight: 500 !important;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #94a3b8;
    padding: 40px 20px;
    font-size: 14px;
    border-top: 1px solid #e2e8f0;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA & ML PIPELINE (Sama seperti logika Anda)
# =====================================================
@st.cache_data
def load_and_clean_data():
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

try:
    data, kolom_numerik = load_and_clean_data()
    X = data[kolom_numerik]
    y = data["kategori"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
    model = GaussianNB()
    model.fit(X_train, y_train)
    akurasi = accuracy_score(y_test, model.predict(X_test))
except Exception as e:
    st.error(f"Gagal memuat data. Pastikan file CSV tersedia. Error: {e}")
    st.stop()

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.markdown("<h1 style='font-size: 26px; margin-bottom: 25px;'>🌍 AQ-CARE</h1>", unsafe_allow_html=True)
menu = st.sidebar.radio(
    "Menu Navigasi",
    ["🏠 Beranda", "📊 Dashboard Analitik", "🔍 Prediksi Kualitas", "ℹ️ Informasi Sistem"],
    label_visibility="collapsed"
)

# =====================================================
# HOME PAGE
# =====================================================
if menu == "🏠 Beranda":
    st.markdown("""
    <div class="hero-card">
        <h1 style='font-size: 38px; margin-bottom: 10px; font-weight: 700;'>🌍 AQ-CARE</h1>
        <h3 style='font-size: 20px; font-weight: 400; opacity: 0.9; margin-bottom: 20px;'>Smart Air Quality Prediction System</h3>
        <p style='font-size: 16px; max-width: 800px; line-height: 1.6; opacity: 0.85;'>
            AQ-CARE memanfaatkan kecerdasan buatan (Machine Learning) dengan algoritma Gaussian Naive Bayes 
            untuk mengklasifikasikan dan memprediksi tingkat kelayakan udara secara presisi, cerdas, dan real-time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Section
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Sampel Data", f"{len(data):,}")
    col2.metric("🎯 Akurasi Pengujian", f"{akurasi*100:.2f}%")
    col3.metric("🧠 Intelijen Model", "Gaussian NB")

    st.markdown("<br>", unsafe_allow_html=True)
    st.image(
        "https://images.unsplash.com/photo-1519608487953-e999c86e7455", 
        use_container_width=True, 
        caption="Mari bersama jaga langit tetap biru dan udara tetap bersih."
    )

# =====================================================
# DASHBOARD PAGE
# =====================================================
elif menu == "📊 Dashboard Analitik":
    st.markdown("""
    <div class="content-card">
        <h2>📊 Dashboard Analitik Kualitas Udara</h2>
        <p style='color: #64748b;'>Visualisasi ringkas dari distribusi data ISPU untuk memantau tren pencemaran lingkungan.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        kategori_count = data["kategori"].value_counts().reset_index()
        kategori_count.columns = ["Kategori", "Jumlah"]
        
        fig1 = px.pie(
            kategori_count, names="Kategori", values="Jumlah",
            title="<b>Proporsi Kategori Udara</b>",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#0f172a')
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        fig2 = px.histogram(
            data, x="pm_sepuluh", nbins=25,
            title="<b>Distribusi Konsentrasi PM10</b>",
            color_discrete_sequence=['#0ea5e9']
        )
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#0f172a')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PREDICTION PAGE
# =====================================================
elif menu == "🔍 Prediksi Kualitas":
    st.markdown("""
    <div class="content-card">
        <h2>🔍 Kalkulator Prediksi Kualitas Udara</h2>
        <p style='color: #64748b;'>Masukkan parameter konsentrasi polutan di bawah ini untuk mendapatkan hasil analisis instan.</p>
    </div>
    """, unsafe_allow_html=True)

    # Mengelompokkan input ke dalam Card yang rapi
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    
    lokasi = st.text_input("📍 Nama Lokasi Pengujian", "Jakarta Pusat, DKI Jakarta")
    
    st.markdown("<hr style='border: 0.5px solid #e2e8f0; margin: 20px 0;'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        pm10 = st.number_input("🌫️ PM10 (µg/m³)", 0.0, 500.0, 50.0, step=1.0)
        co = st.number_input("🚗 CO (µg/m³)", 0.0, 100.0, 10.0, step=0.5)
    with col2:
        pm25 = st.number_input("💨 PM2.5 (µg/m³)", 0.0, 500.0, 70.0, step=1.0)
        o3 = st.number_input("☀️ O3 (µg/m³)", 0.0, 500.0, 20.0, step=1.0)
    with col3:
        so2 = st.number_input("🏭 SO2 (µg/m³)", 0.0, 500.0, 30.0, step=1.0)
        no2 = st.number_input("🔥 NO2 (µg/m³)", 0.0, 500.0, 15.0, step=1.0)

    st.markdown("<br>", unsafe_allow_html=True)
    prediksi_btn = st.button("🚀 Jalankan Prediksi Kecerdasan Buatan")
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

        # Hasil Prediksi Banner
        st.markdown(f"### 📍 Hasil Analisis untuk {lokasi}:")
        if hasil == "BAIK":
            st.success(f"🌱 **KUALITAS UDARA AMAT BAIK** — Udara sangat segar dan aman untuk beraktivitas di luar ruangan.")
        elif hasil == "SEDANG":
            st.warning(f"⚠️ **KUALITAS UDARA SEDANG** — Kualitas udara berstatus wajar namun sensitif bagi sebagian kelompok rentan.")
        else:
            st.error(f"🚨 **KUALITAS UDARA TIDAK SEHAT** — Sangat disarankan menggunakan masker dan mengurangi mobilitas luar.")

        # Chart Probabilitas yang Lebih Bersih
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            "Kategori": model.classes_,
            "Probabilitas Keyakinan": probabilitas
        })

        fig = px.bar(
            prob_df, x="Kategori", y="Probabilitas Keyakinan",
            color="Kategori", text_auto='.2f',
            title="<b>Tingkat Keyakinan Prediksi (Confidence Score)</b>",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#0f172a'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# ABOUT PAGE
# =====================================================
elif menu == "ℹ️ Informasi Sistem":
    st.markdown("""
    <div class="content-card">
        <h2>ℹ️ Tentang AQ-CARE</h2>
        <p style='line-height: 1.6; color: #475569;'>
            Sistem ini dibangun dengan dedikasi tinggi untuk memberikan informasi prediktif yang akurat mengenai indeks standar pencemar udara (ISPU). 
            Menggunakan riset komparatif berbasis data historis DKI Jakarta 2023.
        </p>
        
        <h3 style='margin-top: 30px;'>📌 Variabel Polutan Utama</h3>
        <p style='color: #475569;'>Sistem mendeteksi 6 unsur senyawa kritikal berbahaya:</p>
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 20px;'>
            <div style='padding:15px; background:#f1f5f9; border-radius:10px; text-align:center;'><b>PM10 & PM2.5</b><br>Partikulat Debu</div>
            <div style='padding:15px; background:#f1f5f9; border-radius:10px; text-align:center;'><b>SO2</b><br>Sulfur Dioksida</div>
            <div style='padding:15px; background:#f1f5f9; border-radius:10px; text-align:center;'><b>CO</b><br>Karbon Monoksida</div>
            <div style='padding:15px; background:#f1f5f9; border-radius:10px; text-align:center;'><b>O3</b><br>Ozon Permukaan</div>
            <div style='padding:15px; background:#f1f5f9; border-radius:10px; text-align:center;'><b>NO2</b><br>Nitrogen Dioksida</div>
            <div style='padding:15px; background:#f1f5f9; border-radius:10px; text-align:center;'><b>ISPU</b><br>Standar Indeks</div>
        </div>

        <h3>🧠 Mengapa Menggunakan Gaussian Naive Bayes?</h3>
        <p style='line-height: 1.6; color: #475569;'>
            Algoritma ini mengasumsikan bahwa data kontinu pada masing-masing variabel mengikuti <b>Distribusi Normal (Gaussian)</b>. 
            Metode ini dipilih karena sangat efisien dalam memproses data dengan skala komputasi cepat, membutuhkan sedikit data latihan, dan menghasilkan akurasi klasifikasi probabilitas yang optimal untuk dataset sensor lingkungan.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    <b>AQ-CARE Dashboard</b> • Smart Environment Framework © 2026<br>
    Built with modern Minimalist Engineering Design.
</div>
""", unsafe_allow_html=True)
