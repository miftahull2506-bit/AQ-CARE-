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
    page_title="AQ-CARE | Smart Air Quality Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for Landing Screen Navigation
if "start_prediction" not in st.session_state:
    st.session_state.start_prediction = False

# =====================================================
# MULTI-LANGUAGE DICTIONARY (ID & EN)
# =====================================================
LANG = {
    "ID": {
        "nav_home": "🏠 Beranda",
        "nav_dash": "📊 Dashboard Analitik",
        "nav_pred": "🔍 Prediksi Kualitas",
        "nav_info": "ℹ️ Panduan & Informasi",
        "hero_title": "🌍 AQ-CARE",
        "hero_subtitle": "Sistem Deteksi Kualitas Udara Pintar",
        "hero_desc": "AQ-CARE adalah aplikasi berbasis kecerdasan buatan (AI) yang dirancang khusus untuk memantau, menganalisis, dan memprediksi apakah udara di sekitar kita sehat atau tidak secara instan dan mudah dipahami.",
        "metric_data": "📊 Total Data Diperiksa",
        "metric_acc": "🎯 Akurasi Deteksi AI",
        "metric_method": "🧠 Teknologi Sistem",
        "dash_title": "📊 Grafik Pantauan Udara",
        "dash_subtitle": "Melihat rangkuman kondisi kebersihan udara berdasarkan data historis secara visual.",
        "chart_pie": "<b>Proporsi Kategori Udara</b>",
        "chart_hist": "<b>Tingkat Kepadatan Debu Udara (PM10)</b>",
        "pred_title": "🔍 Kalkulator Prediksi Udara",
        "pred_subtitle": "Cek kelayakan udara di lokasi Anda menggunakan kecerdasan buatan.",
        "input_loc": "📍 Tulis Nama Lokasi Anda",
        "btn_predict": "🚀 Mulai Hitung Kualitas Udara",
        "res_title": "### 📍 Hasil Analisis untuk",
        "res_baik": "🌱 **KUALITAS UDARA AMAT BAIK** — Udara sangat segar dan bersih! Sangat aman untuk jalan-jalan atau olahraga di luar rumah.",
        "res_sedang": "⚠️ **KUALITAS UDARA SEDANG** — Udara cukup aman, tetapi bagi yang sensitif (seperti penderita asma/lansia) sebaiknya mulai berhati-hati.",
        "res_buruk": "🚨 **KUALITAS UDARA TIDAK SEHAT** — Udara kotor! Sangat disarankan memakai masker dan kurangi aktivitas di luar ruangan.",
        "chart_conf": "<b>Persentase Keyakinan Kecerdasan Buatan (AI)</b>",
        "info_title": "ℹ️ Panduan Mudah AQ-CARE",
        "info_desc": "Aplikasi ini diciptakan untuk membantu masyarakat awam mengenali kondisi udara di lingkungan sekitar demi menjaga kesehatan keluarga.",
        "var_title": "📌 Mengenal 6 Musuh Utama di Udara",
        "var_subtitle": "Sistem kami mendeteksi 6 jenis kotoran dan gas berbahaya yang sering melayang di sekitar kita:",
        "why_title": "🧠 Bagaimana Cara Kerja AI di Aplikasi Ini?",
        "why_desc": "Aplikasi ini menggunakan metode statistik pintar yang mempelajari ribuan data kualitas udara di masa lalu. AI akan melihat pola dari angka-angka polutan yang Anda masukkan, membandingkannya dengan pola data yang sudah dipelajari, lalu memberikan kesimpulan instan apakah udara tersebut masuk kategori Baik, Sedang, atau Tidak Sehat.",
        "edu_title": "💡 Tips & Panduan Kesehatan Udara (Standar WHO)",
        "edu_desc": "Klik di sini untuk melihat info batas aman dan dampak kesehatan bagi tubuh kita.",
        "footer_text": "Didesain dengan kemudahan akses untuk semua orang."
    },
    "EN": {
        "nav_home": "🏠 Home",
        "nav_dash": "📊 Analytical Dashboard",
        "nav_pred": "🔍 Quality Prediction",
        "nav_info": "ℹ️ Guides & Information",
        "hero_title": "🌍 AQ-CARE",
        "hero_subtitle": "Smart Air Quality Detection System",
        "hero_desc": "AQ-CARE is an artificial intelligence (AI) application specifically designed to monitor, analyze, and predict whether the air around us is healthy or not instantly and easily.",
        "metric_data": "📊 Total Data Checked",
        "metric_acc": "🎯 AI Detection Accuracy",
        "metric_method": "🧠 System Technology",
        "dash_title": "📊 Air Monitoring Charts",
        "dash_subtitle": "Visually view the summary of air cleanliness conditions based on historical data.",
        "chart_pie": "<b>Air Category Proportion</b>",
        "chart_hist": "<b>Air Dust Density Level (PM10)</b>",
        "pred_title": "🔍 Air Quality Calculator",
        "pred_subtitle": "Check the air quality at your location using artificial intelligence.",
        "input_loc": "📍 Enter Your Location Name",
        "btn_predict": "🚀 Start Calculating Air Quality",
        "res_title": "### 📍 Analysis Results for",
        "res_baik": "🌱 **EXCELLENT AIR QUALITY** — The air is very fresh and clean! Highly safe for walking or outdoor sports.",
        "res_sedang": "⚠️ **MODERATE AIR QUALITY** — The air is fair, but sensitive groups (like asthma patients/elderly) should start being cautious.",
        "res_buruk": "🚨 **UNHEALTHY AIR QUALITY** — Dirty air! It is highly recommended to wear a mask and reduce outdoor activities.",
        "chart_conf": "<b>Artificial Intelligence (AI) Confidence Percentage</b>",
        "info_title": "ℹ️ AQ-CARE Easy Guide",
        "info_desc": "This application was created to help everyone recognize air conditions in the surrounding environment to protect family health.",
        "var_title": "📌 Meet the 6 Main Enemies in the Air",
        "var_subtitle": "Our system detects 6 types of hazardous dirt and gases that often float around us:",
        "why_title": "🧠 How Does the AI Work in This App?",
        "why_desc": "This app uses a smart statistical method that learns from thousands of historical air quality data points. The AI looks at patterns from the pollutant numbers you enter, compares them with past patterns, and gives an instant conclusion whether the air is Good, Moderate, or Unhealthy.",
        "edu_title": "💡 Air Health Guidelines & Tips (WHO Standard)",
        "edu_desc": "Click here to see safe limit info and health impacts on our bodies.",
        "footer_text": "Designed with ease of access for everyone."
    }
}

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

section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0;
    box-shadow: 4px 0 10px rgba(0,0,0,0.02);
}

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

.content-card {
    background: #ffffff;
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    margin-bottom: 25px;
}
.content-card h1, .content-card h2, .content-card h3 {
    color: #0f172a !important;
    font-weight: 600;
}

[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
}

.stButton>button {
    background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    height: 52px !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
    transition: all 0.3s ease !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.3) !important;
}

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
# DATA PIPELINE
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
    st.error(f"Gagal memuat data / Data failed to load. Error: {e}")
    st.stop()

# =====================================================
# SIDEBAR MULTI-LANGUAGE NAVIGATION
# =====================================================
st.sidebar.markdown("<h1 style='font-size: 26px; margin-bottom: 5px;'>🌍 AQ-CARE</h1>", unsafe_allow_html=True)

lang_choice = st.sidebar.segmented_control(
    "Language / Bahasa",
    options=["ID", "EN"],
    default="ID",
    label_visibility="collapsed"
)

txt = LANG[lang_choice]
st.sidebar.markdown("<hr style='margin: 15px 0; border: 0.5px solid #e2e8f0;'>", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Menu Navigation",
    [txt["nav_home"], txt["nav_dash"], txt["nav_pred"], txt["nav_info"]],
    label_visibility="collapsed"
)

# Reset screen prediction if switching menu
if menu != txt["nav_pred"]:
    st.session_state.start_prediction = False

# =====================================================
# HOME PAGE
# =====================================================
if menu == txt["nav_home"]:
    st.markdown(f"""
    <div class="hero-card">
        <h1 style='font-size: 38px; margin-bottom: 10px; font-weight: 700;'>{txt["hero_title"]}</h1>
        <h3 style='font-size: 20px; font-weight: 400; opacity: 0.9; margin-bottom: 20px;'>{txt["hero_subtitle"]}</h3>
        <p style='font-size: 16px; max-width: 800px; line-height: 1.6; opacity: 0.85;'>{txt["hero_desc"]}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric(txt["metric_data"], f"{len(data):,}")
    col2.metric(txt["metric_acc"], f"{akurasi*100:.2f}%")
    col3.metric(txt["metric_method"], "Smart AI System")

    st.markdown("<br>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1519608487953-e999c86e7455", use_container_width=True)

# =====================================================
# DASHBOARD PAGE
# =====================================================
elif menu == txt["nav_dash"]:
    st.markdown(f"""
    <div class="content-card">
        <h2>{txt["dash_title"]}</h2>
        <p style='color: #64748b;'>{txt["dash_subtitle"]}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        kategori_count = data["kategori"].value_counts().reset_index()
        kategori_count.columns = ["Kategori", "Jumlah"]
        
        fig1 = px.pie(
            kategori_count, names="Kategori", values="Jumlah",
            title=txt["chart_pie"],
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#0f172a')
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        fig2 = px.histogram(
            data, x="pm_sepuluh", nbins=25,
            title=txt["chart_hist"],
            color_discrete_sequence=['#0ea5e9']
        )
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#0f172a')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PREDICTION PAGE (WITH LANDING INITIAL VIEW)
# =====================================================
elif menu == txt["nav_pred"]:
    
    # 1. TAMPILAN AWAL (WELCOME SCREEN)
    if not st.session_state.start_prediction:
        st.markdown(f"""
        <div class="hero-card" style="background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%); text-align: center;">
            <h1 style='font-size: 36px; font-weight: 700; margin-bottom: 10px;'>🔍 {txt["pred_title"]}</h1>
            <p style='font-size: 18px; opacity: 0.9; max-width: 700px; margin: 0 auto 30px auto;'>
                {txt["pred_subtitle"]} Aplikasi kami siap mengalkulasi kualitas polutan secara cepat dengan kecerdasan buatan.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Penjelasan Alur Singkat Orang Awam
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='content-card' style='text-align:center; height:180px;'><h4>📍 Langkah 1</h4><p style='color:#64748b;'>Masukkan nama kota atau lokasi tempat tinggal Anda saat ini.</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='content-card' style='text-align:center; height:180px;'><h4>🌫️ Langkah 2</h4><p style='color:#64748b;'>Isi nilai indikator polutan udara (bisa didapatkan dari data sensor lokal atau perkiraan).</p></div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='content-card' style='text-align:center; height:180px;'><h4>✨ Langkah 3</h4><p style='color:#64748b;'>Klik tombol proses dan kecerdasan buatan akan menilai kelayakannya secara instan!</p></div>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tombol Gerbang Masuk Ke Kalkulator Input
        if st.button("🚀 " + ("Mulai Analisis Sekarang" if lang_choice == "ID" else "Start Analysis Now")):
            st.session_state.start_prediction = True
            st.rerun()

    # 2. TAMPILAN HALAMAN INPUT (SETELAH KLIK TOMBOL MULAI)
    else:
        st.markdown(f"""
        <div class="content-card">
            <h2>{txt["pred_title"]}</h2>
            <p style='color: #64748b;'>{txt["pred_subtitle"]}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        lokasi = st.text_input(txt["input_loc"], "Palu, Sulawesi Tengah")
        st.markdown("<hr style='border: 0.5px solid #e2e8f0; margin: 20px 0;'>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            pm10 = st.number_input("🌫️ Kepadatan Debu / PM10 (µg/m³)", 0.0, 500.0, 50.0, step=1.0)
            co = st.number_input("🚗 Gas Kendaraan / CO (µg/m³)", 0.0, 100.0, 10.0, step=0.5)
        with col2:
            pm25 = st.number_input("💨 Partikel Halus / PM2.5 (µg/m³)", 0.0, 500.0, 70.0, step=1.0)
            o3 = st.number_input("☀️ Gas Lapisan Ozon / O3 (µg/m³)", 0.0, 500.0, 20.0, step=1.0)
        with col3:
            so2 = st.number_input("🏭 Asap Pabrik / SO2 (µg/m³)", 0.0, 500.0, 30.0, step=1.0)
            no2 = st.number_input("🔥 Gas Pembakaran / NO2 (µg/m³)", 0.0, 500.0, 15.0, step=1.0)

        st.markdown("<br>", unsafe_allow_html=True)
        prediksi_btn = st.button(txt["btn_predict"])
        
        # Tombol Kembali ke Welcome Screen
        if st.button("⬅️ " + ("Kembali ke Awal" if lang_choice == "ID" else "Back to Welcome Screen")):
            st.session_state.start_prediction = False
            st.rerun()
            
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

            st.markdown(f"{txt['res_title']} **{lokasi}**:")
            if hasil == "BAIK":
                st.success(txt["res_baik"])
            elif hasil == "SEDANG":
                st.warning(txt["res_sedang"])
            else:
                st.error(txt["res_buruk"])

            # Chart persentase keyakinan model
            st.markdown("<div class='content-card'>", unsafe_allow_html=True)
            prob_df = pd.DataFrame({
                "Kategori": model.classes_,
                "Persentase Keyakinan System (%)": probabilitas * 100
            })

            fig = px.bar(
                prob_df, x="Kategori", y="Persentase Keyakinan System (%)",
                color="Kategori", text_auto='.1f',
                title=txt["chart_conf"],
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#0f172a')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# ABOUT & EDUCATION PAGE (ANTI-RIBET RUMUS)
# =====================================================
elif menu == txt["nav_info"]:
    st.markdown(f"""
    <div class="content-card">
        <h2>{txt["info_title"]}</h2>
        <p style='line-height: 1.6; color: #475569;'>{txt["info_desc"]}</p>
        
        <h3 style='margin-top: 30px;'>{txt["var_title"]}</h3>
        <p style='color: #475569;'>{txt["var_subtitle"]}</p>
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 20px;'>
            <div style='padding:15px; background:#f1f5f9; border-radius:10px; text-align:center;'><b>PM10 & PM2.5</b><br>Debu & Partikel Kecil</div>
            <div style='padding:15px; background:#f1f5f9; border-radius:10px; text-align:center;'><b>SO2</b><br>Gas Asap Pabrik Industri</div>
            <div style='padding:15px; background:#f1f5f9; border-radius:10px; text-align:center;'><b>CO</b><br>Gas Racun Asap Kendaraan</div>
            <div style='padding:15px; background:#f1f5f9; border-radius:10px; text-align:center;'><b>O3</b><br>Gas Lapisan Ozon Permukaan</div>
            <div style='padding:15px; background:#f1f5f9; border-radius:10px; text-align:center;'><b>NO2</b><br>Gas Hasil Pembakaran Tinggi</div>
            <div style='padding:15px; background:#f1f5f9; border-radius:10px; text-align:center;'><b>ISPU</b><br>Indeks Standar Kualitas Udara</div>
        </div>

        <h3>{txt["why_title"]}</h3>
        <p style='line-height: 1.6; color: #475569;'>{txt["why_desc"]}</p>
    </div>
    """, unsafe_allow_html=True)

    # Widget Edukasi Informasi Kesehatan Tambahan (Biar Lebih Ramai)
    with st.expander(txt["edu_title"]):
        st.markdown(f"<p style='color:#64748b;'>{txt['edu_desc']}</p>", unsafe_allow_html=True)
        edu_data = {
            "Indikator / Indicator": ["PM2.5", "PM10", "O3 (Ozon)", "CO", "SO2", "NO2"],
            "Ambang Batas Aman / Safe Limit (WHO)": ["15 µg/m³ (24h)", "45 µg/m³ (24h)", "100 µg/m³ (8h)", "4 mg/m³ (24h)", "40 µg/m³ (24h)", "25 µg/m³ (24h)"],
            "Dampak Bagi Kesehatan / Health Impact": [
                "Bisa masuk paru-paru dalam, batuk, sesak / Severe respiratory risks",
                "Iritasi tenggorokan & saluran pernapasan / Lung irritation",
                "Memicu kambuhnya penyakit asma / Triggers asthma",
                "Membuat pusing, mual, lemas / Headache, nausea",
                "Dapat memicu hujan asam dan batuk parah / Irritation",
                "Risiko infeksi paru-paru pada anak-anak / Infection risks"
            ]
        }
        st.table(pd.DataFrame(edu_data))

# =====================================================
# FOOTER
# =====================================================
st.markdown(f"""
<div class="footer">
    <b>AQ-CARE AI Dashboard</b> • Smart Environment Framework © 2026<br>
    {txt["footer_text"]}
</div>
""", unsafe_allow_html=True)
