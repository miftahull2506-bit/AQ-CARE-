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
    page_title="AQ-CARE | Multi-Language AI Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# MULTI-LANGUAGE DICTIONARY (ID & EN)
# =====================================================
LANG = {
    "ID": {
        "nav_home": "🏠 Beranda",
        "nav_dash": "📊 Dashboard Analitik",
        "nav_pred": "🔍 Prediksi Kualitas",
        "nav_info": "ℹ️ Informasi Sistem",
        "hero_title": "🌍 AQ-CARE",
        "hero_subtitle": "Sistem Prediksi Kualitas Udara Pintar",
        "hero_desc": "AQ-CARE memanfaatkan kecerdasan buatan (Machine Learning) dengan algoritma Gaussian Naive Bayes untuk mengklasifikasikan dan memprediksi tingkat kelayakan udara secara presisi, cerdas, dan real-time berdasarkan parameter sensor lingkungan.",
        "metric_data": "📊 Total Sampel Data",
        "metric_acc": "🎯 Akurasi Pengujian",
        "metric_method": "🧠 Intelijen Model",
        "dash_title": "📊 Dashboard Analitik Kualitas Udara",
        "dash_subtitle": "Visualisasi ringkas dari distribusi data ISPU untuk memantau tren pencemaran lingkungan.",
        "chart_pie": "<b>Proporsi Kategori Udara</b>",
        "chart_hist": "<b>Distribusi Konsentrasi PM10</b>",
        "pred_title": "🔍 Kalkulator Prediksi Kualitas Udara",
        "pred_subtitle": "Masukkan parameter konsentrasi polutan di bawah ini untuk mendapatkan hasil analisis instan dari kecerdasan buatan.",
        "input_loc": "📍 Nama Lokasi Pengujian",
        "btn_predict": "🚀 Jalankan Prediksi Kecerdasan Buatan",
        "res_title": "### 📍 Hasil Analisis untuk",
        "res_baik": "🌱 **KUALITAS UDARA AMAT BAIK** — Udara sangat segar dan aman untuk beraktivitas di luar ruangan.",
        "res_sedang": "⚠️ **KUALITAS UDARA SEDANG** — Kualitas udara berstatus wajar namun sensitif bagi sebagian kelompok rentan.",
        "res_buruk": "🚨 **KUALITAS UDARA TIDAK SEHAT** — Sangat disarankan menggunakan masker dan mengurangi mobilitas luar.",
        "chart_conf": "<b>Tingkat Keyakinan Prediksi (Confidence Score)</b>",
        "info_title": "ℹ️ Tentang AQ-CARE",
        "info_desc": "Sistem ini dibangun dengan dedikasi tinggi untuk memberikan informasi prediktif yang akurat mengenai indeks standar pencemar udara (ISPU). Menggunakan riset komparatif berbasis data historis DKI Jakarta.",
        "var_title": "📌 Variabel Polutan Utama",
        "var_subtitle": "Sistem mendeteksi 6 unsur senyawa kritikal berbahaya:",
        "why_title": "🧠 Mengapa Menggunakan Gaussian Naive Bayes?",
        "why_desc": "Algoritma ini mengasumsikan bahwa data kontinu pada masing-masing variabel mengikuti Distribusi Normal (Gaussian). Metode ini dipilih karena sangat efisien dalam memproses data dengan skala komputasi cepat, membutuhkan sedikit data latihan, dan menghasilkan akurasi klasifikasi probabilitas yang optimal untuk dataset sensor lingkungan.",
        "edu_title": "💡 Panduan Ambang Batas Aman Polutan (WHO Standard)",
        "edu_desc": "Klik untuk melihat tabel ambang batas aman indikator polutan udara.",
        "footer_text": "Built with modern Minimalist Engineering Design."
    },
    "EN": {
        "nav_home": "🏠 Home",
        "nav_dash": "📊 Analytical Dashboard",
        "nav_pred": "🔍 Quality Prediction",
        "nav_info": "ℹ️ System Information",
        "hero_title": "🌍 AQ-CARE",
        "hero_subtitle": "Smart Air Quality Prediction System",
        "hero_desc": "AQ-CARE utilizes Artificial Intelligence (Machine Learning) with the Gaussian Naive Bayes algorithm to classify and predict air quality levels precisely, intelligently, and in real-time based on environmental sensor parameters.",
        "metric_data": "📊 Total Data Samples",
        "metric_acc": "🎯 Testing Accuracy",
        "metric_method": "🧠 Model Intelligence",
        "dash_title": "📊 Air Quality Analytical Dashboard",
        "dash_subtitle": "Concise visualization of ISPU data distribution to monitor environmental pollution trends.",
        "chart_pie": "<b>Air Category Proportion</b>",
        "chart_hist": "<b>PM10 Concentration Distribution</b>",
        "pred_title": "🔍 Air Quality Prediction Calculator",
        "pred_subtitle": "Enter the pollutant concentration parameters below to get instant analysis results from our AI model.",
        "input_loc": "📍 Test Location Name",
        "btn_predict": "🚀 Run Artificial Intelligence Prediction",
        "res_title": "### 📍 Analysis Results for",
        "res_baik": "🌱 **EXCELLENT AIR QUALITY** — The air is very fresh and safe for outdoor activities.",
        "res_sedang": "⚠️ **MODERATE AIR QUALITY** — Air quality is fair but may be sensitive to some vulnerable groups.",
        "res_buruk": "🚨 **UNHEALTHY AIR QUALITY** — It is highly recommended to wear masks and reduce outdoor mobility.",
        "chart_conf": "<b>Prediction Confidence Score</b>",
        "info_title": "ℹ️ About AQ-CARE",
        "info_desc": "This system was built with high dedication to provide accurate predictive information regarding the air pollutant standard index (ISPU). Utilizing comparative research based on historical data from DKI Jakarta.",
        "var_title": "📌 Key Pollutant Variables",
        "var_subtitle": "The system detects 6 critical hazardous compound elements:",
        "why_title": "🧠 Why Use Gaussian Naive Bayes?",
        "why_desc": "This algorithm assumes that the continuous data for each variable follows a Normal (Gaussian) Distribution. This method was chosen because it is highly efficient in processing data with fast computational scales, requires minimal training data, and yields optimal probability classification accuracy for environmental sensor datasets.",
        "edu_title": "💡 Pollutant Safety Threshold Guide (WHO Standard)",
        "edu_desc": "Click to view the safety threshold table for air pollutant indicators.",
        "footer_text": "Built with modern Minimalist Engineering Design."
    }
}

# =====================================================
# CUSTOM MODERN CSS (Google & Premium Tech Vibe)
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

.stTextInput input, .stNumberInput input {
    border-radius: 10px !important;
    border: 1px solid #cbd5e1 !important;
    background-color: #ffffff !important;
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

# Language Selector component
lang_choice = st.sidebar.segmented_control(
    "Language / Bahasa",
    options=["ID", "EN"],
    default="ID",
    label_visibility="collapsed"
)

# Active language configuration shortcut
txt = LANG[lang_choice]

st.sidebar.markdown("<hr style='margin: 15px 0; border: 0.5px solid #e2e8f0;'>", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Menu Navigation",
    [txt["nav_home"], txt["nav_dash"], txt["nav_pred"], txt["nav_info"]],
    label_visibility="collapsed"
)

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
    col3.metric(txt["metric_method"], "Gaussian NB")

    st.markdown("<br>", unsafe_allow_html=True)
    st.image(
        "https://images.unsplash.com/photo-1519608487953-e999c86e7455", 
        use_container_width=True
    )

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
# PREDICTION PAGE
# =====================================================
elif menu == txt["nav_pred"]:
    st.markdown(f"""
    <div class="content-card">
        <h2>{txt["pred_title"]}</h2>
        <p style='color: #64748b;'>{txt["pred_subtitle"]}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    lokasi = st.text_input(txt["input_loc"], "Jakarta Pusat, ID")
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
    prediksi_btn = st.button(txt["btn_predict"])
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

        # Chart Confidence Score
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            "Kategori": model.classes_,
            "Probabilitas Keyakinan": probabilitas
        })

        fig = px.bar(
            prob_df, x="Kategori", y="Probabilitas Keyakinan",
            color="Kategori", text_auto='.2f',
            title=txt["chart_conf"],
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#0f172a')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# ABOUT & EDUCATION PAGE
# =====================================================
elif menu == txt["nav_info"]:
    st.markdown(f"""
    <div class="content-card">
        <h2>{txt["info_title"]}</h2>
        <p style='line-height: 1.6; color: #475569;'>{txt["info_desc"]}</p>
        
        <h3 style='margin-top: 30px;'>{txt["var_title"]}</h3>
        <p style='color: #475569;'>{txt["var_subtitle"]}</p>
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 20px;'>
            <div style='padding:15px; background:#f1f5f9; border-radius:10px; text-align:center;'><b>PM10 & PM2.5</b><br>Particulate Matter</div>
            <div style='padding:15px; background:#f1f5f9; border-radius:10px; text-align:center;'><b>SO2</b><br>Sulfur Dioxide</div>
            <div style='padding:15px; background:#f1f5f9; border-radius:10px; text-align:center;'><b>CO</b><br>Carbon Monoxide</div>
            <div style='padding:15px; background:#f1f5f9; border-radius:10px; text-align:center;'><b>O3</b><br>Ozone</div>
            <div style='padding:15px; background:#f1f5f9; border-radius:10px; text-align:center;'><b>NO2</b><br>Nitrogen Dioxide</div>
            <div style='padding:15px; background:#f1f5f9; border-radius:10px; text-align:center;'><b>ISPU</b><br>Air Index Standard</div>
        </div>

        <h3>{txt["why_title"]}</h3>
        <p style='line-height: 1.6; color: #475569;'>{txt["why_desc"]}</p>
    </div>
    """, unsafe_allow_html=True)

    # Menambahkan Widget Edukasi Tambahan Biar Rame & Informatif
    with st.expander(txt["edu_title"]):
        st.markdown(f"<p style='color:#64748b;'>{txt['edu_desc']}</p>", unsafe_allow_html=True)
        edu_data = {
            "Indikator / Indicator": ["PM2.5", "PM10", "O3 (Ozon)", "CO", "SO2", "NO2"],
            "Ambang Batas Aman / Safe Limit (WHO)": ["15 µg/m³ (24h)", "45 µg/m³ (24h)", "100 µg/m³ (8h)", "4 mg/m³ (24h)", "40 µg/m³ (24h)", "25 µg/m³ (24h)"],
            "Dampak Kesehatan / Health Impact": [
                "Risiko pernapasan tinggi / Severe respiratory risks",
                "Iritasi paru-paru / Lung irritation",
                "Memicu asma / Triggers asthma",
                "Sakit kepala, sesak / Headache, asphyxia",
                "Hujan asam, iritasi / Acid rain, irritation",
                "Infeksi pernapasan akut / Acute respiratory infection"
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
