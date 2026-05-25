import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="AQ-CARE",
    page_icon="🌍",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
}

[data-testid="stSidebar"] * {
    color: white;
}

/* Card */
.card {
    background-color: rgba(255,255,255,0.15);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}

/* Judul */
h1 {
    color: white !important;
    font-size: 48px !important;
    font-weight: 700 !important;
}

h2, h3 {
    color: white !important;
}

/* Metric */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.2);
    border-radius: 18px;
    padding: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.15);
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #ff9966, #ff5e62);
    color: white;
    border: none;
    border-radius: 15px;
    padding: 14px;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #ff5e62, #ff9966);
}

/* Input */
input {
    border-radius: 10px !important;
}

/* Footer */
.footer {
    text-align:center;
    color:white;
    font-size:15px;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD DATA
# ======================================================
data = pd.read_csv(
    "Filedata Data Indeks Standar Pencemar Udara ISPU di Provinsi DKI Jakarta 2023.csv"
)

# ======================================================
# DATA CLEANING
# ======================================================
data.replace("–", np.nan, inplace=True)
data.dropna(inplace=True)

data.columns = data.columns.str.lower()
data.columns = data.columns.str.strip()

kolom_numerik = [
    "pm_sepuluh",
    "pm_duakomalima",
    "sulfur_dioksida",
    "karbon_monoksida",
    "ozon",
    "nitrogen_dioksida"
]

for col in kolom_numerik:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data.dropna(inplace=True)

# Hapus kategori tidak valid
data = data[data["kategori"] != "TIDAK ADA DATA"]

# Merge kelas
data["kategori"] = data["kategori"].replace({
    "SANGAT TIDAK SEHAT": "TIDAK SEHAT"
})

# ======================================================
# FITUR & TARGET
# ======================================================
X = data[kolom_numerik]
y = data["kategori"]

# ======================================================
# SPLIT DATA
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=123,
    stratify=y
)

# ======================================================
# MODEL GNB
# ======================================================
model = GaussianNB()
model.fit(X_train, y_train)

# ======================================================
# AKURASI
# ======================================================
prediksi = model.predict(X_test)
akurasi = accuracy_score(y_test, prediksi)

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🌍 AQ-CARE")

st.sidebar.markdown("""
### Smart Air Quality Prediction
Gaussian Naive Bayes Based System
""")

lokasi = st.sidebar.text_input(
    "📍 Lokasi",
    "Palu, Sulawesi Tengah"
)

st.sidebar.markdown("## 📊 Input Polutan")

pm10 = st.sidebar.number_input("PM10", 0.0, 500.0, 50.0)
pm25 = st.sidebar.number_input("PM2.5", 0.0, 500.0, 70.0)
so2 = st.sidebar.number_input("SO2", 0.0, 500.0, 30.0)
co = st.sidebar.number_input("CO", 0.0, 100.0, 10.0)
o3 = st.sidebar.number_input("O3", 0.0, 500.0, 20.0)
no2 = st.sidebar.number_input("NO2", 0.0, 500.0, 15.0)

prediksi_btn = st.sidebar.button("🔍 Prediksi Sekarang")

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<div class="card">
<h1>🌫️ AQ-CARE</h1>
<h3>Air Quality Prediction System</h3>
<p style='color:white;font-size:18px;'>
Prediksi kualitas udara berbasis Machine Learning menggunakan metode Gaussian Naive Bayes.
</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ======================================================
# METRIC
# ======================================================
col1, col2, col3 = st.columns(3)

col1.metric("📍 Lokasi", lokasi)
col2.metric("📊 Total Data", len(data))
col3.metric("🎯 Akurasi Model", f"{akurasi*100:.2f}%")

st.write("")

# ======================================================
# VISUALISASI DATA
# ======================================================
st.markdown("## 📈 Distribusi Kategori Udara")

fig_kategori = px.pie(
    data,
    names="kategori",
    title="Distribusi Kategori Kualitas Udara",
    hole=0.5
)

fig_kategori.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)

st.plotly_chart(fig_kategori, use_container_width=True)

# ======================================================
# PREDIKSI
# ======================================================
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

    st.markdown("## 🔍 Hasil Prediksi")

    if hasil == "BAIK":
        st.success(f"🌿 Kualitas Udara: {hasil}")

    elif hasil == "SEDANG":
        st.warning(f"⚠️ Kualitas Udara: {hasil}")

    else:
        st.error(f"❌ Kualitas Udara: {hasil}")

    # ==================================================
    # GRAFIK PROBABILITAS
    # ==================================================
    kelas = model.classes_

    prob_df = pd.DataFrame({
        "Kategori": kelas,
        "Probabilitas": probabilitas
    })

    fig = px.bar(
        prob_df,
        x="Kategori",
        y="Probabilitas",
        color="Kategori",
        text_auto='.2f',
        title="📊 Probabilitas Prediksi"
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.1)',
        font_color='white'
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==================================================
    # SARAN
    # ==================================================
    st.markdown("## 💡 Rekomendasi")

    if hasil == "BAIK":
        st.info("""
        ✅ Udara aman untuk aktivitas luar ruangan  
        ✅ Cocok untuk olahraga  
        ✅ Tidak perlu masker
        """)

    elif hasil == "SEDANG":
        st.warning("""
        ⚠️ Kelompok sensitif sebaiknya mengurangi aktivitas luar ruangan  
        ⚠️ Gunakan masker bila diperlukan
        """)

    else:
        st.error("""
        ❌ Hindari aktivitas luar ruangan  
        ❌ Gunakan masker  
        ❌ Tutup ventilasi rumah
        """)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<div class="footer">
AQ-CARE © 2026 <br>
Gaussian Naive Bayes Air Quality Prediction System
</div>
""", unsafe_allow_html=True)
