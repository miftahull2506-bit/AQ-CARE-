import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px

# ==========================================
# CONFIG WEB
# ==========================================
st.set_page_config(
    page_title="AQ-CARE",
    page_icon="🌍",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}

.stApp {
    background: linear-gradient(to right, #0f172a, #1e293b);
}

h1, h2, h3 {
    color: #38bdf8;
}

[data-testid="stMetric"] {
    background-color: #1e293b;
    border-radius: 15px;
    padding: 15px;
}

.stButton>button {
    background-color: #38bdf8;
    color: black;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA
# ==========================================
data = pd.read_csv(
    "Filedata Data Indeks Standar Pencemar Udara ISPU di Provinsi DKI Jakarta 2023.csv"
)

# ==========================================
# DATA CLEANING
# ==========================================
data.replace("–", np.nan, inplace=True)
data.dropna(inplace=True)

# rapikan nama kolom
data.columns = data.columns.str.lower()
data.columns = data.columns.str.strip()

# ubah numerik
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

# hapus kategori tidak valid
data = data[data["kategori"] != "TIDAK ADA DATA"]

# merge kelas
data["kategori"] = data["kategori"].replace({
    "SANGAT TIDAK SEHAT": "TIDAK SEHAT"
})

# ==========================================
# FITUR DAN TARGET
# ==========================================
X = data[kolom_numerik]
y = data["kategori"]

# ==========================================
# SPLIT DATA
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=123,
    stratify=y
)

# ==========================================
# MODEL
# ==========================================
model = GaussianNB()
model.fit(X_train, y_train)

# ==========================================
# AKURASI
# ==========================================
prediksi = model.predict(X_test)
akurasi = accuracy_score(y_test, prediksi)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🌍 AQ-CARE")

st.sidebar.markdown("""
Smart Air Quality Prediction System powered by Gaussian Naive Bayes
""")

lokasi = st.sidebar.text_input(
    "📍 Lokasi",
    "Palu, Sulawesi Tengah"
)

pm10 = st.sidebar.number_input("PM10", 0.0, 500.0, 50.0)
pm25 = st.sidebar.number_input("PM2.5", 0.0, 500.0, 70.0)
so2 = st.sidebar.number_input("SO2", 0.0, 500.0, 30.0)
co = st.sidebar.number_input("CO", 0.0, 100.0, 10.0)
o3 = st.sidebar.number_input("O3", 0.0, 500.0, 20.0)
no2 = st.sidebar.number_input("NO2", 0.0, 500.0, 15.0)

prediksi_btn = st.sidebar.button("🔍 Prediksi")

# ==========================================
# HEADER
# ==========================================
st.title("🌫️ AQ-CARE")
st.subheader("Web-Based Air Quality Prediction System")

# ==========================================
# METRIC
# ==========================================
col1, col2, col3 = st.columns(3)

col1.metric("📍 Lokasi", lokasi)
col2.metric("📊 Jumlah Data", len(data))
col3.metric("🎯 Akurasi Model", f"{akurasi*100:.2f}%")

st.divider()

# ==========================================
# PREDIKSI
# ==========================================
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

    st.success(f"🌍 Kategori Kualitas Udara: {hasil}")

    # ======================================
    # PROBABILITAS
    # ======================================
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
        title="Probabilitas Prediksi"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ======================================
    # SARAN
    # ======================================
    if hasil == "BAIK":
        st.info("✅ Udara aman untuk aktivitas luar ruangan.")

    elif hasil == "SEDANG":
        st.warning("⚠️ Kelompok sensitif disarankan mengurangi aktivitas luar ruangan.")

    else:
        st.error("❌ Udara tidak sehat. Gunakan masker dan batasi aktivitas luar ruangan.")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")

st.markdown("""
<center>
AQ-CARE © 2026 <br>
Gaussian Naive Bayes Air Quality Prediction System
</center>
""", unsafe_allow_html=True)