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
    page_title="AQ-CARE",
    page_icon="🌍",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #0ea5e9, #2563eb, #1e3a8a);
    color: white;
}

/* CARD */
.card {
    background: rgba(255,255,255,0.12);
    padding: 30px;
    border-radius: 25px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
    margin-bottom: 20px;
}

/* TITLE */
h1, h2, h3, h4 {
    color: white !important;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg,#06b6d4,#3b82f6);
    color: white;
    border-radius: 15px;
    border: none;
    font-size: 18px;
    font-weight: bold;
    height: 50px;
    width: 100%;
}

/* INPUT */
.stTextInput>div>div>input,
.stNumberInput input {
    border-radius: 12px;
}

/* METRIC */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 15px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.15);
}

/* FOOTER */
.footer {
    text-align:center;
    color:white;
    padding:20px;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================
data = pd.read_csv(
    "Filedata Data Indeks Standar Pencemar Udara ISPU di Provinsi DKI Jakarta 2023.csv"
)

# =====================================================
# DATA CLEANING
# =====================================================
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

data = data[data["kategori"] != "TIDAK ADA DATA"]

data["kategori"] = data["kategori"].replace({
    "SANGAT TIDAK SEHAT": "TIDAK SEHAT"
})

# =====================================================
# FITUR & TARGET
# =====================================================
X = data[kolom_numerik]
y = data["kategori"]

# =====================================================
# SPLIT DATA
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=123,
    stratify=y
)

# =====================================================
# MODEL
# =====================================================
model = GaussianNB()
model.fit(X_train, y_train)

prediksi = model.predict(X_test)
akurasi = accuracy_score(y_test, prediksi)

# =====================================================
# SIDEBAR MENU
# =====================================================
st.sidebar.title("🌍 AQ-CARE")

menu = st.sidebar.radio(
    "Navigasi",
    ["🏠 Home", "📊 Dashboard", "🔍 Prediksi", "ℹ️ About"]
)

# =====================================================
# HOME
# =====================================================
if menu == "🏠 Home":

    st.markdown("""
    <div class="card">
        <h1>🌍 AQ-CARE</h1>
        <h3>Smart Air Quality Prediction System</h3>

        <br>

        <p>
        AQ-CARE merupakan website berbasis Machine Learning
        yang digunakan untuk memprediksi kualitas udara menggunakan
        metode Gaussian Naive Bayes.
        </p>

        <p>
        Sistem ini membantu masyarakat mengetahui kondisi udara
        berdasarkan konsentrasi polutan secara cepat dan interaktif.
        </p>

    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("📊 Total Data", len(data))
    col2.metric("🎯 Akurasi Model", f"{akurasi*100:.2f}%")
    col3.metric("🧠 Metode", "Gaussian NB")

    st.write("")

    st.image(
        "https://images.unsplash.com/photo-1519608487953-e999c86e7455",
        use_container_width=True
    )

# =====================================================
# DASHBOARD
# =====================================================
elif menu == "📊 Dashboard":

    st.markdown("""
    <div class="card">
        <h2>📊 Dashboard Kualitas Udara</h2>

        <p>
        Dashboard ini menampilkan visualisasi data polutan udara
        berdasarkan dataset ISPU DKI Jakarta 2023.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Distribusi kategori
    kategori_count = data["kategori"].value_counts().reset_index()
    kategori_count.columns = ["Kategori", "Jumlah"]

    fig1 = px.pie(
        kategori_count,
        names="Kategori",
        values="Jumlah",
        title="Distribusi Kategori Udara"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Grafik PM10
    fig2 = px.histogram(
        data,
        x="pm_sepuluh",
        nbins=30,
        title="Distribusi PM10"
    )

    st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# PREDIKSI
# =====================================================
elif menu == "🔍 Prediksi":

    st.markdown("""
    <div class="card">
        <h2>🌫️ Prediksi Kualitas Udara</h2>

        <p>
        Masukkan nilai polutan udara untuk mengetahui kategori kualitas udara.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        lokasi = st.text_input(
            "📍 Lokasi",
            "Palu, Sulawesi Tengah"
        )

        pm10 = st.number_input("PM10", 0.0, 500.0, 50.0)
        pm25 = st.number_input("PM2.5", 0.0, 500.0, 70.0)
        so2 = st.number_input("SO2", 0.0, 500.0, 30.0)

    with col2:

        co = st.number_input("CO", 0.0, 100.0, 10.0)
        o3 = st.number_input("O3", 0.0, 500.0, 20.0)
        no2 = st.number_input("NO2", 0.0, 500.0, 15.0)

    st.write("")

    prediksi_btn = st.button("🔍 Jalankan Prediksi")

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

        st.write("")

        if hasil == "BAIK":
            st.success(f"🌿 Kategori Udara: {hasil}")

        elif hasil == "SEDANG":
            st.warning(f"⚠️ Kategori Udara: {hasil}")

        else:
            st.error(f"❌ Kategori Udara: {hasil}")

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
            title="📈 Probabilitas Prediksi"
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.1)',
            font_color='white'
        )

        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ABOUT
# =====================================================
elif menu == "ℹ️ About":

    st.markdown("""
    <div class="card">

    <h1>ℹ️ About AQ-CARE</h1>

    <p>
    AQ-CARE adalah sistem prediksi kualitas udara berbasis Machine Learning
    menggunakan metode Gaussian Naive Bayes.
    </p>

    <p>
    Website ini dikembangkan untuk membantu masyarakat dalam
    memahami kondisi kualitas udara secara interaktif dan modern.
    </p>

    <br>

    <h3>📌 Variabel</h3>

    <ul>
        <li>PM10</li>
        <li>PM2.5</li>
        <li>SO2</li>
        <li>CO</li>
        <li>O3</li>
        <li>NO2</li>
    </ul>

    <br>

    <h3>🧠 Metode Gaussian Naive Bayes</h3>

    <p>
    Gaussian Naive Bayes merupakan algoritma klasifikasi
    berbasis probabilitas yang menggunakan Teorema Bayes
    dengan asumsi distribusi normal pada setiap fitur numerik.
    </p>

    <p>
    Model ini cocok digunakan untuk klasifikasi data kualitas udara
    karena memiliki proses komputasi yang cepat dan akurat.
    </p>

    </div>
    """, unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
AQ-CARE © 2026 <br>
Smart Air Quality Prediction System
</div>
""", unsafe_allow_html=True)
