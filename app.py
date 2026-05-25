import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
from streamlit_option_menu import option_menu

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

/* Background */
.stApp {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

/* Navbar */
.navbar {
    background-color: rgba(255,255,255,0.15);
    padding: 10px;
    border-radius: 15px;
}

/* Card */
.card {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    border-radius: 25px;
    padding: 30px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg,#ff9966,#ff5e62);
    color: white;
    border: none;
    border-radius: 15px;
    padding: 12px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
}

/* Metric */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.2);
    border-radius: 18px;
    padding: 15px;
}

/* Text */
h1, h2, h3, h4 {
    color: white !important;
}

p {
    color: white;
    font-size: 18px;
}

/* Footer */
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

# Hapus kategori tidak valid
data = data[data["kategori"] != "TIDAK ADA DATA"]

# Merge kelas
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
# MODEL GNB
# =====================================================
model = GaussianNB()
model.fit(X_train, y_train)

prediksi = model.predict(X_test)
akurasi = accuracy_score(y_test, prediksi)

# =====================================================
# MENU NAVIGASI
# =====================================================
selected = option_menu(
    menu_title=None,
    options=["Home", "Prediksi", "About"],
    icons=["house", "activity", "info-circle"],
    orientation="horizontal",
)

# =====================================================
# HOME PAGE
# =====================================================
if selected == "Home":

    st.markdown("""
    <div class="card">
        <h1>🌍 AQ-CARE</h1>
        <h3>Smart Air Quality Prediction System</h3>
        <br>
        <p>
        AQ-CARE adalah sistem prediksi kualitas udara berbasis 
        <b>Gaussian Naive Bayes</b> yang digunakan untuk 
        memprediksi kategori kualitas udara berdasarkan data polutan.
        </p>
        <br>
        <p>
        Sistem ini membantu masyarakat mengetahui kondisi udara 
        secara cepat dan memberikan rekomendasi kesehatan.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns(3)

    col1.metric("📊 Total Data", len(data))
    col2.metric("🎯 Akurasi Model", f"{akurasi*100:.2f}%")
    col3.metric("🧠 Metode", "Gaussian NB")

    st.write("")

    st.image(
        "https://images.unsplash.com/photo-1521207418485-99c705420785",
        use_container_width=True
    )

# =====================================================
# HALAMAN PREDIKSI
# =====================================================
elif selected == "Prediksi":

    st.markdown("""
    <div class="card">
        <h2>🌫️ Prediksi Kualitas Udara</h2>
        <p>
        Masukkan nilai polutan udara untuk mengetahui kategori kualitas udara.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        lokasi = st.text_input("📍 Lokasi", "Palu, Sulawesi Tengah")

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

        # HASIL
        if hasil == "BAIK":
            st.success(f"🌿 Kategori Udara: {hasil}")

        elif hasil == "SEDANG":
            st.warning(f"⚠️ Kategori Udara: {hasil}")

        else:
            st.error(f"❌ Kategori Udara: {hasil}")

        # GRAFIK
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

        # SARAN
        st.write("")

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

# =====================================================
# ABOUT PAGE
# =====================================================
elif selected == "About":

    st.markdown("""
    <div class="card">
        <h1>ℹ️ About AQ-CARE</h1>
        <br>
        <p>
        AQ-CARE merupakan website prediksi kualitas udara berbasis 
        Machine Learning menggunakan metode Gaussian Naive Bayes.
        </p>

        <p>
        Website ini dikembangkan untuk membantu masyarakat 
        mengetahui kondisi kualitas udara berdasarkan konsentrasi polutan.
        </p>

        <br>

        <h3>📌 Variabel yang Digunakan</h3>

        <ul style='color:white;font-size:18px;'>
            <li>PM10</li>
            <li>PM2.5</li>
            <li>SO2</li>
            <li>CO</li>
            <li>O3</li>
            <li>NO2</li>
        </ul>

        <br>

        <h3>🧠 Metode</h3>

        <p>
        Gaussian Naive Bayes menggunakan Teorema Bayes dengan asumsi
        distribusi normal pada setiap fitur numerik.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
AQ-CARE © 2026 <br>
Gaussian Naive Bayes Air Quality Prediction System
</div>
""", unsafe_allow_html=True)
