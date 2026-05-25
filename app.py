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
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Navbar */
.navbar {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 20px;
}

/* Card */
.card {
    background: rgba(255,255,255,0.10);
    backdrop-filter: blur(15px);
    border-radius: 25px;
    padding: 35px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.1);
}

/* Input */
.stTextInput>div>div>input,
.stNumberInput>div>div>input {
    background-color: rgba(255,255,255,0.15);
    color: white;
    border-radius: 15px;
    border: none;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
    border: none;
    border-radius: 15px;
    padding: 14px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
    transition: 0.3s;
    box-shadow: 0 4px 15px rgba(0,114,255,0.4);
}

.stButton>button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg,#0072ff,#00c6ff);
}

/* Metric */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.15);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.25);
}

/* Text */
h1, h2, h3, h4 {
    color: white !important;
}

p, li {
    color: #ecf0f1;
    font-size: 17px;
}

/* Footer */
.footer {
    text-align:center;
    color:white;
    padding:20px;
    font-size:14px;
}

/* Hero Section */
.hero {
    text-align:center;
    padding:60px 20px;
}

.hero-title {
    font-size:60px;
    font-weight:700;
    color:white;
}

.hero-sub {
    font-size:22px;
    color:#dfefff;
    margin-top:10px;
}

/* Recommendation box */
.rekom {
    background: rgba(255,255,255,0.1);
    padding:20px;
    border-radius:20px;
    margin-top:20px;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.2);
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
# NAVIGATION MENU
# =====================================================
selected = option_menu(
    menu_title=None,
    options=["Home", "Prediksi", "Dashboard", "About"],
    icons=["house", "activity", "bar-chart", "info-circle"],
    orientation="horizontal",
)

# =====================================================
# HOME PAGE
# =====================================================
if selected == "Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">🌍 AQ-CARE</div>
        <div class="hero-sub">
        Smart Air Quality Prediction System <br>
        Using Gaussian Naive Bayes Machine Learning
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns(3)

    col1.metric("📊 Total Data", len(data))
    col2.metric("🎯 Akurasi Model", f"{akurasi*100:.2f}%")
    col3.metric("🧠 Metode", "Gaussian NB")

    st.write("")

    st.markdown("""
    <div class="card">
        <h2>✨ Tentang AQ-CARE</h2>

        <p>
        AQ-CARE adalah website prediksi kualitas udara berbasis 
        Machine Learning menggunakan metode Gaussian Naive Bayes.
        </p>

        <p>
        Sistem ini dapat membantu pengguna mengetahui kondisi kualitas udara 
        berdasarkan konsentrasi polutan secara cepat dan interaktif.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.image(
        "https://images.unsplash.com/photo-1506744038136-46273834b3fb",
        use_container_width=True
    )

# =====================================================
# PREDIKSI PAGE
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
            title="📊 Probabilitas Prediksi"
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.08)',
            font_color='white'
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="rekom">
        <h3>💡 Rekomendasi</h3>
        </div>
        """, unsafe_allow_html=True)

        if hasil == "BAIK":
            st.info("""
            ✅ Udara aman untuk aktivitas luar ruangan  
            ✅ Cocok untuk olahraga  
            ✅ Tidak perlu masker
            """)

        elif hasil == "SEDANG":
            st.warning("""
            ⚠️ Kelompok sensitif disarankan mengurangi aktivitas luar ruangan  
            ⚠️ Gunakan masker bila diperlukan
            """)

        else:
            st.error("""
            ❌ Hindari aktivitas luar ruangan  
            ❌ Gunakan masker  
            ❌ Tutup ventilasi rumah
            """)

# =====================================================
# DASHBOARD PAGE
# =====================================================
elif selected == "Dashboard":

    st.markdown("""
    <div class="card">
        <h2>📈 Dashboard Kualitas Udara</h2>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    fig1 = px.histogram(
        data,
        x="kategori",
        color="kategori",
        title="Distribusi Kategori Kualitas Udara"
    )

    fig1.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.08)',
        font_color='white'
    )

    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        data,
        x="pm_sepuluh",
        y="pm_duakomalima",
        color="kategori",
        title="Sebaran PM10 vs PM2.5"
    )

    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.08)',
        font_color='white'
    )

    st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# ABOUT PAGE
# =====================================================
elif selected == "About":

    st.markdown("""
    <div class="card">

    <h1>ℹ️ About AQ-CARE</h1>

    <p>
    AQ-CARE merupakan website prediksi kualitas udara berbasis 
    Machine Learning menggunakan metode Gaussian Naive Bayes.
    </p>

    <h3>📌 Variabel</h3>

    <ul>
        <li>PM10</li>
        <li>PM2.5</li>
        <li>SO2</li>
        <li>CO</li>
        <li>O3</li>
        <li>NO2</li>
    </ul>

    <h3>🧠 Metode</h3>

    <p>
    Gaussian Naive Bayes menggunakan Teorema Bayes 
    dengan asumsi distribusi normal pada setiap fitur numerik.
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
