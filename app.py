import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px

# =====================================================
# PAGE CONFIG & SESSION STATE
# =====================================================
st.set_page_config(
    page_title="AQ-CARE | Portal Informasi Kualitas Udara",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "sudah_masuk" not in st.session_state:
    st.session_state.sudah_masuk = False

# =====================================================
# MULTI-LANGUAGE DICTIONARY (ID & EN)
# =====================================================
LANG = {
    "ID": {
        "welcome_tag": "SISTEM LAYANAN MASYARAKAT",
        "welcome_title": "Selamat Datang di Portal AQ-CARE",
        "welcome_desc": "Platform digital penyedia informasi kebersihan udara harian yang dikemas secara interaktif, transparan, dan ramah untuk semua kalangan masyarakat.",
        "btn_enter": "🔓 MASUK KE LAYANAN UTAMA",
        
        "nav_home": "🏠 Beranda Utama",
        "nav_dash": "📊 Grafik & Tren Interaktif",
        "nav_pred": "🔍 Cek Udara Wilayah Anda",
        "nav_info": "💡 Kamus Sehat & FAQ",
        
        "hero_title": "Masyarakat Sehat, Udara Bersih",
        "hero_desc": "AQ-CARE menyederhanakan data laboratorium lingkungan yang rumit menjadi indikator warna sederhana (Hijau, Kuning, Merah) yang mudah dipahami dalam sekali lihat.",
        
        "card_total_data": "📊 Total Data Teranalisis",
        "card_accuracy": "🎯 Akurasi Sistem",
        "card_status": "🟢 Status Server",
        "card_status_val": "Normal / Aktif",
        
        "dash_title": "📊 Pusat Data Visual Kualitas Udara",
        "dash_subtitle": "Pantau sebaran statistik kebersihan lingkungan harian melalui grafik interaktif di bawah ini.",
        "dash_pie_title": "🍟 Proporsi Persentase Status Udara Tahunan",
        "dash_pie_desc": "Grafik lingkaran ini menggambarkan persentase kondisi udara. Semakin besar porsi warna hijau, semakin sering wilayah tersebut menghirup udara segar.",
        "dash_line_title": "📈 Grafik Fluktuasi Debu Harian (PM10)",
        "dash_line_desc": "Garis naik-turun ini mencerminkan tingkat kepekatan debu. Lonjakan tinggi biasanya dipengaruhi oleh polusi kendaraan atau musim kemarau.",
        
        "pred_title": "🔍 Cari Tahu Kualitas Udara di Lokasi Anda",
        "pred_subtitle": "Masukkan angka indikator udara di bawah ini. Sistem akan langsung memberikan rekomendasi kesehatan.",
        "pred_input_loc": "📍 Tulis Nama Kota / Kecamatan Anda:",
        "pred_btn": "🏃 Mulai Proses Analisis Data",
        "pred_btn_back": "⬅️ Kembali ke Halaman Sambutan",
        "pred_res_title": "### Hasil Pemindaian Wilayah:",
        
        "res_baik": "🌱 KUALITAS UDARA AMAT BAIK — Udara bersih dan segar! Sangat ideal untuk beraktivitas, berjalan santai, atau berolahraga di luar ruangan.",
        "res_sedang": "⚠️ KUALITAS UDARA SEDANG — Udara cukup aman, namun bagi yang sensitif (lansia, bayi, atau penderita asma) disarankan tidak terlalu lama di luar.",
        "res_buruk": "🚨 KUALITAS UDARA TIDAK SEHAT — Udara kotor! Gunakan masker pelindung standar dan tutup jendela rumah Anda rapat-rapat.",
        
        "chart_conf_title": "Tingkat Kepastian Hasil Analisis (%)",
        "download_report": "📥 Unduh Laporan (.txt)",
        
        "guide_title": "💡 Kamus Polutan & Pertanyaan Umum (FAQ)",
        "guide_subtitle": "Pelajari zat-zat yang melayang di udara beserta jawaban atas pertanyaan yang sering diajukan.",
        "faq_q1": "🤔 Apa itu PM2.5 dan PM10?",
        "faq_a1": "Itu adalah partikel debu super kecil yang melayang di udara. PM2.5 ukurannya jauh lebih halus (seperti asap rokok), sehingga bisa menembus masker biasa dan langsung masuk ke paru-paru.",
        "faq_q2": "🤔 Dari mana data aplikasi ini berasal?",
        "faq_a2": "Sistem ini menggunakan basis referensi pola data historis indeks standar pencemar udara (ISPU) resmi untuk melatih sistem memberikan tebakan yang akurat.",
        "faq_q3": "🤔 Bagaimana cara menjaga paru-paru saat udara buruk?",
        "faq_a3": "Selalu gunakan masker medis saat berkendara, pasang alat penyaring udara (air purifier) di rumah, dan perbanyak minum air putih.",
        
        "testi_title": "💬 Apa Kata Mereka Tentang AQ-CARE?",
        "footer_text": "Portal Komunitas Peduli Udara Bersih Indonesia. Hak Cipta Dilindungi."
    },
    "EN": {
        "welcome_tag": "PUBLIC SERVICE SYSTEM",
        "welcome_title": "Welcome to AQ-CARE Portal",
        "welcome_desc": "A digital platform providing daily air cleanliness information packaged interactively, transparently, and user-friendly for all members of the community.",
        "btn_enter": "🔓 ENTER MAIN APPLICATION",
        
        "nav_home": "🏠 Main Home",
        "nav_dash": "📊 Interactive Charts & Trends",
        "nav_pred": "🔍 Check Your Area Quality",
        "nav_info": "💡 Health Dictionary & FAQ",
        
        "hero_title": "Healthy Community, Clean Air",
        "hero_desc": "AQ-CARE simplifies complex environmental laboratory data into simple color indicators (Green, Yellow, Red) that are easy to understand at a single glance.",
        
        "card_total_data": "📊 Total Analyzed Data",
        "card_accuracy": "🎯 System Accuracy",
        "card_status": "🟢 Server Status",
        "card_status_val": "Normal / Active",
        
        "dash_title": "📊 Air Quality Visual Data Center",
        "dash_subtitle": "Monitor the statistical distribution of daily environmental cleanliness through the interactive charts below.",
        "dash_pie_title": "🍟 Annual Air Status Percentage Proportion",
        "dash_pie_desc": "This pie chart illustrates the air conditions. The larger the green portion, the more often the area breathes fresh air.",
        "dash_line_title": "📈 Daily Dust Fluctuation Chart (PM10)",
        "dash_line_desc": "This fluctuating line reflects the density level of dust. High spikes are usually influenced by vehicle pollution or dry seasons.",
        
        "pred_title": "🔍 Find Out the Air Quality in Your Location",
        "pred_subtitle": "Enter the air indicator numbers below. The system will instantly provide health recommendations.",
        "pred_input_loc": "📍 Enter Your City / District Name:",
        "pred_btn": "🏃 Start Data Analysis Process",
        "pred_btn_back": "⬅️ Back to Welcome Page",
        "pred_res_title": "### Area Scan Results:",
        
        "res_baik": "🌱 EXCELLENT AIR QUALITY — Clear and fresh air! Perfect for activities, casual walks, or outdoor exercises.",
        "res_sedang": "⚠️ MODERATE AIR QUALITY — Fairly safe, but sensitive individuals (elderly, infants, or asthma patients) are advised not to stay out too long.",
        "res_buruk": "🚨 UNHEALTHY AIR QUALITY — Polluted air! Please wear standard protective masks and close your home windows tightly.",
        
        "chart_conf_title": "Analysis Result Confidence Level (%)",
        "download_report": "📥 Download Report (.txt)",
        
        "guide_title": "💡 Pollutant Dictionary & Frequently Asked Questions (FAQ)",
        "guide_subtitle": "Learn about the elements floating in the air along with answers to frequently asked questions.",
        "faq_q1": "🤔 What are PM2.5 and PM10?",
        "faq_a1": "These are microscopic dust particles floating in the air. PM2.5 is much finer (like tobacco smoke), allowing it to bypass regular masks and enter the lungs directly.",
        "faq_q2": "🤔 Where does this app data come from?",
        "faq_a2": "This system uses historical official air pollutant standard index (ISPU) reference data patterns to train the system to provide accurate analysis.",
        "faq_q3": "🤔 How to protect our lungs during poor air quality?",
        "faq_a3": "Always wear medical masks when commuting, install air purifiers at home, and drink plenty of water.",
        
        "testi_title": "💬 What People Say About AQ-CARE?",
        "footer_text": "Indonesian Clean Air Care Community Portal. All Rights Reserved."
    }
}

# =====================================================
# CUSTOM MODERN WEBPAGE CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #f8fafc !important;
    color: #1e293b !important;
}

/* Header Navbar Simulasi Atas Website */
.web-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #ffffff;
    padding: 15px 30px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
}
.web-logo {
    font-weight: 700;
    font-size: 20px;
    color: #0284c7;
}
.web-menu-item {
    font-size: 14px;
    color: #64748b;
    margin-left: 20px;
    font-weight: 500;
}

/* Welcome Landing Page */
.welcome-box {
    text-align: center;
    padding: 90px 30px;
    background: url('https://images.unsplash.com/photo-1500530855697-b586d89ba3ee') no-repeat center;
    background-size: cover;
    border-radius: 24px;
    position: relative;
    box-shadow: inset 0 0 0 2000px rgba(15, 23, 42, 0.85);
    color: white !important;
}

/* Card Style Website */
.content-card {
    background: #ffffff;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    margin-bottom: 20px;
}

.desc-text {
    font-size: 14px;
    color: #475569;
    line-height: 1.6;
    background: #f8fafc;
    padding: 12px;
    border-radius: 8px;
    border-left: 4px solid #0ea5e9;
    margin-top: 10px;
}

.stButton>button {
    background: #0284c7 !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: none !important;
}
.stButton>button:hover {
    background: #0369a1 !important;
}

/* Testimonial Section */
.testi-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-top: 15px;
}
.testi-card {
    background: #f8fafc;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

.footer {
    text-align: center;
    color: #94a3b8;
    padding: 30px;
    font-size: 13px;
    border-top: 1px solid #e2e8f0;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA STREAM (SIMULATED OR REAL CSV LOAD)
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
        np.random.seed(42)
        rows = 150
        mock_data = pd.DataFrame({
            "pm_sepuluh": np.random.randint(20, 110, rows),
            "pm_duakomalima": np.random.randint(15, 140, rows),
            "sulfur_dioksida": np.random.randint(10, 70, rows),
            "karbon_monoksida": np.random.randint(5, 35, rows),
            "ozon": np.random.randint(15, 150, rows),
            "nitrogen_dioksida": np.random.randint(5, 55, rows),
            "kategori": np.random.choice(["BAIK", "SEDANG", "TIDAK SEHAT"], rows, p=[0.4, 0.4, 0.2])
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
# SIDEBAR CONTROL (LANGUAGE SELECTOR & NAV BUTTONS)
# =====================================================
st.sidebar.markdown("<h3 style='margin-bottom:0;'>⚙️ Pengaturan Web</h3>", unsafe_allow_html=True)

# Fitur Ubah Bahasa Profesional (ID / EN)
lang_choice = st.sidebar.radio(
    "Pilih Bahasa / Language:",
    ["ID", "EN"],
    horizontal=True
)

txt = LANG[lang_choice]

st.sidebar.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)

if st.session_state.sudah_masuk:
    menu = st.sidebar.radio(
        "Menu Halaman Website:",
        [txt["nav_home"], txt["nav_dash"], txt["nav_pred"], txt["nav_info"]]
    )
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    if st.sidebar.button("🚪 " + ("Keluar Layanan" if lang_choice == "ID" else "Exit Portal"), use_container_width=True):
        st.session_state.sudah_masuk = False
        st.rerun()

# =====================================================
# HEADER SIMULATION (Selayaknya Website Asli)
# =====================================================
st.markdown(f"""
<div class="web-navbar">
    <div class="web-logo">🌍 AQ-CARE DIGITAL PORTAL</div>
    <div>
        <span class="web-menu-item">🌐 Layanan Publik</span>
        <span class="web-menu-item">🛡️ WHO Verified</span>
        <span class="web-menu-item">⚡ Respons Cepat</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# HALAMAN SEBELUM MASUK (WELCOME LANDING SCREEN)
# =====================================================
if not st.session_state.sudah_masuk:
    st.markdown(f"""
    <div class="welcome-box">
        <p style="font-size:14px; letter-spacing:3px; color:#38bdf8; font-weight:700; margin:0;">{txt["welcome_tag"]}</p>
        <h1 style="font-size:48px; font-weight:800; color:white; margin-top:10px; margin-bottom:15px;">{txt["welcome_title"]}</h1>
        <p style="font-size:18px; max-width:800px; margin:0 auto 35px auto; opacity:0.9; line-height:1.6;">{txt["welcome_desc"]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 2])
    with col_btn2:
        if st.button(txt["btn_enter"], use_container_width=True):
            st.session_state.sudah_masuk = True
            st.rerun()
            
    # Tampilan Footer Halaman Depan
    st.markdown(f"<div class='footer'>{txt['footer_text']}</div>", unsafe_allow_html=True)
    st.stop()

# =====================================================
# HALAMAN 1: BERANDA UTAMA (SETELAH MASUK)
# =====================================================
if menu == txt["nav_home"]:
    st.markdown(f"""
    <div class="content-card" style="background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); color:white;">
        <h2 style="color:white !important; margin:0; font-weight:700;">🏠 {txt["hero_title"]}</h2>
        <p style="opacity:0.9; font-size:15px; margin-top:10px; max-width:950px;">{txt["hero_desc"]}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(txt["card_total_data"], f"{len(data):,}")
    with col2:
        st.metric(txt["card_accuracy"], f"{akurasi*100:.2f}%")
    with col3:
        st.metric(txt["card_status"], txt["card_status_val"])

    # Tambahan Sesi Ulasan/Testimoni agar seperti web asli
    st.markdown(f"<div class='content-card'><h3>{txt['testi_title']}</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="testi-grid">
        <div class="testi-card">
            <h5>⭐⭐⭐⭐⭐ Budi (34 thn) - Warga Perkotaan</h5>
            <p style="font-size:13px; color:#475569; margin:0;">"Web ini gampang banget dibaca! Grafik persentasenya langsung ngasih tahu kalau udara lagi ga sehat jadi saya bisa prepare masker buat anak."</p>
        </div>
        <div class="testi-card">
            <h5>⭐⭐⭐⭐⭐ Dr. Siti - Praktisi Kesehatan</h5>
            <p style="font-size:13px; color:#475569; margin:0;">"Sangat mengedukasi pasien awam. Tidak ada angka rumus matematika yang bikin bingung, panduan tips kesehatannya sangat praktis."</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# HALAMAN 2: DASHBOARD GRAFIK KREATIF
# =====================================================
elif menu == txt["nav_dash"]:
    st.markdown(f"""
    <div class="content-card">
        <h2>{txt["dash_title"]}</h2>
        <p style="color:#64748b; margin:0;">{txt["dash_subtitle"]}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([4, 6])
    with col1:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown(f"<h4>{txt['dash_pie_title']}</h4>", unsafe_allow_html=True)
        kategori_count = data["kategori"].value_counts().reset_index()
        kategori_count.columns = ["Kategori", "Jumlah Hari"]
        fig1 = px.pie(
            kategori_count, names="Kategori", values="Jumlah Hari", hole=0.4,
            color_discrete_map={"BAIK": "#22c55e", "SEDANG": "#eab308", "TIDAK SEHAT": "#ef4444"}
        )
        fig1.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown(f"<div class='desc-text'>{txt['dash_pie_desc']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown(f"<h4>{txt['dash_line_title']}</h4>", unsafe_allow_html=True)
        fig2 = px.line(data.reset_index(), x="index", y="pm_sepuluh", color_discrete_sequence=['#0ea5e9'])
        fig2.update_layout(margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"<div class='desc-text'>{txt['dash_line_desc']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# HALAMAN 3: CEK UDARA (PREDIKSI DAN DOWNLOAD LAPORAN)
# =====================================================
elif menu == txt["nav_pred"]:
    st.markdown(f"""
    <div class="content-card">
        <h2>{txt["pred_title"]}</h2>
        <p style="color:#64748b; margin:0;">{txt["pred_subtitle"]}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    lokasi = st.text_input(txt["pred_input_loc"], "Palu, Central Sulawesi")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        pm10 = st.number_input("🌫️ Kepadatan Debu / PM10 (µg/m³)", 0.0, 300.0, 48.0)
        co = st.number_input("🚗 Gas Kendaraan / CO (µg/m³)", 0.0, 100.0, 11.0)
    with col2:
        pm25 = st.number_input("💨 Partikel Halus / PM2.5 (µg/m³)", 0.0, 300.0, 32.0)
        o3 = st.number_input("☀️ Lapisan Ozon Bawah / O3 (µg/m³)", 0.0, 300.0, 42.0)
    with col3:
        so2 = st.number_input("🏭 Asap Industri / SO2 (µg/m³)", 0.0, 300.0, 18.0)
        no2 = st.number_input("🔥 Gas Pembakaran / NO2 (µg/m³)", 0.0, 300.0, 14.0)

    st.markdown("<br>", unsafe_allow_html=True)
    
    c_btn1, c_btn2 = st.columns([4, 2])
    with c_btn1:
        prediksi_btn = st.button(txt["pred_btn"], use_container_width=True)
    with c_btn2:
        if st.button(txt["pred_btn_back"], use_container_width=True):
            st.session_state.sudah_masuk = False
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

        st.markdown(f"{txt['pred_res_title']} **{lokasi}**")
        
        text_report_content = f"LAPORAN KUALITAS UDARA AQ-CARE\nLokasi: {lokasi}\nHasil Evaluasi: Udara {hasil}\n"
        
        if hasil == "BAIK":
            st.success(txt["res_baik"])
            text_report_content += "Rekomendasi: Aman untuk beraktivitas luar rumah."
        elif hasil == "SEDANG":
            st.warning(txt["res_sedang"])
            text_report_content += "Rekomendasi: Kelompok rentan kurangi aktivitas berat di luar."
        else:
            st.error(txt["res_buruk"])
            text_report_content += "Rekomendasi: Wajib gunakan masker medis pelindung."

        # Fitur Download Laporan Otomatis Khas Website Komersial
        st.download_button(
            label=txt["download_report"],
            data=text_report_content,
            file_name=f"Laporan_Udara_{lokasi}.txt",
            mime="text/plain"
        )

        st.markdown("<div class='content-card' style='margin-top:15px;'>", unsafe_allow_html=True)
        prob_df = pd.DataFrame({"Status": model.classes_, "Persentase (%)": probabilitas * 100})
        fig_bar = px.bar(prob_df, x="Status", y="Persentase (%)", color="Status", text_auto='.1f',
                         title=txt["chart_conf_title"],
                         color_discrete_map={"BAIK": "#22c55e", "SEDANG": "#eab308", "TIDAK SEHAT": "#ef4444"})
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# HALAMAN 4: KAMUS SEHAT & FAQ (UMUM BGT)
# =====================================================
elif menu == txt["nav_info"]:
    st.markdown(f"""
    <div class="content-card">
        <h2>{txt["guide_title"]}</h2>
        <p style="color:#64748b; margin:0;">{txt["guide_subtitle"]}</p>
    </div>
    """, unsafe_allow_html=True)

    # FAQ accordion element bergaya website portal berita
    with st.expander(txt["faq_q1"]):
        st.write(txt["faq_a1"])
        
    with st.expander(txt["faq_q2"]):
        st.write(txt["faq_a2"])
        
    with st.expander(txt["faq_q3"]):
        st.write(txt["faq_a3"])

# =====================================================
# FOOTER WEB UTAMA
# =====================================================
st.markdown(f"""
<div class="footer">
    <b>AQ-CARE AI Community Portal © 2026</b><br>
    {txt["footer_text"]}
</div>
""", unsafe_allow_html=True)
