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
# MULTI-LANGUAGE DICTIONARY (ID & EN COMPLETELY)
# =====================================================
LANG = {
    "ID": {
        "web_navbar_title": "🌍 AQ-CARE PORTAL DIGITAL",
        "web_nav_badge1": "🌐 Layanan Publik",
        "web_nav_badge2": "🛡️ Terverifikasi Standar",
        "web_nav_badge3": "⚡ Respons Cepat",
        
        "welcome_tag": "SISTEM LAYANAN MASYARAKAT",
        "welcome_title": "Selamat Datang di Portal AQ-CARE",
        "welcome_desc": "Platform digital penyedia informasi kebersihan udara harian yang dikemas secara interaktif, transparan, dan ramah untuk semua kalangan masyarakat.",
        "btn_enter": "🔓 MASUK KE LAYANAN UTAMA",
        
        "nav_home": "🏠 Beranda Utama",
        "nav_dash": "📊 Grafik & Tren Interaktif",
        "nav_pred": "🔍 Cek Udara Wilayah Anda",
        "nav_info": "💡 Kamus Sehat & FAQ Lengkap",
        
        "hero_title": "Masyarakat Sehat, Udara Bersih",
        "hero_desc": "AQ-CARE menyederhanakan data laboratorium lingkungan yang rumit menjadi indikator warna sederhana (Hijau, Kuning, Merah) yang mudah dipahami dalam sekali lihat tanpa rumus matematika yang membingungkan.",
        
        "card_total_data": "📊 Total Data Teranalisis",
        "card_accuracy": "🎯 Akurasi Sistem",
        "card_status": "🟢 Status Server",
        "card_status_val": "Normal / Aktif",
        
        "dash_title": "📊 Pusat Data Visual Kualitas Udara",
        "dash_subtitle": "Pantau sebaran statistik kebersihan lingkungan harian melalui grafik interaktif di bawah ini.",
        "dash_pie_title": "Proporsi Persentase Status Udara Tahunan",
        "dash_pie_desc": "Grafik lingkaran ini menggambarkan persentase kondisi udara. Semakin besar porsi warna hijau, semakin sering wilayah tersebut menghirup udara segar.",
        "dash_line_title": "Grafik Fluktuasi Debu Harian (PM10)",
        "dash_line_desc": "Garis naik-turun ini mencerminkan tingkat kepekatan debu. Lonjakan tinggi biasanya dipengaruhi oleh polusi kendaraan atau musim kemarau.",
        "graph_axis_index": "Sampel Hari ke-",
        "graph_axis_pm10": "Tingkat Kepadatan Debu",
        
        "pred_title": "🔍 Cari Tahu Kualitas Udara di Lokasi Anda",
        "pred_subtitle": "Masukkan angka indikator parameter udara di bawah ini. Sistem cerdas akan langsung memproses dan memberikan rekomendasi kesehatan.",
        "pred_input_loc": "📍 Tulis Nama Kota / Kecamatan Anda:",
        "pred_btn": "🏃 Mulai Proses Analisis Data",
        "pred_btn_back": "⬅️ Kembali ke Halaman Sambutan",
        "pred_res_title": "### Hasil Pemindaian Wilayah:",
        
        "label_pm10": "🌫️ Kepadatan Debu Kasar / PM10 (µg/m³)",
        "label_co": "🚗 Gas Kendaraan Bermotor / CO (µg/m³)",
        "label_pm25": "💨 Partikel Debu Halus / PM2.5 (µg/m³)",
        "label_o3": "☀️ Lapisan Ozon Bawah / O3 (µg/m³)",
        "label_so2": "🏭 Asap Industri Pabrik / SO2 (µg/m³)",
        "label_no2": "🔥 Gas Hasil Pembakaran / NO2 (µg/m³)",
        
        "cat_baik": "BAIK",
        "cat_sedang": "SEDANG",
        "cat_buruk": "TIDAK SEHAT",
        
        "res_baik": "🌱 KUALITAS UDARA AMAT BAIK — Udara bersih dan segar! Sangat ideal untuk beraktivitas, berjalan santai, atau berolahraga di luar ruangan tanpa khawatir.",
        "res_sedang": "⚠️ KUALITAS UDARA SEDANG — Udara cukup aman, namun bagi yang sensitif (lansia, bayi, atau penderita asma) disarankan tidak terlalu lama beraktivitas berat di luar.",
        "res_buruk": "🚨 KUALITAS UDARA TIDAK SEHAT — Udara kotor! Gunakan masker pelindung standar (N95/Medis) dan tutup jendela rumah Anda rapat-rapat.",
        
        "chart_conf_title": "Tingkat Kepastian Hasil Analisis (%)",
        "download_report": "📥 Unduh Laporan Resmi (.txt)",
        
        "guide_title": "💡 Ensiklopedia Polutan & Pertanyaan Umum (FAQ)",
        "guide_subtitle": "Pelajari secara mendalam zat-zat berbahaya yang melayang di udara beserta panduan mitigasi kesehatan.",
        
        "pol_title_pm25": "💨 Partikulat Makro & Mikro (PM10 & PM2.5)",
        "pol_desc_pm25": "PM10 adalah partikel udara dengan diameter kurang dari 10 mikrometer, sedangkan PM2.5 jauh lebih kecil (kurang dari 2.5 mikrometer). Sumber utamanya berasal dari debu jalanan, pembakaran lahan, dan asap kendaraan. Karena ukurannya yang sangat mikro, PM2.5 berbahaya karena tidak dapat disaring oleh bulu hidung manusia, sehingga langsung menembus alveolus paru-paru dan masuk ke sistem peredaran darah, memicu penyakit kardiovaskular serta ISPA.",
        
        "pol_title_co": "🚗 Karbon Monoksida (CO)",
        "pol_desc_co": "Gas yang sama sekali tidak berwarna, tidak berbau, dan tidak berasa. Zat ini dihasilkan dari pembakaran bahan bakar fosil yang tidak sempurna pada mesin kendaraan dan mesin pabrik. Bahaya utamanya adalah sifat CO yang sangat mudah mengikat Hemoglobin (Hb) di dalam darah manusia, bahkan 200 kali lebih kuat daripada oksigen. Menghirup CO kadar tinggi menyebabkan tubuh kekurangan oksigen secara mendadak, memicu pusing kepala, mual, pingsan, hingga kerusakan sistem saraf pusat.",
        
        "pol_title_so2": "🏭 Sulfur Dioksida (SO2)",
        "pol_desc_so2": "Gas berbau tajam dan sangat perih di tenggorokan yang terbentuk dari pembakaran bahan bakar yang mengandung sulfur, seperti batu bara dan minyak bumi pada pembangkit listrik atau kilang industri. Ketika SO2 bertemu dengan uap air di udara, ia membentuk asam korosif yang merusak jaringan sensitif mukosa hidung. Paparan jangka pendek langsung memicu serangan asma akut, batuk berdahak, serta iritasi mata parah.",
        
        "pol_title_o3": "☀️ Ozon Permukaan (O3)",
        "pol_desc_o3": "Berbeda dengan lapisan ozon di stratosfer yang melindungi bumi dari radiasi ultraviolet, ozon permukaan adalah polutan sekunder yang merugikan. Gas ini terbentuk di permukaan tanah akibat reaksi kimia antara gas hidrokarbon (VOC) dan Nitrogen Dioksida (NOx) di bawah terik sinar matahari yang menyengat. Gas ozon ini bersifat oksidator kuat yang mampu mengikis elastisitas jaringan paru-paru, memicu nyeri dada saat bernapas dalam, serta menurunkan fungsi imun pernapasan.",
        
        "pol_title_no2": "🔥 Nitrogen Dioksida (NO2)",
        "pol_desc_no2": "Zat gas berwarna cokelat kemerahan dan berbau menyengat yang dilepaskan dari hasil pembakaran bersuhu tinggi, seperti pada kompor gas, pembakaran sampah, dan mesin diesel. NO2 berperan besar sebagai komponen utama pembentuk kabut asap (smog). Gas ini sangat toksik bagi saluran napas, merusak sel-sel silia penapis bakteri di paru-paru, sehingga tubuh menjadi jauh lebih rentan terhadap infeksi virus dan bronkitis kronis.",
        
        "faq_q1": "🤔 Bagaimana cara membaca indikator kualitas udara secara mandiri?",
        "faq_a1": "Cara termudah adalah melihat klasifikasi warnanya. Nilai di bawah 50 (Hijau) berarti lingkungan bersih sempurna. Nilai 51-100 (Kuning) berarti sedang dan aman untuk masyarakat umum. Sedangkan nilai di atas 100 (Merah) mengindikasikan polusi tinggi berbahaya.",
        "faq_q2": "🤔 Mengapa sistem prediksi AI AQ-CARE menggunakan metode Naive Bayes?",
        "faq_a2": "Metode Naive Bayes sangat andal dan efisien dalam memproses data klasifikasi berbasis probabilitas matematika. Metode ini mampu mengukur kontribusi masing-masing gas polutan secara simultan untuk menghasilkan keputusan status udara secara instan tanpa membebani memori komputasi.",
        "faq_q3": "🤔 Tindakan apa yang paling efektif saat polusi udara mencapai level merah?",
        "faq_a3": "1. Hindari aktivitas fisik berat di luar rumah seperti jogging atau bersepeda.\n2. Nyalakan air purifier yang dilengkapi filter HEPA di dalam ruangan.\n3. Gunakan masker respirator khusus (seperti N95 atau KN95) jika terpaksa harus keluar rumah, karena masker kain biasa tidak mampu menyaring debu mikro PM2.5.",
        
        "testi_title": "💬 Apa Kata Mereka Tentang AQ-CARE?",
        "testi_user1": "<h5>⭐⭐⭐⭐• Budi (34 thn) - Warga Perkotaan</h5>",
        "testi_comment1": '"Web ini gampang banget dibaca! Grafik persentasenya langsung ngasih tahu kalau udara lagi ga sehat jadi saya bisa bersiap memakai masker untuk keluarga."',
        "testi_user2": "<h5>⭐⭐⭐⭐⭐ Dr. Siti - Praktisi Kesehatan</h5>",
        "testi_comment2": '"Sangat mengedukasi pasien awam. Tidak ada angka rumus matematika yang bikin bingung, panduan tips kesehatannya sangat komprehensif dan praktis."',
        
        "footer_text": "Portal Komunitas Peduli Udara Bersih Indonesia. Hak Cipta Dilindungi.",
        "sidebar_setting_title": "⚙️ Pengaturan Web",
        "sidebar_lang_label": "Pilih Bahasa / Language:",
        "sidebar_menu_title": "Menu Halaman Website:",
        "sidebar_logout_btn": "Keluar Layanan"
    },
    "EN": {
        "web_navbar_title": "🌍 AQ-CARE DIGITAL PORTAL",
        "web_nav_badge1": "🌐 Public Service",
        "web_nav_badge2": "🛡️ Standard Verified",
        "web_nav_badge3": "⚡ Fast Response",
        
        "welcome_tag": "PUBLIC SERVICE SYSTEM",
        "welcome_title": "Welcome to AQ-CARE Portal",
        "welcome_desc": "A digital platform providing daily air cleanliness information packaged interactively, transparently, and user-friendly for all members of the community.",
        "btn_enter": "🔓 ENTER MAIN APPLICATION",
        
        "nav_home": "🏠 Main Home",
        "nav_dash": "📊 Interactive Charts & Trends",
        "nav_pred": "🔍 Check Your Area Quality",
        "nav_info": "💡 Health Dictionary & Full FAQ",
        
        "hero_title": "Healthy Community, Clean Air",
        "hero_desc": "AQ-CARE simplifies complex environmental laboratory data into simple color indicators (Green, Yellow, Red) that are easy to understand at a single glance without confusing mathematical formulas.",
        
        "card_total_data": "📊 Total Analyzed Data",
        "card_accuracy": "🎯 System Accuracy",
        "card_status": "🟢 Server Status",
        "card_status_val": "Normal / Active",
        
        "dash_title": "📊 Air Quality Visual Data Center",
        "dash_subtitle": "Monitor the statistical distribution of daily environmental cleanliness through the interactive charts below.",
        "dash_pie_title": "Annual Air Status Percentage Proportion",
        "dash_pie_desc": "This pie chart illustrates the air condition percentages. The larger the green portion, the more often the area breathes fresh air.",
        "dash_line_title": "Daily Dust Fluctuation Chart (PM10)",
        "dash_line_desc": "This fluctuating line reflects the density level of dust. High spikes are usually influenced by vehicle pollution or dry seasons.",
        "graph_axis_index": "Day Sample No.",
        "graph_axis_pm10": "Dust Density Level",
        
        "pred_title": "🔍 Find Out the Air Quality in Your Location",
        "pred_subtitle": "Enter the air parameter indicator numbers below. The intelligent system will instantly process and provide health recommendations.",
        "pred_input_loc": "📍 Enter Your City / District Name:",
        "pred_btn": "🏃 Start Data Analysis Process",
        "pred_btn_back": "⬅️ Back to Welcome Page",
        "pred_res_title": "### Area Scan Results:",
        
        "label_pm10": "🌫️ Coarse Dust Density / PM10 (µg/m³)",
        "label_co": "🚗 Motor Vehicle Gas / CO (µg/m³)",
        "label_pm25": "💨 Fine Dust Particles / PM2.5 (µg/m³)",
        "label_o3": "☀️ Ground-Level Ozone / O3 (µg/m³)",
        "label_so2": "🏭 Industrial Factory Smoke / SO2 (µg/m³)",
        "label_no2": "🔥 Combustion Gas Emissions / NO2 (µg/m³)",
        
        "cat_baik": "GOOD",
        "cat_sedang": "MODERATE",
        "cat_buruk": "UNHEALTHY",
        
        "res_baik": "🌱 EXCELLENT AIR QUALITY — Clear and fresh air! Perfect for activities, casual walks, or outdoor exercises without any worries.",
        "res_sedang": "⚠️ MODERATE AIR QUALITY — Fairly safe, but sensitive individuals (elderly, infants, or asthma patients) are advised to avoid prolonged heavy outdoor activities.",
        "res_buruk": "🚨 UNHEALTHY AIR QUALITY — Polluted air! Please wear standard protective masks (N95/Medical) and close your home windows tightly.",
        
        "chart_conf_title": "Analysis Result Confidence Level (%)",
        "download_report": "📥 Download Official Report (.txt)",
        
        "guide_title": "💡 Pollutant Encyclopedia & Frequently Asked Questions (FAQ)",
        "guide_subtitle": "Learn deeply about the dangerous substances floating in the air along with health mitigation guidelines.",
        
        "pol_title_pm25": "💨 Macro & Micro Particulate Matter (PM10 & PM2.5)",
        "pol_desc_pm25": "PM10 refers to air particles with a diameter under 10 micrometers, while PM2.5 is much smaller (less than 2.5 micrometers). The main sources are road dust, agricultural burning, and vehicle exhaust. Due to its microscopic size, PM2.5 is exceptionally hazardous because it bypasses human nasal hair filtration, penetrating directly into the lung alveoli and blood stream, triggering cardiovascular diseases and acute respiratory infections.",
        
        "pol_title_co": "🚗 Carbon Monoxide (CO)",
        "pol_desc_co": "A completely colorless, odorless, and tasteless gas. This substance is produced by incomplete combustion of fossil fuels in vehicles and industrial machinery. Its primary danger lies in its extreme affinity for hemoglobin (Hb) in human blood, binding 200 times stronger than oxygen. Inhaling high concentrations of CO deprives the body of oxygen rapidly, causing headaches, nausea, fainting, and central nervous system damage.",
        
        "pol_title_so2": "🏭 Sulfur Dioxide (SO2)",
        "pol_desc_so2": "A sharp, choking gas formed by burning sulfur-containing fuels like coal and oil in power plants or refineries. When SO2 mixes with water vapor in the atmosphere, it forms corrosive acids that irritate the sensitive mucosal tissues of the respiratory tract. Short-term exposure immediately triggers acute asthma attacks, wheezing, and severe eye irritation.",
        
        "pol_title_o3": "☀️ Ground-Level Ozone (O3)",
        "pol_desc_o3": "Unlike the stratospheric ozone layer protecting the earth from UV rays, ground-level ozone is a harmful secondary pollutant. It is formed near the ground through chemical reactions between volatile organic compounds (VOCs) and Nitrogen Oxides (NOx) under intense sunlight. This ozone gas is a powerful oxidant that erodes lung tissue elasticity, causes chest pain during deep breathing, and suppresses immune system responses.",
        
        "pol_title_no2": "🔥 Nitrogen Dioxide (NO2)",
        "pol_desc_no2": "A reddish-brown, foul-smelling gas released from high-temperature combustion processes, such as gas stoves, waste burning, and diesel engines. NO2 is a major component of urban smog. The gas is highly toxic to the airways, damaging the ciliated cells that filter bacteria out of the lungs, making the body much more susceptible to viral infections and chronic bronchitis.",
        
        "faq_q1": "🤔 How do I interpret the air quality indicators on my own?",
        "faq_a1": "The easiest way is to follow the color codes. Values under 50 (Green) indicate a perfectly clean environment. Values between 51-100 (Yellow) signify moderate and safe conditions for the general public. Meanwhile, values above 100 (Red) point to high, hazardous pollution levels.",
        "faq_q2": "🤔 Why does the AQ-CARE AI prediction model utilize Naive Bayes?",
        "faq_a2": "The Naive Bayes method is highly reliable and efficient for mathematical probability-based classification tasks. It measures the simultaneous contribution of each pollutant gas to output instant air status decisions without exhausting computing memory.",
        "faq_q3": "🤔 What are the most effective measures to take when pollution levels reach red?",
        "faq_a3": "1. Avoid heavy physical strain outdoors, such as jogging or cycling.\n2. Turn on indoor air purifiers equipped with a certified HEPA filter.\n3. Always wear specialized respiratory masks (like N95 or KN95) if you must step outside, as regular cloth masks cannot filter microscopic PM2.5 dust.",
        
        "testi_title": "💬 What People Say About AQ-CARE?",
        "testi_user1": "<h5>⭐⭐⭐⭐• Budi (34 yo) - Urban Resident</h5>",
        "testi_comment1": '"This website is so easy to read! The percentage charts instantly inform me when the air gets unhealthy, so I can prepare masks for my family."',
        "testi_user2": "<h5>⭐⭐⭐⭐⭐ Dr. Siti - Health Practitioner</h5>",
        "testi_comment2": '"Extremely educational for everyday citizens. There are no confusing mathematical formulas, and the health guidelines are highly comprehensive and practical."',
        
        "footer_text": "Indonesia Clean Air Care Community Portal. All Rights Reserved.",
        "sidebar_setting_title": "⚙️ Web Settings",
        "sidebar_lang_label": "Pilih Bahasa / Language:",
        "sidebar_menu_title": "Website Page Menu:",
        "sidebar_logout_btn": "Exit Portal"
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
# SIDEBAR CONTROL & LANGUAGE INITIALIZATION
# =====================================================
lang_choice = st.sidebar.radio(
    "Pilih Bahasa / Language:",
    ["ID", "EN"],
    horizontal=True
)

txt = LANG[lang_choice]

# PERBAIKAN UTAMA: Global Map untuk menyelaraskan kategori grafik dengan bahasa pilihan
label_map = {"BAIK": txt["cat_baik"], "SEDANG": txt["cat_sedang"], "TIDAK SEHAT": txt["cat_buruk"]}

st.sidebar.markdown(f"<h3 style='margin-bottom:0;'>{txt['sidebar_setting_title']}</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)

if st.session_state.sudah_masuk:
    menu = st.sidebar.radio(
        txt["sidebar_menu_title"],
        [txt["nav_home"], txt["nav_dash"], txt["nav_pred"], txt["nav_info"]]
    )
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    if st.sidebar.button("🚪 " + txt["sidebar_logout_btn"], use_container_width=True):
        st.session_state.sudah_masuk = False
        st.rerun()

# =====================================================
# HEADER SIMULATION
# =====================================================
st.markdown(f"""
<div class="web-navbar">
    <div class="web-logo">{txt["web_navbar_title"]}</div>
    <div>
        <span class="web-menu-item">{txt["web_nav_badge1"]}</span>
        <span class="web-menu-item">{txt["web_nav_badge2"]}</span>
        <span class="web-menu-item">{txt["web_nav_badge3"]}</span>
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

    st.markdown(f"<div class='content-card'><h3>{txt['testi_title']}</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="testi-grid">
        <div class="testi-card">
            {txt["testi_user1"]}
            <p style="font-size:13px; color:#475569; margin:0;">{txt["testi_comment1"]}</p>
        </div>
        <div class="testi-card">
            {txt["testi_user2"]}
            <p style="font-size:13px; color:#475569; margin:0;">{txt["testi_comment2"]}</p>
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
        
        kategori_count["Kategori"] = kategori_count["Kategori"].map(label_map)
        
        fig1 = px.pie(
            kategori_count, names="Kategori", values="Jumlah Hari", hole=0.4,
            color="Kategori",
            color_discrete_map={txt["cat_baik"]: "#22c55e", txt["cat_sedang"]: "#eab308", txt["cat_buruk"]: "#ef4444"}
        )
        fig1.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown(f"<div class='desc-text'>{txt['dash_pie_desc']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown(f"<h4>{txt['dash_line_title']}</h4>", unsafe_allow_html=True)
        
        fig2 = px.line(
            data.reset_index(), x="index", y="pm_sepuluh", 
            color_discrete_sequence=['#0ea5e9'],
            labels={"index": txt["graph_axis_index"], "pm_sepuluh": txt["graph_axis_pm10"]}
        )
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
        pm10 = st.number_input(txt["label_pm10"], 0.0, 300.0, 48.0)
        co = st.number_input(txt["label_co"], 0.0, 100.0, 11.0)
    with col2:
        pm25 = st.number_input(txt["label_pm25"], 0.0, 300.0, 32.0)
        o3 = st.number_input(txt["label_o3"], 0.0, 300.0, 42.0)
    with col3:
        so2 = st.number_input(txt["label_so2"], 0.0, 300.0, 18.0)
        no2 = st.number_input(txt["label_no2"], 0.0, 300.0, 14.0)

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
        
        status_terjemahan = txt["cat_baik"] if hasil == "BAIK" else (txt["cat_sedang"] if hasil == "SEDANG" else txt["cat_buruk"])
        text_report_content = f"AQ-CARE SYSTEM AIR QUALITY REPORT\nLocation: {lokasi}\nResult Status: {status_terjemahan}\n"
        
        if hasil == "BAIK":
            st.success(txt["res_baik"])
            text_report_content += "Recommendation: Safe for all outdoor physical activities."
        elif hasil == "SEDANG":
            st.warning(txt["res_sedang"])
            text_report_content += "Recommendation: Sensitive groups should reduce strenuous outdoor activities."
        else:
            st.error(txt["res_buruk"])
            text_report_content += "Recommendation: Mandatory use of protective masks and keep doors closed."

        st.download_button(
            label=txt["download_report"],
            data=text_report_content,
            file_name=f"AQ_Report_{lokasi}.txt",
            mime="text/plain"
        )

        st.markdown("<div class='content-card' style='margin-top:15px;'>", unsafe_allow_html=True)
        
        prob_df = pd.DataFrame({"Status": model.classes_, "Persentase (%)": probabilitas * 100})
        prob_df["Status"] = prob_df["Status"].map(label_map)
        
        fig_bar = px.bar(
            prob_df, x="Status", y="Persentase (%)", color="Status", text_auto='.1f',
            title=txt["chart_conf_title"],
            color_discrete_map={txt["cat_baik"]: "#22c55e", txt["cat_sedang"]: "#eab308", txt["cat_buruk"]: "#ef4444"}
        )
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# HALAMAN 4: KAMUS SEHAT & DATA EDUKASI POLUTAN MENDALAM
# =====================================================
elif menu == txt["nav_info"]:
    st.markdown(f"""
    <div class="content-card">
        <h2>{txt["guide_title"]}</h2>
        <p style="color:#64748b; margin:0;">{txt["guide_subtitle"]}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown(f"### 🔬 {txt['pol_title_pm25']}")
        st.write(txt["pol_desc_pm25"])
        st.markdown("<hr style='border: 0.5px solid #f1f5f9;'>", unsafe_allow_html=True)
        
    with st.container():
        st.markdown(f"### 🚗 {txt['pol_title_co']}")
        st.write(txt["pol_desc_co"])
        st.markdown("<hr style='border: 0.5px solid #f1f5f9;'>", unsafe_allow_html=True)
        
    with st.container():
        st.markdown(f"### 🏭 {txt['pol_title_so2']}")
        st.write(txt["pol_desc_so2"])
        st.markdown("<hr style='border: 0.5px solid #f1f5f9;'>", unsafe_allow_html=True)
        
    with st.container():
        st.markdown(f"### ☀️ {txt['pol_title_o3']}")
        st.write(txt["pol_desc_o3"])
        st.markdown("<hr style='border: 0.5px solid #f1f5f9;'>", unsafe_allow_html=True)
        
    with st.container():
        st.markdown(f"### 🔥 {txt['pol_title_no2']}")
        st.write(txt["pol_desc_no2"])
        
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander(txt["faq_q1"]):
        st.write(txt["faq_a1"])
        
    with st.expander(txt["faq_q2"]):
        st.write(txt["faq_a2"])
        
    with st.expander(txt["faq_q3"]):
        st.write(txt["faq_a3"])

# =====================================================
# FOOTER WEBSITE
# =====================================================
st.markdown(f"""
<div class="footer">
    <b>AQ-CARE AI Community Portal © 2026</b><br>
    {txt["footer_text"]}
</div>
""", unsafe_allow_html=True)
